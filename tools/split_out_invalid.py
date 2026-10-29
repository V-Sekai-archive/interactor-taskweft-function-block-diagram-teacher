"""Split a published FBD corpus's per-candidate tables into ok and invalid.

A candidate is ok when it parses, compiles, runs and its effect matches the
intent; anything else is invalid, whether it was refused or ran to the wrong
effect. The verdict is read from `<family>_scores`, never inferred from the rank
label, and the two are cross-checked so a wrong verdict rule cannot pass.

Three tables are per candidate — `<family>_candidates`, `<family>_scores` and the
joined `<family>` view — and all three split. `<family>_root` is one row per
intent and does not.

Nothing is dropped: the ok side and the invalid side are asserted to partition
the input, table by table, before anything is written.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
TREE = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"
SPLITS = [("", "train"), ("test/", "test"), ("evaluation/", "evaluation")]
# The repo suffix is not always the family name: the fbd family publishes as
# chibifire/taskweft-fbd-editscore-train.
PAIRS = [("editscore", "fbd"), ("react", "react"), ("harness", "harness"),
         ("compose", "compose"), ("plan", "plan"), ("godot", "godot"),
         ("trainer", "trainer"), ("udon", "udon")]
VERDICT_COLS = ("parses", "compiles", "runs", "effect_matches")

# rank1 compiles, runs and matches; rank3 compiles and misses; rank5 is refused.
EXPECTED_RANK = {1: "ok", 3: "invalid", 5: "invalid"}

# table key -> (published path stem, invalid-side path stem)
TABLES = {
    "candidates": ("{f}_candidates", "{f}_candidates_invalid"),
    "scores": ("{f}_scores", "{f}_scores_invalid"),
    "joined": ("{f}", "{f}_invalid"),
}


def curl(url: str, out: str, token: str) -> None:
    subprocess.run(["curl", "-sL", "-H", "Authorization: Bearer " + token, url, "-o", out], check=True)


def verdict(score: dict, rule=VERDICT_COLS) -> str:
    return "ok" if all(score.get(c) for c in rule) else "invalid"


def tree_paths(repo: str, token: str) -> set:
    raw = subprocess.run(["curl", "-sL", "-H", "Authorization: Bearer " + token,
                          TREE.format(repo=repo)], capture_output=True, text=True, check=True).stdout
    got = json.loads(raw)
    if not isinstance(got, list):
        raise SystemExit(f"{repo}: the tree API answered {str(got)[:200]}")
    return {e["path"] for e in got if e["type"] == "file"}


def split_candidates(cand: pa.Table, score: pa.Table, rule=VERDICT_COLS):
    """(ok, invalid, disagreements) for a flat candidate table."""
    by_key = {(r["row_key"], r["candidate"]): r for r in score.to_pylist()}
    ok, bad, disagree = [], [], []
    for r in cand.to_pylist():
        s = by_key.get((r["row_key"], r["candidate"]))
        if s is None:
            raise SystemExit(f"no score row for {r['row_key']} {r['candidate']}")
        v = verdict(s, rule)
        want = EXPECTED_RANK.get(r["rank"])
        if want is not None and want != v:
            disagree.append((r["row_key"], r["candidate"], r["rank"], v))
        (ok if v == "ok" else bad).append(r)
    mk = lambda rows: pa.Table.from_pylist(rows, schema=cand.schema)
    return mk(ok), mk(bad), disagree


def split_flat(score: pa.Table, verdicts: dict):
    """(ok, invalid) for a flat per-candidate table that carries no rank."""
    ok, bad = [], []
    for r in score.to_pylist():
        (ok if verdicts[(r["row_key"], r["candidate"])] == "ok" else bad).append(r)
    mk = lambda rows: pa.Table.from_pylist(rows, schema=score.schema)
    return mk(ok), mk(bad)


def split_joined(joined: pa.Table, verdicts: dict):
    """(ok, invalid) for the nested view, filtering each row's candidate list."""
    ok, bad = [], []
    for r in joined.to_pylist():
        keep = [c for c in r["candidates"] if verdicts[(r["key"], c["candidate"])] == "ok"]
        drop = [c for c in r["candidates"] if verdicts[(r["key"], c["candidate"])] != "ok"]
        if keep:
            ok.append({**r, "candidates": keep})
        if drop:
            bad.append({**r, "candidates": drop})
    mk = lambda rows: pa.Table.from_pylist(rows, schema=joined.schema)
    return mk(ok), mk(bad)


def assert_partition(before: pa.Table, ok: pa.Table, bad: pa.Table, label: str, keys) -> None:
    kb, ko, ki = keys(before), keys(ok), keys(bad)
    if len(ko) + len(ki) != len(kb):
        raise SystemExit(f"{label}: partition lost rows {len(ko)}+{len(ki)} != {len(kb)}")
    if set(ko) & set(ki):
        raise SystemExit(f"{label}: partition overlaps on {len(set(ko) & set(ki))} keys")
    # both sides are sorted here: the input is in the writer's order, so comparing
    # a sorted split against it unsorted fails on correct data.
    if sorted(ko + ki) != sorted(kb):
        raise SystemExit(f"{label}: partition is not the input set")


def self_test() -> int:
    """A plausible-but-wrong verdict rule must be caught, or the gate is decoration."""
    cand = pa.Table.from_pylist([
        {"row_key": "x/1", "candidate": "rank1", "rank": 1},
        {"row_key": "x/1", "candidate": "rank3", "rank": 3},
        {"row_key": "x/1", "candidate": "rank5", "rank": 5},
    ])
    score = pa.Table.from_pylist([
        {"row_key": "x/1", "candidate": "rank1", "parses": True, "compiles": True, "runs": True, "effect_matches": True},
        {"row_key": "x/1", "candidate": "rank3", "parses": True, "compiles": True, "runs": True, "effect_matches": False},
        {"row_key": "x/1", "candidate": "rank5", "parses": False, "compiles": False, "runs": False, "effect_matches": False},
    ])
    ok, bad, d = split_candidates(cand, score)
    assert not d, f"the correct rule disagrees with the ranks: {d}"
    assert len(ok) == 1 and len(bad) == 2
    # `runs` alone calls the wrong-plan mutant ok, which is the mistake this gate exists for.
    _, bad_runs, d2 = split_candidates(cand, score, rule=("runs",))
    assert len(d2) == 1, f"a rule that calls rank3 ok was not caught: {d2}"
    assert len(bad_runs) == 1, "the wrong rule did not move rank3 into ok"
    # the partition holds even when a rule moves everything to one side
    ok_all, bad_all, _ = split_candidates(cand, score, rule=())
    assert_partition(cand, ok_all, bad_all, "all-ok", lambda t: [(r["row_key"], r["candidate"]) for r in t.to_pylist()])
    assert len(ok_all) == 3 and len(bad_all) == 0

    # the partition gate must fail on a broken partition, or it certifies nothing.
    # A gate that only ever passes is the defect this file was written against: the
    # first version compared a sorted split against the input's own order and so
    # failed on correct data, which a control would have caught immediately.
    def catches(label, before, ok, bad, keys) -> bool:
        try:
            assert_partition(before, ok, bad, label, keys)
        except SystemExit:
            return True
        return False

    keys_flat = lambda t: [(r["row_key"], r["candidate"]) for r in t.to_pylist()]
    mk = lambda rows: pa.Table.from_pylist(rows, schema=cand.schema)
    full_ok, full_bad, _ = split_candidates(cand, score)
    assert catches("lost", cand, full_ok, mk(full_bad.to_pylist()[:-1]), keys_flat), \
        "a lost row was not caught"
    # ok and bad must total 3 keys here, or the length check fires instead of the
    # overlap check and the control proves the wrong branch.
    assert catches("overlap", cand, mk(full_ok.to_pylist() + [full_bad.to_pylist()[0]]),
                   mk([full_bad.to_pylist()[0]]), keys_flat), "an overlap was not caught"
    assert catches("substituted", cand, full_ok,
                   mk(full_bad.to_pylist()[:-1] + [dict(full_bad.to_pylist()[-1], candidate="rank9")]),
                   keys_flat), "a substituted key was not caught"
    # and it must pass on the correct split, so the controls above are not noise
    assert not catches("correct", cand, full_ok, full_bad, keys_flat)

    # the nested view must keep the row on whichever side has candidates, and drop
    # it only when that side is empty
    joined = pa.Table.from_pylist([{"key": "x/1", "candidates": [
        {"candidate": "rank1", "rank": 1}, {"candidate": "rank3", "rank": 3}, {"candidate": "rank5", "rank": 5}]}])
    v = {("x/1", "rank1"): "ok", ("x/1", "rank3"): "invalid", ("x/1", "rank5"): "invalid"}
    jok, jbad = split_joined(joined, v)
    assert jok.num_rows == 1 and jbad.num_rows == 1, (jok.num_rows, jbad.num_rows)
    assert len(jok.to_pylist()[0]["candidates"]) == 1 and len(jbad.to_pylist()[0]["candidates"]) == 2
    print("ok  self-test: 8 of 8 controls fired (correct rule agrees, `runs` alone is caught,")
    print("    the wrong rule moves rank3, an all-ok rule still partitions, the nested view splits,")
    print("    and the partition gate catches a lost row, an overlap and a substituted key)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="chibifire/taskweft-fbd-{family}-train")
    ap.add_argument("--families", default=",".join(f for _, f in PAIRS))
    ap.add_argument("--workdir", default="work/split_work")
    ap.add_argument("--out", default="work/split_out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    token = os.environ.get("HFTOK") or ""
    if not token:
        raise SystemExit("set HFTOK")
    os.makedirs(a.workdir, exist_ok=True)
    if a.write:
        os.makedirs(a.out, exist_ok=True)
    want = set(a.families.split(","))

    print(f"{'family':9s} {'split':11s} {'cands':>6s} {'ok':>6s} {'invalid':>8s} "
          f"{'scores ok/inv':>15s} {'joined ok/inv':>15s} {'disagree':>9s}")
    tot = collections.Counter()
    problems = []
    for suffix, fam in PAIRS:
        if fam not in want:
            continue
        repo = a.repo.format(family=suffix)
        have = tree_paths(repo, token)
        for pre, name in SPLITS:
            paths = {k: (f"{pre}data/{stem.format(f=fam)}/{name}-00000-of-00001.parquet")
                     for k, (stem, _) in TABLES.items()}
            missing = [k for k, p in paths.items() if p not in have]
            if missing:
                problems.append(f"{fam}/{name}: missing {missing}")
                print(f"{fam:9s} {name:11s}   (missing tables: {missing})")
                continue
            local = {}
            for k, p in paths.items():
                dst = os.path.join(a.workdir, f"{fam}_{name}_{k}.parquet")
                curl(HF.format(repo=repo, path=p), dst, token)
                local[k] = pq.read_table(dst)

            ok_c, bad_c, disagree = split_candidates(local["candidates"], local["scores"])
            vmap = {(r["row_key"], r["candidate"]): verdict(r) for r in local["scores"].to_pylist()}
            ok_s, bad_s = split_flat(local["scores"], vmap)
            ok_j, bad_j = split_joined(local["joined"], vmap)

            keys_flat = lambda t: [(r["row_key"], r["candidate"]) for r in t.to_pylist()]
            # the joined view keeps a row per intent on both sides, so its
            # partition is over the nested candidates, not the rows.
            keys_nested = lambda t: [(r["key"], c["candidate"]) for r in t.to_pylist() for c in r["candidates"]]
            assert_partition(local["candidates"], ok_c, bad_c, f"{fam}/{name}/candidates", keys_flat)
            assert_partition(local["scores"], ok_s, bad_s, f"{fam}/{name}/scores", keys_flat)
            assert_partition(local["joined"], ok_j, bad_j, f"{fam}/{name}/joined", keys_nested)

            tot["cands"] += local["candidates"].num_rows
            tot["ok"] += ok_c.num_rows
            tot["invalid"] += bad_c.num_rows
            tot["disagree"] += len(disagree)
            print(f"{fam:9s} {name:11s} {local['candidates'].num_rows:6d} {ok_c.num_rows:6d} "
                  f"{bad_c.num_rows:8d} {str(ok_s.num_rows)+'/'+str(bad_s.num_rows):>15s} "
                  f"{str(ok_j.num_rows)+'/'+str(bad_j.num_rows):>15s} {len(disagree):9d}")
            if disagree:
                problems.append(f"{fam}/{name}: {len(disagree)} rank/verdict disagreements")

            if a.write:
                for tbl, (stem, istem) in TABLES.items():
                    ok_t = {"candidates": ok_c, "scores": ok_s, "joined": ok_j}[tbl]
                    bad_t = {"candidates": bad_c, "scores": bad_s, "joined": bad_j}[tbl]
                    pq.write_table(ok_t, os.path.join(a.out, f"{pre.replace('/', '_')}data__{stem.format(f=fam)}__{name}.parquet"))
                    pq.write_table(bad_t, os.path.join(a.out, f"{pre.replace('/', '_')}data__{istem.format(f=fam)}__{name}.parquet"))

    print(f"{'TOTAL':9s} {'':11s} {tot['cands']:6d} {tot['ok']:6d} {tot['invalid']:8d} "
          f"{'':>15s} {'':>15s} {tot['disagree']:9d}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
