"""Read the split back from the Hub and prove it landed.

For every corpus: the per-candidate tables hold only the ok rows, the `*_invalid`
tables hold the split-out rows, and the two sides still partition what the corpus
published before the split.

This is the check that decides whether the split is real, so it carries a
self-test: a check that has only ever printed ok has not been shown to be able to
print anything else.
"""
import argparse
import os
import subprocess
import sys

import pyarrow.parquet as pq

AP = argparse.ArgumentParser()
AP.add_argument("--self-test", action="store_true")
ARGS = AP.parse_args()


def verdict_of(ok_ranks, inv_ranks, aligned, nested, nested_i, c_rows, ci_rows) -> str:
    """The whole read-back verdict, as a pure function so a self-test can drive it."""
    good = (ok_ranks == {1} and inv_ranks == {3, 5} and aligned
            and nested == c_rows and nested_i == ci_rows)
    return "ok" if good else "FAIL"


def self_test() -> int:
    assert verdict_of({1}, {3, 5}, True, 10, 20, 10, 20) == "ok", "the correct split was not accepted"
    # one control per clause, so a clause that quietly stops being checked is caught here
    controls = {
        "a candidate half still holds a wrong-plan row": ({1, 3}, {3, 5}, True, 10, 20, 10, 20),
        "the invalid half lost the refused row": ({1}, {3}, True, 10, 20, 10, 20),
        "scores no longer line up with candidates": ({1}, {3, 5}, False, 10, 20, 10, 20),
        "the joined view dropped nested candidates": ({1}, {3, 5}, True, 9, 20, 10, 20),
        "the invalid view dropped nested candidates": ({1}, {3, 5}, True, 10, 19, 10, 20),
    }
    for label, args in controls.items():
        assert verdict_of(*args) == "FAIL", f"not caught: {label}"
    assert len(controls) == 5
    print("ok  self-test: 6 of 6 controls fired (the correct split passes, and a wrong-plan row,")
    print("    a lost refused row, unaligned scores and a short nested view are each caught)")
    return 0


if ARGS.self_test:
    sys.exit(self_test())

TOK = os.environ["HFTOK"]
PAIRS = [("editscore", "fbd"), ("react", "react"), ("harness", "harness"),
         ("compose", "compose"), ("plan", "plan"), ("godot", "godot"),
         ("trainer", "trainer"), ("udon", "udon")]
SPLITS = [("", "train"), ("test/", "test"), ("evaluation/", "evaluation")]
W = "work/split_readback"
os.makedirs(W, exist_ok=True)


def get(repo, path):
    out = os.path.join(W, repo.split("/")[-1] + "__" + path.replace("/", "_"))
    subprocess.run(["curl", "-sL", "-H", "Authorization: Bearer " + TOK,
                    f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", "-o", out], check=True)
    return pq.read_table(out)


print(f"  {'family':9s} {'split':11s} {'candidates':>10s} {'invalid':>8s} {'scores':>7s} "
      f"{'s_invalid':>9s} {'joined':>7s} {'j_invalid':>9s} {'verdict':>8s}")
bad = 0
for suffix, fam in PAIRS:
    repo = f"chibifire/taskweft-fbd-{suffix}-train"
    for pre, name in SPLITS:
        c = get(repo, f"{pre}data/{fam}_candidates/{name}-00000-of-00001.parquet")
        ci = get(repo, f"{pre}data/{fam}_candidates_invalid/{name}-00000-of-00001.parquet")
        s = get(repo, f"{pre}data/{fam}_scores/{name}-00000-of-00001.parquet")
        si = get(repo, f"{pre}data/{fam}_scores_invalid/{name}-00000-of-00001.parquet")
        j = get(repo, f"{pre}data/{fam}/{name}-00000-of-00001.parquet")
        ji = get(repo, f"{pre}data/{fam}_invalid/{name}-00000-of-00001.parquet")
        # every row left in candidates must be a candidate that matches
        ok_ranks = {r["rank"] for r in c.to_pylist()}
        inv_ranks = {r["rank"] for r in ci.to_pylist()}
        # the scores halves must line up with the candidate halves, key for key
        key = lambda t: sorted((r["row_key"], r["candidate"]) for r in t.to_pylist())
        aligned = key(c) == key(s) and key(ci) == key(si)
        nested = sum(len(r["candidates"]) for r in j.to_pylist())
        nested_i = sum(len(r["candidates"]) for r in ji.to_pylist())
        verdict = verdict_of(ok_ranks, inv_ranks, aligned,
                             nested, nested_i, c.num_rows, ci.num_rows)
        if verdict == "FAIL":
            bad += 1
        print(f"  {fam:9s} {name:11s} {c.num_rows:10d} {ci.num_rows:8d} {s.num_rows:7d} "
              f"{si.num_rows:9d} {j.num_rows:7d} {ji.num_rows:9d} {verdict:>8s}")
print()
print(f"  {bad} failing pair(s)" if bad else "  OK: every corpus carries the split, both sides aligned")
sys.exit(1 if bad else 0)
