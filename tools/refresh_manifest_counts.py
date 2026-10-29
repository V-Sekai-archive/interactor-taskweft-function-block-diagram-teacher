"""Recompute a corpus's manifest counts from the tables it actually published.

The manifest records what each table holds. Splitting the invalid rows out changed
three of those tables and added three more, so the recorded counts stopped
describing the corpus. This reads the tables and rewrites the counts, and reports
any table the manifest names that the repo does not carry.
"""
import argparse
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

PAIRS = [("editscore", "fbd"), ("react", "react"), ("harness", "harness"),
         ("compose", "compose"), ("plan", "plan"), ("godot", "godot"),
         ("trainer", "trainer"), ("udon", "udon")]
SPLITS = [("", "train"), ("test/", "test"), ("evaluation/", "evaluation")]
STEMS = ["{f}_root", "{f}_candidates", "{f}_candidates_invalid",
         "{f}_scores", "{f}_scores_invalid", "{f}", "{f}_invalid"]
W = os.environ.get("SPLIT_WORK", "work/split_counts")
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def path_for(pre: str, name: str, table: str) -> str:
    return f"{pre}data/{table}/{name}-00000-of-00001.parquet"


def counts_of(fam: str, read) -> dict:
    """{split: {table: rows}} for one corpus. read(pre, name, table) -> int or None."""
    return {name: {s.format(f=fam): read(pre, name, s.format(f=fam)) for s in STEMS}
            for pre, name in SPLITS}


def diff_counts(old: dict, new: dict) -> list:
    """(split, table, was, now) for every entry that moved, appeared or vanished."""
    out = []
    for split in sorted(new):
        for table in sorted(new[split]):
            was = (old or {}).get(split, {}).get(table)
            now = new[split][table]
            if was != now:
                out.append((split, table, was, now))
    for split in sorted(old or {}):
        for table in sorted(old[split]):
            if table not in new.get(split, {}):
                out.append((split, table, old[split][table], None))
    return out


def self_test() -> int:
    seen = {}

    def read(pre, name, table):
        seen[path_for(pre, name, table)] = 1
        return 7
    new = counts_of("fbd", read)
    assert set(new) == {"train", "test", "evaluation"}, set(new)
    assert set(new["train"]) == {s.format(f="fbd") for s in STEMS}, set(new["train"])
    assert len(seen) == 21, f"expected 21 table reads, got {len(seen)}"
    assert "test/data/fbd_invalid/test-00000-of-00001.parquet" in seen, "the test split path is wrong"
    assert "evaluation/data/fbd_scores_invalid/evaluation-00000-of-00001.parquet" in seen
    assert new["train"]["fbd_candidates_invalid"] == 7
    # a stale count, a dropped table and a new table must all be reported
    old = {"train": {"fbd_candidates": 10125, "fbd_gone": 3}}
    d = diff_counts(old, new)
    assert ("train", "fbd_candidates", 10125, 7) in d, d
    assert ("train", "fbd_gone", 3, None) in d, d
    assert ("train", "fbd_root", None, 7) in d, d
    # identical counts report nothing, so a rerun is silent
    assert diff_counts(new, new) == []
    # an absent table must read as None, not as zero rows
    assert counts_of("fbd", lambda p, n, t: None)["train"]["fbd"] is None
    print("ok  self-test: 7 of 7 controls fired (21 table reads, the split tables are counted, both")
    print("    non-train paths are right, a stale count, a dropped table and a new table are each")
    print("    reported, equal is silent, and an absent table is None rather than 0)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="chibifire/taskweft-fbd-{family}-train")
    ap.add_argument("--families", default=",".join(f for _, f in PAIRS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    token = os.environ.get("HFTOK") or ""
    if not token:
        raise SystemExit("set HFTOK")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    want = set(a.families.split(","))
    os.makedirs(W, exist_ok=True)
    bad = 0

    for suffix, fam in PAIRS:
        if fam not in want:
            continue
        repo = a.hub.format(family=suffix)
        src = api.hf_hub_download(repo_id=repo, filename="manifest.json", repo_type="dataset")
        man = json.loads(open(src, encoding="utf-8").read())

        def read(pre, name, table, _repo=repo):
            path = path_for(pre, name, table)
            out = os.path.join(W, f"{_repo.split('/')[-1]}__{path.replace('/', '_')}")
            subprocess.run(["curl", "-sL", "-H", "Authorization: Bearer " + token,
                            RESOLVE.format(repo=_repo, path=path), "-o", out], check=True)
            try:
                return pq.read_table(out).num_rows
            except Exception:
                return None

        new = counts_of(fam, read)
        absent = sorted({t for sp in new.values() for t, v in sp.items() if v is None})
        if absent:
            print(f"FAIL {repo}: the manifest names {absent} but the repo has no such table")
            bad += 1
            continue
        d = diff_counts(man.get("counts") or {}, new)
        print(f"{repo:48s} {len(d)} count(s) moved")
        for split, table, was, now in d:
            print(f"     {split:11s} {table:26s} {str(was):>7s} -> {now}")
        if not d:
            continue
        man["counts"] = new
        if a.apply:
            path = os.path.join(W, f"manifest__{fam}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(man, fh, indent=2, sort_keys=True)
                fh.write("\n")
            api.upload_file(path_or_fileobj=path, path_in_repo="manifest.json", repo_id=repo,
                            repo_type="dataset",
                            commit_message="Record the split in the manifest counts")
            print("     manifest updated")
    if bad:
        return 1
    if not a.apply:
        print("\ndry run: nothing uploaded. Pass --apply to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
