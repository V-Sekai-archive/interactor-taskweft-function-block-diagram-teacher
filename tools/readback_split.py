"""Read the split back from the Hub and prove it landed.

For every corpus: the per-candidate tables hold only the ok rows, the `*_invalid`
tables hold the split-out rows, and the two sides still partition what the corpus
published before the split.
"""
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

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
        verdict = "ok" if (ok_ranks == {1} and inv_ranks == {3, 5} and aligned
                           and nested == c.num_rows and nested_i == ci.num_rows) else "FAIL"
        if verdict == "FAIL":
            bad += 1
        print(f"  {fam:9s} {name:11s} {c.num_rows:10d} {ci.num_rows:8d} {s.num_rows:7d} "
              f"{si.num_rows:9d} {j.num_rows:7d} {ji.num_rows:9d} {verdict:>8s}")
print()
print(f"  {bad} failing pair(s)" if bad else "  OK: every corpus carries the split, both sides aligned")
sys.exit(1 if bad else 0)
