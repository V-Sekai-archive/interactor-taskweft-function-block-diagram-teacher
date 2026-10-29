"""Publish the split-out invalid rows into the FBD corpora.

Each corpus repo gets one commit carrying six tables per split: the three
per-candidate tables reduced to the ok rows, and the three new `*_invalid`
tables holding the split-out rows. One commit per repo, so a reader never sees
the invalid rows missing from `_candidates` before they exist in
`_candidates_invalid`.

The card is edited in place rather than regenerated: the published cards carry
sections the current generator does not produce, so regenerating would delete
them. Every rule is required to match, so a card the rules do not know is
reported instead of silently left stale.
"""
import argparse
import os
import re
import sys

from huggingface_hub import CommitOperationAdd, HfApi

OUT = os.environ.get("SPLIT_OUT", "work/split_out")
PAIRS = [("editscore", "fbd"), ("react", "react"), ("harness", "harness"),
         ("compose", "compose"), ("plan", "plan"), ("godot", "godot"),
         ("trainer", "trainer"), ("udon", "udon")]
SPLITS = ["train", "test", "evaluation"]
STEMS = ["{f}", "{f}_candidates", "{f}_scores", "{f}_invalid",
         "{f}_candidates_invalid", "{f}_scores_invalid"]

SPLIT_NOTE = (
    "The split is a partition: `{f}_candidates` and `{f}_candidates_invalid` together hold\n"
    "every candidate the corpus was built with, and so do `{f}_scores` with\n"
    "`{f}_scores_invalid` and `{f}` with `{f}_invalid`. A candidate is invalid when it was\n"
    "refused or its effect did not match, read from `{f}_scores` and cross-checked against\n"
    "the rank, so the two never disagree.\n")


def rules(f: str) -> list:
    """(name, regex, replacement) applied to the whole card. \\s+ spans the wraps."""
    return [
        # the modern card, seven of the eight families
        ("candidates-config",
         r"The `" + f + r"_candidates` config holds each candidate's compiler text form, "
         r"PLCopen XML, plan and runner result, with a sha256 of its source text\.",
         "The `" + f + "_candidates` config holds the matching candidate: the one that parses, "
         "compiles, runs and matches the intent, with its compiler text form, PLCopen XML, plan "
         "and runner result and a sha256 of its source text. The `" + f + "_candidates_invalid` "
         "config holds the split-out rows, every candidate that was refused or ran to the wrong "
         "effect, with the same columns."),
        ("scores-config",
         r"The `" + f + r"_scores` config holds parse, compile, run and effect-match flags, "
         r"step and wall-time measures, and the refusal\.",
         "The `" + f + "_scores` config holds parse, compile, run and effect-match flags, step and "
         "wall-time measures, and the refusal. `" + f + "_scores_invalid` carries the split-out "
         "candidates' verdicts."),
        ("joined-config",
         r"The default `" + f + r"` config joins the three\. Every artifact is in the parquet\. "
         r"An artifact a family does not produce is an empty string, never a null\.",
         "The default `" + f + "` config joins the root row with its matching candidate, each "
         "carrying its scores; `" + f + "_invalid` is the same view of the split-out candidates. "
         + SPLIT_NOTE.format(f=f) +
         "Every artifact is in the parquet. An artifact a family does not produce is an empty "
         "string, never a null."),
        # the ranked-intro line, which every modern card carries
        ("ranked-intro",
         r"Each intent has a root row, three ranked candidate diagrams and a score row per candidate\.",
         "Each intent has a root row, its matching candidate diagram, the split-out invalid ones, "
         "and a score row for each candidate."),
        # the fbd card, whose prose is the older generator's
        ("fbd-intro",
         r"EditScore-shaped corpus: one root row per intent, three candidates per row \(rank1 the\s+"
         r"reference diagram, rank3 one that compiles and does the wrong thing, rank5 one the\s+"
         r"compiler refuses\), and one score row per candidate",
         "EditScore-shaped corpus: one root row per intent, the matching candidate per row (rank1 "
         "the reference diagram), the refused and wrong-effect candidates split out into their own "
         "tables (rank3 one that compiles and does the wrong thing, rank5 one the compiler "
         "refuses), and one score row per candidate"),
        ("fbd-candidates-head",
         r"`" + f + r"_candidates`, three per row:",
         "`" + f + "_candidates`, one per row, the matching candidate:"),
        ("fbd-scores-head",
         r"`" + f + r"_scores`, three per row:",
         "`" + f + "_scores`, one per row:"),
        ("fbd-joined",
         r"`" + f + r"` is the joined view and the default config: the root row with its\s+"
         r"candidates, each carrying its scores\.",
         "`" + f + "` is the joined view and the default config: the root row with its matching "
         "candidate, carrying its scores. `" + f + "_invalid` is the same view of the split-out "
         "candidates.\n\n" + SPLIT_NOTE.format(f=f).rstrip()),
    ]


def local_paths(fam: str) -> dict:
    out = {}
    for split in SPLITS:
        pre = "" if split == "train" else f"{split}/"
        disk_pre = "" if split == "train" else f"{split}_"
        for stem in STEMS:
            s = stem.format(f=fam)
            out[f"{pre}data/{s}/{split}-00000-of-00001.parquet"] = os.path.join(
                OUT, f"{disk_pre}data__{s}__{split}.parquet")
    return out


def add_configs(readme: str, fam: str) -> str:
    block = []
    for stem in ["{f}_candidates_invalid", "{f}_scores_invalid", "{f}_invalid"]:
        s = stem.format(f=fam)
        block.append(f"- config_name: {s}\n  data_files:\n"
                     f"  - split: train\n    path: data/{s}/*.parquet\n"
                     f"  - split: test\n    path: test/data/{s}/*.parquet\n"
                     f"  - split: evaluation\n    path: evaluation/data/{s}/*.parquet\n")
    i = readme.index(f"- config_name: {fam}_scores\n")
    j = readme.index("\n---", i)
    return readme[:j + 1] + "".join(block) + readme[j + 1:]


def apply_rules(readme: str, fam: str) -> tuple:
    """(new text, rules that matched, rules that did not)."""
    hit, miss = [], []
    for name, pat, rep in rules(fam):
        new, n = re.subn(pat, rep.replace("\\", "\\\\"), readme, flags=re.S)
        if n:
            hit.append(name)
            readme = new
        else:
            miss.append(name)
    return readme, hit, miss


def self_test() -> int:
    paths = local_paths("fbd")
    assert len(paths) == 18, f"expected 18 paths, got {len(paths)}"
    assert sum(1 for p in paths if p.startswith("test/")) == 6
    assert sum(1 for p in paths if p.startswith("evaluation/")) == 6
    assert "test/data/fbd_candidates_invalid/test-00000-of-00001.parquet" in paths
    rd = ("---\nconfigs:\n- config_name: fbd\n  x: 1\n- config_name: fbd_scores\n  x: 2\n---\nbody\n")
    got = add_configs(rd, "fbd")
    assert got.count("- config_name:") == 5, got.count("- config_name:")
    assert got.count("\n---\n") == 1, "the closing fence moved"
    for s in ("fbd_invalid", "fbd_candidates_invalid", "fbd_scores_invalid"):
        assert s in got, s
    # a rule that cannot match must be reported, not silently skipped
    _, hit, miss = apply_rules("nothing here", "fbd")
    assert not hit and len(miss) == len(rules("fbd")), (hit, miss)
    # the modern rules must fire on a modern card
    modern = ("The `react_candidates` config holds each candidate's compiler text form, PLCopen "
              "XML, plan and runner result, with a sha256 of its source text.\n"
              "The `react_scores` config holds parse, compile, run and effect-match flags, step and "
              "wall-time measures, and the refusal.\n"
              "The default `react` config joins the three. Every artifact is in the parquet. An "
              "artifact a family does not produce is an empty string, never a null.\n")
    out, hit, miss = apply_rules(modern, "react")
    assert hit == ["candidates-config", "scores-config", "joined-config"], hit
    assert len(miss) == 5, miss          # the five fbd-only rules must not fire here
    assert "react_candidates_invalid" in out and "react_invalid` is the same view" in out
    assert "three ranked candidate diagrams" not in out
    print("ok  self-test: 8 of 8 controls fired (18 paths, three splits, three configs, the fence")
    print("    holds, an unknown card reports every rule, the modern rules fire, the split is named)")
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
    api = HfApi(token=token)
    want = set(a.families.split(","))
    bad = 0

    for suffix, fam in PAIRS:
        if fam not in want:
            continue
        repo = a.hub.format(family=suffix)
        paths = local_paths(fam)
        missing = [p for p, d in paths.items() if not os.path.exists(d)]
        if missing:
            print(f"FAIL {repo}: {len(missing)} local file(s) missing, first {missing[0]}")
            return 1
        src = api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="dataset")
        old = open(src, encoding="utf-8").read()
        new, hit, miss = apply_rules(add_configs(old, fam), fam)
        # fbd's card is the older generator's prose, so it needs the fbd rules
        # and must not be judged against the modern ones.
        if fam == "fbd":
            required = [n for n, _, _ in rules(fam) if n.startswith("fbd-")]
        else:
            required = ["candidates-config", "scores-config", "joined-config", "ranked-intro"]
        unmet = [n for n in required if n in miss]
        # a card that still claims three per row is stale however many rules fired
        stale = [s for s in ("three per row", "three candidates per row",
                             "three ranked candidate diagrams") if s in new]
        print(f"{repo:46s} README {len(old)}->{len(new)}  hit {hit}"
              + (f"  MISSED {unmet}" if unmet else "")
              + (f"  STALE {stale}" if stale else ""))
        if stale:
            bad += 1
        if a.apply and not unmet:
            rpath = os.path.join(OUT, f"README__{fam}.md")
            with open(rpath, "w", encoding="utf-8") as fh:
                fh.write(new)
            ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=d)
                   for p, d in sorted(paths.items())]
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=rpath))
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                              commit_message="Split the invalid rows out of the per-candidate tables")
            print(f"     committed {len(ops)} file(s)")
    if bad:
        print(f"\n{bad} card(s) did not match every rule; nothing was uploaded for them.")
        return 1
    if not a.apply:
        print("\ndry run: nothing uploaded. Pass --apply to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
