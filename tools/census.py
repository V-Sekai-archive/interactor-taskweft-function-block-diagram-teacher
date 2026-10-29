"""The corpus census, and the gate on it (RFD66606-2236, step 2 precondition).

Reads every stage directory given, counts rows per family, template and frame, the
block kinds each rank1 program uses, the program length histogram, the tokens per
row in the text form under Gemma 4's tokenizer, and the two holdout axes: a family
or a block kind named as held out must appear in no training row. A leak is a
refusal; `--self-test` plants one and must be refused.

    python tools/census.py work/stage work/stage_react work/stage_harness --holdout-blocks TOF,RS,GE,MOD
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

BLOCK_RE = re.compile(r"^\w+ = ([A-Z_]+)(?:\[|\()", re.M)


SIG_RE = re.compile(r"CALL\[([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\]")


def blocks_of(text: str) -> list[str]:
    """Block kinds, with a CALL counted by its signature so an API surface is enumerated."""
    return [b for b in BLOCK_RE.findall(text) if b != "CALL"] + SIG_RE.findall(text)


def rows_of(stage: Path, split_dir: Path) -> list[dict]:
    fam = json.loads((stage / "manifest.json").read_text(encoding="utf-8")).get("family", "fbd")
    if not (split_dir / f"{fam}_root").is_dir():
        return []
    root = pq.read_table(split_dir / f"{fam}_root").to_pylist()
    cands = pq.read_table(split_dir / f"{fam}_candidates").to_pylist()
    ref = {c["row_key"]: c for c in cands if c["candidate"] == "rank1"}
    out = []
    for r in root:
        c = ref[r["key"]]
        # The columns carry the program itself since payload moved into the parquet;
        # fbd_path is the staged copy and is provenance, not the source of truth.
        text, is_xml = c.get("fbd_text"), False
        if not text:
            text, is_xml = c.get("fbd_xml"), True
        if not text:
            path = stage / c["fbd_path"]
            text, is_xml = path.read_text(encoding="utf-8"), path.suffix == ".xml"
        if is_xml:
            text = re.sub(r'typeName="([A-Z_]+)"', lambda m: f"\n_ = {m.group(1)}(", text)
        out.append({"family": fam, "template": r["template_id"], "frame": r.get("frame_id", 0),
                    "blocks": blocks_of(text), "text": text})
    return out


def tokens(texts: list[str]) -> list[int] | None:
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", "tokenizer.json"))
    except Exception as e:  # noqa: BLE001
        print(f"tokens: not counted ({e.__class__.__name__}); pass --no-tokens to silence", file=sys.stderr)
        return None
    return [len(tok.encode(t).ids) for t in texts]


def census(stages: list[Path], holdout_families: set[str], holdout_blocks: set[str], count_tokens: bool) -> tuple[dict, list[str]]:
    train, test, holdout = [], [], []
    for s in stages:
        train += rows_of(s, s / "data")
        if (s / "test" / "data").is_dir():
            test += rows_of(s, s / "test" / "data")
        for legacy in ("holdout", "evaluation"):
            if (s / legacy / "data").is_dir() and any((s / legacy / "data").iterdir()):
                holdout += rows_of(s, s / legacy / "data")
    problems = []
    fam_counts = collections.Counter(r["family"] for r in train)
    tpl_counts = collections.Counter(r["template"] for r in train)
    block_counts = collections.Counter(b for r in train for b in set(r["blocks"]))
    lengths = collections.Counter(len(r["blocks"]) for r in train)
    for r in train:
        if r["template"] in holdout_families:
            problems.append(f"held-out family {r['template']} leaked into training")
            break
    leaked = {b for r in train for b in r["blocks"] if b in holdout_blocks}
    for b in sorted(leaked):
        problems.append(f"held-out block kind {b} leaked into training")
    for b in sorted(holdout_blocks):
        if not any(b in r["blocks"] for r in holdout):
            problems.append(f"held-out block kind {b} appears in no holdout row either; the axis is empty")
    tok = tokens([r["text"] for r in train]) if count_tokens else None
    report = {
        "train_rows": len(train), "test_rows": len(test), "evaluation_rows": len(holdout),
        "families": dict(fam_counts), "templates": dict(sorted(tpl_counts.items())),
        "frames": dict(collections.Counter(f"{r['template']}#{r['frame']}" for r in train)),
        "block_kinds": dict(sorted(block_counts.items())),
        "block_kinds_present": len(block_counts),
        "length_histogram": dict(sorted(lengths.items())),
        "holdout_families": sorted(holdout_families), "holdout_blocks": sorted(holdout_blocks),
        "tokens": None if tok is None else {"mean": round(statistics.mean(tok), 1), "p50": statistics.median(tok),
                                             "max": max(tok), "total": sum(tok)},
    }
    return report, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", type=Path)
    ap.add_argument("--holdout-families", default="")
    ap.add_argument("--holdout-blocks", default="")
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--min-rows", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    hf = {f for f in args.holdout_families.split(",") if f}
    hb = {b for b in args.holdout_blocks.split(",") if b}
    report, problems = census(args.stages, hf, hb, not args.no_tokens)
    if args.self_test:
        # plant a leak: the most common block kind declared held out
        planted = next(iter(sorted(report["block_kinds"], key=report["block_kinds"].get, reverse=True)))
        _, planted_problems = census(args.stages, hf, hb | {planted}, False)
        if not any("leaked" in p for p in planted_problems):
            sys.exit(f"self-test: planted held-out block {planted} was not refused; the census is decoration")
        print(f"self-test: planted {planted} refused")
    if args.min_rows and report["train_rows"] < args.min_rows:
        problems.append(f"{report['train_rows']} train rows, below the floor of {args.min_rows}")
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for p in problems:
        print("FAIL", p)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
