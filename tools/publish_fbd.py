"""Publish a written stage to the Hub, the way publish_dress_on.py does.

The token is the `hf_token` field of this desk's `agents/<cn>` row in OpenBao, read
with a cert login that stores nothing. The upload is `upload_folder` with no delete
patterns, so a rerun adds shards. The README's `configs` block gives the viewer one
config per table with the joined view as the default; without it every directory
folds into one table. Nothing leaves until the stage passes the three refusals below.

    python tools/publish_fbd.py --stage work/stage --hub chibifire/taskweft-fbd-editscore-train
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import re
import sys
from pathlib import Path

FORBIDDEN = ("c:/users", "c:\\users", "/home/", "/private/tmp")
# A drive letter or a mount point names the desk even when no user name is in it.
# A drive letter needs a boundary before it: Godot prints `Camera3D:/root/...`,
# whose "D:/" is not a drive.
LOCAL_PATH = re.compile(r"(?i)(?:(?<![A-Za-z0-9_])[a-z]:[\\\\/]|/mnt/[a-z]/|/users/|/home/)")


def refuse_if_absolute(root: Path) -> None:
    hits = []
    for p in root.rglob("*.parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(p)
        for col in t.column_names:
            if "string" in str(t.schema.field(col).type):
                for v in t.column(col).to_pylist():
                    if isinstance(v, str) and LOCAL_PATH.search(v):
                        hits.append(f"{p.relative_to(root)}:{col}")
                        break
    if hits:
        sys.exit(f"REFUSED: {len(hits)} column(s) name a local filesystem: {hits[:5]}")


def refuse_if_forbidden(root: Path) -> None:
    bad = [str(p) for p in root.rglob("*") if any(m in str(p).lower() for m in FORBIDDEN)]
    if bad:
        sys.exit(f"REFUSED: blocked marker in staged path: {bad[:5]}")


def refuse_if_text_names_a_desk(root: Path) -> list[str]:
    """Every staged JSON and Markdown file is read, not just its name: the manifest
    is uploaded and once carried an absolute path naming the operator."""
    hits = []
    for p in sorted(root.rglob("*")):
        # rows/ is the writer's scratch tree and is not uploaded; its contents
        # reach the Hub only through the parquet columns, checked there instead.
        if p.relative_to(root).parts[:1] == ("rows",):
            continue
        if p.suffix.lower() not in (".json", ".md", ".txt", ".cff") or not p.is_file():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if LOCAL_PATH.search(line):
                hits.append(f"{p.relative_to(root)}:{n}")
                break
    return hits


def self_test() -> int:
    """Three controls: a clean tree passes, a planted Windows path and a planted
    WSL mount are each seen."""
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "manifest.json").write_text('{"compiler": "taskweft_fbd_compiler.exe"}', encoding="utf-8")
        if refuse_if_text_names_a_desk(root):
            fails.append("a clean manifest was refused")
        (root / "planted.json").write_text('{"godot_path": "C:\\\\Users\\\\someone\\\\godot.exe"}', encoding="utf-8")
        if not refuse_if_text_names_a_desk(root):
            fails.append("the planted local path was not seen; the gate is decoration")
        (root / "planted.json").write_text('{"p": "/mnt/c/fabric-starforged/x"}', encoding="utf-8")
        if not refuse_if_text_names_a_desk(root):
            fails.append("the planted WSL mount was not seen")
    for f in fails:
        print("FAIL", f)
    print(f"{3 - len(fails)} of 3 controls fired.")
    return 1 if fails else 0


def hf_token_from_bao() -> str:
    """This desk's Bao agent row, or the `bao-hf-token` helper where the desk has one.

    The row path and its client certs are the Windows desk's. A desk without them
    reads the same vault through its own helper, so the publisher runs on either
    rather than only where the certs happen to be installed.
    """
    bao = os.path.expanduser("~/bin/bao.exe")
    if not os.path.exists(bao):
        helper = shutil.which("bao-hf-token")
        if not helper:
            sys.exit("FAIL: no Bao client on this desk: neither ~/bin/bao.exe nor bao-hf-token")
        # the helper prints a note about the vault node above the token itself
        hf = subprocess.run([helper], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1].strip()
        if not hf.startswith("hf_"):
            sys.exit("FAIL: bao-hf-token answered no hf_ token")
        return hf
    env = dict(os.environ, BAO_ADDR="https://100.124.200.34:8200", BAO_TLS_SERVER_NAME="weftspun-bao.internal",
               BAO_CACERT=os.path.expanduser("~/.magi/ca-bundle.pem"),
               BAO_CLIENT_CERT=os.path.expanduser("~/.magi/v3-leaf-int.pem"),
               BAO_CLIENT_KEY=os.path.expanduser("~/.magi/bao-client-v3.key"))
    # the login prints a note about not storing the token above the token itself
    tok = subprocess.run([bao, "login", "-method=cert", "-no-store", "-field=token"], env=env,
                         capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1].strip()
    env["BAO_TOKEN"] = tok
    hf = subprocess.run([bao, "kv", "get", "-field=hf_token", "agents/magi-16739d.agents.weftspun"], env=env,
                        capture_output=True, text=True, timeout=60).stdout.strip()
    if not hf.startswith("hf_"):
        sys.exit("FAIL: no hf_token in bao agents row")
    return hf


SCORED = {
    "fbd": "the compiler and a runner that performed the plan",
    "react": "the compiler's reference scan on three constructed input traces per row",
    "harness": "the compiler's reference scan, rank1's outputs the reference for the others",
    "compose": "the compiler's reference scan over the composed controller's traces",
    "plan": "the compiler's step lowering, compared step for step against rank1's plan",
    "godot": "the engine itself: api_runner.gd performed the calls on the fixture scene and the returns were read back",
    "trainer": "the trainer config: the calls were applied to the mjlab task config and the term table read back",
}


def readme(manifest: dict, hub: str = "chibifire/taskweft-fbd-editscore-train") -> str:
    family = manifest.get("family", "fbd")
    names = [family, f"{family}_root", f"{family}_candidates", f"{family}_scores",
             f"{family}_candidates_invalid", f"{family}_scores_invalid", f"{family}_invalid"]
    scored = SCORED.get(family, SCORED["fbd"])
    splits = [("train", "data"), ("test", "test/data"), ("evaluation", "evaluation/data")]
    present = [(sp, d) for sp, d in splits if manifest.get(f"{sp}_rows", 0) > 0]
    configs = "".join(
        f"- config_name: {n}\n" + ("  default: true\n" if n == family else "")
        + "  data_files:\n" + "".join(f"  - split: {sp}\n    path: {d}/{n}/*.parquet\n" for sp, d in present)
        for n in names)
    held = manifest.get("holdout_families", []) + manifest.get("holdout_blocks", [])
    counts = ", ".join(f"{manifest.get(f'{sp}_rows', 0)} {sp}" for sp, _ in splits)
    return f"""---
license: mit
task_categories:
- text-generation
tags:
- iec-61131-3
- function-block-diagram
- plcopen
- taskweft
- editscore
configs:
{configs}---

# {hub.split("/")[-1]}

Intents and the IEC 61131-3 Function Block Diagrams that carry them out, as an
EditScore-shaped corpus: one root row per intent, the matching candidate per row
(rank1 the reference diagram), the refused and wrong-effect candidates split out
into their own tables (rank3 one that compiles and does the wrong thing, rank5 one
the compiler refuses), and one score row per candidate from {scored}. Every row is
constructed from a template and a seed, so the labels are true by construction and
the corpus regenerates from the seeds. Controls were asserted on every row before
the emit.

Three splits. `train` trains. `test` is the same distribution with every tenth seed
held out: the gate after training. `evaluation` is whole held-out families and block
kinds ({", ".join(held) or "none"}), never trained or tuned on.

## What a row carries

Every artefact is in the parquet; nothing points outside it.

`{family}_root`, one row per intent: `key`, `intent`, `template_id`, `seed`,
`frame_id`, `blocks` (the block kinds and signatures the reference diagram uses),
`traces` (the input traces as JSON strings, empty where the family has none), and
the EditScore stub columns.

`{family}_candidates`, one per row, the matching candidate: `candidate` and
`rank`, `fbd_text` (the compiler's text form, verbatim), `fbd_xml` (the PLCopen
XML, verbatim), `plan_json` and `result_json` (what the compiler planned and what
the runner returned, where the family has a runner), `fbd_sha` (sha256 of the text
the candidate was built from), `fbd_path` (where the writer kept it on disk) and
`provenance`. An artefact a family does not produce is an empty string, never a
null.

`{family}_candidates_invalid`, the split-out rows: every candidate that was
refused or ran to the wrong effect, with the same columns as `{family}_candidates`.

`{family}_scores`, one per row: `parses`, `compiles`, `runs`, `effect_matches`,
`steps`, `wall_ms`, `refusal`. `{family}_scores_invalid` carries the split-out
candidates' verdicts.

`{family}` is the joined view and the default config: the root row with its
matching candidate, each carrying its scores. `{family}_invalid` is the same view
of the split-out candidates.

The split is a partition: `{family}_candidates` and `{family}_candidates_invalid`
together hold every candidate the corpus was built with, and so do
`{family}_scores` with `{family}_scores_invalid` and `{family}` with
`{family}_invalid`. A candidate is invalid when it was refused or its effect did
not match, read from `{family}_scores` and cross-checked against the rank, so the
two never disagree.

Rows: {manifest["rows"]} ({counts}).
Templates: {json.dumps(manifest["templates"])}. Compiler: taskweft-fbd-compiler
{manifest["compiler_sha"]}. Source: v-sekai-fabric/taskweft-fbd-teacher.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--readme-only", action="store_true", help="upload the dataset card alone")
    ap.add_argument("--self-test", action="store_true", help="run the local-path controls and exit")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(self_test())

    stage: Path = args.stage
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    refuse_if_absolute(stage)
    refuse_if_forbidden(stage)
    named = refuse_if_text_names_a_desk(stage)
    if named:
        sys.exit(f"REFUSED: {len(named)} staged file(s) name a local filesystem: {named[:5]}")
    (stage / "README.md").write_text(readme(manifest, args.hub), encoding="utf-8")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_from_bao())
    api.create_repo(args.hub, repo_type="dataset", exist_ok=True, private=args.private)
    if args.readme_only:
        api.upload_file(path_or_fileobj=str(stage / "README.md"), path_in_repo="README.md",
                        repo_id=args.hub, repo_type="dataset")
    else:
        # rows/ is the writer's scratch tree; the parquets carry every row
        if any(f.startswith("rows/") for f in api.list_repo_files(args.hub, repo_type="dataset")):
            api.delete_folder(path_in_repo="rows", repo_id=args.hub, repo_type="dataset")
        api.upload_folder(folder_path=str(stage), repo_id=args.hub, repo_type="dataset",
                          ignore_patterns=["rows/**", "README.md.bak"])
    files = api.list_repo_files(args.hub, repo_type="dataset")
    parquets = [f for f in files if f.endswith(".parquet")]
    print(f"published {args.hub}: {len(files)} file(s), {len(parquets)} parquet(s)")
    if not parquets:
        sys.exit("FAIL: the read-back lists no parquet")


if __name__ == "__main__":
    main()
