"""Write the EditScore-shaped FBD corpus: root, candidates and scores, three ZStandard
parquets in Essential Tuple Normal Form, plus the joined view the dataset viewer reads.

Every row is constructed from a template and a seed, so the labels are true by
construction: the compiler (taskweft-fbd-compiler `check` and `plan`) and a runner
that performs the plan's steps in a scratch directory score every candidate, and the
controls are asserted on every row before anything is written. A holdout of one row
in ten is written apart and never trains.

Each candidate also keeps the plan it ran, in `plan_json`, and what running it returned,
in `result_json`: the plan its own program compiles to, or with `--plan-source rectgtn`
the RECTGTN domain's own plan for `--plan-domain`. A candidate the compiler refuses has
no plan and keeps an empty slot rather than borrowing one.

    python tools/write_fbd_rows.py --rows 5000 --out work/stage
    python tools/write_fbd_rows.py --rows 20 --out work/smoke --negative-control
    python tools/write_fbd_rows.py --rows 20 --out work/smoke --plan-source rectgtn \
        --plan-domain ../taskweft/priv/smoke/blocks_world.jsonld \
        --plan-problem ../taskweft/priv/plans/problems/blocks_world_1a.jsonld
"""
from __future__ import annotations

import argparse
import re
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbd_templates import TEMPLATES, Row, row_for  # noqa: E402
from react_templates import REACT_TEMPLATES, ReactRow, react_row_for  # noqa: E402
from harness_templates import ensure_generated, harness_names, harness_row  # noqa: E402
from compose_templates import PAIRS, compose_row_for  # noqa: E402
from plan_rows import GOALS, plan_row  # noqa: E402
from godot_api_templates import GODOT_TEMPLATES, godot_row_for, signatures_of  # noqa: E402
from trainer_api_templates import TRAINER_TEMPLATES, trainer_row_for  # noqa: E402
from udon_templates import TEMPLATES as UDON_TEMPLATES, udon_row  # noqa: E402
from census import blocks_of  # noqa: E402
from rectgtn_plan import plan_json_of as rectgtn_plan_json  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CANDIDATES = [("rank1", 1), ("rank3", 3), ("rank5", 5)]
STUB = ("fbd", "intent_to_fbd", "instruction_following", "input_intent", "text")
REACT_STUB = ("react", "intent_to_controller", "instruction_following", "input_intent", "text")
STUBS = {
    "fbd": STUB, "react": REACT_STUB,
    "harness": ("harness", "intent_to_block", "instruction_following", "input_intent", "text"),
    "compose": ("compose", "intent_to_composed_controller", "instruction_following", "input_intent", "text"),
    "plan": ("plan", "goal_to_plan", "instruction_following", "input_intent", "text"),
    "godot": ("godot", "intent_to_api_calls", "instruction_following", "input_intent", "text"),
    "trainer": ("trainer", "intent_to_trainer_config", "instruction_following", "input_intent", "text"),
    "udon": ("udon", "udonsharp_to_fbd", "translation", "input_source", "text"),
}

_trainer_local = threading.local()


class TrainerServer:
    """One config runner per worker thread, in WSL, one JSON line per plan."""

    def __init__(self) -> None:
        # a script file rather than bash -c: the interop layer mangles a long -c string
        cmd = ["wsl.exe", "--", "bash", "/mnt/c/fabric-starforged/3-interactor/taskweft-fbd-teacher/scripts/serve_trainer.sh"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", bufsize=1)
        for line in self.p.stdout:
            if line.startswith("{") and json.loads(line).get("ready"):
                break
        else:
            raise RuntimeError("the trainer config runner did not come up")

    def ask(self, steps: list[dict]) -> dict:
        self.p.stdin.write(json.dumps({"steps": steps}) + "\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError("the trainer config runner went away")


def perform_in_trainer(plan_json: str) -> tuple[bool, dict, str]:
    srv = getattr(_trainer_local, "srv", None)
    if srv is None:
        srv = _trainer_local.srv = TrainerServer()
    res = srv.ask(json.loads(plan_json)["steps"])
    return bool(res.get("ok")), res.get("table", {}), res.get("error", "")

GODOT_PROJECT = HERE.parent / "taskweft-godot-sandbox" / "priv" / "godot_project"


def find_godot() -> Path:
    env = os.environ.get("TASKWEFT_GODOT")
    if env:
        return Path(env)
    sys.exit("FAIL: the godot family needs TASKWEFT_GODOT, the path to a Godot 4.5 executable")


def godot_record() -> dict:
    """The engine that scored the stage, by name and sha256. A published manifest
    carries no local path: it would name the desk and its user."""
    if not os.environ.get("TASKWEFT_GODOT"):
        return {"godot_name": "none", "godot_sha": "none", "godot_version": "none"}
    g = find_godot()
    if not g.is_file():
        sys.exit(f"FAIL: TASKWEFT_GODOT names no file: {g}")
    sha = hashlib.sha256(g.read_bytes()).hexdigest()
    r = subprocess.run([str(g), "--version"], capture_output=True, text=True, timeout=60)
    version = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "unknown"
    return {"godot_name": g.name, "godot_sha": sha, "godot_version": version}


def perform_in_godot(plan_json: str, row_dir: Path, name: str) -> tuple[bool, list[dict], str]:
    """The API runner performs the plan on the fixture scene; returns (ran, results, error)."""
    plan = row_dir / f"{name}.plan.json"
    out = row_dir / f"{name}.result.json"
    plan.write_text(plan_json, encoding="utf-8")
    cmd = [str(find_godot()), "--headless", "--path", str(GODOT_PROJECT), "--script", "res://scripts/api_runner.gd",
           "--", "--plan", str(plan.resolve()), "--out", str(out.resolve())]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, [], "godot timed out"
    m = re.search(r"^exit (\d+)", r.stdout, re.M)
    if not m or not out.is_file():
        return False, [], (r.stderr.strip() or r.stdout.strip())[-400:] or "godot wrote no result"
    results = json.loads(out.read_text(encoding="utf-8")).get("results", [])
    if m.group(1) != "0":
        failed = next((x for x in results if not x.get("ok")), None)
        return False, results, f"{failed['command']} did not run" if failed else f"godot exit {m.group(1)}"
    return True, results, ""


def find_compiler(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
    elif os.environ.get("FBD_COMPILER"):
        p = Path(os.environ["FBD_COMPILER"])
    else:
        exe = "taskweft_fbd_compiler.exe" if os.name == "nt" else "taskweft_fbd_compiler"
        p = HERE.parent / "taskweft-fbd-compiler" / ".lake" / "build" / "bin" / exe
    if not p.is_file():
        sys.exit(f"FAIL: no compiler at {p}; build taskweft-fbd-compiler or pass --compiler")
    # the signature tables sit beside the compiler's source, not the writer's cwd
    sigs = p.resolve().parents[3] / "sigs"
    if "TASKWEFT_SIGS_DIR" not in os.environ and sigs.is_dir():
        os.environ["TASKWEFT_SIGS_DIR"] = str(sigs)
    return p


def compiler_sha(compiler: Path) -> str:
    repo = compiler.parents[3]
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def compile_one(compiler: Path, xml: Path) -> tuple[bool, bool, dict | None, str]:
    """(parses, compiles, plan, refusal). compiles means check and plan both answered."""
    chk = subprocess.run([str(compiler), "check", str(xml)], capture_output=True, text=True, timeout=60)
    if chk.returncode != 0:
        return False, False, None, chk.stderr.strip() or chk.stdout.strip()
    pl = subprocess.run([str(compiler), "plan", str(xml)], capture_output=True, text=True, timeout=60)
    if pl.returncode != 0:
        return True, False, None, pl.stderr.strip() or pl.stdout.strip()
    return True, True, json.loads(pl.stdout), ""


def perform(plan: dict, scratch: Path) -> tuple[bool, int, str]:
    """Perform the plan's steps the way the host scene does; stop at the first failure."""
    lines: list[str] = []
    done = 0
    for step in plan["steps"]:
        kind = step["kind"]
        code = 1
        if kind == "write":
            target = scratch / step["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(step["content"], encoding="utf-8")
            code = 0
        elif kind == "read":
            code = 0 if (scratch / step["path"]).is_file() else 1
        elif kind == "terminal":
            command = sys.executable if step["command"] == "python" else step["command"]
            try:
                run = subprocess.run([command, *step["args"]], cwd=scratch, capture_output=True,
                                     text=True, timeout=30)
                code = run.returncode
                if run.stdout:
                    lines.append(run.stdout.rstrip("\n"))
            except (OSError, subprocess.TimeoutExpired) as e:
                lines.append(f"{step['action']} failed to start: {e}")
                code = 1
        lines.append(f"{step['action']} exit {code}")
        if code != 0:
            return False, done, "\n".join(lines)
        done += 1
    return True, done, "\n".join(lines)


def score_candidate(compiler: Path, row: Row, name: str, xml: Path, negative_control: bool) -> dict:
    t0 = time.perf_counter()
    parses, compiles, plan, refusal = compile_one(compiler, xml)
    runs, steps, effect = False, 0, False
    out = ""
    if compiles and plan is not None:
        scratch = Path(tempfile.mkdtemp(prefix="fbd_"))
        try:
            runs, steps, out = perform(plan, scratch)
            effect = runs and row.expect(scratch, out)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    # The negative control on the writer: hand rank5's verdict to rank1 and the
    # controls below must refuse the emit, or they certify nothing.
    if negative_control and name == "rank1":
        parses, compiles, runs, effect = False, False, False, False
    return {"candidate": name, "parses": parses, "compiles": compiles, "runs": runs,
            "effect_matches": effect, "steps": steps, "wall_ms": wall_ms, "refusal": refusal,
            "_plan": json.dumps(plan, sort_keys=True) if plan is not None else "",
            "_result": _run_record(runs, steps, out) if plan is not None else ""}


def _run_record(ran: bool, steps: int, log: str) -> str:
    """What running a plan returned, as the corpus stores it: text, never a null."""
    return json.dumps({"ran": ran, "steps": steps, "log": log}, sort_keys=True)


def _keep_plan(row_dir: Path, name: str, plan_json: str, result_json: str) -> None:
    """A row keeps the plan it ran and what running it returned; the corpus reads both."""
    (row_dir / f"{name}.plan.json").write_text(plan_json, encoding="utf-8")
    (row_dir / f"{name}.result.json").write_text(result_json, encoding="utf-8")


_rectgtn_cache: list[tuple[str, str]] = []


def rectgtn_row_plan() -> tuple[str, str]:
    """The RECTGTN domain's own plan, and its temporal verdict, once per run.

    The planner's answer does not vary by candidate, so it is asked for once. The
    verdict is the domain's own measurement, which is what the corpus records.
    """
    if not _rectgtn_cache:
        domain = os.environ.get("TASKWEFT_PLAN_DOMAIN", "")
        if not domain:
            raise SystemExit("FAIL: --plan-source rectgtn needs --plan-domain, a domain JSON-LD")
        merged = json.loads(Path(domain).read_text(encoding="utf-8"))
        problem = os.environ.get("TASKWEFT_PLAN_PROBLEM", "")
        if problem:
            merged = {**merged, **json.loads(Path(problem).read_text(encoding="utf-8"))}
        text = rectgtn_plan_json(merged)
        _rectgtn_cache.append((text, json.dumps(json.loads(text)["temporal"], sort_keys=True)))
    return _rectgtn_cache[0]


def assert_controls(key: str, scores: dict[str, dict]) -> None:
    r1, r3, r5 = scores["rank1"], scores["rank3"], scores["rank5"]
    if not (r1["compiles"] and r1["runs"] and r1["effect_matches"]):
        raise SystemExit(f"identity control failed on {key}: rank1 {r1}")
    if not r3["compiles"] or r3["effect_matches"]:
        raise SystemExit(f"effect control failed on {key}: rank3 must compile and miss the effect, got {r3}")
    if r5["compiles"]:
        raise SystemExit(f"negative control failed on {key}: rank5 compiled: {r5}")


def _read(path: Path) -> str:
    """A missing artefact is an empty string; ETNF forbids the null."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def build_row(compiler: Path, out: Path, template_id: str, seed: int, negative_control: bool) -> dict:
    row = row_for(template_id, seed)
    key = f"fbd/{template_id}/{seed}"
    row_dir = out / "rows" / template_id / str(seed)
    row_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict] = {}
    cands = []
    for name, rank in CANDIDATES:
        xml_path = row_dir / f"{name}.plcopen.xml"
        xml_path.write_text(getattr(row, name), encoding="utf-8")
        score = score_candidate(compiler, row, name, xml_path, negative_control)
        plan_text = score.pop("_plan")
        result_text = score.pop("_result")
        # The RECTGTN domain's own plan stands in for a candidate the compiler accepted;
        # a refused candidate has no plan, and its slot stays empty rather than borrowing one.
        if os.environ.get("TASKWEFT_PLAN_SOURCE") == "rectgtn" and score["compiles"]:
            plan_text, result_text = rectgtn_row_plan()
        _keep_plan(row_dir, name, plan_text, result_text)
        scores[name] = score
        cands.append({"candidate": name, "rank": rank,
                      "fbd_text": "",
                      "fbd_xml": getattr(row, name),
                      "plan_json": _read(row_dir / f"{name}.plan.json"),
                      "result_json": _read(row_dir / f"{name}.result.json"),
                      "fbd_path": f"rows/{template_id}/{seed}/{name}",
                      "fbd_sha": hashlib.sha256(getattr(row, name).encode("utf-8")).hexdigest(),
                      "provenance": f"constructed:{template_id}:seed:{seed}"})
    assert_controls(key, scores)
    xml_blocks = sorted(set(re.findall(r'typeName="([A-Z_]+)"', row.rank1)))
    return {"key": key, "intent": row.intent, "template_id": template_id, "seed": seed, "frame_id": row.frame_id,
            "blocks": xml_blocks, "traces": [],
            "candidates": cands, "scores": [scores[n] for n, _ in CANDIDATES]}


def run_compiler(compiler: Path, *args: str) -> tuple[int, str, str]:
    p = subprocess.run([str(compiler), *args], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def _same_outputs(a: list[list[dict]], b: list[list[dict]]) -> bool:
    if len(a) != len(b):
        return False
    for ta, tb in zip(a, b):
        if len(ta) != len(tb):
            return False
        for oa, ob in zip(ta, tb):
            if set(oa) != set(ob):
                return False
            for k in oa:
                x, y = oa[k], ob[k]
                if isinstance(x, str) or isinstance(y, str):
                    if x != y:
                        return False
                elif isinstance(x, bool) or isinstance(y, bool):
                    if bool(x) != bool(y):
                        return False
                elif abs(float(x) - float(y)) > 1e-6:
                    return False
    return True


def score_react(compiler: Path, row: ReactRow, name: str, fbd: Path, traces: list[Path], negative_control: bool,
                reference: list[list[dict]] | None = None) -> dict:
    """A controller: parses (the text form reads), compiles (the netlist builds), runs (the
    reference scan finishes every trace without a fault), effect (its outputs are the
    numbers the intent implies), steps = ticks simulated."""
    t0 = time.perf_counter()
    code, xml, err = run_compiler(compiler, "to-xml", str(fbd))
    parses = code == 0
    refusal = "" if parses else (err.strip() or "to-xml refused")
    compiles, runs, effect, steps = False, False, False, 0
    if parses:
        fbd.with_suffix(".plcopen.xml").write_text(xml, encoding="utf-8")
        code, _, err = run_compiler(compiler, "check", str(fbd))
        compiles = code == 0
        if not compiles:
            refusal = err.strip()
    if compiles and row.host == "godot":
        # an API program: the engine performs the calls and the returns are the effect
        code, out, err = run_compiler(compiler, "plan", str(fbd))
        if code != 0:
            refusal = err.strip()
            outs = []
        else:
            runs, results, err = perform_in_godot(out, fbd.parent, name)
            outs = [[{"results": json.dumps(results, sort_keys=True)}]]
            steps = len(results)
            if not runs:
                refusal = err
    elif compiles and row.host == "trainer":
        # a configuration program: the runner applies it to the task config in WSL and
        # the term table read back from the config is the effect
        code, out, err = run_compiler(compiler, "plan", str(fbd))
        if code != 0:
            refusal = err.strip()
            outs = []
        else:
            runs, tbl, err = perform_in_trainer(out)
            outs = [[{"table": json.dumps(tbl, sort_keys=True)}]]
            steps = len(json.loads(out)["steps"])
            _keep_plan(fbd.parent, name, out, _run_record(runs, steps, json.dumps(tbl, sort_keys=True)))
            if not runs:
                refusal = err
    elif compiles and not traces:
        # a step program from the planner: the effect is the plan itself, compared
        # step for step against rank1's (the host would perform mix and git, which a
        # scratch directory cannot)
        code, out, err = run_compiler(compiler, "plan", str(fbd))
        runs = code == 0
        outs = [[{"steps": json.dumps(json.loads(out)["steps"], sort_keys=True)}]] if runs else []
        steps = len(json.loads(out)["steps"]) if runs else 0
        _keep_plan(fbd.parent, name, out if runs else "",
                   _run_record(runs, steps, json.dumps(outs[0][0]["steps"], sort_keys=True) if runs else ""))
        if not runs:
            refusal = err.strip()
    elif compiles:
        outs: list[list[dict]] = []
        runs = True
        for trace in traces:
            code, out, err = run_compiler(compiler, "sim", str(fbd), str(trace))
            lines = [json.loads(l) for l in out.splitlines() if l.strip()]
            if code != 0:
                runs = False
                refusal = err.strip()
                break
            if row.expect is not None and any("fault" in l for l in lines):
                # a template with its own expectation never faults; a fault is a failure to run
                runs = False
                refusal = next(l["fault"] for l in lines if "fault" in l)
                break
            # differential rows keep a fault as an output: both sides must fault on the same tick
            outs.append([l["out"] if "out" in l else {"_fault": l["fault"]} for l in lines])
            steps += len(lines)
    else:
        outs = []
    if compiles and row.host == "godot":
        effect = runs and row.expect(json.loads(outs[0][0]["results"]) if outs else [])
    elif compiles and row.host == "trainer":
        effect = runs and row.expect(json.loads(outs[0][0]["table"]) if outs else {})
    elif compiles and row.expect is not None:
        effect = runs and row.expect(outs)
    elif compiles:
        # differential: rank1's own outputs are the reference for the others
        effect = runs and (reference is None or _same_outputs(outs, reference))
    wall_ms = int((time.perf_counter() - t0) * 1000)
    if negative_control and name == "rank1":
        parses, compiles, runs, effect = False, False, False, False
    score = {"candidate": name, "parses": parses, "compiles": compiles, "runs": runs,
             "effect_matches": effect, "steps": steps, "wall_ms": wall_ms, "refusal": refusal}
    score["_outputs"] = outs if runs else None
    return score


def controller_row_for(family: str, template_id: str, seed: int, compiler: Path) -> ReactRow:
    if family == "react":
        return react_row_for(template_id, seed)
    if family == "harness":
        return harness_row(template_id[len("harness_"):], seed, compiler)
    if family == "compose":
        return compose_row_for(template_id, seed)
    if family == "godot":
        return godot_row_for(template_id, seed)
    if family == "trainer":
        return trainer_row_for(template_id, seed)
    if family == "udon":
        return udon_row(template_id, seed, compiler, find_godot())
    if family == "plan":
        return plan_row(int(template_id.split("#")[1]), seed, compiler)
    raise ValueError(family)


def build_react_row(compiler: Path, out: Path, template_id: str, seed: int, negative_control: bool, family: str = "react") -> dict:
    row = controller_row_for(family, template_id, seed, compiler)
    template_id = row.template_id
    key = f"{family}/{template_id}/{seed}"
    row_dir = out / "rows" / template_id / str(seed)
    row_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for k, text in enumerate(row.traces):
        tp = row_dir / f"trace_{k}.json"
        tp.write_text(text + "\n", encoding="utf-8")
        traces.append(tp)
    scores: dict[str, dict] = {}
    cands = []
    reference = None
    for name, rank in CANDIDATES:
        fbd = row_dir / f"{name}.fbd"
        fbd.write_text(getattr(row, name), encoding="utf-8")
        scores[name] = score_react(compiler, row, name, fbd, traces, negative_control, reference)
        if name == "rank1":
            reference = scores[name].pop("_outputs")
        else:
            scores[name].pop("_outputs", None)
        xml = row_dir / f"{name}.plcopen.xml"
        cands.append({"candidate": name, "rank": rank,
                      "fbd_text": getattr(row, name),
                      "fbd_xml": xml.read_text(encoding="utf-8") if xml.is_file() else "",
                      "plan_json": _read(row_dir / f"{name}.plan.json"),
                      "result_json": _read(row_dir / f"{name}.result.json"),
                      "fbd_path": f"rows/{template_id}/{seed}/{name}",
                      "fbd_sha": hashlib.sha256(getattr(row, name).encode("utf-8")).hexdigest(),
                      "provenance": f"constructed:{family}:{template_id}:seed:{seed}"})
    assert_controls(key, scores)
    return {"key": key, "intent": row.intent, "template_id": template_id, "seed": seed, "frame_id": row.frame_id,
            "blocks": sorted(set(signatures_of(row.rank1) if row.host in ("godot", "trainer") else blocks_of(row.rank1))),
            "traces": list(row.traces),
            "candidates": cands, "scores": [scores[n] for n, _ in CANDIDATES]}


def tables(rows: list[dict], stub: tuple = STUB) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    task_type, dimension, input_column, input_asset_kind = stub[1], stub[2], stub[3], stub[4]
    root = pa.table({
        "key": [r["key"] for r in rows],
        "task_type": [task_type] * len(rows),
        "dimension": [dimension] * len(rows),
        "input_column": [input_column] * len(rows),
        "input_asset_kind": [input_asset_kind] * len(rows),
        "intent": [r["intent"] for r in rows],
        "template_id": [r["template_id"] for r in rows],
        "seed": pa.array([r["seed"] for r in rows], pa.int64()),
        "frame_id": pa.array([r.get("frame_id", 0) for r in rows], pa.int64()),
        "blocks": pa.array([",".join(r.get("blocks", [])) for r in rows], pa.string()),
        "traces": pa.array([r.get("traces", []) for r in rows], pa.list_(pa.string())),
        "provenance": ["constructed:template"] * len(rows),
    })
    cand_rows = [{"row_key": r["key"], **c} for r in rows for c in r["candidates"]]
    candidates = pa.Table.from_pylist(cand_rows)
    score_rows = [{"row_key": r["key"], **s} for r in rows for s in r["scores"]]
    scores = pa.Table.from_pylist(score_rows)
    joined = pa.Table.from_pylist([{
        "key": r["key"], "intent": r["intent"], "template_id": r["template_id"], "seed": r["seed"],
        "candidates": [{**c, "scores": s} for c, s in zip(r["candidates"], r["scores"])],
    } for r in rows])
    return root, candidates, scores, joined


def write_stage(stage: Path, rows: list[dict], split: str, family: str = "fbd") -> dict:
    stub = STUBS[family]
    root, candidates, scores, joined = tables(rows, stub)
    counts = {}
    for name, table in [(f"{family}_root", root), (f"{family}_candidates", candidates), (f"{family}_scores", scores), (family, joined)]:
        for col in table.column_names:
            n = table.column(col).null_count
            if n:
                raise SystemExit(f"FAIL: {name}.{col} carries {n} null(s); ETNF forbids them")
        d = stage / "data" / name
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / f"{split}-00000-of-00001.parquet", compression="zstd")
        counts[name] = table.num_rows
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=HERE / "work" / "stage")
    ap.add_argument("--compiler", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--family", choices=list(STUBS), default="fbd",
                    help="fbd: operating-system step programs; react: per-frame controllers; harness: every block kind; "
                         "compose: two controllers in one program; plan: the planner's goals")
    ap.add_argument("--holdout-families", default="", help="template ids written to the evaluation split, never train")
    ap.add_argument("--holdout-blocks", default="", help="block kinds whose rows go to the evaluation split, never train")
    ap.add_argument("--negative-control", action="store_true",
                    help="hand rank5's verdict to rank1; the controls must refuse the emit")
    ap.add_argument("--plan-source", choices=["compiler", "rectgtn"], default="compiler",
                    help="compiler: the plan the candidate's own program compiles to; "
                         "rectgtn: the RECTGTN domain's own plan for --plan-domain")
    ap.add_argument("--plan-domain", default=None, help="a domain JSON-LD (with --plan-source rectgtn)")
    ap.add_argument("--plan-problem", default=None, help="a problem JSON-LD, composed over the domain")
    args = ap.parse_args()

    os.environ["TASKWEFT_PLAN_SOURCE"] = args.plan_source
    if args.plan_domain:
        os.environ["TASKWEFT_PLAN_DOMAIN"] = args.plan_domain
    if args.plan_problem:
        os.environ["TASKWEFT_PLAN_PROBLEM"] = args.plan_problem

    compiler = find_compiler(args.compiler)
    if compiler_sha(compiler) == "unknown":
        sys.exit(f"FAIL: the compiler at {compiler} has no committed sha (built outside its repository); build it in-tree")
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    family = args.family
    if family == "fbd":
        template_ids = list(TEMPLATES)
    elif family == "react":
        template_ids = list(REACT_TEMPLATES)
    elif family == "harness":
        ensure_generated(compiler)
        template_ids = [f"harness_{n}" for n in harness_names()]
    elif family == "compose":
        template_ids = [f"compose_{i}" for i in range(len(PAIRS))]
    elif family == "godot":
        template_ids = list(GODOT_TEMPLATES)
    elif family == "trainer":
        template_ids = list(TRAINER_TEMPLATES)
    elif family == "udon":
        template_ids = list(UDON_TEMPLATES)
    else:
        template_ids = [f"plan#{i}" for i in range(len(GOALS))]
    if family == "fbd":
        builder = build_row
    else:
        builder = lambda c, o, t, sd, nc: build_react_row(c, o, t, sd, nc, family)  # noqa: E731
    holdout_families = {f for f in args.holdout_families.split(",") if f}
    holdout_blocks = {b for b in args.holdout_blocks.split(",") if b}
    jobs = [(template_ids[(i // 10) % len(template_ids)], i) for i in range(args.rows)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: builder(compiler, out, j[0], j[1], args.negative_control), jobs))
    wall = time.perf_counter() - t0

    def evaluation(r: dict) -> bool:
        return r["template_id"] in holdout_families or any(b in holdout_blocks for b in r.get("blocks", []))
    eval_rows = [r for r in rows if evaluation(r)]
    rest = [r for r in rows if not evaluation(r)]
    train = [r for r in rest if r["seed"] % 10 != 0]
    test = [r for r in rest if r["seed"] % 10 == 0]
    counts = {"train": write_stage(out, train, "train", family),
              "test": write_stage(out / "test", test, "test", family),
              "evaluation": write_stage(out / "evaluation", eval_rows, "evaluation", family)}
    manifest = {
        "stub": STUBS[family], "family": family, "rows": len(rows), "train_rows": len(train), "test_rows": len(test),
        "evaluation_rows": len(eval_rows), "holdout_families": sorted(holdout_families), "holdout_blocks": sorted(holdout_blocks),
        "splits": {"train": "trains", "test": "same-distribution seed holdout (every tenth seed), the gate after training",
                   "evaluation": "held-out families and block kinds, never trained or tuned on"},
        "templates": {t: sum(1 for r in rows if r["template_id"] == t) for t in template_ids},
        "controls": "rank1 compiles, runs and matches; rank3 compiles and misses; rank5 is refused; asserted on every row",
        "plan_source": args.plan_source, "plan_domain": args.plan_domain or "",
        "compiler": Path(compiler).name, "compiler_sha": compiler_sha(compiler), **godot_record(),
        "wall_s": round(wall, 1), "counts": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
