"""Self-check: the RECTGTN domain plans a real domain, and its answer is the
plan `plan_json` carries.

Run against a taskweft checkout (`TASKWEFT_DIR`, default `../taskweft`):

    python tools/proof_rectgtn_plan.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rectgtn_plan import plan_json_of  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TW = Path(os.environ.get("TASKWEFT_DIR", HERE.parent / "taskweft"))

domain_path = TW / "priv/smoke/blocks_world.jsonld"
problem_path = TW / "priv/plans/problems/blocks_world_1a.jsonld"
expected_path = TW / "priv/plans/expected/blocks_world__blocks_world_1a_expected.json"
for p in (domain_path, problem_path, expected_path):
    if not p.is_file():
        sys.exit(f"FAIL: no {p}; point TASKWEFT_DIR at a taskweft checkout")

domain = json.loads(domain_path.read_text(encoding="utf-8"))
problem = json.loads(problem_path.read_text(encoding="utf-8"))
expected = json.loads(expected_path.read_text(encoding="utf-8"))

plan = json.loads(plan_json_of({**domain, **problem}))
print("temporal:", json.dumps(plan["temporal"]["consistent"]))
print("plan:", json.dumps(plan["plan"]))

if plan["plan"] != expected["plan"]:
    sys.exit(f"FAIL: the domain planned {plan['plan']}, the checkout expects {expected['plan']}")
if set(plan["temporal"]) != {"consistent", "origin", "steps", "total"}:
    sys.exit(f"FAIL: the temporal block is not intrinsic: {sorted(plan['temporal'])}")
print("OK: the RECTGTN domain plans blocks_world_1a as the checkout expects, "
      "and its answer carries the intrinsic temporal block")
