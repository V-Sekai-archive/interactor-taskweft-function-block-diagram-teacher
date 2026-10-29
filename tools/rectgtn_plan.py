"""The RECTGTN domain's own plan, the way the door reaches it.

`plan_json` carries the plan of a row, and the plan of a goal is the RECTGTN
domain's: the planner in `taskweft-nif` (`Taskweft.plan/1`), whose answer holds
the intrinsic temporal block. One planner process per worker thread, one JSON
line per domain, so a batch does not pay the planner's start-up per row.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NIF_DIR = Path(os.environ.get("TASKWEFT_NIF_DIR", HERE.parent / "taskweft-nif"))
SERVER = Path(__file__).resolve().parent / "rectgtn_plan_server.exs"

_local = threading.local()


class PlanServer:
    """One RECTGTN planner process, spoken to one domain at a time."""

    def __init__(self) -> None:
        if not (NIF_DIR / "mix.exs").is_file():
            raise RuntimeError(f"no taskweft-nif checkout at {NIF_DIR}; set TASKWEFT_NIF_DIR")
        self.p = subprocess.Popen(["mix", "run", str(SERVER)], cwd=NIF_DIR, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", bufsize=1)
        self.out, self.inp = self.p.stdout, self.p.stdin
        if self.out is None or self.inp is None:
            raise RuntimeError("the RECTGTN planner's pipes did not open")
        for line in self.out:
            if line.startswith("{"):
                if json.loads(line).get("ready"):
                    break
                continue
        else:
            raise RuntimeError("the RECTGTN planner did not come up")

    def plan(self, domain: dict) -> dict:
        """The domain's plan JSON, parsed. Raises when the planner refuses."""
        self.inp.write(json.dumps({"domain": domain}) + "\n")
        self.inp.flush()
        for line in self.out:
            if line.startswith("{"):
                answer = json.loads(line)
                if not answer.get("ok"):
                    raise RuntimeError(f"the RECTGTN planner refused: {answer.get('error')}")
                plan = answer["plan"]
                return json.loads(plan) if isinstance(plan, str) else plan
        raise RuntimeError("the RECTGTN planner went away")


def server() -> PlanServer:
    srv = getattr(_local, "srv", None)
    if srv is None:
        srv = _local.srv = PlanServer()
    return srv


def plan_of(domain: dict) -> dict:
    return server().plan(domain)


def plan_json_of(domain: dict) -> str:
    """The plan as the text the corpus stores: compact, key order kept."""
    return json.dumps(plan_of(domain), sort_keys=True)
