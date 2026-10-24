"""Intents and the per-frame controllers they mean (RFD66606-2237), constructed so the
labels are true by construction.

Each template turns a seed into an intent, the reference controller in the text form
(rank1), a controller that compiles and reacts wrongly (rank3: a wrong gain, swapped
axes, a timer too long, a wrong style), and one the compiler refuses (rank5: a BOOL
into an arithmetic pin, an unknown input, an output written twice, an output declared
as an input). Every row carries constructed input traces; the effect check compares
the reference scan's outputs on those traces against the numbers the intent implies.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable

from frames import pick

TICKS = 12
# A binary fraction, so n frames of dt sum to exactly n * dt and a hold of n frames
# expires on the frame the intent names (0.1 s frames put 0.8 s at 0.7999999999999999).
DT = 0.125
EPS = 1e-6


@dataclass
class ReactRow:
    template_id: str
    seed: int
    intent: str
    rank1: str
    rank3: str
    rank5: str
    traces: list[str]
    expect: Callable[[list[list[dict]]], bool] | None
    frame_id: int = 0
    host: str = ""


def _trace(ticks: list[dict]) -> str:
    return json.dumps({"dt": DT, "ticks": ticks})


def _sweep(rng: random.Random) -> list[dict]:
    # The first tick pins the stick to its corners, so a halved speed cap or a
    # swapped axis shows on every trace rather than only when the draw happens to.
    ticks = [{"lx": round(rng.uniform(-1.0, 1.0), 3), "ly": round(rng.uniform(-1.0, 1.0), 3),
              "btn": False, "tracker_ok": True} for _ in range(TICKS)]
    ticks[0]["lx"], ticks[0]["ly"] = 1.0, -1.0
    return ticks


def _pulse(hold_ticks: int) -> list[dict]:
    return [{"lx": 0.0, "ly": 0.0, "btn": i == 1, "tracker_ok": True} for i in range(TICKS)]


def _drop(at: int) -> list[dict]:
    return [{"lx": 0.0, "ly": 0.0, "btn": False, "tracker_ok": i < at} for i in range(TICKS)]


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= EPS


def _limit(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _fmt(x: float) -> str:
    s = f"{x:.9f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


HEAD = "in lx : REAL\nin ly : REAL\nin btn : BOOL\nin tracker_ok : BOOL\n"


def t_walk_from_stick(seed: int) -> ReactRow:
    rng = random.Random(seed)
    gain = round(rng.uniform(0.8, 1.5), 2)
    cap = round(rng.uniform(0.6, 2.0), 2)
    wrong_gain = round(gain + 0.5, 2)

    def prog(g: float, c: float, mn: str = "0.0") -> str:
        return (f"program walk_from_stick\n{HEAD}out move_x : REAL\nout move_z : REAL\nout speed : REAL\n"
                f"mx = MUL(IN1=lx, IN2={_fmt(g)})\nmz = MUL(IN1=ly, IN2={_fmt(g)})\n"
                f"sp = LIMIT(MN={mn}, IN=mx.OUT, MX={_fmt(c)})\n"
                f"move_x = mx.OUT\nmove_z = mz.OUT\nspeed = sp.OUT\n")

    traces = [_sweep(rng), _sweep(rng), _drop(6)]

    def expect(outs: list[list[dict]]) -> bool:
        for ticks, out in zip(traces, outs):
            for t, o in zip(ticks, out):
                want = _limit(t["lx"] * gain, 0.0, cap)
                if not (_close(o["move_x"], t["lx"] * gain) and _close(o["move_z"], t["ly"] * gain)
                        and _close(o["speed"], want)):
                    return False
        return True

    fid, intent = pick("walk_from_stick", seed, gain=gain, cap=cap)
    return ReactRow("walk_from_stick", seed, intent,
                    prog(gain, cap), prog(wrong_gain, cap), prog(gain, cap, mn="TRUE"),
                    [_trace(t) for t in traces], expect, fid)


def t_face_the_stick(seed: int) -> ReactRow:
    rng = random.Random(seed)

    def prog(x: str, z: str) -> str:
        return (f"program face_the_stick\n{HEAD}out face_x : REAL\nout face_z : REAL\n"
                f"fx = MOVE(IN={x})\nfz = MOVE(IN={z})\nface_x = fx.OUT\nface_z = fz.OUT\n")

    traces = [_sweep(rng), _sweep(rng), _sweep(rng)]

    def expect(outs: list[list[dict]]) -> bool:
        for ticks, out in zip(traces, outs):
            for t, o in zip(ticks, out):
                if not (_close(o["face_x"], t["lx"]) and _close(o["face_z"], t["ly"])):
                    return False
        return True

    fid, intent = pick("face_the_stick", seed)
    return ReactRow("face_the_stick", seed, intent,
                    prog("lx", "ly"), prog("ly", "lx"), prog("lx", "lz"),
                    [_trace(t) for t in traces], expect, fid)


def t_tracker_lost_freezes(seed: int) -> ReactRow:
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    ms = n * 125

    def prog(pt: str) -> str:
        return (f"program tracker_lost_freezes\n{HEAD}out chain_frozen : BOOL\n"
                f"nt = NOT(IN=tracker_ok)\nlost = TON(IN=nt.OUT, PT={pt})\nchain_frozen = lost.Q\n")

    drops = [rng.randint(2, 5) for _ in range(3)]
    traces = [_drop(at) for at in drops]

    def expect(outs: list[list[dict]]) -> bool:
        for at, out in zip(drops, outs):
            for i, o in enumerate(out):
                want = i >= at + n - 1
                if bool(o["chain_frozen"]) != want:
                    return False
        return True

    fid, intent = pick("tracker_lost_freezes", seed, ms=ms)
    return ReactRow("tracker_lost_freezes", seed, intent,
                    prog(f"T#{ms}ms"), prog(f"T#{2 * ms}ms"), prog("TRUE"),
                    [_trace(t) for t in traces], expect, fid)


def t_button_sequence_then_idle(seed: int) -> ReactRow:
    rng = random.Random(seed)
    style = rng.randint(1, 9)
    n = rng.randint(3, 8)
    ms = n * 125

    def prog(k: int, tail: str = "style = st.OUT\n") -> str:
        return (f"program button_sequence_then_idle\n{HEAD}out style : INT\nvar greeting : BOOL\n"
                f"press = R_TRIG(CLK=btn)\nhold = TON(IN=greeting, PT=T#{ms}ms)\n"
                f"g = SR(S1=press.Q, R=hold.Q)\nst = SEL(G=g.Q1, IN0=0, IN1={k})\n"
                f"greeting = g.Q1\n{tail}")

    traces = [_pulse(n), _pulse(n), _drop(4)]

    def expect(outs: list[list[dict]]) -> bool:
        for idx, out in enumerate(outs):
            for i, o in enumerate(out):
                want = style if (idx < 2 and 1 <= i <= n) else 0
                if int(o["style"]) != want:
                    return False
        return True

    fid, intent = pick("button_sequence_then_idle", seed, style=style, ms=ms)
    return ReactRow("button_sequence_then_idle", seed, intent,
                    prog(style), prog(style + 1 if style < 9 else style - 1),
                    prog(style, tail="style = st.OUT\nstyle = press.Q\n"),
                    [_trace(t) for t in traces], expect, fid)


def t_speed_limited_run(seed: int) -> ReactRow:
    rng = random.Random(seed)
    gain = round(rng.uniform(1.5, 3.0), 2)
    cap = round(rng.uniform(1.0, 2.5), 2)

    def prog(c: float, decl: str = "out speed : REAL\n") -> str:
        return (f"program speed_limited_run\n{HEAD}{decl}"
                f"mx = MUL(IN1=lx, IN2={_fmt(gain)})\nsp = LIMIT(MN=0.0, IN=mx.OUT, MX={_fmt(c)})\nspeed = sp.OUT\n")

    traces = [_sweep(rng), _sweep(rng), _sweep(rng)]

    def expect(outs: list[list[dict]]) -> bool:
        for ticks, out in zip(traces, outs):
            for t, o in zip(ticks, out):
                if not _close(o["speed"], _limit(t["lx"] * gain, 0.0, cap)):
                    return False
        return True

    fid, intent = pick("speed_limited_run", seed, gain=gain, cap=cap)
    return ReactRow("speed_limited_run", seed, intent,
                    prog(cap), prog(round(cap / 2, 2)), prog(cap, decl="in speed : REAL\n"),
                    [_trace(t) for t in traces], expect, fid)


REACT_TEMPLATES = {
    "walk_from_stick": t_walk_from_stick,
    "face_the_stick": t_face_the_stick,
    "tracker_lost_freezes": t_tracker_lost_freezes,
    "button_sequence_then_idle": t_button_sequence_then_idle,
    "speed_limited_run": t_speed_limited_run,
}


def react_row_for(template_id: str, seed: int) -> ReactRow:
    return REACT_TEMPLATES[template_id](seed)
