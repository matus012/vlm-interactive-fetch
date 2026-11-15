"""G1-rest: the house left alone stays put.

    python -m sim.world.rest_check --seeds 0 1 2 3 --out runs/1w/rest.json

Per (seed, blocker): house + robot (retract arm, base holding its spawn pose with its own position actuators),
boxes / crate start generate.SETTLE_DROP above their support. Settle SETTLE_S of sim time, record every box's and the
blocker's pose, then step G1_REST_SIM_S, checking every CHECK_S: no NaN in qpos/qvel; drift = max over checks of the
position distance (and rotation angle) from the post-settle pose. Pass: every box and the crate / cabinet drift
<= G1_REST_MAX_DRIFT. Nothing is frozen: plain mj_step, the world's timestep and solver as generated.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim.constants import G1_REST_MAX_DRIFT, G1_REST_SIM_S
from sim.world.generate import LAYOUT, SETTLE_DROP, generate
from sim.world.house import load_house

SETTLE_S = 1.0
CHECK_S = 1.0


def _rot_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


def rest_one(seed: int, blocker: str, sim_s: float = G1_REST_SIM_S) -> dict[str, Any]:
    w = generate(LAYOUT, seed, blocker)
    m, d = load_house(w)
    names = [b["name"] for b in w.boxes] + ([w.blocker["body"]] if w.blocker["body"] else [])
    ids = [m.body(n).id for n in names]
    mujoco.mj_step(m, d, nstep=int(round(SETTLE_S / m.opt.timestep)))
    p0 = {n: d.xpos[i].copy() for n, i in zip(names, ids)}
    q0 = {n: d.xquat[i].copy() for n, i in zip(names, ids)}
    drift = {n: 0.0 for n in names}
    rot = {n: 0.0 for n in names}
    nan = False
    per = int(round(CHECK_S / m.opt.timestep))
    t = time.perf_counter()
    for _ in range(int(round(sim_s / CHECK_S))):
        mujoco.mj_step(m, d, nstep=per)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            nan = True
            break
        for n, i in zip(names, ids):
            drift[n] = max(drift[n], float(np.linalg.norm(d.xpos[i] - p0[n])))
            rot[n] = max(rot[n], _rot_deg(q0[n], d.xquat[i]))
    wall = time.perf_counter() - t
    boxes = [n for n in names if n.startswith("box_")]
    out: dict[str, Any] = {
        "seed": seed, "blocker": blocker, "door": w.blocker["door"], "sim_s": sim_s, "settle_s": SETTLE_S,
        "settle_drop_m": SETTLE_DROP, "nan": nan,
        "surfaces": {b["name"]: b["surface"] for b in w.boxes},
        "drift_m": drift, "rot_drift_deg": rot,
        "max_box_drift_m": max(drift[n] for n in boxes), "max_box_rot_deg": max(rot[n] for n in boxes),
        "blocker_drift_m": drift.get(w.blocker["body"]) if w.blocker["body"] else None,
        "blocker_rot_deg": rot.get(w.blocker["body"]) if w.blocker["body"] else None,
        "steps_per_s": sim_s / m.opt.timestep / wall, "timestep": m.opt.timestep,
    }
    out["pass"] = bool(not nan and all(v <= G1_REST_MAX_DRIFT for v in drift.values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sim-s", type=float, default=G1_REST_SIM_S)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [(s, b) for s in a.seeds for b in ("none", "crate", "cabinet")]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(rest_one, *zip(*runs), [a.sim_s] * len(runs)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"gate": "G1-rest", "bar_m": G1_REST_MAX_DRIFT, "pass": all(r["pass"] for r in res), "runs": res}
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    for r in res:
        bd = "-" if r["blocker_drift_m"] is None else f'{r["blocker_drift_m"] * 1e3:.4f}'
        print(f's{r["seed"]} {r["blocker"]:7s} box {r["max_box_drift_m"] * 1e3:.4f} mm {r["max_box_rot_deg"]:.4f} deg'
              f' blocker {bd} mm nan {r["nan"]} {r["steps_per_s"]:.0f} steps/s pass {r["pass"]}')
    print(f'G1-rest {"PASS" if summary["pass"] else "FAIL"} -> {a.out}')


if __name__ == "__main__":
    main()
