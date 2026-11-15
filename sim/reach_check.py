"""G1-reach (task 002): every place in places.yaml, G1_REACH_TRIES tries each, start = a seeded random OTHER place.

    python -m sim.reach_check --out runs/1g/reach.json [--no-boxes]

World: generate(seed=0, blocker="none"), its boxes as generated (not on the known map); `--no-boxes` drops the box
bodies from the XML (same house otherwise), for the case where a box on a corridor breaks tries. Per try: fresh
MjData, base set at the start place pose (set_base, a start-state set, not a drive), arm set to LOOK_ARM_QPOS,
settle START_SETTLE_S, then `go_to` the goal place. Pass: final pos error <= G1_REACH_POS_TOL and yaw error <=
G1_REACH_YAW_TOL_DEG after go_to's settle. Gate: G1_REACH_TRIES / G1_REACH_TRIES for every place.
Start draw: rng = numpy default_rng(--rng-seed); for place i, try k: start = rng.choice(the 6 other places), in
place order then try order.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import mujoco
import numpy as np

from sim.body import run
from sim.constants import G1_REACH_POS_TOL, G1_REACH_TRIES, G1_REACH_YAW_TOL_DEG
from sim.look import set_arm
from sim.nav.follow import go_to
from sim.robot import GRIPPER_ACT
from sim.world.generate import World, generate, load_places
from sim.world.house import GRIP_OPEN, house_spec, set_base
from sim.world.occupancy import known_map

START_SETTLE_S = 0.5


def compile_house(w: World) -> mujoco.MjModel:
    import warnings

    spec, _ = house_spec(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()


def start_state(m: mujoco.MjModel, pose: tuple[float, float, float]) -> mujoco.MjData:
    d = mujoco.MjData(m)
    set_base(m, d, pose)
    set_arm(m, d)  # LOOK_ARM_QPOS, qpos + ctrl, forward
    d.ctrl[m.actuator(GRIPPER_ACT).id] = GRIP_OPEN
    run(m, d, START_SETTLE_S)
    return d


def pose_of(p: dict) -> tuple[float, float, float]:
    x, y, yaw_deg = p["pose"]
    return float(x), float(y), math.radians(float(yaw_deg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--no-boxes", action="store_true")
    a = ap.parse_args()
    w = generate(seed=0, blocker="none")
    if a.no_boxes:
        w.xml = re.sub(r'    <body name="box_[a-z]+".*?</body>\n', "", w.xml, flags=re.S)
        assert "box_" not in w.xml
    m = compile_house(w)
    grid = known_map(w.layout)
    places = load_places()
    rng = np.random.default_rng(a.rng_seed)
    tries, per_place = [], {}
    wall0, steps0 = time.time(), 0
    for p in places:
        others = [q for q in places if q["name"] != p["name"]]
        rows = []
        for k in range(G1_REACH_TRIES):
            s = others[int(rng.integers(len(others)))]
            d = start_state(m, pose_of(s))
            t0 = d.time
            r = go_to(m, d, grid, pose_of(p))
            steps0 += int(round(d.time / m.opt.timestep))
            row = {"goal": p["name"], "start": s["name"], "try": k, "pass": r["reached"],
                   "pos_err_m": r["pos_err_m"], "yaw_err_deg": r["yaw_err_deg"], "body_pos_err_m": r.get("body_pos_err_m"),
                   "body_yaw_err_deg": r.get("body_yaw_err_deg"), "path_len_m": r["path_len_m"],
                   "sim_s": r["sim_s"], "max_track_err_m": r["max_track_err_m"], "contacts": r["contacts"],
                   "waypoints": r["waypoints"], "start_settle_s": t0}
            rows.append(row)
            print(f"{p['name']:>14} <- {s['name']:<14} {'PASS' if row['pass'] else 'FAIL'} pos {row['pos_err_m']:.2e} "
                  f"yaw {row['yaw_err_deg']:.2e} len {row['path_len_m']} sim {row['sim_s']:.1f}s "
                  f"track {row['max_track_err_m']:.2e} contacts {row['contacts']}", flush=True)
        per_place[p["name"]] = {
            "passed": sum(r["pass"] for r in rows), "tries": len(rows),
            "max_pos_err_m": max(max(r["pos_err_m"], r["body_pos_err_m"]) for r in rows),
            "max_yaw_err_deg": max(max(r["yaw_err_deg"], r["body_yaw_err_deg"]) for r in rows),
            "max_track_err_m": max(r["max_track_err_m"] for r in rows),
            "contact_tries": sum(bool(r["contacts"]) for r in rows),
            "wall_furniture_contact_tries": sum(any(not k.startswith("box_") for k in r["contacts"]) for r in rows),
        }
        tries += rows
    wall = time.time() - wall0
    gate = all(v["passed"] == v["tries"] == G1_REACH_TRIES for v in per_place.values())
    res = {"gate": "G1-reach", "pass": gate, "world": {"seed": 0, "blocker": "none", "boxes": not a.no_boxes,
           "layout_sha": w.layout_sha, "box_poses": w.boxes}, "rng_seed": a.rng_seed,
           "bars": {"pos_tol_m": G1_REACH_POS_TOL, "yaw_tol_deg": G1_REACH_YAW_TOL_DEG, "tries": G1_REACH_TRIES},
           "per_place": per_place, "wall_s": wall, "sim_steps": steps0, "steps_per_s": steps0 / wall,
           "mujoco": mujoco.__version__, "tries": tries}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for n, v in per_place.items():
        print(f"{n:>14}: {v['passed']}/{v['tries']} max pos {v['max_pos_err_m']:.2e} m, max yaw "
              f"{v['max_yaw_err_deg']:.2e} deg, max track {v['max_track_err_m']:.2e} m, tries with contacts "
              f"{v['contact_tries']} (wall/furniture {v['wall_furniture_contact_tries']})")
    print(f"G1-reach {'PASS' if gate else 'FAIL'}; wall {wall:.1f} s, {steps0} steps, {steps0 / wall:.0f} steps/s -> {a.out}")


if __name__ == "__main__":
    main()
