"""Task 014 (E4) world gates on dev seeds (house = NAV116_HOUSE).

    MUJOCO_GL=egl NAV116_HOUSE=A python -m sim.e4_gates --n 20 --out OUT.json

- C2 relation: make_task_e4's place_ref draws on dev seeds 2000.. (the generator rejects ambiguous draws); reports the
  anchor counts and the yield (tasks / seeds tried).
- C3 move: for the first n dev recovery tasks: a fresh SimRobot, base set at the target's room point (its first room
  point), look, locate the target with the harness installed (sim.e4.install_move). Bar: every task whose target is
  located there gets moved (a candidate out of view from the robot and pickable exists); the moved box is then at a
  floor spot of another room, not the robot's.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import mujoco

from sim import tasks as T
from sim.e4 import install_move, make_task_e4
from sim.grasp import tilt_deg
from sim.world import generate as gen

REST_S, REST_XY_M, REST_TILT_DEG = 1.0, 0.01, 5.0


def _rest_and_pick(r, row: dict, box: str, room: str) -> None:
    body = r.d.body(r.body_of[box])
    xy0 = body.xpos[:2].copy()
    for _ in range(math.ceil(REST_S / r.m.opt.timestep)):
        mujoco.mj_step(r.m, r.d)
    disp, tilt = float(math.dist(body.xpos[:2], xy0)), float(tilt_deg(body.xmat))
    row.update(rest_disp_m=disp, rest_tilt_deg=tilt, rest_ok=disp < REST_XY_M and tilt < REST_TILT_DEG)
    nav = r.navigate(r.room_name[room])
    r.look()
    found = r.locate(box).pose is not None
    res = r.pick(box) if found else None
    row.update(nav=nav.status, located_again=found,
               pick_ok=bool(res and res.ok), pick_reason=None if res is None else res.reason,
               held=r.held() == box)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from sim.adapter import SimRobot
    from sim.world.house import set_base
    rel, tried = [], 0
    for seed in range(2000, 2400):
        tried += 1
        try:
            rel.append(make_task_e4(seed, "place_ref"))
        except T.NoTask:
            continue
        if len(rel) == a.n:
            break
    c2 = {"tasks": len(rel), "seeds_tried": tried, "anchors": dict(collections.Counter(t["anchor"] for t in rel))}
    rows = []
    seed = 2000
    while len(rows) < a.n and seed < 3000:
        try:
            t = make_task_e4(seed, "recovery")
        except T.NoTask:
            seed += 1
            continue
        seed += 1
        box = t["targets"][0]["object"]
        w = T.world(t["seed"], "none")
        b = next(x for x in w.boxes if T.box_name(x["colour"]) == box)
        x, y = gen.room_points(b["room"])[0]
        r = SimRobot(t["seed"], "none")
        set_base(r.m, r.d, (x, y, 0.0))
        log = install_move(r, t)
        r.look()
        found = r.locate(box).pose is not None
        row = {"task_id": t["task_id"], "box_room": b["room"], "located": found, "moved": log["moved"],
               "to": log.get("to"), "tried": log.get("tried"), "view_px": log.get("view_px")}
        if log["moved"]:
            nx, ny = (float(v) for v in r.d.body(r.body_of[box]).xpos[:2])
            row["at_spot"] = gen.spot_ok(w.layout, log["to"][0], nx, ny)
            row["other_room"] = log["to"][0] != b["room"]
            _rest_and_pick(r, row, box, log["to"][0])
        rows.append(row)
    located = [r for r in rows if r["located"]]
    ok = [r for r in located if r["moved"] and r.get("at_spot") and r.get("other_room")]
    moved = [r for r in rows if r["moved"]]
    rest = [r for r in moved if r.get("rest_ok")]
    picks = [r for r in moved if r.get("pick_ok")]
    c3 = {"n": len(rows), "located": len(located), "moved_ok": len(ok), "moved": len(moved), "rest_pass": len(rest),
          "rest_max_disp_m": max((r["rest_disp_m"] for r in moved), default=None),
          "rest_max_tilt_deg": max((r["rest_tilt_deg"] for r in moved), default=None),
          "located_again": sum(1 for r in moved if r.get("located_again")), "pick_ok": len(picks),
          "pick_held": sum(1 for r in moved if r.get("held")),
          "pick_reasons": dict(collections.Counter(r.get("pick_reason") for r in moved if not r.get("pick_ok"))),
          "pick_rate": len(picks) / len(moved) if moved else None,
          "pass": len(ok) == len(located) and len(located) > 0 and len(rest) == len(moved)}
    out = {"house": gen.HOUSE, "c2": c2, "c3": c3 | {"rows": rows}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"house": gen.HOUSE, "c2": c2, "c3": c3}))


if __name__ == "__main__":
    main()
