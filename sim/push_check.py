"""Dev gates G2-push-crate / G2-push-cabinet (task 003) on `sim.push.push_object`, dev seeds 1000+.

    python -m sim.push_check --out runs/2c/push_dev.json [--seeds 10] [--first 1000]

Crate (per seed): generate(seed, blocker="crate") (house A: door d_bed or d_sto by seed), then JITTER in this harness (the
generator is not touched here): rng = default_rng([seed, 2]); crate moved by U(-JIT_M, JIT_M) along the passage,
U(-JIT_M, JIT_M) across it and yawed by U(-JIT_DEG, JIT_DEG), by writing its free-joint qpos before the settle.
Robot: START_D in front of the doorway on the hall side, centred on the door axis, facing it; arm LOOK_ARM_QPOS;
settle. Then push_object, then go_to the far room's room point (the door's non-hall room) (through the doorway). Pass: push ok, go_to reached,
sim finite. Bar >= 9/10.
Cabinet (per seed): generate(seed, blocker="cabinet") (house A: d_lk1 or d_lk2), robot START_D in front on the living-room
side (house A), push_object. Pass: ok False with reason "not movable", cabinet moved <= G2_CABINET_MAX_MOVE, robot-cabinet
penetration <= G2_CABINET_MAX_PEN, sim finite. Bar 10/10.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim.body import run
from sim.constants import G2_CABINET_MAX_MOVE, G2_CABINET_MAX_PEN
from sim.look import robot_geoms
from sim.nav.follow import go_to
from sim.push import push_object
from sim.reach_check import compile_house, pose_of, start_state
from sim.tasks import detour_pair
from sim.world.generate import generate, load_places
from sim.world.occupancy import known_map

JIT_M, JIT_DEG = 0.03, 3.0
START_D = 1.2
SETTLE_S = 0.5


def room_place(room: str, places: list[dict[str, Any]] | None = None) -> str:
    """The VLM place name of a layout room: its first room-kind place in places.yaml (living_room -> "living room")."""
    return next(p["name"] for p in places or load_places() if p["kind"] == "room" and p["room"] == room)


def behind_room(layout: dict[str, Any], door: str) -> str:
    """The room behind a crate door: the room of its `between` that the door cuts off from the hall (house A: the
    non-hall room; house B's bedroom door joins the living room and the bedroom)."""
    from sim.tasks import cut_off_layout
    return next(r for r in layout["doors"][door]["between"] if r in cut_off_layout(layout, door))


def near_room(layout: dict[str, Any], door: str) -> str:
    """The hall side of a crate door: the room of its `between` that stays reachable (house A: the hall)."""
    return next(r for r in layout["doors"][door]["between"] if r != behind_room(layout, door))


def cabinet_rooms(layout: dict[str, Any], door: str) -> tuple[str, str]:
    """(start room, goal room) of a cabinet door: goal = the cabinet room (tasks.detour_pair), start = the other side."""
    goal = detour_pair(layout)["room"]
    return next(r for r in layout["doors"][door]["between"] if r != goal), goal


def cabinet_axes(layout: dict[str, Any], door: str) -> tuple[int, int, float]:
    """(along, across, sign) coordinate indices (0 = x, 1 = y) for a cabinet door: along = its normal, sign = +1 when
    the goal room lies on the + side of the door from the start room."""
    k = 0 if "x" in layout["doors"][door]["wall"] else 1
    start, goal = cabinet_rooms(layout, door)
    c = {r: [(v[0] + v[2]) / 2, (v[1] + v[3]) / 2] for r, v in ((r, layout["rooms"][r]["rect"]) for r in (start, goal))}
    return k, 1 - k, 1.0 if c[goal][k] > c[start][k] else -1.0


def door_dict(layout: dict[str, Any], name: str) -> dict[str, Any]:
    """The door as the adapter (sim/adapter.py `_door_dict`) passes it, plus "centre" for this harness."""
    dd = layout["doors"][name]
    (axis, c), = dd["wall"].items()
    centre = (c, dd["center"]) if axis == "x" else (dd["center"], c)
    return {"id": name, "name": name, "axis": axis, "line": float(c), "center": float(dd["center"]),
            "width": float(layout["door_width"]), "between": list(dd["between"]), "centre": centre}


def start_pose(door: dict[str, Any], near_room: str, layout: dict[str, Any]) -> tuple[float, float, float]:
    """START_D in front of the door on near_room's side, facing the door."""
    k = 0 if door["axis"] == "x" else 1
    x0, y0, x1, y1 = layout["rooms"][near_room]["rect"]
    side = 1.0 if ((x0 + x1) / 2, (y0 + y1) / 2)[k] > door["centre"][k] else -1.0
    p = list(door["centre"])
    p[k] += side * START_D
    yaw = math.atan2(-side if k == 1 else 0.0, -side if k == 0 else 0.0)
    return float(p[0]), float(p[1]), yaw


def jitter(m: mujoco.MjModel, d: mujoco.MjData, body: str, door: dict[str, Any], seed: int) -> list[float]:
    rng = np.random.default_rng([seed, 2])
    da, dl, dyaw = rng.uniform(-JIT_M, JIT_M), rng.uniform(-JIT_M, JIT_M), math.radians(rng.uniform(-JIT_DEG, JIT_DEG))
    j = m.body(body).jntadr[0]
    a = m.jnt_qposadr[j]
    if door["axis"] == "x":
        d.qpos[a] += da
        d.qpos[a + 1] += dl
    else:
        d.qpos[a] += dl
        d.qpos[a + 1] += da
    q = d.qpos[a + 3: a + 7].copy()
    r = np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])
    mujoco.mju_mulQuat(d.qpos[a + 3: a + 7], r, q)
    mujoco.mj_forward(m, d)
    return [float(da), float(dl), math.degrees(dyaw)]


def finite(d: mujoco.MjData) -> bool:
    return bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())


def crate_trial(seed: int, places: dict[str, dict]) -> dict[str, Any]:
    w = generate(seed=seed, blocker="crate")
    m = compile_house(w)
    grid = known_map(w.layout)
    door = door_dict(w.layout, w.blocker["door"])
    far = behind_room(w.layout, door["name"])
    d = start_state(m, start_pose(door, near_room(w.layout, door["name"]), w.layout))  # settles START_SETTLE_S: crate at rest
    jit = jitter(m, d, "crate", door, seed)
    run(m, d, SETTLE_S)
    t0 = time.time()
    r = push_object(m, d, grid, "crate", {k: v for k, v in door.items() if k != "centre"})  # adapter shape
    g = go_to(m, d, grid, pose_of(places[room_place(far, list(places.values()))]))
    crate_contacts = {k: v for k, v in g["contacts"].items() if "crate" in k}
    ok = bool(r["ok"] and g["reached"] and finite(d))
    return {"seed": seed, "door": door["name"], "jitter_along_lat_m_yaw_deg": jit, "pass": ok,
            "push_ok": r["ok"], "push_reason": r["reason"], "goto_reached": g["reached"],
            "goto_pos_err_m": g.get("pos_err_m"), "goto_crate_contacts": crate_contacts, "sim_ok": finite(d),
            "push": r["detail"], "wall_s": time.time() - t0}


def cabinet_trial(seed: int) -> dict[str, Any]:
    w = generate(seed=seed, blocker="cabinet")
    m = compile_house(w)
    grid = known_map(w.layout)
    door = door_dict(w.layout, w.blocker["door"])
    d = start_state(m, start_pose(door, cabinet_rooms(w.layout, door["name"])[0], w.layout))
    c0 = d.body("cabinet").xpos.copy()
    rg, cg = robot_geoms(m), np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("cabinet").id])
    pen = [0.0]

    def cb(dd: mujoco.MjData) -> None:
        g = dd.contact.geom[: dd.ncon]
        hit = (np.isin(g[:, 0], rg) & np.isin(g[:, 1], cg)) | (np.isin(g[:, 1], rg) & np.isin(g[:, 0], cg))
        if hit.any():
            pen[0] = max(pen[0], float(-dd.contact.dist[: dd.ncon][hit].min()))

    t0 = time.time()
    r = push_object(m, d, grid, "cabinet", door, cb=cb)
    move = float(np.linalg.norm(d.body("cabinet").xpos - c0))
    ok = bool(not r["ok"] and r["reason"] == "not movable" and move <= G2_CABINET_MAX_MOVE
              and pen[0] <= G2_CABINET_MAX_PEN and finite(d))
    return {"seed": seed, "door": door["name"], "pass": ok, "push_ok": r["ok"], "push_reason": r["reason"],
            "cabinet_move_m": move, "pen_max_m": pen[0], "sim_ok": finite(d), "push": r["detail"],
            "wall_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first", type=int, default=1000)
    a = ap.parse_args()
    places = {p["name"]: p for p in load_places()}
    seeds = list(range(a.first, a.first + a.seeds))
    crate = [crate_trial(s, places) for s in seeds]
    for t in crate:
        print(f"crate   seed {t['seed']} {t['door']} pass={t['pass']} push={t['push_ok']}/{t['push_reason']!r} "
              f"goto={t['goto_reached']} tilt={t['push']['tilt_deg']:.1f} fmax={t['push']['fmax_n']:.0f}", flush=True)
    cab = [cabinet_trial(s) for s in seeds]
    for t in cab:
        print(f"cabinet seed {t['seed']} {t['door']} pass={t['pass']} reason={t['push_reason']!r} "
              f"move={t['cabinet_move_m'] * 1e3:.2f} mm pen={t['pen_max_m'] * 1e3:.2f} mm", flush=True)
    n_c, n_k = sum(t["pass"] for t in crate), sum(t["pass"] for t in cab)
    out = {"seeds": seeds, "jitter": {"along_lat_m": JIT_M, "yaw_deg": JIT_DEG}, "start_d_m": START_D,
           "G2-push-crate-dev": {"pass": n_c, "n": len(crate), "bar": 9, "gate_pass": n_c >= 9},
           "G2-push-cabinet-dev": {"pass": n_k, "n": len(cab), "bar": len(cab), "gate_pass": n_k == len(cab)},
           "crate": crate, "cabinet": cab, "mujoco": mujoco.__version__}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("G2-push-crate-dev", "G2-push-cabinet-dev")}))


if __name__ == "__main__":
    main()
