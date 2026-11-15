"""Fixtures for tests/test_grasp.py: a house with the robot at a box's room point, other-box discs, a settle probe."""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from sim.body import run
from sim.look import LOOK_ARM_QPOS, set_arm
from sim.world.generate import BOX_REACH, World, furniture_rect, generate, load_places, room_point
from sim.world.house import load_house
from sim.world.occupancy import Grid, known_map

PLACE_SURFACES = ("kitchen table", "storage bench")


def surfaces(w: World) -> dict[str, dict[str, Any]]:
    """Place surfaces: rect + top from layout furniture, approach from places.yaml (yaw in rad)."""
    pl = {p["name"]: p for p in load_places()}
    out = {}
    for f in w.layout["furniture"]:
        if f.get("surface") in PLACE_SURFACES:
            x, y, yaw = pl[f["surface"]]["pose"]
            out[f["surface"]] = {"name": f["surface"], "rect": list(furniture_rect(f)), "top_z": float(f["box"][4]),
                                 "approach": (float(x), float(y), math.radians(yaw))}
    return out


def setup(seed: int, body: str) -> tuple[World, mujoco.MjModel, mujoco.MjData, Grid, dict[str, Any]]:
    w = generate(seed=seed, blocker="none")
    b = next(x for x in w.boxes if x["name"] == body)
    rx, ry = room_point(b["room"])
    m, d = load_house(w, (rx, ry, 0.0))
    set_arm(m, d, LOOK_ARM_QPOS)
    run(m, d, 1.0)
    return w, m, d, known_map(w.layout), b


def others(m: mujoco.MjModel, d: mujoco.MjData, w: World, body: str) -> list[tuple[float, float, float]]:
    return [(float(d.body(b["name"]).xpos[0]), float(d.body(b["name"]).xpos[1]), BOX_REACH)
            for b in w.boxes if b["name"] != body]


def still(m: mujoco.MjModel, d: mujoco.MjData, body: str, seconds: float) -> tuple[float, float]:
    """Run `seconds`; max displacement from the start and max linear speed of `body`."""
    p0 = d.body(body).xpos.copy()
    disp, speed = 0.0, 0.0
    v6 = np.zeros(6)
    bid = m.body(body).id
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        disp = max(disp, float(np.linalg.norm(d.body(body).xpos - p0)))
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
        speed = max(speed, float(np.linalg.norm(v6[3:])))
    return disp, speed
