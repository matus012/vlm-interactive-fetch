"""Scorer input: settle the sim with the controls held, then report the boxes, surfaces and robot state.

    snapshot(m, d, world, held_fn, settle_s) -> dict   # schema: agent/robot_api.py (plus two extra keys below)

- Steps round(settle_s / timestep) physics steps with d.ctrl unchanged. Per box (VLM name "<colour> box"): final
  centre, tilt = angle of the box's 12 cm (local z) axis from world z, max displacement from its pose at the start of
  the window, max linear speed (free-joint world-frame velocity) over the window including its first state.
- support from geometry at the end: lowest box corner z (from the full orientation, so a box on its side is right);
  a furniture top with a `surface` name whose footprint rect contains the centre xy and whose height is within
  SCORE_ON_TOP_TOL of that corner -> that surface; else within SCORE_ON_TOP_TOL of z = 0 -> "floor"; else None.
- surfaces: layout place_surfaces, rect = furniture footprint [x0, y0, x1, y1], top_z = furniture height.
- held = held_fn(m, d) (pad-contact logic belongs to the adapter). gripper_open = gripper command is GRIP_OPEN.
  arm_retracted = the arm's joint targets are exactly LOOK_ARM_QPOS or the Menagerie `retract` pose (commanded
  state: no tolerance invented); extra key arm_track_err_rad = max |qpos - target| over the arm joints, so a
  reader can see whether the arm actually got there.
- sim_ok = qpos / qvel finite and MuJoCo raised no bad-qacc / instability warning since the last reset.
  Extra key n_steps = physics steps taken.
"""
from __future__ import annotations

import functools
import math
from typing import Callable

import mujoco
import numpy as np

from sim.constants import BOX_HALF, SCORE_ON_TOP_TOL
from sim.look import LOOK_ARM_QPOS
from sim.robot import ARM_JOINTS, GRIPPER_ACT, robot_spec
from sim.world.generate import World, furniture_rect
from sim.world.house import GRIP_OPEN

HeldFn = Callable[[mujoco.MjModel, mujoco.MjData], "str | None"]


@functools.cache
def retract_qpos() -> tuple[float, ...]:
    r = robot_spec()
    key, joints = list(r.key("retract").qpos), [j.name for j in r.joints]
    return tuple(float(key[joints.index(j)]) for j in ARM_JOINTS)


def box_bottom(d: mujoco.MjData, body: str) -> float:
    """Lowest corner z of a fetch box."""
    R = d.body(body).xmat.reshape(3, 3)
    return float(d.body(body).xpos[2] - np.abs(R[2]) @ np.asarray(BOX_HALF))


def support_of(world: World, x: float, y: float, bottom: float) -> str | None:
    for f in world.layout["furniture"]:
        x0, y0, x1, y1 = furniture_rect(f)
        if f.get("surface") and x0 <= x <= x1 and y0 <= y <= y1 and abs(bottom - f["box"][4]) <= SCORE_ON_TOP_TOL:
            return f["surface"]
    return "floor" if abs(bottom) <= SCORE_ON_TOP_TOL else None


def surfaces(world: World) -> dict[str, dict]:
    return {f["surface"]: {"rect": list(furniture_rect(f)), "top_z": float(f["box"][4])}
            for f in world.layout["furniture"] if f.get("surface") in world.layout["place_surfaces"]}


def snapshot(m: mujoco.MjModel, d: mujoco.MjData, world: World, held_fn: HeldFn, settle_s: float = 2.0) -> dict:
    names = {f"{b['colour']} box": b["name"] for b in world.boxes}
    dof = {k: int(m.jnt_dofadr[m.body(b).jntadr[0]]) for k, b in names.items()}
    p0 = {k: d.body(b).xpos.copy() for k, b in names.items()}
    disp = dict.fromkeys(names, 0.0)
    speed = {k: float(np.linalg.norm(d.qvel[a: a + 3])) for k, a in dof.items()}
    n = int(round(settle_s / m.opt.timestep))
    for _ in range(n):
        mujoco.mj_step(m, d)
        for k, b in names.items():
            disp[k] = max(disp[k], float(np.linalg.norm(d.body(b).xpos - p0[k])))
            speed[k] = max(speed[k], float(np.linalg.norm(d.qvel[dof[k]: dof[k] + 3])))
    boxes = {}
    for k, b in names.items():
        x, y, z = (float(v) for v in d.body(b).xpos)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(d.body(b).xmat[8])))))
        boxes[k] = {"pos": [x, y, z], "tilt_deg": tilt, "max_disp_settle_m": disp[k], "max_speed_settle": speed[k],
                    "support": support_of(world, x, y, box_bottom(d, b))}
    arm = [m.actuator(j).id for j in ARM_JOINTS]
    target = tuple(float(v) for v in d.ctrl[arm])
    qarm = d.qpos[[m.joint(j).qposadr[0] for j in ARM_JOINTS]]
    warn = sum(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                             mujoco.mjtWarning.mjWARN_BADQVEL))
    return {"boxes": boxes, "surfaces": surfaces(world), "held": held_fn(m, d),
            "gripper_open": float(d.ctrl[m.actuator(GRIPPER_ACT).id]) == GRIP_OPEN,
            "arm_retracted": target in (tuple(LOOK_ARM_QPOS), retract_qpos()),
            "arm_track_err_rad": float(np.abs(qarm - np.asarray(target)).max()),
            "sim_ok": bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and warn == 0),
            "settle_s": n * m.opt.timestep, "n_steps": n}
