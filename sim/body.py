"""TidyBot body skills for Phase 0: smooth base driving and a scripted top-down grasp of a fetch box.

Base: position actuators `joint_x/y/th` (ctrl = target qpos) follow a min-jerk setpoint, then settle.
Arm: damped-least-squares IK on `pinch_site` (position + orientation, arm joints only, base fixed),
joint targets interpolated into the arm position actuators. Physics is plain MuJoCo; nothing is welded.
"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from sim.robot import ARM_JOINTS, BASE_JOINTS, GRIPPER_ACT, tidybot_spec

BOX = "fetch_box"
BOX_HALF = (0.03, 0.10, 0.06)  # 6 x 20 x 12 cm; grasped across the 6 cm (box x) side
BOX_MASS = 0.3
BOX_REL = (0.55, 0.0)  # box centre in the base frame for a grasp: in front of the bumper, in arm reach
TABLE_POS = (1.6, 0.0)  # 0.75 m table, near edge 1.35 m from the base centre
TABLE_TOP_Z = 0.75
GRIP_OPEN, GRIP_CLOSED = 0.0, 255.0
PINCH = "pinch_site"
# Base speed cap (task 002 item 4): the min-jerk setpoint's peak speed (1.875 x mean) never exceeds these.
# 0.5 m/s: at the 300 N base force limit the 69 kg base stops from it in 29 mm (< the 5 cm reach tolerance) and the
# servo lag at it is 25 mm (kv/kp x v). 1.0 rad/s: rim speed 0.28 m/s, below the linear cap; a 90 deg look turn ~3 s.
# Belong in sim/constants.py once track W's constants merge (BASE_VMAX, BASE_WMAX).
BASE_VMAX = 0.5
BASE_WMAX = 1.0
MINJERK_PEAK = 1.875  # peak / mean speed of the min-jerk profile

StepCb = Callable[[mujoco.MjData], None] | None


def body_scene(box_xy: tuple[float, float] = BOX_REL, box_yaw: float = 0.0, table: bool = True) -> mujoco.MjSpec:
    """TidyBot + mast camera + a 0.3 kg fetch box on the floor (free joint) + optionally a 0.75 m table."""
    spec = tidybot_spec()
    hx, hy, hz = BOX_HALF
    box = spec.worldbody.add_body(
        name=BOX, pos=[box_xy[0], box_xy[1], hz], quat=[math.cos(box_yaw / 2), 0, 0, math.sin(box_yaw / 2)]
    )
    box.add_freejoint()
    box.add_geom(
        name=BOX, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], mass=BOX_MASS, rgba=[0.85, 0.45, 0.15, 1]
    )
    if table:
        tx, ty = TABLE_POS
        t = spec.worldbody.add_body(name="table", pos=[tx, ty, 0.0])
        t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.40, 0.02], pos=[0, 0, TABLE_TOP_Z - 0.02],
                   rgba=[0.55, 0.40, 0.25, 1])
        for sx in (-1, 1):
            for sy in (-1, 1):
                t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (TABLE_TOP_Z - 0.04) / 2],
                           pos=[sx * 0.22, sy * 0.37, (TABLE_TOP_Z - 0.04) / 2], rgba=[0.35, 0.25, 0.15, 1])
    return spec


def reset(m: mujoco.MjModel, key: str = "retract") -> mujoco.MjData:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)  # box qpos = its body pose (MjSpec fills the key)
    mujoco.mj_forward(m, d)
    return d


def _ids(m: mujoco.MjModel, joints: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qadr = np.array([m.joint(j).qposadr[0] for j in joints])
    dadr = np.array([m.joint(j).dofadr[0] for j in joints])
    act = np.array([m.actuator(j).id for j in joints])
    return qadr, dadr, act


def _minjerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(m: mujoco.MjModel, d: mujoco.MjData, seconds: float, cb: StepCb = None) -> None:
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        if cb is not None:
            cb(d)


def track(m: mujoco.MjModel, d: mujoco.MjData, act: np.ndarray, q0: np.ndarray, q1: np.ndarray,
          seconds: float, cb: StepCb = None) -> None:
    """Min-jerk setpoint from q0 to q1 on the given position actuators over `seconds` of sim time."""
    n = max(1, int(round(seconds / m.opt.timestep)))
    for i in range(1, n + 1):
        d.ctrl[act] = q1 if i == n else q0 + (q1 - q0) * _minjerk(i / n)  # ends exactly at q1 (q0 + (q1 - q0) need
        # not round to q1: snapshot's arm_retracted compares the targets exactly, 2j place 3000)
        mujoco.mj_step(m, d)
        if cb is not None:
            cb(d)


def base_pose(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    return d.qpos[_ids(m, BASE_JOINTS)[0]].copy()


def drive_seconds(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest min-jerk duration from base setpoint q0 to q1 (x, y, th) that keeps the speed caps."""
    return MINJERK_PEAK * max(math.hypot(q1[0] - q0[0], q1[1] - q0[1]) / BASE_VMAX, abs(q1[2] - q0[2]) / BASE_WMAX)


def drive_base(m: mujoco.MjModel, d: mujoco.MjData, goal: tuple[float, float, float], seconds: float = 0.0,
               settle: float = 1.0, cb: StepCb = None) -> dict[str, float]:
    """Drive to goal (x, y, theta rad) along a smooth setpoint (shortest turn), settle, return the pose error.

    The setpoint takes max(seconds, drive_seconds(...)): it never exceeds BASE_VMAX / BASE_WMAX; `seconds` can only
    slow it down. Adds "drive_s" (setpoint duration) to the result."""
    _, _, act = _ids(m, BASE_JOINTS)
    q0 = base_pose(m, d)
    c0 = d.ctrl[act].copy()
    q1 = np.array([goal[0], goal[1], q0[2] + wrap(goal[2] - q0[2])])
    seconds = max(seconds, drive_seconds(c0, q1))
    track(m, d, act, c0, q1, seconds, cb)
    run(m, d, settle, cb)
    q = base_pose(m, d)
    pos_err = float(math.hypot(q[0] - goal[0], q[1] - goal[1]))
    # cross-check with the body frame (independent of joint bookkeeping)
    xb = d.body("base_link").xpos
    yaw_b = math.atan2(d.body("base_link").xmat[3], d.body("base_link").xmat[0])
    return {
        "pos_err_m": pos_err,
        "yaw_err_deg": abs(math.degrees(wrap(q[2] - goal[2]))),
        "body_pos_err_m": float(math.hypot(xb[0] - goal[0], xb[1] - goal[1])),
        "body_yaw_err_deg": abs(math.degrees(wrap(yaw_b - goal[2]))),
        "drive_s": seconds,
    }


def ik(m: mujoco.MjModel, d: mujoco.MjData, pos: np.ndarray, rot: np.ndarray, iters: int = 300,
       damping: float = 1e-3) -> tuple[np.ndarray, float, float]:
    """Arm joint angles putting `pinch_site` at pos with orientation rot (3x3, world), base held at d's pose.

    Returns (q_arm, position error m, rotation error rad). Works on a scratch copy of d."""
    qadr, dadr, _ = _ids(m, ARM_JOINTS)
    s = mujoco.MjData(m)
    s.qpos[:] = d.qpos
    sid = m.site(PINCH).id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS]].T
    limited = m.jnt_limited[[m.joint(j).id for j in ARM_JOINTS]].astype(bool)
    for _ in range(iters):
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        cur = s.site_xmat[sid].reshape(3, 3)
        ep = pos - s.site_xpos[sid]
        er = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(m, s, jp, jr, sid)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        q = s.qpos[qadr] + np.clip(dq, -0.2, 0.2)
        q[limited] = np.clip(q[limited], lo[limited], hi[limited])
        s.qpos[qadr] = q
    mujoco.mj_kinematics(m, s)
    cur = s.site_xmat[sid].reshape(3, 3)
    er = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
    return s.qpos[qadr].copy(), float(np.linalg.norm(pos - s.site_xpos[sid])), float(np.linalg.norm(er))


def grasp_rot(box_xmat: np.ndarray) -> np.ndarray:
    """Top-down gripper frame: site z (approach) = world down, site x (finger closing axis) = box x (6 cm side)."""
    bx = box_xmat.reshape(3, 3)[:, 0].copy()
    bx[2] = 0.0
    bx /= np.linalg.norm(bx)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([bx, np.cross(z, bx), z])


def box_in_gripper(m: mujoco.MjModel, d: mujoco.MjData) -> dict[str, float | bool]:
    """Box near pinch_site and touched by a pad on each finger."""
    box_b = m.body(BOX).id
    touch = {"left": False, "right": False}
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if box_b not in (b1, b2):
            continue
        other = m.body(b2 if b1 == box_b else b1).name
        for side in touch:
            if other == f"{side}_pad":
                touch[side] = True
    dist = float(np.linalg.norm(d.body(BOX).xpos - d.site(PINCH).xpos))
    return {"pinch_dist_m": dist, "left_pad": touch["left"], "right_pad": touch["right"],
            "held": bool(touch["left"] and touch["right"] and dist < 0.05)}


def grasp_and_lift(m: mujoco.MjModel, d: mujoco.MjData, lift: float = 0.22, hold: float = 2.0,
                   cb: StepCb = None) -> dict[str, float | bool]:
    """Scripted pick of the fetch box: pre-grasp above, descend, close, lift `lift` m, hold `hold` s."""
    qadr, _, act = _ids(m, ARM_JOINTS)
    g = m.actuator(GRIPPER_ACT).id
    box = d.body(BOX)
    z0 = float(box.xpos[2])
    rot = grasp_rot(box.xmat)
    grasp_p = box.xpos.copy()
    grasp_p[2] = z0 + 0.02  # pinch slightly above the box centre: pads cover the upper box half, tips clear the floor
    d.ctrl[g] = GRIP_OPEN
    out: dict[str, float | bool] = {"box_z0": z0}
    q_cur = d.ctrl[act].copy()
    for name, dz, sec in (("pre", 0.15, 2.5), ("grasp", 0.0, 1.5)):
        q, ep, er = ik(m, d, grasp_p + [0, 0, dz], rot)
        out[f"ik_{name}_pos_err_m"], out[f"ik_{name}_rot_err_rad"] = ep, er
        track(m, d, act, q_cur, q, sec, cb)
        q_cur = q
    run(m, d, 0.3, cb)
    d.ctrl[g] = GRIP_CLOSED
    run(m, d, 1.0, cb)
    q, ep, er = ik(m, d, grasp_p + [0, 0, lift], rot)
    out["ik_lift_pos_err_m"], out["ik_lift_rot_err_rad"] = ep, er
    track(m, d, act, q_cur, q, 2.0, cb)
    run(m, d, 0.2, cb)
    out["lift_m"] = float(d.body(BOX).xpos[2] - z0)
    run(m, d, hold, cb)
    out["lift_after_hold_m"] = float(d.body(BOX).xpos[2] - z0)
    out.update(box_in_gripper(m, d))
    out["pass"] = bool(out["lift_after_hold_m"] >= 0.18 and out["held"])
    return out
