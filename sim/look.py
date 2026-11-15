"""Look/drive arm pose and the 4-heading look capture from the mast camera.

Phase 0 caveat 3: in `retract` the arm hangs in front of the mast camera and hides the floor 0.25-0.75 m ahead.
LOOK_ARM_QPOS folds the arm backwards beside the mast, below the camera's lower frustum plane (35 deg pitch +
36.9 deg half-FOV = 71.9 deg below horizontal), inside the base footprint radius and clear of the mast pole.
It was found by a local random search over the 7 arm joints on mesh vertices (cost: vertices inside the camera
frustum, xy radius > 0.265 m, z < 0.36 m, < 6 cm from the mast axis, contacts), then checked by rendering,
dynamics (contacts after settling) and a joint-space interpolation from retract (tests/test_limits.py, 1r report).
"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from sim.body import StepCb, _ids, base_pose, drive_base, track
from sim.robot import ARM_JOINTS, BASE_JOINTS, CAM_H, CAM_W, MAST_CAM

LOOK_ARM_QPOS = (2.563, 0.517, 3.087, -2.57, -0.102, -0.463, 1.064)  # joint_1..joint_7, rad
HEADINGS_DEG = (0.0, 90.0, 180.0, 270.0)  # absolute (world) base yaw of the 4 look photos
ARM_WMAX = 1.0  # rad/s peak joint speed when moving the arm between poses

Photo = tuple[float, np.ndarray, np.ndarray, np.ndarray]  # heading_deg, rgb uint8 HxWx3, depth m HxW, seg geom id HxW
SetYaw = Callable[[mujoco.MjModel, mujoco.MjData, float], None]


def robot_geoms(m: mujoco.MjModel) -> np.ndarray:
    """Ids of every geom on the robot (bodies under base_link: base, mast, arm, gripper)."""
    root = m.body("base_link").id
    return np.array([g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root])


def arm_geoms(m: mujoco.MjModel) -> np.ndarray:
    """Robot geoms on the moving arm + gripper (below gen3/base_link, which is fixed to the top plate)."""
    root = m.body("gen3/base_link").id
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == root:
            continue
        while b not in (0, root):
            b = int(m.body_parentid[b])
        if b == root:
            out.append(g)
    return np.array(out)


def geom_points(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray) -> np.ndarray:
    """World points on the given geoms: all mesh vertices, the 8 bounding-box corners for primitives."""
    pts = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            v = np.array([c + h * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])
        pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    return np.vstack(pts)


def robot_radius_xy(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray | None = None) -> float:
    """Max horizontal distance of any point of `geoms` (default: all robot geoms) from the base centre."""
    p = geom_points(m, d, robot_geoms(m) if geoms is None else geoms) - d.body("base_link").xpos
    return float(np.hypot(p[:, 0], p[:, 1]).max())


def robot_contacts(m: mujoco.MjModel, d: mujoco.MjData) -> list[tuple[str, str]]:
    """Active contacts that touch a robot geom, as (body1, body2) names."""
    rg = set(robot_geoms(m).tolist())
    return [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)
            for c in d.contact[: d.ncon] if c.geom1 in rg or c.geom2 in rg]


def mast_clearance(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """Min distance (m) from the arm to the mast pole. The pole is visual-only (contype 0), so contacts miss it."""
    pole = m.geom("mast_pole").id
    ft = np.zeros(6)
    return float(min(mujoco.mj_geomDistance(m, d, int(g), pole, 1.0, ft) for g in arm_geoms(m)))


def set_arm(m: mujoco.MjModel, d: mujoco.MjData, q: tuple[float, ...] = LOOK_ARM_QPOS) -> None:
    """Put the arm at q directly (qpos + actuator targets), zero its velocity, forward. For world/scene setup."""
    qadr, dadr, act = _ids(m, ARM_JOINTS)
    d.qpos[qadr], d.ctrl[act], d.qvel[dadr] = q, q, 0.0
    mujoco.mj_forward(m, d)


def move_arm(m: mujoco.MjModel, d: mujoco.MjData, q: tuple[float, ...] = LOOK_ARM_QPOS, cb: StepCb = None) -> float:
    """Joint-space min-jerk move of the arm targets from the current ones to q (peak <= ARM_WMAX). Returns seconds."""
    _, _, act = _ids(m, ARM_JOINTS)
    q0, q1 = d.ctrl[act].copy(), np.asarray(q, dtype=float)
    sec = max(0.5, 1.875 * float(np.abs(q1 - q0).max()) / ARM_WMAX)
    track(m, d, act, q0, q1, sec, cb)
    return sec


def set_yaw_direct(m: mujoco.MjModel, d: mujoco.MjData, yaw: float) -> None:
    """Fast mode (coverage gate): set the base yaw qpos and its setpoint directly, zero base velocity, forward.

    This is a teleport of the yaw only, allowed for look capture; the driven mode is `drive_yaw`."""
    qadr, dadr, act = _ids(m, BASE_JOINTS)
    d.qpos[qadr[2]], d.ctrl[act[2]] = yaw, yaw
    d.qvel[dadr] = 0.0
    mujoco.mj_forward(m, d)


def drive_yaw(m: mujoco.MjModel, d: mujoco.MjData, yaw: float) -> None:
    """Physical mode (clip): turn in place through the joint_th setpoint (speed-capped drive_base), then settle."""
    x, y, _ = base_pose(m, d)
    drive_base(m, d, (float(x), float(y), yaw))


def render_look(r: mujoco.Renderer, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB, depth (m) and per-pixel geom id (-1 = no geom) from the mast camera with an open renderer."""
    r.update_scene(d, camera=MAST_CAM)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(d, camera=MAST_CAM)
    depth = r.render().astype(np.float32).copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=MAST_CAM)
    s = r.render()
    r.disable_segmentation_rendering()
    seg = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1).astype(np.int32)
    return rgb, depth, seg


def look_photos(m: mujoco.MjModel, d: mujoco.MjData, set_base_yaw: SetYaw = set_yaw_direct) -> list[Photo]:
    """Mast photos at absolute base yaw 0/90/180/270 deg: (heading_deg, rgb, depth, seg). Arm pose is the caller's
    (use set_arm / move_arm to LOOK_ARM_QPOS first). The renderer is closed before returning."""
    out: list[Photo] = []
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    try:
        for h in HEADINGS_DEG:
            set_base_yaw(m, d, math.radians(h))
            out.append((h, *render_look(r, d)))
    finally:
        r.close()
    return out
