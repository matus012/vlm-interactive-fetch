"""TidyBot (Menagerie `stanford_tidybot`) plus the project's mast camera, built with MjSpec.

The Menagerie copy is never edited: the mast and its camera are added to the loaded spec.
Mast camera: 90 deg horizontal FOV at 640x480 (fx = fy = 320 px), RGB + depth, on a mast at the back of the base,
pitched down so it sees floor boxes near the robot and a 0.75 m table top 1-2 m away.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = Path(os.environ.get("MENAGERIE_DIR", REPO / "ext" / "mujoco_menagerie"))
TIDYBOT_SCENE = MENAGERIE / "stanford_tidybot" / "scene.xml"
TIDYBOT_XML = MENAGERIE / "stanford_tidybot" / "tidybot.xml"  # robot only, no floor / light: for attaching into a world

MAST_CAM = "mast"
CAM_W, CAM_H = 640, 480
CAM_HFOV_DEG = 90.0
# MuJoCo's fovy is the vertical FOV; square pixels -> fy = fx = (W/2) / tan(hfov/2).
CAM_FOVY_DEG = math.degrees(2.0 * math.atan(math.tan(math.radians(CAM_HFOV_DEG) / 2.0) * CAM_H / CAM_W))
MAST_XY = (-0.20, 0.0)  # base frame, behind the arm mount (arm base at x = 0, z = 0.335)
MAST_TOP_Z = 1.30  # camera height above the floor (base frame z = 0 at the floor)
MAST_PITCH_DEG = 35.0  # down from horizontal

BASE_JOINTS = ("joint_x", "joint_y", "joint_th")
ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER_ACT = "fingers_actuator"  # 0 = open, 255 = closed

# Base force limit (task 002 item 4, deviation-log row 1). Menagerie gains kept: kp 1e6 / kv 5e4 (x, y), 5e4 / 1e3 (th).
# - 300 N linear (the L1 cap): the base subtree is 69.1 kg, so 300 N gives 4.3 m/s^2, about 10x the 0.41 m/s^2 peak a
#   min-jerk move needs at the 0.5 m/s cap (1 m move), and stops from 0.5 m/s in 29 mm. Pushing a 4 kg crate at
#   friction 1 needs 39 N, so 300 N pushes it with 7x margin. Into a welded cabinet the push is bounded by 300 N.
# - 85 N.m yaw = 300 N at the footprint radius (0.284 m): same rim force. Yaw inertia about joint_th is ~2.0 kg m^2
#   (retract), so 85 N.m gives ~43 rad/s^2 against ~0.5 rad/s^2 needed by a pi turn at the 1 rad/s cap.
# - With the force saturated the position actuator acts as a velocity servo (kp/kv = 20 1/s) with no integrator, so
#   no wind-up: measured settle and overshoot are in tests/test_limits.py and the 1r report.
# R5 (task 003): the PLANAR NORM is capped at BASE_FORCE_N, as a per-axis forcerange of BASE_AXIS_FORCE_N =
# 300/sqrt(2) = 212.1 N on joint_x and joint_y (world-frame slides, so the norm bound holds in every direction; along an
# axis the base gets only 212 N). Chosen over a per-step norm cap because it lives in the model (MjSpec), needs no
# controller, and the crate still pushes with it (sim/push_check.py, runs/2c/).
# Integrator note (002's 225 N peak, explained in runs/2c/peak_225.json): under implicitfast the kv = 5e4 velocity term
# is integrated implicitly, so an UNCLAMPED actuator_force reads (M + h kv) / M = (69.1 + 100) / 69.1 = 2.45x the force
# that accelerates the base; a CLAMPED actuator drops the implicit term and the clamp value is the real force.
BASE_FORCE_N = 300.0
BASE_AXIS_FORCE_N = BASE_FORCE_N / math.sqrt(2.0)
BASE_TORQUE_NM = 85.0


def add_mast_camera(spec: mujoco.MjSpec) -> None:
    """Add a thin visual mast and the `mast` camera to `base_link` of a TidyBot spec."""
    base = spec.body("base_link")
    x, y = MAST_XY
    z0 = 0.335  # top plate
    half = (MAST_TOP_Z - z0) / 2.0
    mast = base.add_body(name="mast", pos=[x, y, z0 + half])
    mast.add_geom(
        name="mast_pole",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.02, half, 0],
        contype=0,
        conaffinity=0,
        group=2,
        mass=0.0,
        rgba=[0.2, 0.2, 0.2, 1],
    )
    th = math.radians(MAST_PITCH_DEG)
    # camera looks along its -z, image up = +y; forward = base +x pitched down by th, image right = base -y
    cam = base.add_camera(name=MAST_CAM, pos=[x, y, MAST_TOP_Z], fovy=CAM_FOVY_DEG, resolution=[CAM_W, CAM_H])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [0.0, -1.0, 0.0, math.sin(th), 0.0, math.cos(th)]


def add_base_limits(spec: mujoco.MjSpec) -> None:
    """Finite forcerange on the base actuators: +-BASE_AXIS_FORCE_N on joint_x/y (planar norm <= BASE_FORCE_N),
    +-BASE_TORQUE_NM on joint_th."""
    for name, lim in (("joint_x", BASE_AXIS_FORCE_N), ("joint_y", BASE_AXIS_FORCE_N), ("joint_th", BASE_TORQUE_NM)):
        a = spec.actuator(name)
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-lim, lim]


def tidybot_spec(scene: Path | str = TIDYBOT_SCENE) -> mujoco.MjSpec:
    """Menagerie TidyBot scene with the mast camera and the base force limit added."""
    spec = mujoco.MjSpec.from_file(str(scene))
    add_mast_camera(spec)
    add_base_limits(spec)
    return spec


def robot_spec() -> mujoco.MjSpec:
    """The robot alone (no floor, no light) with the project's additions, ready to attach into a world spec.

    Base joints `joint_x/y/th` are a slide-slide-hinge chain on the attachment frame: attach at the world origin and
    the base qpos is the world pose (x, y, yaw). Every project addition to the robot goes here (mast camera, base force limit)."""
    spec = mujoco.MjSpec.from_file(str(TIDYBOT_XML))
    add_mast_camera(spec)
    add_base_limits(spec)
    return spec


def load_tidybot(scene: Path | str = TIDYBOT_SCENE) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = tidybot_spec(scene).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("retract").id)
    mujoco.mj_forward(m, d)
    return m, d


def camera_intrinsics(m: mujoco.MjModel, cam: str = MAST_CAM, width: int = CAM_W, height: int = CAM_H) -> np.ndarray:
    """3x3 pinhole K for a MuJoCo camera at the given render size (fovy is vertical, pixels square)."""
    fovy = math.radians(float(m.cam_fovy[m.camera(cam).id]))
    f = (height / 2.0) / math.tan(fovy / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def render_rgbd(
    m: mujoco.MjModel, d: mujoco.MjData, cam: str = MAST_CAM, width: int = CAM_W, height: int = CAM_H
) -> tuple[np.ndarray, np.ndarray]:
    """One RGB (uint8 HxWx3) + depth (float32 metres, HxW) frame. The renderer is closed before returning."""
    r = mujoco.Renderer(m, height, width)
    try:
        r.update_scene(d, camera=cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cam)
        depth = r.render().astype(np.float32).copy()
    finally:
        r.close()
    return rgb, depth
