"""Look arm pose (task 002 item 6): no contacts, inside the footprint, clears the floor ahead of the mast camera."""
import math

import mujoco
import numpy as np

from sim import constants as C
from sim.body import BOX, _ids, body_scene, reset, run
from sim.look import (LOOK_ARM_QPOS, arm_geoms, drive_yaw, robot_geoms, look_photos, mast_clearance, move_arm, render_look, robot_contacts,
                      robot_radius_xy, set_arm)
from sim.robot import ARM_JOINTS, BASE_JOINTS, CAM_H, CAM_W


def test_look_pose_contacts_radius() -> None:
    m = body_scene((-3.0, 3.0), table=False).compile()
    d = reset(m)
    hit: list = []
    move_arm(m, d, LOOK_ARM_QPOS, cb=lambda dd: hit.extend(robot_contacts(m, dd)))  # from retract, interpolated
    run(m, d, 2.0)
    assert hit == [] and robot_contacts(m, d) == []
    assert np.abs(d.qpos[_ids(m, ARM_JOINTS)[0]] - LOOK_ARM_QPOS).max() < 0.02
    assert robot_radius_xy(m, d, arm_geoms(m)) <= C.ROBOT_RADIUS
    base_only = robot_radius_xy(m, d, np.setdiff1d(robot_geoms(m), arm_geoms(m)))  # 0.28419 m (constant rounds it)
    assert robot_radius_xy(m, d) <= base_only  # the arm does not widen the footprint
    assert mast_clearance(m, d) > 0.0


def test_box_ahead_visible() -> None:
    m = body_scene((0.55, 0.0), table=False).compile()
    d = reset(m)
    set_arm(m, d)
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    try:
        _, _, seg = render_look(r, d)
    finally:
        r.close()
    assert (seg == m.geom(BOX).id).sum() >= C.G1_COVER_MIN_PX


def test_look_photos_both_modes() -> None:
    m = body_scene((-3.0, 3.0), table=False).compile()
    for mode in (None, drive_yaw):
        d = reset(m)
        set_arm(m, d)
        photos = look_photos(m, d) if mode is None else look_photos(m, d, mode)
        assert [p[0] for p in photos] == [0.0, 90.0, 180.0, 270.0]
        for _, rgb, depth, seg in photos:
            assert rgb.shape == (CAM_H, CAM_W, 3) and depth.shape == (CAM_H, CAM_W) and seg.shape == (CAM_H, CAM_W)
            assert np.isfinite(depth).all()
        yaw = d.qpos[_ids(m, BASE_JOINTS)[0][2]]
        assert abs(math.degrees((yaw - 1.5 * math.pi + math.pi) % (2 * math.pi) - math.pi)) <= C.G0A_GOAL_YAW_TOL_DEG
