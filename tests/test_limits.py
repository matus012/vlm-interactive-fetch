"""Base force limit + speed cap (task 002 item 4): forcerange finite, crate push works, cabinet push stays bounded."""
import math

import mujoco
import numpy as np

from sim import constants as C
from sim.body import BASE_VMAX, BASE_WMAX, _ids, base_pose, body_scene, drive_base, reset, run
from sim.robot import BASE_AXIS_FORCE_N, BASE_FORCE_N, BASE_JOINTS, BASE_TORQUE_NM, robot_spec, tidybot_spec

FAR = (-3.0, 3.0)  # park the Phase 0 fetch box out of the way


def test_forcerange_set_and_finite() -> None:
    assert 0 < BASE_FORCE_N <= 300.0 and 0 < BASE_TORQUE_NM < math.inf
    for spec in (robot_spec(), tidybot_spec()):
        m = spec.compile()
        _, _, act = _ids(m, BASE_JOINTS)
        assert m.actuator_forcelimited[act].all()
        fr = m.actuator_forcerange[act]
        assert np.isfinite(fr).all()
        assert np.allclose(fr, [[-BASE_AXIS_FORCE_N, BASE_AXIS_FORCE_N]] * 2 + [[-BASE_TORQUE_NM, BASE_TORQUE_NM]])
        assert math.hypot(fr[0, 1], fr[1, 1]) <= BASE_FORCE_N + 1e-9  # R5: planar norm <= 300 N


def test_speed_cap_and_settle() -> None:
    """Setpoint and base speed stay under the caps; goals reached within G0a tolerances without overshoot."""
    m = body_scene(FAR, table=False).compile()
    d = reset(m)
    qa, da, _ = _ids(m, BASE_JOINTS)
    for goal in ((1.5, -0.5, math.radians(120.0)), (0.0, 0.0, 0.0)):
        g = np.array(goal)
        q0 = base_pose(m, d)
        u = (g[:2] - q0[:2]) / np.linalg.norm(g[:2] - q0[:2])
        v, over = [0.0, 0.0], [0.0]

        def cb(dd: mujoco.MjData) -> None:
            v[0] = max(v[0], math.hypot(dd.qvel[da[0]], dd.qvel[da[1]]))
            v[1] = max(v[1], abs(dd.qvel[da[2]]))
            over[0] = max(over[0], float((dd.qpos[qa[:2]] - g[:2]) @ u))

        e = drive_base(m, d, goal, cb=cb)
        assert e["pos_err_m"] <= C.G0A_GOAL_POS_TOL and e["yaw_err_deg"] <= C.G0A_GOAL_YAW_TOL_DEG
        assert v[0] <= BASE_VMAX * 1.02 and v[1] <= BASE_WMAX * 1.02
        assert over[0] <= 0.001  # <= 1 mm past the goal along the drive direction


def test_turn_in_place() -> None:
    m = body_scene(FAR, table=False).compile()
    d = reset(m)
    e = drive_base(m, d, (0.0, 0.0, math.pi))
    assert e["yaw_err_deg"] <= C.G0A_GOAL_YAW_TOL_DEG and e["pos_err_m"] <= C.G0A_GOAL_POS_TOL


def _scene_with(kind: str, x: float) -> mujoco.MjModel:
    spec = body_scene(FAR, table=False)
    half = C.CRATE_HALF if kind == "crate" else C.CABINET_HALF
    b = spec.worldbody.add_body(name=kind, pos=[x, 0.0, half[2]])
    if kind == "crate":
        b.add_freejoint()
        b.add_geom(name=kind, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=C.CRATE_MASS, friction=[1, 0.005, 1e-4])
    else:  # no joint: welded to the world
        b.add_geom(name=kind, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half))
    return spec.compile()


def test_crate_push_moves_crate() -> None:
    """At the R5 cap (212 N per axis) a straight push wedges the crate (sim/push.py doc); the wiggle push moves it."""
    from sim.push import push_line

    m = _scene_with("crate", 0.60)  # crate near face 0.375 m, base bumper ~0.24 m
    d = reset(m)
    c0 = d.body("crate").xpos.copy()
    drive_base(m, d, (0.66, 0.0, 0.0), seconds=6.6, settle=0.0)  # 0.1 m/s mean, straight
    wedged = float(d.body("crate").xpos[0] - c0[0])
    d = reset(m)
    tilt = [0.0]

    def cb(dd: mujoco.MjData) -> None:
        tilt[0] = max(tilt[0], math.degrees(2 * math.acos(min(1.0, abs(dd.body("crate").xquat[0])))))

    r = push_line(m, d, "crate", np.array([1.0, 0.0]), 1.0, cb=cb)
    dc = d.body("crate").xpos - c0
    assert wedged < 0.4  # 0.66 m of base travel, 0.11 m in contact range lost to the gap: the crate creeps
    assert np.isfinite(d.qpos).all() and not r["stalled"]
    assert dc[0] >= 0.8 and abs(dc[1]) < 0.05 and abs(dc[2]) < 0.01  # pushed ~0.87 m along, stays on the floor
    assert tilt[0] < 5.0
    assert base_pose(m, d)[0] > 0.99


def test_cabinet_push_bounded() -> None:
    m = _scene_with("cabinet", 0.591)  # near face 0.341 m: blocked after ~6 cm
    d = reset(m)
    qa, _, act = _ids(m, BASE_JOINTS)
    cg = m.geom("cabinet").id
    fmax, pen = [0.0], [0.0]

    def cb(dd: mujoco.MjData) -> None:
        fmax[0] = max(fmax[0], float(np.hypot(*dd.actuator_force[act[:2]])))
        p = [-c.dist for c in dd.contact[: dd.ncon] if cg in (c.geom1, c.geom2)]
        pen[0] = max([pen[0], *p])

    drive_base(m, d, (1.5, 0.0, 0.0), settle=0.0, cb=cb)
    x_end = base_pose(m, d)[0]
    run(m, d, 5.0, cb)  # 5 s more of pushing into it
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert fmax[0] <= BASE_FORCE_N + 1e-6
    assert pen[0] < 0.005
    assert abs(base_pose(m, d)[0] - x_end) < 0.001  # no creep, no blow-up
    assert d.ctrl[act[0]] - d.qpos[qa[0]] > 1.0  # tracking error grows: what the Phase 2 blocked detector reads
