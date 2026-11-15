"""Fast G0a-base / G0a-grasp checks: one goal, one grasp trial (full gate: python -m sim.body_check)."""
import math

from sim.body import BOX, body_scene, drive_base, grasp_and_lift, reset


def test_base_reaches_goal() -> None:
    m = body_scene(table=False).compile()
    d = reset(m)
    e = drive_base(m, d, (-0.8, 1.2, math.radians(-45.0)))
    assert e["pos_err_m"] <= 0.02 and e["yaw_err_deg"] <= 2.0
    assert e["body_pos_err_m"] <= 0.02 and e["body_yaw_err_deg"] <= 2.0


def test_grasp_lift_hold() -> None:
    m = body_scene(table=False).compile()
    assert abs(m.body_mass[m.body(BOX).id] - 0.3) < 1e-9
    d = reset(m)
    r = grasp_and_lift(m, d)
    assert r["lift_after_hold_m"] >= 0.18
    assert r["held"] and r["left_pad"] and r["right_pad"]
