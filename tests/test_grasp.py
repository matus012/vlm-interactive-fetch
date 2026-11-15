"""Pick / place helpers and one full pick (floor box) + place (storage bench)."""
import math

import numpy as np

from sim.constants import G2_PICK_LIFT, ROBOT_RADIUS
from sim.grasp import (JOINT_MARGIN, _edge_dist, foot_free, holding, joint_margin, mark_discs, path_ok, pick_box,
                       place_box, retreat_knots, top_down_rot, true_box_pose)
from grasp_helpers import others, setup, still, surfaces
from sim.world.generate import load_layout
from sim.world.occupancy import known_map


def test_top_down_rot_is_rotation() -> None:
    r = top_down_rot(0.7)
    assert np.allclose(r.T @ r, np.eye(3)) and abs(np.linalg.det(r) - 1.0) < 1e-12
    assert np.allclose(r[:, 2], [0, 0, -1]) and abs(math.atan2(r[1, 0], r[0, 0]) - 0.7) < 1e-12


def test_roll_turns_about_the_closing_axis() -> None:
    r0, r = top_down_rot(0.7), top_down_rot(0.7, math.radians(17.0))
    assert np.allclose(r[:, 0], r0[:, 0]) and abs(math.degrees(math.acos(-r[2, 2])) - 17.0) < 1e-9


def test_path_ok_and_retreat_reject_unreachable() -> None:
    """2j fix 2: a knot at a joint limit or an IK error over IK_TOL is never followed; a pinch far out of reach has
    no retreat path (None, not a bad path)."""
    w, m, d, grid, b = setup(1000, "box_red")
    q = np.array([0.0, 2.24, 0.0, 0.0, 0.0, 0.0, 0.0])  # joint_2 at its limit
    assert joint_margin(m, [q]) < 1e-9 and not path_ok(m, [q], 0.0)
    q[1] = 2.24 - 2 * JOINT_MARGIN
    assert path_ok(m, [q], 0.0) and not path_ok(m, [q], 0.01)
    far = d.body("base_link").xpos + [1.5, 0.0, 0.9]
    assert retreat_knots(m, d, q, far, far[:2], 0.0) is None


def test_edge_dist() -> None:
    rect = [7.0, 0.65, 8.1, 1.35]  # kitchen table
    assert abs(_edge_dist(7.55, 1.95, -math.pi / 2, rect) - 0.6) < 1e-9
    assert math.isinf(_edge_dist(7.55, 1.95, math.pi / 2, rect))


def test_footprint_and_discs() -> None:
    g = known_map(load_layout())
    assert foot_free(g, 4.5, 5.25, 0.0)  # hall room point
    assert not foot_free(g, 7.55, 1.30, 0.0)  # over the kitchen table
    # facing the table from 0.30 m (front face 0.06 m from the edge) is free; the planner's disc inflation is not
    assert foot_free(g, 7.55, 1.35 + 0.30, -math.pi / 2, 0.03)
    assert not g.inflate(ROBOT_RADIUS + 0.09).free(7.55, 1.35 + 0.30)
    gd = mark_discs(g, [(4.5, 5.25, 0.1)])
    assert not gd.free(4.5, 5.25) and g.free(4.5, 5.25)


def test_pick_then_place() -> None:
    w, m, d, grid, b = setup(1000, "box_red")  # bedroom floor box
    assert abs(m.body_mass[m.body("box_red").id] - 0.3) < 1e-9
    r = pick_box(m, d, grid, "box_red", others(m, d, w, "box_red"), pose_fn=lambda: true_box_pose(d, "box_red"))
    det = r["detail"]
    assert r["ok"] and det["lift_above_support_m"] >= G2_PICK_LIFT
    assert det["pads"]["left_pad"] and det["pads"]["right_pad"]
    assert det["carry_xy_radius_m"] <= ROBOT_RADIUS + 0.01
    surf = surfaces(w)["storage bench"]
    p = place_box(m, d, mark_discs(grid, others(m, d, w, "box_red")), surf, "box_red", det["in_hand_carry"])
    assert p["ok"], p
    assert not holding(m, d, "box_red")
    disp, speed = still(m, d, "box_red", 2.0)
    assert disp <= 0.005 and speed <= 0.02
    assert p["detail"]["tilt_deg"] <= 15.0 and p["detail"]["centre_margin_m"] >= 0.02
