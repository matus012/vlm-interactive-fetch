"""Task 006 (row 17 addendum 3): the pushed object goes on the planner map; an object that cuts a room's
places off is blocking; a push that moved things drops the located box poses (perceived mode)."""
import math

import numpy as np
import pytest

import sim.push as P
from agent.robot_api import Pose
from agent.skills import Skills
from sim.adapter import AVOID_MARGIN, PUSHED_MAP_MARGIN, SimRobot
from sim.constants import CRATE_HALF
from tune_replay import set_xy


def test_room_cutting_object_is_not_not_blocking(monkeypatch) -> None:
    """Off every corridor and place at the call, but its room's places cut off -> pushed at (clearing push), never
    "not blocking"; no clearing push here -> stuck."""
    from test_push import _crate_1000

    Pm, m, d, grid, door = _crate_1000(monkeypatch, _blocks=lambda *a: (False, False),
                                       room_places_reachable=lambda grid, poly: False,
                                       _clear_plan=lambda *a, **k: None)
    r = Pm.push_object(m, d, grid, "crate", door)
    assert r["detail"]["start_cuts_place"] and r["reason"] == "stuck" and r["detail"]["why"] == "no room to clear"


def test_contact_footprint() -> None:
    """Object squared up in front of the bumper: base + u (BASE_HALF + half) + the planned offset, yaw of u."""
    x, y, yaw = P.contact_footprint({"base": [1.0, 2.0, 0.3], "u": [0.0, 1.0], "off": [0.07, 0.0]}, (0.225, 0.225))
    assert (x, y) == pytest.approx((1.07, 2.0 + P.BASE_HALF + 0.225)) and yaw == pytest.approx(math.pi / 2)


def _cell(robot: SimRobot, x: float, y: float) -> bool:
    i, j = robot.grid.world_to_cell(x, y)
    return bool(robot.grid.occ[j, i])


def _fake_push(monkeypatch, robot: SimRobot, end_xy: tuple[float, float], ok: bool, moved: float,
               contact: dict | None, seen: list) -> None:
    """push_object stubbed: the crate teleported to end_xy, the given verdict / moved / last_contact; records the grid."""
    def fake(m, d, grid, body, door, cb=None, held_geoms=(), stop_path=None, **_):
        seen.append(grid.occ.copy())
        set_xy(robot, body, end_xy)
        return {"ok": ok, "reason": "" if ok else "stuck",
                "detail": {"obj_moved_m": moved, **({"last_contact": contact} if contact else {})}}
    monkeypatch.setattr(P, "push_object", fake)


def test_unpushed_crate_stays_off_the_map() -> None:
    robot = SimRobot(1000, "crate")
    assert robot._pushed == {} and np.array_equal(robot.grid.occ, robot._known.occ)
    x, y = (float(v) for v in robot.d.body("crate").xpos[:2])
    assert not _cell(robot, x, y)
    grid, _, _ = robot._nav_grid(None, ())
    assert np.array_equal(grid.occ, robot._known.occ)


def test_perceived_footprint_from_contact_not_truth(monkeypatch) -> None:
    """Perceived: the map entry is where push's contact geometry puts the crate, not where the sim has it; the entry
    covers half + PUSHED_MAP_MARGIN; every later drive's grid has it (navigate grid; robot.grid for pick / place); a
    later push of the same object plans on the map without its own entry and replaces it."""
    robot = SimRobot(1000, "crate")
    seen: list = []
    contact = {"base": [4.0, 6.4, 0.0], "u": [1.0, 0.0], "off": [0.0, 0.0]}  # -> crate centre (4.466, 6.4)
    _fake_push(monkeypatch, robot, (4.5, 4.4), True, 1.0, contact, seen)  # truth 2 m away from the estimate
    r = robot.push("crate")
    cx = 4.0 + P.BASE_HALF + CRATE_HALF[0]
    assert r.ok and robot._pushed["crate"][:2] == pytest.approx((cx, 6.4))
    h = CRATE_HALF[0] + PUSHED_MAP_MARGIN
    assert _cell(robot, cx + h - 0.02, 6.4) and not _cell(robot, cx + h + 0.03, 6.4)
    assert not _cell(robot, 4.5, 4.4)  # the true pose is not on the map
    assert r.detail["map_eval"]["xy_err_m"] > 0.9  # report only
    grid, _, _ = robot._nav_grid(None, ())
    assert grid.occ[robot.grid.world_to_cell(cx, 6.4)[::-1]]
    contact2 = {"base": [3.0, 4.0, 0.0], "u": [0.0, -1.0], "off": [0.0, 0.0]}
    _fake_push(monkeypatch, robot, (3.0, 3.5), True, 0.5, contact2, seen)
    robot.push("crate")
    assert not seen[-1][robot.grid.world_to_cell(cx, 6.4)[::-1]]  # its own old entry off push's grid
    assert not _cell(robot, cx, 6.4) and _cell(robot, 3.0, 4.0 - P.BASE_HALF - CRATE_HALF[0])  # replaced


def test_truth_footprint_and_stuck_threshold(monkeypatch) -> None:
    """Truth mode: the true end pose + AVOID_MARGIN. A stuck push maps the object only if it moved >= 3 cm."""
    robot = SimRobot(1000, "crate", poses="truth")
    seen: list = []
    _fake_push(monkeypatch, robot, (4.5, 6.4), False, 0.02, None, seen)
    robot.push("crate")
    assert robot._pushed == {}
    _fake_push(monkeypatch, robot, (4.5, 6.4), False, 0.05, None, seen)
    robot.push("crate")
    assert robot._pushed["crate"][:2] == pytest.approx((4.5, 6.4), abs=1e-3)
    assert robot._pushed["crate"][3] == pytest.approx(CRATE_HALF[0] + AVOID_MARGIN)


def test_perceived_no_contact_stroke_no_entry(monkeypatch) -> None:
    robot = SimRobot(1000, "crate")
    _fake_push(monkeypatch, robot, (4.5, 6.4), True, 1.0, None, [])
    robot.push("crate")
    assert robot._pushed == {}


@pytest.mark.parametrize("moved,dropped", [(0.2, True), (0.005, False)])
def test_push_that_moved_things_drops_located(monkeypatch, moved, dropped) -> None:
    """Perceived: a push that moved the object >= 1 cm drops every located box pose (adapter and Skills): pick says
    "failed: not located" until the box is located again."""
    robot = SimRobot(1000, "crate")
    s = Skills(robot)
    for n in ("red box", "green box"):
        s.located[n] = robot._located[n] = Pose(1.0, 1.0, 0.1, 0.0)
    _fake_push(monkeypatch, robot, tuple(robot.d.body("crate").xpos[:2]), False, moved,
               {"base": [0.0, 0.0, 0.0], "u": [1.0, 0.0], "off": [0.0, 0.0]}, [])
    s.push("crate")
    assert (s.located == {} and robot._located == {}) is dropped
    if dropped:
        assert s.pick("red box").text == "failed: not located"


def test_truth_mode_keeps_located(monkeypatch) -> None:
    robot = SimRobot(1000, "crate", poses="truth")
    s = Skills(robot)
    s.located["red box"] = robot._located["red box"] = Pose(1.0, 1.0, 0.1, 0.0)
    _fake_push(monkeypatch, robot, tuple(robot.d.body("crate").xpos[:2]), True, 0.5, None, [])
    s.push("crate")
    assert "red box" in s.located


def test_push_records_last_contact() -> None:
    """A real push (seed 1000 crate in the storage door from the hall): last_contact is the base's own pose at a bumper
    contact, and the estimate lands within PUSHED_MAP_MARGIN of the true centre."""
    from test_push import _crate_1000

    mp = pytest.MonkeyPatch()
    try:
        Pm, m, d, grid, door = _crate_1000(mp)
        r = Pm.push_object(m, d, grid, "crate", door)
    finally:
        mp.undo()
    lc = r["detail"]["last_contact"]
    assert r["ok"] and len(lc["base"]) == 3 and np.linalg.norm(lc["u"]) == pytest.approx(1.0)
    x, y, _ = P.contact_footprint(lc, CRATE_HALF[:2])
    assert math.hypot(x - d.body("crate").xpos[0], y - d.body("crate").xpos[1]) <= PUSHED_MAP_MARGIN
