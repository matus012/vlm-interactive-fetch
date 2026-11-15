"""push_object (task 003 `push`): crate cleared out of its doorway and off the path, cabinet -> "not movable"."""
import math

import mujoco
import numpy as np

from sim import constants as C
from sim.body import _ids
from sim.nav.follow import go_to
from sim.push import PROGRESS_M, PROGRESS_S, _rect_clear, push_object
from sim.push_check import cabinet_trial, crate_trial, door_dict, start_pose
from sim.reach_check import compile_house, pose_of, start_state
from sim.robot import BASE_JOINTS
from sim.world.generate import generate, load_places
from sim.world.occupancy import known_map

PLACES = {p["name"]: p for p in load_places()}


def test_rect_clear() -> None:
    sq = lambda a, l, h: [(a - h, l - h), (a + h, l - h), (a + h, l + h), (a - h, l + h)]
    assert _rect_clear(sq(1.0, 0.0, 0.225), 0.6, 0.5)  # 0.775 > 0.6
    assert not _rect_clear(sq(0.8, 0.0, 0.225), 0.6, 0.5)
    assert _rect_clear(sq(0.8, 0.75, 0.225), 0.6, 0.5)  # 0.525 > 0.5 across
    assert not _rect_clear(sq(0.0, 0.0, 0.225), 0.6, 0.5)


def test_crate_storage_side_push() -> None:
    t = crate_trial(1000, PLACES)  # d_sto, side plan at 1.1 m
    assert t["pass"] and t["push"]["plan"][0] == "side" and t["push"]["corridor_clear"]
    assert t["push"]["tilt_deg"] < 5.0 and not t["goto_crate_contacts"]


def test_crate_bedroom_shallow_side_push() -> None:
    t = crate_trial(1003, PLACES)  # d_bed: the bed rules out the 1.1 m side push -> 0.84 m
    assert t["pass"] and tuple(t["push"]["plan"]) == ("side", 1, 0.84, 0.07)
    assert abs(t["push"]["obj_lat"]) > 0.7 and not t["goto_crate_contacts"]


def test_cabinet_not_movable_and_released() -> None:
    t = cabinet_trial(1000)
    assert t["pass"] and t["push_reason"] == "not movable"  # never moved (005 row 10)
    assert t["push"]["why"] == "no progress" and t["push"]["obj_moved_m"] < PROGRESS_M
    assert t["cabinet_move_m"] <= C.G2_CABINET_MAX_MOVE and t["pen_max_m"] <= C.G2_CABINET_MAX_PEN
    assert t["push"]["through"]["sat_run_max_s"] <= PROGRESS_S + 0.01  # never left pressing at the limit


def test_crate_from_contact() -> None:
    """Robot starts touching the crate (where a blocked navigate stops) and the adapter's door shape is used."""
    w = generate(seed=1000, blocker="crate")
    m = compile_house(w)
    grid = known_map(w.layout)
    door = door_dict(w.layout, w.blocker["door"])
    x, y, yaw = start_pose(door, "hall", w.layout)
    k = 0 if door["axis"] == "x" else 1
    p = [x, y]
    p[k] = door["centre"][k] - math.copysign(C.CRATE_HALF[0] + 0.241, door["centre"][k] - p[k])  # bumper on the face
    d = start_state(m, (p[0], p[1], yaw))
    r = push_object(m, d, grid, "crate", {kk: v for kk, v in door.items() if kk != "centre"})
    assert r["ok"], r
    far = next(rr for rr in door["between"] if rr != "hall")
    g = go_to(m, d, grid, pose_of(PLACES[far]))
    assert g["reached"] and np.isfinite(d.qpos).all()
    _, _, act = _ids(m, BASE_JOINTS)
    assert float(np.hypot(*d.actuator_force[act[:2]])) < 50.0  # ends at rest, not pressing


def test_crate_keepout() -> None:
    """2j fix 4: no final crate position within 0.80 m of a place pose (other doors' door places included) or inside
    any door's R4 corridor; the pushed door's own door place does not count."""
    import numpy as np

    from sim.push import LONG_PAST, keepout_ok

    assert not keepout_ok(np.array([7.1, 6.0]), 0.225, "d_sto")  # 0.40 m from the storage-bench approach (round 1)
    assert keepout_ok(np.array([7.1, 4.5]), 0.225, "d_sto")
    assert not keepout_ok(np.array([8.2, 5.25]), 0.225, "d_sto")  # 0.70 m past the storage room point (old LONG_PAST)
    assert keepout_ok(np.array([7.5 + LONG_PAST, 5.25]), 0.225, "d_sto")  # the long plan's end now
    assert not keepout_ok(np.array([3.75, 2.6]), 0.225, None)  # in d_liv's corridor
    assert keepout_ok(np.array([4.5, 6.4]), 0.225, "d_bed")  # hall, 1.15 m from the hall point, off every corridor
    assert not keepout_ok(np.array([3.4, 5.0]), 0.225, "d_sto")  # beside d_bed: its door place (other door) + corridor


def _crate_1000(monkeypatch, **patch):
    import sim.push as P

    w = generate(seed=1000, blocker="crate")
    m = compile_house(w)
    door = door_dict(w.layout, w.blocker["door"])
    d = start_state(m, start_pose(door, "hall", w.layout))
    for k, v in patch.items():
        monkeypatch.setattr(P, k, v)
    return P, m, d, known_map(w.layout), {k: v for k, v in door.items() if k != "centre"}


def test_stalled_side_push_with_clear_end_is_pushed(monkeypatch) -> None:
    """Row 17: the side push is forced into the storage wall (plan fixed, SIDE_LAT 3 m) and stalls, but the crate's
    ACTUAL end is off every corridor and >= 0.80 m from every place pose -> pushed (the plan's stall is in detail)."""
    P, m, d, grid, door = _crate_1000(monkeypatch, _choose_plan=lambda *a, **k: ("side", -1, 1.1, 0.0), SIDE_LAT=3.0)
    r = P.push_object(m, d, grid, "crate", door)
    det = r["detail"]
    assert r["ok"] and det["side"]["stalled"] and det["why"] in ("no progress", "saturated")
    assert not det["end_in_corridor"] and not det["end_near_place"] and det["obj_moved_m"] > 0.5
    assert not any(P._blocks(*P._aabb(m, d, "crate")))


def test_planned_end_ok_actual_end_near_place_is_stuck(monkeypatch) -> None:
    """Row 17 / tuning blocked-8217 step 12: a long push that ends where the plan said, but by the storage room point
    (LONG_PAST -0.6: crate centre 0.6 m short of it, corridor clear), with no clearing push -> never "pushed": stuck."""
    P, m, d, grid, door = _crate_1000(monkeypatch, _choose_plan=lambda *a, **k: ("long",), LONG_PAST=-0.6,
                                      _clear_plan=lambda *a, **k: None)
    r = P.push_object(m, d, grid, "crate", door)
    det = r["detail"]
    assert det["door_result"] == [True, ""] and not det["end_in_corridor"] and det["end_near_place"]
    assert not r["ok"] and r["reason"] == "stuck" and det["why"] == "no room to clear" and det["obj_moved_m"] > 0.5


def test_planned_end_near_place_is_cleared(monkeypatch) -> None:
    """Same long push, clearing push on: the crate is pushed on off the room point -> pushed, actual end clear."""
    P, m, d, grid, door = _crate_1000(monkeypatch, _choose_plan=lambda *a, **k: ("long",), LONG_PAST=-0.6)
    r = P.push_object(m, d, grid, "crate", door)
    det = r["detail"]
    assert r["ok"], (det["why"], det.get("clear_plan"), det.get("clear_approach"))
    assert det["clear_plan"] is not None and not det["end_in_corridor"] and not det["end_near_place"]
    assert not any(P._blocks(*P._aabb(m, d, "crate")))


def test_not_blocking_moves_nothing(monkeypatch) -> None:
    """Row 17: crate already off every corridor and >= 0.80 m from every place (hall, (4.5, 6.4)) -> "not blocking",
    nothing moved (crate, robot)."""
    P, m, d, grid, door = _crate_1000(monkeypatch)
    j = m.body("crate").jntadr[0]
    d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 2] = (4.5, 6.4)
    mujoco.mj_forward(m, d)
    q0, c0, t0 = d.qpos[:3].copy(), d.body("crate").xpos.copy(), d.time
    r = P.push_object(m, d, grid, "crate", door)
    assert not r["ok"] and r["reason"] == "not blocking" and r["detail"]["why"] == "not blocking"
    assert d.time == t0 and np.allclose(d.qpos[:3], q0) and np.allclose(d.body("crate").xpos, c0)


def test_under_1cm_never_pushed(monkeypatch) -> None:
    """Row 17: a push that moves the object < PUSHED_MIN_M never says "pushed", even if the plan ran and the end check
    passes (stubbed: blocking at the call, clear after; motion stubbed out, so nothing moves)."""
    calls = iter([(True, False)])
    noop = {"stalled": False, "why": None, "obj_disp_m": 0.0, "fmax_n": 0.0, "pen_max_m": 0.0, "sat_steps": 0,
            "sat_run_max_s": 0.0}
    P, m, d, grid, door = _crate_1000(
        monkeypatch, _blocks=lambda *a: next(calls, (False, False)), _choose_plan=lambda *a, **k: ("long",),
        go_to=lambda *a, **k: {"reached": True, "contacts": {}}, push_line=lambda *a, **k: noop,
        drive_base=lambda *a, **k: None, room_places_reachable=lambda *a: True)  # end check passes (row 17 add. 2 too)
    r = P.push_object(m, d, grid, "crate", door)
    assert P.PUSHED_MIN_M == 0.01 and r["detail"]["obj_moved_m"] < P.PUSHED_MIN_M
    assert r["detail"]["door_result"] == [True, ""] and not r["ok"] and r["reason"] == "stuck"  # not pushed against
    assert r["detail"]["why"] == "not moved"


def test_8217_replay_push_end_passes_keepout(tmp_path) -> None:
    """Tuning blocked-8217: steps 1-10 replayed (holding the red box, navigate(storage) blocked by the crate in d_sto),
    then push(crate) until it says pushed (<= 3 calls): every push that says pushed leaves the crate's ACTUAL pose
    clear of every corridor and place; then navigate(storage) and place(storage bench) work."""
    from tune_replay import replay
    from sim.push import _aabb, _blocks

    robot, rt = replay("blocked-8217", tmp_path)
    assert rt.steps[-1]["result"] == "failed: blocked" and robot.held() == "red box"
    for _ in range(3):
        r = rt.call("push", {"object": "crate"})
        if r.text == "pushed":
            break
        assert r.text in ("failed: stuck", "failed: not movable"), r.text
    assert r.text == "pushed" and not any(_blocks(*_aabb(robot.m, robot.d, "crate"))) and robot.held() == "red box"
    assert r.meta["detail"]["obj_moved_m"] >= 0.01
    assert rt.call("navigate", {"place": "storage"}).text == "reached"
    assert rt.call("place", {"surface": "storage bench"}).text == "placed"


def test_8217_logged_parked_crate_is_cleared(tmp_path) -> None:
    """The logged run's state before step 12 (PERUN; the laptop replay diverges at step 10): crate at (7.016, 5.761),
    off the d_sto corridor but 0.70 m from the storage room point and 0.52 m from the storage-bench approach. Rebuilt as
    steps 1-10 + the crate put there. push(crate) is a clearing push (+x) -> pushed, actual end clear, box held."""
    from tune_replay import replay, set_xy
    from sim.push import _aabb, _blocks

    robot, rt = replay("blocked-8217", tmp_path)
    set_xy(robot, "crate", (7.016272309357157, 5.761268260344077))
    assert _blocks(*_aabb(robot.m, robot.d, "crate")) == (False, True)
    r = rt.call("push", {"object": "crate"})
    assert r.text == "pushed", (r.text, r.meta["detail"]["why"])
    det = r.meta["detail"]
    assert det["clear_plan"]["u"] == [1.0, 0.0] and "door_result" not in det
    assert not any(_blocks(*_aabb(robot.m, robot.d, "crate"))) and robot.held() == "red box"
    assert rt.call("place", {"surface": "storage bench"}).text == "placed"


def test_skills_push_labels() -> None:
    """Skills maps the robot's reason: "not movable" / "not blocking" -> the same failed: text, anything else ->
    failed: stuck."""
    from agent.fake_robot import FakeRobot
    from agent.robot_api import ActResult
    from agent.skills import Skills

    r = FakeRobot({"red box": ("bedroom", 2.5, 5.0, "floor")}, blockers={"crate": "storage door"})
    s = Skills(r)
    for reason, text in (("not movable", "failed: not movable"), ("stuck", "failed: stuck"),
                         ("not blocking", "failed: not blocking"), ("sim error", "failed: stuck")):
        r.push = lambda obj, reason=reason: ActResult(False, reason)
        assert s.push("crate").text == text


def test_no_feasible_plan_is_stuck_not_not_movable(monkeypatch) -> None:
    """Ruling 2026-11-13 (row 10): no feasible plan, nothing pushed -> "stuck", never "not movable" (that is only for an
    object pushed against that never moved, e.g. the cabinet). Stubbed: the crate blocks a place pose at the call, no
    clearing push exists."""
    P, m, d, grid, door = _crate_1000(monkeypatch, _blocks=lambda *a: (False, True), _clear_plan=lambda *a, **k: None)
    r = P.push_object(m, d, grid, "crate", door)
    assert not r["ok"] and r["detail"]["obj_moved_m"] < P.PROGRESS_M
    assert r["detail"]["why"] not in ("no progress", "saturated") and r["reason"] == "stuck", r["detail"]["why"]
