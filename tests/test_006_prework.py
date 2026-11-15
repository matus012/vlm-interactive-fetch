"""Task 006 pre-work: one "held" (deviation 18), push end keeps its room's places reachable (row 17 addendum 2),
labeller tripwire pushed -> blocked on the same object (row 14 addendum)."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

import sim.grasp as G
from agent.runtime import label_failure
from agent.scorer import score
from agent.skills import Skills
from sim.adapter import SimRobot

MAIN_RUNS = Path("runs") / "005"  # main checkout's logs


# ---- 1. one "held" ------------------------------------------------------------------------------------------------

def _one_pad(monkeypatch, robot: SimRobot, box: str) -> str:
    """The box's centre put on the pinch site, and pad_contacts stubbed to report the left pad only (a box hanging on
    one pad, 005 tuning 8217)."""
    b = robot.body_of[box]
    j = robot.m.body(b).jntadr[0]
    a = robot.m.jnt_qposadr[j]
    robot.d.qpos[a:a + 3] = robot.d.site(G.PINCH).xpos
    mujoco.mj_forward(robot.m, robot.d)
    real = G.pad_contacts
    monkeypatch.setattr(G, "pad_contacts", lambda m, d, body: {**real(m, d, body), "left_pad": body == b,
                                                                 "right_pad": False})
    return b


def test_one_pad_box_is_held_and_place_does_not_say_nothing_held(monkeypatch) -> None:
    robot = SimRobot(1000)
    b = _one_pad(monkeypatch, robot, "red box")
    assert G.holding(robot.m, robot.d, b) and robot.held() == "red box"
    calls = []
    monkeypatch.setattr(G, "place_box", lambda *a, **k: calls.append(a) or {"ok": False, "reason": "unreachable",
                                                                            "detail": {}})
    r = Skills(robot).place("storage bench")
    assert r.text == "failed: unreachable" and len(calls) == 1  # precondition passed, place_box ran


def test_box_off_the_pinch_is_not_held() -> None:
    robot = SimRobot(1000)
    assert robot.held() is None and robot.held_any() is None
    assert Skills(robot).place("storage bench").text == "failed: nothing held"


@pytest.mark.parametrize("between,pads,held", [(True, 0, "red box"), (False, 1, "red box"), (True, 2, "red box"),
                                               (False, 0, None)])
def test_scorer_held_is_conservative(monkeypatch, between, pads, held) -> None:
    """Scorer input: "held" = between the pads (grasp.holding) OR any pad contact; the held rule fails for either."""
    robot = SimRobot(1000)
    b = robot.body_of["red box"]
    monkeypatch.setattr(G, "holding", lambda m, d, body: between and body == b)
    monkeypatch.setattr(robot, "pad_touch", lambda: {"red box": pads} if pads else {})
    snap = robot.snapshot(0.1)
    assert snap["held"] == held and snap["holding"] == ("red box" if between else None)
    s = score(snap, {"targets": [{"object": "red box", "surface": "kitchen table"}], "step_limit": 20}, 1)
    assert s["rules"]["held"] is (held is None) and s["detail"]["holding"] == snap["holding"]


def _st(call: str, args: dict, result: str, ok: bool, **meta) -> dict:
    return {"call": call, "args": args, "result": result, "ok": ok, **({"meta": meta} if meta else {})}


def test_lost_box_uses_holding() -> None:
    """The labeller's lost-box rule reads grasp.holding (the scorer detail's "holding"), not the conservative "held":
    a box hanging off one pad at the end (held, not holding) is lost; logs without "holding" fall back to "held"."""
    steps = [_st("pick", {"object": "red box"}, "holding red box", True),
             _st("place", {"surface": "storage bench"}, "failed: blocked", False)]
    end = lambda det: {"sim_ok": True, "score": {"pass": False, "detail": {"targets": {}, **det}}}  # noqa: E731
    assert label_failure(steps, end({"held": "red box", "holding": None}), False) == "skill"
    assert label_failure(steps, end({"held": "red box", "holding": "red box"}), False) == "vlm"
    assert label_failure(steps, end({"held": "red box"}), False) == "vlm"  # 005 logs
    assert label_failure(steps + [_st("place", {"surface": "storage bench"}, "failed: nothing held", False)],
                         end({"held": "red box", "holding": "red box"}), False) == "skill"


# ---- 2. push end keeps its room ----------------------------------------------------------------------------------

def test_room_places_reachable() -> None:
    from sim.push import _sq, room_places_reachable
    from sim.world.generate import load_layout
    from sim.world.occupancy import known_map

    lay = load_layout()
    grid = known_map(lay)
    for r in lay["rooms"].values():  # a speck in each room: every room's places connected on the known map
        c = np.array([(r["rect"][0] + r["rect"][2]) / 2 + 0.01, (r["rect"][1] + r["rect"][3]) / 2 + 0.01])
        assert room_places_reachable(grid, _sq(c, 0.01))
    assert not room_places_reachable(grid, _sq(np.array([6.6, 5.25]), 0.225))  # crate just inside the storage door
    assert not room_places_reachable(grid, _sq(np.array([2.6, 5.25]), 0.225))  # ... the bedroom door


def test_end_cutting_room_places_off_is_not_pushed(monkeypatch) -> None:
    """Stubbed motion: blocking at the call, then the crate ends 0.6 m inside the storage door (corridor / place tests
    stubbed clear): its room's places are cut off from the door, no clearing push -> stuck, never pushed."""
    from test_push import _crate_1000

    calls = iter([(True, False)])

    def teleport(m, d, body, u, dist, cb=None):
        j = m.body(body).jntadr[0]
        d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 2] = (6.6, 5.25)
        mujoco.mj_forward(m, d)
        return {"stalled": False, "why": None, "obj_disp_m": 0.5, "fmax_n": 0.0, "pen_max_m": 0.0, "sat_steps": 0,
                "sat_run_max_s": 0.0}

    P, m, d, grid, door = _crate_1000(
        monkeypatch, _blocks=lambda *a: next(calls, (False, False)), _choose_plan=lambda *a, **k: ("long",),
        go_to=lambda *a, **k: {"reached": True, "contacts": {}}, push_line=teleport,
        drive_base=lambda *a, **k: None, _clear_plan=lambda *a, **k: None)
    r = P.push_object(m, d, grid, "crate", door)
    det = r["detail"]
    assert det["door_result"] == [True, ""] and det["end_cuts_place"] and det["obj_moved_m"] > 0.3
    assert not r["ok"] and r["reason"] == "stuck" and det["why"] == "no room to clear"


def _reject_end(P, monkeypatch, end: np.ndarray) -> None:
    """room_places_reachable stubbed: False only for a footprint centred at `end`."""
    real = P.room_places_reachable
    monkeypatch.setattr(P, "room_places_reachable",
                        lambda grid, poly: not np.allclose(poly.mean(0), end, atol=1e-6) and real(grid, poly))


def test_clear_plan_skips_end_that_cuts_room(monkeypatch) -> None:
    """Plan choice, clearing push: the end _clear_plan picks is rejected when it would cut its room; another is taken."""
    import sim.push as P
    from test_last_stop import _crate_1000

    m, d, grid, _ = _crate_1000()
    j = m.body("crate").jntadr[0]
    d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 2] = (4.5, 6.4)
    mujoco.mj_forward(m, d)
    c, h = P._aabb(m, d, "crate")
    ext, rad = P.footprint(m, d)
    path = np.array([(4.0, 6.4), (5.2, 6.4)])
    u0, s0, *_ = P._clear_plan(m, d, grid, "crate", c, h, ext, rad, path)
    _reject_end(P, monkeypatch, c + u0 * s0)
    u1, s1, *_ = P._clear_plan(m, d, grid, "crate", c, h, ext, rad, path)
    assert not np.allclose(c + u1 * s1, c + u0 * s0)


def test_choose_plan_skips_end_that_cuts_room(monkeypatch) -> None:
    """Plan choice, door push (crate in d_sto, robot in the hall): the first plan's end rejected -> another plan."""
    import sim.push as P
    from sim.world.generate import room_point
    from test_last_stop import _crate_1000

    m, d, grid, _ = _crate_1000()
    C, nv, tv = np.array([6.0, 5.25]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    world = lambda a, l: C + nv * a + tv * l  # noqa: E731
    l_o = float((d.body("crate").xpos[:2] - C) @ tv)
    a_long = float((np.array(room_point("storage")) - C) @ nv) + P.LONG_PAST
    ext, rad = P.footprint(m, d)
    args = (m, d, grid, "crate", world, l_o, 0.225, a_long, "d_sto", l_o, ext, rad)
    p0 = P._choose_plan(*args)
    assert p0[0] == "side"
    _reject_end(P, monkeypatch, world(p0[2], p0[1] * P.SIDE_LAT))
    assert P._choose_plan(*args) != p0


@pytest.mark.parametrize("poses", ["truth", "perceived"])
@pytest.mark.parametrize("key,box", [("blocked-9200@count5a", "red box"), ("blocked-8204@tune5d", "yellow box")])
def test_005_replay_carry_out_plans_around_pushed_crate(tmp_path, key, box, poses) -> None:
    """005 counted 9200 / tuning 8204 (tune5d), was a strict xfail until task 006 5c (row 17 addendum 3): the crate
    push in the prefix now puts the crate's end footprint on the planner map, so the later drives plan around it: the
    pick no longer stops on the crate, and the carry-out from the pick pose reaches the kitchen table."""
    from tune_replay import replay

    robot, rt = replay(key, tmp_path, poses)
    push = next(s for s in rt.steps if s["call"] == "push")
    assert push["result"] == "pushed" and "crate" in robot._pushed
    assert push["meta"]["detail"]["map_footprint"] is not None
    assert rt.steps[-1]["result"] == f"holding {box}"
    assert rt.call("navigate", {"place": "kitchen table", "via": "hall"}).text == "reached"


# ---- 3. tripwire ---------------------------------------------------------------------------------------------------

def test_tripwire_pushed_then_blocked_on_same_object() -> None:
    base = {"sim_ok": True}
    pushed = _st("push", {"object": "crate"}, "pushed", True)
    nav = lambda *objs: _st("navigate", {"place": "storage"}, "failed: blocked", False,  # noqa: E731
                            contacts={f"{o}|49": 0.001 for o in objs})
    pick = _st("pick", {"object": "red box"}, "failed: blocked", False,
               detail={"go_to": [{"reached": True, "contacts": {}}, {"reached": False, "contacts": {"crate|49": 0.001}}]})
    place = _st("place", {"surface": "kitchen table"}, "failed: blocked", False,
                detail={"go_to": {"reached": False, "contacts": {"box_red|77": 1e-4, "crate|49": 0.001}}})
    assert label_failure([pushed, nav("crate")], base, False) == "skill"
    assert label_failure([pushed, pick], base, False) == "skill"
    assert label_failure([pushed, place], base, False) == "skill"
    assert label_failure([nav("crate"), pushed], base, False) == "vlm"  # blocked before the push
    assert label_failure([pushed, nav("cabinet")], base, False) == "vlm"  # another object
    assert label_failure([pushed, nav()], base, False) == "vlm"  # no contact logged
    assert label_failure([_st("push", {"object": "crate"}, "failed: stuck", False), nav("crate")], base, False) == "vlm"
    assert label_failure([pushed, nav("crate")], base, True) == "nav"  # scripted rules unchanged


@pytest.mark.skipif(not (MAIN_RUNS / "count5a" / "blocked-9200").exists(), reason="005 logs not in this checkout")
def test_tripwire_relabels_005_counted_9200() -> None:
    recs = [json.loads(x) for x in (MAIN_RUNS / "count5a" / "blocked-9200" / "log.jsonl").read_text().splitlines()]
    end = next(r for r in recs if r.get("event") == "end")
    assert end["label"] == "vlm" and label_failure([r for r in recs if "step" in r], end, False) == "skill"
