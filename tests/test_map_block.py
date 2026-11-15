"""Row 17 addendum 4 (task 006 5d): the adapter's map-blocked memory (analogue of fix (d)) and push on the planner map.
Set when a drive / pick fails on planning, with the pushed objects whose map entry alone stops the plan; cleared by a
drive that reaches its goal or a successful pick; read only by push (static, as test_last_stop)."""
from pathlib import Path

import numpy as np
import pytest

import sim.push as P
from agent.robot_api import Pose
from sim.adapter import SimRobot
from test_last_stop import _crate_1000 as _crate_1000_raw
from test_last_stop import _reads

ROOT = Path(__file__).resolve().parents[1]


def _map_crate(robot: SimRobot, x: float, y: float, h: float = 0.365) -> None:
    """Test fixture: a pushed-crate map entry at (x, y) (as `_map_pushed` would stamp it)."""
    robot._pushed["crate"] = (x, y, 0.0, h, h)
    robot.grid = robot._map_grid()


def test_only_push_reads_map_block() -> None:
    hits = {str(p.relative_to(ROOT)): _reads(p.read_text(), "_map_block")
            for d in ("sim", "agent") for p in (ROOT / d).rglob("*.py")}
    hits = {k: v for k, v in hits.items() if v}
    assert set(hits) == {"sim/adapter.py"} and {f for f, _ in hits["sim/adapter.py"]} == {"push"}, hits


def test_static_check_catches_navigate_read() -> None:
    src = (ROOT / "sim/adapter.py").read_text()
    mut = src.replace("        p = self._places[place]\n", "        p = self._places[place]\n        _ = self._map_block\n", 1)
    assert mut != src and ("navigate", mut.count("\n", 0, mut.index("_ = self._map_block")) + 1) in _reads(mut, "_map_block")


def test_navigate_no_path_sets_and_reached_clears() -> None:
    """Crate's map entry in the storage door: navigate(storage) -> no path; remembered with the crate (the plan passes
    without its entry). A drive that reaches its goal clears it."""
    r = SimRobot(1000, "crate")
    _map_crate(r, 6.0, 5.25)
    assert r.navigate("storage").status == "no_path"
    mb = r._map_block
    assert mb["bodies"] == {"crate"} and not mb["plan"](r.grid) and mb["plan"](r._map_grid("crate"))
    assert r.navigate("hall").status == "reached" and r._map_block is None


def test_no_path_not_from_a_pushed_entry_names_nobody() -> None:
    """The plan fails with the crate's entry removed too (goal sealed by a second, unpushed obstacle): bodies empty,
    so push gets no need_plan."""
    r = SimRobot(1000, "crate")
    _map_crate(r, 6.0, 5.25)
    door = r._door_gap("d_sto", 0.4)
    occ = r._known.occ.copy()
    r._stamp_rect(occ, door)
    r._known = type(r._known)(r._known.origin, r._known.res, occ)
    r.grid = r._map_grid()
    assert r.navigate("storage").status == "no_path" and r._map_block["bodies"] == set()


def test_pick_no_pregrasp_sets_and_pick_ok_clears(monkeypatch) -> None:
    """Perceived pick, located box with the crate's map entry over it: no pre-grasp pose -> remembered with the crate
    (pre-grasp feasible at the located pose without it). A successful pick clears it."""
    r = SimRobot(1000, "crate")
    t = r.true_pose("red box")  # test fixture: the located pose = where the box is
    r._located["red box"] = Pose(t.x, t.y, t.z, t.yaw)
    _map_crate(r, t.x, t.y, 0.6)
    res = r.pick("red box")
    assert res.reason == "unreachable" and res.detail["why"] == "no pre-grasp pose / path"
    assert r._map_block["bodies"] == {"crate"} and r._map_block["plan"](r._map_grid("crate"))
    import sim.grasp as G

    r._located["red box"] = Pose(t.x, t.y, t.z, t.yaw)
    monkeypatch.setattr(G, "pick_box", lambda *a, **k: {"ok": True, "reason": "", "detail": {}})
    assert r.pick("red box").ok and r._map_block is None


def test_push_gets_need_plan_only_for_a_named_body(monkeypatch) -> None:
    r = SimRobot(1000, "crate")
    _map_crate(r, 6.0, 5.25)
    r.navigate("storage")
    seen: list = []

    def fake(m, d, grid, body, door, **k):
        seen.append(k)
        return {"ok": False, "reason": "not blocking", "detail": {"obj_moved_m": 0.0}}

    monkeypatch.setattr(P, "push_object", fake)
    r.push("crate")
    assert seen[-1]["need_plan"] is r._map_block["plan"] and seen[-1]["map_margin"] == pytest.approx(0.14)
    r._map_block["bodies"] = set()
    r.push("crate")
    assert seen[-1]["need_plan"] is None


def test_push_tests_read_the_map_entry_not_the_sim_pose() -> None:
    """push_object with est: the crate truly stands in the storage door, but its map entry (est) is in the hall clear
    of every corridor / place: "not blocking", nothing moved. Without est (the sim pose) it is pushed at."""
    m, d, grid, door = _crate_1000_raw()
    r = P.push_object(m, d, grid, "crate", door, est=lambda det: (4.5, 6.4, 0.0), map_margin=0.14)
    assert r["reason"] == "not blocking" and r["detail"]["obj_moved_m"] == 0.0
    assert P._blocks(*P._aabb(m, d, "crate"))[0]  # the sim pose is in the corridor


def test_map_blocked_push_moves_until_plan_passes() -> None:
    """Crate in the hall at (4.5, 6.4), clear by row 17; need_plan = a plan that needs the cells around (4.5, 6.4)
    free: the crate is blocking, gets the clearing push, and "pushed" only with need_plan passing on its end's map
    footprint."""
    import mujoco

    m, d, grid, door = _crate_1000_raw()
    j = m.body("crate").jntadr[0]
    d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 2] = (4.5, 6.4)
    mujoco.mj_forward(m, d)
    est = lambda det: (P.contact_footprint(det["last_contact"], (0.225, 0.225)) if "last_contact" in det  # noqa: E731
                       else (4.5, 6.4, 0.0))
    need = lambda g: not g.occ[g.world_to_cell(4.5, 6.4)[::-1]]  # noqa: E731
    r = P.push_object(m, d, grid, "crate", door, est=est, map_margin=0.14, need_plan=need)
    det = r["detail"]
    assert det["start_blocks_plan"] and r["reason"] != "not blocking"
    assert r["ok"], det["why"]
    assert not det["end_blocks_plan"] and np.linalg.norm(np.array(det["obj_end"][:2]) - (4.5, 6.4)) > 0.3


PUSH_TEXTS = {"pushed", "failed: stuck", "failed: not movable"}  # never "failed: not blocking" in these replays


def _texts_clean(rt) -> None:
    from agent.leak import scan

    texts = [s["result"] for s in rt.steps]
    assert not any(scan(t) for t in texts) and not any("plan" in t or "map" in t for t in texts)


def test_g5_9204_replay_place_after_push(tmp_path) -> None:
    """G5 counted run 1, blocked-9204, perceived: carrying the red box, navigate(storage) stopped on the crate. Logged:
    push -> pushed, navigate -> no path, place -> unreachable, push -> not blocking (x2). Now the push's tests read the
    crate's map entry (pushed or stuck) and the robot steps off it, so navigate(storage) and place(storage bench)
    succeed."""
    from tune_replay import replay

    robot, rt = replay("blocked-9204@g5run1", tmp_path, poses="perceived")
    assert rt.steps[-1]["result"] == "failed: blocked"
    r = rt.call("push", {"object": "crate"})
    assert r.text in PUSH_TEXTS and r.text != "failed: not movable", r.text
    assert rt.call("navigate", {"place": "storage"}).text == "reached"
    assert rt.call("place", {"surface": "storage bench"}).text == "placed"
    _texts_clean(rt)


def test_g5_9225_replay_map_blocked_pick(tmp_path) -> None:
    """G5 counted run 1, blocked-9225, perceived: after push(crate) -> pushed, pick(green box) -> unreachable (no
    pre-grasp pose: the box sits behind the crate's map entry); logged push -> not blocking. Now the failed pick is
    remembered with the crate; push treats it as blocking (map-blocked) and moves it until the pre-grasp plan passes on
    the map; after locating again, pick holds the box. The memory is not in the packet."""
    from agent.context import build_packet
    from agent.leak import scan
    from tune_replay import replay

    robot, rt = replay("blocked-9225@g5run1", tmp_path, poses="perceived")
    for c in (("push", {"object": "crate"}), ("navigate", {"place": "storage"}), ("look", {}),
              ("locate", {"object": "green box"})):
        rt.call(*c)
    r = rt.call("pick", {"object": "green box"})
    assert r.text == "failed: unreachable" and robot._map_block["bodies"] == {"crate"}
    pk = build_packet(robot, rt.skills, rt, rt.task)
    mem, robot._map_block = robot._map_block, None
    assert build_packet(robot, rt.skills, rt, rt.task) == pk and not scan(pk.user)
    robot._map_block = mem
    r = rt.call("push", {"object": "crate"})
    det = r.meta["detail"]
    assert r.text in PUSH_TEXTS and det["map_blocked"] and det["start_blocks_plan"], (r.text, det.get("why"))
    if r.text == "pushed":
        rt.call("look", {})
        rt.call("locate", {"object": "green box"})
        assert rt.call("pick", {"object": "green box"}).text == "holding green box"
    _texts_clean(rt)
