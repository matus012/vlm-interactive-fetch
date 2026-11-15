"""sim.e4.install_move (C3 harness move): state restore on rejection, what an accepted move changes, one attempt."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sim import e4
from sim import tasks as T
from sim.world import generate as gen


def _find_task() -> dict:
    for seed in range(2000, 3000):
        try:
            return e4.make_task_e4(seed, "recovery")
        except T.NoTask:
            continue
    raise AssertionError("no recovery task")


@pytest.fixture(scope="module")
def task() -> dict:
    return _find_task()


def _setup(task: dict, **over):
    from sim.adapter import SimRobot
    from sim.world.house import set_base
    t = task | over
    box = t["targets"][0]["object"]
    w = T.world(t["seed"], "none")
    b = next(x for x in w.boxes if T.box_name(x["colour"]) == box)
    x, y = gen.room_points(b["room"])[0]
    r = SimRobot(t["seed"], "none")
    set_base(r.m, r.d, (x, y, 0.0))
    log = e4.install_move(r, t)
    r.look()
    return r, t, box, b, log


def _snap(r):
    return r.d.qpos.copy(), r.d.qvel.copy(), r.d.act.copy()


def _same(r, s) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(_snap(r), s))


def _slice(r, box):
    j = r.m.body(r.body_of[box]).jntadr[0]
    a, v = int(r.m.jnt_qposadr[j]), int(r.m.jnt_dofadr[j])
    return a, v


def test_all_rejected_no_move_state_identical(task, monkeypatch):
    monkeypatch.setattr(e4, "_view_px", lambda robot, body: 10**6)
    r, t, box, b, log = _setup(task)
    s = _snap(r)
    assert r.locate(box).pose is not None
    assert log["moved"] is False and log["failed"] == "no candidate" and len(log["tried"]) == len(t["move"])
    assert _same(r, s)


def test_own_room_candidate_skipped_state_identical(task):
    box = task["targets"][0]["object"]
    own = [next(x for x in T.world(task["seed"], "none").boxes if T.box_name(x["colour"]) == box)["room"], 1.0, 1.0, 0.0]
    r, t, box, b, log = _setup(task, move=[own])
    s = _snap(r)
    assert r.locate(box).pose is not None
    assert log["moved"] is False and log["tried"] == [[own[0], "robot's room"]]
    assert _same(r, s)


def test_rejection_then_accept_changes_only_target_slice(task, monkeypatch):
    real, calls = e4._view_px, []

    def view(robot, body):
        calls.append(1)
        return 10**6 if len(calls) == 1 else real(robot, body)
    monkeypatch.setattr(e4, "_view_px", view)
    r, t, box, b, log = _setup(task)
    q0, v0, a0 = _snap(r)
    assert r.locate(box).pose is not None
    assert log["moved"] is True and log["tried"] == [[t["move"][0][0], "px 1000000"]] and len(calls) == 2 and log["to"] == t["move"][1]
    a, v = _slice(r, box)
    mask = np.ones(q0.size, bool)
    mask[a:a + 7] = False
    assert np.array_equal(r.d.qpos[mask], q0[mask])
    assert not np.array_equal(r.d.qpos[a:a + 2], q0[a:a + 2])
    assert np.array_equal(r.d.qvel[v:v + 6], np.zeros(6))
    vm = np.ones(v0.size, bool)
    vm[v:v + 6] = False
    assert np.array_equal(r.d.qvel[vm], v0[vm]) and np.array_equal(r.d.act, a0)
    cand = log["to"]
    assert np.allclose(r.d.qpos[a:a + 2], cand[1:3]) and cand[0] != b["room"]


def test_accept_first_candidate(task):
    r, t, box, b, log = _setup(task)
    q0 = r.d.qpos.copy()
    assert r.locate(box).pose is not None
    assert log["moved"] is True
    a, v = _slice(r, box)
    mask = np.ones(q0.size, bool)
    mask[a:a + 7] = False
    assert np.array_equal(r.d.qpos[mask], q0[mask])
    assert np.array_equal(r.d.qvel[v:v + 6], np.zeros(6))


def test_failed_locate_never_moves(task):
    r, t, box, b, log = _setup(task)
    from sim.world.house import set_base
    # robot far from the box and facing a wall: put it in another room's point, look, box out of view
    other = next(k for k in r.layout["rooms"] if k != b["room"] and k in r.layout["spawn_regions"])
    x, y = gen.room_points(other)[0]
    set_base(r.m, r.d, (x, y, 0.0))
    r.look()
    s = _snap(r)
    assert r.locate(box).pose is None
    assert log["moved"] is False and log["tried"] is None
    assert _same(r, s)


def test_second_locate_never_moves_again(task, monkeypatch):
    r, t, box, b, log = _setup(task)
    assert r.locate(box).pose is not None and log["moved"]
    to, calls = list(log["to"]), []
    monkeypatch.setattr(e4, "_set_box", lambda *a: calls.append(a))
    r.look()
    r.locate(box)
    r.locate(box)
    assert log["to"] == to and calls == []


def test_other_object_never_triggers(task):
    r, t, box, b, log = _setup(task)
    other = next(n for n, o in r.objects().items() if o["class"] == "box" and n != box)
    s = _snap(r)
    r.locate(other)
    assert log["tried"] is None and log["moved"] is False and _same(r, s)
