"""Fix (d) (task 005): the adapter's last-stop memory. push(object) treats an object that stopped the last drive
as blocking and moves it clear of that drive's path; only push reads the memory; it never reaches the VLM."""
import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest

from agent.context import build_packet
from agent.leak import scan
from sim.adapter import SimRobot
from tune_replay import replay

ROOT = Path(__file__).resolve().parents[1]
PUSH_TEXTS = {"pushed", "failed: stuck", "failed: not movable", "failed: not blocking"}


def _reads(src: str, attr: str = "_last_stop") -> list[tuple[str | None, int]]:
    """(enclosing function name, line) of every read of attr (attribute load, or the name as a string)."""
    out: list[tuple[str | None, int]] = []

    def walk(node: ast.AST, fn: str | None) -> None:
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if (isinstance(ch, ast.Attribute) and ch.attr == attr and isinstance(ch.ctx, ast.Load)) or \
                    (isinstance(ch, ast.Constant) and ch.value == attr):
                out.append((f, ch.lineno))
            walk(ch, f)

    walk(ast.parse(src), None)
    return out


def test_only_push_reads_last_stop() -> None:
    """Static: in sim/ and agent/, the memory is read only inside SimRobot.push (navigate never re-plans around a
    blocker; nothing else may see it)."""
    hits = {str(p.relative_to(ROOT)): _reads(p.read_text()) for d in ("sim", "agent") for p in (ROOT / d).rglob("*.py")}
    hits = {k: v for k, v in hits.items() if v}
    assert set(hits) == {"sim/adapter.py"} and {f for f, _ in hits["sim/adapter.py"]} == {"push"}, hits


def test_static_check_catches_navigate_read() -> None:
    src = (ROOT / "sim/adapter.py").read_text()
    mut = src.replace("        p = self._places[place]\n", "        p = self._places[place]\n        _ = self._last_stop\n", 1)
    assert mut != src and ("navigate", mut.count("\n", 0, mut.index("_ = self._last_stop")) + 1) in _reads(mut)


def test_note_drive_records_and_forgets() -> None:
    """A drive stopped on the crate is remembered with its path from the stop (projected) to the goal; a later drive
    that reaches its goal forgets it."""
    r = SimRobot(1000, "crate")
    nav = {"reached": False, "waypoints": [(1.0, 1.0), (3.0, 1.0), (3.0, 4.0)], "contacts": {"crate|bumper": 0.001},
           "blocked": {"why": "bumper", "pose": [2.0, 1.1, 0.0]}}
    r._note_drive(nav)
    assert r._last_stop["body"] == "crate" and np.allclose(r._last_stop["path"], [(2.0, 1.0), (3.0, 1.0), (3.0, 4.0)])
    r._note_drive({**nav, "contacts": {"wall_3|bumper": 0.001}})  # not on a blocker: unchanged
    assert r._last_stop["body"] == "crate"
    n = r.navigate("hall")
    assert n.status == "reached" and r._last_stop is None


def _crate_1000(monkeypatch=None):
    from sim.push_check import door_dict, start_pose
    from sim.reach_check import compile_house, start_state
    from sim.world.generate import generate
    from sim.world.occupancy import known_map

    w = generate(seed=1000, blocker="crate")
    m = compile_house(w)
    door = door_dict(w.layout, w.blocker["door"])
    d = start_state(m, start_pose(door, "hall", w.layout))
    return m, d, known_map(w.layout), {k: v for k, v in door.items() if k != "centre"}


def test_stopped_object_is_blocking_when_row17_says_clear() -> None:
    """Crate in the hall at (4.5, 6.4): row 17 says not blocking (test_push), but with a stop path through it push
    moves it and ends with its footprint > ROBOT_RADIUS + PATH_CLEAR off the path and row 17 met."""
    import sim.push as P
    from sim.constants import ROBOT_RADIUS

    m, d, grid, door = _crate_1000()
    j = m.body("crate").jntadr[0]
    d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 2] = (4.5, 6.4)
    mujoco.mj_forward(m, d)
    assert not any(P._blocks(*P._aabb(m, d, "crate")))
    path = np.array([(4.0, 6.4), (5.2, 6.4)])
    r = P.push_object(m, d, grid, "crate", door, stop_path=path)
    det = r["detail"]
    assert det["stopped_last_drive"] and det["start_on_path"] and r["reason"] != "not blocking"
    assert r["ok"], (det["why"], det.get("clear_plan"))
    assert det["end_path_dist_m"] > ROBOT_RADIUS + P.PATH_CLEAR and not any(P._blocks(*P._aabb(m, d, "crate")))


def test_no_clear_end_is_stuck() -> None:
    """Crate in d_sto, stop path sweeping the whole house: the door push moves it, no clearing push can end off the
    path -> stuck (moved, so not "not movable")."""
    import sim.push as P

    m, d, grid, door = _crate_1000()
    path = np.array([(x, y) for i, y in enumerate(np.arange(0.0, 7.01, 0.25)) for x in ((0.0, 9.0) if i % 2 else (9.0, 0.0))])
    r = P.push_object(m, d, grid, "crate", door, stop_path=path)
    det = r["detail"]
    assert not r["ok"] and r["reason"] == "stuck" and det["why"] == "no room to clear" and det["obj_moved_m"] > 0.5
    assert det["end_on_path"]


@pytest.mark.parametrize("task_id,nxt", [("blocked-8204", ("pick", {"object": "yellow box"})),
                                         ("blocked-8208", ("pick", {"object": "yellow box"})),
                                         ("blocked-8206", ("navigate", {"place": "kitchen table"}))])
def test_tune5c_replay_push_clears_stopped_path(tmp_path, monkeypatch, task_id, nxt) -> None:
    """Tuning round 3: the logged calls up to the drive the pushed crate stopped (logged: push -> "not blocking",
    run lost). Now push(crate) -> pushed, and the call that was blocked succeeds. The memory is not in the packet
    (identical with it removed) nor in any result text.
    Task 006 5c: with the pushed-object map (row 17 addendum 3) the prefix's earlier push puts the crate on the map and
    the drive is no longer stopped (test_006_prework); the prefix is replayed with that map entry suppressed to rebuild
    the 005 state fix (d) is for (a drive stopped by a crate not on the map), then the map is live again."""
    with monkeypatch.context() as mp:
        mp.setattr(SimRobot, "_map_pushed", lambda self, obj, det: None)
        robot, rt = replay(task_id, tmp_path)
    assert rt.steps[-1]["result"] == "failed: blocked" and robot._last_stop["body"] == "crate"
    pk = build_packet(robot, rt.skills, rt, rt.task)
    mem, robot._last_stop = robot._last_stop, None
    assert build_packet(robot, rt.skills, rt, rt.task) == pk and not scan(pk.user)
    robot._last_stop = mem
    r = rt.call("push", {"object": "crate"})
    assert r.text == "pushed", r.meta["detail"].get("why")
    assert robot._last_stop is None
    r2 = rt.call(*nxt)
    assert r2.ok, r2.text
    texts = [s["result"] for s in rt.steps]
    assert {t for s, t in zip(rt.steps, texts) if s["call"] == "push"} <= PUSH_TEXTS
    assert not any(scan(t) for t in texts) and not any("path" in t or "stop" in t for t in texts)
