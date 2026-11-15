"""Task 014 (E4): E3c (agent/e3c.py) reads only the packet text (truth-free, with a mutant check), and every rule of its
docstring does what it says, on synthetic packets; one dev-seed run through the real Runtime (skip with E3C_SKIP_SIM=1)."""
from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path

import pytest

from agent.context import Packet
from agent.e3c import ScriptPolicy, info_of, surface_of

SRC = Path(__file__).resolve().parents[1] / "agent" / "e3c.py"
PLACES = """Places (for navigate):
- hall: room look spot, in hall
- bedroom: room look spot, in bedroom
- storage: room look spot, in storage
- living room: room look spot, in living room
- kitchen: room look spot, in kitchen
- kitchen east: room look spot, in kitchen
- kitchen table: surface, in kitchen
- storage bench: surface, in storage
- bedroom door: door, between hall and bedroom
- storage door: door, between hall and storage
- living room door: door, between hall and living room
- kitchen north door: door, between living room and kitchen
- kitchen south door: door, between living room and kitchen
Surfaces (for place): kitchen table, storage bench.
Objects: red box, blue box, green box, yellow box, crate, cabinet."""
BOXES = ["red box", "blue box", "green box", "yellow box"]


def packet(task: str, last: str | None) -> Packet:
    user = f"Task: {task}\nStep 1 of 20.\nYou are in: hall.\n{PLACES}\nHistory:\n(none)\nLast result: {last or '(none)'}\nNo photos."
    return Packet("rules", user, [], [])


def play(task: str, answer, n: int = 80) -> list[tuple[str, dict]]:  # noqa: ANN001 - answer(call) -> result
    s, calls, last = ScriptPolicy("e3c"), [], None
    for _ in range(n):
        r = s(packet(task, last), 0)
        if not r.tool_calls:
            break
        c = (r.tool_calls[0]["name"], json.loads(r.tool_calls[0]["arguments"]))
        calls.append(c)
        if c[0] == "done":
            break
        last = answer(c)
    return calls


def audit(src: str) -> list[str]:
    """Problems with a source: imports, packet attribute reads, forbidden names (the purity contract)."""
    tree, bad = ast.parse(src), []
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    if mods != {"__future__", "json", "re", "collections.abc", "agent.context", "agent.e3", "agent.vlm"}:
        bad.append(f"imports {mods}")
    reads = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id == "packet"}
    if reads != {"user"}:
        bad.append(f"packet reads {reads}")
    names = sorted(a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module.startswith("agent.")
                   for a in n.names)
    if names != ["Packet", "Reply", "parse"]:
        bad.append(f"agent imports {names}")
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if loads & {"seed", "open", "getattr", "eval", "exec", "__import__", "globals", "vars", "compile"}:
        bad.append(f"names {loads}")
    if [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr.startswith("__")]:
        bad.append("dunder attribute")
    return bad


def test_imports_and_reads_only_the_packet_text() -> None:
    assert audit(SRC.read_text()) == []


@pytest.mark.parametrize("mutant", [
    ("        info = info_of(packet.user)", "        info = info_of(packet.user)\n        _ = packet.photos"),
    ("        info = info_of(packet.user)", "        info = info_of(packet.user)\n        _ = seed"),
    ("import re\n", "import re\nimport os\n"),
    ("        info = info_of(packet.user)", "        info = info_of(packet.user)\n        _ = getattr(packet, 'poses')"),
])
def test_audit_catches_a_mutant_that_reads_something_forbidden(mutant: tuple[str, str]) -> None:
    src = SRC.read_text()
    assert mutant[0] in src
    assert audit(src.replace(*mutant, 1)), mutant


def test_surface_parse() -> None:
    for task, want in [("Bring the box next to the sofa to the storage bench.", "storage bench"),
                       ("Bring something to drink to the kitchen table.", "kitchen table"),
                       ("Bring the box on the nightstand to the kitchen table.", "kitchen table"),
                       ("Bring the box by the bed to the storage bench.", "storage bench"),
                       ("Bring the box by the kitchen table to the storage bench.", "storage bench")]:
        assert surface_of(packet(task, None).user) == want, task
        assert info_of(packet(task, None).user)["targets"] == [(None, want)]


def test_colour_task_with_a_hint_parses_the_colour_target() -> None:
    for t in ("Bring the yellow box to the kitchen table.", "Bring the yellow box to the kitchen table; someone left it in the bedroom."):
        i = info_of(packet(t, None).user)
        assert i["targets"] == [("yellow box", "kitchen table")] and not i["colourless"], t


def world(box_room: dict[str, str], pick: list[str] | None = None, moved: dict | None = None):  # noqa: ANN201
    """box -> room; locate answers 'found' for a box in the room the robot is in. `pick` = the pick answers in order
    (default: holding). `moved`: after the first successful locate the boxes in it change room (E4 'recovery')."""
    st, picks, first = {"at": "hall"}, list(pick or []), {"done": False}
    rooms = {"kitchen east": "kitchen"}

    def ans(c: tuple[str, dict]) -> str:
        n, a = c
        if n == "navigate":
            st["at"] = a["place"]
            return "reached"
        if n == "locate":
            if box_room.get(a["object"]) == rooms.get(st["at"], st["at"]):
                if moved and not first["done"]:
                    first["done"] = True
                    box_room.update(moved)
                return f"found: {a['object']} in {st['at']}, on floor"
            return "failed: not in view"
        if n == "pick":
            return picks.pop(0) if picks else f"holding {a['object']}"
        return {"look": "4 photos", "place": "placed"}[n]
    return ans


def test_no_colour_locates_in_colour_order_at_each_spot_and_the_first_found_is_picked() -> None:
    calls = play("Bring something to drink to the kitchen table.", world({"green box": "storage", "yellow box": "storage",
                                                                            "blue box": "living room"}))
    loc = lambda: [("locate", {"object": b}) for b in BOXES]  # noqa: E731
    assert calls[:2] == [("look", {}), ("locate", {"object": "red box"})]
    assert calls[:1] + loc() + [("navigate", {"place": "bedroom"}), ("look", {})] + loc() == calls[:1 + 4 + 2 + 4]
    j = calls.index(("navigate", {"place": "storage"}))
    assert calls[j + 1:] == [("look", {}), ("locate", {"object": "red box"}), ("locate", {"object": "blue box"}),
                             ("locate", {"object": "green box"}), ("pick", {"object": "green box"}),
                             ("navigate", {"place": "kitchen table"}), ("place", {"surface": "kitchen table"}),
                             ("done", {"note": "script e3c: delivered green box"})]


def test_no_colour_nothing_found_gives_done() -> None:
    calls = play("Bring the box on the nightstand to the storage bench.", world({}))
    assert [c[1]["place"] for c in calls if c[0] == "navigate"] == ["bedroom", "storage", "living room", "kitchen", "kitchen east"]
    assert calls[-1] == ("done", {"note": "script e3c: delivered no box"})
    assert sum(c[0] == "locate" for c in calls) == 4 * 6


def test_failed_pick_looks_and_locates_again_then_picks() -> None:
    for fail in ("failed: missed", "failed: slipped", "failed: not in view", "failed: not located", "error: x"):
        calls = play("Bring the yellow box to the kitchen table.", world({"yellow box": "storage"}, [fail]))
        i = calls.index(("pick", {"object": "yellow box"}))
        assert calls[i + 1:i + 5] == [("look", {}), ("locate", {"object": "yellow box"}), ("pick", {"object": "yellow box"}),
                                      ("navigate", {"place": "kitchen table"})], fail
        assert calls[-1] == ("done", {"note": "script e3c: delivered yellow box"})


def test_failed_pick_box_moved_resumes_the_search_from_the_next_room() -> None:
    # recovery category: the box is in storage at first; after the first successful locate it is in the kitchen
    calls = play("Bring something to drink to the kitchen table.",
                 world({"blue box": "storage"}, ["failed: missed"], moved={"blue box": "kitchen"}))
    i = calls.index(("pick", {"object": "blue box"}))
    assert calls[i + 1:i + 4] == [("look", {}), ("locate", {"object": "blue box"}), ("navigate", {"place": "living room"})]
    assert calls[i + 4:i + 6] == [("look", {}), ("locate", {"object": "blue box"})]
    assert calls[i + 6:i + 11] == [("navigate", {"place": "kitchen"}), ("look", {}), ("locate", {"object": "blue box"}),
                                   ("pick", {"object": "blue box"}), ("navigate", {"place": "kitchen table"})]
    assert calls[-1] == ("done", {"note": "script e3c: delivered blue box"})


def test_second_failed_pick_no_second_recovery() -> None:
    calls = play("Bring the yellow box to the kitchen table.", world({"yellow box": "storage"}, ["failed: missed", "failed: slipped"]))
    assert [c[0] for c in calls].count("pick") == 2
    assert calls[-2:] == [("pick", {"object": "yellow box"}), ("done", {"note": "script e3c: delivered no box"})]


def test_recovery_is_once_per_run_across_two_boxes() -> None:
    calls = play("Green box to the kitchen table, red box to the storage bench.",
                 world({"green box": "storage", "red box": "bedroom"}, ["failed: missed", "holding", "failed: missed"]))
    assert [c[0] for c in calls].count("pick") == 3  # green (fail, retry ok), red (fail, no retry)
    assert calls[-1] == ("done", {"note": "script e3c: delivered green box"})


def test_failed_pick_nothing_found_anywhere_gives_done() -> None:
    calls = play("Bring the yellow box to the kitchen table.",
                 world({"yellow box": "kitchen"}, ["failed: missed"], moved={"yellow box": "nowhere"}))
    assert [c[0] for c in calls].count("pick") == 1 and calls[-1][0] == "done"
    assert [c[1]["place"] for c in calls if c[0] == "navigate"][-1] == "kitchen east"


def test_blocked_pick_is_not_a_failed_pick() -> None:
    # blocked pick -> push (not movable), go round via both doors still blocked: no look/locate recovery
    n = {"pick": 0}
    base = world({"yellow box": "kitchen"})

    def ans(c: tuple[str, dict]) -> str:
        if c[0] == "pick":
            return "failed: blocked"
        if c[0] == "push":
            return "failed: not movable"
        if c[0] == "navigate" and c[1].get("via"):
            return "reached"
        return base(c)
    calls = play("Bring the yellow box to the kitchen table.", ans)
    i = calls.index(("pick", {"object": "yellow box"}))
    assert ("look", {}) not in calls[i:] and calls[-1][0] == "done"


def test_e3b_blocked_rule_kept() -> None:
    base, st = world({"yellow box": "bedroom"}), {"cleared": False}

    def ans(c: tuple[str, dict]) -> str:
        if c[0] == "push":
            st["cleared"] = True
            return "pushed"
        if c == ("navigate", {"place": "kitchen table"}) and not st["cleared"]:
            return "failed: blocked"
        return base(c)
    calls = play("Bring the yellow box to the kitchen table.", ans)
    i = calls.index(("navigate", {"place": "kitchen table"}))
    assert calls[i + 1:i + 3] == [("push", {"object": "crate"}), ("navigate", {"place": "kitchen table"})]
    assert calls[-1] == ("done", {"note": "script e3c: delivered yellow box"})


@pytest.mark.skipif(os.environ.get("E3C_SKIP_SIM") == "1", reason="E3C_SKIP_SIM=1")
def test_dev_seed_no_colour_run_ends_in_done_within_the_step_limit() -> None:
    from agent.loop import end_run, run_agent
    from agent.runtime import Runtime
    from agent.skills import Skills
    from sim.adapter import SimRobot
    from sim.tasks import make_task
    seed = 1000
    task = make_task(seed, "fetch")
    surface = task["targets"][0]["surface"]
    task["instruction"] = f"Bring something to drink to the {surface}."  # scorer still uses targets
    with tempfile.TemporaryDirectory() as td:
        robot = SimRobot(task["seed"], task["blocker"], photo_dir=Path(td) / "photos", poses="perceived")
        skills = Skills(robot)
        rt = Runtime(skills, robot, task, Path(td) / "log.jsonl", scripted=False)
        run_agent(robot, skills, rt, task, ScriptPolicy("e3c"), td)
        end = end_run(robot, rt, task)
        calls = [(s["call"], s["result"]) for s in rt.steps]
    print(calls, end["score"])
    assert end["done_called"] and calls[-1][0] == "done" and end["steps_used"] <= task["step_limit"], calls
