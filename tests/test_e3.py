"""Task 010 (E3): the fixed scripts read only the VLM-visible packet text (E3-truthfree), and every branch does what
agent/e3.py's docstring says, on synthetic packets (no sim)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agent.context import Packet
from agent.e3 import ScriptPolicy, parse

SRC = Path(__file__).resolve().parents[1] / "agent" / "e3.py"
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


def packet(task: str, last: str | None) -> Packet:
    user = f"Task: {task}\nStep 1 of 20.\nYou are in: hall.\n{PLACES}\nHistory:\n(none)\nLast result: {last or '(none)'}\nNo photos."
    return Packet("rules", user, [], [])


def play(variant: str, task: str, answer) -> list[tuple[str, dict]]:  # noqa: ANN001 - answer(call) -> result
    s, calls, last = ScriptPolicy(variant), [], None
    for _ in range(60):
        r = s(packet(task, last), 0)
        if not r.tool_calls:
            break
        c = (r.tool_calls[0]["name"], json.loads(r.tool_calls[0]["arguments"]))
        calls.append(c)
        if c[0] == "done":
            break
        last = answer(c)
    return calls


def test_imports_and_reads_only_the_packet_text() -> None:
    tree = ast.parse(SRC.read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods == {"__future__", "json", "re", "collections.abc", "agent.context", "agent.vlm"}, mods
    reads = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id == "packet"}
    assert reads == {"user"}, reads
    names = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module.startswith("agent.")}
    assert names == {"Packet", "Reply"} and all(len(n.names) == 1 for n in ast.walk(tree)
                                                if isinstance(n, ast.ImportFrom) and n.module.startswith("agent.")), names
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not loads & {"seed", "open", "getattr", "eval", "exec", "__import__", "globals", "vars", "compile"}, loads
    assert not [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr.startswith("__")]


def test_parse() -> None:
    i = parse(packet("Green box to the kitchen table, red box to the storage bench.", "failed: blocked").user)
    assert i["targets"] == [("green box", "kitchen table"), ("red box", "storage bench")]
    assert i["spots"] == ["hall", "bedroom", "storage", "living room", "kitchen", "kitchen east"]
    assert i["round_doors"] == ["kitchen north door", "kitchen south door"]
    assert i["last"] == "failed: blocked"


def found_in(room: str, box: str = "green box"):  # noqa: ANN201
    state = {"at": "hall"}

    def answer(c: tuple[str, dict]) -> str:
        n, a = c
        if n == "navigate":
            state["at"] = a["place"]
            return "reached"
        if n == "locate":
            return f"found: {box} in {room}, on floor" if state["at"] == room else "failed: not in view"
        return {"look": "4 photos", "pick": f"holding {box}", "place": "placed"}.get(n, "?")
    return answer


def test_search_in_order_then_deliver() -> None:
    calls = play("e3a", "Bring the green box to the kitchen table.", found_in("storage"))
    assert calls == [("look", {}), ("locate", {"object": "green box"}), ("navigate", {"place": "bedroom"}), ("look", {}),
                     ("locate", {"object": "green box"}), ("navigate", {"place": "storage"}), ("look", {}),
                     ("locate", {"object": "green box"}), ("pick", {"object": "green box"}),
                     ("navigate", {"place": "kitchen table"}), ("place", {"surface": "kitchen table"}),
                     ("done", {"note": "script e3a: delivered green box"})]


def test_not_found_anywhere_gives_done() -> None:
    calls = play("e3a", "Bring the green box to the kitchen table.", found_in("nowhere"))
    assert [c[1]["place"] for c in calls if c[0] == "navigate"] == ["bedroom", "storage", "living room", "kitchen", "kitchen east"]
    assert calls[-1] == ("done", {"note": "script e3a: delivered no box"})


def blocked(push_answer: dict[str, str], room: str = "kitchen", box_room: str = "bedroom"):  # noqa: ANN201
    """navigate to `room` is blocked until pushed / taken with via='kitchen south door'; the box is in `box_room`."""
    base, state = found_in(box_room), {"cleared": False}

    def answer(c: tuple[str, dict]) -> str:
        n, a = c
        if n == "push":
            r = push_answer[a["object"]]
            state["cleared"] |= r == "pushed"
            return r
        if n == "navigate" and a["place"] in (room, "kitchen table") and not state["cleared"]:
            return "reached" if a.get("via") == "kitchen south door" else "failed: blocked"
        return base(c)
    return answer


@pytest.mark.parametrize("variant", ["e3a", "e3b"])
def test_crate_pushed_then_step_retried(variant: str) -> None:
    calls = play(variant, "Bring the green box to the kitchen table.", blocked({"crate": "pushed"}))
    i = calls.index(("navigate", {"place": "kitchen table"}))
    assert calls[i + 1:i + 3] == [("push", {"object": "crate"}), ("navigate", {"place": "kitchen table"})]
    assert ("place", {"surface": "kitchen table"}) in calls


def test_no_crate_pushes_cabinet_e3a_gives_up() -> None:
    ans = blocked({"crate": "error: unknown object 'crate'", "cabinet": "failed: not movable"})
    calls = play("e3a", "Bring the green box to the kitchen table.", ans)
    i = calls.index(("navigate", {"place": "kitchen table"}))
    assert calls[i + 1:] == [("push", {"object": "crate"}), ("push", {"object": "cabinet"}),
                             ("done", {"note": "script e3a: delivered no box"})]


def test_e3b_goes_round_by_the_doors_in_order() -> None:
    ans = blocked({"crate": "error: unknown object 'crate'", "cabinet": "failed: not movable"})
    calls = play("e3b", "Bring the green box to the kitchen table.", ans)
    i = calls.index(("push", {"object": "cabinet"}))
    assert calls[i + 1:i + 3] == [("navigate", {"place": "kitchen table", "via": "kitchen north door"}),
                                  ("navigate", {"place": "kitchen table", "via": "kitchen south door"})]
    assert calls[i + 3] == ("place", {"surface": "kitchen table"})


def test_blocked_search_moves_on_to_the_next_room() -> None:
    ans = blocked({"crate": "failed: stuck"}, room="storage", box_room="living room")
    calls = play("e3a", "Bring the green box to the kitchen table.", ans)
    navs = [c[1]["place"] for c in calls if c[0] == "navigate"]
    assert navs[:3] == ["bedroom", "storage", "living room"]


def test_e3b_pick_blocked_goes_round_via_the_found_spot() -> None:
    base, n = found_in("kitchen"), {"pick": 0}

    def ans(c: tuple[str, dict]) -> str:
        if c[0] == "pick":
            n["pick"] += 1
            return "failed: blocked" if n["pick"] == 1 else "holding green box"
        if c[0] == "push":
            return "error: unknown object 'crate'" if c[1]["object"] == "crate" else "failed: not movable"
        return base(c)
    calls = play("e3b", "Bring the green box to the storage bench.", ans)
    i = calls.index(("push", {"object": "cabinet"}))
    assert calls[i + 1:i + 3] == [("navigate", {"place": "kitchen", "via": "kitchen north door"}), ("pick", {"object": "green box"})]


def test_two_box_second_search_restarts_the_order() -> None:
    state = {"at": "hall"}
    where = {"green box": "storage", "red box": "bedroom"}

    def ans(c: tuple[str, dict]) -> str:
        n, a = c
        if n == "navigate":
            state["at"] = a["place"]
            return "reached"
        if n == "locate":
            return "found" if where[a["object"]] == state["at"] else "failed: not in view"
        return {"look": "4 photos", "pick": "holding", "place": "placed"}[n]
    calls = play("e3a", "Green box to the kitchen table, red box to the storage bench.", ans)
    navs = [c[1]["place"] for c in calls if c[0] == "navigate"]
    assert navs == ["bedroom", "storage", "kitchen table", "bedroom", "storage bench"]
    assert calls[-1] == ("done", {"note": "script e3a: delivered green box, red box"})
