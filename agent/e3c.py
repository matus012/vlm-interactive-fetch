"""Task 014 (E4): E3c = E3b plus two fixed rules. Same interface as agent/e3.py: `ScriptPolicy('e3c')` is a callable
`(packet, seed) -> Reply`, one instance per run, zero calls after done. Stdlib only; reads ONLY `packet.user` (never
photos, poses, maps, seeds or the sim; tests/test_e3c.py checks the imports and the attribute reads). agent/e3.py is
imported (parse) and stays unchanged.

E3c is E3b (search in order, push, else go round; see agent/e3.py) with:
(i) No colour in the task text ("Bring something to drink to the kitchen table.", "Bring the box next to the sofa to
    the storage bench."): the target surface is the "Surfaces (for place)" name in the text (if two occur, the one after
    the LAST " to the "). Search as E3b does (look in the hall, then each room look spot in packet order, the hall
    skipped), and at each spot, after look, locate the boxes in the fixed colour order red, blue, green, yellow: the
    FIRST box that answers "found..." is the target. Then pick / navigate to the surface / place / done as E3b. No box
    found anywhere -> done. Coloured tasks ("Bring the yellow box to the kitchen table; someone left it in the
    bedroom.") parse as E3b (the hint is ignored) and locate only that box.
(ii) After a failed pick, ONCE per run: the pick result starts with "failed" or "error" and the pick step did not meet
    the blocked rule. Real pick results (agent/skills.py, sim/adapter.py pick, sim/grasp.py): "failed: not located",
    "failed: missed", "failed: slipped", "failed: unreachable", "failed: not a box", "failed: already holding
    something", "failed: hand full", "failed: sim error", "error: unknown object ..."; "failed: blocked" (pick stopped on
    a crate / cabinet) goes through the blocked / push / go-round rule instead, never through this recovery.
    Recovery: look where the robot stands, locate the same box; found -> pick again; not found -> resume the search
    from the next look spot after the current one in the order (hall, then the packet order), navigate / look / locate
    that box at each; found -> pick again. A second failed pick (or nothing found) -> done.
Everything else is E3b, including the blocked / push / go-round rule and the done note.
"""
from __future__ import annotations

import json
import re
from collections.abc import Generator

from agent.context import Packet
from agent.e3 import parse
from agent.vlm import Reply

Call = tuple[str, dict]
Gen = Generator[Call, str, str]
BOXES = ("red box", "blue box", "green box", "yellow box")


def surface_of(user: str) -> str | None:
    """The place-surface named in the task text: the one after the last ' to the ' (else the last one named)."""
    lines = user.splitlines()
    task = lines[0].removeprefix("Task: ")
    names = next(ln for ln in lines if ln.startswith("Surfaces (for place): ")).removeprefix(
        "Surfaces (for place): ").rstrip(".").split(", ")
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    tail = task[task.lower().rfind(" to the "):] if " to the " in task.lower() else ""
    hit = re.search(alt, tail, flags=re.IGNORECASE) or (list(re.finditer(alt, task, flags=re.IGNORECASE)) or [None])[-1]
    return next((n for n in names if hit and n.lower() == hit.group().lower()), None)


def info_of(user: str) -> dict:
    info = parse(user)
    info["colourless"] = not info["targets"]
    if info["colourless"] and (s := surface_of(user)):
        info["targets"] = [(None, s)]
    return info


def policy(info: dict) -> Gen:
    order, hall, spots = ["hall", *info["spots"][1:]], info["spots"][0], info["spots"][1:]
    state = {"here": hall, "blocked": False, "recovered": False}

    def act(name: str, args: dict, back: str | None) -> Gen:
        """E3b's step with the blocked / push / go-round rule (agent/e3.py); sets state['blocked'] when it fired."""
        r = yield (name, args)
        if r != "failed: blocked":
            return r
        state["blocked"] = True
        p = yield ("push", {"object": "crate"})
        if p.startswith("error: unknown object"):
            p = yield ("push", {"object": "cabinet"})
        if p == "pushed":
            return (yield (name, args))
        if p in ("failed: not movable", "failed: stuck"):
            for door in info["round_doors"]:
                if name == "navigate":
                    r = yield (name, {"place": args["place"], "via": door})
                else:
                    r = yield ("navigate", {"place": back, "via": door})
                    if r == "reached":
                        r = yield (name, args)
                if r != "failed: blocked":
                    return r
            return r
        return p

    def locate(cands: tuple[str, ...]) -> Gen:
        for box in cands:
            if (yield ("locate", {"object": box})).startswith("found"):
                return box
        return None

    def scan(todo: list[str], cands: tuple[str, ...]) -> Gen:
        for spot in todo:
            if (yield from act("navigate", {"place": spot}, None)) != "reached":
                continue
            state["here"] = spot
            yield ("look", {})
            if (box := (yield from locate(cands))) is not None:
                return box
        return None

    def pick(box: str) -> Gen:
        state["blocked"] = False
        r = yield from act("pick", {"object": box}, state["here"])
        if r.startswith("holding") or state["blocked"] or state["recovered"]:
            return r
        state["recovered"] = True  # rule (ii), once per run
        yield ("look", {})
        if (yield ("locate", {"object": box})).startswith("found") or (
                (yield from scan(order[order.index(state["here"]) + 1:], (box,))) is not None):
            return (yield from act("pick", {"object": box}, state["here"]))
        return r

    yield ("look", {})
    delivered: list[str] = []
    for i, (want, surface) in enumerate(info["targets"]):
        cands = BOXES if want is None else (want,)
        state["here"] = hall
        box = (yield from locate(cands)) if i == 0 else None
        if box is None:
            box = yield from scan(spots, cands)
        if box is None:
            break
        if not (yield from pick(box)).startswith("holding"):
            break
        if (yield from act("navigate", {"place": surface}, None)) != "reached":
            break
        if (yield from act("place", {"surface": surface}, surface)) != "placed":
            break
        state["here"] = surface
        delivered.append(box)
    yield ("done", {"note": f"script e3c: delivered {', '.join(delivered) or 'no box'}"})
    return "done"


class ScriptPolicy:
    """`script(packet, seed) -> Reply`, like a VLM; one instance per run. After done it returns zero calls."""

    def __init__(self, variant: str) -> None:
        assert variant == "e3c", variant
        self.variant, self.model = variant, f"script:{variant}"
        self.gen: Gen | None = None
        self.over = False

    def __call__(self, packet: Packet, seed: int) -> Reply:
        info = info_of(packet.user)
        try:
            if self.over:
                raise StopIteration
            if self.gen is None:
                self.gen = policy(info)
                name, args = next(self.gen)
            else:
                name, args = self.gen.send(info["last"])
        except StopIteration:
            self.over = True
            return Reply(content=None, reasoning=f"script {self.variant}: over", tool_calls=[], latency_s=0.0)
        return Reply(content=None, reasoning=f"script {self.variant}", latency_s=0.0,
                     tool_calls=[{"name": name, "arguments": json.dumps(args)}])
