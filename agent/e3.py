"""Task 010 (E3): fixed scripts in place of the VLM. Same interface as agent/vlm.py: `script(packet, seed) -> Reply`.
Stdlib only. A script reads ONLY `packet.user` (the text the VLM gets: task, places with kind and rooms, last result);
never photos, poses, maps, seeds or the sim (tests/test_e3.py checks the imports and the attribute reads).

E3a "search in order, always push" (the owner's wording):
- Room order: the room look spots in the packet's place order (= places.yaml), the hall skipped after the first look.
- Start: look, then locate target 1 (the robot starts in the hall). Not found -> for each look spot in order:
  navigate there, look, locate the target; found -> stop searching.
- Found -> pick it, navigate to its surface, place it. Then the next target (two-box), searching again from the first
  room of the order. Every target placed, or a target not found in any room -> done.
- Any step answering "failed: blocked" -> push the blocker, then retry that step once. The result does not name the
  blocker, so: push(crate); if that answers "error: unknown object 'crate'" (no crate in this house), push(cabinet).
- Push fails (anything but "pushed") -> the step fails: while searching, move on to the next room; while delivering
  (pick / navigate to the surface / place), give up: done.
- Any other failure: same (searching: next room; delivering: done).
E3b "search in order, push, else go round": E3a, plus: when push answers "failed: not movable" or "failed: stuck",
go round with via. Doors are told apart only by name: the go-round doors are the doors whose two rooms are shared
with another door ("kitchen north door", "kitchen south door"), tried in place order until one is not blocked. A
blocked navigate is retried as navigate(place, via=door); a blocked pick / place (no via in their schema) as
navigate(<the look spot where the box was found | the surface>, via=door), then the step again.
The robot's own step limit ends a run that does not reach done.
"""
from __future__ import annotations

import json
import re
from collections.abc import Generator

from agent.context import Packet
from agent.vlm import Reply

Call = tuple[str, dict]
Gen = Generator[Call, str, str]


def parse(user: str) -> dict:
    """Task targets, look spots, go-round doors and the last result from the packet text."""
    lines = user.splitlines()
    task = lines[0].removeprefix("Task: ")
    surfaces = next(ln for ln in lines if ln.startswith("Surfaces (for place): "))
    surfaces = surfaces.removeprefix("Surfaces (for place): ").rstrip(".").split(", ")
    targets = re.findall(r"(red|blue|green|yellow) box to the (" + "|".join(map(re.escape, surfaces)) + r")",
                         task, flags=re.IGNORECASE)
    places = [re.match(r"- (.+?): (room look spot|surface|door), (in|between) (.+)$", ln) for ln in lines]
    places = [m.groups() for m in places if m]
    spots = [p[0] for p in places if p[1] == "room look spot"]
    doors = [(p[0], tuple(sorted(p[3].split(" and ")))) for p in places if p[1] == "door"]
    round_doors = [d for d, pair in doors if sum(pair == q for _, q in doors) > 1]
    last = next(ln for ln in lines if ln.startswith("Last result: ")).removeprefix("Last result: ")
    return {"targets": [(f"{c.lower()} box", s) for c, s in targets], "spots": spots, "round_doors": round_doors,
            "last": None if last == "(none)" else last}


def policy(info: dict, variant: str) -> Gen:
    hall, spots = info["spots"][0], info["spots"][1:]

    def act(name: str, args: dict, back: str | None) -> Gen:
        """One step with the blocked rule; returns the step's final result. `back`: where a go-round navigate goes
        for a pick / place (None for navigate itself)."""
        r = yield (name, args)
        if r != "failed: blocked":
            return r
        p = yield ("push", {"object": "crate"})
        if p.startswith("error: unknown object"):
            p = yield ("push", {"object": "cabinet"})
        if p == "pushed":
            return (yield (name, args))
        if variant == "e3b" and p in ("failed: not movable", "failed: stuck"):
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

    yield ("look", {})
    delivered: list[str] = []  # the first target is first looked for in the hall
    here = hall
    for i, (box, surface) in enumerate(info["targets"]):
        found = (yield ("locate", {"object": box})).startswith("found") if i == 0 else False
        for spot in ([] if found else spots):
            if (yield from act("navigate", {"place": spot}, None)) != "reached":
                continue  # move on
            here = spot
            yield ("look", {})
            if (yield ("locate", {"object": box})).startswith("found"):
                found = True
                break
        if not found:
            break
        if not (yield from act("pick", {"object": box}, here)).startswith("holding"):
            break
        if (yield from act("navigate", {"place": surface}, None)) != "reached":
            break
        if (yield from act("place", {"surface": surface}, surface)) != "placed":
            break
        here = surface
        delivered.append(box)
    yield ("done", {"note": f"script {variant}: delivered {', '.join(delivered) or 'no box'}"})
    return "done"


class ScriptPolicy:
    """`script(packet, seed) -> Reply`, like a VLM; one instance per run. After done it returns zero calls."""

    def __init__(self, variant: str) -> None:
        assert variant in ("e3a", "e3b"), variant
        self.variant, self.model = variant, f"script:{variant}"
        self.gen: Gen | None = None
        self.over = False

    def __call__(self, packet: Packet, seed: int) -> Reply:
        info = parse(packet.user)
        try:
            if self.over:
                raise StopIteration
            if self.gen is None:
                self.gen = policy(info, self.variant)
                name, args = next(self.gen)
            else:
                name, args = self.gen.send(info["last"])
        except StopIteration:
            self.over = True
            return Reply(content=None, reasoning=f"script {self.variant}: over", tool_calls=[], latency_s=0.0)
        return Reply(content=None, reasoning=f"script {self.variant}", latency_s=0.0,
                     tool_calls=[{"name": name, "arguments": json.dumps(args)}])
