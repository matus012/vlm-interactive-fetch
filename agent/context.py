"""Context builder (task 004, brief §One run step 3): the packet the VLM sees each step. Stdlib only.

    build_packet(robot, skills, runtime, task) -> Packet(system, user, photos, labels)

`system` = the rules text (same every step and task); `user` = task, step counter, current room, places with their
kind, surfaces, the fixed object vocabulary, one-line history, last result and the photo labels; `photos` = the PNG
paths of the latest `look` set or `navigate` photo (Skills.photos), in order; `labels[i]` = "Photo <i+1>", the text
placed right before photo i in the user message. Names only: no poses, coordinates, map or robot identity. Text the
VLM itself wrote and the runtime echoes back (its call arguments, an error quoting them) has its digits masked as '#',
so a VLM that invents a number never makes the packet carry one (agent/leak.py scans every packet).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.robot_api import RobotAPI
from agent.runtime import Runtime
from agent.skills import Skills

OBJECTS = ("red box", "blue box", "green box", "yellow box", "crate", "cabinet")  # fixed, every task
KIND = {"room": "room look spot", "surface": "surface", "door": "door"}

SYSTEM = """You control a mobile robot with an arm and a camera on a mast, in a house. You see only the camera photos \
and the names of rooms, places, surfaces and objects. Each step you call exactly one tool.

Skills:
- navigate(place): drive to a named place. Returns "reached" or a failure, and one new photo. Optional via: a room or \
door name to route through. It can fail with "failed: blocked" when something blocks a door on the way.
- look(): turn in place and take one photo in each of four directions. Returns four new photos.
- locate(object): find the named object, but only if it is visible in the current photos. Returns the room and what \
it rests on, or "failed: not in view".
- pick(object): pick up a box. Needs that box located first. It drives to the box itself.
- place(surface): put the held box down on the named surface. Needs a held box. It drives to the surface itself by \
its own route; it cannot route round a blocked door.
- pick and place can fail with "failed: blocked" when something blocks a door on their way.
- push(object): try to clear a crate or cabinet that blocks a door, with the base. Returns "pushed", "failed: not \
movable" (it never moved), "failed: stuck" (it moved but could not be cleared) or "failed: not blocking" (it blocks \
nothing). Push only a crate or cabinet, never a box.
- done(note): end the run.

Rules:
- Call exactly one tool per step.
- Use only the names listed in the message. An unknown name or a bad call returns an error and still uses a step.
- The object list names every possible object; a house holds only some of them (a crate or a cabinet, not both).
- What they look like: each box is a small coloured block (red, blue, green or yellow); the crate is a purple cube \
about knee high; the cabinet is a tall grey cabinet. Name an object only after you have seen it.
- Some rooms have two look spots (for example "kitchen" and "kitchen east"). Each look spot is a place. Doors are \
places too.
- navigate reaches any place directly, so no stops on the way are needed.
- A look at the same spot again gives the same photos.
- locate only works on objects in the current photos.
- Carry one box at a time.
- When a way is blocked, you choose: try push, or route round with via. A door blocked by something push reports "failed: not \
movable" stays blocked; to go round it, name the other door in via. via must name a door or room on a different \
route (a door is listed with the two rooms it joins).
- If a door on the way to a surface was blocked and not cleared, navigate to that surface first (with via if needed), \
then place.
- Call done right after the last place that finishes the task; do not spend steps checking.
- Running out of steps without calling done counts as a failure."""


@dataclass
class Packet:
    system: str
    user: str
    photos: list[str | None] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def _mask(s: str) -> str:
    return re.sub(r"\d", "#", s)


def fmt_call(step: dict) -> str:
    """'<call>(<args>)' as the VLM wrote it (digits masked); an error step without a single call says so."""
    if not step["call"]:
        return "no single tool call"
    a = step["args"]
    if isinstance(a, str):
        return _mask(f"{step['call']}({a})")
    return _mask(f"{step['call']}({', '.join(f'{k}={v!r}' for k, v in a.items())})")


def fmt_result(step: dict) -> str:
    return _mask(step["result"]) if step["result"].startswith("error") else step["result"]


def build_packet(robot: RobotAPI, skills: Skills, runtime: Runtime, task: dict) -> Packet:
    places = robot.places()
    lines = []
    for n, v in places.items():
        where = f"between {' and '.join(v['between'])}" if "between" in v else f"in {v['room']}"
        lines.append(f"- {n}: {KIND[v['kind']]}, {where}")
    surfaces = [n for n, v in places.items() if v["kind"] == "surface"]
    hist = [f"{s['step']}. {fmt_call(s)} -> {fmt_result(s)}" for s in runtime.steps]
    labels = [f"Photo {i + 1}" for i in range(len(skills.photos))]
    if len(labels) > 1:
        photo_line = f"Current photos ({labels[0]} to {labels[-1]}, from your last look) follow this text."
    elif labels:
        photo_line = f"Current photo ({labels[0]}, from your last navigate) follows this text."
    else:
        photo_line = "No photos yet: navigate or look to get some."
    user = "\n".join([
        f"Task: {task['instruction']}",
        f"Step {runtime.steps_used + 1} of {runtime.limit}.",
        f"You are in: {robot.robot_room()}.",
        "Places (for navigate):", *lines,
        f"Surfaces (for place): {', '.join(surfaces)}.",
        f"Objects: {', '.join(OBJECTS)}.",
        "History:", *(hist or ["(none)"]),
        f"Last result: {fmt_result(runtime.steps[-1]) if runtime.steps else '(none)'}",
        photo_line,
    ])
    return Packet(SYSTEM, user, [p.path for p in skills.photos], labels)
