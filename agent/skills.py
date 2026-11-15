"""The six skills + done, as the brief's table. Talks only to RobotAPI. Stdlib only. Never puts poses in a result string.

Decisions: push on a box returns "failed: not movable" without calling the robot (boxes are carried, not pushed);
push returns "failed: not movable" only when the object never moved, else "failed: stuck" (deviation 10), and
"failed: not blocking" when the object blocked no door corridor and no place pose at the call (nothing moved, row 17);
pick / place return "failed: blocked" when their own drive stopped on the crate / cabinet (row 16);
an `error:` result is a bad call (unknown name), a `failed:` result is a valid call that did not work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.robot_api import PX_VISIBLE, Photo, Pose, RobotAPI


@dataclass
class Result:
    text: str
    photos: list[Photo] = field(default_factory=list)
    ok: bool = True
    meta: dict = field(default_factory=dict)  # internal, logged (never shown to the VLM)


def _res(text: str, photos: list[Photo] | None = None, **meta) -> Result:
    return Result(text, photos or [], not text.startswith(("failed", "error")), meta)


def check_names(robot: RobotAPI, name: str, args: dict) -> str | None:
    """Argument VALUES against the world's names. Returns 'error: ...' or None."""
    places, objs = robot.places(), robot.objects()
    rooms = {p for p, v in places.items() if v["kind"] == "room"}
    doors = {p for p, v in places.items() if v["kind"] == "door"}
    surfaces = {p for p, v in places.items() if v["kind"] == "surface"}
    if name == "navigate":
        if args["place"] not in places:
            return f"error: unknown place {args['place']!r}"
        if args.get("via") is not None and args["via"] not in rooms | doors:
            return f"error: unknown place {args['via']!r}"
    elif name in ("locate", "pick", "push") and args["object"] not in objs:
        return f"error: unknown object {args['object']!r}"
    elif name == "place" and args["surface"] not in surfaces:
        return f"error: unknown surface {args['surface']!r}"
    return None


class Skills:
    def __init__(self, robot: RobotAPI) -> None:
        self.robot = robot
        self.photos: list[Photo] = []  # latest look / photo set
        self.seen: set[str] = set()  # object names ever >= PX_VISIBLE px
        self.located: dict[str, Pose] = {}  # runtime-only (the adapter's located pose: perceived in perceived mode)

    def _take(self, photos: list[Photo]) -> None:
        self.photos = photos
        for p in photos:
            self.seen |= {o for o, px in p.visible.items() if px >= PX_VISIBLE}

    def navigate(self, place: str, via: str | None = None) -> Result:
        if (e := check_names(self.robot, "navigate", {"place": place, "via": via})):
            return _res(e)
        avoid = sorted(o for o in self.seen if self.robot.objects()[o]["class"] == "box" and o != self.robot.held())
        r = self.robot.navigate(place, via, avoid)
        if r.photo is not None:
            self._take([r.photo])
        text = {"reached": "reached", "blocked": "failed: blocked", "no_path": "failed: no path"}[r.status]
        return _res(text, [r.photo] if r.photo else [], sim_s=r.sim_s, contacts=r.contacts, detail=r.detail)

    def look(self) -> Result:
        photos = self.robot.look()
        self._take(photos)
        return _res(f"{len(photos)} photos", photos)

    def locate(self, object: str) -> Result:  # noqa: A002 (the VLM-facing argument name)
        if (e := check_names(self.robot, "locate", {"object": object})):
            return _res(e)
        loc = self.robot.locate(object)  # the adapter's current photos = self.photos (latest look / navigate photo)
        if loc.pose is None:
            return _res("failed: not in view", **loc.meta)
        self.located[object] = loc.pose
        return _res(f"found: {object} in {loc.room}, on {loc.support}", **loc.meta)

    def pick(self, object: str) -> Result:  # noqa: A002
        if (e := check_names(self.robot, "pick", {"object": object})):
            return _res(e)
        if self.robot.objects()[object]["class"] != "box":
            return _res("failed: not a box")
        if object not in self.located:
            return _res("failed: not located")
        if self.robot.held() is not None:
            return _res("failed: already holding something")
        r = self.robot.pick(object)
        if r.ok:
            self.located.pop(object, None)
            return _res(f"holding {object}", detail=r.detail)
        return _res(f"failed: {r.reason}", detail=r.detail)

    def place(self, surface: str) -> Result:
        if (e := check_names(self.robot, "place", {"surface": surface})):
            return _res(e)
        if self.robot.held() is None:
            return _res("failed: nothing held")
        r = self.robot.place(surface)
        return _res("placed" if r.ok else f"failed: {r.reason}", detail=r.detail)

    def push(self, object: str) -> Result:  # noqa: A002
        if (e := check_names(self.robot, "push", {"object": object})):
            return _res(e)
        if self.robot.objects()[object]["class"] == "box":
            return _res("failed: not movable")
        r = self.robot.push(object)
        for o in r.detail.get("dropped_located", ()):  # the push moved things: locate again before pick (row 17 add. 3)
            self.located.pop(o, None)
        if r.ok:
            return _res("pushed", detail=r.detail)
        # "not movable" only when the object never moved; moved then stalled / not cleared -> "stuck" (005 row 10);
        # "not blocking": nothing to clear, nothing moved (row 17)
        return _res(f"failed: {r.reason}" if r.reason in ("not movable", "not blocking") else "failed: stuck",
                    detail=r.detail)

    def done(self, note: str) -> Result:
        return _res("done", note=note)
