"""Pure-Python RobotAPI for tests (no MuJoCo). Rooms are rectangles joined by named doors; a blocker sits in a door.

Visibility: an object in the robot's room has `PX` px in the photo whose heading quadrant holds its bearing; `look`
gives all four quadrants. Objects in other rooms are never visible. Knobs for tests: `pick_fail`, `place_fail`.
"""
from __future__ import annotations

import math
from collections import deque

from agent.robot_api import PX_VISIBLE, ActResult, Located, NavResult, Photo, Pose

PX = 400
ROOMS = {"hall": (3.0, 4.5, 6.0, 6.0), "bedroom": (0.0, 4.5, 3.0, 6.0), "storage": (6.0, 4.5, 9.0, 6.0),
         "living room": (0.0, 0.0, 4.5, 3.5), "kitchen": (4.5, 0.0, 9.0, 3.5)}
PLACES = {  # name: (kind, room, x, y)
    "hall": ("room", "hall", 4.5, 5.25), "bedroom": ("room", "bedroom", 1.5, 5.25),
    "storage": ("room", "storage", 7.5, 5.25), "living room": ("room", "living room", 2.25, 1.75),
    "kitchen": ("room", "kitchen", 6.75, 1.75), "kitchen east": ("room", "kitchen", 8.5, 2.0),
    "kitchen table": ("surface", "kitchen", 7.55, 1.95), "storage bench": ("surface", "storage", 7.5, 5.95),
    "bedroom door": ("door", "bedroom", 3.0, 5.25), "storage door": ("door", "storage", 6.0, 5.25),
    "living room door": ("door", "living room", 3.6, 4.5), "kitchen north door": ("door", "kitchen", 4.5, 2.5),
    "kitchen south door": ("door", "kitchen", 4.5, 0.8)}
DOORS = {"bedroom door": ("bedroom", "hall"), "storage door": ("hall", "storage"), "living room door": ("hall", "living room"),
         "kitchen north door": ("living room", "kitchen"), "kitchen south door": ("living room", "kitchen")}
SURFACES = {"kitchen table": ((7.2, 1.5, 7.9, 2.4), 0.75), "storage bench": ((7.0, 5.7, 8.0, 6.0), 0.45)}


class FakeRobot:
    def __init__(self, boxes: dict[str, tuple[str, float, float, str]], blockers: dict[str, str] | None = None,
                 start: str = "hall") -> None:
        """boxes: name -> (room, x, y, support). blockers: 'crate'|'cabinet' -> door name it sits in."""
        self.pos = {n: [x, y, 0.06 if s == "floor" else 0.8, 0.0] for n, (_, x, y, s) in boxes.items()}
        self.room_ = {n: r for n, (r, *_rest) in boxes.items()}
        self.support = {n: s for n, (_, _x, _y, s) in boxes.items()}
        self.cls = {n: "box" for n in boxes}
        self.blockers = dict(blockers or {})
        for b, d in self.blockers.items():
            self.cls[b] = b
            self.room_[b] = DOORS[d][0]
            self.pos[b] = [*PLACES[d][2:], 0.2, 0.0]
            self.support[b] = "floor"
        self.pushed: set[str] = set()
        self.room = start
        self.xy = (PLACES[start][2], PLACES[start][3])
        self.held_: str | None = None
        self.pick_fail: set[str] = set()
        self.place_fail: set[str] = set()
        self.t = 0.0
        self.n_photos = 0
        self.current: list[Photo] = []  # the latest look / navigate photo set (what locate sees)

    # --- world facts ---
    def places(self) -> dict[str, dict]:
        return {n: {"kind": k, "room": r} for n, (k, r, *_x) in PLACES.items()}

    def objects(self) -> dict[str, dict]:
        return {n: {"class": c, "colour": n.split()[0] if c == "box" else None} for n, c in self.cls.items()}

    # --- photos ---
    def _photo(self, heading: float, index: int = 1) -> Photo:
        self.n_photos += 1
        vis = {}
        for n in self.pos:
            if n == self.held_ or self.room_[n] != self.room:
                continue
            bearing = math.degrees(math.atan2(self.pos[n][1] - self.xy[1], self.pos[n][0] - self.xy[0])) % 360
            if abs((bearing - heading + 180) % 360 - 180) <= 45:
                vis[n] = PX
        return Photo(index, heading, None, vis)

    def look(self) -> list[Photo]:
        self.t += 8.0
        self.current = [self._photo(h, i + 1) for i, h in enumerate((0.0, 90.0, 180.0, 270.0))]
        return self.current

    def photo(self) -> Photo:
        self.current = [self._photo(0.0)]
        return self.current[0]

    def locate(self, obj: str) -> Located:
        px = max((p.visible.get(obj, 0) for p in self.current), default=0)
        if px < PX_VISIBLE:
            return Located(None, meta={"max_px": px})
        return Located(Pose(*self.pos[obj]), self.room_[obj], self.support_of(obj), {"max_px": px})

    # --- navigation ---
    def _route(self, a: str, b: str, avoid_door: str | None = None) -> list[str] | None:
        """Doors crossed going from room a to room b (BFS on the known map)."""
        q, seen = deque([(a, [])]), {a}
        while q:
            r, path = q.popleft()
            if r == b:
                return path
            for d, (u, v) in DOORS.items():
                if r in (u, v) and d != avoid_door:
                    nxt = v if r == u else u
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, path + [d]))
        return None

    def navigate(self, place: str, via: str | None, avoid: list[str]) -> NavResult:
        goal = PLACES[place][1]
        doors: list[str] | None
        if via is None:
            doors = self._route(self.room, goal)
        elif via in DOORS:
            u, v = DOORS[via]
            cands = []
            for a, b in ((u, v), (v, u)):
                r1, r2 = self._route(self.room, a), self._route(b, goal)
                if r1 is not None and r2 is not None:
                    cands.append(r1 + [via] + r2)
            doors = min(cands, key=len) if cands else None
        else:
            r1, r2 = self._route(self.room, PLACES[via][1]), self._route(PLACES[via][1], goal)
            doors = None if r1 is None or r2 is None else r1 + r2
        if doors is None:
            return NavResult("no_path", None, 0.0, None)
        cur = self.room
        for d in doors:
            if any(b in self.blockers and self.blockers[b] == d and b not in self.pushed for b in self.blockers):
                self.room = cur
                self.t += 5.0
                return NavResult("blocked", self.photo(), 5.0, None, detail=f"fake: blocked at {d}")
            cur = DOORS[d][1] if cur == DOORS[d][0] else DOORS[d][0]
        self.room, self.xy = goal, (PLACES[place][2], PLACES[place][3])
        self.t += 10.0
        return NavResult("reached", self.photo(), 10.0, 5.0)

    # --- manipulation ---
    def pick(self, obj: str) -> ActResult:
        if obj in self.pick_fail or self.room_[obj] != self.room:
            return ActResult(False, "missed")
        self.held_, self.support[obj], self.pos[obj][2] = obj, "", 0.5
        return ActResult(True)

    def place(self, surface: str) -> ActResult:
        if surface in self.place_fail or PLACES[surface][1] != self.room:
            return ActResult(False, "too far" if PLACES[surface][1] != self.room else "slipped")
        (x0, y0, x1, y1), top = SURFACES[surface]
        o = self.held_
        self.pos[o] = [(x0 + x1) / 2, (y0 + y1) / 2, top + 0.06, 0.0]
        self.support[o], self.room_[o], self.held_ = surface, self.room, None
        return ActResult(True)

    def push(self, obj: str) -> ActResult:
        if self.cls[obj] != "crate" or obj not in self.blockers:
            return ActResult(False, "not movable")
        self.pushed.add(obj)
        return ActResult(True)

    def held(self) -> str | None:
        return self.held_

    def robot_room(self) -> str:
        return self.room

    # --- truth ---
    def true_pose(self, obj: str) -> Pose:
        return Pose(*self.pos[obj])

    def room_of(self, obj: str) -> str:
        return self.room_[obj]

    def support_of(self, obj: str) -> str:
        return self.support[obj] or "floor"

    def sim_ok(self) -> bool:
        return True

    def sim_time(self) -> float:
        return self.t

    def snapshot(self, settle_s: float = 2.0) -> dict:
        boxes = {n: {"pos": self.pos[n][:3], "tilt_deg": 0.0, "max_disp_settle_m": 0.0, "max_speed_settle": 0.0,
                     "support": self.support[n] or None} for n, c in self.cls.items() if c == "box"}
        return {"boxes": boxes, "surfaces": {n: {"rect": list(r), "top_z": z} for n, (r, z) in SURFACES.items()},
                "held": self.held_, "gripper_open": self.held_ is None, "arm_retracted": True, "sim_ok": True,
                "settle_s": settle_s}
