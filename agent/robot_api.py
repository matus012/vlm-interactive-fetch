"""Phase 2 contract (task 003): what the skills / runtime / scorer may ask of the robot adapter. Stdlib only.

The adapter (`sim/adapter.py`, the only module the agent side talks to; it imports MuJoCo) implements `RobotAPI`.
Skills (`agent/skills.py`), the runtime (`agent/runtime.py`) and the scorer (`agent/scorer.py`) import no MuJoCo
(gate G2-arch). Poses stay on this side of the VLM: skills never put coordinates into a result string.

Names (exactly what the VLM will see):
- objects: "red box", "blue box", "green box", "yellow box", "crate", "cabinet" (only those in the world)
- places: `sim/world/places.yaml` names; kind room | surface | door. Doors (R3): "bedroom door", "storage door",
  "living room door", "kitchen north door" (d_lk1), "kitchen south door" (d_lk2).
- surfaces (place targets): "kitchen table", "storage bench"; supports a box may rest on: "floor", those two,
  "coffee table", "nightstand".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

PX_VISIBLE = 100  # an object counts as seen / in view with >= 100 px in a photo (= G1-cover bar): segmentation px in
# truth mode; colour-mask px for fetch boxes in perceived mode (task 006; sim.perception.IN_VIEW_PX)


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float  # rad


@dataclass
class Photo:
    index: int  # 1..4 within a look; 1 for a single photo
    heading_deg: float  # absolute base heading the photo was taken at
    path: str | None  # PNG written by the adapter (None when photos are not saved)
    visible: dict[str, int]  # object name -> px (only > 0): segmentation px; perceived mode: mask px for fetch boxes


@dataclass
class Located:
    """`locate` in the current photos (the latest look, or the photo navigate returned). pose None = not in view."""
    pose: Pose | None
    room: str = ""  # room name (places.yaml) of the pose
    support: str = ""  # "floor", a surface name, or "none"
    meta: dict = field(default_factory=dict)  # internal log (px, perceived pose; eval-only truth), never to the VLM


@dataclass
class NavResult:
    status: Literal["reached", "blocked", "no_path"]
    photo: Photo | None  # one photo at the final heading
    sim_s: float
    path_len_m: float | None
    contacts: dict[str, float] = field(default_factory=dict)  # "objname|geom" -> max penetration m (unseen boxes etc.)
    detail: str = ""  # internal (why blocked: tracking error / no progress), never shown to the VLM


@dataclass
class ActResult:
    ok: bool
    reason: str = ""  # short, VLM-safe: "not located", "missed", "slipped", "not movable", "nothing held", ...
    detail: dict = field(default_factory=dict)  # internal numbers (forces, lift height, in-hand pose, ...)


class RobotAPI(Protocol):
    # --- world facts (names only, no poses) ---
    def places(self) -> dict[str, dict]: ...  # name -> {"kind": room|surface|door, "room": str}
    def objects(self) -> dict[str, dict]: ...  # name -> {"class": box|crate|cabinet, "colour": str|None}

    # --- actions (all physics inside the adapter) ---
    def navigate(self, place: str, via: str | None, avoid: list[str]) -> NavResult: ...
    def look(self) -> list[Photo]: ...  # look arm pose, headings 0/90/180/270
    def photo(self) -> Photo: ...  # one photo at the current heading
    def locate(self, obj: str) -> Located: ...  # the object in the current photos (Phase 5: perceived for boxes)
    def pick(self, obj: str) -> ActResult: ...  # adapter plans its own pre-grasp base pose from the located pose
    def place(self, surface: str) -> ActResult: ...  # approach point, put down upright, open, retract
    def push(self, obj: str) -> ActResult: ...  # with the base; ok=False, reason "not movable" (never moved) | "stuck" | "not blocking"
    def held(self) -> str | None: ...
    def robot_room(self) -> str: ...  # room name (places.yaml) the base centre is in; outside every room: the nearest

    # --- truth (truth-mode locate, evaluation; never shown to the VLM) ---
    def true_pose(self, obj: str) -> Pose: ...
    def room_of(self, obj: str) -> str: ...  # room name as in places.yaml rooms ("living room", ...)
    def support_of(self, obj: str) -> str: ...  # "floor" or a surface / spawn-surface name
    def sim_ok(self) -> bool: ...  # no NaN / blow-up so far

    # --- scorer input ---
    def snapshot(self, settle_s: float = 2.0) -> dict: ...  # steps settle_s of sim time; schema below


# snapshot() schema (scorer input; plain JSON-able):
# {
#   "boxes":    {"red box": {"pos": [x, y, z], "tilt_deg": float,   # angle of the box's 12 cm axis from world z
#                            "max_disp_settle_m": float,             # max displacement during the settle window
#                            "max_speed_settle": float,              # max linear speed (m/s) during the settle window
#                            "support": "floor" | surface name | None}},
#   "surfaces": {"kitchen table": {"rect": [x0, y0, x1, y1], "top_z": float}, "storage bench": {...}},
#   "held": "red box" | None,     # object between the fingers with both pads in contact at the end
#   "gripper_open": bool, "arm_retracted": bool,
#   "sim_ok": bool, "settle_s": float,
# }
