"""The seeded house: layout.yaml + seed + blocker variant -> world MJCF text (no robot) + truth labels.

    python -m sim.world.generate --seed N --blocker none|crate|cabinet --out DIR   # DIR/house.xml + DIR/labels.json

Structure (static, one body each, label class in labels.json): walls (merged room edges with 1.0 m door gaps), one
coloured visual floor patch per room over a shared plane floor, furniture by kind (table = top slab + 4 legs, open
underneath; solid = one block; sofa = seat + back), point lights from the YAML + a soft headlight, an `overview`
camera above the house.
Fetch boxes (all 4 colours, free joints, matte), by seed: room uniform among rooms with a spawn region; if the room
has a spawn surface, the box goes on it with P = SURFACE_P (surface picked uniformly), else on the floor; floor spot
by rejection sampling in the room's region with `spot_ok` and `route_clear` (deviation 9: off every planned route
corridor between two places; also checked on surfaces); surface spot uniform on the top inset by BOX_REACH (the
box footprint stays on the top at any yaw; the floor rules do not apply on a surface) by rejection with
`surface_spot_ok` (reachable from standable floor in front of an edge); no two boxes closer than
MIN_BOX_SEP centre to centre; yaw uniform [0, pi). Every box starts SETTLE_DROP above its support and drops onto it.
Blocker (own RNG stream, so the same seed gives the same boxes in every variant): crate (free joint) or cabinet (no
joint = welded) centred in a seeded door from the YAML lists, cabinet long side across the doorway.
Determinism: every number is written with `fs` (fixed 6 significant digits; quaternions 9 decimals), lists keep YAML
order, RNG = numpy PCG64 seeded with [seed, stream]. Same YAML + seed + blocker -> byte-identical XML.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.constants import (BOX_COLOURS, BOX_HALF, BOX_MASS, CABINET_HALF, CABINET_RGBA, CRATE_HALF, CRATE_MASS,
                           CRATE_RGBA, ROBOT_RADIUS, ROUTE_MARGIN, refuse_test_seed)

# E2 (task 011): NAV116_HOUSE=B selects house B's world files, read once at import (every default below follows);
# unset / A = house A, the freeze-006 files. Set it in the environment before Python starts, never in code.
HOUSE = os.environ.get("NAV116_HOUSE", "A")
assert HOUSE in ("A", "B"), f"NAV116_HOUSE={HOUSE!r}: A or B"
LAYOUT = Path(__file__).with_name("layout.yaml" if HOUSE == "A" else "layout_b.yaml")
PLACES = Path(__file__).with_name("places.yaml" if HOUSE == "A" else "places_b.yaml")
SURFACE_P = 0.25  # P(box on a spawn surface | its room has one)
MIN_BOX_SEP = 0.40
BOX_REACH = math.hypot(BOX_HALF[0], BOX_HALF[1])  # box footprint radius in xy: inset of a surface top
SETTLE_DROP = 0.0005  # every free body starts 0.5 mm above its support
LIGHT_Z = 1.9
PATCH_Z = 0.001  # floor patch: visual-only slab, half height 1 mm, top at 2 mm above the plane
TABLE_TOP_T = 0.04  # table / bench top slab thickness
LEG_HALF = 0.025
WALL_MIN, FURN_MIN, ROOM_PT_MIN = 0.35, 0.30, 0.80  # spawn rules (layout.yaml header)
# Task 003 fix round 2 (2j): a floor box >= SURF_MIN from every spawn-surface footprint (a floor box must not block the
# only standing spot in front of a surface, seed 1016); a surface box within SURF_EDGE_MAX (box centre to edge) of a
# top edge with standable floor in front of it: some base centre on the line STAND_OFF outside that edge, within the
# edge's span, is >= STAND_OFF from every wall and every other furniture footprint, and the box centre is within
# STAND_OFF + SURF_EDGE_MAX of it. SURF_EDGE_MAX: 0.30 m (L1) tightened to 0.24 m by measurement
# (runs/gate2r2/surface_*.json: nightstand south edge, pick_box ok at 0.245 m, "no pre-grasp pose" from 0.255 m).
SURF_MIN = 0.60
SURF_EDGE_MAX = 0.24
STAND_OFF = ROBOT_RADIUS + 0.02  # robot footprint radius + follow.CLEARANCE (the planner's true clearance)
# Doorway corridor: door width + DOOR_PAD across, DOOR_DEPTH each side of the wall line along the passage. Phase 1
# rule: 0.8 m deep. R4a (task 003): each door +- 0.6 m along the passage takes no box; the larger depth is kept, so
# the 0.8 m corridor already contains the R4a one (same width, deeper) and R4a removes no further spot.
DOOR_PAD = 0.3
R4A_DOOR_DEPTH = 0.6
DOOR_DEPTH = max(0.8, R4A_DOOR_DEPTH)
# E1 (task 009, context row 22): EASINGS False lifts rows 8 and 9 for the E1 extension only (no route-corridor rule;
# floor boxes >= FURN_MIN from spawn surfaces as before fix round 2; surface boxes anywhere on the inset top).
# Process-wide, set once by set_easings before the first world is drawn (sim.tasks caches per (seed, blocker)).
# Default True = freeze-006, byte for byte.
EASINGS = True
_DRAWN: set[bool] = set()  # the EASINGS values worlds were drawn with in this process
KEEPOUT_KINDS = ("room", "surface")  # places whose pose gets the ROOM_PT_MIN keep-out (R4a: + approach points);
# kind door is handled by the corridor rule
# E4 C1 (task 014): LABELS True prints a use word on every fetch box. Process-wide, set once by set_labels before the
# first world is drawn. Default False = no label: XML and world byte-identical to before the switch existed.
# Per seed: rng default_rng([seed, 7]) permutes LABEL_WORDS' 4 categories over the boxes (BOX_COLOURS order), then
# draws one word per box (same rng, same order). The world's box dicts get "label": {"category", "word"}.
# Texture: one cube texture per box (MuJoCo maps one square image onto each box face, stretched to the face),
# multiplied by the geom rgba, so white = box colour and the black word stays black. Faces: +-x (20 x 12 cm, the big
# sides) and +z (top, 6 x 20 cm) carry the word, upright with the box upright and read left to right from outside;
# on the top it runs along body +y with letter tops towards body -x. +-y and -z are plain. label_band gives the
# word's ink band per face for measurements. PNGs are written at draw time to LABEL_DIR/<WORD>.png (content depends
# on the word only; atomic replace, so concurrent processes are safe) and referenced by absolute path.
LABELS = False
_LABELS_DRAWN: set[bool] = set()  # the LABELS values worlds were drawn with in this process
LABEL_WORDS: dict[str, tuple[str, ...]] = {"drink": ("JUICE", "MILK", "WATER"), "wipe": ("SPONGE", "CLOTH"),
                                           "read": ("BOOK", "NEWS"), "wash": ("SOAP",)}
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_DIR = Path(tempfile.gettempdir()) / "nav116_labels"
LABEL_TEX = 256  # px per cube face of the texture
LABEL_RES = 10  # px per mm of the layout canvas (word placement and ink band are measured there)
LABEL_MARGIN = 0.08  # free border on every side of a labelled face, as a fraction of the face's shorter side


def set_easings(on: bool) -> None:
    """E1 switch (row 22). Refuses a change once a world was drawn with the other value."""
    global EASINGS
    if _DRAWN - {on}:
        raise RuntimeError(f"set_easings({on}): worlds already drawn with easings={EASINGS} in this process")
    EASINGS = on


def set_labels(on: bool) -> None:
    """E4 C1 switch. Refuses a change once a world was drawn with the other value."""
    global LABELS
    if _LABELS_DRAWN - {on}:
        raise RuntimeError(f"set_labels({on}): worlds already drawn with labels={LABELS} in this process")
    LABELS = on


def _draw_labels(seed: int, boxes: list[dict[str, Any]]) -> None:
    rng = np.random.default_rng([seed, 7])
    cats = list(LABEL_WORDS)
    perm = rng.permutation(len(cats))
    for b, k in zip(boxes, perm):
        words = LABEL_WORDS[cats[int(k)]]
        b["label"] = {"category": cats[int(k)], "word": words[int(rng.integers(len(words)))]}


def _face_mm(face: str) -> tuple[int, int]:
    """(width along the reading direction, height) of a labelled face in mm: side = +-x face, top = +z face."""
    return ((round(2000 * BOX_HALF[1]), round(2000 * BOX_HALF[2])) if face == "side"
            else (round(2000 * BOX_HALF[1]), round(2000 * BOX_HALF[0])))


@functools.cache
def _label_canvas(word: str, face: str) -> tuple[Any, tuple[float, float, float, float]]:
    """The face's layout canvas (PIL L image, white, black word, as large as fits inside the margin, centred) and the
    word's ink box as fractions (a0, a1, b0, b1): a along the reading direction, b from the letter tops down."""
    from PIL import Image, ImageDraw, ImageFont

    wmm, hmm = _face_mm(face)
    W, H = wmm * LABEL_RES, hmm * LABEL_RES
    pad = LABEL_MARGIN * min(W, H)
    size = 1
    while True:
        l, t, r, b = ImageFont.truetype(LABEL_FONT, size + 1).getbbox(word)
        if r - l > W - 2 * pad or b - t > H - 2 * pad:
            break
        size += 1
    font = ImageFont.truetype(LABEL_FONT, size)
    l, t, r, b = font.getbbox(word)
    im = Image.new("L", (W, H), 255)
    ImageDraw.Draw(im).text(((W - (r - l)) / 2 - l, (H - (b - t)) / 2 - t), word, font=font, fill=0)
    ink = np.asarray(im) < 128
    cols, rows = np.nonzero(ink.any(0))[0], np.nonzero(ink.any(1))[0]
    return im, (cols[0] / W, (cols[-1] + 1) / W, rows[0] / H, (rows[-1] + 1) / H)


def label_band(word: str, face: str) -> tuple[float, float, float, float]:
    """Ink box of `word` on a labelled face ("side" = the +-x faces, "top" = +z) as fractions (a0, a1, b0, b1):
    a along the reading direction (side: body +y on +x, body -y on -x; top: body +y), b from the letter tops down
    (side: body -z from the top edge; top: body +x from the -x edge)."""
    return _label_canvas(word, face)[1]


def label_png(word: str) -> Path:
    """Write LABEL_DIR/<word>.png: the cube texture strip (6 rows of LABEL_TEX px squares, MuJoCo order +x -x +y -y
    +z -z). Square = the face canvas resized to the square, then turned so MuJoCo's per-face mapping (measured by
    rendering) shows it upright: +x and +z rotate 90 deg ccw, -x 90 deg cw."""
    from PIL import Image

    n = LABEL_TEX
    side = _label_canvas(word, "side")[0].resize((n, n), Image.LANCZOS)
    top = _label_canvas(word, "top")[0].resize((n, n), Image.LANCZOS)
    blank = Image.new("L", (n, n), 255)
    faces = [side.transpose(Image.ROTATE_90), side.transpose(Image.ROTATE_270), blank, blank,
             top.transpose(Image.ROTATE_90), blank]
    strip = Image.new("L", (n, 6 * n), 255)
    for i, f in enumerate(faces):
        strip.paste(f, (0, i * n))
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    path = LABEL_DIR / f"{word}.png"
    tmp = path.with_name(f".{word}.{os.getpid()}.png")
    strip.convert("RGB").save(tmp)
    os.replace(tmp, path)
    return path


def fs(*v: float) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def r6(v: float) -> float:
    """The value as it is written into the XML."""
    return float(f"{float(v):.6g}")


def quat_s(yaw: float) -> str:
    return " ".join(f"{v:.9f}" for v in (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def load_layout(path: Path | str = LAYOUT) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_places(path: Path | str = PLACES) -> list[dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text())["places"]


def room_points(room: str, places: list[dict[str, Any]] | None = None) -> list[tuple[float, float]]:
    """Every room point of a room (kind room), in places.yaml order. The kitchen has two (G1-cover fallback, 002)."""
    pts = [(float(p["pose"][0]), float(p["pose"][1])) for p in places or load_places()
           if p["kind"] == "room" and p["room"] == room]
    if not pts:
        raise KeyError(room)
    return pts


def room_point(room: str, places: list[dict[str, Any]] | None = None) -> tuple[float, float]:
    """The room's first (main) room point."""
    for p in places or load_places():
        if p["kind"] == "room" and p["room"] == room:
            return float(p["pose"][0]), float(p["pose"][1])
    raise KeyError(room)


# ---- geometry from the layout ---------------------------------------------------------------------------------

def wall_segments(layout: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    """Wall pieces (axis, line coord, a, b): axis 'x' = wall on x = line from y=a to y=b; 'y' likewise. Room edges
    merged per line, door gaps cut out, ends extended by half a thickness so corners close."""
    lines: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for r in layout["rooms"].values():
        x0, y0, x1, y1 = (float(v) for v in r["rect"])
        for key, iv in ((("x", x0), (y0, y1)), (("x", x1), (y0, y1)), (("y", y0), (x0, x1)), (("y", y1), (x0, x1))):
            lines.setdefault(key, []).append(iv)
    hw, w = layout["wall"]["thickness"] / 2.0, layout["door_width"] / 2.0
    gaps: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for d in layout["doors"].values():
        (axis, c), = d["wall"].items()
        gaps.setdefault((axis, float(c)), []).append((d["center"] - w, d["center"] + w))
    out = []
    for key in sorted(lines):
        ivs = sorted(lines[key])
        merged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            pieces = [(a - hw, b + hw)]
            for ga, gb in sorted(gaps.get(key, [])):
                nxt = []
                for pa, pb in pieces:
                    if gb <= pa or ga >= pb:
                        nxt.append((pa, pb))
                        continue
                    if ga > pa:
                        nxt.append((pa, ga))
                    if gb < pb:
                        nxt.append((gb, pb))
                pieces = nxt
            out += [(key[0], key[1], pa, pb) for pa, pb in pieces if pb - pa > 1e-6]
    return out


def wall_rects(layout: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Wall pieces as xy rectangles [x0, y0, x1, y1]."""
    hw = layout["wall"]["thickness"] / 2.0
    return [(c - hw, a, c + hw, b) if axis == "x" else (a, c - hw, b, c + hw)
            for axis, c, a, b in wall_segments(layout)]


def furniture_rect(f: dict[str, Any]) -> tuple[float, float, float, float]:
    cx, cy, sx, sy, _ = f["box"]
    return cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2


def door_corridor(layout: dict[str, Any], d: dict[str, Any]) -> tuple[float, float, float, float]:
    (axis, c), = d["wall"].items()
    half = (layout["door_width"] + DOOR_PAD) / 2.0
    if axis == "x":
        return c - DOOR_DEPTH, d["center"] - half, c + DOOR_DEPTH, d["center"] + half
    return d["center"] - half, c - DOOR_DEPTH, d["center"] + half, c + DOOR_DEPTH


def rect_dist(x: float, y: float, r: tuple[float, float, float, float]) -> float:
    dx = max(r[0] - x, 0.0, x - r[2])
    dy = max(r[1] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)


def spot_ok(layout: dict[str, Any], room: str, x: float, y: float,
            places: list[dict[str, Any]] | None = None) -> bool:
    """Floor spawn rules (layout.yaml header): inside the room's spawn region, >= 0.35 m from the room's wall faces,
    >= 0.30 m from every furniture footprint, outside every door corridor (door width + 0.3 m, 0.8 m deep each side
    of the wall line), >= 0.60 m from every spawn-surface footprint, >= 0.80 m from every room point and every surface approach point of the room (R4a)."""
    reg = layout["spawn_regions"].get(room)
    if reg is None or not (reg[0] <= x <= reg[2] and reg[1] <= y <= reg[3]):
        return False
    x0, y0, x1, y1 = layout["rooms"][room]["rect"]
    if min(x - x0, x1 - x, y - y0, y1 - y) - layout["wall"]["thickness"] / 2.0 < WALL_MIN:
        return False
    surf_min = SURF_MIN if EASINGS else FURN_MIN  # row 8 (E1: lifted)
    if any(rect_dist(x, y, furniture_rect(f)) < (surf_min if f.get("surface") in layout["spawn_surfaces"] else FURN_MIN)
           for f in layout["furniture"]):
        return False
    if any(rect_dist(x, y, door_corridor(layout, d)) == 0.0 for d in layout["doors"].values()):
        return False
    return all(math.hypot(x - rx, y - ry) >= ROOM_PT_MIN for rx, ry in keepout_points(room, places))


@functools.cache
def _route_segments(layout_json: str) -> tuple[np.ndarray, np.ndarray]:
    """Segment starts / ends of every planned route of the layout (route_segments)."""
    from sim.nav.astar import plan_path  # lazy: occupancy and follow import this module
    from sim.nav.follow import CLEARANCE, DISC_MARGIN
    from sim.world.occupancy import known_map

    layout = json.loads(layout_json)
    g = known_map(layout)
    ends = list(dict.fromkeys([(float(p["pose"][0]), float(p["pose"][1])) for p in load_places()]
                              + [(float(layout["spawn"][0]), float(layout["spawn"][1]))]))
    a: list[tuple[float, float]] = []
    b: list[tuple[float, float]] = []
    for s, t in itertools.permutations(ends, 2):
        path = plan_path(g.occ, g.origin, g.res, s, t, ROBOT_RADIUS, CLEARANCE + DISC_MARGIN)
        if path is None:
            raise RuntimeError(f"no route {s} -> {t} on the known map")
        a += path[:-1]
        b += path[1:]
    return np.array(a), np.array(b)


def route_segments(layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Deviation 9: the routes navigate drives on the static known map (walls + furniture; no boxes, no blocker),
    planned exactly as follow.go_to plans them (plan_path, ROBOT_RADIUS, CLEARANCE + DISC_MARGIN, no via), between
    every ordered pair of distinct endpoints = every places.yaml pose (room points incl. kitchen east, surface approach
    points, doors) + the hall spawn (layout.spawn; the gates start there). Both directions of a pair are planned (A*
    ties can make them differ). Per layout, not per seed: cached (~3 s once per process)."""
    return _route_segments(json.dumps(layout, sort_keys=True))


def route_clear(segs: tuple[np.ndarray, np.ndarray], x: float, y: float) -> bool:
    """Deviation 9: the box footprint stays > ROBOT_RADIUS + ROUTE_MARGIN from every route polyline. Footprint =
    the circumscribed disc of radius BOX_REACH (yaw-free, so the check runs before the yaw draw and spawn_spots / the
    G1-cover grid use the same rule): box centre > ROBOT_RADIUS + ROUTE_MARGIN + BOX_REACH from every segment."""
    if not EASINGS:  # row 9 (E1: lifted)
        return True
    return route_dist(segs, x, y) > ROBOT_RADIUS + ROUTE_MARGIN + BOX_REACH


def route_dist(segs: tuple[np.ndarray, np.ndarray], x: float, y: float) -> float:
    """Distance from (x, y) to the nearest route segment."""
    a, b = segs
    ab = b - a
    p = np.array([x, y])
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(a + t[:, None] * ab - p).T).min())


def keepout_points(room: str, places: list[dict[str, Any]] | None = None) -> list[tuple[float, float]]:
    """Poses of the room's places of kind room or surface (the robot stands there); doors excluded."""
    return [(float(p["pose"][0]), float(p["pose"][1])) for p in places or load_places()
            if p["kind"] in KEEPOUT_KINDS and p["room"] == room]


def surfaces_in(layout: dict[str, Any], room: str) -> list[dict[str, Any]]:
    return [f for f in layout["furniture"] if f["room"] == room and f.get("surface") in layout["spawn_surfaces"]]


def surface_top_rect(f: dict[str, Any]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = furniture_rect(f)
    return x0 + BOX_REACH, y0 + BOX_REACH, x1 - BOX_REACH, y1 - BOX_REACH


def standable(layout: dict[str, Any], x: float, y: float, skip: dict[str, Any] | None = None) -> bool:
    """A base centre at (x, y) is >= STAND_OFF from every wall piece and every furniture footprint but `skip`."""
    rects = wall_rects(layout) + [furniture_rect(g) for g in layout["furniture"] if g is not skip]
    return all(rect_dist(x, y, r) >= STAND_OFF for r in rects)


def surface_spot_ok(layout: dict[str, Any], f: dict[str, Any], x: float, y: float) -> bool:
    """Surface spawn rule (fix round 2): inside the inset top, and a standing point (1 cm samples along the line
    STAND_OFF outside one of the four top edges, within that edge's span) is standable and within
    STAND_OFF + SURF_EDGE_MAX of the box centre."""
    ix0, iy0, ix1, iy1 = surface_top_rect(f)
    if not (ix0 - 1e-9 <= x <= ix1 + 1e-9 and iy0 - 1e-9 <= y <= iy1 + 1e-9):
        return False
    if not EASINGS:  # row 8 (E1: lifted): any spot on the top the box fits on
        return True
    return edge_reachable(layout, f, x, y)


def edge_reachable(layout: dict[str, Any], f: dict[str, Any], x: float, y: float) -> bool:
    """Row 8's edge rule alone: a standable point STAND_OFF outside a top edge within STAND_OFF + SURF_EDGE_MAX."""
    x0, y0, x1, y1 = furniture_rect(f)
    xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / 0.01)) + 1))
    ys = np.linspace(y0, y1, max(2, int(round((y1 - y0) / 0.01)) + 1))
    stands = ([(float(u), y0 - STAND_OFF) for u in xs] + [(float(u), y1 + STAND_OFF) for u in xs]
              + [(x0 - STAND_OFF, float(v)) for v in ys] + [(x1 + STAND_OFF, float(v)) for v in ys])
    near = [(sx, sy) for sx, sy in stands if math.hypot(x - sx, y - sy) <= STAND_OFF + SURF_EDGE_MAX]
    return any(standable(layout, sx, sy, f) for sx, sy in near)


def spawn_spots(layout: dict[str, Any], room: str, step: float,
                places: list[dict[str, Any]] | None = None) -> list[tuple[float, float, float, str]]:
    """Every box centre the generator can draw in a room, on a `step` grid: (x, y, z, surface). Floor: grid over the
    spawn region (anchored at its corner) filtered by spot_ok and route_clear; each spawn surface: grid over its inset
    top filtered by surface_spot_ok and route_clear. z = box centre at rest (no drop offset)."""
    places = places or load_places()
    out: list[tuple[float, float, float, str]] = []
    reg = layout["spawn_regions"].get(room)
    if reg is None:
        return out

    def grid(a: float, b: float) -> list[float]:
        n = int(math.floor((b - a) / step + 1e-9))
        return [a + i * step for i in range(n + 1)]

    segs = route_segments(layout)
    for y in grid(reg[1], reg[3]):
        for x in grid(reg[0], reg[2]):
            if spot_ok(layout, room, x, y, places) and route_clear(segs, x, y):
                out.append((r6(x), r6(y), r6(BOX_HALF[2]), "floor"))
    for f in surfaces_in(layout, room):
        x0, y0, x1, y1 = surface_top_rect(f)
        xs = grid(x0, x1) or [(x0 + x1) / 2]
        ys = grid(y0, y1) or [(y0 + y1) / 2]
        # centre the grid on the top
        ox, oy = ((x1 - x0) - (xs[-1] - xs[0])) / 2, ((y1 - y0) - (ys[-1] - ys[0])) / 2
        for y in ys:
            for x in xs:
                if not (surface_spot_ok(layout, f, x + ox, y + oy) and route_clear(segs, x + ox, y + oy)):
                    continue
                out.append((r6(x + ox), r6(y + oy), r6(f["box"][4] + BOX_HALF[2]), f["surface"]))
    return out


# ---- the world ------------------------------------------------------------------------------------------------

@dataclass
class World:
    xml: str
    labels: dict[str, dict[str, Any]]
    boxes: list[dict[str, Any]]
    blocker: dict[str, Any]
    seed: int
    layout_sha: str
    layout: dict[str, Any] = field(repr=False)


def _draw_boxes(layout: dict[str, Any], seed: int, places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng([seed, 0])
    rooms = [r for r in layout["rooms"] if r in layout["spawn_regions"]]
    segs = route_segments(layout)
    boxes: list[dict[str, Any]] = []
    for colour in BOX_COLOURS:
        for _ in range(10000):
            room = rooms[int(rng.integers(len(rooms)))]
            surfs = surfaces_in(layout, room)
            on_surf = bool(surfs) and float(rng.random()) < SURFACE_P
            if on_surf:
                f = surfs[int(rng.integers(len(surfs)))]
                x0, y0, x1, y1 = surface_top_rect(f)
                x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
                z, surface = f["box"][4] + BOX_HALF[2], f["surface"]
                # deviation 9 on surface boxes too: free (no surface spot of the 0.05 m grid is in a corridor; the
                # routes clear furniture), so it removes nothing today and holds if the layout changes
                ok = surface_spot_ok(layout, f, x, y) and route_clear(segs, x, y)
            else:
                reg = layout["spawn_regions"][room]
                x, y = float(rng.uniform(reg[0], reg[2])), float(rng.uniform(reg[1], reg[3]))
                z, surface = BOX_HALF[2], "floor"
                ok = spot_ok(layout, room, x, y, places) and route_clear(segs, x, y)
            if ok and all(math.hypot(x - b["x"], y - b["y"]) >= MIN_BOX_SEP for b in boxes):
                break
        else:
            raise RuntimeError(f"no spot for box {colour} (seed {seed})")
        yaw = float(rng.uniform(0.0, math.pi))
        boxes.append({"name": f"box_{colour}", "colour": colour, "room": room, "surface": surface,
                      "x": r6(x), "y": r6(y), "z": r6(z + SETTLE_DROP), "yaw": r6(yaw)})
    return boxes


def _draw_blocker(layout: dict[str, Any], seed: int, variant: str) -> dict[str, Any]:
    if variant == "none":
        return {"variant": "none", "door": None, "body": None}
    assert variant in ("crate", "cabinet"), variant
    rng = np.random.default_rng([seed, 1])
    doors = layout["blockers"][variant]["doors"]
    door = doors[int(rng.integers(len(doors)))]
    d = layout["doors"][door]
    (axis, c), = d["wall"].items()
    x, y = (c, d["center"]) if axis == "x" else (d["center"], c)
    half = CRATE_HALF if variant == "crate" else CABINET_HALF
    # cabinet: long side (half y) across the doorway, i.e. along the wall
    yaw = 0.0 if axis == "x" else math.pi / 2
    z = half[2] + (SETTLE_DROP if variant == "crate" else 0.0)
    return {"variant": variant, "door": door, "body": variant, "x": r6(x), "y": r6(y), "z": r6(z), "yaw": r6(yaw)}


def _rgba(rgb: list[float] | tuple[float, ...]) -> str:
    return fs(*rgb[:3], rgb[3] if len(rgb) > 3 else 1.0)


def _box_geom(name: str, half: tuple[float, ...], pos: tuple[float, ...], rgba: str, extra: str = "") -> str:
    return f'      <geom name="{name}" type="box" size="{fs(*half)}" pos="{fs(*pos)}" rgba="{rgba}"{extra}/>'


def _furniture(f: dict[str, Any]) -> list[str]:
    cx, cy, sx, sy, h = f["box"]
    rgba = _rgba(f["rgb"])
    n = f["id"]
    lines = [f'    <body name="{n}" pos="{fs(cx, cy, 0)}">']
    if f["kind"] == "table":
        lines.append(_box_geom(f"{n}_top", (sx / 2, sy / 2, TABLE_TOP_T / 2), (0, 0, h - TABLE_TOP_T / 2), rgba))
        lh = (h - TABLE_TOP_T) / 2
        for i, (ux, uy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            lines.append(_box_geom(f"{n}_leg{i}", (LEG_HALF, LEG_HALF, lh),
                                   (ux * (sx / 2 - LEG_HALF), uy * (sy / 2 - LEG_HALF), lh), rgba))
    elif f["kind"] == "solid":
        lines.append(_box_geom(f"{n}_body", (sx / 2, sy / 2, h / 2), (0, 0, h / 2), rgba))
    elif f["kind"] == "sofa":
        seat_h, back_t = 0.45, 0.2  # seat 0.45 m high; back along the -x (wall) side, full height
        lines.append(_box_geom(f"{n}_seat", (sx / 2, sy / 2, seat_h / 2), (0, 0, seat_h / 2), rgba))
        lines.append(_box_geom(f"{n}_back", (back_t / 2, sy / 2, h / 2), (-sx / 2 + back_t / 2, 0, h / 2), rgba))
    else:
        raise ValueError(f["kind"])
    lines.append("    </body>")
    return lines


def generate(layout_path: Path | str = LAYOUT, seed: int = 0, blocker: str = "none", *,
             sealed_draw: bool = False) -> World:
    refuse_test_seed(seed, sealed_draw)
    _DRAWN.add(EASINGS)
    _LABELS_DRAWN.add(LABELS)
    raw = Path(layout_path).read_bytes()
    layout = yaml.safe_load(raw)
    places = load_places()
    boxes = _draw_boxes(layout, seed, places)
    if LABELS:
        _draw_labels(seed, boxes)
    blk = _draw_blocker(layout, seed, blocker)
    labels: dict[str, dict[str, Any]] = {}
    H, T = layout["wall"]["height"], layout["wall"]["thickness"]
    xs = [v for r in layout["rooms"].values() for v in (r["rect"][0], r["rect"][2])]
    ys = [v for r in layout["rooms"].values() for v in (r["rect"][1], r["rect"][3])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    L = [f'<mujoco model="{layout["house"]}_s{seed}_{blocker}">',
         '  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>',
         '  <visual>',
         '    <headlight diffuse="0.25 0.25 0.25" ambient="0.15 0.15 0.15" specular="0 0 0"/>',
         '    <global offwidth="1280" offheight="960"/>',
         '    <quality shadowsize="0"/>',
         '  </visual>',
         '  <asset>',
         '    <material name="matte" specular="0" shininess="0" reflectance="0"/>',
         *([line for b in boxes for line in (
             f'    <texture name="label_{b["colour"]}" type="cube" file="{label_png(b["label"]["word"])}" '
             'gridsize="6 1" gridlayout="RLUDFB"/>',
             f'    <material name="label_{b["colour"]}" texture="label_{b["colour"]}" specular="0" shininess="0" '
             'reflectance="0"/>')] if LABELS else []),
         '  </asset>',
         '  <default>',
         '    <geom material="matte"/>',
         '  </default>',
         '  <worldbody>',
         f'    <geom name="floor" type="plane" size="{fs(max(xs) + 2, max(ys) + 2, 0.05)}" pos="{fs(cx, cy, 0)}" '
         'rgba="0.35 0.35 0.35 1"/>',
         f'    <camera name="overview" pos="{fs(cx, cy, 13)}" fovy="45"/>']
    labels["floor"] = {"class": "floor", "colour": None, "room": None, "surface": None}
    for i, (x, y) in enumerate(layout["lights"]):
        L.append(f'    <light name="light{i}" pos="{fs(x, y, LIGHT_Z)}" cutoff="180" diffuse="0.6 0.6 0.6" specular="0 0 0" '
                 'attenuation="1 0.3 0.1" castshadow="false"/>')
    for room, r in layout["rooms"].items():
        x0, y0, x1, y1 = r["rect"]
        n = f"floor_{room}"
        L.append(f'    <body name="{n}">')
        L.append(_box_geom(n, ((x1 - x0) / 2, (y1 - y0) / 2, PATCH_Z), ((x0 + x1) / 2, (y0 + y1) / 2, PATCH_Z),
                           _rgba(r["floor"]), ' contype="0" conaffinity="0"'))
        L.append("    </body>")
        labels[n] = {"class": "floor", "colour": None, "room": room, "surface": None}
    for i, (x0, y0, x1, y1) in enumerate(wall_rects(layout)):
        n = f"wall{i:02d}"
        L.append(f'    <body name="{n}">')
        L.append(_box_geom(n, ((x1 - x0) / 2, (y1 - y0) / 2, H / 2), ((x0 + x1) / 2, (y0 + y1) / 2, H / 2),
                           "0.92 0.9 0.86 1"))
        L.append("    </body>")
        labels[n] = {"class": "wall", "colour": None, "room": None, "surface": None}
    for f in layout["furniture"]:
        L += _furniture(f)
        labels[f["id"]] = {"class": f["id"], "colour": None, "room": f["room"], "surface": f.get("surface")}
    for b in boxes:
        L.append(f'    <body name="{b["name"]}" pos="{fs(b["x"], b["y"], b["z"])}" quat="{quat_s(b["yaw"])}">')
        L.append('      <freejoint/>')
        L.append(_box_geom(b["name"], BOX_HALF, (0, 0, 0), _rgba(BOX_COLOURS[b["colour"]]), f' mass="{fs(BOX_MASS)}"'
                           + (f' material="label_{b["colour"]}"' if LABELS else "")))
        L.append("    </body>")
        labels[b["name"]] = {"class": "box", "colour": b["colour"], "room": b["room"], "surface": b["surface"]}
    if blk["variant"] != "none":
        n = blk["body"]
        L.append(f'    <body name="{n}" pos="{fs(blk["x"], blk["y"], blk["z"])}" quat="{quat_s(blk["yaw"])}">')
        if n == "crate":
            L.append('      <freejoint/>')
            L.append(_box_geom(n, CRATE_HALF, (0, 0, 0), _rgba(CRATE_RGBA), f' mass="{fs(CRATE_MASS)}"'))
        else:
            L.append(_box_geom(n, CABINET_HALF, (0, 0, 0), _rgba(CABINET_RGBA)))
        L.append("    </body>")
        room = layout["doors"][blk["door"]]["between"]
        labels[n] = {"class": n, "colour": "purple" if n == "crate" else "grey", "room": None, "surface": None,
                     "door": blk["door"], "between": room}
    L += ["  </worldbody>", "</mujoco>", ""]
    return World(xml="\n".join(L), labels=labels, boxes=boxes, blocker=blk, seed=seed,
                 layout_sha=hashlib.sha256(raw).hexdigest(), layout=layout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=Path, default=LAYOUT)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--blocker", choices=("none", "crate", "cabinet"), default="none")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    w = generate(a.layout, a.seed, a.blocker)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "house.xml").write_text(w.xml)
    (a.out / "labels.json").write_text(json.dumps(
        {"seed": w.seed, "blocker": w.blocker, "layout_sha": w.layout_sha, "boxes": w.boxes, "labels": w.labels},
        indent=1) + "\n")
    print(f"wrote {a.out}/house.xml ({len(w.xml)} B) + labels.json; boxes "
          + ", ".join(f'{b["colour"]}@{b["room"]}/{b["surface"]}' for b in w.boxes) + f"; blocker {w.blocker}")


if __name__ == "__main__":
    main()
