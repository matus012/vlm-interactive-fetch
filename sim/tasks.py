"""Seeded tasks (brief §Tasks) on the Phase 1 house generator, plus the truth a hand-written script needs.

    make_task(seed, ttype, sealed_draw=False, want=None) -> task dict
    dev_tasks(ttype, n, start=DEV_SEEDS.start) -> the first n valid tasks at dev seeds >= start
    script_truth(task) -> {"pairs", "blocker", "detour_via", "wrong_box", "wrong_surface"}

Task schema (scorer input, agent/scorer.py): {"task_id": "<type>-<seed>", "type", "seed", "blocker":
"none"|"crate"|"cabinet", "instruction", "targets": [{"object": "red box", "surface": "kitchen table"}, ...],
"step_limit": 20|30, "hall_px": {box: max seg px over the 4 hall photos}} (hall_px is an extra key: the evidence
for the visibility filter, renderer-dependent near the bar).

Seed spaces: dev = DEV_SEEDS (1000-9999). Test = TEST_SEEDS (100000-199999), disjoint, NOT drawn in Phase 2:
make_task refuses a test seed unless sealed_draw=True, which nothing in Phase 2 passes (the sealed draw is at the end
of Phase 4). Any other seed is refused too.

World per type: generate(LAYOUT, seed, blocker) — fetch / two_box "none", blocked "crate", push_or_detour "cabinet";
boxes do not depend on the blocker variant (own RNG stream), the blocker door does (seeded by the generator).
Target filter, in this order, over (box, surface) candidates in colour order x place_surfaces order:
- the box is out of sight from the hall spawn: robot at layout.spawn, arm LOOK_ARM_QPOS, base yaw 0/90/180/270
  (set directly, as G1-cover), segmentation px of the box geom < PX_OUT_OF_SIGHT (10, deviation 13; was PX_VISIBLE
  100 before task 005) in every photo, in the task's world (the blocker can hide a box); cached per (seed, blocker);
- the box is not already on the surface;
- type rule. blocked: the crate's door cuts rooms off from the hall in the door graph (bedroom or storage); the box
  or the surface is in a cut-off room. push_or_detour: the cabinet door does not cut anything off (the other
  kitchen door is the detour); box in the kitchen -> storage bench, or box outside the kitchen -> kitchen table;
  and the known-map plan hall -> box room point -> surface approach point passes through the cabinet's door
  (route_uses_door), so the cabinet is actually met.
  two_box: two different boxes, one per place surface.
- deviation 15 (task 005): every target box has a feasible pre-grasp pose with every other box avoided
  (pregrasp_ok: grasp.plan_pregrasp on a fresh SimRobot of the task's world, all other boxes as avoid discs, planning
  only, the check of sim/pregrasp_survey.py). E1 (task 009, row 22): off when the generator's EASINGS is False, and
  the task dict then carries "easings": False (no such key with the default, so freeze-006 tasks are unchanged).
`want` (the balanced sampler, sim/sampler.py) keeps only candidates whose FIRST target matches {"room": layout room
key, "support": "floor" | spawn surface name, "surface": place surface}, any subset of the keys. two_box without
want: first target to place_surfaces[0] (Phase 2 order); with want: either order, the first target matches want and
the second is any valid partner (one ordered pair is drawn uniformly).
Then one candidate is drawn with numpy PCG64 seeded [seed, 2, type index]. A seed with no candidate raises
NoTask; dev_tasks skips it. Same (seed, type) -> same task.
"""
from __future__ import annotations

import functools
import math

import mujoco
import numpy as np

from agent.robot_api import PX_VISIBLE
from sim import grasp
from sim.adapter import SimRobot
from sim.constants import (DEV_SEEDS, PX_OUT_OF_SIGHT, ROBOT_RADIUS, STEP_LIMIT, STEP_LIMIT_TWO_BOX, TEST_SEEDS)
from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, set_arm, set_yaw_direct
from sim.nav.astar import plan_path
from sim.nav.follow import CLEARANCE, DISC_MARGIN
from sim.robot import CAM_H, CAM_W, MAST_CAM
from sim.world import generate as gen
from sim.world.generate import LAYOUT, World, generate, load_places
from sim.world.house import load_house, set_base
from sim.world.occupancy import known_map

TYPES = ("fetch", "blocked", "push_or_detour", "two_box")
BLOCKER = {"fetch": "none", "blocked": "crate", "push_or_detour": "cabinet", "two_box": "none"}
DOOR_NAMES = {p["door"]: p["name"] for p in load_places() if p["kind"] == "door"}  # layout door id -> VLM place name


DETOUR_PAIR = {"room": "kitchen", "surface": "storage bench"}  # house A; a layout may set detour_pair (task 011)


def detour_pair(layout: dict) -> dict[str, str]:
    """push_or_detour pairing: a box in `room` goes to `surface`, any other box to the other place surface."""
    return layout.get("detour_pair", DETOUR_PAIR)


class NoTask(ValueError):
    """The seed has no valid target for the type."""


_SEALED_OK: set[int] = set()  # test-space seeds a make_task(..., sealed_draw=True) call has authorised


@functools.cache
def world(seed: int, blocker: str) -> World:
    """generate() for the helpers below. A test-space seed is refused (generate's guard) unless make_task authorised
    it with sealed_draw=True first, so no helper (hall_px, look_px, script_truth, ...) opens the sealed space silently."""
    return generate(LAYOUT, seed, blocker, sealed_draw=seed in _SEALED_OK)


def box_name(colour: str) -> str:
    return f"{colour} box"


def look_px(seed: int, blocker: str, xy: tuple[float, float]) -> dict[str, int]:
    """Max segmentation px per fetch box over the 4 look photos from base position xy."""
    w = world(seed, blocker)
    m, d = load_house(w)
    set_base(m, d, (xy[0], xy[1], 0.0))
    set_arm(m, d, LOOK_ARM_QPOS)
    gid = {box_name(b["colour"]): m.geom(b["name"]).id for b in w.boxes}
    out = dict.fromkeys(gid, 0)
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    try:
        r.enable_segmentation_rendering()
        for h in HEADINGS_DEG:
            set_yaw_direct(m, d, math.radians(h))
            r.update_scene(d, camera=MAST_CAM)
            s = r.render()
            geom = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
            for k, g in gid.items():
                out[k] = max(out[k], int((geom == g).sum()))
    finally:
        r.close()
    return out


@functools.cache
def hall_px(seed: int, blocker: str) -> dict[str, int]:
    x, y, _ = world(seed, blocker).layout["spawn"]
    return look_px(seed, blocker, (float(x), float(y)))


@functools.cache
def point_px(seed: int, blocker: str, place: str) -> dict[str, int]:
    p = next(q for q in load_places() if q["name"] == place)
    return look_px(seed, blocker, (float(p["pose"][0]), float(p["pose"][1])))


def surface_room(w: World, surface: str) -> str:
    return next(f["room"] for f in w.layout["furniture"] if f.get("surface") == surface)


def cut_off(w: World, door: str) -> set[str]:
    return cut_off_layout(w.layout, door)


def cut_off_layout(layout: dict, door: str) -> set[str]:
    """Rooms unreachable from the hall in the door graph once `door` is closed."""
    seen, todo = {"hall"}, ["hall"]
    while todo:
        r = todo.pop()
        for k, d in layout["doors"].items():
            if k != door and r in d["between"]:
                for o in d["between"]:
                    if o not in seen:
                        seen.add(o)
                        todo.append(o)
    return set(layout["rooms"]) - seen


def crosses_door(layout: dict, path: list[tuple[float, float]], door: str) -> bool:
    """Some polyline segment crosses the door's wall line inside the opening."""
    d = layout["doors"][door]
    (axis, c), = d["wall"].items()
    i = 0 if axis == "x" else 1
    for p, q in zip(path, path[1:]):
        if (p[i] - c) * (q[i] - c) <= 0 and p[i] != q[i]:
            u = (c - p[i]) / (q[i] - p[i])
            if abs(p[1 - i] + u * (q[1 - i] - p[1 - i]) - d["center"]) <= layout["door_width"] / 2:
                return True
    return False


def route_uses_door(seed: int, blocker: str, box: str, surface: str, door: str) -> bool:
    """The known-map plan (the follower's own planner and clearance, blockers not on the map) hall -> the box's room
    point -> the surface approach point goes through `door`. Without this a cabinet in the kitchen door the planner
    does not use would never block anything (push_or_detour would be a plain fetch)."""
    w = world(seed, blocker)
    g = known_map(w.layout)
    xy = {p["name"]: (float(p["pose"][0]), float(p["pose"][1])) for p in load_places()}
    legs = [("hall", room_point_for(seed, blocker, box)), (room_point_for(seed, blocker, box), surface)]
    for a, b in legs:
        path = plan_path(g.occ, g.origin, g.res, xy[a], xy[b], ROBOT_RADIUS, CLEARANCE + DISC_MARGIN)
        if path is not None and crosses_door(w.layout, path, door):
            return True
    return False


def check_seed(seed: int, sealed_draw: bool) -> None:
    if seed in TEST_SEEDS:
        if not sealed_draw:
            raise PermissionError(f"seed {seed} is in the sealed test space; Phase 2 draws dev seeds only")
    elif seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is in neither the dev nor the test space")


@functools.cache
def pregrasp_ok(seed: int, blocker: str) -> frozenset[str]:
    """Deviation 15: the boxes with >= 1 pre-grasp candidate when every other box is an avoid disc (as pick gets them
    once all boxes are seen), on a fresh SimRobot of the task's world; planning only (= sim/pregrasp_survey.py)."""
    r = SimRobot(seed, blocker, world=world(seed, blocker), sealed_draw=seed in _SEALED_OK)
    boxes = [o for o, v in r.objects().items() if v["class"] == "box"]
    r.seen |= set(boxes)
    return frozenset(b for b in boxes if grasp.plan_pregrasp(r.m, r.d, r.grid, r.body_of[b], r._seen_discs({b})))


def start_of(seed: int, blocker: str, box: str) -> tuple[str, str]:
    """(layout room key, support: "floor" or the spawn surface name) of the box at spawn."""
    b = next(b for b in world(seed, blocker).boxes if box_name(b["colour"]) == box)
    return b["room"], b["surface"]


@functools.cache
def candidates(seed: int, ttype: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Every valid target list for the seed (module docstring filter), in draw order; two_box: ordered pairs."""
    blocker = BLOCKER[ttype]
    w = world(seed, blocker)
    px = hall_px(seed, blocker)
    room = {box_name(b["colour"]): b["room"] for b in w.boxes}
    support = {box_name(b["colour"]): b["surface"] for b in w.boxes}
    surfaces = list(w.layout["place_surfaces"])
    cand = [(b, s) for b in room for s in surfaces if px[b] < PX_OUT_OF_SIGHT and support[b] != s]
    if ttype == "blocked":
        cut = cut_off(w, w.blocker["door"])
        cand = [(b, s) for b, s in cand if room[b] in cut or surface_room(w, s) in cut]
    elif ttype == "push_or_detour":
        assert not cut_off(w, w.blocker["door"]), w.blocker
        dp = detour_pair(w.layout)
        cand = [(b, s) for b, s in cand if (room[b] == dp["room"]) == (s == dp["surface"])
                and route_uses_door(seed, blocker, b, s, w.blocker["door"])]
    if gen.EASINGS:  # row 15 (E1: lifted)
        ok = pregrasp_ok(seed, blocker) if cand else frozenset()
        cand = [(b, s) for b, s in cand if b in ok]
    if ttype == "two_box":
        return tuple((t1, t2) for t1 in cand for t2 in cand if t2[0] != t1[0] and t2[1] != t1[1])
    return tuple((t,) for t in cand)


def make_task(seed: int, ttype: str, sealed_draw: bool = False, want: dict[str, str] | None = None) -> dict:
    check_seed(seed, sealed_draw)
    if sealed_draw and seed in TEST_SEEDS:
        _SEALED_OK.add(seed)
    blocker = BLOCKER[ttype]
    cand = list(candidates(seed, ttype))
    if ttype == "two_box" and want is None:
        first = world(seed, blocker).layout["place_surfaces"][0]
        cand = [c for c in cand if c[0][1] == first]
    if want is not None:
        def match(t: tuple[str, str]) -> bool:
            r, sup = start_of(seed, blocker, t[0])
            return all({"room": r, "support": sup, "surface": t[1]}[k] == v for k, v in want.items())
        cand = [c for c in cand if match(c[0])]
    if not cand:
        raise NoTask(f"{ttype} seed {seed}")
    rng = np.random.default_rng([seed, 2, TYPES.index(ttype)])
    targets = list(cand[int(rng.integers(len(cand)))])
    if ttype == "two_box":
        instruction = (f"{targets[0][0].capitalize()} to the {targets[0][1]}, "
                       f"{targets[1][0]} to the {targets[1][1]}.")
    else:
        instruction = f"Bring the {targets[0][0]} to the {targets[0][1]}."
    px = hall_px(seed, blocker)
    task = {"task_id": f"{ttype}-{seed}", "type": ttype, "seed": seed, "blocker": blocker,
            "instruction": instruction, "targets": [{"object": b, "surface": s} for b, s in targets],
            "step_limit": STEP_LIMIT_TWO_BOX if ttype == "two_box" else STEP_LIMIT,
            "hall_px": {b: px[b] for b, _ in targets}}
    task = task if gen.HOUSE == "A" else task | {"house": gen.HOUSE}  # E2 (task 011): no key in house A
    return task if gen.EASINGS else task | {"easings": False}


def dev_tasks(ttype: str, n: int, start: int = DEV_SEEDS.start) -> list[dict]:
    out = []
    for seed in range(start, DEV_SEEDS.stop):
        if len(out) == n:
            break
        try:
            out.append(make_task(seed, ttype))
        except NoTask:
            continue
    if len(out) < n:
        raise NoTask(f"only {len(out)} {ttype} dev tasks from seed {start}")
    return out


def room_point_for(seed: int, blocker: str, box: str) -> str:
    """The room point (places.yaml kind room) of the box's room whose look sees the box most (>= PX_VISIBLE when any
    point sees it; places.yaml order breaks ties)."""
    w = world(seed, blocker)
    room = next(b["room"] for b in w.boxes if box_name(b["colour"]) == box)
    pts = [p["name"] for p in load_places() if p["kind"] == "room" and p["room"] == room]
    seen = [p for p in pts if point_px(seed, blocker, p)[box] >= PX_VISIBLE]
    return seen[0] if seen else max(pts, key=lambda p: point_px(seed, blocker, p)[box])


def script_truth(task: dict) -> dict:
    """What the correct / wrong scripts (track 2e) need; never shown to a VLM."""
    seed, blocker = task["seed"], task["blocker"]
    w = world(seed, blocker)
    pairs = [{"box": t["object"], "room_point": room_point_for(seed, blocker, t["object"]), "surface": t["surface"]}
             for t in task["targets"]]
    door = w.blocker["door"]
    detour = None
    if blocker == "cabinet":
        # the other cabinet door (house A: the other living / kitchen door, as before; task 011: layout-driven)
        other = [k for k in w.layout["blockers"]["cabinet"]["doors"] if k != door]
        doors = w.layout["doors"]
        if sorted(doors[other[0]]["between"]) == sorted(doors[door]["between"]):
            detour = DOOR_NAMES[other[0]]  # twin door (house A): navigate's door-via closes the blocked twin
        else:
            # task 011 (house B): the doors join different rooms, so a door via does not close the blocked door
            # (adapter._nav_grid); go round through the room on the far side of the other cabinet door instead
            far = next(r for r in doors[other[0]]["between"] if r != detour_pair(w.layout)["room"])
            detour = next(p["name"] for p in load_places() if p["kind"] == "room" and p["room"] == far)
    targets = {t["object"] for t in task["targets"]}
    cut = cut_off(w, door) if blocker == "crate" else set()
    others = [box_name(b["colour"]) for b in w.boxes if box_name(b["colour"]) not in targets]
    # prefer a wrong box outside a crate-cut room that a room point sees
    room = {box_name(b["colour"]): b["room"] for b in w.boxes}
    wb = min(others, key=lambda b: (room[b] in cut,
                                    point_px(seed, blocker, room_point_for(seed, blocker, b))[b] < PX_VISIBLE))
    surfaces = list(w.layout["place_surfaces"])
    return {"pairs": pairs, "blocker": None if blocker == "none" else blocker, "detour_via": detour,
            "wrong_box": {"box": wb, "room_point": room_point_for(seed, blocker, wb)},
            "wrong_surface": next(s for s in surfaces if s != task["targets"][0]["surface"])}
