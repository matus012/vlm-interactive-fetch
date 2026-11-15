"""Balanced task-set sampler (task 005 pre-work 4 and 9, deviation 13).

    draw_set(ttype, n, start, stop=None, exclude=(), sealed_draw=False, min_surface=0) -> n task dicts

Quota plan, from a numpy PCG64 seeded [start, 3, type index, n] (the set's own seed):
- room: the target box's room, uniform over the rooms that can hold a target for the type (`eligible`): n // R slots
  per room, the n % R extra slots to rooms drawn without replacement. Type rules (sim.tasks): blocked needs the box
  or the surface in a room the crate's door cuts off (crate doors d_bed / d_sto: bedroom / storage); push_or_detour
  pairs a kitchen box with the storage bench and any other box with the kitchen table; fetch / two_box: any. With
  the current layout every spawn room (bedroom, storage, living_room, kitchen) is eligible for every type.
- support: in a room with a spawn surface a target can start on (TARGET_SPAWN_SURFACES: the nightstand; never the
  coffee table, which the < 10 px rule always rejects), round(SURFACE_P x room slots) slots on that surface (SURFACE_P = 0.25, the
  generator's own share: the natural share of a box spawning there), the rest floor; then surface slots are added, one
  at a time to the surface room with the lowest surface fraction (layout order on ties), until >= min_surface
  (MIN_SURFACE_COUNTED 2 for a counted set of 10, MIN_SURFACE_SEALED 5 per 25 in the sealed set; 0 = natural only).
- surface: the target place surface, uniform (n // 2 each, the odd slot to a surface drawn by the RNG); a slot whose
  type rule allows one surface only gets it (push_or_detour: always; blocked: a kitchen / living-room box -> storage
  bench), the free slots, in RNG order, take the surface furthest below its quota (place_surfaces order on ties).
- slots are put in RNG order. Then seeds are scanned in order (start .. stop, minus exclude); each seed fills the
  first open slot (plan order) for which make_task(seed, ttype, want=slot) has a candidate, at most one task per
  seed. Deterministic: same arguments -> same tasks.
two_box: the slot constrains the first target (room, support, surface); the second is drawn by make_task among the
valid partners (another box, the other place surface; all filters), one ordered pair uniformly.
Every task carries "slot": {"room", "support", "surface"} (its first target's start and surface).

    python -m sim.sampler --stats fetch 7000 7300   # reject rates of the < 10 px and pre-grasp filters
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from agent.robot_api import PX_VISIBLE
from sim.constants import DEV_SEEDS, PX_OUT_OF_SIGHT, SEALED_EXCLUDE, TEST_SEEDS
from sim.tasks import (BLOCKER, TYPES, NoTask, box_name, cut_off_layout, detour_pair, hall_px, make_task, pregrasp_ok,
                       world)
from sim.world.generate import SURFACE_P, load_layout, surfaces_in

# Spawn surfaces a target box can start on. The coffee table is not one: through the hall-living door it is in view
# of the hall spawn, so the < 10 px out-of-sight rule (deviation 13) rejects every coffee-table box (seeds 9000-9199:
# 130 coffee-table boxes, hall px min 52, 0 below 10; nightstand: 83 boxes, all 0 px; runs/4b/coffee.py, 2026-11-13).
TARGET_SPAWN_SURFACES = ("nightstand",)
# E2 (task 011): a layout may name its own (layout key target_spawn_surfaces); absent = house A's tuple above.


def target_spawn_surfaces(layout: dict) -> tuple[str, ...]:
    return tuple(layout.get("target_spawn_surfaces", TARGET_SPAWN_SURFACES))


def slot_ok(layout: dict, ttype: str, room: str, surface: str) -> bool:
    """The type rule allows a target box in `room` going to `surface` (some blocker door of the type)."""
    sroom = next(f["room"] for f in layout["furniture"] if f.get("surface") == surface)
    if ttype == "blocked":
        return any(room in c or sroom in c for c in (cut_off_layout(layout, d) for d in layout["blockers"]["crate"]["doors"]))
    if ttype == "push_or_detour":
        dp = detour_pair(layout)
        return (room == dp["room"]) == (surface == dp["surface"])
    return True


def eligible(layout: dict, ttype: str) -> list[str]:
    return [r for r in layout["spawn_regions"] if any(slot_ok(layout, ttype, r, s) for s in layout["place_surfaces"])]


def plan(ttype: str, n: int, start: int, min_surface: int = 0) -> list[dict[str, str]]:
    layout = load_layout()
    rng = np.random.default_rng([start, 3, TYPES.index(ttype), n])
    rooms = eligible(layout, ttype)
    count = dict.fromkeys(rooms, n // len(rooms))
    for i in rng.choice(len(rooms), n % len(rooms), replace=False):
        count[rooms[int(i)]] += 1
    surf_of = {}
    for r in rooms:
        fs = [f for f in surfaces_in(layout, r) if f["surface"] in target_spawn_surfaces(layout)]
        assert len(fs) <= 1, r
        if fs:
            surf_of[r] = fs[0]["surface"]
    k = {r: int(np.floor(SURFACE_P * count[r] + 0.5)) for r in surf_of}
    while sum(k.values()) < min_surface:
        open_ = [r for r in surf_of if k[r] < count[r]]
        if not open_:
            raise ValueError(f"{ttype}: min_surface {min_surface} > surface-room slots")
        r = min(open_, key=lambda r: k[r] / count[r])
        k[r] += 1
    slots = [{"room": r, "support": surf_of[r] if i < k.get(r, 0) else "floor"} for r in rooms for i in range(count[r])]
    places = list(layout["place_surfaces"])
    quota = dict.fromkeys(places, n // 2)
    if n % 2:
        quota[places[int(rng.integers(2))]] += 1
    order = [int(i) for i in rng.permutation(n)]
    for i in order:  # forced slots first
        opts = [s for s in places if slot_ok(layout, ttype, slots[i]["room"], s)]
        if len(opts) == 1:
            slots[i]["surface"] = opts[0]
            quota[opts[0]] -= 1
    for i in order:
        if "surface" not in slots[i]:
            s = max(places, key=lambda s: quota[s])  # first max = place_surfaces order on ties
            slots[i]["surface"] = s
            quota[s] -= 1
    return [slots[i] for i in rng.permutation(n)]


def draw_set(ttype: str, n: int, start: int, stop: int | None = None, exclude: frozenset[int] | set[int] = frozenset(),
             sealed_draw: bool = False, min_surface: int = 0) -> list[dict]:
    if sealed_draw:
        assert SEALED_EXCLUDE <= set(exclude), "the sealed draw must exclude every test seed a test touches"
    space = TEST_SEEDS if sealed_draw else DEV_SEEDS
    seeds = range(start, space.stop if stop is None else stop)
    assert seeds.start in space and seeds.stop - 1 in space, (seeds, space)
    slots = plan(ttype, n, start, min_surface)
    out: list[dict | None] = [None] * n
    for seed in seeds:
        if seed in exclude:
            continue
        for i, want in enumerate(slots):
            if out[i] is None:
                try:
                    out[i] = make_task(seed, ttype, sealed_draw=sealed_draw, want=want) | {"slot": dict(want)}
                    break
                except NoTask:
                    continue
        if all(out):
            return out  # type: ignore[return-value]
    raise NoTask(f"{ttype}: {sum(o is None for o in out)} of {n} slots unfilled in {seeds}")


def table(tasks: list[dict]) -> dict[str, dict[str, int]]:
    """room / support / surface counts of a set (first targets)."""
    return {k: dict(sorted(Counter(t["slot"][k] for t in tasks).items())) for k in ("room", "support", "surface")}


def stats(ttype: str, seeds: range) -> dict:
    """Reject rates of the two task-005 filters over the seeds. px: boxes (not on a place surface, so every box) that
    pass the old < 100 px bar, and of those the ones the < 10 px rule rejects. pre-grasp: target boxes left after every
    other filter (out of sight, type rule) for which pregrasp_ok finds no pose; seeds whose task it removes."""
    blocker = BLOCKER[ttype]
    old = new = boxes = pg_rej = 0
    no_task = 0
    hits = []
    for s in seeds:
        px = hall_px(s, blocker)
        boxes += len(px)
        old += sum(v < PX_VISIBLE for v in px.values())
        new += sum(v < PX_OUT_OF_SIGHT for v in px.values())
        ok = pregrasp_ok(s, blocker)
        out = [box_name(b["colour"]) for b in world(s, blocker).boxes if px[box_name(b["colour"])] < PX_OUT_OF_SIGHT]
        bad = [b for b in out if b not in ok]
        pg_rej += len(bad)
        hits += [{"seed": s, "box": b} for b in bad]
        try:
            make_task(s, ttype)
        except NoTask:
            no_task += 1
    return {"type": ttype, "seeds": [seeds.start, seeds.stop], "boxes": boxes, "out_of_sight_lt100": old,
            "out_of_sight_lt10": new, "px_rule_rejects": old - new, "px_reject_rate": (old - new) / old,
            "out_of_sight_boxes_no_pregrasp": pg_rej, "pregrasp_reject_rate": pg_rej / new, "pregrasp_hits": hits,
            "seeds_without_task": no_task}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs=3, metavar=("TYPE", "START", "STOP"), required=True)
    a = ap.parse_args()
    print(json.dumps(stats(a.stats[0], range(int(a.stats[1]), int(a.stats[2])))))


if __name__ == "__main__":
    main()
