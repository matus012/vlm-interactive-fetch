"""E1 helpers (task 009, context row 22): which eased rule each box of a world breaks, measured with the freeze-006
rules whatever the EASINGS switch says.

    box_flags(w) -> {box name: {"on_route": bool, "back_of_surface": bool, "near_surface_floor": bool}}

on_route: the box footprint is within ROBOT_RADIUS + ROUTE_MARGIN of a planned route (row 9 would reject it).
back_of_surface: a surface box with no standable edge point within STAND_OFF + SURF_EDGE_MAX (row 8's edge rule; on
the nightstand this is the back half, away from the open south edge; the box's own "surface" names which top).
near_surface_floor: a floor box closer than SURF_MIN to a spawn-surface footprint (row 8's floor rule).
"""
from __future__ import annotations

from typing import Any

from sim.constants import ROBOT_RADIUS, ROUTE_MARGIN
from sim.world.generate import (BOX_REACH, SURF_MIN, World, edge_reachable, furniture_rect, rect_dist, route_dist,
                                route_segments)


def box_flags(w: World) -> dict[str, dict[str, bool]]:
    layout: dict[str, Any] = w.layout
    segs = route_segments(layout)
    surf = {f["surface"]: f for f in layout["furniture"] if f.get("surface") in layout["spawn_surfaces"]}
    out: dict[str, dict[str, bool]] = {}
    for b in w.boxes:
        x, y = b["x"], b["y"]
        floor = b["surface"] == "floor"
        out[b["name"]] = {
            "on_route": route_dist(segs, x, y) <= ROBOT_RADIUS + ROUTE_MARGIN + BOX_REACH,
            "back_of_surface": not floor and not edge_reachable(layout, surf[b["surface"]], x, y),
            "near_surface_floor": floor and any(rect_dist(x, y, furniture_rect(f)) < SURF_MIN for f in surf.values()),
        }
    return out


def check_seal_005() -> dict:
    """Test (a): the seal-005 draw rule (sim.seal, sealed mode) re-run in memory at this tree with the switch at its
    default; only the sha256 comparison and the counts leave this function, never a task."""
    import hashlib
    import json

    from hpc.seal_regen import SEALED_SHA256
    from sim.constants import MIN_SURFACE_SEALED, SEALED_EXCLUDE, TEST_SEEDS
    from sim.sampler import draw_set
    from sim.seal import N_PER_TYPE, gl_renderer
    from sim.tasks import TYPES
    from sim.world import generate as gen

    assert gen.EASINGS
    assert "NVIDIA" in gl_renderer(), "test (a) needs NVIDIA px (llvmpipe px differ): an instrument failure, not a verdict"
    used: set[int] = set()
    tasks = []
    for t in TYPES:
        ts = draw_set(t, N_PER_TYPE, TEST_SEEDS.start, stop=TEST_SEEDS.stop, exclude=SEALED_EXCLUDE | used,
                      sealed_draw=True, min_surface=MIN_SURFACE_SEALED)
        used |= {x["seed"] for x in ts}
        tasks += ts
    data = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in tasks).encode()
    sha = hashlib.sha256(data).hexdigest()
    return {"n": len(tasks), "sha256_equal_seal_005": sha == SEALED_SHA256}


if __name__ == "__main__":
    import json
    import sys

    from sim.seal import gl_renderer
    assert sys.argv[1:] == ["--check-seal-005"], "usage: python -m sim.e1 --check-seal-005"
    print(json.dumps({"renderer": gl_renderer(), **check_seal_005()}))
