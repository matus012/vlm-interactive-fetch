"""Phase 1 world: G1-det (byte-identical MJCF), spawn rules, known map, places, house + robot, short rest smoke."""
import math

import mujoco
import numpy as np
import pytest

from sim.constants import BOX_COLOURS, G1_COVER_GRID, ROBOT_RADIUS
from sim.world.generate import (LAYOUT, MIN_BOX_SEP, generate, load_layout, load_places, spawn_spots, spot_ok,
                                surfaces_in)
from sim.world.house import load_house
from sim.world.occupancy import known_map

SEEDS = range(5)
BLOCKERS = ("none", "crate", "cabinet")


@pytest.mark.parametrize("blocker", BLOCKERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_g1_det_byte_identical(seed: int, blocker: str) -> None:
    a, b = generate(LAYOUT, seed, blocker), generate(LAYOUT, seed, blocker)
    assert a.xml.encode() == b.xml.encode()
    assert a.labels == b.labels and a.boxes == b.boxes and a.layout_sha == b.layout_sha


def test_g1_det_seeds_differ() -> None:
    poses = [tuple((b["x"], b["y"], b["yaw"]) for b in generate(LAYOUT, s).boxes) for s in SEEDS]
    assert len(set(poses)) == len(poses)
    # the blocker draw is its own RNG stream: boxes do not depend on the variant
    assert all(generate(LAYOUT, s, k).boxes == generate(LAYOUT, s).boxes for s in SEEDS for k in BLOCKERS)


@pytest.mark.parametrize("seed", range(20))
def test_boxes_follow_spawn_rules(seed: int) -> None:
    layout, w = load_layout(), generate(LAYOUT, seed, "crate")
    assert sorted(b["colour"] for b in w.boxes) == sorted(BOX_COLOURS)
    for b in w.boxes:
        assert b["room"] != "hall" and 0.0 <= b["yaw"] < math.pi
        if b["surface"] == "floor":
            assert spot_ok(layout, b["room"], b["x"], b["y"])
        else:
            assert b["surface"] in [f["surface"] for f in surfaces_in(layout, b["room"])]
    for i, a in enumerate(w.boxes):
        for c in w.boxes[i + 1:]:
            assert math.hypot(a["x"] - c["x"], a["y"] - c["y"]) >= MIN_BOX_SEP


def test_blocker_doors() -> None:
    layout = load_layout()
    for s in SEEDS:
        assert generate(LAYOUT, s, "crate").blocker["door"] in layout["blockers"]["crate"]["doors"]
        assert generate(LAYOUT, s, "cabinet").blocker["door"] in layout["blockers"]["cabinet"]["doors"]


def test_spawn_spots() -> None:
    layout = load_layout()
    assert spawn_spots(layout, "hall", G1_COVER_GRID) == []
    for room in layout["spawn_regions"]:
        spots = spawn_spots(layout, room, G1_COVER_GRID)
        floor = [s for s in spots if s[3] == "floor"]
        assert floor and all(spot_ok(layout, room, x, y) for x, y, _, _ in floor)
        assert {s[3] for s in spots} - {"floor"} == {f["surface"] for f in surfaces_in(layout, room)}


def test_known_map_places_doors_connectivity() -> None:
    layout = load_layout()
    g = known_map(layout)
    assert g.occ.shape == (140, 180)
    gi = g.inflate(ROBOT_RADIUS + 0.02)
    lab_raw, lab = g.components(), gi.components()
    assert lab_raw.max() == 1  # all free space of the house is one region
    comps = set()
    for p in load_places():
        x, y = p["pose"][:2]
        assert gi.free(x, y), p["name"]
        ix, iy = gi.world_to_cell(x, y)
        comps.add(int(lab[iy, ix]))
    assert len(comps) == 1  # every place reachable from every other in the inflated map
    for name, d in layout["doors"].items():  # door centre on the wall line, and both sides, free when inflated
        (axis, c), = d["wall"].items()
        for off in (-0.1, 0.0, 0.1):
            x, y = (c + off, d["center"]) if axis == "x" else (d["center"], c + off)
            assert gi.free(x, y), name
            ix, iy = gi.world_to_cell(x, y)
            assert int(lab[iy, ix]) in comps, name


def test_house_loads_with_robot() -> None:
    w = generate(LAYOUT, 0, "cabinet")
    m, d = load_house(w)
    assert m.camera("mast").id >= 0
    x, y, yaw = w.layout["spawn"]
    for j, v in zip(("joint_x", "joint_y", "joint_th"), (x, y, math.radians(yaw))):
        assert d.qpos[m.joint(j).qposadr[0]] == pytest.approx(v) and d.ctrl[m.actuator(j).id] == pytest.approx(v)
    assert d.body("base_link").xpos[:2] == pytest.approx([x, y])
    assert d.qpos[m.joint("joint_4").qposadr[0]] == pytest.approx(-2.54818071)
    for b in w.boxes:
        assert d.body(b["name"]).xpos == pytest.approx([b["x"], b["y"], b["z"]])
    assert m.body_jntnum[m.body("cabinet").id] == 0  # welded


def test_rest_smoke() -> None:
    from sim.world.rest_check import rest_one

    r = rest_one(1, "crate", sim_s=10.0)
    assert r["pass"] and not r["nan"]


def test_r4a_keepouts() -> None:
    """R4a (task 003): surface approach points get the room-point keep-out; doorway corridors (door +- 0.6 m along
    the passage, door width + pad across) take no box; kind-door places are ignored by the point keep-out."""
    from sim.world.generate import DOOR_PAD, R4A_DOOR_DEPTH, ROOM_PT_MIN

    layout, places = load_layout(), load_places()
    approach = [p for p in places if p["kind"] == "surface"]
    assert approach
    half = (layout["door_width"] + DOOR_PAD) / 2.0
    counts = {}
    for room in layout["spawn_regions"]:
        floor = [s for s in spawn_spots(layout, room, G1_COVER_GRID) if s[3] == "floor"]
        counts[room] = len(floor)
        for x, y, _, _ in floor:
            for p in approach:
                if p["room"] == room:
                    assert math.hypot(x - p["pose"][0], y - p["pose"][1]) >= ROOM_PT_MIN, (room, x, y, p["name"])
            for d in layout["doors"].values():
                (axis, c), = d["wall"].items()
                along, across = (x - c, y - d["center"]) if axis == "x" else (y - c, x - d["center"])
                assert not (abs(along) <= R4A_DOOR_DEPTH and abs(across) <= half), (room, x, y)
    # spots per room after R4a on the G1-cover grid (before: bedroom 23, storage 47, living_room 58, kitchen 53)
    # fix round 2 (floor >= 0.60 m from spawn-surface footprints): bedroom 23 -> 22, living_room 58 -> 49
    # fix round 3 (deviation 9, route corridors): storage 39 -> 38, living_room 49 -> 19, kitchen 49 -> 12
    assert counts == {"bedroom": 22, "storage": 38, "living_room": 19, "kitchen": 12}
    # the approach point keep-out bites: a spot 0.5 m in front of the kitchen-table approach point is rejected
    ax, ay = next(p["pose"][:2] for p in approach if p["name"] == "kitchen table")
    assert spot_ok(layout, "kitchen", 7.55, 2.45) is False and math.hypot(7.55 - ax, 2.45 - ay) < ROOM_PT_MIN
    assert spot_ok(layout, "kitchen", 7.55, 2.45, [p for p in places if p["kind"] != "surface"]) is True
    # a kind-door place does not add a point keep-out
    door = {"name": "x door", "kind": "door", "room": "kitchen", "pose": [5.6, 2.0, 0.0]}
    assert spot_ok(layout, "kitchen", 5.6, 2.0) is True
    assert spot_ok(layout, "kitchen", 5.6, 2.0, places + [door]) is True


def test_surface_spots_reachable_rule() -> None:
    """Fix round 2: surface spots only near a top edge with standable floor in front (nightstand: south edge only, the
    bed and the walls box in the other three); floor spots >= SURF_MIN from every spawn-surface footprint."""
    from sim.world.generate import (STAND_OFF, SURF_EDGE_MAX, SURF_MIN, furniture_rect, rect_dist, standable,
                                    surface_spot_ok)

    layout = load_layout()
    ns = next(f for f in layout["furniture"] if f["id"] == "nightstand")
    x0, y0, x1, y1 = furniture_rect(ns)
    cx = (x0 + x1) / 2
    assert standable(layout, cx, y0 - STAND_OFF, ns)
    assert not standable(layout, cx, y1 + STAND_OFF, ns) and not standable(layout, x0 - STAND_OFF, 6.7, ns)
    assert not standable(layout, x1 + STAND_OFF, 6.7, ns)
    assert surface_spot_ok(layout, ns, cx, y0 + SURF_EDGE_MAX - 0.005)
    assert not surface_spot_ok(layout, ns, cx, y0 + SURF_EDGE_MAX + 0.005)
    for s in range(200):
        for b in generate(LAYOUT, s).boxes:
            if b["surface"] != "floor":
                f = next(g for g in layout["furniture"] if g.get("surface") == b["surface"])
                assert surface_spot_ok(layout, f, b["x"], b["y"]), (s, b)
            else:
                assert all(rect_dist(b["x"], b["y"], furniture_rect(f)) >= SURF_MIN for f in layout["furniture"]
                           if f.get("surface") in layout["spawn_surfaces"]), (s, b)


def test_route_corridor_rule() -> None:
    """Deviation 9 (fix round 3): no generated fetch box (floor or surface) within the robot's swept corridor
    (ROBOT_RADIUS + ROUTE_MARGIN) of any planned route between two places; footprint = BOX_REACH disc."""
    from sim.constants import ROUTE_MARGIN
    from sim.nav.astar import plan_path
    from sim.nav.follow import CLEARANCE, DISC_MARGIN
    from sim.world.generate import BOX_REACH, route_segments

    layout, places = load_layout(), load_places()
    a, b = route_segments(layout)
    g = known_map(layout)
    xy = [tuple(p["pose"][:2]) for p in places]
    # the cached set holds go_to's own plan for a pair (hall -> kitchen east) and its reverse
    for s, t in ((xy[0], xy[5]), (xy[5], xy[0])):
        path = plan_path(g.occ, g.origin, g.res, s, t, ROBOT_RADIUS, CLEARANCE + DISC_MARGIN)
        segs = set(zip(map(tuple, a.tolist()), map(tuple, b.tolist())))
        assert all((tuple(p), tuple(q)) in segs for p, q in zip(path, path[1:]))
    lim = ROBOT_RADIUS + ROUTE_MARGIN + BOX_REACH
    ab = b - a
    for s in range(300):
        for box in generate(LAYOUT, s).boxes:
            p = np.array([box["x"], box["y"]])
            t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
            dist = float(np.hypot(*(a + t[:, None] * ab - p).T).min())
            assert dist > lim, (s, box["name"], box["room"], dist)
