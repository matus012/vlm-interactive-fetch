"""Task generator (sim/tasks.py): determinism, type constraints, hall-visibility filter, seed split, script truth."""
from __future__ import annotations

import copy

import pytest

import sim.tasks as T
from agent.robot_api import PX_VISIBLE
from sim.constants import PX_OUT_OF_SIGHT
from sim.constants import ROBOT_RADIUS
from sim.world.generate import load_places
from sim.nav.astar import path_length, plan_path
from sim.nav.follow import CLEARANCE, DISC_MARGIN
from sim.world.occupancy import known_map

N = 5  # tasks per type here (rendering is slow under software EGL); the report's 10 come from the same function


@pytest.fixture(scope="module")
def tasks() -> dict[str, list[dict]]:
    return {t: T.dev_tasks(t, N) for t in T.TYPES}


def place_xy(name: str) -> tuple[float, float]:
    p = next(q for q in load_places() if q["name"] == name)
    return float(p["pose"][0]), float(p["pose"][1])


def closed_map(layout: dict, closed: str | None):
    """Known map with door `closed` filled (wall line +- one thickness, door width)."""
    g = known_map(layout)
    if closed is not None:
        d = layout["doors"][closed]
        (axis, c), = d["wall"].items()
        t, w = layout["wall"]["thickness"], layout["door_width"] / 2
        x0, y0, x1, y1 = ((c - t, d["center"] - w, c + t, d["center"] + w) if axis == "x"
                          else (d["center"] - w, c - t, d["center"] + w, c + t))
        i0, j0 = g.world_to_cell(x0, y0)
        i1, j1 = g.world_to_cell(x1, y1)
        g.occ[j0: j1 + 1, i0: i1 + 1] = True
    return g


def connected(layout: dict, a: tuple[float, float], b: tuple[float, float], closed: str | None) -> bool:
    """a and b in one free component of the known map (inflated by the robot radius) with door `closed` filled."""
    g = closed_map(layout, closed)
    lab = g.inflate(ROBOT_RADIUS + 0.02).components()
    (ia, ja), (ib, jb) = g.world_to_cell(*a), g.world_to_cell(*b)
    return bool(lab[ja, ia]) and lab[ja, ia] == lab[jb, ib]


def test_seed_split_refuses_test_space() -> None:
    assert not set(T.DEV_SEEDS) & set(T.TEST_SEEDS)
    for s in (100000, 150000, 199999):
        with pytest.raises(PermissionError):
            T.make_task(s, "fetch")
    for s in (999, 10000, 99999, 200000):
        with pytest.raises(ValueError):
            T.make_task(s, "fetch")
    with pytest.raises(T.NoTask):
        T.dev_tasks("fetch", 1, start=T.DEV_SEEDS.stop)


def test_determinism(tasks: dict[str, list[dict]]) -> None:
    first = {t: v[:2] for t, v in tasks.items()}
    T.world.cache_clear(), T.hall_px.cache_clear(), T.point_px.cache_clear()
    T.candidates.cache_clear(), T.pregrasp_ok.cache_clear()
    for t, v in first.items():
        assert [T.make_task(k["seed"], t) for k in v] == v
        assert v[0]["task_id"] == f"{t}-{v[0]['seed']}"


def test_types(tasks: dict[str, list[dict]]) -> None:
    for t, v in tasks.items():
        assert len({k["seed"] for k in v}) == N and all(k["seed"] in T.DEV_SEEDS for k in v)
        for k in v:
            w = T.world(k["seed"], k["blocker"])
            assert k["type"] == t and k["blocker"] == T.BLOCKER[t]
            assert k["step_limit"] == (30 if t == "two_box" else 20)
            assert len(k["targets"]) == (2 if t == "two_box" else 1)
            for g in k["targets"]:
                assert g["surface"] in w.layout["place_surfaces"]
            if t == "two_box":
                a, b = k["targets"]
                assert a["object"] != b["object"] and a["surface"] != b["surface"]
                assert k["instruction"] == (f"{a['object'].capitalize()} to the {a['surface']}, "
                                            f"{b['object']} to the {b['surface']}.")
            else:
                g = k["targets"][0]
                assert k["instruction"] == f"Bring the {g['object']} to the {g['surface']}."


def box_room(k: dict, box: str) -> str:
    return next(b["room"] for b in T.world(k["seed"], k["blocker"]).boxes if T.box_name(b["colour"]) == box)


def test_blocked_door_is_the_only_route(tasks: dict[str, list[dict]]) -> None:
    hall = place_xy("hall")
    for k in tasks["blocked"]:
        w = T.world(k["seed"], "crate")
        door, g = w.blocker["door"], k["targets"][0]
        assert door in w.layout["blockers"]["crate"]["doors"]
        box_pt = place_xy(T.room_point_for(k["seed"], "crate", g["object"]))
        goal = place_xy(g["surface"])
        assert connected(w.layout, hall, box_pt, None) and connected(w.layout, hall, goal, None)
        # closing the crate's door cuts the hall off from the box's room or from the target surface
        assert not (connected(w.layout, hall, box_pt, door) and connected(w.layout, hall, goal, door)), k


def test_push_or_detour_route_needs_the_kitchen(tasks: dict[str, list[dict]]) -> None:
    hall = place_xy("hall")
    for k in tasks["push_or_detour"]:
        w = T.world(k["seed"], "cabinet")
        door, g = w.blocker["door"], k["targets"][0]
        assert door in ("d_lk1", "d_lk2")
        in_kitchen = box_room(k, g["object"]) == "kitchen"
        assert g["surface"] == ("storage bench" if in_kitchen else "kitchen table")
        assert connected(w.layout, hall, place_xy("kitchen"), door)  # the other kitchen door is the detour
        # the cabinet is on the planned route: closing its door makes the hall -> box -> surface plan longer
        rp = T.room_point_for(k["seed"], "cabinet", g["object"])
        legs = [("hall", rp), (rp, g["surface"])]

        def length(closed: str | None) -> float:
            m = closed_map(w.layout, closed)
            return sum(path_length(plan_path(m.occ, m.origin, m.res, place_xy(a), place_xy(b), ROBOT_RADIUS,
                                             CLEARANCE + DISC_MARGIN)) for a, b in legs)

        assert length(door) > length(None) + 0.1, k
        tr = T.script_truth(k)
        assert tr["blocker"] == "cabinet"
        assert tr["detour_via"] == {"d_lk1": "kitchen south door", "d_lk2": "kitchen north door"}[door]


def test_hall_visibility_filter(tasks: dict[str, list[dict]]) -> None:
    rejected = 0
    for v in tasks.values():
        for k in v:
            px = T.hall_px(k["seed"], k["blocker"])
            for g in k["targets"]:
                assert px[g["object"]] < PX_OUT_OF_SIGHT and k["hall_px"][g["object"]] == px[g["object"]]
            rejected += sum(p >= PX_VISIBLE for p in px.values())
    assert rejected > 0
    # seed 1000: the hall look sees red, blue and green (>= 100 px), so every type must pick yellow
    px = T.hall_px(1000, "none")
    assert {b for b, p in px.items() if p >= PX_VISIBLE} == {"red box", "blue box", "green box"}
    assert T.make_task(1000, "fetch")["targets"][0]["object"] == "yellow box"
    with pytest.raises(T.NoTask):
        T.make_task(1000, "two_box")  # only one box hidden from the hall


def test_box_on_target_surface_is_not_a_target(monkeypatch: pytest.MonkeyPatch) -> None:
    real_world = T.world
    base = T.make_task(1000, "fetch")
    box, surf = base["targets"][0]["object"], base["targets"][0]["surface"]

    def fake(seed: int, blocker: str):
        w = copy.copy(real_world(seed, blocker))
        w.boxes = [dict(b, surface=surf) if T.box_name(b["colour"]) == box else b for b in w.boxes]
        return w

    monkeypatch.setattr(T, "world", fake)
    T.candidates.cache_clear()  # candidates() caches per (seed, type): drop the real world's entry, then the fake's
    k = T.make_task(1000, "fetch")
    T.candidates.cache_clear()
    assert k["targets"][0] != base["targets"][0]
    assert k["targets"][0]["object"] == box and k["targets"][0]["surface"] != surf  # the other surface is left


def test_script_truth(tasks: dict[str, list[dict]]) -> None:
    for t, v in tasks.items():
        for k in v[:2]:
            tr = T.script_truth(k)
            assert set(tr) == {"pairs", "blocker", "detour_via", "wrong_box", "wrong_surface"}
            assert [(p["box"], p["surface"]) for p in tr["pairs"]] == [(g["object"], g["surface"])
                                                                        for g in k["targets"]]
            for p in tr["pairs"] + [tr["wrong_box"]]:
                pl = next(q for q in load_places() if q["name"] == p["room_point"])
                assert pl["kind"] == "room" and pl["room"] == box_room(k, p["box"])
                assert T.point_px(k["seed"], k["blocker"], p["room_point"])[p["box"]] >= PX_VISIBLE, (k, p)
            assert tr["wrong_box"]["box"] not in {g["object"] for g in k["targets"]}
            assert tr["wrong_surface"] != k["targets"][0]["surface"]
            assert tr["blocker"] == (None if k["blocker"] == "none" else k["blocker"])
            if k["blocker"] != "cabinet":
                assert tr["detour_via"] is None
