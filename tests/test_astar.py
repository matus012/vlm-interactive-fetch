from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import yaml

from sim.constants import GRID_RES, ROBOT_RADIUS
from sim.nav.astar import astar, inflate, path_length, plan_path, segment_headings

RES = GRID_RES
ORIGIN = (0.0, 0.0)
ROOT = Path(__file__).resolve().parents[1]


def grid(w_m: float, h_m: float) -> np.ndarray:
    return np.zeros((round(h_m / RES), round(w_m / RES)), bool)


def box(occ: np.ndarray, x0: float, y0: float, x1: float, y1: float, v: bool = True) -> None:
    occ[round(y0 / RES):round(y1 / RES), round(x0 / RES):round(x1 / RES)] = v


def wall_with_gap(gap: float) -> np.ndarray:
    occ = grid(6.0, 4.0)
    box(occ, 2.95, 0.0, 3.05, 4.0)
    box(occ, 2.95, 2.0 - gap / 2, 3.05, 2.0 + gap / 2, False)
    return occ


def assert_clear(occ: np.ndarray, path: list[tuple[float, float]], r: float) -> None:
    blocked = inflate(occ, r, RES)
    for a, b in zip(path, path[1:]):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.01) + 1)
        for t in np.linspace(0, 1, n):
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            assert not blocked[int(y // RES), int(x // RES)], (x, y)


def test_straight_corridor() -> None:
    occ = grid(6.0, 1.5)
    box(occ, 0, 0, 6, 0.2)
    box(occ, 0, 1.3, 6, 1.5)
    p = plan_path(occ, ORIGIN, RES, (0.5, 0.75), (5.5, 0.75), 0.284)
    assert p is not None and len(p) == 2
    assert abs(path_length(p) - 5.0) < 1e-9
    assert_clear(occ, p, 0.304)


def test_door_gap_passable_1m_not_half() -> None:
    occ = wall_with_gap(1.0)
    p = plan_path(occ, ORIGIN, RES, (1.0, 2.0), (5.0, 2.0), ROBOT_RADIUS, 0.02)
    assert p is not None and len(p) <= 4
    assert_clear(occ, p, ROBOT_RADIUS + 0.02)
    assert plan_path(wall_with_gap(0.5), ORIGIN, RES, (1.0, 2.0), (5.0, 2.0), ROBOT_RADIUS, 0.02) is None


def two_rooms() -> np.ndarray:
    occ = grid(6.0, 6.0)
    box(occ, 2.95, 0.0, 3.05, 6.0)
    box(occ, 2.95, 0.75, 3.05, 1.75, False)  # door low
    box(occ, 2.95, 4.25, 3.05, 5.25, False)  # door high
    return occ


def test_two_doors_block_one() -> None:
    occ = two_rooms()
    p = plan_path(occ, ORIGIN, RES, (1.0, 1.5), (5.0, 1.5), 0.284)
    assert p is not None and path_length(p) < 4.2  # low door, near straight
    occ2 = occ.copy()
    box(occ2, 2.95, 0.75, 3.05, 1.75)  # block the low door
    q = plan_path(occ2, ORIGIN, RES, (1.0, 1.5), (5.0, 1.5), 0.284)
    assert q is not None and path_length(q) > 7.0
    assert any(abs(x - 3.0) < 0.3 and 4.25 <= y <= 5.25 for x, y in q) or any(3.0 - 0.01 < x < 3.0 + 0.01 for x, _ in q)
    assert_clear(occ2, q, 0.304)
    crossing = [(a, b) for a, b in zip(q, q[1:]) if min(a[0], b[0]) < 3.0 < max(a[0], b[0])]
    assert crossing and all(4.25 < a[1] + (3.0 - a[0]) / (b[0] - a[0]) * (b[1] - a[1]) < 5.25 for a, b in crossing)


def test_via_forces_third_room() -> None:
    occ = grid(9.0, 6.0)
    box(occ, 2.95, 0.0, 3.05, 6.0)
    box(occ, 5.95, 0.0, 6.05, 6.0)
    box(occ, 0.0, 2.95, 9.0, 3.05)
    box(occ, 2.95, 4.0, 3.05, 5.0, False)
    box(occ, 5.95, 4.0, 6.05, 5.0, False)
    box(occ, 2.95, 0.75, 3.05, 1.75, False)
    box(occ, 5.95, 0.75, 6.05, 1.75, False)
    box(occ, 3.5, 2.95, 4.5, 3.05, False)  # door into the third room (top-middle block x 3..6, y 3..6)
    via = np.zeros_like(occ)
    box(via, 3.0, 3.0, 6.0, 6.0)
    via = via.astype(bool)
    start, goal = (1.0, 1.25), (5.0, 1.25)
    direct = plan_path(occ, ORIGIN, RES, start, goal, 0.284)
    routed = plan_path(occ, ORIGIN, RES, start, goal, 0.284, via_cells=via)
    assert direct is not None and routed is not None
    assert path_length(routed) > path_length(direct) + 2.0
    assert any(via[int(y // RES), int(x // RES)] for x, y in routed)
    assert_clear(occ, routed, 0.304)


def test_via_xy_is_a_waypoint() -> None:
    """Two doors in one wall; via_xy at the far door's centre forces the route through that door (task 003 R3)."""
    occ = grid(6.0, 4.0)
    box(occ, 2.95, 0.0, 3.05, 4.0)
    box(occ, 2.95, 0.5, 3.05, 1.5, False)  # near door, y 1.0
    box(occ, 2.95, 2.5, 3.05, 3.5, False)  # far door, y 3.0
    start, goal = (1.0, 1.0), (5.0, 1.0)
    direct = plan_path(occ, ORIGIN, RES, start, goal, 0.284)
    routed = plan_path(occ, ORIGIN, RES, start, goal, 0.284, via_xy=(3.0, 3.0))
    assert direct is not None and routed is not None
    assert (3.0, 3.0) in routed and routed[0] == start and routed[-1] == goal
    assert path_length(routed) > path_length(direct) + 1.5
    assert_clear(occ, routed, 0.304)


def test_unreachable() -> None:
    occ = grid(6.0, 4.0)
    box(occ, 2.95, 0.0, 3.05, 4.0)
    assert plan_path(occ, ORIGIN, RES, (1.0, 2.0), (5.0, 2.0), 0.284) is None
    free = ~inflate(occ, 0.304, RES)
    assert astar(free, (20, 40), (100, 40)) is None


def test_snap_start() -> None:
    occ = wall_with_gap(1.0)
    p = plan_path(occ, ORIGIN, RES, (2.8, 1.0), (5.0, 2.0), 0.284)  # 0.2 m from wall: inside inflation, snaps
    assert p is not None and p[0] != (2.8, 1.0)
    assert plan_path(occ, ORIGIN, RES, (3.0, 1.0), (5.0, 2.0), 0.284) is None  # on the wall, nearest free > 0.15 m


def test_corner_cutting() -> None:
    free = np.ones((5, 5), bool)
    free[1, 2] = False  # blocks (2,1)
    free[2, 1] = False
    path = astar(free, (0, 0), (4, 4))
    assert path is not None
    for a, b in zip(path, path[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            assert free[a[1], b[0]] and free[b[1], a[0]]


def test_deterministic_and_headings() -> None:
    occ = two_rooms()
    a = plan_path(occ, ORIGIN, RES, (1.0, 3.0), (5.0, 3.0), 0.284)
    assert a == plan_path(occ, ORIGIN, RES, (1.0, 3.0), (5.0, 3.0), 0.284)
    h = segment_headings(a, 1.5, start_yaw=0.0)
    assert [t[2] for t in h] == [0.0] * (len(a) - 1) + [1.5]
    assert all(t[2] == 1.5 for t in segment_headings(a, 1.5))


def test_runtime_house_size() -> None:
    occ = grid(9.0, 7.0)
    assert occ.shape == (140, 180)
    for x in (3.0, 6.0):
        box(occ, x - 0.05, 0.0, x + 0.05, 7.0)
        box(occ, x - 0.05, 5.0, x + 0.05, 6.0, False)
        box(occ, x - 0.05, 1.0, x + 0.05, 2.0, False)
    box(occ, 0.0, 3.45, 9.0, 3.55)
    box(occ, 3.5, 3.45, 4.5, 3.55, False)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        p = plan_path(occ, ORIGIN, RES, (0.5, 6.5), (8.5, 0.5), ROBOT_RADIUS)
        ts.append(time.perf_counter() - t)
        assert p is not None
    print(f"runtime 180x140: worst {max(ts) * 1e3:.0f} ms, median {sorted(ts)[2] * 1e3:.0f} ms")
    assert max(ts) < 0.2


# ---- demo on the real layout (test-only rasteriser; the official known map is sim/world/occupancy.py) -----------------

def raster_layout() -> np.ndarray:
    lay = yaml.safe_load((ROOT / "sim/world/layout.yaml").read_text())
    t = lay["wall"]["thickness"]
    occ = np.zeros((round(7.0 / RES), round(9.0 / RES)), bool)
    for r in lay["rooms"].values():
        x0, y0, x1, y1 = r["rect"]
        for x in (x0, x1):
            box(occ, max(x - t / 2, 0), y0, min(x + t / 2, 9.0), y1)
        for y in (y0, y1):
            box(occ, x0, max(y - t / 2, 0), x1, min(y + t / 2, 7.0))
    w = lay["door_width"]
    for d in lay["doors"].values():
        (ax, v), = d["wall"].items()
        c = d["center"]
        if ax == "x":
            box(occ, v - t / 2, c - w / 2, v + t / 2, c + w / 2, False)
        else:
            box(occ, c - w / 2, v - t / 2, c + w / 2, v + t / 2, False)
    for f in lay["furniture"]:
        cx, cy, sx, sy, _ = f["box"]
        box(occ, cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)
    return occ


def test_demo_layout_png() -> None:
    from PIL import Image, ImageDraw

    occ = raster_layout()
    blocked = inflate(occ, ROBOT_RADIUS + 0.02, RES)
    start = (4.5, 5.25)
    goals = {"kitchen_table": (7.55, 1.95), "storage": (7.5, 5.25)}
    s = 6
    px = np.full(occ.shape + (3,), 255, np.uint8)
    px[blocked] = (255, 205, 205)
    px[occ] = (60, 60, 60)
    img = Image.fromarray(px[::-1]).resize((occ.shape[1] * s, occ.shape[0] * s), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for name, g in goals.items():
        p = plan_path(occ, ORIGIN, RES, start, g, ROBOT_RADIUS)
        assert p is not None, name
        assert_clear(occ, p, ROBOT_RADIUS + 0.02)
        print(f"{name}: {len(p)} waypoints, length {path_length(p):.2f} m")
        pts = [(x / RES * s, img.height - y / RES * s) for x, y in p]
        d.line(pts, fill=(0, 90, 220), width=3)
        for q in pts:
            d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=(220, 40, 40))
    out = ROOT / "runs/1n"
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "astar_demo.png")
