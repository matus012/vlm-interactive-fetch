"""A* planner on a bool occupancy grid (numpy only).

Conventions: grids are indexed ``occ[iy, ix]`` (True = occupied); cells are ``(ix, iy)`` tuples; the world position of
cell centre (ix, iy) is ``origin + (ix + 0.5, iy + 0.5) * res``. Metres / radians.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

Cell = tuple[int, int]
Point = tuple[float, float]
_SQ2 = math.sqrt(2.0)
_NBRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, _SQ2), (1, -1, _SQ2), (-1, 1, _SQ2), (-1, -1, _SQ2)]


def inflate(occ: np.ndarray, radius_m: float, res: float) -> np.ndarray:
    """Disk dilation of ``occ`` by ``radius_m`` (cells whose centre is within radius of an occupied centre)."""
    r = radius_m / res
    n = int(math.floor(r))
    out = occ.copy()
    h, w = occ.shape
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r + 1e-9:
                ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
                xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
                out[yd, xd] |= occ[ys, xs]
    return out


def _snap(free: np.ndarray, c: Cell, max_cells: int) -> Cell | None:
    h, w = free.shape
    x, y = c
    if 0 <= x < w and 0 <= y < h and free[y, x]:
        return c
    best: tuple[float, int, int] | None = None
    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            d = math.hypot(dx, dy)
            xx, yy = x + dx, y + dy
            if d <= max_cells and 0 <= xx < w and 0 <= yy < h and free[yy, xx] and (best is None or (d, yy, xx) < best):
                best = (d, yy, xx)
    return None if best is None else (best[2], best[1])


def astar(free: np.ndarray, start: Cell, goal: Cell, allowed: np.ndarray | None = None,
          snap_cells: int = 3) -> list[Cell] | None:
    """8-connected A* (octile heuristic, no diagonal through an occupied orthogonal neighbour) over ``free``
    (True = traversable; pass the *inflated* grid, inverted). ``allowed`` further restricts traversable cells.
    Start/goal in a blocked cell snap to the nearest free cell within ``snap_cells`` (3 = 0.15 m at 5 cm). Ties break
    on (f, h, iy, ix), so output is deterministic. Returns the cell path incl. both ends, or None."""
    g_ok = free if allowed is None else free & allowed
    s, g = _snap(g_ok, start, snap_cells), _snap(g_ok, goal, snap_cells)
    if s is None or g is None:
        return None
    h_, w_ = g_ok.shape

    def hfun(x: int, y: int) -> float:
        dx, dy = abs(x - g[0]), abs(y - g[1])
        return (dx + dy) + (_SQ2 - 2.0) * min(dx, dy)

    best = {s: 0.0}
    parent: dict[Cell, Cell] = {}
    heap = [(hfun(*s), hfun(*s), s[1], s[0])]
    done: set[Cell] = set()
    while heap:
        _, _, y, x = heapq.heappop(heap)
        c = (x, y)
        if c in done:
            continue
        if c == g:
            path = [c]
            while c in parent:
                c = parent[c]
                path.append(c)
            return path[::-1]
        done.add(c)
        gc = best[c]
        for dx, dy, cost in _NBRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w_ and 0 <= ny < h_) or not g_ok[ny, nx]:
                continue
            if dx and dy and not (g_ok[y, nx] and g_ok[ny, x]):
                continue
            n = (nx, ny)
            ng = gc + cost
            if ng < best.get(n, math.inf) - 1e-12:
                best[n] = ng
                parent[n] = c
                hn = hfun(nx, ny)
                heapq.heappush(heap, (ng + hn, hn, ny, nx))
    return None


def _to_world(c: Cell, origin_xy: Point, res: float) -> Point:
    return (origin_xy[0] + (c[0] + 0.5) * res, origin_xy[1] + (c[1] + 0.5) * res)


def _to_cell(p: Point, origin_xy: Point, res: float) -> Cell:
    return (int(math.floor((p[0] - origin_xy[0]) / res)), int(math.floor((p[1] - origin_xy[1]) / res)))


def _los(blocked: np.ndarray, a: Point, b: Point, origin_xy: Point, res: float) -> bool:
    """True if the segment a-b only touches free cells of the inflated grid (sampled every res/10)."""
    n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (res / 10.0))) + 1)
    t = np.linspace(0.0, 1.0, n)
    ix = np.floor((a[0] + t * (b[0] - a[0]) - origin_xy[0]) / res).astype(int)
    iy = np.floor((a[1] + t * (b[1] - a[1]) - origin_xy[1]) / res).astype(int)
    h, w = blocked.shape
    if (ix < 0).any() or (iy < 0).any() or (ix >= w).any() or (iy >= h).any():
        return False
    return not blocked[iy, ix].any()


def _simplify(pts: list[Point], blocked: np.ndarray, origin_xy: Point, res: float) -> list[Point]:
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _los(blocked, pts[i], pts[j], origin_xy, res):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan_path(occ: np.ndarray, origin_xy: Point, res: float, start_xy: Point, goal_xy: Point,
              robot_radius: float, clearance: float = 0.02, via_cells: np.ndarray | None = None,
              via_xy: Point | None = None) -> list[Point] | None:
    """Plan start -> goal on ``occ`` inflated by ``robot_radius + clearance``; return a few straight-segment waypoints
    (world metres, first = start, last = goal) or None.

    Clearance is NOT continuous-exact: inflation is cell centre to cell centre and the line-of-sight test accepts any
    point of a free cell, so the polyline is only guaranteed ``robot_radius + clearance - res*sqrt(2)`` from occupied
    geometry. Callers that need a true clearance c pass ``clearance = c + res*sqrt(2)`` (``sim.nav.follow`` does:
    ``CLEARANCE + DISC_MARGIN``; G1-reach round 1 grazed door jambs without it).

    Endpoints: if the start/goal cell is free it is used as given; if it is inside the inflated obstacles it snaps to
    the nearest free cell within 0.15 m and the path begins/ends at that cell's centre; else None.

    ``via_cells`` (bool mask, same shape as occ, e.g. a room's cells): the path is forced through that room. Rule: if
    the start or goal is already inside the mask, nothing changes. Otherwise the via point is the free cell of the
    mask nearest the mask's bounding-box centre (candidates tried in that order, up to 8, until both legs are
    plannable); the path is start -> via point -> goal, each leg simplified separately so the via point is kept.

    ``via_xy`` (a world point, e.g. a door centre; task 003 R3): the path is start -> via_xy -> goal, two legs planned
    and simplified separately, so the via point is always a waypoint (snapped like an endpoint). Wins over via_cells."""
    blocked = inflate(occ, robot_radius + clearance, res)
    free = ~blocked
    snap = int(round(0.15 / res))
    sc, gc = _to_cell(start_xy, origin_xy, res), _to_cell(goal_xy, origin_xy, res)

    def leg(a: Cell, b: Cell, pa: Point | None, pb: Point | None) -> list[Point] | None:
        cp = astar(free, a, b, snap_cells=snap)
        if cp is None:
            return None
        pts = [_to_world(c, origin_xy, res) for c in cp]
        if pa is not None and _to_cell(pa, origin_xy, res) == cp[0]:
            pts[0] = pa
        if pb is not None and _to_cell(pb, origin_xy, res) == cp[-1]:
            pts[-1] = pb
        return _simplify(pts, blocked, origin_xy, res)

    if via_xy is not None:
        vc = _to_cell(via_xy, origin_xy, res)
        a, b = leg(sc, vc, start_xy, via_xy), leg(vc, gc, via_xy, goal_xy)
        return None if a is None or b is None else a + b[1:]
    if via_cells is None or via_cells[sc[1], sc[0]] or via_cells[gc[1], gc[0]]:
        return leg(sc, gc, start_xy, goal_xy)

    ys, xs = np.nonzero(via_cells & free)
    if len(xs) == 0:
        return None
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    order = np.lexsort((xs, ys, (xs - cx) ** 2 + (ys - cy) ** 2))[:8]
    for k in order:
        v = (int(xs[k]), int(ys[k]))
        a, b = leg(sc, v, start_xy, None), leg(v, gc, None, goal_xy)
        if a is not None and b is not None:
            return a + b[1:]
    return None


def path_length(path: list[Point]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def segment_headings(path: list[Point], final_yaw: float, start_yaw: float | None = None
                     ) -> list[tuple[float, float, float]]:
    """Setpoints (x, y, yaw) for the holonomic base, one per waypoint (path[0] included, so the follower can
    start with a rotate-in-place if needed).

    Rule: hold ``start_yaw`` (the current heading) while translating through intermediate waypoints, and take
    ``final_yaw`` at the last waypoint (the follower rotates there, shortest way). With ``start_yaw=None`` the base
    turns to ``final_yaw`` straight away and holds it for the whole path. Yaw is never tied to the motion
    direction: the base is holonomic, and a fixed yaw keeps the mast camera and arm orientation predictable."""
    hold = final_yaw if start_yaw is None else start_yaw
    n = len(path)
    return [(x, y, final_yaw if i == n - 1 else hold) for i, (x, y) in enumerate(path)]
