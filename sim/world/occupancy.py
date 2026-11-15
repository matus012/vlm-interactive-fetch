"""Known map: an occupancy grid rasterised from the layout's walls (door gaps open) and furniture footprints only.

Fetch boxes and blockers are NOT on it (the robot does not know them). Grid covers the house rectangle exactly
(origin = its lower-left corner), so the outer walls fill the border cells and all free space is indoors.
A cell is occupied if it overlaps a wall or footprint rectangle (conservative). `occ[iy, ix]`.

    python -m sim.world.occupancy --out runs/1w/map.png   # PNG: black = occupied, grey = inflated, white = free
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.constants import GRID_RES
from sim.world.generate import furniture_rect, load_layout, wall_rects

EPS = 1e-9


@dataclass
class Grid:
    origin: tuple[float, float]
    res: float
    occ: np.ndarray  # bool [ny, nx]

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """(ix, iy) of the cell containing (x, y)."""
        return int(math.floor((x - self.origin[0]) / self.res)), int(math.floor((y - self.origin[1]) / self.res))

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        """Cell centre."""
        return self.origin[0] + (ix + 0.5) * self.res, self.origin[1] + (iy + 0.5) * self.res

    def free(self, x: float, y: float) -> bool:
        ix, iy = self.world_to_cell(x, y)
        return 0 <= iy < self.occ.shape[0] and 0 <= ix < self.occ.shape[1] and not self.occ[iy, ix]

    def inflate(self, radius: float) -> Grid:
        """Occupied = within `radius` (cell centre to cell centre) of an occupied cell."""
        r = int(math.floor(radius / self.res + EPS))
        ny, nx = self.occ.shape
        pad = np.pad(self.occ, r, constant_values=True)
        out = np.zeros_like(self.occ)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if math.hypot(dx, dy) * self.res <= radius + EPS:
                    out |= pad[r + dy: r + dy + ny, r + dx: r + dx + nx]
        return Grid(self.origin, self.res, out)

    def components(self) -> np.ndarray:
        """4-connected labels of free cells (0 = occupied, 1.. = component id)."""
        lab = np.zeros(self.occ.shape, dtype=np.int32)
        ny, nx = self.occ.shape
        n = 0
        for sy, sx in zip(*np.nonzero(~self.occ)):
            if lab[sy, sx]:
                continue
            n += 1
            lab[sy, sx] = n
            q = deque([(sy, sx)])
            while q:
                y, x = q.popleft()
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < ny and 0 <= xx < nx and not self.occ[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        q.append((yy, xx))
        return lab

    def save_png(self, path: Path | str, inflated: Grid | None = None) -> None:
        from PIL import Image

        img = np.full(self.occ.shape, 255, np.uint8)
        if inflated is not None:
            img[inflated.occ] = 170
        img[self.occ] = 0
        Image.fromarray(img[::-1]).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(path)


def known_map(layout: dict[str, Any], res: float = GRID_RES) -> Grid:
    xs = [v for r in layout["rooms"].values() for v in (r["rect"][0], r["rect"][2])]
    ys = [v for r in layout["rooms"].values() for v in (r["rect"][1], r["rect"][3])]
    ox, oy = min(xs), min(ys)
    nx, ny = int(round((max(xs) - ox) / res)), int(round((max(ys) - oy) / res))
    occ = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x1, y1 in wall_rects(layout) + [furniture_rect(f) for f in layout["furniture"]]:
        i0, i1 = max(0, math.floor((x0 - ox) / res + EPS)), min(nx, math.ceil((x1 - ox) / res - EPS))
        j0, j1 = max(0, math.floor((y0 - oy) / res + EPS)), min(ny, math.ceil((y1 - oy) / res - EPS))
        occ[j0:j1, i0:i1] = True
    return Grid((ox, oy), res, occ)


def main() -> None:
    from sim.constants import ROBOT_RADIUS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    g = known_map(load_layout())
    gi = g.inflate(ROBOT_RADIUS + 0.02)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    g.save_png(a.out, gi)
    lab = gi.components()
    print(f"grid {g.occ.shape[1]}x{g.occ.shape[0]} @ {g.res} m, free {int((~g.occ).sum())}, inflated free "
          f"{int((~gi.occ).sum())}, inflated components {lab.max()} -> {a.out}")


if __name__ == "__main__":
    main()
