"""One mast-camera photo of a small test scene: robot, 2 coloured fetch boxes on the floor, a 0.75 m table with a box.

    python -m sim.photo out.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.robot import render_rgbd, tidybot_spec

BOX_HALF = (0.03, 0.10, 0.06)  # 6 x 20 x 12 cm
TABLE_TOP = 0.75
TABLE_XY = (1.9, 0.0)  # robot frame; robot at the origin facing +x


def _box(spec: mujoco.MjSpec, name: str, xyz: tuple[float, float, float], yaw: float, rgba: list[float]) -> None:
    b = spec.worldbody.add_body(name=name, pos=list(xyz), euler=[0, 0, yaw])
    b.add_freejoint()
    b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF), mass=0.3, rgba=rgba)


def make_scene_spec() -> mujoco.MjSpec:
    spec = tidybot_spec()
    tx, ty = TABLE_XY
    t = spec.worldbody.add_body(name="table", pos=[tx, ty, 0])
    brown = [0.45, 0.28, 0.12, 1]
    t.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.45, 0.02],
               pos=[0, 0, TABLE_TOP - 0.02], rgba=brown)
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        t.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, (TABLE_TOP - 0.04) / 2],
                   pos=[sx * 0.46, sy * 0.41, (TABLE_TOP - 0.04) / 2], rgba=brown)
    _box(spec, "box_red", (0.75, 0.30, BOX_HALF[2]), 0.4, [0.9, 0.1, 0.1, 1])
    _box(spec, "box_blue", (0.95, -0.35, BOX_HALF[2]), -0.3, [0.1, 0.2, 0.9, 1])
    _box(spec, "box_yellow", (tx - 0.1, 0.1, TABLE_TOP + BOX_HALF[2]), 1.2, [0.95, 0.85, 0.1, 1])
    return spec


def load_test_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = make_scene_spec().compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("retract").id)
    for _ in range(500):  # let the boxes settle
        mujoco.mj_step(m, d)
    return m, d


def render_photo(path: Path) -> np.ndarray:
    m, d = load_test_scene()
    rgb, _ = render_rgbd(m, d)
    Image.fromarray(rgb).save(path)
    return rgb


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    img = render_photo(out)
    print(f"wrote {out} {img.shape} mean {img.mean():.1f}")
