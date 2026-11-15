"""Perceived locate (sim/perception.py, task 006 item 1) and its truth-free static check (item 3, module part)."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from sim import perception as pp
from sim.constants import BOX_HALF

SRC = Path(pp.__file__)
ROOTS = ("locate_from_rgbd", "locate_photos")
BANNED_NAMES = {"mujoco", "true_pose", "MjData", "MjModel"}
BANNED_ATTRS = {"xpos", "xquat", "xmat", "xipos", "ximat", "qpos", "body", "geom", "geom_xpos", "true_pose",
                "body_of", "enable_segmentation_rendering"}


def truth_reads(source: str, roots: tuple[str, ...] = ROOTS) -> tuple[set[str], list[str]]:
    """AST scan: the module functions reachable from `roots` by direct calls (visited set) and every truth read in
    them: the sim module, a body / geom pose attribute, true_pose, or anything named / keyed 'seg' or 'truth'."""
    funcs = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    seen, todo, hits = set(), list(roots), []
    while todo:
        f = todo.pop()
        if f in seen or f not in funcs:
            continue
        seen.add(f)
        for node in ast.walk(funcs[f]):
            word = None
            if isinstance(node, ast.Name):
                word = node.id
                if node.id in funcs:
                    todo.append(node.id)
                if node.id in BANNED_NAMES:
                    hits.append(f"{f}: name {node.id}")
            elif isinstance(node, ast.Attribute):
                word = node.attr
                if node.attr in BANNED_ATTRS:
                    hits.append(f"{f}: attribute .{node.attr}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                word = node.value
            elif isinstance(node, ast.arg):
                word = node.arg
            if word is not None and ("seg" in word.lower() or "truth" in word.lower()):
                hits.append(f"{f}: '{word}'")
    return seen, hits


def test_locate_reads_no_truth() -> None:
    seen, hits = truth_reads(SRC.read_text())
    # the scan is not vacuous: the whole pipeline is in the call graph
    assert {"_locate", "colour_mask", "rgb_to_hsv", "backproject", "reject_outliers", "fit_box", "_extents",
            "_place"} <= seen, seen
    assert "mast_photos" not in seen
    assert hits == []


MUTANTS = {
    "box pose in the fit": ("    top = float(P[:, 2].max())\n",
                            "    top = float(_SIM_D.body('box_red').xpos[2]) + 0.06\n"),
    "true_pose in the entry": ("    return _locate([(p[\"rgb\"]",
                               "    _ = true_pose(colour)\n    return _locate([(p[\"rgb\"]"),
    "segmentation from the photo dict": ("    masks = [colour_mask(rgb, colour) for rgb, *_ in views]\n",
                                         "    masks = [colour_mask(rgb, colour) for rgb, *_ in views]\n"
                                         "    masks = [m & (PHOTOS[0]['seg'] >= 0) for m in masks]\n"),
    "sim data in a helper": ("def _place(lo: float, hi: float, length: float, cam: float) -> float:\n",
                             "def _place(lo: float, hi: float, length: float, cam: float) -> float:\n"
                             "    mujoco.mj_forward(_M, _D)\n"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_truth_leak_mutant_is_caught(name: str, tmp_path: Path) -> None:
    old, new = MUTANTS[name]
    src = SRC.read_text()
    assert src.count(old) == 1, name
    p = tmp_path / "perception_mutant.py"
    p.write_text(src.replace(old, new))
    _, hits = truth_reads(p.read_text())
    assert hits, name


def _box_face_points(c: np.ndarray, yaw: float, cam: np.ndarray, step: float = 0.004) -> np.ndarray:
    """Points on the faces of an upright box that face the camera (top + up to two sides), on a regular grid."""
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    h = np.array(BOX_HALF)
    pts = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = R[:, ax] * sgn
            fc = c + n * h[ax]
            if n @ (cam - fc) <= 0:
                continue
            a, b = [k for k in range(3) if k != ax]
            ga, gb = np.meshgrid(np.arange(-h[a], h[a] + 1e-9, step), np.arange(-h[b], h[b] + 1e-9, step))
            pts.append(fc + ga.reshape(-1, 1) * R[:, a] + gb.reshape(-1, 1) * R[:, b])
    return np.vstack(pts)


@pytest.mark.parametrize("yaw", [0.0, 0.3, 1.0, 1.5708, 2.4, 3.1])
@pytest.mark.parametrize("cam", [(-1.5, 0.2, 1.3), (0.4, -2.5, 1.3), (2.0, 2.0, 1.3)])
def test_fit_box_recovers_pose(yaw: float, cam: tuple[float, float, float]) -> None:
    c = np.array([0.1, -0.05, 0.06])
    cam_pos = np.array(cam)
    P = _box_face_points(c, yaw, cam_pos)
    centre, size, est = pp.fit_box(pp.reject_outliers(P), cam_pos)
    assert np.linalg.norm(centre - c) < 0.005
    e = (est - yaw) % math.pi
    assert math.degrees(min(e, math.pi - e)) < 1.0
    assert abs(size[2] - 0.12) < 0.005


def test_fit_uses_known_size_when_far_side_is_hidden() -> None:
    """Only the camera-facing long side (no top): the centre is half the box depth behind it."""
    c, yaw, cam = np.array([0.0, 0.0, 0.06]), 0.0, np.array([-3.0, 0.0, 0.10])
    P = _box_face_points(c, yaw, cam)
    P = P[np.abs(P[:, 0] + BOX_HALF[0]) < 1e-9]  # the -x face only
    P = np.vstack([P, [[-BOX_HALF[0] + 0.001, 0.0, 0.12]] * 3])  # 3 top-edge points so the top height is found
    centre, _, est = pp.fit_box(P, cam)
    assert abs(centre[0] - c[0]) < 0.005 and abs(centre[1] - c[1]) < 0.005


def test_projection_round_trip() -> None:
    K = np.array([[320.0, 0, 320], [0, 320.0, 240], [0, 0, 1]])
    th = math.radians(35)
    R = np.column_stack([[0, -1, 0], [math.sin(th), 0, math.cos(th)], [-math.cos(th), 0, math.sin(th)]])  # mast
    pos = np.array([0.0, 0.0, 1.3])
    depth = np.full((480, 640), 2.0, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[100, 200] = mask[400, 600] = True
    P = pp.backproject(depth, K, pos, R, mask)
    uv = pp.project(P, K, pos, R)
    assert np.allclose(uv, [[200.5, 100.5], [600.5, 400.5]], atol=1e-6)


def test_not_in_view_on_empty_photo() -> None:
    rgb = np.full((480, 640, 3), 128, np.uint8)
    out = pp.locate_from_rgbd(rgb, np.ones((480, 640), np.float32), np.eye(3), (np.zeros(3), np.eye(3)), "red")
    assert out["in_view"] is False and out["pos"] is None and out["mask_px"] == 0


def test_locate_in_the_house() -> None:
    """End to end on a development-range house (seed 9752, never a scored seed): look photos from the living-room
    room point; the blue box on the coffee table within the G5-pose tolerances. (sim/photo.py's small scene is not
    used: its Menagerie floor is reflective, so the mask also catches the box's mirror image.)"""
    from sim.pose_check import put_robot, seed_state, yaw_err_deg

    _, m, d, truth = seed_state(9752)
    put_robot(m, d, {"x": 2.25, "y": 1.75})
    res = pp.locate_photos(pp.mast_photos(m, d), "blue")
    t = truth["blue"]
    assert t["support"] == "coffee table" and res["in_view"]
    assert np.linalg.norm(res["pos"] - t["pos"]) <= 0.02
    assert yaw_err_deg(res["yaw"], t["yaw"]) <= 10.0
