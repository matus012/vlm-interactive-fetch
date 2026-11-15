"""Perceived `locate` (task 006 item 1): the named fetch box from one mast RGB + depth photo.

Offline only: nothing here is wired into the adapter or the skills yet (Phase 5 / task 006 does that).

`locate_from_rgbd(rgb, depth, K, cam_pose, colour)` (one photo) and `locate_photos(photos, colour)` (several, e.g.
the 4 look photos: in view = >= IN_VIEW_PX mask px in one of them; the fit uses the mask points of all of them, so a box
cut by the image edge between two headings is fitted whole - every dev failure before this was such a cut):
1. HSV mask for the colour (HSV_RANGES below; calibration in the docstring there). In view = >= IN_VIEW_PX mask px.
2. Back-projection of every mask pixel with the pinhole K (fx = fy = 320, Phase 0) and the camera world pose
   (pos, R with the camera axes as columns, MuJoCo convention: looks along -z, image up = +y). MuJoCo depth is planar
   camera-z in metres and exact (no noise model).
3. Outlier rejection: points farther than OUTLIER_R from the median point are dropped (twice, re-centred); then the
   top face height is the K-th highest z and points outside the box's known height band below it are dropped.
4. Model-based box fit on the ground-projected points (the box is upright, so every visible face - top and sides -
   projects inside its 6 x 20 cm footprint): yaw = the angle minimising the area of the trimmed (TRIM_PCT) xy extent
   rectangle (0.5 deg grid over [0, 90), then 0.05 deg refinement); the longer extent is the 20 cm axis. The centre
   uses the known size: along each box axis, if the seen extent covers the known length the midpoint is used, else
   the edge nearest the camera is taken as a true face and the centre is half a box length behind it. Centre z =
   top face - BOX_HALF[2]. `size` reports the measured extents; `yaw` is the 6 cm (body x) axis, in [0, pi).

Truth-free: `locate_from_rgbd`, `locate_photos` and their callees read only their arguments (tests/test_perception.py scans them).
`mast_photos` (render RGB + depth + camera pose from a sim state at the look headings) is for the offline harness
`sim/pose_check.py` only; it also returns segmentation, which the harness uses for selection and comparison.
"""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from sim.constants import BOX_HALF
from sim.look import HEADINGS_DEG, set_yaw_direct
from sim.robot import CAM_H, CAM_W, MAST_CAM, camera_intrinsics

IN_VIEW_PX = 100  # in view = >= 100 mask px (the old segmentation bar, agent.robot_api.PX_VISIBLE; task 006 item 1)

# HSV masks: colour -> (hue_lo_deg, hue_hi_deg, s_min, v_min); hue_lo > hue_hi wraps through 0 deg.
# Calibrated on development seeds 9750-9799 only, never on the scored 9800-9999 (`python -m sim.pose_check --seeds
# 9750:9800 --out runs/5s/calib --calibrate`, runs/5s/calib/hsv_stats.json): every other seed, every 3rd candidate
# position of the G5-pose rule, 4 headings; box px = segmentation px of each box geom, background = every 7th other px.
# - Box hue medians: red 0.0, blue 232.5, green 125.0, yellow 53.3 deg; 1-99 % quantiles within +-8 deg of the
#   median except red's 99 % (51 deg: blended edge px - seg marks a px whose colour is partly floor). Hue windows
#   are about +-15 deg (blue 215-250, yellow 45-62); the brown furniture, the most saturated background, is 24-28 deg.
# - Saturation: >= 98.0 % of every box's px have s >= 0.70 (the rest are blended edge px); background s 99.99 %
#   quantile 0.644 (brown furniture, s 0.64) over 43.8 M sampled px; only 315 sampled background px reach 0.70, at box
#   hues (edge blends just outside the seg silhouette). s_min = 0.70.
# - Value: box v 0.1 % quantile 0.11-0.18 (shaded faces); v_min = 0.10 only drops near-black px where hue is noise.
# Check on the dev views (runs/5s/dev): 0 in-view disagreements (mask >= 100 px vs seg >= 100 px) over 157 views x
# 4 photos x 4 colours.
HSV_RANGES: dict[str, tuple[float, float, float, float]] = {
    "red": (345.0, 15.0, 0.70, 0.10),
    "blue": (215.0, 250.0, 0.70, 0.10),
    "green": (110.0, 140.0, 0.70, 0.10),
    "yellow": (45.0, 62.0, 0.70, 0.10),
}

OUTLIER_R = 0.15  # m from the median point; the box's half diagonal is 0.12 m
TOP_K_FRAC = 0.02  # top face z = the K-th highest point z, K = max(3, 2 % of the points)
Z_BAND_MARGIN = 0.01  # m: keep z in [top - 2 * BOX_HALF[2] - margin, top + margin / 2]
TRIM_PCT = 0.5  # % trimmed at each end of the extents in the yaw search


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 HxWx3 -> hue (deg, [0, 360)), saturation, value ([0, 1]) arrays."""
    c = rgb.astype(np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(axis=-1)
    delta = v - c.min(axis=-1)
    s = np.where(v > 0, delta / np.maximum(v, 1e-9), 0.0)
    dd = np.maximum(delta, 1e-9)
    h = np.where(v == r, ((g - b) / dd) % 6.0, np.where(v == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0)) * 60.0
    return np.where(delta > 0, h, 0.0), s, v


def colour_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    h_lo, h_hi, s_min, v_min = HSV_RANGES[colour]
    h, s, v = rgb_to_hsv(rgb)
    hue = (h >= h_lo) & (h <= h_hi) if h_lo <= h_hi else (h >= h_lo) | (h <= h_hi)
    return hue & (s >= s_min) & (v >= v_min)


def backproject(depth: np.ndarray, K: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """World points (N x 3) of the mask pixels with valid depth. Pixel (row v, col u) centre is (u + 0.5, v + 0.5)."""
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0.0))
    z = depth[v, u].astype(np.float64)
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, -(v + 0.5 - K[1, 2]) / K[1, 1] * z, -z], axis=1)
    return cam_pos + pc @ cam_mat.T


def project(P: np.ndarray, K: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """World points (N x 3) -> pixel coordinates (N x 2, u = column, v = row, continuous; pixel i spans [i, i+1))."""
    pc = (P - cam_pos) @ cam_mat
    z = -pc[:, 2]
    return np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], -K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)


def reject_outliers(P: np.ndarray) -> np.ndarray:
    for _ in range(2):
        P = P[np.linalg.norm(P - np.median(P, axis=0), axis=1) <= OUTLIER_R]
    if len(P) == 0:
        return P
    zs = np.sort(P[:, 2])[::-1]
    top = zs[min(len(zs), max(3, int(TOP_K_FRAC * len(zs)))) - 1]
    keep = (P[:, 2] >= top - 2 * BOX_HALF[2] - Z_BAND_MARGIN) & (P[:, 2] <= top + Z_BAND_MARGIN / 2)
    return P[keep]


def _extents(xy: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trimmed extents of xy along (cos, sin)(th) [a] and its normal [b] for every angle in th: lo/hi arrays."""
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    a = c * xy[None, :, 0] + s * xy[None, :, 1]
    b = -s * xy[None, :, 0] + c * xy[None, :, 1]
    pa = np.percentile(a, [TRIM_PCT, 100 - TRIM_PCT], axis=1)
    pb = np.percentile(b, [TRIM_PCT, 100 - TRIM_PCT], axis=1)
    return pa[0], pa[1], pb[0], pb[1]


def _place(lo: float, hi: float, length: float, cam: float) -> float:
    """Centre coordinate on one box axis from the seen extent [lo, hi], the known length and the camera coordinate."""
    if hi - lo >= length or lo <= cam <= hi:
        return (lo + hi) / 2.0
    return lo + length / 2.0 if cam < lo else hi - length / 2.0


def fit_box(P: np.ndarray, cam_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre xyz, measured size [short, long, height], yaw of the short axis in [0, pi)) from inlier points."""
    top = float(P[:, 2].max())
    xy0 = P[:, :2].mean(axis=0)
    xy = P[:, :2] - xy0
    th = np.radians(np.arange(0.0, 90.0, 0.5))
    alo, ahi, blo, bhi = _extents(xy, th)
    t0 = float(th[np.argmin((ahi - alo) * (bhi - blo))])
    th = t0 + np.radians(np.arange(-0.5, 0.5001, 0.05))
    alo, ahi, blo, bhi = _extents(xy, th)
    i = int(np.argmin((ahi - alo) * (bhi - blo)))
    t, ea, eb = float(th[i]), float(ahi[i] - alo[i]), float(bhi[i] - blo[i])
    lo_a, hi_a, lo_b, hi_b = float(alo[i]), float(ahi[i]), float(blo[i]), float(bhi[i])
    if ea > eb:  # a is the long axis: rotate so that a is the short (body x) axis
        t += math.pi / 2.0
        lo_a, hi_a, lo_b, hi_b = lo_b, hi_b, -hi_a, -lo_a
        ea, eb = eb, ea
    ua, ub = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    cxy = cam_pos[:2] - xy0
    ca = _place(lo_a, hi_a, 2 * BOX_HALF[0], float(cxy @ ua))
    cb = _place(lo_b, hi_b, 2 * BOX_HALF[1], float(cxy @ ub))
    centre_xy = xy0 + ca * ua + cb * ub
    centre = np.array([centre_xy[0], centre_xy[1], top - BOX_HALF[2]])
    return centre, np.array([ea, eb, top - float(P[:, 2].min())]), t % math.pi


def _locate(views: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], colour: str
            ) -> dict[str, Any]:
    """views: (rgb, depth, K, cam_pos, cam_mat) per photo. In view = >= IN_VIEW_PX mask px in one photo; the fit
    uses the mask points of every photo (world frame, exact depth), with the camera of the photo with most mask px."""
    masks = [colour_mask(rgb, colour) for rgb, *_ in views]
    px = [int(mk.sum()) for mk in masks]
    i = int(np.argmax(px))
    out: dict[str, Any] = {"in_view": px[i] >= IN_VIEW_PX, "mask_px": px[i], "photo": i, "n_inliers": 0,
                           "pos": None, "size": None, "yaw": None}
    if px[i] < IN_VIEW_PX:
        return out
    P = reject_outliers(np.vstack([backproject(dp, K, cp, cm, mk) for (_, dp, K, cp, cm), mk in zip(views, masks)]))
    out["n_inliers"] = int(len(P))
    if len(P) < 3:
        out["in_view"] = False
        return out
    c, size, yaw = fit_box(P, views[i][3])
    out.update(pos=c, size=size, yaw=yaw)
    return out


def locate_from_rgbd(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray,
                     cam_pose: tuple[np.ndarray, np.ndarray], colour: str) -> dict[str, Any]:
    """The named box in one photo. cam_pose = (camera world position (3,), camera rotation 3x3, axes as columns).
    Returns in_view, mask_px, n_inliers, pos (xyz, m) / size (short, long, height, m) / yaw (rad, [0, pi)) or None."""
    return _locate([(rgb, depth, K, np.asarray(cam_pose[0], float), np.asarray(cam_pose[1], float).reshape(3, 3))],
                   colour)


def locate_photos(photos: list[dict[str, Any]], colour: str) -> dict[str, Any]:
    """The named box over several photos (keys rgb, depth, K, cam_pos, cam_mat), e.g. the 4 look photos: what an
    adapter `locate` over the current photos does. A box cut by the edge between two headings is fitted whole."""
    return _locate([(p["rgb"], p["depth"], p["K"], np.asarray(p["cam_pos"], float),
                     np.asarray(p["cam_mat"], float).reshape(3, 3)) for p in photos], colour)


def mast_photos(m: mujoco.MjModel, d: mujoco.MjData, headings_deg: tuple[float, ...] = HEADINGS_DEG,
                r: mujoco.Renderer | None = None) -> list[dict[str, Any]]:
    """HARNESS ONLY: mast photos at the given absolute base yaws (set directly, as look_px / G1-cover), each a dict
    heading, rgb, depth, seg (geom id, -1 none), K, cam_pos, cam_mat (3x3). The arm pose is the caller's.
    Same three passes as sim.look.render_look (RGB, depth, segmentation); the geom test compares with the int value of
    mjOBJ_GEOM (render_look compares with the enum object, elementwise in Python: 0.2 s per photo)."""
    K = camera_intrinsics(m)
    cid = m.camera(MAST_CAM).id
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    out = []
    own = r is None
    r = mujoco.Renderer(m, CAM_H, CAM_W) if own else r
    try:
        for h in headings_deg:
            set_yaw_direct(m, d, math.radians(h))
            r.update_scene(d, camera=MAST_CAM)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(d, camera=MAST_CAM)
            depth = r.render().astype(np.float32).copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=MAST_CAM)
            s = r.render()
            r.disable_segmentation_rendering()
            seg = np.where(s[..., 1] == geom, s[..., 0], -1).astype(np.int32)
            out.append({"heading": float(h), "rgb": rgb, "depth": depth, "seg": seg, "K": K,
                        "cam_pos": d.cam_xpos[cid].copy(), "cam_mat": d.cam_xmat[cid].reshape(3, 3).copy()})
    finally:
        if own:
            r.close()
    return out
