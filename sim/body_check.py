"""Phase 0 / 0a body check: G0a-cam, G0a-img, G0a-base, G0a-grasp (+ optional clip). Writes body_check.json + images.

    python -m sim.body_check --out runs/0a/<tag> [--clip showcase/0a_body.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.body import BOX, BOX_REL, body_scene, drive_base, grasp_and_lift, reset
from sim.robot import MAST_CAM, camera_intrinsics, render_rgbd

GOALS_DEG = ((1.0, 0.5, 90.0), (-0.8, 1.2, -45.0), (0.3, -1.0, 180.0))
POS_TOL_M, YAW_TOL_DEG = 0.02, 2.0  # G0a-base bar
N_TRIALS = 5
CLIP_W, CLIP_H, CLIP_EVERY = 640, 480, 17  # 17 steps x 2 ms -> 29.4 fps


def steps_per_s(m: mujoco.MjModel, wall: float = 5.0) -> dict[str, float]:
    d = reset(m)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < wall:
        for _ in range(1000):
            mujoco.mj_step(m, d)
        n += 1000
    dt = time.perf_counter() - t0
    return {"steps": n, "wall_s": dt, "steps_per_s": n / dt, "realtime_x": n * m.opt.timestep / dt}


IMG_BOX_XY = (0.8, 0.45)  # front-left: the retracted arm hides the floor straight ahead of the mast


def mast_images(out: Path) -> dict[str, object]:
    m = body_scene(IMG_BOX_XY).compile()
    d = reset(m)
    rgb, depth = render_rgbd(m, d)
    far = m.vis.map.zfar * m.stat.extent
    sky = depth >= 0.99 * far
    Image.fromarray(rgb).save(out / "mast_rgb.png")
    np.save(out / "mast_depth.npy", depth.astype(np.float32))
    vis = np.clip(depth / 5.0, 0, 1)  # visualisation only: 0-5 m
    vis = (255 * (1 - vis)).astype(np.uint8)  # near = bright
    vis[sky] = 0
    Image.fromarray(vis).save(out / "mast_depth_vis.png")
    K = camera_intrinsics(m)
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "resolution": [int(v) for v in m.cam_resolution[m.camera(MAST_CAM).id]],
        "depth_finite": bool(np.isfinite(depth).all()), "far_plane_m": float(far),
        "sky_frac": float(sky.mean()), "depth_min_m": float(depth[~sky].min()), "depth_max_m": float(depth[~sky].max()),
        "img_box_xy": list(IMG_BOX_XY),
        "files": ["mast_rgb.png", "mast_depth.npy", "mast_depth_vis.png"],
    }


def grasp_trials(seed: int = 0) -> list[dict[str, object]]:
    """5 independent trials, fresh MjData each; box at the fixed spot with a small seeded jitter (+-1 cm, +-5 deg)."""
    rng = np.random.default_rng(seed)
    res = []
    for i in range(N_TRIALS):
        dx, dy = rng.uniform(-0.01, 0.01, 2)
        yaw = math.radians(rng.uniform(-5, 5))
        m = body_scene((BOX_REL[0] + dx, BOX_REL[1] + dy), yaw, table=False).compile()
        d = reset(m)
        r = grasp_and_lift(m, d)
        res.append({"trial": i, "box_dx_m": float(dx), "box_dy_m": float(dy), "box_yaw_deg": math.degrees(yaw), **r})
    return res


def drive_and_clip(out: Path, clip: Path | None) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Base to the 3 goals, then a grasp-and-lift of a box placed in front of goal 3. Optionally recorded."""
    gx, gy, gth = GOALS_DEG[-1]
    th = math.radians(gth)
    box_xy = (gx + math.cos(th) * BOX_REL[0] - math.sin(th) * BOX_REL[1],
              gy + math.sin(th) * BOX_REL[0] + math.cos(th) * BOX_REL[1])
    m = body_scene(box_xy, th).compile()
    d = reset(m)
    box0 = d.body(BOX).xpos.copy()
    cb, writer, renderer, frames = None, None, None, {}
    if clip is not None:
        import imageio.v2 as imageio

        clip.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(clip, fps=1.0 / (CLIP_EVERY * m.opt.timestep), codec="libx264", quality=7,
                                    macro_block_size=16)
        renderer = mujoco.Renderer(m, CLIP_H, CLIP_W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        wide = ((0.2, 0.1, 0.3), 4.3, 135.0, -35.0)
        close = ((box_xy[0] + 0.25, box_xy[1], 0.3), 1.5, 90.0, -15.0)  # side view of the grasp
        shot = {"view": wide, "n": 0}

        def cb(dd: mujoco.MjData) -> None:
            shot["n"] += 1
            if shot["n"] % CLIP_EVERY:
                return
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = shot["view"]
            renderer.update_scene(dd, camera=cam)
            img = renderer.render()
            writer.append_data(img)
            frames["last"] = img

    try:
        goals = []
        for (x, y, deg) in GOALS_DEG:
            e = drive_base(m, d, (x, y, math.radians(deg)), seconds=3.0, settle=1.0, cb=cb)
            if clip is not None:
                Image.fromarray(frames["last"]).save(out / f"clip_goal{len(goals) + 1}.png")
            goals.append({"goal": [x, y, deg], **e,
                          "pass": e["pos_err_m"] <= POS_TOL_M and e["yaw_err_deg"] <= YAW_TOL_DEG})
        moved = float(np.linalg.norm(d.body(BOX).xpos - box0))
        if clip is not None:
            shot["view"] = close
        g = grasp_and_lift(m, d, cb=cb)
        if clip is not None:
            Image.fromarray(frames["last"]).save(out / "clip_lifted.png")
        g["box_moved_during_drive_m"] = moved
        g["clip_s"] = d.time
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
    return goals, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = body_scene().compile()
    res: dict[str, object] = {
        "host": socket.gethostname(), "python": platform.python_version(), "mujoco": mujoco.__version__,
        "timestep_s": m.opt.timestep, "box_mass_kg": float(m.body_mass[m.body(BOX).id]),
        "box_friction": m.geom_friction[m.geom(BOX).id].tolist(), "box_placement": "floor",
    }
    res["mast"] = mast_images(a.out)
    res["G0a_cam_pass"] = abs(res["mast"]["fx"] - 320) <= 0.5 and abs(res["mast"]["fy"] - 320) <= 0.5
    goals, clip_grasp = drive_and_clip(a.out, a.clip)
    res["base_goals"] = goals
    res["G0a_base_pass"] = all(g["pass"] for g in goals)
    res["clip_grasp"] = clip_grasp
    res["clip"] = str(a.clip) if a.clip else None
    trials = grasp_trials()
    res["grasp_trials"] = trials
    res["G0a_grasp_n_pass"] = sum(t["pass"] for t in trials)
    res["G0a_grasp_pass"] = res["G0a_grasp_n_pass"] == N_TRIALS
    res["sim_speed"] = steps_per_s(m)
    (a.out / "body_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("host", "mujoco", "G0a_cam_pass", "G0a_base_pass", "G0a_grasp_n_pass")}
                     | {"steps_per_s": res["sim_speed"]["steps_per_s"], "depth": [res["mast"]["depth_min_m"],
                                                                                 res["mast"]["depth_max_m"]]}))


if __name__ == "__main__":
    main()
