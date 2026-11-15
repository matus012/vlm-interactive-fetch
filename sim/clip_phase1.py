"""Phase 1 clip (task 002 Done): the house, the path follower and one look, rendered from the run itself.

    MUJOCO_GL=egl python -m sim.clip_phase1 --out showcase/phase1_world.mp4 --record runs/1g/clip_record.json

Rule (written before any render, task 002 / honest-demo-from-evaluator-records): seed 0, blocker none; the robot
starts at the hall room point (set_base + LOOK_ARM_QPOS + 0.5 s settle, as in the G1-reach tries) and drives
hall -> kitchen table approach with go_to (the same code as the gate), then go_to the kitchen room point, then turns
through the 4 look headings 0/90/180/270 deg there (drive_yaw's call: drive_base to the same x, y and the heading,
default settle; drive_yaw itself takes no step callback, so the call is repeated here with one), showing the
mast-camera photo of each heading. One run, no retakes; whatever go_to returns is what the captions say.

Layout 1280x720: top band = task line; left = the world's fixed `overview` camera (top-down) with the planned path
drawn as scene-only markers (not in the mast view, not in the physics); right = live mast camera, below it the 4 look
photos as they are taken; bottom band = current step and the go_to result (reached, errors, contacts) copied from the
result dict. Sim time runs at SPEED x (badge on screen); the photo holds are marked "hold".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.body import base_pose, drive_base
from sim.look import HEADINGS_DEG
from sim.constants import ROBOT_RADIUS
from sim.nav.astar import plan_path
from sim.nav.follow import CLEARANCE, DISC_MARGIN, go_to
from sim.reach_check import compile_house, pose_of, start_state
from sim.robot import MAST_CAM
from sim.world.generate import generate, load_places
from sim.world.occupancy import known_map

W, H, FPS, SPEED = 1280, 720, 30, 4.0
OV_W, OV_H = 840, 630  # overview panel (4:3)
MAST_W, MAST_H = 440, 330
THUMB_W, THUMB_H = 200, 150
HOLD_S = 1.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TASK = "Phase 1: drive hall -> kitchen table (go_to on the known map), then look at 4 headings in the kitchen"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def _add_path(scn: mujoco.MjvScene, path: list[tuple[float, float]], rgba: tuple[float, ...]) -> None:
    for a, b in zip(path, path[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.03, np.array([*a, 0.05]), np.array([*b, 0.05]))
        scn.ngeom += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    a = ap.parse_args()
    w = generate(seed=0, blocker="none")
    m = compile_house(w)
    grid = known_map(w.layout)
    P = {p["name"]: p for p in load_places()}
    d = start_state(m, pose_of(P["hall"]))

    r_ov = mujoco.Renderer(m, OV_H, OV_W)
    r_mast = mujoco.Renderer(m, MAST_H, MAST_W)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    f_big, f_mid, f_small, f_res = _font(22), _font(18), _font(15), _font(13)
    st: dict[str, Any] = {"step": "start: hall room point", "results": [], "paths": [], "thumbs": [], "next_t": d.time,
                          "n": 0, "badge": f"sim time x{SPEED:g}"}

    def frame(dd: mujoco.MjData) -> np.ndarray:
        r_ov.update_scene(dd, camera="overview")
        for i, p in enumerate(st["paths"]):
            _add_path(r_ov.scene, p, (0.1, 0.8, 1.0, 0.8) if i == len(st["paths"]) - 1 else (0.6, 0.6, 0.6, 0.6))
        ov = r_ov.render()
        r_mast.update_scene(dd, camera=MAST_CAM)
        mast = r_mast.render()
        img = Image.new("RGB", (W, H), (18, 18, 22))
        img.paste(Image.fromarray(ov), (0, 45))
        img.paste(Image.fromarray(mast), (OV_W, 45))
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), TASK, font=f_big, fill=(240, 240, 240))
        dr.text((10, 50), "overview camera (fixed, top-down); planned path in cyan", font=f_small, fill=(255, 255, 0))
        dr.text((OV_W + 6, 50), "mast camera (live)", font=f_small, fill=(255, 255, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        for i, (hdg, th) in enumerate(st["thumbs"]):
            x, y = OV_W + (i % 2) * THUMB_W, 45 + MAST_H + (i // 2) * THUMB_H
            img.paste(Image.fromarray(th), (x, y))
            dr.text((x + 4, y + 2), f"look {hdg:g} deg", font=f_small, fill=(255, 255, 0), stroke_width=2,
                    stroke_fill=(0, 0, 0))
        x, y, th = base_pose(m, dd)
        dr.text((10, 688), f"{st['step']}   |   base x {x:.2f} y {y:.2f} yaw {int(round(math.degrees(th)))} deg   |   "
                           f"t_sim {dd.time:.1f} s", font=f_mid, fill=(240, 240, 240))
        for i, line in enumerate(st["results"]):  # go_to results, inside the overview panel's bottom margin
            dr.text((10, 620 + 22 * i), line, font=f_res, fill=(160, 255, 160))
        dr.text((OV_W - 190, 50), st["badge"], font=f_mid, fill=(255, 140, 0))
        return np.asarray(img)

    def cb(dd: mujoco.MjData) -> None:
        if dd.time >= st["next_t"]:
            writer.append_data(frame(dd))
            st["n"] += 1
            st["next_t"] += SPEED / FPS

    def hold(seconds: float) -> None:
        st["badge"] = "hold (sim paused)"
        img = frame(d)
        for _ in range(int(seconds * FPS)):
            writer.append_data(img)
            st["n"] += 1
        st["badge"] = f"sim time x{SPEED:g}"

    record: dict[str, Any] = {"rule": __doc__.split("Rule")[1].split("Layout")[0].strip(), "seed": 0,
                              "blocker": "none", "speed": SPEED, "fps": FPS, "legs": [], "looks": []}
    hold(1.5)
    for goal in ("kitchen table", "kitchen"):
        q = base_pose(m, d)  # draw the path go_to will plan (same call, same arguments; asserted equal below)
        st["paths"].append(plan_path(grid.occ, grid.origin, grid.res, (float(q[0]), float(q[1])),
                                     pose_of(P[goal])[:2], ROBOT_RADIUS, CLEARANCE + DISC_MARGIN))
        st["step"] = f"go_to('{goal}')"
        r = go_to(m, d, grid, pose_of(P[goal]), cb=cb)
        assert r["waypoints"] == st["paths"][-1], "drawn path != go_to's path"
        walls = sum(not k.startswith("box_") for k in r["contacts"])
        st["results"].append(f"{goal}: reached={r['reached']} | err {r['pos_err_m'] * 1000:.2f} mm, "
                             f"{r['yaw_err_deg']:.3f} deg (bar 50 mm, 3 deg) | {r['path_len_m']:.2f} m, "
                             f"{r['sim_s']:.1f} s sim | contacts: wall/furn {walls}, box {len(r['contacts']) - walls}")
        record["legs"].append({"goal": goal, **{k: v for k, v in r.items()}})
        hold(1.0)
    for hdg in HEADINGS_DEG:
        st["step"] = f"look: turn to {hdg:g} deg (drive_base, same x, y)"
        x, y, _ = base_pose(m, d)
        e = drive_base(m, d, (float(x), float(y), math.radians(hdg)), cb=cb)
        r_mast.update_scene(d, camera=MAST_CAM)
        st["thumbs"].append((hdg, np.asarray(Image.fromarray(r_mast.render()).resize((THUMB_W, THUMB_H)))))
        st["step"] = f"look photo {hdg:g} deg taken (yaw err {e['yaw_err_deg']:.3f} deg)"
        record["looks"].append({"heading_deg": hdg, **e, "t_sim": d.time})
        hold(HOLD_S)
    st["step"] = "done: 2 go_to legs + 4 look photos"
    hold(2.0)
    writer.close()
    r_ov.close()
    r_mast.close()
    record.update(frames=st["n"], duration_s=st["n"] / FPS, t_sim_end=d.time)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(f"{a.out}: {st['n']} frames, {st['n'] / FPS:.1f} s; record -> {a.record}")


if __name__ == "__main__":
    main()
