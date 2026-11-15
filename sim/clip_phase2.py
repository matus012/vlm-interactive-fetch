"""Phase 2 clip (task 003): one correct blocked-path script run, rendered from the run itself.

    MUJOCO_GL=egl python -m sim.clip_phase2 --out showcase/phase2_skills.mp4 --check runs/clip2/replay_check.txt

Rule: the pre-registered rule (2026-11-13-003-clip-rule) (seed 1001, runs/clip2/selection.txt). The script is re-run exactly as
sim/gate2.py `script_trial` runs it (make_task -> script_truth -> SimRobot -> Runtime -> CORRECT["blocked"] -> score);
frames come from wrapping mujoco.mj_step (as gate2.StepWatch does), so the adapter is untouched. The run's call list
(name, args, result, ok, sim_s) is compared with the counted log.jsonl; any difference -> no clip is written (exit 1).
Uniform speed-up SPEED x for the whole run (badge on screen), then a static end card.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.runtime import Runtime
from agent.scorer import score
from agent.scripts import CORRECT
from agent.skills import Skills
from sim.adapter import SimRobot
from sim.robot import MAST_CAM
from sim.tasks import make_task, script_truth

SEED = 1001
LOG = Path("runs/gate2r3/script/blocked_s1001_-/log.jsonl")
W, H, FPS, SPEED = 1280, 720, 30, 7.0
OV_W, OV_H = 840, 630
MAST_W, MAST_H = 440, 330
TOP = 45
END_S = 3.5
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CONTEXT = "blocked-path scripts, round 3: 9/10 passed (seed 1013 failed: its push stalled part-way)"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def fmt(name: str, args: dict) -> str:
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    a = ap.parse_args()
    task = make_task(SEED, "blocked")
    truth = script_truth(task)
    robot = SimRobot(SEED, task["blocker"])
    m, d = robot.m, robot.d
    tdir = Path(tempfile.mkdtemp())
    rt = Runtime(Skills(robot), robot, task, tdir / "log.jsonl")

    r_ov, r_mast = mujoco.Renderer(m, OV_H, OV_W), mujoco.Renderer(m, MAST_H, MAST_W)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp.mp4")
    writer = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    f_big, f_mid, f_small, f_hist = font(22, True), font(19), font(15), font(14)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = m.cam_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overview")]
    cam.lookat[2] = 0
    cam.distance, cam.elevation, cam.azimuth = 10.4, -90, 90
    st: dict[str, Any] = {"cur": None, "hist": [], "next_t": d.time, "n": 0}

    def frame() -> np.ndarray:
        r_ov.update_scene(d, camera=cam)
        s = r_ov.scene  # scene-only halo on the target box (not in the physics, not in the mast view)
        g = s.geoms[s.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.22, 0, 0]), np.zeros(3), np.eye(3).ravel(),
                            np.array([1, 1, 0, 0.35], dtype=np.float32))
        g.pos[:] = d.body(robot.body_of[task["targets"][0]["object"]]).xpos
        s.ngeom += 1
        ov = r_ov.render()
        r_mast.update_scene(d, camera=MAST_CAM)
        mast = r_mast.render()
        img = Image.new("RGB", (W, H), (18, 18, 22))
        img.paste(Image.fromarray(ov), (0, TOP))
        img.paste(Image.fromarray(mast), (OV_W, TOP))
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), "Task: " + task["instruction"], font=f_big, fill=(240, 240, 240))
        dr.text((10, TOP + 4), "overview (top-down)  yellow halo = the red box, purple = blocker", font=f_small, fill=(255, 255, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        dr.text((OV_W + 6, TOP + 4), "mast camera (live)", font=f_small, fill=(255, 255, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        dr.text((OV_W - 150, TOP + 4), f"speed x{SPEED:g}", font=f_big, fill=(255, 140, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        dr.text((OV_W + 6, TOP + MAST_H + 6), "calls so far (name(args) -> result)", font=f_small, fill=(255, 255, 0))
        for i, (c, r) in enumerate(st["hist"][-7:]):
            dr.text((OV_W + 6, TOP + MAST_H + 32 + 36 * i), c, font=f_hist,
                    fill=(200, 200, 200))
            dr.text((OV_W + 26, TOP + MAST_H + 32 + 36 * i + 16), "-> " + r, font=f_hist, fill=(160, 255, 160))
        c, r = st["cur"] or ("(starting)", None)
        dr.rectangle((0, 675, W, H), fill=(30, 30, 40))
        dr.text((10, 680), "call: ", font=f_mid, fill=(255, 255, 0))
        dr.text((70, 680), c, font=f_mid, fill=(240, 240, 240))
        dr.text((640, 680), "result: " + (r if r is not None else "..."), font=f_mid,
                fill=(160, 255, 160) if r is not None else (255, 200, 120))
        dr.text((10, 702), f"sim t = {d.time:6.1f} s", font=f_small, fill=(170, 170, 170))
        return np.asarray(img)

    orig_step = mujoco.mj_step

    def step(mm: mujoco.MjModel, dd: mujoco.MjData, *args: Any, **kw: Any) -> None:
        orig_step(mm, dd, *args, **kw)
        if dd is d and dd.time >= st["next_t"]:
            writer.append_data(frame())
            st["n"] += 1
            st["next_t"] += SPEED / FPS

    orig_call = rt.call

    def call(name: str, args: dict | None = None):
        c = fmt(name, args or {})
        st["cur"] = (c, None)
        res = orig_call(name, args)
        st["cur"] = (c, res.text)
        st["hist"].append((c, res.text))
        writer.append_data(frame())
        st["n"] += 1
        return res

    rt.call = call
    mujoco.mj_step = step
    try:
        CORRECT["blocked"](rt, task, truth)
        snap = robot.snapshot(2.0)
    finally:
        mujoco.mj_step = orig_step
    sc = score(snap, task, len(rt.steps) + rt.n_refused)
    rt.finish(sc)

    # ---- replay check against the counted log
    logged = [json.loads(l) for l in LOG.read_text().splitlines()]
    logged = [s for s in logged if "step" in s]
    rows, same = [], len(logged) == len(rt.steps)
    for i in range(max(len(logged), len(rt.steps))):
        L = logged[i] if i < len(logged) else None
        R = rt.steps[i] if i < len(rt.steps) else None
        key = lambda s: None if s is None else (s["call"], s["args"], s["result"], s["ok"], round(s["sim_s"], 6))
        eq = key(L) == key(R)
        same &= eq
        rows.append(f"{i + 1:2d} {'MATCH' if eq else 'DIFF '} log: {key(L)}\n   {'' :5} run: {key(R)}")
    head = f"replay check seed {SEED}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(logged)} logged, {len(rt.steps)} re-run); " \
           f"compared (call, args, result, ok, sim_s); scorer pass={sc['pass']} failed={sc['failed']}"
    a.check.parent.mkdir(parents=True, exist_ok=True)
    a.check.write_text(head + "\n" + "\n".join(rows) + "\n")
    print(head)
    if not same:
        writer.close()
        tmp.unlink()
        sys.exit(1)

    # ---- end card
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    dr.text((60, 50), "Task: " + task["instruction"], font=font(30, True), fill=(240, 240, 240))
    dr.text((60, 120), f"Scorer verdict: {'PASS' if sc['pass'] else 'FAIL'}", font=font(44, True),
            fill=(120, 255, 120) if sc["pass"] else (255, 110, 110))
    dr.text((60, 200), "Rules checked:", font=font(26), fill=(255, 255, 0))
    expl = {"surface": "target box rests on the target surface (upright, still)", "wrong_box": "no other box on that surface",
            "held": "nothing held at the end", "steps": f"steps used {len(rt.steps)} <= limit {task['step_limit']}",
            "sim": "simulation healthy (no blow-up / NaN)"}
    for i, (k, v) in enumerate(sc["rules"].items()):
        dr.text((80, 245 + 40 * i), f"{'ok ' if v else 'FAIL'}  {k}: {expl[k]}", font=font(24),
                fill=(160, 255, 160) if v else (255, 110, 110))
    dr.text((60, 500), f"Run: {len(rt.steps)} calls, {d.time:.0f} s of sim time, shown at x{SPEED:g} (uniform, nothing cut)",
            font=font(22), fill=(200, 200, 200))
    dr.text((60, 600), CONTEXT, font=font(24), fill=(255, 190, 90))
    arr = np.asarray(img)
    for _ in range(int(END_S * FPS)):
        writer.append_data(arr)
        st["n"] += 1
    writer.close()
    tmp.replace(a.out)
    print(f"{a.out}: {st['n']} frames, {st['n'] / FPS:.1f} s, sim {d.time:.1f} s, {a.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
