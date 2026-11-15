"""Phase 5 clip (task 006): selection per the pre-registered rule (2026-11-13-006-clip-rule), rendering by sim/clip_phase3.render
(perceived poses) with a locate card after each locate step: the mast photo(s) the perception used with the colour mask
(magenta) and the fitted box (green), and the caption "camera fit vs truth" (truth = the step's meta.eval, evaluation only).

    MUJOCO_GL=egl python -m sim.clip_phase5 [select|render]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sim import clip_phase3 as c3
from sim.clip_phase3 import H, W, font
from sim.perception import colour_mask, project
from sim.pose_check import box_corners

RUNS = Path("runs/006/g5run2")
OUT = Path("runs/clip5")
CONTEXT = "Phase 5, perceived poses, counted run 2: fetch 10/10, blocked 10/10, push-or-detour 7/10, two-box 10/10"
CARD_S = 4.0
Z, CROP = 4, 100  # zoom factor and crop size (px of the photo) around the fitted box


def select() -> str:
    res = {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("fetch-*/result.json"))}
    passed = sorted((v["steps_used"], v["seed"], k) for k, v in res.items() if v["pass"])
    i = (len(passed) - 1) // 2  # lower median
    med = passed[i][0]
    main = min((s, k) for st, s, k in passed if st == med)[1]
    L = [f"Selection per the pre-registered rule (2026-11-13-006-clip-rule), applied mechanically to {RUNS}",
         f"fetch runs: {len(res)}, passed {len(passed)}",
         f"passed steps_used sorted: {[(st, k) for st, s, k in passed]}",
         f"n={len(passed)} ({'odd' if len(passed) % 2 else 'even: lower median'}): element {i + 1} -> median {med}",
         f"ties at {med}: {[k for st, s, k in passed if st == med]} -> lowest seed -> {main}"]
    for k, v in sorted(res.items()):
        why = "CHOSEN" if k == main else f"not chosen: steps_used {v['steps_used']}" + (
            f" = median, seed {v['seed']} not lowest" if v["steps_used"] == med else f" != median {med}")
        L.append(f"  {k}: pass={v['pass']} steps={v['steps_used']}  {why}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return main


def wire(dr: ImageDraw.ImageDraw, P: np.ndarray, f: dict[str, Any], off: tuple[float, float], zoom: float, width: int) -> None:
    uv = (project(P, f["K"], f["cam_pos"], f["cam_mat"]) - off) * zoom
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                dr.line([tuple(uv[i]), tuple(uv[j])], fill=(0, 255, 0), width=width)


def card(robot: Any, rt: Any, name: str, r: Any) -> np.ndarray | None:
    if name != "locate":
        return None
    step = rt.steps[-1]
    obj = step["args"]["object"]
    res = robot._cur[obj]
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    dr.text((20, 12), f"locate(object={obj!r}) -> {step['result']}", font=font(26, True), fill=(240, 240, 240))
    dr.text((20, 52), "colour mask = magenta, fitted box = green (sim/perception.py on the mast photos; sim paused)",
            font=font(19), fill=(255, 255, 0))
    frames = robot._frames
    colour = robot._objects[obj]["colour"]
    masks = [colour_mask(f["rgb"], colour) for f in frames]
    used = [k for k in np.argsort([-int(m.sum()) for m in masks]) if masks[k].any()]
    best = int(res["photo"])
    used = [best] + [k for k in used if k != best]
    fit = None
    if res["pos"] is not None:
        y = float(res["yaw"])
        R = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        fit = box_corners(np.asarray(res["pos"], float), R)

    def tinted(k: int) -> Image.Image:
        a = frames[k]["rgb"].astype(np.float32)
        a[masks[k]] = 0.35 * a[masks[k]] + 0.65 * np.array([255, 0, 255])
        return Image.fromarray(a.astype(np.uint8))

    heads = ["0", "90", "180", "270"] if len(frames) == 4 else ["current"]
    main = tinted(best)
    md = ImageDraw.Draw(main)
    if fit is not None:
        wire(md, fit, frames[best], (0, 0), 1, 2)
    img.paste(main, (20, 90))
    dr.text((24, 94), f"photo used for the fit (heading {heads[best]}), {int(masks[best].sum())} mask px", font=font(16, True),
            fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if fit is not None:
        uv = project(fit.mean(0)[None], frames[best]["K"], frames[best]["cam_pos"], frames[best]["cam_mat"])[0]
        off = (min(max(int(uv[0]) - CROP // 2, 0), main.width - CROP), min(max(int(uv[1]) - CROP // 2, 0), main.height - CROP))
        zm = main.crop((off[0], off[1], off[0] + CROP, off[1] + CROP)).resize((CROP * Z, CROP * Z), Image.NEAREST)
        zd = ImageDraw.Draw(zm)
        wire(zd, fit, frames[best], off, Z, 3)
        img.paste(zm, (680, 90))
        dr.text((684, 94), f"zoom x{Z}", font=font(16, True), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    for n, k in enumerate(used[1:4]):
        t = tinted(k).resize((160, 120))
        img.paste(t, (1100, 112 + 150 * n))
        dr.text((1104, 116 + 150 * n), f"{heads[k]}: {int(masks[k].sum())} px", font=font(14, True), fill=(255, 255, 255),
                stroke_width=2, stroke_fill=(0, 0, 0))
    if len(used) > 1:
        dr.text((1100, 90), "other photos used", font=font(14), fill=(200, 200, 200))
    e = step["meta"]["eval"]
    cap = (f"camera fit vs truth: {e['pos_err_m'] * 100:.1f} cm, {e['yaw_err_deg']:.1f} deg" if "pos_err_m" in e
           else "camera fit vs truth: not located")
    dr.text((20, 600), cap, font=font(40, True), fill=(120, 255, 120))
    dr.text((20, 655), "(evaluation only: truth comes from the simulator, the robot never sees it)", font=font(22),
            fill=(200, 200, 200))
    return np.asarray(img)


def main() -> None:
    """OUT/<task_id> = the run's log.jsonl + result.json (copied from RUNS) + photos/ (rsync from PERUN, see the task)."""
    k = select()
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        return
    entry = next(t for t in json.loads(Path("tasksets/phase4_fetch_counted.json").read_text())["tasks"] if t["task_id"] == k)
    ok = c3.render(OUT / k, entry, Path("showcase/phase5_perceived.mp4"), None, CONTEXT, OUT, poses="perceived",
                   card=card, card_s=CARD_S)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
