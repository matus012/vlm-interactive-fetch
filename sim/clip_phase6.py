"""Phase 6 failure gallery (task 007): one captioned slideshow per failed run of the sealed test pass 1.

    python -m sim.clip_phase6 select | render [task_id ...] | index

Built ONLY from pass 1's logged artefacts (log.jsonl, result.json, the logged photos in runs/clip6/runs/<id>/photos);
no sim is run, no replay check. One slide per logged step: the photo(s) the VLM was given at that step
(extra.packet.photos, mapped by file name), the call and its result, the last sentence of its reasoning, the history so far.
Every slide is shown for the same time (uniform), so the clip is <= 14.5 s.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim.clip_phase3 import H, W, excerpt, fit, fmt, font, wrap

RUNS = Path("runs/007/pass1")
OUT = Path("runs/clip6")
SHOW = Path("showcase/failures/test")
FPS, STEP_S, END_S = 30, 11.5, 3.0
CONTEXT = "sealed test, pass 1: fetch 24/25, blocked 19/25, push-or-detour 20/25, two-box 23/25"
MEANING = {"vlm": "the AI's choices (search, order, giving up)", "skill": "a robot skill (pick / place / push) failed"}
PLAIN = {
    "blocked-100024": "vlm: the AI explored before pushing the crate (step 15); the box was on the table at step 20 with no step left to call done.",
    "blocked-100025": "vlm: the AI explored before pushing the crate and made 2 malformed calls; the box was on target, but done was never called.",
    "blocked-100031": "vlm: 15 exploration steps before the push, then the steps ran out with the yellow box only just located.",
    "blocked-100042": "vlm: locate was called 12 times in a row; the box sat just under the camera's visibility bar (97 of 100 px); the steps ran out while holding it.",
    "blocked-100046": "vlm: the AI never searched the kitchen; the blue box was barely visible (85 px) from the living room.",
    "blocked-100054": "skill: the box was located from 4.3 m away; the first pick tipped it onto its side and every later pick slipped.",
    "fetch-100005": "skill: the nightstand box was located from 1.8 m with an 8.5 cm / 72 deg pose error; 7 picks from the same spot and wrong pose all failed.",
    "push_or_detour-100043": "skill: the red box was located from 4.2 m; the first pick tipped it onto its side and every later pick slipped or missed.",
    "push_or_detour-100058": "vlm: 11 exploration steps; the box was picked at step 20 and the run ended holding it.",
    "push_or_detour-100065": "skill: carrying the box, the robot drove into the cabinet and the impact knocked the box out of the grip.",
    "push_or_detour-100090": "skill: the box was knocked out of the grip by the cabinet, twice; the AI shares the blame: it routed through the blocked north door after using the south-door detour.",
    "push_or_detour-100101": "vlm: the box was on target at step 20, but done was never called.",
    "two_box-100070": "skill: the red box was located from 4.1 m; the second pick reused that far pose and tipped the box, then 7 more picks slipped.",
    "two_box-100100": "skill: a 7.1 cm pose error at the nightstand; the box fell, was picked upside down and placed upside down, so the 15 deg upright rule failed it; the AI's done note wrongly claimed success.",
}
EXPL = {"surface": "every target box rests upright on its target surface", "wrong_box": "no other box on that surface",
        "held": "nothing held at the end", "steps": "done called within the step limit", "sim": "simulation healthy"}


def select() -> None:
    ids = [p.parent.name for p in sorted(RUNS.glob("*/result.json")) if not json.loads(p.read_text())["pass"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.txt").write_text("\n".join(ids) + "\n")
    (OUT / "selection_note.txt").write_text(
        f"{len(ids)} runs with pass:false in {RUNS}. No replay check: the gallery is rendered from pass 1's logged "
        "artefacts only (log.jsonl, result.json, logged photos); no sealed task is re-run in the sim.\n")
    for k in ids:
        (OUT / "runs" / k).mkdir(parents=True, exist_ok=True)
        for f in ("log.jsonl", "result.json"):
            shutil.copy(RUNS / k / f, OUT / "runs" / k / f)
    print(len(ids), ids)


def slide(task: dict, steps: list[dict], k: int, d: Path, res: dict) -> np.ndarray:
    s = steps[k]
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    f_big, f_mid, f_small, f_hist = font(22, True), font(19), font(15), font(14)
    dr.text((14, 10), "Task: " + task["instruction"], font=f_big, fill=(240, 240, 240))
    b = task["blocker"]
    dr.text((14, 40), f"Blocker: {b}" if b != "none" else "Blocker: none", font=f_mid, fill=(255, 190, 90) if b != "none" else (170, 170, 170))
    ph = [Image.open(d / "photos" / Path(p).name).convert("RGB") for p in s["extra"]["packet"]["photos"] if p]
    names = [Path(p).name for p in s["extra"]["packet"]["photos"] if p]
    if len(ph) == 4:
        for i, (p, n) in enumerate(zip(ph, names)):
            img.paste(p.resize((320, 240)), (14 + 324 * (i % 2), 76 + 244 * (i // 2)))
            dr.text((20 + 324 * (i % 2), 80 + 244 * (i // 2)), n.split("_")[1][:-4], font=f_small, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    elif ph:
        img.paste(ph[0].resize((640, 480)), (14, 76))
        dr.text((20, 80), "photo " + names[0][:-4], font=f_small, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    else:
        dr.text((200, 300), "no photo given at this step", font=f_mid, fill=(170, 170, 170))
    dr.text((14, 566), "VLM's reasoning, last sentence (verbatim):", font=f_small, fill=(255, 255, 0))
    for i, ln in enumerate(wrap(dr, "“" + excerpt(s["extra"]["vlm"]["reasoning"]) + "”", f_mid, 640)[:3]):
        dr.text((14, 586 + 24 * i), ln, font=f_mid, fill=(170, 200, 255))
    x0 = 676
    dr.text((x0, 76), "steps so far", font=f_small, fill=(255, 255, 0))
    lo = max(0, k + 1 - 16)
    for i in range(lo, k + 1):
        t = steps[i]
        cur = i == k
        dr.text((x0, 100 + 24 * (i - lo)), fit(dr, f"{i + 1}. {fmt(t['call'], t['args'])} -> {t['result']}", f_hist, W - x0 - 10),
                font=f_hist, fill=(255, 255, 255) if cur else (160, 255, 160) if t["ok"] else (255, 150, 150))
    dr.rectangle((0, 674, W, H), fill=(30, 30, 40))
    line = f"step {k + 1}/{len(steps)}:  "
    dr.text((10, 684), line, font=f_mid, fill=(240, 240, 240))
    dr.text((10 + dr.textlength(line, font=f_mid), 684), fit(dr, f"{fmt(s['call'], s['args'])} -> {s['result']}", f_mid, W - 200),
            font=f_mid, fill=(160, 255, 160) if s["ok"] else (255, 150, 150))
    return np.asarray(img)


def endcard(task: dict, res: dict) -> np.ndarray:
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    dr.text((50, 30), "Task: " + task["instruction"], font=font(30, True), fill=(240, 240, 240))
    dr.text((50, 85), "Scorer verdict: FAIL", font=font(44, True), fill=(255, 110, 110))
    dr.text((50, 155), "Failed rules:", font=font(24), fill=(255, 255, 0))
    for i, k in enumerate(res["failed"]):
        dr.text((70, 192 + 34 * i), f"FAIL  {k}: {EXPL[k]}", font=font(23), fill=(255, 110, 110))
    dr.text((50, 320), f"Label: {res['label']} = {MEANING[res['label']]}", font=font(26, True), fill=(255, 190, 90))
    dr.text((50, 370), f"Steps used {res['steps_used']} (limit {task['step_limit']}), blocker: {task['blocker']}, done called: {res['done_called']}",
            font=font(22), fill=(200, 200, 200))
    for i, ln in enumerate(wrap(dr, "What went wrong: " + PLAIN[res["task_id"]], font(24), W - 100)[:4]):
        dr.text((50, 430 + 34 * i), ln, font=font(24), fill=(255, 255, 255))
    dr.text((50, 640), CONTEXT, font=font(22), fill=(255, 190, 90))
    return np.asarray(img)


def render(k: int | str, V: dict) -> None:
    d = OUT / "runs" / k
    lines = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
    steps, res = lines[:-1], json.loads((d / "result.json").read_text())
    n = len(steps)
    per = min(45, int(STEP_S * FPS / n))
    out = SHOW / f"{k}.mp4"
    SHOW.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    for i in range(n):
        a = slide(V[k], steps, i, d, res)
        for _ in range(per):
            w.append_data(a)
    e = endcard(V[k], res)
    for _ in range(int(END_S * FPS)):
        w.append_data(e)
    w.close()
    print(f"{out}: {n} slides x {per / FPS:.2f} s, {(n * per + int(END_S * FPS)) / FPS:.1f} s, {out.stat().st_size / 1e6:.2f} MB")


def index() -> None:
    rows = ["| task_id | type | label | failed rules | steps | what went wrong |", "|---|---|---|---|---|---|"]
    for k in (OUT / "selection.txt").read_text().split():
        r = json.loads((RUNS / k / "result.json").read_text())
        rows.append(f"| {k} | {k.rsplit('-', 1)[0]} | {r['label']} | {', '.join(r['failed'])} | {r['steps_used']} | {PLAIN[k]} |")
    (SHOW / "index.md").write_text("Sealed test, pass 1: failed runs (14 of 100), rendered from the logged artefacts only (no replay). Labels: "
                                   + "; ".join(f"{k} = {v}" for k, v in MEANING.items()) + "\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    elif sys.argv[1] == "index":
        index()
    else:
        V = {t["task_id"]: t for t in map(json.loads, (RUNS / "verified.jsonl").read_text().splitlines()) if t}
        for k in sys.argv[2:] or (OUT / "selection.txt").read_text().split():
            render(k, V)
