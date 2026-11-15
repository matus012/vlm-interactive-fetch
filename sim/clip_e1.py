"""Task 009 E1 clip (<= 30 s): selection per the pre-registered rule (2026-11-14-009-clip-rule), slides by sim/clip_phase6.slide from
the E1 counted pass's logged artefacts only (no sealed task is re-run), title / verdict / results cards.

    python -m sim.clip_e1      # -> showcase/e1_no_easings.mp4, runs/009/clip_selection.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim.clip_phase3 import FPS, H, W, font, wrap
from sim.clip_phase6 import slide

RUNS = Path("runs/009/pass1")
ANALYSIS = Path("runs/009/e1_analysis.json")
OUT = Path("showcase/e1_no_easings.mp4")
BG, WHITE, GREY, YEL, ORG, GRN, RED = (18, 18, 22), (240, 240, 240), (200, 200, 200), (255, 255, 0), (255, 190, 90), (160, 255, 160), (255, 110, 110)
RUN_S = 8.0
EXPL = {"surface": "the box is not on the named surface", "held": "the box is still in the gripper",
        "steps": "no done within the step limit", "upright": "the box is not upright", "still": "the box is still moving"}


def lower_median(rows: list[tuple[int, int, str]]) -> str:
    rows = sorted(rows)
    med = rows[(len(rows) - 1) // 2][0]
    return min((seed, k) for st, seed, k in rows if st == med)[1]


def select() -> tuple[str, str]:
    rs = {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("*/result.json"))}
    a = json.loads(ANALYSIS.read_text())
    nopg = {f["task_id"] for f in a["failures"] if "target_no_pregrasp" in f["cases"]}
    ok = lower_median([(v["steps_used"], v["seed"], k) for k, v in rs.items() if k.startswith("push_or_detour-") and v["pass"]])
    bad = lower_median([(rs[k]["steps_used"], rs[k]["seed"], k) for k in nopg])
    Path("runs/009/clip_selection.txt").write_text(f"rule: the pre-registered rule (2026-11-14-009-clip-rule)\npassed: {ok}\nfailed: {bad} "
                                                   f"(one of {len(nopg)} no-pre-grasp failures)\n")
    return ok, bad


def card(lines: list[tuple[str, int, bool, tuple[int, int, int]]], y: int = 60) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    for text, size, bold, col in lines:
        f = font(size, bold)
        for ln in wrap(dr, text, f, W - 120) if text else [""]:
            dr.text((60, y), ln, font=f, fill=col)
            y += size + 7
        y += 14
    return np.asarray(img)


def verdict(task: dict, res: dict, what: str) -> np.ndarray:
    ok = res["pass"]
    lines = [("Task: " + task["instruction"], 30, True, WHITE),
             (f"Scorer verdict: {'PASS' if ok else 'FAIL'}", 44, True, GRN if ok else RED)]
    lines += [(f"FAIL  {k}: {EXPL.get(k, k)}", 24, False, RED) for k in res["failed"]]
    lines += [(f"Steps used {res['steps_used']} of {task['step_limit']}; done called: {res['done_called']}"
               + ("" if ok else f"; label: {res['label']}"), 24, False, GREY), ("What happened: " + what, 26, False, WHITE),
              ("E1 sealed test, one counted pass. Slideshow of the logged photos and steps (not re-run).", 22, False, ORG)]
    return card(lines, y=80)


def results(a: dict) -> np.ndarray:
    img = Image.fromarray(card([("E1: same AI, easings removed, 100 new sealed tasks", 34, True, WHITE)], y=30))
    dr = ImageDraw.Draw(img)
    dr.text((60, 100), "type", font=font(22, True), fill=YEL)
    for x, t in ((470, "E1"), (700, "base test"), (950, "E1, arm can reach")):
        dr.text((x, 100), t, font=font(22, True), fill=YEL)
    names = {"fetch": "fetch a box", "blocked": "blocked door", "push_or_detour": "push or detour", "two_box": "two boxes"}
    y = 140
    for t, v in a["per_type"].items():
        dr.text((60, y), names[t], font=font(25), fill=WHITE)
        dr.text((470, y), f"{v['e1']['pass']}/25", font=font(25, True), fill=GRN if v['e1']['pass'] >= 18 else RED)
        dr.text((700, y), f"{v['sealed_005']['pass']}/25", font=font(25), fill=GREY)
        n = 25 - v["n_no_pregrasp"]
        dr.text((950, y), f"{v['e1']['pass']}/{n}", font=font(25), fill=GREY)
        y += 42
    tot = a["total"]
    dr.text((60, y + 10), "overall", font=font(28, True), fill=WHITE)
    dr.text((470, y + 10), f"{tot['e1']['pass']}/100", font=font(28, True), fill=WHITE)
    dr.text((700, y + 10), f"{tot['sealed_005']['pass']}/100", font=font(28), fill=GREY)
    dr.text((950, y + 10), f"{tot['e1']['pass']}/{100 - a['n_no_pregrasp']}", font=font(28), fill=GREY)
    f = font(24)
    text = (f"{a['n_no_pregrasp']} of the new tasks asked for a box the arm cannot reach from anywhere (the case the "
            "easings filtered out); every one failed. On the rest the AI did as well as on the base test; boxes on the "
            "robot's routes and the missing detour hint made no clear difference.")
    for i, ln in enumerate(wrap(dr, text, f, W - 120)):
        dr.text((60, y + 70 + 32 * i), ln, font=f, fill=ORG)
    return np.asarray(img)


def main() -> None:
    a = json.loads(ANALYSIS.read_text())
    nopg = [f["task_id"] for f in a["failures"] if "target_no_pregrasp" in f["cases"]]
    a["n_no_pregrasp"] = len(nopg)
    for t in a["per_type"]:
        a["per_type"][t]["n_no_pregrasp"] = sum(k.startswith(t + "-") for k in nopg)
    V = {t["task_id"]: t for t in map(json.loads, (RUNS / "verified.jsonl").read_text().splitlines())}
    ok, bad = select()
    what = {ok: "carrying the box, the robot met the cabinet in the kitchen door; push said 'not movable'. Without the hint the AI "
                "first tried a route through the blocked door, then went round by the other kitchen door and placed the box.",
            bad: "the box lies on the back of the nightstand, where the arm cannot reach it from any standing spot. pick "
                 "answered 'unreachable' 6 times; the AI gave up with done at step 20."}
    assert (ok, bad) == ("push_or_detour-160031", "fetch-160001"), "captions were written for these two runs"
    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])

    def hold(arr: np.ndarray, s: float) -> None:
        for _ in range(int(s * FPS)):
            w.append_data(arr)

    hold(card([("Extension E1: was the house made easy?", 44, True, WHITE),
               ("The frozen AI again, on 100 new sealed tasks drawn without the easings: boxes may lie on the robot's routes, "
                "anywhere on a nightstand or near furniture, also where the arm cannot reach, and the hint 'a door blocked by "
                "something unmovable stays blocked, go round' is removed from the AI's rules.", 26, False, GREY),
               ("Next: one passed run, one failed run, the results.", 24, False, YEL)], y=150), 2.5)
    for k in (ok, bad):
        d = RUNS / k
        lines = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
        steps, res = lines[:-1], json.loads((d / "result.json").read_text())
        per = max(1, int(RUN_S * FPS / len(steps)))
        for i in range(len(steps)):
            hold(slide(V[k], steps, i, d, res), per / FPS)
        hold(verdict(V[k], res, what[k]), 2.25)
    hold(results(a), 6.5)
    w.close()
    print(OUT, OUT.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
