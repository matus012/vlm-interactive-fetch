"""Task 015 E4 clip (<= 30 s): selection per the pre-registered rule (2026-11-15-015-clip-rule) (runs/014/clip_selection.txt); the
same place-referenced task done by the frozen VLM agent and by the E3c script, slides by sim/clip_phase6.slide from the
E4 counted pass's logged artefacts only (no sealed task is re-run), a verdict card after each, the results card.

    python -m sim.clip_e4      # -> showcase/e4_needs_vlm.mp4
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim import clip_phase6 as c6
from sim.clip_e1 import EXPL, GREY, GRN, ORG, RED, WHITE, YEL, card
from sim.clip_phase3 import FPS, font, wrap, W

RUNS, ANALYSIS = Path("runs/014/pass1"), Path("runs/014/e4_analysis.json")
SEL, OUT = Path("runs/014/clip_selection.txt"), Path("showcase/e4_needs_vlm.mp4")
RUN_S = 6.0
EXPL4 = EXPL | {"wrong_box": "a box that was not asked for is on the named surface"}
# written after reading the selected runs' calls (render asserts the selection they were written for)
CAPTIONS = {
    "for": "place_ref-181001",
    "vlm": "the AI drove straight to the bedroom, found the yellow box on the nightstand and placed it on the storage "
           "bench.",
    "e3c": "the script cannot use 'on the nightstand': it took the first box it located (blue, in storage) and placed "
           "the wrong box.",
}
CATS = (("place_ref", "the box next to the sofa / on the nightstand"), ("recovery", "the box is moved after it is found"),
        ("hint", "'...; someone left it in the bedroom'"))


def verdict(task: dict, r: dict, who: str, what: str) -> np.ndarray:
    ok = r["pass"]
    lines = [(f"{who}: " + task["instruction"], 30, True, WHITE),
             (f"Scorer verdict: {'PASS' if ok else 'FAIL'}", 44, True, GRN if ok else RED)]
    lines += [(f"FAIL  {k}: {EXPL4.get(k, k)}", 24, False, RED) for k in r["failed"]]
    lines += [(f"Steps used {r['steps_used']} of {task['step_limit']}", 24, False, GREY), ("What happened: " + what, 26, False, WHITE),
              ("E4 sealed test, one counted pass. Slideshow of the logged photos and steps (not re-run).", 22, False, ORG)]
    return card(lines, y=80)


def results(a: dict) -> np.ndarray:
    img = Image.fromarray(card([("E4: tasks that need judgement, 100 locked tasks (both houses)", 32, True, WHITE)], y=30))
    dr = ImageDraw.Draw(img)
    for x, t in ((60, "kind of task"), (720, "AI"), (900, "best script"), (1100, "")):
        dr.text((x, 100), t, font=font(22, True), fill=YEL)
    y = 140
    for c, name in CATS:
        e = a["per_category"][c]
        best = e[e["best_script"]]
        dr.text((60, y), name, font=font(23), fill=WHITE)
        dr.text((720, y), f"{e['vlm']['pass']}/{e['vlm']['n']}", font=font(25, True), fill=WHITE)
        dr.text((900, y), f"{best['pass']}/{best['n']}", font=font(25), fill=GREY)
        dr.text((60, y + 30), e["verdict"], font=font(20), fill=ORG)
        y += 78
    f = font(22)
    msg = ("By the rule written before the run the AI does not clearly add value on E4: it wins when it must read the "
           "scene, but when the box is moved it trusted the old photos and never found it again; a script that "
           "always looks again after a failed pick got 8.")
    for i, ln in enumerate(wrap(dr, msg, f, W - 120)):
        dr.text((60, y + 10 + 29 * i), ln, font=f, fill=ORG)
    return np.asarray(img)


def main() -> None:
    k = SEL.read_text().split("picked: ")[1].split()[0]
    assert CAPTIONS["for"] == k, "captions were written for another selection"
    a = json.loads(ANALYSIS.read_text())
    V = {t["task_id"]: t for f in RUNS.glob("verified_*.jsonl") for t in map(json.loads, f.read_text().splitlines()) if t}
    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])

    def hold(arr: np.ndarray, sec: float) -> None:
        for _ in range(int(sec * FPS)):
            w.append_data(arr)

    hold(card([("Extension E4: tasks that need the AI's judgement", 40, True, WHITE),
               ("The same locked task, done by the frozen AI and by the best fixed script (E3c).", 28, False, GREY),
               ("Task: " + V[k]["instruction"], 30, True, YEL),
               ("No colour is named: the robot has to find the box that is on the nightstand.", 26, False, WHITE)], y=180), 3.0)
    for who, d, key in (("AI (frozen agent)", RUNS / k, "vlm"), ("Script E3c", RUNS / "e3c" / k, "e3c")):
        logs = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
        steps, r = logs[:-1], json.loads((d / "result.json").read_text())
        per = max(1, int(RUN_S * FPS / len(steps)))
        for i in range(len(steps)):
            hold(c6.slide(V[k], steps, i, d, r), per / FPS)
        hold(verdict(V[k], r, who, CAPTIONS[key]), 3.0)
    hold(results(a), 6.5)
    w.close()
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
