"""Task 011 E2 clip (<= 30 s): selection per the pre-registered rule (2026-11-14-011-clip-rule); house plans card, slides by
sim/clip_phase6.slide from the house B counted pass's logged artefacts only (no sealed task is re-run), verdict cards,
results card. Every task shown must carry "house": "B" (review 011 F7).

    python -m sim.clip_e2 select      # -> runs/011/clip_selection.txt
    python -m sim.clip_e2 render      # -> showcase/e2_new_house.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim import clip_phase6 as c6
from sim.clip_e1 import EXPL, GREY, GRN, ORG, RED, WHITE, YEL, card, lower_median
from sim.clip_phase3 import FPS, H, W, font, wrap

RUNS, ANALYSIS = Path("runs/011/pass1"), Path("runs/011/e2_analysis.json")
SEL, OUT = Path("runs/011/clip_selection.txt"), Path("showcase/e2_new_house.mp4")
PLANS = (Path("runs/011/house_a_plan.png"), Path("runs/011/house_b_plan.png"))
TYPES = ("fetch", "blocked", "push_or_detour", "two_box")
NAMES = {"fetch": "fetch a box", "blocked": "blocked door", "push_or_detour": "push or detour", "two_box": "two boxes"}
RUN_S = 7.5
# filled in after reading the selected runs' calls (render asserts the selection they were written for)
CAPTIONS: dict[str, str] = {
    "for": "push_or_detour-170027 push_or_detour-170024",
    "push_or_detour-170027": "carrying the box from the storage room, the robot was blocked at the kitchen. The AI went "
                             "round through the hall (a room, not a door) and placed the box.",
    "push_or_detour-170024": "the cabinet blocked the living room-kitchen door; the AI went round through the hall and "
                             "picked the box in the kitchen. After that every drive out of the kitchen, the hall route "
                             "included, came back 'blocked'; it ran out of steps holding the box.",
    "results": "Same score as house A. Cabinet tasks dipped to 17 of 25: the rules' hint 'name the other door in via' "
               "does not force a way round in this layout (5 runs tried it, 1 passed); going round via a room passed 15 of 18. The "
               "fixed script B relied on twin doors and fell from 88 to 70.",
}


def res() -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("*/result.json"))}


def select() -> tuple[str, str | None]:
    rs = res()
    ok_type = "push_or_detour" if any(v["pass"] for k, v in rs.items() if k.startswith("push_or_detour-")) else "fetch"
    ok = lower_median([(v["steps_used"], v["seed"], k) for k, v in rs.items() if k.startswith(ok_type + "-") and v["pass"]])
    fails = {t: [k for k, v in rs.items() if k.startswith(t + "-") and not v["pass"]] for t in TYPES}
    ft = max(TYPES, key=lambda t: len(fails[t]))  # first max = TYPES order on ties
    bad = lower_median([(rs[k]["steps_used"], rs[k]["seed"], k) for k in fails[ft]]) if fails[ft] else None
    SEL.write_text(f"rule: the pre-registered rule (2026-11-14-011-clip-rule)\npassed: {ok}\nfailed: {bad} (type with most failures: {ft}, "
                   f"{len(fails[ft])})\n")
    return ok, bad


def plans_card() -> np.ndarray:
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    dr.text((40, 20), "Extension E2: the same frozen AI in a house it has never seen", font=font(34, True), fill=WHITE)
    for i, (p, label) in enumerate(zip(PLANS, ("house A (all earlier tests)", "house B (new, designed blind)"))):
        im = Image.open(p).convert("RGB")
        im.thumbnail((600, 560))
        x = 40 + i * 620
        img.paste(im, (x, 100))
        dr.text((x, 72), label, font=font(22, True), fill=YEL)
    dr.text((40, 672), "Same rooms, furniture, boxes, crate and cabinet; new floor plan and doors. 100 new locked tasks.",
            font=font(21), fill=ORG)
    return np.asarray(img)


def verdict(task: dict, r: dict, what: str) -> np.ndarray:
    ok = r["pass"]
    lines = [("House B: " + task["instruction"], 30, True, WHITE),
             (f"Scorer verdict: {'PASS' if ok else 'FAIL'}", 44, True, GRN if ok else RED)]
    lines += [(f"FAIL  {k}: {EXPL.get(k, k)}", 24, False, RED) for k in r["failed"]]
    lines += [(f"Steps used {r['steps_used']} of {task['step_limit']}; done called: {r['done_called']}"
               + ("" if ok else f"; label: {r['label']}"), 24, False, GREY), ("What happened: " + what, 26, False, WHITE),
              ("Sealed house B task, one counted pass. Slideshow of the logged photos and steps (not re-run).", 22, False, ORG)]
    return card(lines, y=80)


def results(a: dict) -> np.ndarray:
    img = Image.fromarray(card([("E2: house A vs house B, frozen AI, 100 locked tasks each", 34, True, WHITE)], y=30))
    dr = ImageDraw.Draw(img)
    for x, t in ((60, "type"), (470, "house A"), (700, "house B")):
        dr.text((x, 100), t, font=font(22, True), fill=YEL)
    y = 140
    for t in TYPES + ("all",):
        v = a["per_type"][t]
        big = t == "all"
        dr.text((60, y), "overall" if big else NAMES[t], font=font(28 if big else 25, big), fill=WHITE)
        dr.text((470, y), f"{v['A_vlm']['pass']}/{v['A_vlm']['n']}", font=font(28 if big else 25), fill=GREY)
        b = v["B_vlm"]["pass"] / v["B_vlm"]["n"]
        dr.text((700, y), f"{v['B_vlm']['pass']}/{v['B_vlm']['n']}", font=font(28 if big else 25, True), fill=GRN if b >= 0.7 else RED)
        y += 50 if big else 42
    f = font(23)
    for i, ln in enumerate(wrap(dr, CAPTIONS["results"], f, W - 120)):
        dr.text((60, y + 20 + 31 * i), ln, font=f, fill=ORG)
    return np.asarray(img)


def render() -> None:
    lines = SEL.read_text().splitlines()
    ok, bad = lines[1].split(": ")[1], lines[2].split(": ")[1].split(" ")[0]
    bad = None if bad == "None" else bad
    assert CAPTIONS.get("for") == f"{ok} {bad}", "captions were written for another selection"
    a = json.loads(ANALYSIS.read_text())
    V = {t["task_id"]: t for t in map(json.loads, (RUNS / "verified.jsonl").read_text().splitlines())}
    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])

    def hold(arr: np.ndarray, sec: float) -> None:
        for _ in range(int(sec * FPS)):
            w.append_data(arr)

    hold(plans_card(), 3.5)
    for k in filter(None, (ok, bad)):
        assert V[k].get("house") == "B", k
        d = RUNS / k
        logs = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
        steps, r = logs[:-1], json.loads((d / "result.json").read_text())
        per = max(1, int(RUN_S * FPS / len(steps)))
        for i in range(len(steps)):
            hold(c6.slide(V[k], steps, i, d, r), per / FPS)
        hold(verdict(V[k], r, CAPTIONS[k]), 2.25)
    hold(results(a), 6.0)
    w.close()
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    {"select": select, "render": render}[sys.argv[1]]()
