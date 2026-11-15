"""Task 010 E3 clip (<= 30 s): selection per the pre-registered rule (2026-11-14-010-clip-rule); slides by sim/clip_phase6.slide from
logged artefacts only (VLM: runs/007/pass1 with its photos fetched to runs/010/clip/vlm/<id>; script: runs/010/pass1/
<e3a|e3b>/<id>), verdict cards and a results card. No sealed task is re-run.

    python -m sim.clip_e3 select      # -> runs/010/clip_selection.txt
    python -m sim.clip_e3 render      # -> showcase/e3_no_vlm.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim import clip_phase6 as c6
from sim.clip_e1 import BG, EXPL, GREY, GRN, ORG, RED, WHITE, YEL, card
from sim.clip_phase3 import FPS, W, font, wrap

BASE, RUNS, VLM_PHOTOS = Path("runs/007/pass1"), Path("runs/010/pass1"), Path("runs/010/clip/vlm")
SEL, ANALYSIS, OUT = Path("runs/010/clip_selection.txt"), Path("runs/010/e3_analysis.json"), Path("showcase/e3_no_vlm.mp4")
TYPES = ("fetch", "blocked", "push_or_detour", "two_box")
NAMES = {"fetch": "fetch a box", "blocked": "blocked door", "push_or_detour": "push or detour", "two_box": "two boxes"}
RUN_S = 8.0
# filled in after reading the two selected runs' calls (render asserts the selection they were written for)
CAPTIONS: dict[str, str] = {
    "for": "e3a push_or_detour-100035",
    "vlm": "the AI went straight to the storage room and picked the box. The cabinet blocked the kitchen; push said "
           "'not movable', so it went round by the kitchen south door and placed the box.",
    "script": "the script searched the bedroom, then the storage room, and picked the box. The cabinet blocked the "
              "kitchen; push said 'not movable'. 'Always push' has no other move, so it gave up holding the box.",
    "results": "Script B, which also goes round when push fails, matches the AI (88 vs 86 of 100). The 'always push' "
               "script fails almost every cabinet task (2 of 25). On these tasks the AI's clear contribution is choosing "
               "to go round; it also needs fewer steps for one box (median 9 vs 13).",
}


def res(d: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(d.glob("*/result.json"))}


def select() -> tuple[str, str]:
    base = res(BASE)
    for s in ("e3a", "e3b"):
        sc = res(RUNS / s)
        gap = {t: sum(r["pass"] for k, r in base.items() if k.startswith(t + "-"))
               - sum(r["pass"] for k, r in sc.items() if k.startswith(t + "-")) for t in TYPES}
        t = max(TYPES, key=lambda t: abs(gap[t]))  # first max = TYPES order on ties
        vlm_wins = gap[t] >= 0
        cand = [k for k in base if k.startswith(t + "-") and base[k]["pass"] == vlm_wins and sc[k]["pass"] != vlm_wins]
        if cand:
            k = min(cand, key=lambda k: base[k]["seed"])
            SEL.write_text(f"rule: the pre-registered rule (2026-11-14-010-clip-rule)\nscript: {s}\ntype: {t} (VLM - {s} = {gap[t]})\n"
                           f"task: {k}\n")
            return s, k
    raise SystemExit("VLM and both scripts agree on every task")


def verdict(task: dict, r: dict, who: str, what: str) -> np.ndarray:
    ok = r["pass"]
    lines = [(f"{who}: " + task["instruction"], 30, True, WHITE),
             (f"Scorer verdict: {'PASS' if ok else 'FAIL'}", 44, True, GRN if ok else RED)]
    lines += [(f"FAIL  {k}: {EXPL.get(k, k)}", 24, False, RED) for k in r["failed"]]
    lines += [(f"Steps used {r['steps_used']} of {task['step_limit']}; done called: {r['done_called']}", 24, False, GREY),
              ("What happened: " + what, 26, False, WHITE),
              ("Sealed test task. Slideshow of the logged photos and steps (not re-run).", 22, False, ORG)]
    return card(lines, y=80)


def results(a: dict) -> np.ndarray:
    img = Image.fromarray(card([("E3: the AI vs a fixed script, same 100 sealed tasks", 34, True, WHITE)], y=30))
    dr = ImageDraw.Draw(img)
    for x, t in ((60, "type"), (430, "AI (Qwen)"), (650, "script A: always push"), (970, "script B: + go round")):
        dr.text((x, 100), t, font=font(22, True), fill=YEL)
    y = 140
    for t in TYPES + ("all",):
        v = a["per_type"][t] if t != "all" else a["total"]["all"]
        big = t == "all"
        dr.text((60, y), "overall" if big else NAMES[t], font=font(28 if big else 25, big), fill=WHITE)
        for x, k in ((430, "vlm"), (650, "e3a"), (970, "e3b")):
            dr.text((x, y), f"{v[k]['pass']}/{v[k]['n']}", font=font(28 if big else 25, k == "vlm"), fill=GRN if k == "vlm" else GREY)
        y += 50 if big else 42
    f = font(23)
    for i, ln in enumerate(wrap(dr, CAPTIONS["results"], f, W - 120)):
        dr.text((60, y + 20 + 31 * i), ln, font=f, fill=ORG)
    return np.asarray(img)


def render() -> None:
    lines = SEL.read_text().splitlines()
    s, k = lines[1].split(": ")[1], lines[3].split(": ")[1]
    assert CAPTIONS.get("for") == f"{s} {k}", "captions were written for another selection"
    a = json.loads(ANALYSIS.read_text())
    V = {t["task_id"]: t for t in map(json.loads, (RUNS / "verified.jsonl").read_text().splitlines())}
    draw_text = ImageDraw.ImageDraw.text  # a script has no reasoning: relabel that panel on script slides

    def relabel(on: bool) -> None:
        def text(self, xy, t, *args, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
            if on and isinstance(t, str) and t.startswith("VLM's reasoning"):
                t = "Fixed script: no reasoning, the next rule in its list:"
            return draw_text(self, xy, t, *args, **kw)
        ImageDraw.ImageDraw.text = text

    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])

    def hold(arr: np.ndarray, sec: float) -> None:
        for _ in range(int(sec * FPS)):
            w.append_data(arr)

    hold(card([("Extension E3: what does the AI add?", 44, True, WHITE),
               ("The same 100 sealed tasks, same robot skills and judge, with the AI replaced by a fixed script that "
                "sees only what the AI sees: script A searches the rooms in a fixed order and always pushes; script B also "
                "goes round by the other door when push fails.", 26, False, GREY),
               (f"Next: one task, first the AI, then script {s[-1].upper()}; then the results.", 24, False, YEL)], y=150), 2.5)
    for who, d, cap in (("AI", VLM_PHOTOS / k, CAPTIONS["vlm"]), (f"Script {s[-1].upper()}", RUNS / s / k, CAPTIONS["script"])):
        relabel(who != "AI")
        logs = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
        steps, r = logs[:-1], json.loads((d / "result.json").read_text())
        per = max(1, int(RUN_S * FPS / len(steps)))
        for i in range(len(steps)):
            hold(c6.slide(V[k], steps, i, d, r), per / FPS)
        relabel(False)
        hold(verdict(V[k], r, who, cap), 2.25)
    hold(results(a), 6.0)
    w.close()
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    {"select": select, "render": render}[sys.argv[1]]()
