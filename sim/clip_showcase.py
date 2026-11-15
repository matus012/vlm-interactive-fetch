"""Phase 7 showcase video (task 008): selection per the pre-registered rule (2026-11-13-008-video-rule), rendering by
sim/clip_phase3.render (dev passes, replayed; perceived poses) and sim/clip_phase6.render (sealed failure, logged
artefacts only, never re-run), plus title/table/closing cards, joined into showcase/116_showcase.mp4.

    MUJOCO_GL=egl python -m sim.clip_showcase select | dev | fail | cards | join
`select` writes runs/clip7/selection.txt. `dev` replays the chosen dev runs (a run whose replay differs from its log is
dropped for the next one in the rule's order, and the selection file says so). Inputs are copied to runs/clip7/runs/<id>.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from sim import clip_phase3 as c3
from sim import clip_phase6 as c6
from sim.clip_phase3 import FPS, H, W, font, wrap

DEV = Path("runs/006/g5run2")
SEALED = Path("runs/007/pass1")
OUT = Path("runs/clip7")
SEG = OUT / "seg"
TYPES = ("fetch", "blocked", "push_or_detour", "two_box")
PLAIN = {"fetch": "fetch", "blocked": "blocked-path", "push_or_detour": "push-or-detour", "two_box": "two-box"}
DEV_S, FAIL_STEP_S = 16.0, 9.0
CONTEXT = "practice run of the frozen system (a development task, not part of the sealed test)"
BG, WHITE, YEL, ORG, GRN = (18, 18, 22), (240, 240, 240), (255, 255, 0), (255, 190, 90), (160, 255, 160)


def order(t: str) -> list[str]:
    """Passing runs of type t in the rule's order: the lower-median run first, then by (steps distance, seed)."""
    rs = {p.parent.name: json.loads(p.read_text()) for p in sorted(DEV.glob(f"{t}-*/result.json"))}
    passed = sorted((v["steps_used"], v["seed"], k) for k, v in rs.items() if v["pass"])
    med = passed[(len(passed) - 1) // 2][0]
    return [k for st, s, k in sorted(passed, key=lambda x: (abs(x[0] - med), x[1]))]


def sealed_pick() -> str:
    rs = {p.parent.name: json.loads(p.read_text()) for p in sorted(SEALED.glob("*/result.json"))}
    bad = {k: v for k, v in rs.items() if not v["pass"]}
    typ = max(TYPES, key=lambda t: sum(k.startswith(t + "-") for k in bad))
    f = sorted((v["steps_used"], v["seed"], k) for k, v in bad.items() if k.startswith(typ + "-"))
    med = f[(len(f) - 1) // 2][0]
    return min((sd, k) for st, sd, k in f if st == med)[1]  # ties at the median -> lowest seed


def select() -> None:
    L = [f"Selection per the pre-registered rule (2026-11-13-008-video-rule), applied mechanically to {DEV} and {SEALED}", ""]
    for t in TYPES:
        o = order(t)
        rs = {k: json.loads((DEV / k / "result.json").read_text()) for k in o}
        L.append(f"{t}: passing runs in rule order (steps distance from the lower median, then seed): "
                 f"{[(k, rs[k]['steps_used']) for k in o]}\n  first = {o[0]} (steps_used {rs[o[0]]['steps_used']})")
    f = sealed_pick()
    L.append(f"\nsealed failure: type with most failed runs, lower-median steps_used of its failed runs, ties -> lowest seed: {f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def dev() -> bool:
    """Render the first run in rule order whose replay reproduces the log, per type; append the outcome to selection.txt."""
    c3.MAX_CLIP_S = DEV_S
    SEG.mkdir(parents=True, exist_ok=True)
    notes, ok = [], True
    for t in TYPES:
        tasks = json.loads(Path(f"tasksets/phase4_{t}_counted.json").read_text())["tasks"]
        for k in order(t):
            entry = next(e for e in tasks if e["task_id"] == k)
            good = c3.render(OUT / "runs" / k, entry, SEG / f"{t}.mp4", None, CONTEXT, OUT, poses="perceived")
            notes.append(f"{t}: replay of {k} {'reproduced the log: USED' if good else 'DIFFERED from the log: dropped, next in rule order'}")
            if good:
                (SEG / f"{t}.run").write_text(k)
                break
        else:
            ok = False
    with (OUT / "selection.txt").open("a") as f:
        f.write("\nReplay outcome (c3.render, perceived poses; any result/verdict difference drops the run):\n" + "\n".join(notes) + "\n")
    return ok


def fail() -> None:
    k = sealed_pick()
    for f in ("log.jsonl", "result.json"):
        assert (OUT / "runs" / k / f).exists()
    c6.OUT, c6.SHOW, c6.STEP_S = OUT, SEG, FAIL_STEP_S
    draw_text = ImageDraw.ImageDraw.text  # drop the photo file index ("0005_") from the slide label: it is not a caption
    ImageDraw.ImageDraw.text = lambda self, xy, text, *a, **kw: draw_text(self, xy, re.sub(r"photo \d{4}_", "photo ", text), *a, **kw)
    c6.render(k, {t["task_id"]: t for t in map(json.loads, (SEALED / "verified.jsonl").read_text().splitlines()) if t})
    (SEG / "failure.run").write_text(k)
    (SEG / "failure.mp4").unlink(missing_ok=True)
    (SEG / f"{k}.mp4").rename(SEG / "failure.mp4")
    with (OUT / "selection.txt").open("a") as f:
        f.write(f"\nSealed failure rendered from logged artefacts only (no replay): {k}\n")


def card(lines: list[tuple[str, int, bool, tuple[int, int, int]]], x: int = 60, y: int = 40, gap: int = 14) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    for text, size, bold, col in lines:
        f = font(size, bold)
        for ln in wrap(dr, text, f, W - 2 * x) if text else [""]:
            dr.text((x, y), ln, font=f, fill=col)
            y += size + gap // 2
        y += gap
    return img


def title_card() -> Image.Image:
    return card([("TidyBot fetches boxes in a five-room house", 46, True, WHITE), ("", 20, False, WHITE),
                 ("A language-and-vision AI (Qwen3.8-27B) looks at camera photos and object names and chooses the robot's "
                  "skills: search, pick, place, push.", 28, False, (200, 200, 200)),
                 ("Everything here is a MuJoCo simulation.", 28, False, ORG), ("", 20, False, WHITE),
                 ("Next: the final sealed-test score, then 4 practice runs, 1 failure, and the limits.", 26, False, YEL)], y=150)


def table_card() -> Image.Image:
    img = card([("Sealed test: 100 new tasks, one counted pass, no retries", 36, True, WHITE)])
    dr = ImageDraw.Draw(img)
    rows = [("fetch a box", "24 of 25", "likely between 80 and 99 %"), ("blocked door (push the crate first)", "19 of 25", "likely between 57 and 89 %"),
            ("push or detour (cabinet stays put)", "20 of 25", "likely between 61 and 91 %"), ("two boxes", "23 of 25", "likely between 75 and 98 %")]
    y = 110
    for a, b, c in rows:
        dr.text((60, y), a, font=font(27), fill=WHITE)
        dr.text((640, y), b, font=font(27, True), fill=GRN)
        dr.text((800, y), c, font=font(23), fill=(200, 200, 200))
        y += 46
    dr.line((60, y + 4, W - 60, y + 4), fill=(90, 90, 100), width=2)
    dr.text((60, y + 16), "overall", font=font(32, True), fill=WHITE)
    dr.text((640, y + 16), "86 of 100", font=font(32, True), fill=GRN)
    dr.text((60, y + 70), "Ranges are 95 % intervals: with only 25 tasks per type they are wide.", font=font(22), fill=(200, 200, 200))
    f = font(25)
    for i, ln in enumerate(wrap(dr, "The camera finds the boxes; the crate/cabinet positions, place's landing check, push's knowledge of "
                                "nearby boxes and the grip check still come from the simulator.", f, W - 120)):
        dr.text((60, y + 118 + 33 * i), ln, font=f, fill=ORG)
    return img


def closing_card() -> Image.Image:
    return card([("Known limits", 40, True, WHITE),
                 ("- Only 25 tasks per type: the blocked-door and push-or-detour ranges reach below 70 %.", 24, False, (220, 220, 220)),
                 ("- A run counts only if the AI says done within 20 steps (30 for two boxes); 3 of the 14 failures had the box in place and ran out of steps.", 24, False, (220, 220, 220)),
                 ("- Still from the simulator, not the camera: crate and cabinet positions, place's landing check, push's knowledge of nearby boxes, the grip check.", 24, False, (220, 220, 220)),
                 ("- Not shown: a real robot, other houses or objects, or the same result on every re-run (the AI's sampling is random).", 24, False, (220, 220, 220)),
                 ("", 10, False, WHITE),
                 ("Full failure gallery: showcase/failures/INDEX.md (earlier phases + all 14 sealed-test failures, one clip each).", 23, False, ORG),
                 ("Reports and plain summary: see RESULTS.md.", 23, False, ORG)], y=30, gap=10)


def stamp(arr: np.ndarray, text: str, text2: str | None, end: bool) -> np.ndarray:
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    for i, t in enumerate(filter(None, (text, text2))):
        f = font(18, True)
        x, y = W - 10 - int(dr.textlength(t, font=f)), (8 if not end else 664 if text2 is None else 540) + 26 * i
        dr.rectangle((x - 6, y - 3, W - 4, y + 24), fill=(40, 30, 10))
        dr.text((x, y), t, font=f, fill=ORG)
    return np.asarray(img)


def join() -> None:
    parts: list[tuple[str, object]] = [("still", (title_card(), 2.5)), ("still", (table_card(), 8.0))]
    parts += [("clip", (SEG / f"{t}.mp4", f"{PLAIN[t]} practice run, frozen system", None, c3.END_S)) for t in TYPES]
    parts += [("clip", (SEG / "failure.mp4", "1 of the 14 sealed-test failures", "slideshow of logged photos and steps (not re-run)", c6.END_S)),
              ("still", (closing_card(), 8.0))]
    out = Path("showcase/116_showcase.mp4")
    out.parent.mkdir(exist_ok=True)
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for kind, a in parts:
        if kind == "still":
            img, s = a
            fr = np.asarray(img)
            for _ in range(int(s * FPS)):
                w.append_data(fr)
        else:
            p, t1, t2, end_s = a
            rd = imageio.get_reader(p)
            n, n_end = rd.count_frames(), int(end_s * FPS)
            for i, fr in enumerate(rd):
                w.append_data(stamp(fr, t1, t2, end=i >= n - n_end))
    w.close()
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    {"select": select, "dev": lambda: sys.exit(0 if dev() else 1), "fail": fail, "cards": lambda: None, "join": join}[sys.argv[1]]()
