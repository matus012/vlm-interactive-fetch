"""Phase 3 clips (task 004): replay logged VLM runs through the same runtime and sim, render them.

    MUJOCO_GL=egl python -m sim.clip_phase3 [select|render]   (default: both)

Rule: the pre-registered rule (2026-11-13-004-clip-rule). `select` writes runs/clip3/selection.txt. `render` replays each chosen run
exactly as sim/agent_run.py runs it (make_task -> SimRobot -> Skills -> Runtime(scripted=False) -> run_agent), with a
StubVLM that returns the logged tool calls of every step (zero / several / invalid ones too). Frames come from wrapping
mujoco.mj_step. The replay is compared with the logged run (call, args, validation, result, ok, sim_s; scorer verdict,
rules, label); any result/verdict difference -> no clip (exit 1). sim_s differences are reported (max abs), not blocking.
One uniform speed-up per clip, chosen so the clip is <= 30 s; nothing cut.
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.loop import end_run, run_agent
from agent.runtime import Runtime
from agent.skills import Skills
from agent.vlm import StubVLM
from sim.adapter import SimRobot
from sim.agent_run import TASK_KEYS
from sim.robot import MAST_CAM
from sim.tasks import make_task

RUNS = Path("runs/004")
OUT = Path("runs/clip3")
W, H, FPS = 1280, 720, 30
OV_W, OV_H, MAST_W, MAST_H, TOP = 840, 630, 440, 330, 45
END_S, MAX_CLIP_S = 3.5, 30.0
PH_W, PH_H = 104, 78
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CONTEXT = "counted run 1: 9/10 passed (fetch-6000 placed the box at step 20 but never called done: step limit)"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def fmt(name: str, args: Any) -> str:
    if not name:
        return "(no single tool call)"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return f"{name}({args})"
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def fit(dr: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> str:
    while dr.textlength(text, font=f) > width and len(text) > 1:
        text = text[:-2] + "…"
    return text


def wrap(dr: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and dr.textlength(cur + " " + w, font=f) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def excerpt(reasoning: str | None, n: int = 120) -> str:
    """Verbatim slice of the VLM's reasoning: its last sentence (the one before it too when that is very short),
    cut at n chars (the end kept: that is where the decision is)."""
    if not reasoning or not reasoning.strip():
        return "(no reasoning logged)"
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", reasoning.strip()) if s.strip()]
    t = sents[-1]
    if len(t) < 30 and len(sents) > 1:
        t = sents[-2] + " " + t
    return t if len(t) <= n else "…" + t[-(n - 1):].split(" ", 1)[-1]


def load(task_dir: Path) -> tuple[list[dict], dict, dict]:
    lines = [json.loads(x) for x in (task_dir / "log.jsonl").read_text().splitlines() if x.strip()]
    return lines[:-1], lines[-1], json.loads((task_dir / "result.json").read_text())


def all_results() -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) | {"set": p.parent.parent.name, "dir": str(p.parent)}
            for p in sorted(RUNS.glob("*1/fetch-*/result.json"))}


def select() -> dict:
    res = all_results()
    counted = {k: v for k, v in res.items() if v["set"] == "count1"}
    tune = {k: v for k, v in res.items() if v["set"] == "tune1"}
    passed = sorted((v["steps_used"], v["seed"], k) for k, v in counted.items() if v["pass"])
    med = passed[(len(passed) - 1) // 2][0]  # lower median
    main = min((s, k) for st, s, k in passed if st == med)[1]
    fails_c = [k for k, v in counted.items() if not v["pass"]]
    fails_t = [k for k, v in tune.items() if not v["pass"]]
    L = [f"Selection per the pre-registered rule (2026-11-13-004-clip-rule), applied mechanically to {RUNS}",
         f"counted runs: {len(counted)}, passed {len(passed)}; tuning runs: {len(tune)}, passed {len(tune) - len(fails_t)}",
         f"passed counted runs, steps_used sorted: {[(st, k) for st, s, k in passed]}",
         f"n passed = {len(passed)} (odd): median = element {(len(passed) - 1) // 2 + 1} of the sorted steps_used = {med}",
         f"ties at {med}: {[k for st, s, k in passed if st == med]} -> lowest seed -> MAIN = {main}", "",
         f"failed counted: {fails_c}; failed tuning: {fails_t} -> failure clips"]
    for k, v in sorted(res.items(), key=lambda kv: (kv[1]["set"], kv[0])):
        if k == main and v["set"] == "count1":
            why = "CHOSEN main clip (lowest-seed run at the median steps_used)"
        elif not v["pass"]:
            why = f"CHOSEN failure clip (failed {v['failed']}, label {v['label']}, steps {v['steps_used']})"
        elif v["set"] == "tune1":
            why = "not chosen: tuning run that passed (rule: only failed tuning runs get a clip)"
        elif v["steps_used"] == med:
            why = f"not chosen: steps_used {med} = median but seed {v['seed']} > {main.split('-')[1]} (tie -> lowest seed)"
        else:
            why = f"not chosen: passed, steps_used {v['steps_used']} != median {med}"
        L.append(f"{v['set']}/{k}: pass={v['pass']} steps={v['steps_used']}  {why}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return {"main": main, "fails": [("count1", k) for k in fails_c] + [("tune1", k) for k in fails_t]}


def render(d_run: Path, entry: dict, out: Path, note: str | None, context: str = CONTEXT, outdir: Path = OUT,
           poses: str = "truth", card: Any = None, card_s: float = 0.0) -> bool:
    """Phase 4 (sim/clip_phase4.py) calls this with its own run dir / taskset entry / context line / output dir.
    Phase 5 (sim/clip_phase5.py): poses="perceived"; card(robot, rt, name, ok) -> a 1280x720 RGB array or None, shown for
    card_s s after each call (frozen sim; the budget of card_s per card is the caller's: it passes the total as extra)."""
    tid = entry["task_id"]
    steps_log, end_log, res = load(d_run)
    task = make_task(entry["seed"], entry["type"], want=entry.get("slot"))
    assert {k: task[k] for k in TASK_KEYS} == {k: entry[k] for k in TASK_KEYS}
    n_cards = sum(1 for x in steps_log if x["call"] == "locate") if card is not None else 0
    card_s_each, card_s = card_s, card_s * n_cards
    total_sim = res["timing"]["sim_s"] + 2.0
    speed = math.ceil(total_sim / (MAX_CLIP_S - END_S - 1.5 - card_s))
    tdir = Path(tempfile.mkdtemp())
    robot = SimRobot(task["seed"], task["blocker"], photo_dir=tdir / "photos", poses=poses)
    m, d = robot.m, robot.d
    rt = Runtime(Skills(robot), robot, task, tdir / "log.jsonl", scripted=False)
    r_ov, r_mast = mujoco.Renderer(m, OV_H, OV_W), mujoco.Renderer(m, MAST_H, MAST_W)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    f_big, f_mid, f_small, f_hist = font(22, True), font(19), font(15), font(13)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = m.cam_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overview")]
    cam.lookat[2] = 0
    cam.distance, cam.elevation, cam.azimuth = 10.4, -90, 90
    st: dict[str, Any] = {"k": 0, "cur": "(starting)", "res": None, "why": "", "ph": [], "hist": [], "next_t": d.time, "n": 0}
    n_steps = len(steps_log)
    target_obj = " + ".join(t["object"] for t in task["targets"])
    blk = robot.world.blocker["body"] if robot.world.blocker["variant"] != "none" else None
    blk_rgb = {"crate": (0.7, 0.2, 1.0), "cabinet": (0.9, 0.9, 0.9)}.get(blk)

    def frame() -> np.ndarray:
        r_ov.update_scene(d, camera=cam)
        s = r_ov.scene  # scene-only halo on the target box (not in the physics, not in the mast view)
        halos = [(robot.body_of[t["object"]], 0.22, (1, 1, 0)) for t in task["targets"]]
        if blk:
            halos.append((blk, 0.5, blk_rgb))
        for body, rad, rgb in halos:
            g = s.geoms[s.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), np.zeros(3), np.eye(3).ravel(),
                                np.array([*rgb, 0.35 if body != blk else 0.5], dtype=np.float32))
            g.pos[:] = d.body(body).xpos
            s.ngeom += 1
        ov = r_ov.render()
        r_mast.update_scene(d, camera=MAST_CAM)
        img = Image.new("RGB", (W, H), (18, 18, 22))
        img.paste(Image.fromarray(ov), (0, TOP))
        img.paste(Image.fromarray(r_mast.render()), (OV_W, TOP))
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), "Task: " + task["instruction"], font=f_big, fill=(240, 240, 240))
        dr.text((10, TOP + 4), f"overview (top-down)  yellow halo = {target_obj}", font=f_small, fill=(255, 255, 0),
                stroke_width=2, stroke_fill=(0, 0, 0))
        if blk:
            dr.text((10, TOP + 24), f"{'purple' if blk == 'crate' else 'grey'} halo = the {blk} (blocker)", font=f_small,
                    fill=tuple(int(255 * c) for c in blk_rgb), stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((OV_W + 6, TOP + 4), "mast camera (live)", font=f_small, fill=(255, 255, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        dr.text((OV_W - 150, TOP + 4), f"speed x{speed}", font=f_big, fill=(255, 140, 0), stroke_width=2,
                stroke_fill=(0, 0, 0))
        x0, y = OV_W + 6, TOP + MAST_H + 6
        dr.text((x0, y), f"photos the VLM was given at step {st['k']}" if st["ph"] else "photos the VLM was given: none yet",
                font=f_small, fill=(255, 255, 0))
        for i, p in enumerate(st["ph"]):
            img.paste(p, (OV_W + 6 + i * (PH_W + 4), y + 22))
        y += 22 + PH_H + 8
        dr.text((x0, y), "VLM's reasoning (verbatim excerpt):", font=f_small, fill=(255, 255, 0))
        for i, ln in enumerate(wrap(dr, "“" + st["why"] + "”", f_small, MAST_W - 12)[:3]):
            dr.text((x0, y + 20 + 18 * i), ln, font=f_small, fill=(170, 200, 255))
        y += 20 + 18 * 3 + 6
        for i, (c, r) in enumerate(st["hist"][-4:]):
            dr.text((x0, y + 17 * i), fit(dr, f"{c} -> {r}", f_hist, MAST_W - 12), font=f_hist, fill=(160, 255, 160))
        dr.rectangle((0, 675, W, H), fill=(30, 30, 40))
        line = f"{st['k']}/{n_steps}. {st['cur']} -> " if st["k"] else "(starting)"
        dr.text((10, 680), line, font=f_mid, fill=(240, 240, 240))
        if st["res"] is not None:
            dr.text((10 + dr.textlength(line, font=f_mid), 680), st["res"], font=f_mid, fill=(160, 255, 160))
        elif st["k"]:
            dr.text((10 + dr.textlength(line, font=f_mid), 680), "...", font=f_mid, fill=(255, 200, 120))
        dr.text((10, 702), f"sim t = {d.time:6.1f} s", font=f_small, fill=(170, 170, 170))
        return np.asarray(img)

    orig_step = mujoco.mj_step

    def step(mm: mujoco.MjModel, dd: mujoco.MjData, *a: Any, **kw: Any) -> None:
        orig_step(mm, dd, *a, **kw)
        if dd is d and dd.time >= st["next_t"]:
            writer.append_data(frame())
            st["n"] += 1
            st["next_t"] += speed / FPS

    def hook(orig):
        def wrapped(*a: Any, **kw: Any):
            k = len(rt.steps)
            if k < n_steps:  # logged step k+1: the photos and reasoning that belong to this call
                ex = steps_log[k]["extra"]
                st["ph"] = [Image.open(d_run / "photos" / Path(p).name).convert("RGB").resize((PH_W, PH_H))
                            for p in ex["packet"]["photos"] if p]
                st["why"] = excerpt(ex["vlm"]["reasoning"])
            name, args = (a[0], a[1]) if orig == orig_call else ("", {})
            st.update(k=k + 1, cur=fmt(name, args), res=None)
            r = orig(*a, **kw)
            st.update(res=r.text)
            st["hist"].append((f"{k + 1}. {st['cur']}", r.text))
            writer.append_data(frame())
            st["n"] += 1
            if card is not None and orig == orig_call and (c := card(robot, rt, name, r)) is not None:
                for _ in range(int(card_s_each * FPS)):
                    writer.append_data(c)
                    st["n"] += 1
            return r
        return wrapped

    orig_call, orig_err = rt.call, rt.error_step
    rt.call, rt.error_step = hook(orig_call), hook(orig_err)
    items = [s["extra"]["vlm"]["tool_calls"] for s in steps_log]
    mujoco.mj_step = step
    try:
        run_agent(robot, rt.skills, rt, task, StubVLM(items), tdir)
        end = end_run(robot, rt, task)
    finally:
        mujoco.mj_step = orig_step
    sc = end["score"]

    # ---- replay check against the logged run
    mine = rt.steps
    rows, same, dts = [], len(mine) == n_steps, []
    for i in range(max(n_steps, len(mine))):
        L = steps_log[i] if i < n_steps else None
        R = mine[i] if i < len(mine) else None
        key = lambda s: None if s is None else (s["call"], s["args"], s["validation"], s["result"], s["ok"])
        eq = key(L) == key(R)
        same &= eq
        dt = abs(L["sim_s"] - R["sim_s"]) if L and R else float("nan")
        dts.append(dt)
        rows.append(f"{i + 1:2d} {'MATCH' if eq else 'DIFF '} sim_s diff {dt:.2e}\n   log: {key(L)}\n   run: {key(R)}")
    ls = end_log["score"]
    verdict_same = (sc["pass"], sc["failed"], sc["rules"], end["label"], end["steps_used"]) == (
        ls["pass"], ls["failed"], ls["rules"], end_log["label"], end_log["steps_used"])
    same &= verdict_same
    mx = max(dts) if dts else 0.0
    head = (f"replay check {tid}: {'IDENTICAL' if same else 'DIFFERENT'} ({n_steps} logged, {len(mine)} re-run); "
            f"result strings+calls+args+validation {'equal' if all(r.split()[1] == 'MATCH' for r in rows) else 'DIFFER'}; "
            f"verdict logged pass={ls['pass']} failed={ls['failed']} label={end_log['label']} steps={end_log['steps_used']} | "
            f"replay pass={sc['pass']} failed={sc['failed']} label={end['label']} steps={end['steps_used']} "
            f"({'same' if verdict_same else 'DIFFERENT'}); sim_s max abs diff {mx:.3e} s "
            f"({'<= 1e-6 ok' if mx <= 1e-6 else 'ABOVE 1e-6: mismatch to report'}); "
            f"sim total {d.time:.2f} s, speed x{speed}")
    outdir.mkdir(parents=True, exist_ok=True)
    chk = outdir / f"{tid}_replay.txt"
    chk.write_text(head + "\n" + "\n".join(rows) + "\n")
    print(head)
    if not same:
        writer.close()
        tmp.unlink()
        return False

    # ---- end card
    img = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    dr.text((60, 40), "Task: " + task["instruction"], font=font(30, True), fill=(240, 240, 240))
    dr.text((60, 100), f"Scorer verdict: {'PASS' if sc['pass'] else 'FAIL'}", font=font(44, True),
            fill=(120, 255, 120) if sc["pass"] else (255, 110, 110))
    dr.text((60, 175), "Rules checked:", font=font(26), fill=(255, 255, 0))
    expl = {"surface": "every target box rests on the target surface (upright, still)", "wrong_box": "no other box on that surface",
            "held": "nothing held at the end",
            "steps": f"steps used {sc['detail']['steps_used']} <= limit {task['step_limit']}"
                     + ("" if end["done_called"] else " (done was never called)"),
            "sim": "simulation healthy (no blow-up / NaN)"}
    for i, (k, v) in enumerate(sc["rules"].items()):
        dr.text((80, 220 + 38 * i), f"{'ok ' if v else 'FAIL'}  {k}: {expl[k]}", font=font(24),
                fill=(160, 255, 160) if v else (255, 110, 110))
    dr.text((60, 430), f"Run: {len(mine)} calls by Qwen3.8-27B, {d.time:.0f} s of sim time, shown at x{speed} (uniform, nothing cut)",
            font=font(21), fill=(200, 200, 200))
    if note:
        dr.text((60, 540), note, font=font(22), fill=(255, 255, 255))
    dr.text((60, 600), fit(dr, context, font(21), W - 80), font=font(21), fill=(255, 190, 90))
    arr = np.asarray(img)
    for _ in range(int(END_S * FPS)):
        writer.append_data(arr)
        st["n"] += 1
    writer.close()
    tmp.replace(out)
    print(f"{out}: {st['n']} frames, {st['n'] / FPS:.1f} s, speed x{speed}, {out.stat().st_size / 1e6:.2f} MB")
    return True


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    sel = select()
    if mode == "select":
        return
    only = sys.argv[2:]
    ts = {n: json.loads(Path(f"tasksets/phase3_{n}.json").read_text())["tasks"] for n in ("counted", "tuning")}
    ent = lambda s, k: next(t for t in ts["counted" if s == "count1" else "tuning"] if t["task_id"] == k)
    jobs = [("count1", sel["main"], Path("showcase/phase3_vlm_fetch.mp4"), None)]
    for s, k in sel["fails"]:
        jobs.append((s, k, Path(f"showcase/failures/{k}.mp4"),
                     None if s == "count1" else "(tuning run of job 159661, not a counted run)"))
    ok = True
    for s, k, out, note in jobs:
        if only and k not in only:
            continue
        ok &= render(RUNS / s / k, ent(s, k), out, note)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
