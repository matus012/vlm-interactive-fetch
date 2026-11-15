"""Run scorer (brief §One run "A run passes only if all hold"), from a sim-state snapshot. Stdlib only (G2-arch).

    score(snapshot, task, steps_used) -> {"pass", "rules", "failed", "detail"}

Rules, in the fixed order of `failed`:
- surface   every target box rests on its target surface: centre xy inside the surface rect shrunk by
            SCORE_SURFACE_MARGIN on every side; box bottom (centre z - BOX_HALF z, i.e. upright) within
            SCORE_ON_TOP_TOL of top_z; tilt <= SCORE_UPRIGHT_DEG; still: max displacement <= SCORE_STILL_DISP and
            max speed <= SCORE_STILL_SPEED over a settle window of >= SCORE_STILL_S sim seconds.
- wrong_box no box that is not targeted at a target surface ends on that surface. "On" = the snapshot's support
            is that surface, or the centre is inside the (unshrunk) rect and above top_z (tilt-proof: a box lying
            on its side on a table is still on it). In a two-box task the other task's box counts as wrong on a
            surface it is not targeted at.
- held      nothing held at the end (snapshot held is None; the adapter's held is conservative: between the pads OR
            any pad contact, deviation 18). detail also carries snapshot "holding" (between the pads) for the labeller.
- steps     steps_used <= task step_limit.
- sim       snapshot sim_ok and every box number finite (no NaN / blow-up).
Snapshot schema: agent/robot_api.py. Task schema: sim/tasks.py.
"""
from __future__ import annotations

import math

from sim.constants import (BOX_HALF, SCORE_ON_TOP_TOL, SCORE_STILL_DISP, SCORE_STILL_S, SCORE_STILL_SPEED,
                           SCORE_SURFACE_MARGIN, SCORE_UPRIGHT_DEG)

RULES = ("surface", "wrong_box", "held", "steps", "sim")
BOX_HALF_Z = BOX_HALF[2]


def on_target(box: dict, surf: dict, settle_s: float) -> tuple[bool, dict]:
    x, y, z = box["pos"]
    x0, y0, x1, y1 = surf["rect"]
    m = SCORE_SURFACE_MARGIN
    inside = x0 + m <= x <= x1 - m and y0 + m <= y <= y1 - m
    gap = z - BOX_HALF_Z - surf["top_z"]
    on_top = abs(gap) <= SCORE_ON_TOP_TOL
    upright = box["tilt_deg"] <= SCORE_UPRIGHT_DEG
    still = (box["max_disp_settle_m"] <= SCORE_STILL_DISP and box["max_speed_settle"] <= SCORE_STILL_SPEED
             and settle_s >= SCORE_STILL_S)
    edge = min(x - x0, x1 - x, y - y0, y1 - y)  # distance of the centre to the nearest rect edge (m)
    return inside and on_top and upright and still, {
        "edge_m": edge, "bottom_gap_m": gap, "tilt_deg": box["tilt_deg"], "inside": inside, "on_top": on_top,
        "upright": upright, "still": still, "max_disp_settle_m": box["max_disp_settle_m"],
        "max_speed_settle": box["max_speed_settle"]}


def on_surface(box: dict, name: str, surf: dict) -> bool:
    x, y, z = box["pos"]
    x0, y0, x1, y1 = surf["rect"]
    return box.get("support") == name or (x0 <= x <= x1 and y0 <= y <= y1 and z > surf["top_z"])


def score(snapshot: dict, task: dict, steps_used: int) -> dict:
    boxes, surfaces, settle_s = snapshot["boxes"], snapshot["surfaces"], snapshot["settle_s"]
    targets = {}
    surface_ok = True
    for t in task["targets"]:
        ok, det = on_target(boxes[t["object"]], surfaces[t["surface"]], settle_s)
        targets[t["object"]] = dict(det, surface=t["surface"], ok=ok)
        surface_ok = surface_ok and ok
    wrong = []
    for s in dict.fromkeys(t["surface"] for t in task["targets"]):
        meant = {t["object"] for t in task["targets"] if t["surface"] == s}
        wrong += [f"{b}@{s}" for b, v in boxes.items() if b not in meant and on_surface(v, s, surfaces[s])]
    finite = all(math.isfinite(v) for b in boxes.values()
                 for v in (*b["pos"], b["tilt_deg"], b["max_disp_settle_m"], b["max_speed_settle"]))
    rules = {"surface": surface_ok, "wrong_box": not wrong, "held": snapshot["held"] is None,
             "steps": steps_used <= task["step_limit"], "sim": bool(snapshot["sim_ok"]) and finite}
    failed = [r for r in RULES if not rules[r]]
    return {"pass": not failed, "rules": rules, "failed": failed,
            "detail": {"targets": targets, "wrong_boxes": wrong, "held": snapshot["held"], "holding": snapshot.get("holding"), "steps_used": steps_used,
                       "step_limit": task["step_limit"], "settle_s": settle_s, "finite": finite}}
