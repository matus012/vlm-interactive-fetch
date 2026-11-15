"""E4 C1 gate (task 014): are the printed box labels legible from the room point, and does locate still work?

    MUJOCO_GL=egl NAV116_HOUSE=A python -m sim.e4_c1_gate legibility --out DIR   # (a), per house (run once per house)
    MUJOCO_GL=egl NAV116_HOUSE=A python -m sim.e4_c1_gate pose --out DIR         # (b), G5-pose with labels on

(a) Legibility, labels on, dev seeds 1000-1199, every box: robot at the first room point of the box's room
(room_points order), LOOK_ARM_QPOS, base yaw at each look heading (set_yaw_direct; no physics step, the look_px
pattern). Per photo: segmentation px of the box geom, and the letter height = pixel distance between the projected
top and bottom of the word's ink band (label_band, at the band's centre along the reading direction) on the labelled
face most turned to the camera (max cos between the outward face normal and the face-centre-to-camera direction;
faces +x, -x, +z; 0 if that cos <= 0 or either point projects outside the image). Best heading = max letter height
(ties: more seg px). Pass = letter height >= LETTER_PX_MIN and seg px >= SEG_PX_MIN in that photo. Bar >= BAR_FRAC.
Example photos (house A only): the first EXAMPLES placements (distinct words, seg px >= SEG_PX_MIN, pass or not) with camera-box distance in the G5-pose mid band.
(b) sim.pose_check's G5-pose run (seeds 9800-9999, 50 views, >= 45/50, fused 4 photos), unchanged, with labels on.
Pixel measurements need the NVIDIA renderer (asserted).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from sim.constants import BOX_HALF, G1_COVER_MIN_PX
from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, set_arm, set_yaw_direct
from sim.perception import project
from sim.robot import CAM_H, CAM_W, MAST_CAM, camera_intrinsics
from sim.seal import gl_renderer
from sim.world import generate as gen
from sim.world.house import load_house, set_base

SEEDS = range(1000, 1200)
LETTER_PX_MIN = 12.0  # task 014 spec
SEG_PX_MIN = G1_COVER_MIN_PX  # 100 px, the in-view bar
BAR_FRAC = 0.90  # task 014 spec
MID_BAND = (1.6, 2.4)  # sim.pose_check BANDS "mid"
EXAMPLES = 4


def band_points(word: str, face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-frame (top, bottom) points of the word's ink band at its centre along the reading direction, and the
    face's outward normal. Face mapping as in sim.world.generate (label_band docstring)."""
    hx, hy, hz = BOX_HALF
    a0, a1, b0, b1 = gen.label_band(word, "top" if face == "+z" else "side")
    a = (a0 + a1) / 2
    if face == "+x":
        pts = [np.array([hx, -hy + a * 2 * hy, hz - b * 2 * hz]) for b in (b0, b1)]
        n = np.array([1.0, 0, 0])
    elif face == "-x":
        pts = [np.array([-hx, hy - a * 2 * hy, hz - b * 2 * hz]) for b in (b0, b1)]
        n = np.array([-1.0, 0, 0])
    else:
        pts = [np.array([-hx + b * 2 * hx, -hy + a * 2 * hy, hz]) for b in (b0, b1)]
        n = np.array([0, 0, 1.0])
    return pts[0], pts[1], n


def letter_px(word: str, pos: np.ndarray, R: np.ndarray, K: np.ndarray, cam_pos: np.ndarray,
              cam_mat: np.ndarray) -> tuple[float, str, float, float]:
    """(letter height px, face, cos) on the labelled face most turned to the camera (the gate), plus (info) the max
    letter height over every labelled face turned to the camera (cos > 0)."""
    faces = []
    for face in ("+x", "-x", "+z"):
        top, bot, n = band_points(word, face)
        centre = pos + R @ (n * np.array(BOX_HALF))
        v = cam_pos - centre
        c = float((R @ n) @ v / np.linalg.norm(v))
        uv = project(np.stack([pos + R @ top, pos + R @ bot]), K, cam_pos, cam_mat)
        inside = bool(((uv[:, 0] >= 0) & (uv[:, 0] < CAM_W) & (uv[:, 1] >= 0) & (uv[:, 1] < CAM_H)).all())
        faces.append((face, c, float(np.linalg.norm(uv[0] - uv[1])) if c > 0 and inside else 0.0))
    face, c, lh = max(faces, key=lambda f: f[1])
    return lh, face, c, max(f[2] for f in faces)


def legibility(out: Path, seeds: range = SEEDS) -> dict[str, Any]:
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
    for seed in seeds:
        w = gen.generate(seed=seed, blocker="none")
        m, d = load_house(w)
        K = camera_intrinsics(m)
        cid = m.camera(MAST_CAM).id
        r = mujoco.Renderer(m, CAM_H, CAM_W)
        try:
            for b in w.boxes:
                rx, ry = gen.room_points(b["room"])[0]
                set_base(m, d, (rx, ry, 0.0))
                set_arm(m, d, LOOK_ARM_QPOS)
                g = m.geom(b["name"]).id
                body = d.body(b["name"])
                per = []
                for h in HEADINGS_DEG:
                    set_yaw_direct(m, d, math.radians(h))
                    r.enable_segmentation_rendering()
                    r.update_scene(d, camera=MAST_CAM)
                    s = r.render()
                    r.disable_segmentation_rendering()
                    seg = int(((s[..., 1] == geom_t) & (s[..., 0] == g)).sum())
                    cam_pos, cam_mat = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
                    lh, face, c, lh_any = letter_px(b["label"]["word"], body.xpos.copy(), body.xmat.reshape(3, 3).copy(),
                                            K, cam_pos, cam_mat)
                    per.append({"heading": h, "seg_px": seg, "letter_px": round(lh, 2), "face": face,
                                "letter_px_any_face": round(lh_any, 2),
                                "cos": round(c, 3), "dist": float(np.hypot(*(cam_pos[:2] - body.xpos[:2])))})
                k = max(range(4), key=lambda i: (per[i]["letter_px"], per[i]["seg_px"]))
                bh = per[k]
                ok = bh["letter_px"] >= LETTER_PX_MIN and bh["seg_px"] >= SEG_PX_MIN
                any_ok = any(p["letter_px"] >= LETTER_PX_MIN and p["seg_px"] >= SEG_PX_MIN for p in per)
                any_face_ok = any(p["letter_px_any_face"] >= LETTER_PX_MIN and p["seg_px"] >= SEG_PX_MIN for p in per)
                rows.append({"seed": seed, "colour": b["colour"], "word": b["label"]["word"], "room": b["room"],
                             "surface": b["surface"], "room_point": [rx, ry], "best_heading": bh["heading"],
                             "letter_px": bh["letter_px"], "seg_px": bh["seg_px"], "face": bh["face"],
                             "distance": round(bh["dist"], 3), "pass": ok, "pass_any_heading": any_ok,
                             "info_pass_any_face": any_face_ok,
                             "per_heading": per})
                if (gen.HOUSE == "A" and bh["seg_px"] >= SEG_PX_MIN and len(examples) < EXAMPLES and MID_BAND[0] <= bh["dist"] < MID_BAND[1]
                        and b["label"]["word"] not in {e["word"] for e in examples}):
                    set_yaw_direct(m, d, math.radians(bh["heading"]))
                    r.update_scene(d, camera=MAST_CAM)
                    rgb = r.render().copy()
                    u, v = project(body.xpos[None].copy(), K, d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3))[0]
                    name = f"example_{len(examples)}_s{seed}_{b['colour']}_{b['label']['word']}.png"
                    full = Image.fromarray(rgb)
                    crop = full.crop((int(u) - 60, int(v) - 45, int(u) + 60, int(v) + 45)).resize((640, 480),
                                                                                               Image.NEAREST)
                    sheet = Image.new("RGB", (1280, 480))
                    sheet.paste(full, (0, 0))
                    sheet.paste(crop, (640, 0))
                    sheet.save(out / name)
                    full.save(out / name.replace(".png", "_raw.png"))
                    examples.append({"file": name, "seed": seed, "colour": b["colour"], "word": b["label"]["word"],
                                     "heading": bh["heading"], "distance": round(bh["dist"], 3),
                                     "letter_px": bh["letter_px"], "face": bh["face"]})
        finally:
            r.close()
    n = len(rows)
    k = sum(r["pass"] for r in rows)
    lh = np.array([r["letter_px"] for r in rows])
    by_word = {wd: {"n": len(rs), "pass": sum(r["pass"] for r in rs),
                    "letter_px_median": float(np.median([r["letter_px"] for r in rs]))}
               for wd in sorted({r["word"] for r in rows}) for rs in [[r for r in rows if r["word"] == wd]]}
    res = {"gate": "E4-C1 legibility (a)", "house": gen.HOUSE, "renderer": gl_renderer(), "seeds": [seeds[0], seeds[-1]],
           "letter_px_min": LETTER_PX_MIN, "seg_px_min": SEG_PX_MIN, "bar_frac": BAR_FRAC, "n": n, "pass": k,
           "frac": k / n, "verdict": "PASS" if k / n >= BAR_FRAC else "FAIL",
           "pass_any_heading": sum(r["pass_any_heading"] for r in rows),
           "info_pass_any_face_any_heading": sum(r["info_pass_any_face"] for r in rows),
           "fail_seg_lt_min": sum(r["seg_px"] < SEG_PX_MIN for r in rows),
           "fail_letter_lt_min": sum(r["letter_px"] < LETTER_PX_MIN for r in rows),
           "letter_px": {"median": float(np.median(lh)), "p10": float(np.percentile(lh, 10)),
                         "p90": float(np.percentile(lh, 90)), "max": float(lh.max())},
           "by_word": by_word, "examples": examples, "seconds": round(time.time() - t0), "rows": rows}
    (out / f"legibility_{gen.HOUSE}.json").write_text(json.dumps(res, indent=1))
    print(f"house {gen.HOUSE}: {k}/{n} = {k / n:.3f} pass (bar {BAR_FRAC}) -> {res['verdict']}; "
          f"any-heading {res['pass_any_heading']}; info any-face {res['info_pass_any_face_any_heading']}; seg<{SEG_PX_MIN}: {res['fail_seg_lt_min']}, "
          f"letter<{LETTER_PX_MIN}: {res['fail_letter_lt_min']}; letter px {res['letter_px']}; {res['seconds']} s")
    for wd, v in by_word.items():
        print(f"  {wd}: {v}")
    return res


def pose(out: Path) -> None:
    from sim import pose_check

    sys.argv = ["pose_check", "--seeds", "9800:10000", "--out", str(out / f"g5pose_{gen.HOUSE}")]
    pose_check.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("legibility", "pose"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1000:1200", help="legibility seed range (smoke runs only; the gate is 1000:1200)")
    a = ap.parse_args()
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    assert 1000 <= s0 < s1 <= 10000
    assert "NVIDIA" in gl_renderer(), gl_renderer()
    gen.set_labels(True)
    a.out.mkdir(parents=True, exist_ok=True)
    legibility(a.out, range(s0, s1)) if a.part == "legibility" else pose(a.out)


if __name__ == "__main__":
    main()
