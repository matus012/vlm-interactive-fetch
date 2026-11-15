"""G1-cover: can the 4-heading look from a room point see a fetch box at every spot the generator can draw?

    python -m sim.cover_check --out runs/1c [--extra-point ROOM:X:Y]

One house (seed 0, no blocker) with a single fetch box (the other boxes are removed); the box is moved per case by
qpos + mj_forward. Robot at the room point, arm in LOOK_ARM_QPOS (or `retract`), base yaw set directly to
0/90/180/270; per photo the pixels whose segmentation geom id is the box geom are counted.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim.constants import G1_COVER_GRID, G1_COVER_MIN_PX, G1_COVER_YAWS_DEG
from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, render_look, set_arm, set_yaw_direct
from sim.robot import ARM_JOINTS, CAM_H, CAM_W
from sim.world.generate import LAYOUT, generate, load_layout, load_places, room_point, room_points, spawn_spots
from sim.world.house import load_house, set_base

ROOMS = tuple(load_layout()["spawn_regions"])  # layout order; house A: bedroom, storage, living_room, kitchen
BOX = "box_red"


def single_box_world():
    """Seed-0 world with only BOX in it."""
    w = generate(LAYOUT, 0, "none")
    for b in w.boxes:
        if b["name"] != BOX:
            w.xml = re.sub(rf'    <body name="{b["name"]}".*?</body>\n', "", w.xml, flags=re.S)
    return w


class Cover:
    def __init__(self) -> None:
        self.m, self.d = load_house(single_box_world())
        self.retract = tuple(float(v) for v in self.d.qpos[[self.m.joint(j).qposadr[0] for j in ARM_JOINTS]])
        self.gid = self.m.geom(BOX).id
        self.qadr = self.m.joint(self.m.body(BOX).jntadr[0]).qposadr[0]
        self.r = mujoco.Renderer(self.m, CAM_H, CAM_W)

    def place(self, x: float, y: float, z: float, yaw: float) -> None:
        self.d.qpos[self.qadr: self.qadr + 7] = (x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))

    def stand(self, pt: tuple[float, float], look: bool) -> None:
        set_base(self.m, self.d, (pt[0], pt[1], 0.0))
        set_arm(self.m, self.d, LOOK_ARM_QPOS if look else self.retract)
        mujoco.mj_forward(self.m, self.d)

    def count(self) -> int:
        """Box pixels in the mast view at the current state (segmentation = geom id)."""
        _, _, seg = render_look(self.r, self.d)
        return int((seg == self.gid).sum())

    def px(self, pt: tuple[float, float], cases: list[tuple[float, float, float, float]], look: bool) -> np.ndarray:
        """cases = (x, y, z, yaw) -> px[case, heading]."""
        self.stand(pt, look)
        out = np.zeros((len(cases), len(HEADINGS_DEG)), dtype=int)
        for j, h in enumerate(HEADINGS_DEG):
            set_yaw_direct(self.m, self.d, math.radians(h))
            for i, (x, y, z, yaw) in enumerate(cases):
                self.place(x, y, z, yaw)
                mujoco.mj_forward(self.m, self.d)
                out[i, j] = self.count()
        return out

    def photos(self, pt: tuple[float, float], case: tuple[float, float, float, float]) -> list[np.ndarray]:
        self.stand(pt, True)
        self.place(*case)
        out = []
        for h in HEADINGS_DEG:
            set_yaw_direct(self.m, self.d, math.radians(h))
            out.append(render_look(self.r, self.d)[0])
        return out


def room_cases(layout: dict, room: str) -> tuple[list[tuple], list[tuple[float, float, float, float]]]:
    spots = spawn_spots(layout, room, G1_COVER_GRID)
    cases = [(x, y, z, math.radians(yd)) for (x, y, z, _) in spots for yd in G1_COVER_YAWS_DEG]
    return spots, cases


def summarize(layout: dict, room: str, px: np.ndarray, spots: list[tuple]) -> dict:
    ny = len(G1_COVER_YAWS_DEG)
    mx = px.max(axis=1)
    fails = [{"spot": list(spots[i // ny][:3]), "surface": spots[i // ny][3], "yaw_deg": G1_COVER_YAWS_DEG[i % ny],
              "px": px[i].tolist()} for i in range(len(mx)) if mx[i] < G1_COVER_MIN_PX]
    return {"n_spots": len(spots), "n_cases": len(mx), "n_pass": int((mx >= G1_COVER_MIN_PX).sum()),
            "min_max_px": int(mx.min()), "failing": fails}


def plot(layout: dict, room: str, spots: list[tuple], mx: np.ndarray, pt: tuple[float, float], path: Path,
         extra: list[tuple[float, float]] = ()) -> None:
    """Top-down PNG: whole room, spots coloured by min-over-yaws of max-px (red < 100, then dark->bright green)."""
    x0, y0, x1, y1 = layout["rooms"][room]["rect"]
    S = 160
    img = Image.new("RGB", (int((x1 - x0) * S) + 20, int((y1 - y0) * S) + 20), "white")
    dr = ImageDraw.Draw(img)
    P = lambda x, y: (10 + (x - x0) * S, 10 + (y1 - y) * S)
    dr.rectangle([*P(x0, y1), *P(x1, y0)], outline="black")
    for f in layout["furniture"]:
        if f["room"] == room:
            cx, cy, sx, sy, _ = f["box"]
            dr.rectangle([*P(cx - sx / 2, cy + sy / 2), *P(cx + sx / 2, cy - sy / 2)], fill=(200, 190, 170), outline="gray")
    ny = len(G1_COVER_YAWS_DEG)
    worst = mx.reshape(-1, ny).min(axis=1)
    for (x, y, z, s), v in zip(spots, worst):
        px_, py_ = P(x, y)
        c = (220, 30, 30) if v < G1_COVER_MIN_PX else (0, int(min(255, 60 + v * 0.4)), 0)
        dr.ellipse([px_ - 5, py_ - 5, px_ + 5, py_ + 5], fill=c)
        if v < G1_COVER_MIN_PX:
            dr.line([px_ - 8, py_ - 8, px_ + 8, py_ + 8], fill="black", width=2)
            dr.line([px_ - 8, py_ + 8, px_ + 8, py_ - 8], fill="black", width=2)
        elif z > 0.1:
            dr.rectangle([px_ - 6, py_ - 6, px_ + 6, py_ + 6], outline="blue")
    for q in (pt, *extra):
        qx, qy = P(*q)
        dr.polygon([(qx, qy - 9), (qx + 9, qy), (qx, qy + 9), (qx - 9, qy)], fill="blue")
    dr.text((14, 12), f"{room}: min-over-yaw max-px; red X = <{G1_COVER_MIN_PX}; blue diamond = room point", fill="black")
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rooms", nargs="+", default=list(ROOMS))
    ap.add_argument("--look-only", action="store_true", help="skip the retract run and the hall report")
    ap.add_argument("--extra-point", action="append", default=[], help="ROOM:X:Y second room point (proposal)")
    ap.add_argument("--room-pt-min", type=float, default=None,
                    help="override the spawn keep-out around room points (m), e.g. footprint-only 0.44 (check, 002)")
    a = ap.parse_args()
    if a.room_pt_min is not None:
        import sim.world.generate as gen
        gen.ROOM_PT_MIN = a.room_pt_min
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layout, places, c = load_layout(), load_places(), Cover()
    extra = {}
    for e in a.extra_point:
        r, x, y = e.split(":")
        extra.setdefault(r, []).append((float(x), float(y)))
    res: dict = {"look": {}, "retract": {}, "extra_points": {r: v for r, v in extra.items()}, "hall": {}}
    best = None
    for room in a.rooms:
        spots, cases = room_cases(layout, room)
        pts = [*room_points(room, places), *extra.get(room, [])]
        for key, look in (("look", True),) if a.look_only else (("look", True), ("retract", False)):
            # per case: best over room points of the per-heading px
            per = [c.px(p, cases, look) for p in pts]
            px = np.maximum.reduce(per)
            r = summarize(layout, room, px, spots)
            for f in r["failing"]:
                f["dist_room_pt_m"] = round(min(math.hypot(f["spot"][0] - p[0], f["spot"][1] - p[1]) for p in pts), 3)
            r["room_pt_min_m"] = a.room_pt_min
            res[key][room] = r
            print(key, room, {k: v for k, v in r.items() if k != "failing"}, "fails", len(r["failing"]), flush=True)
            if look:
                plot(layout, room, spots, px.max(axis=1), pts[0], a.out / f"cover_{room}.png", pts[1:])
                ok = px.max(axis=1)
                ok_i = [i for i in range(len(ok)) if ok[i] >= G1_COVER_MIN_PX]
                if ok_i:
                    i = min(ok_i, key=lambda i: ok[i])
                    if best is None or ok[i] < best[0]:
                        best = (int(ok[i]), room, pts[0], cases[i], px[i].tolist())
    if best:
        v, room, pt, case, pxs = best
        ph = c.photos(pt, case)
        for h, im in zip(HEADINGS_DEG, ph):
            Image.fromarray(im).save(a.out / f"hardest_{room}_{int(h):03d}.png")
        res["hardest_passing"] = {"room": room, "max_px": v, "case": list(case), "px": pxs}
    # hall visibility: spawn point, look pose, spots of every room, yaw 0 and 90
    hp = room_point("hall", places)
    for room in () if a.look_only else ROOMS:
        spots = spawn_spots(layout, room, G1_COVER_GRID)
        cases = [(x, y, z, math.radians(yd)) for (x, y, z, _) in spots for yd in (0.0, 90.0)]
        mx = c.px(hp, cases, True).max(axis=1)
        vis = mx >= G1_COVER_MIN_PX
        res["hall"][room] = {"n_cases": len(cases), "n_visible": int(vis.sum()),
                             "visible_cases": [[*cases[i][:3], round(math.degrees(cases[i][3])), int(mx[i])]
                                               for i in np.flatnonzero(vis)]}
        print("hall", room, len(cases), int(vis.sum()), flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    (a.out / "cover.json").write_text(json.dumps(res, indent=1) + "\n")
    print("runtime", res["runtime_s"])


if __name__ == "__main__":
    main()
