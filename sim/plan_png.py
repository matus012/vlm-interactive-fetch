"""Top-view floor plan of the active house (NAV116_HOUSE) from its world files, for reports and clips (task 011).

    NAV116_HOUSE=B python -m sim.plan_png runs/011/house_b_plan.png

Rooms (floor colour), walls, doors (gap, crate / cabinet doors marked X / C), furniture with names, places (room look
spots blue, surfaces purple), the hall spawn (red ring). 80 px per metre, same scale for both houses.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from sim.clip_phase3 import font
from sim.world.generate import HOUSE, furniture_rect, load_layout, load_places, wall_rects

PX = 80
PAD = 40


def main() -> None:
    out = Path(sys.argv[1])
    lay, places = load_layout(), load_places()
    xs = [v for r in lay["rooms"].values() for v in (r["rect"][0], r["rect"][2])]
    ys = [v for r in lay["rooms"].values() for v in (r["rect"][1], r["rect"][3])]
    x0, y1 = min(xs), max(ys)
    w, h = int((max(xs) - x0) * PX) + 2 * PAD, int((y1 - min(ys)) * PX) + 2 * PAD + 30

    def P(x: float, y: float) -> tuple[int, int]:
        return int(PAD + (x - x0) * PX), int(PAD + 30 + (y1 - y) * PX)

    def R(r: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        (a, b), (c, d) = P(r[0], r[3]), P(r[2], r[1])
        return a, b, c, d

    img = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.text((PAD, 8), f"house {HOUSE}: {lay['house']} (1 grid = 1 m)", font=font(20, True), fill=(0, 0, 0))
    for name, r in lay["rooms"].items():
        dr.rectangle(R(tuple(r["rect"])), fill=tuple(int(255 * c) for c in r["floor"]))
        cx, cy = (r["rect"][0] + r["rect"][2]) / 2, r["rect"][3] - 0.35
        dr.text(P(cx - 0.6, cy), name.replace("_", " ").upper(), font=font(16, True), fill=(30, 30, 30))
    for wr in wall_rects(lay):
        dr.rectangle(R(wr), fill=(60, 60, 60))
    marks = {d: "X" for d in lay["blockers"]["crate"]["doors"]} | {d: "C" for d in lay["blockers"]["cabinet"]["doors"]}
    for k, d in lay["doors"].items():
        (axis, c), = d["wall"].items()
        x, y = (c, d["center"]) if axis == "x" else (d["center"], c)
        if k in marks:
            dr.text(P(x + 0.08, y + 0.15), marks[k], font=font(18, True), fill=(200, 60, 0))
    for f in lay["furniture"]:
        dr.rectangle(R(furniture_rect(f)), fill=tuple(int(255 * c) for c in f["rgb"]), outline=(0, 0, 0))
        fr = furniture_rect(f)
        dr.text(P(fr[0] + 0.05, (fr[1] + fr[3]) / 2 + 0.1), f["id"].replace("_", " "), font=font(12), fill=(255, 255, 255),
                stroke_width=2, stroke_fill=(0, 0, 0))
    for p in places:
        if p["kind"] == "door":
            continue
        x, y = p["pose"][:2]
        col = (30, 60, 220) if p["kind"] == "room" else (150, 0, 150)
        a, b = P(x, y)
        dr.ellipse((a - 5, b - 5, a + 5, b + 5), fill=col)
        dr.text((a + 7, b - 2), p["name"], font=font(13), fill=col)
    a, b = P(*lay["spawn"][:2])
    dr.ellipse((a - 10, b - 10, a + 10, b + 10), outline=(220, 0, 0), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out, img.size)


if __name__ == "__main__":
    main()
