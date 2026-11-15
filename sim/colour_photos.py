"""G1-colour photo set, truth, selection and scoring (task 002). RULE FIXED BEFORE ANY VLM CALL:

Photos: house generate(seed, "none") for seeds 0, 1, 2, ...; robot at each room point of places.yaml (kind room, all 5
rooms incl. hall) in LOOK_ARM_QPOS, 4 look photos (headings 0/90/180/270, set_yaw_direct), 640x480 RGB + segmentation.
Truth per photo: a colour is PRESENT if its box has >= 100 seg px (G1_COVER_MIN_PX), ABSENT if 0 px; 1-99 px =
ambiguous (neither required nor forbidden).
Selection: iterate photos in order (seed, room in places.yaml order, heading). For each colour take the first photos in
which it is PRESENT, skipping a photo if that colour already has a selected photo from the same (seed, room), until
each colour has 3. The union of the selected photos is the set sent to the VLM (<= 12 photos).
Scoring: one VLM call per selected photo (one image), thinking on. system = SYSTEM, user = photo + QUESTION; parse the
first [...] of the final content. Per (colour, photo of its selection): pass if the true colour is named AND no ABSENT
colour is named in that answer. Gate: 3/3 per colour (red, blue, green, yellow).

    python -m sim.colour_photos --out runs/1v/photos      # renders (locally, MUJOCO_GL=egl), writes PNGs, truth.json, sheet.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from sim.constants import BOX_COLOURS, G1_COVER_MIN_PX

COLOURS = tuple(BOX_COLOURS)  # red, blue, green, yellow
PER_COLOUR = 3
MAX_SEEDS = 30
SYSTEM = "You are looking at one photo from a home robot's camera."
QUESTION = ("Which coloured boxes can you see in this photo? The possible colours are red, blue, green and yellow. "
            "Answer with only a JSON list of the colours you see, e.g. [\"red\"], or [] if none.")


def status(px: int) -> str:
    return "present" if px >= G1_COVER_MIN_PX else "absent" if px == 0 else "ambiguous"


def select(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """colour -> selected photo ids (records in photo order; each has id, seed, room, status{colour})."""
    sel: dict[str, list[str]] = {c: [] for c in COLOURS}
    used: dict[str, set[tuple[int, str]]] = {c: set() for c in COLOURS}
    for r in records:
        for c in COLOURS:
            key = (r["seed"], r["room"])
            if len(sel[c]) < PER_COLOUR and r["status"][c] == "present" and key not in used[c]:
                sel[c].append(r["id"])
                used[c].add(key)
    return sel


def parse_answer(content: str | None) -> list[str] | None:
    """Colours in the first [...] of the content (lowercased); None if there is none or it is not a JSON list."""
    m = re.search(r"\[.*?\]", content or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    return [str(x).strip().lower() for x in v] if isinstance(v, list) else None


def score(truth: dict[str, dict[str, str]], sel: dict[str, list[str]], answers: dict[str, list[str] | None]) -> dict[str, Any]:
    """Per colour: list of (photo id, pass) and its pass count; verdict = 3/3 for every colour."""
    out: dict[str, Any] = {}
    for c in COLOURS:
        rows = []
        for pid in sel[c]:
            a = answers.get(pid)
            ok = a is not None and c in a and not any(o in a for o in COLOURS if truth[pid][o] == "absent")
            rows.append({"photo": pid, "pass": ok})
        out[c] = {"photos": rows, "n_pass": sum(r["pass"] for r in rows), "gate": len(rows) == PER_COLOUR
                  and all(r["pass"] for r in rows)}
    out["all_pass"] = all(out[c]["gate"] for c in COLOURS)
    return out


def render_seed(seed: int) -> list[dict[str, Any]]:
    """Photo records for one seed (no blocker), in places.yaml room order x headings; each with `rgb`."""
    import mujoco

    from sim.look import look_photos, set_arm
    from sim.world.generate import LAYOUT, generate, load_places
    from sim.world.house import load_house, set_base

    w = generate(LAYOUT, seed, "none")
    m, d = load_house(w)
    ids = {b["colour"]: m.geom(b["name"]).id for b in w.boxes}
    recs = []
    for p in load_places():
        if p["kind"] != "room":
            continue
        set_base(m, d, (p["pose"][0], p["pose"][1], 0.0))
        set_arm(m, d)
        mujoco.mj_forward(m, d)
        for h, rgb, _, seg in look_photos(m, d):
            px = {c: int((seg == ids[c]).sum()) for c in COLOURS}
            recs.append({"id": f"s{seed}_{p['room']}_{int(h):03d}", "seed": seed, "room": p["room"], "heading": int(h),
                         "px": px, "status": {c: status(n) for c, n in px.items()}, "rgb": rgb,
                         "boxes": {b["colour"]: [b["room"], b["surface"]] for b in w.boxes}})
    return recs


def build(out: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    recs: list[dict[str, Any]] = []
    sel: dict[str, list[str]] = {}
    for seed in range(MAX_SEEDS):
        recs += render_seed(seed)
        sel = select(recs)
        if all(len(v) == PER_COLOUR for v in sel.values()):
            break
    assert all(len(v) == PER_COLOUR for v in sel.values()), f"selection incomplete after {MAX_SEEDS} seeds: {sel}"
    ids = sorted({i for v in sel.values() for i in v}, key=[r["id"] for r in recs].index)
    out.mkdir(parents=True, exist_ok=True)
    by = {r["id"]: r for r in recs}
    truth = {i: {**by[i]["status"], "_px": by[i]["px"], "_boxes": by[i]["boxes"]} for i in ids}
    cols = 4
    sheet = Image.new("RGB", (cols * 320, math.ceil(len(ids) / cols) * 260), "white")
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(ids):
        im = Image.fromarray(by[i]["rgb"])
        im.save(out / f"{i}.png")
        x, y = (k % cols) * 320, (k // cols) * 260
        sheet.paste(im.resize((320, 240)), (x, y))
        dr.text((x + 3, y + 242), i + " " + ",".join(c[0].upper() + {"present": "+", "absent": "-", "ambiguous": "?"}[by[i]["status"][c]]
                                                   for c in COLOURS), fill="black")
    sheet.save(out / "sheet.png")
    doc = {"selection": sel, "photos": ids, "truth": {i: {c: truth[i][c] for c in COLOURS} for i in ids},
           "px": {i: truth[i]["_px"] for i in ids}, "boxes": {i: truth[i]["_boxes"] for i in ids}, "seeds_used": seed + 1}
    (out / "truth.json").write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    doc = build(ap.parse_args().out)
    print(json.dumps(doc["selection"]), len(doc["photos"]), "photos, seeds used", doc["seeds_used"])
