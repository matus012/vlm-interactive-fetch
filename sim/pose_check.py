"""G5-pose harness (task 006): perceived box poses vs sim truth on 50 mast views.

The view rule is the pre-registered rule (2026-11-13-006-g5pose-rule) (committed before this file). Offline: renders from a seeded
house state, never through the adapter. Truth (box body pose, segmentation) is used here for SELECTION and SCORING
only; perception (`sim.perception.locate_photos`) gets RGB, depth, K and the camera pose.

    MUJOCO_GL=egl python -m sim.pose_check --seeds 9800:10000 --out runs/5s/g5pose          # the scored run
    MUJOCO_GL=egl python -m sim.pose_check --seeds 9750:9800 --out runs/5s/dev --scale 4     # development only
    MUJOCO_GL=egl python -m sim.pose_check --seeds 9750:9800 --out runs/5s/calib --calibrate # HSV stats (dev only)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim.body import run
from sim.constants import BOX_HALF, ROBOT_RADIUS
from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, set_arm
from sim.perception import IN_VIEW_PX, colour_mask, locate_photos, mast_photos, project
from sim.robot import CAM_H, CAM_W, GRIPPER_ACT
from sim.world.generate import BOX_REACH, generate, load_places
from sim.world.house import GRIP_OPEN, load_house, set_base
from sim.world.occupancy import known_map

POS_TOL = 0.02  # m, 3D (brief Phase 5)
YAW_TOL_DEG = 10.0  # mod 180 deg (brief Phase 5)
BAR = 45  # of 50
SETTLE_S = 0.5  # = sim.adapter.START_SETTLE_S
N_RANDOM = 6  # random reachable positions per box room
DIST_MIN, DIST_MAX = 0.8, 4.0
BANDS = (("near", 0.8, 1.6), ("mid", 1.6, 2.4), ("far", 2.4, 4.0))
STRATA = (("bedroom", "nightstand", 9), ("living_room", "coffee table", 9), ("bedroom", "floor", 8),
          ("storage", "floor", 8), ("living_room", "floor", 8), ("kitchen", "floor", 8))
BORROW = ("mid", "near", "far")
DRAW_SEED = 20261113


def band_quota(n: int) -> dict[str, int]:
    return {"near": (n + 2) // 3, "mid": (n + 1) // 3, "far": n // 3}  # 9 -> 3/3/3, 8 -> 3/3/2


def seed_state(seed: int) -> tuple[Any, mujoco.MjModel, mujoco.MjData, dict[str, dict[str, Any]]]:
    """World, model, data after the SimRobot start settle (look arm, gripper open, 0.5 s), truth per box colour."""
    w = generate(seed=seed, blocker="none")
    m, d = load_house(w)
    set_arm(m, d, LOOK_ARM_QPOS)
    d.ctrl[m.actuator(GRIPPER_ACT).id] = GRIP_OPEN
    run(m, d, SETTLE_S)
    truth = {}
    for b in w.boxes:
        bd = d.body(b["name"])
        xm = bd.xmat.reshape(3, 3).copy()
        truth[b["colour"]] = {"pos": bd.xpos.copy(), "xmat": xm, "yaw": math.atan2(xm[1, 0], xm[0, 0]),
                              "room": b["room"], "support": b["surface"], "geom": m.geom(b["name"]).id}
    return w, m, d, truth


def positions(w: Any, seed: int, truth: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule: every room point of each box room, then N_RANDOM random reachable positions per box room."""
    rooms = [r for r in w.layout["rooms"] if r in w.layout["spawn_regions"]]
    out = [{"room": p["room"], "kind": p["name"], "x": float(p["pose"][0]), "y": float(p["pose"][1])}
           for p in load_places() if p["kind"] == "room" and p["room"] in rooms]
    free = known_map(w.layout).inflate(ROBOT_RADIUS + 0.02)
    rng = np.random.default_rng([seed, 5])
    clear = ROBOT_RADIUS + BOX_REACH + 0.05
    for room in rooms:
        x0, y0, x1, y1 = w.layout["rooms"][room]["rect"]
        for k in range(N_RANDOM):
            for _ in range(1000):
                x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
                if free.free(x, y) and all(math.hypot(x - t["pos"][0], y - t["pos"][1]) >= clear
                                           for t in truth.values()):
                    out.append({"room": room, "kind": f"random{k}", "x": x, "y": y})
                    break
    for i, p in enumerate(out):
        p["index"] = i
    return out


def put_robot(m: mujoco.MjModel, d: mujoco.MjData, p: dict[str, Any]) -> None:
    set_base(m, d, (p["x"], p["y"], 0.0))
    set_arm(m, d, LOOK_ARM_QPOS)


def band_of(dist: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo <= dist < hi or (name == "far" and dist == hi):
            return name
    return None


def candidates(seed: int) -> list[dict[str, Any]]:
    w, m, d, truth = seed_state(seed)
    out = []
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    try:
        phs = []
        for p in positions(w, seed, truth):
            put_robot(m, d, p)
            phs.append((p, mast_photos(m, d, r=r)))
    finally:
        r.close()
    for p, ph in phs:
        for colour, t in truth.items():
            if t["room"] != p["room"]:
                continue
            px = [int((q["seg"] == t["geom"]).sum()) for q in ph]
            h = int(np.argmax(px))
            if px[h] < IN_VIEW_PX:
                continue
            dist = float(np.hypot(*(ph[h]["cam_pos"][:2] - t["pos"][:2])))
            if not DIST_MIN <= dist <= DIST_MAX:
                continue
            out.append({"seed": seed, "position": p, "colour": colour, "room": t["room"], "support": t["support"],
                        "distance": dist, "band": band_of(dist), "seg_px": px, "seg_heading": HEADINGS_DEG[h]})
    return out


def draw(cands: list[dict[str, Any]], scale: int = 1) -> tuple[list[dict[str, Any]], list[str]]:
    """The rule's deterministic stratified draw. Returns (views, shortfall notes)."""
    rng = np.random.default_rng(DRAW_SEED)
    taken: set[tuple[int, str]] = set()
    views, notes = [], []
    for room, support, n in STRATA:
        pool = {b: sorted((c for c in cands if c["room"] == room and c["support"] == support and c["band"] == b),
                          key=lambda c: (c["seed"], c["position"]["index"], c["colour"])) for b, _, _ in BANDS}
        perm = {b: [pool[b][i] for i in rng.permutation(len(pool[b]))] for b, _, _ in BANDS}
        quota = {b: q * scale for b, q in band_quota(n).items()}
        got: dict[str, list[dict[str, Any]]] = {b: [] for b in quota}

        def take(b: str, k: int) -> None:
            for c in perm[b]:
                if len(got[b]) >= k:
                    return
                if (c["seed"], c["colour"]) not in taken and c not in got[b]:
                    got[b].append(c)
                    taken.add((c["seed"], c["colour"]))

        for b, _, _ in BANDS:
            take(b, quota[b])
        short = sum(quota.values()) - sum(len(v) for v in got.values())
        if short:
            notes.append(f"{support} / {room}: bands {({b: len(v) for b, v in got.items()})} vs quota {quota}")
            for b in BORROW:
                take(b, len(got[b]) + short)
                short = sum(quota.values()) - sum(len(v) for v in got.values())
                if not short:
                    break
            if short:
                notes.append(f"{support} / {room}: {short} views missing after borrowing")
        for b, _, _ in BANDS:
            views += got[b]
    return views, notes


def yaw_err_deg(est: float, true: float) -> float:
    e = (est - true) % math.pi
    return math.degrees(min(e, math.pi - e))


def box_corners(c: np.ndarray, R: np.ndarray) -> np.ndarray:
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.array(BOX_HALF)
    return c + s @ R.T


EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def score(views: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Per view result rows, in-view disagreement rows, and overlay records (photo + fits) for the sheet."""
    rows, disagree, overlay = [], [], []
    by_seed: dict[int, list[int]] = {}
    for i, v in enumerate(views):
        by_seed.setdefault(v["seed"], []).append(i)
    for seed, idx in sorted(by_seed.items()):
        _, m, d, truth = seed_state(seed)
        for i in idx:
            v = views[i]
            put_robot(m, d, v["position"])
            ph = mast_photos(m, d)
            t = truth[v["colour"]]
            res = locate_photos(ph, v["colour"])
            row = {"view": i, "seed": seed, "colour": v["colour"], "room": v["room"], "support": v["support"],
                   "band": v["band"], "distance": round(v["distance"], 3), "position": v["position"],
                   "seg_heading": v["seg_heading"], "seg_px": v["seg_px"], "photo_heading": HEADINGS_DEG[res["photo"]],
                   "mask_px": res["mask_px"], "n_inliers": res["n_inliers"], "in_view": res["in_view"],
                   "true_pos": [round(float(x), 5) for x in t["pos"]], "true_yaw_deg": round(math.degrees(t["yaw"]) % 180, 2)}
            if res["pos"] is not None:
                err = res["pos"] - t["pos"]
                row.update(pos=[round(float(x), 5) for x in res["pos"]], size=[round(float(x), 4) for x in res["size"]],
                           yaw_deg=round(math.degrees(res["yaw"]), 2), pos_err=float(np.linalg.norm(err)),
                           xy_err=float(np.hypot(err[0], err[1])), z_err=float(err[2]),
                           yaw_err_deg=yaw_err_deg(res["yaw"], t["yaw"]))
                row["pass"] = bool(row["pos_err"] <= POS_TOL and row["yaw_err_deg"] <= YAW_TOL_DEG)
            else:
                row.update(pos=None, pos_err=None, yaw_err_deg=None, **{"pass": False})
            rows.append(row)
            for k, q in enumerate(ph):
                for colour, tt in truth.items():
                    seg = int((q["seg"] == tt["geom"]).sum())
                    mk = int(colour_mask(q["rgb"], colour).sum())
                    if (seg >= IN_VIEW_PX) != (mk >= IN_VIEW_PX):
                        disagree.append({"view": i, "seed": seed, "heading": HEADINGS_DEG[k], "colour": colour,
                                         "seg_px": seg, "mask_px": mk,
                                         "kind": "mask_only" if mk >= IN_VIEW_PX else "seg_only"})
            q = ph[res["photo"]]
            overlay.append({"view": i, "rgb": q["rgb"], "mask": colour_mask(q["rgb"], v["colour"]), "K": q["K"],
                            "cam_pos": q["cam_pos"], "cam_mat": q["cam_mat"], "truth": (t["pos"], t["xmat"]),
                            "fit": res})
    return rows, disagree, overlay


def stats(x: list[float]) -> dict[str, float | int]:
    if not x:
        return {"n": 0}
    a = np.asarray(x)
    return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "worst": float(a.max())}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(key: str) -> dict[str, Any]:
        out = {}
        for g in sorted({r[key] for r in rows}):
            rs = [r for r in rows if r[key] == g]
            ok = [r for r in rs if r["pos_err"] is not None]
            out[g] = {"views": len(rs), "pass": sum(r["pass"] for r in rs), "not_located": len(rs) - len(ok),
                      "pos_err_m": stats([r["pos_err"] for r in ok]), "yaw_err_deg": stats([r["yaw_err_deg"] for r in ok])}
        return out

    ok = [r for r in rows if r["pos_err"] is not None]
    mask_bins = ((100, 300), (300, 1000), (1000, 3000), (3000, 10 ** 9))
    by_mask = {f"{lo}-{hi if hi < 10 ** 9 else 'inf'}": {
        "views": len(rs), "pass": sum(r["pass"] for r in rs), "pos_err_m": stats([r["pos_err"] for r in rs]),
        "yaw_err_deg": stats([r["yaw_err_deg"] for r in rs])}
        for lo, hi in mask_bins for rs in [[r for r in ok if lo <= r["mask_px"] < hi]]}
    return {"views": len(rows), "pass": sum(r["pass"] for r in rows), "pos_err_m": stats([r["pos_err"] for r in ok]),
            "xy_err_m": stats([r["xy_err"] for r in ok]), "yaw_err_deg": stats([r["yaw_err_deg"] for r in ok]),
            "by_band": group("band"), "by_support": group("support"), "by_mask_px": by_mask,
            "mask_px_vs_pos_err_spearman": _spearman([r["mask_px"] for r in ok], [r["pos_err"] for r in ok]),
            "mask_px_vs_yaw_err_spearman": _spearman([r["mask_px"] for r in ok], [r["yaw_err_deg"] for r in ok])}


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _wire(dr: ImageDraw.ImageDraw, P: np.ndarray, o: dict[str, Any], colour: tuple[int, int, int], zoom: float,
          off: tuple[float, float]) -> None:
    uv = project(P, o["K"], o["cam_pos"], o["cam_mat"])
    for i, j in EDGES:
        a, b = (uv[i] - off) * zoom, (uv[j] - off) * zoom
        dr.line([tuple(a), tuple(b)], fill=colour, width=1)


def contact_sheet(overlay: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path, n: int = 8) -> list[int]:
    """Tiles: full photo (half size) + 4x zoom around the true box; mask tinted magenta, fit = green, truth = cyan.
    Views: the 4 worst by position error (not located first), then evenly spread others."""
    order = sorted(range(len(rows)), key=lambda i: -(rows[i]["pos_err"] if rows[i]["pos_err"] is not None else 9.0))
    pick = order[:4]
    rest = [i for i in range(len(rows)) if i not in pick]
    pick += [rest[k] for k in np.linspace(0, len(rest) - 1, n - 4).astype(int)]
    W, H, Z, C = 320, 240, 4.0, 60  # half-size full view; zoom window 2C px of the photo at Z
    tile_w, tile_h = W + int(2 * C * Z) // 2, max(H, int(2 * C * Z) // 2) + 28
    sheet = Image.new("RGB", (2 * tile_w, ((len(pick) + 1) // 2) * tile_h), (255, 255, 255))
    for k, i in enumerate(pick):
        o, r = overlay[i], rows[i]
        img = o["rgb"].astype(np.float32)
        img[o["mask"]] = 0.5 * img[o["mask"]] + 0.5 * np.array([255, 0, 255])
        full = Image.fromarray(img.astype(np.uint8))
        u0, v0 = project(o["truth"][0][None], o["K"], o["cam_pos"], o["cam_mat"])[0]
        off = (u0 - C, v0 - C)
        zoom = full.crop((int(off[0]), int(off[1]), int(off[0]) + 2 * C, int(off[1]) + 2 * C)).resize(
            (int(2 * C * Z) // 2, int(2 * C * Z) // 2), Image.NEAREST)
        off = (int(off[0]), int(off[1]))
        zd = ImageDraw.Draw(zoom)
        _wire(zd, box_corners(*o["truth"]), o, (0, 255, 255), Z / 2, off)
        if o["fit"]["pos"] is not None:
            y = o["fit"]["yaw"]
            R = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
            _wire(zd, box_corners(o["fit"]["pos"], R), o, (0, 255, 0), Z / 2, off)
        small = full.resize((W, H))
        sd = ImageDraw.Draw(small)
        sd.rectangle([off[0] / 2, off[1] / 2, off[0] / 2 + C, off[1] / 2 + C], outline=(255, 255, 0))
        x0, y0 = (k % 2) * tile_w, (k // 2) * tile_h
        sheet.paste(small, (x0, y0 + 28))
        sheet.paste(zoom, (x0 + W, y0 + 28))
        pe = f"{r['pos_err'] * 100:.1f} cm, yaw {r['yaw_err_deg']:.1f} deg" if r["pos_err"] is not None else "not located"
        ImageDraw.Draw(sheet).text((x0 + 4, y0 + 2), f"view {i} seed {r['seed']} {r['colour']} {r['support']} "
                                   f"{r['distance']:.2f} m mask {r['mask_px']} px | {pe} | {'PASS' if r['pass'] else 'FAIL'}",
                                   fill=(0, 0, 0))
        ImageDraw.Draw(sheet).text((x0 + 4, y0 + 14), "magenta = mask, green = fit, cyan = truth", fill=(80, 80, 80))
    sheet.save(path)
    return pick


def calibrate(seeds: range, out: Path) -> dict[str, Any]:
    """HSV statistics behind perception.HSV_RANGES (development seeds only): every other seed, every 3rd candidate
    position, 4 headings; box px = segmentation px of each box geom, background = every 7th other px."""
    from sim.perception import rgb_to_hsv

    box: dict[str, list[np.ndarray]] = {}
    bg = []
    for seed in seeds[::2]:
        w, m, d, truth = seed_state(seed)
        r = mujoco.Renderer(m, CAM_H, CAM_W)
        try:
            for p in positions(w, seed, truth)[::3]:
                put_robot(m, d, p)
                for q in mast_photos(m, d, r=r):
                    hsv = np.stack(rgb_to_hsv(q["rgb"]), axis=-1)
                    anyb = np.zeros(hsv.shape[:2], bool)
                    for c, t in truth.items():
                        mk = q["seg"] == t["geom"]
                        anyb |= mk
                        box.setdefault(c, []).append(hsv[mk])
                    bg.append(hsv[~anyb][::7])
        finally:
            r.close()
    qs = [0.1, 1, 50, 99, 99.9]
    res: dict[str, Any] = {"seeds": [seeds[0], seeds[-1]], "box": {}, "background": {}}
    for c, v in box.items():
        a = np.concatenate(v)
        res["box"][c] = {"px": len(a), "hue_q": dict(zip(map(str, qs), np.percentile(a[:, 0], qs).round(1).tolist())),
                         "s_q": dict(zip(map(str, qs), np.percentile(a[:, 1], qs).round(3).tolist())),
                         "v_q": dict(zip(map(str, qs), np.percentile(a[:, 2], qs).round(3).tolist())),
                         "frac_s_ge_0.70": round(float((a[:, 1] >= 0.70).mean()), 4)}
    b = np.concatenate(bg)
    hi = b[b[:, 1] >= 0.70]
    res["background"] = {"px_sampled": len(b), "s_q": dict(zip(["50", "99", "99.9", "99.99"],
                                                              np.percentile(b[:, 1], [50, 99, 99.9, 99.99]).round(3).tolist())),
                         "n_s_ge_0.70": len(hi), "hue_hist_s_ge_0.70_10deg": np.histogram(hi[:, 0], 36, (0, 360))[0].tolist()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "hsv_stats.json").write_text(json.dumps(res, indent=1))
    return res


def single_photo(result: Path, out: Path) -> dict[str, Any]:
    """Info only (follow-up to G5-pose round 1, not gated): the same views as a scored result.json, located from
    ONE photo each, the heading with the most mask px (`locate_from_rgbd`), i.e. without the 4-photo fusion."""
    from sim.perception import locate_from_rgbd

    rows = json.loads(result.read_text())["views_rows"]
    out_rows = []
    for seed in sorted({r["seed"] for r in rows}):
        _, m, d, truth = seed_state(seed)
        for r in (r for r in rows if r["seed"] == seed):
            put_robot(m, d, r["position"])
            ph = mast_photos(m, d)
            res = [locate_from_rgbd(q["rgb"], q["depth"], q["K"], (q["cam_pos"], q["cam_mat"]), r["colour"]) for q in ph]
            k = max(range(len(res)), key=lambda j: res[j]["mask_px"])
            t, f = truth[r["colour"]], res[k]
            row = {"view": r["view"], "seed": seed, "colour": r["colour"], "support": r["support"], "band": r["band"],
                   "distance": r["distance"], "heading": HEADINGS_DEG[k], "mask_px": f["mask_px"],
                   "photos_with_mask": sum(x["mask_px"] > 0 for x in res), "fused_pass": r["pass"]}
            if f["pos"] is None:
                row.update(pos_err=None, yaw_err_deg=None, **{"pass": False})
            else:
                row.update(pos_err=float(np.linalg.norm(f["pos"] - t["pos"])), yaw_err_deg=yaw_err_deg(f["yaw"], t["yaw"]))
                row["pass"] = bool(row["pos_err"] <= POS_TOL and row["yaw_err_deg"] <= YAW_TOL_DEG)
            out_rows.append(row)
    ok = [r for r in out_rows if r["pos_err"] is not None]
    res = {"source": str(result), "views": len(out_rows), "pass": sum(r["pass"] for r in out_rows),
           "pos_err_m": stats([r["pos_err"] for r in ok]), "yaw_err_deg": stats([r["yaw_err_deg"] for r in ok]),
           "rows": out_rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "single_photo.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9800:10000")
    ap.add_argument("--out", default="runs/5s/g5pose")
    ap.add_argument("--scale", type=int, default=1, help="quota multiplier (development range only)")
    ap.add_argument("--calibrate", action="store_true", help="HSV statistics only (development range only)")
    ap.add_argument("--single-photo", type=Path, help="info: re-locate a scored result.json's views from one photo each")
    a = ap.parse_args()
    if a.single_photo:
        res = single_photo(a.single_photo, Path(a.out))
        print(json.dumps({k: res[k] for k in ("views", "pass", "pos_err_m", "yaw_err_deg")}))
        for r in res["rows"]:
            if not r["pass"] or not r["fused_pass"]:
                print(r)
        return
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.calibrate:
        assert s1 <= 9800 or s0 >= 10000, "calibration never on the scored range 9800-9999"
        print(json.dumps(calibrate(range(s0, s1), out), indent=1))
        return
    t0 = time.time()
    cands = []
    for s in range(s0, s1):
        cands += candidates(s)
    t1 = time.time()
    print(f"{len(cands)} candidates from seeds {s0}-{s1 - 1} in {t1 - t0:.0f} s", flush=True)
    views, notes = draw(cands, a.scale)
    rows, disagree, overlay = score(views)
    t2 = time.time()
    summ = summary(rows)
    pick = contact_sheet(overlay, rows, out / "sheet.png")
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "sim", "tests"], capture_output=True,
                                text=True).stdout.strip())
    res = {"gate": "G5-pose", "rule": "the pre-registered rule (2026-11-13-006-g5pose-rule)", "commit": commit, "dirty": dirty,
           "seeds": [s0, s1 - 1], "scale": a.scale, "bar": f">= {BAR * a.scale} / {50 * a.scale}",
           "pos_tol_m": POS_TOL, "yaw_tol_deg": YAW_TOL_DEG, "passes": summ["pass"], "views": len(rows),
           "verdict": "PASS" if summ["pass"] >= BAR * a.scale and len(rows) == 50 * a.scale else "FAIL",
           "n_candidates": len(cands), "draw_notes": notes, "summary": summ,
           "in_view_disagreements": {"photos_x_colours": 16 * len(rows),
                                     "mask_only": sum(x["kind"] == "mask_only" for x in disagree),
                                     "seg_only": sum(x["kind"] == "seg_only" for x in disagree), "rows": disagree},
           "sheet_views": pick, "seconds": {"candidates": round(t1 - t0), "score": round(t2 - t1)}, "views_rows": rows}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("verdict", "passes", "views", "draw_notes")}), flush=True)
    for k in ("pos_err_m", "yaw_err_deg"):
        print(k, summ[k])
    for g in ("by_band", "by_support", "by_mask_px"):
        for name, v in summ[g].items():
            print(g, name, v["views"], v["pass"], v["pos_err_m"], v["yaw_err_deg"])
    print("in-view disagreements", res["in_view_disagreements"]["mask_only"], res["in_view_disagreements"]["seg_only"])


if __name__ == "__main__":
    main()
