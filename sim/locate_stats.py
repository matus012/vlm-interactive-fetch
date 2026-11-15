"""Perceived-`locate` stats from run logs (task 006 "also report"): every locate step of a perceived-mode run logs its
mask px, the perceived pose and, under meta["eval"], the true pose, seg px and the errors (sim.adapter._locate_eval).

    python -m sim.locate_stats <runs dir> [...]   -> prints JSON; reads every log.jsonl below the dirs

- error (3D position, xy, yaw mod 180 deg): median / p90 / worst, by distance band (horizontal mast camera -> true box
  centre, the G5-pose rule's bands near [0.8, 1.6), mid [1.6, 2.4), far [2.4, 4.0], plus < 0.8 and > 4.0) and by the
  box's true support; within the G5-pose tolerance (<= 2 cm and <= 10 deg) count;
- mask px vs error: the same stats by mask-px band;
- in-view disagreements: `failed: not in view` while seg px >= 100 (missed), and found while seg px < 100 (extra).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PX = 100
BANDS = (("<0.8", 0.0, 0.8), ("near", 0.8, 1.6), ("mid", 1.6, 2.4), ("far", 2.4, 4.0 + 1e-9), (">4.0", 4.0 + 1e-9, 1e9))
PX_BANDS = (("100-499", 100, 500), ("500-1999", 500, 2000), ("2000+", 2000, 10 ** 9))


def rows(dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for log in sorted(d.rglob("log.jsonl")):
            for line in log.read_text().splitlines():
                s = json.loads(line)
                ev = (s.get("meta") or {}).get("eval")
                if s.get("call") != "locate" or ev is None:
                    continue
                out.append({"log": str(log.relative_to(d)), "step": s.get("step"), "object": s["args"].get("object"),
                            "result": s["result"], "found": s["result"].startswith("found"),
                            "mask_px": s["meta"]["max_px"], "seg_px": ev["seg_px"], "support": ev["support"],
                            **{k: ev.get(k) for k in ("pos_err_m", "xy_err_m", "yaw_err_deg", "dist_m")}})
    return out


def _stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    rs = [r for r in rs if r["pos_err_m"] is not None]
    if not rs:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(rs), "within_2cm_10deg": sum(r["pos_err_m"] <= 0.02 and r["yaw_err_deg"] <= 10.0
                                                                   for r in rs)}
    for k, unit in (("pos_err_m", 100.0), ("xy_err_m", 100.0), ("yaw_err_deg", 1.0)):
        v = np.array([r[k] for r in rs]) * unit
        name = k.replace("_m", "_cm") if unit == 100.0 else k
        out[name] = {"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                     "worst": round(float(v.max()), 3)}
    return out


def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    found = [r for r in rs if r["found"]]
    worst = sorted((r for r in found if r["pos_err_m"] is not None), key=lambda r: -r["pos_err_m"])[:5]
    return {
        "locate_calls": len(rs), "found": len(found), "all": _stats(found),
        "by_distance": {b: _stats([r for r in found if r["dist_m"] is not None and lo <= r["dist_m"] < hi])
                        for b, lo, hi in BANDS},
        "by_support": {s: _stats([r for r in found if r["support"] == s]) for s in sorted({r["support"] for r in found})},
        "by_mask_px": {b: _stats([r for r in found if lo <= r["mask_px"] < hi]) for b, lo, hi in PX_BANDS},
        "not_in_view_but_seg_ge_100": [{k: r[k] for k in ("log", "step", "object", "mask_px", "seg_px")}
                                       for r in rs if not r["found"] and r["seg_px"] >= PX],
        "found_but_seg_lt_100": [{k: r[k] for k in ("log", "step", "object", "mask_px", "seg_px")}
                                 for r in found if r["seg_px"] < PX],
        "worst5": [{k: r[k] for k in ("log", "step", "object", "mask_px", "seg_px", "support", "pos_err_m",
                                      "yaw_err_deg", "dist_m")} for r in worst],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    print(json.dumps(summary(rows(ap.parse_args().dirs)), indent=1))


if __name__ == "__main__":
    main()
