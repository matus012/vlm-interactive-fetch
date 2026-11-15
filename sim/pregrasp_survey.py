"""Pick pre-grasp feasibility survey (task 004, tuning fail fetch-4016): for every fetch box of every seed in the
given ranges, count boxes with no pre-grasp candidate (`grasp.plan_pregrasp`) when every OTHER box is seen (avoid
discs, as the adapter passes them to pick). Planning only, no physics: it says whether a pre-grasp pose exists, not
whether the pick then succeeds.

    MUJOCO_GL=egl python -m sim.pregrasp_survey --seeds 4000:4100 7000:7300 --out runs/004/probe/pregrasp_survey.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim import grasp
from sim.adapter import SimRobot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True, help="a:b ranges (dev seeds)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = [s for r in a.seeds for s in range(*map(int, r.split(":")))]
    n = z_none = z_all = 0
    hits = []
    for s in seeds:
        r = SimRobot(s, "none")
        boxes = [o for o, v in r.objects().items() if v["class"] == "box"]
        r.seen |= set(boxes)
        for b in boxes:
            n += 1
            c0 = len(grasp.plan_pregrasp(r.m, r.d, r.grid, r.body_of[b], []))
            ca = len(grasp.plan_pregrasp(r.m, r.d, r.grid, r.body_of[b], r._seen_discs({b})))
            z_none += c0 == 0
            z_all += ca == 0
            if ca == 0:
                hits.append({"seed": s, "box": b, "support": r.support_of(b), "cands_no_avoid": c0})
    res = {"seeds": a.seeds, "n_seeds": len(seeds), "n_boxes": n, "no_pregrasp_no_avoid": z_none,
           "no_pregrasp_others_avoided": z_all, "hits": hits}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "hits"}), hits)


if __name__ == "__main__":
    main()
