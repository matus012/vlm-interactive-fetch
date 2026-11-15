"""Task 009 report numbers (after the E1 counted pass and the oracle split are written; reads task contents).

    python -m sim.e1_analyse --e1 <runs/009/pass1> --verified <verified.jsonl> --oracle <runs/009/oracle>
        --base <runs/007/pass1> --out <json>

Per type: E1 pass / 25 with Wilson 95 % next to the sealed pass 1 (seal-005), and E1 on the oracle-solvable subset.
Per eased case (sim.e1.box_flags on the task's easings-off world; target = the task's target boxes, any = every box):
count and pass rate. Row 15 lifted: targets with no pre-grasp pose (sim.tasks.pregrasp_ok, planning only). Cabinet
tasks: push / detour choices (a `push(cabinet)` call; a navigate / pick / place with a via) in E1 and in the sealed
run. Failure labels, steps, VLM latency, wall. Worlds: easings off, test seeds opened here by adding them to
sim.tasks._SEALED_OK (analysis after the pass, not a draw).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from sim.world.generate import set_easings

set_easings(False)

import sim.tasks as T  # noqa: E402 - after set_easings
from sim.e1 import box_flags  # noqa: E402
from sim.tasks import TYPES, box_name  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (c - h)), round(100 * (c + h)))


def results(d: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in d.glob("*/result.json")}


def rate(rows: list[dict]) -> dict:
    k = sum(r["pass"] for r in rows)
    return {"pass": k, "n": len(rows), "wilson95": wilson(k, len(rows))}


def choices(r: dict) -> dict[str, bool]:
    c = r["calls"]
    return {"blocked_seen": any("failed: blocked" in x for x in c),
            "push_cabinet": any(x.startswith("push(cabinet)") for x in c),
            "via": any(x.split("(")[0] in ("navigate", "pick", "place") and "," in x.split(") ->")[0] for x in c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("e1", "verified", "oracle", "base", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in map(json.loads, Path(a.verified).read_text().splitlines())}
    e1, base = results(Path(a.e1)), results(Path(a.base))
    assert set(e1) == set(tasks) and len(e1) == 100, (len(e1), len(tasks))
    orc = {p.parent.name: json.loads(p.read_text()) for p in Path(a.oracle).glob("*/oracle.json")}
    out: dict = {"per_type": {}, "cases": {}, "cabinet": {}, "oracle_n": len(orc)}
    case_rows: dict[str, list[dict]] = {}
    for tid, t in tasks.items():
        T._SEALED_OK.add(t["seed"])
        w = T.world(t["seed"], t["blocker"])
        fl = box_flags(w)
        surf = {b["name"]: b["surface"] for b in w.boxes}
        targets = {f"box_{x['object'].split()[0]}" for x in t["targets"]}
        ok = T.pregrasp_ok(t["seed"], t["blocker"])
        r = e1[tid]
        r["_oracle"] = orc.get(tid, {}).get("solvable")
        cases = {
            "target_on_route": any(fl[b]["on_route"] for b in targets),
            "any_box_on_route": any(f["on_route"] for f in fl.values()),
            "target_back_nightstand": any(fl[b]["back_of_surface"] and surf[b] == "nightstand" for b in targets),
            "any_back_nightstand": any(f["back_of_surface"] and surf[b] == "nightstand" for b, f in fl.items()),
            "any_back_coffee_table": any(f["back_of_surface"] and surf[b] == "coffee table" for b, f in fl.items()),
            "target_near_surface_floor": any(fl[b]["near_surface_floor"] for b in targets),
            "any_near_surface_floor": any(f["near_surface_floor"] for f in fl.values()),
            "target_no_pregrasp": any(box_name(b.split("_", 1)[1]) not in ok for b in targets),
        }
        cases["any_eased_case"] = any(cases[k] for k in ("any_box_on_route", "any_back_nightstand",
                                                         "any_near_surface_floor", "target_no_pregrasp"))
        r["_cases"] = cases
        for k, v in cases.items():
            if v:
                case_rows.setdefault(k, []).append(r)
    for ty in TYPES:
        rows = [e1[i] for i in tasks if tasks[i]["type"] == ty]
        brow = [r for i, r in base.items() if i.startswith(ty + "-")]
        sol = [r for r in rows if r["_oracle"]]
        out["per_type"][ty] = {
            "e1": rate(rows), "sealed_005": rate(brow), "e1_oracle_solvable": rate(sol),
            "oracle_unsolvable": sum(r["_oracle"] is False for r in rows),
            "labels": dict(Counter(r["label"] for r in rows if not r["pass"])),
            "steps_median": statistics.median(r["steps_used"] for r in rows),
            "vlm_median_s": round(statistics.median(r["timing"]["vlm_median_s"] for r in rows), 2),
            "run_wall_median_s": round(statistics.median(r["timing"]["run_wall_s"] for r in rows), 1),
        }
    allr = list(e1.values())
    out["total"] = {"e1": rate(allr), "sealed_005": rate(list(base.values())),
                    "e1_oracle_solvable": rate([r for r in allr if r["_oracle"]]),
                    "no_eased_case": rate([r for r in allr if not r["_cases"]["any_eased_case"]])}
    out["cases"] = {k: rate(v) for k, v in sorted(case_rows.items())}
    for name, res in (("e1", e1), ("sealed_005", base)):
        rows = [r for i, r in res.items() if i.startswith("push_or_detour-")]
        ch = [choices(r) | {"pass": r["pass"]} for r in rows]
        out["cabinet"][name] = {
            "n": len(ch), "blocked_seen": sum(c["blocked_seen"] for c in ch),
            "push_tried": sum(c["push_cabinet"] for c in ch), "via_used": sum(c["via"] for c in ch),
            "push_and_via": sum(c["push_cabinet"] and c["via"] for c in ch),
            "via_without_push": sum(c["via"] and not c["push_cabinet"] for c in ch),
            "neither_after_block": sum(c["blocked_seen"] and not c["via"] and not c["push_cabinet"] for c in ch),
            "pass_when_push_tried": rate([r for r, c in zip(rows, ch) if c["push_cabinet"]]),
        }
    out["failures"] = [{"task_id": r["task_id"], "label": r["label"], "failed": r["failed"], "steps": r["steps_used"],
                        "oracle_solvable": r["_oracle"], "cases": [k for k, v in r["_cases"].items() if v]}
                       for r in sorted(allr, key=lambda r: r["task_id"]) if not r["pass"]]
    out["leak_hits"] = sum(len(r["leak_hits"]) if isinstance(r["leak_hits"], list) else r["leak_hits"] for r in allr)
    out["rules"] = dict(Counter(r.get("rules") for r in allr))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"total": out["total"], "per_type": {k: v["e1"] for k, v in out["per_type"].items()}}))


if __name__ == "__main__":
    main()
