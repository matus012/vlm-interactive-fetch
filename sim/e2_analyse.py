"""Task 011 report numbers: house B (seal-011) vs house A (seal-005), VLM and the E3 scripts.

    python -m sim.e2_analyse --b <runs/011/pass1> --a-vlm <runs/007/pass1> --a-scripts <runs/010/pass1> --out <json>

<b>/<task_id>/result.json = the frozen agent's counted pass in house B; <b>/e3a, <b>/e3b = agent/e3.py there.
Per type and overall: pass / 25 with Wilson 95 %, for VLM / E3a / E3b in house A and house B; VLM failure labels;
steps median; script outcomes (sim.e3_analyse.outcome) and branch counts; leak hits. Reads result.json only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from sim.e3_analyse import TYPES, branches, outcome, rate, results


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("b", "a-vlm", "a-scripts", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    runs = {("B", "vlm"): results(Path(a.b)), ("A", "vlm"): results(Path(a.a_vlm)),
            **{(h, s): results(Path(d) / s) for h, d in (("A", a.a_scripts), ("B", a.b)) for s in ("e3a", "e3b")}}
    assert all(len(v) == 100 for v in runs.values()), {k: len(v) for k, v in runs.items()}
    out: dict = {"per_type": {}}
    for ty in TYPES + ("all",):
        def rows(d: dict) -> list[dict]:
            return [r for k, r in sorted(d.items()) if ty == "all" or k.startswith(ty + "-")]
        e = {f"{h}_{s}": rate(rows(d)) for (h, s), d in runs.items()}
        e["steps_median"] = {f"{h}_{s}": statistics.median(r["steps_used"] for r in rows(d)) for (h, s), d in runs.items()}
        e["B_vlm_labels"] = dict(Counter(r["label"] for r in rows(runs[("B", "vlm")]) if not r["pass"]))
        e["B_script_outcomes"] = {s: dict(Counter(outcome(r) for r in rows(runs[("B", s)]))) for s in ("e3a", "e3b")}
        out["per_type"][ty] = e
    # review 011 F3: cabinet tasks, every navigate / pick / place with a via: door or room, and its result
    vias: dict[str, Counter] = {"door": Counter(), "room": Counter()}
    for (h, s), d in runs.items():
        if s != "vlm":
            continue
        for k, r in d.items():
            if not k.startswith("push_or_detour-"):
                continue
            for c in r["calls"]:
                head, _, res = c.partition(" -> ")
                if head.split("(")[0] in ("navigate", "pick", "place") and ", " in head:
                    via = head.rsplit(", ", 1)[1].rstrip(")")
                    vias["door" if via.endswith(" door") else "room"][f"{h}: {res}"] += 1
    out["cabinet_vias"] = {k: dict(v) for k, v in vias.items()}
    out["cabinet_runs_with_door_via"] = {h: sum(any(", " in c.split(" -> ")[0] and c.split(" -> ")[0].rstrip(")").endswith(" door")
                                                for c in r["calls"]) for k, r in runs[(h, "vlm")].items()
                                            if k.startswith("push_or_detour-")) for h in ("A", "B")}
    out["B_branches"] = {s: dict(sum((branches(r["calls"]) for r in runs[("B", s)].values()), Counter()))
                         for s in ("e3a", "e3b")}
    vb = runs[("B", "vlm")]
    out["B_vlm"] = {
        "leak_hits": sum(r["leak_hits"] for r in vb.values()),
        "sealed": dict(Counter(r.get("sealed") for r in vb.values())),
        "verified_sha256": dict(Counter(r.get("verified_sha256", "")[:16] for r in vb.values())),
        "vlm_median_s": statistics.median(r["timing"]["vlm_median_s"] for r in vb.values()),
        "run_wall_median_s": statistics.median(r["timing"]["run_wall_s"] for r in vb.values()),
        "failures": [{"task_id": k, "label": r["label"], "failed": r["failed"], "steps": r["steps_used"],
                      "calls": r["calls"]} for k, r in sorted(vb.items()) if not r["pass"]],
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({ty: {k: v[k]["pass"] for k in v if k[:2] in ("A_", "B_") and isinstance(v[k], dict) and "pass" in v[k]}
                      for ty, v in out["per_type"].items()}))


if __name__ == "__main__":
    main()
