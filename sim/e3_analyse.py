"""Task 010 report numbers: the fixed scripts (E3a, E3b) vs the VLM on the sealed 100 (seal-005).

    python -m sim.e3_analyse --runs <runs/010/pass1> --base <runs/007/pass1> --out <json>
    python -m sim.e3_analyse --branches <dir with */result.json>     # branch counts (dev runs)

Per type: pass / 25 with Wilson 95 % for VLM, E3a, E3b; steps median; where each script failed (`outcome`); tasks
the VLM passed and a script failed, and the reverse. Reads result.json only (calls + verdict).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[int, int]:
    """Wilson 95 % interval in % (as sim.e1_analyse; not imported: that module switches the easings off)."""
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (c - h)), round(100 * (c + h)))


def rate(rows: list[dict]) -> dict:
    k = sum(r["pass"] for r in rows)
    return {"pass": k, "n": len(rows), "wilson95": wilson(k, len(rows))}


def results(d: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in d.glob("*/result.json")}

TYPES = ("fetch", "blocked", "push_or_detour", "two_box")


def outcome(r: dict) -> str:
    """Where a script run ended, from its calls (scorer verdict aside)."""
    c = r["calls"]
    if r["pass"]:
        return "pass"
    if not c[-1].startswith("done("):
        return "step limit"
    note = c[-1]
    if "delivered no box" in note and not any(x.startswith("pick(") for x in c):
        return "target not found"
    last_fail = next((x for x in reversed(c[:-1]) if " -> failed" in x or " -> error" in x), "")
    if last_fail.startswith("push("):
        return "blocked, push failed"
    if last_fail.startswith("navigate(") and "via" in last_fail or (last_fail.startswith("navigate(") and "," in last_fail.split(") ->")[0]):
        return "go-round failed"
    if last_fail:
        return f"{last_fail.split('(')[0]} {last_fail.split(' -> ')[1]}"
    return "done, scorer fail"


def branches(calls: list[str]) -> Counter:
    b: Counter = Counter()
    for i, x in enumerate(calls):
        nxt = calls[i + 1] if i + 1 < len(calls) else ""
        if x.endswith("-> failed: blocked"):
            b["blocked"] += 1
        if x.startswith("push(crate) -> pushed"):
            b["crate pushed, step retried"] += nxt.split(" -> ")[0] == calls[i - 1].split(" -> ")[0]
        if x.startswith("push(crate) -> error: unknown object") and nxt.startswith("push(cabinet)"):
            b["no crate -> push cabinet"] += 1
        if x.startswith("push(") and " -> failed" in x:
            b["push failed"] += 1
            b["go round (via)"] += "," in nxt.split(") ->")[0] and nxt.startswith("navigate(")
        if x.startswith("push(") and " -> failed" in x and (nxt.startswith("navigate(") and "," not in nxt.split(") ->")[0]
                                                         and not any(c.startswith("pick(") and "holding" in c for c in calls[:i])):
            b["search moved on (push failed)"] += 1
        if x.startswith("navigate(") and " -> failed" in x and not x.endswith("blocked") and nxt.startswith("navigate("):
            b["search moved on (navigate failed)"] += 1
        if x.startswith("locate(") and "found" in x.split(" -> ")[1][:5]:
            b["target found"] += 1
    if calls and calls[-1].startswith("done(") and "no box" in calls[-1] and not any(c.startswith("pick(") for c in calls):
        b["not found anywhere -> done"] += 1
    if sum(c.startswith("place(") and c.endswith("placed") for c in calls) >= 2:
        b["second box delivered"] += 1
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--branches")
    a = ap.parse_args()
    if a.branches:
        tot: Counter = Counter()
        for r in results(Path(a.branches)).values():
            tot += branches(r["calls"])
        print(json.dumps(dict(sorted(tot.items()))))
        return
    base = results(Path(a.base))
    runs = {s: results(Path(a.runs) / s) for s in ("e3a", "e3b")}
    assert all(len(v) == 100 for v in runs.values()) and len(base) == 100
    out: dict = {"per_type": {}, "total": {}}
    for ty in TYPES + ("all",):
        def rows(d: dict) -> list[dict]:
            return [r for k, r in sorted(d.items()) if ty == "all" or k.startswith(ty + "-")]
        e = {"vlm": rate(rows(base)), **{s: rate(rows(runs[s])) for s in runs}}
        e["steps_median"] = {"vlm": statistics.median(r["steps_used"] for r in rows(base)),
                             **{s: statistics.median(r["steps_used"] for r in rows(runs[s])) for s in runs}}
        e["outcomes"] = {s: dict(Counter(outcome(r) for r in rows(runs[s]))) for s in runs}
        e["vlm_only"] = {s: [k for k, r in sorted(base.items()) if (ty == "all" or k.startswith(ty + "-")) and r["pass"]
                             and not runs[s][k]["pass"]] for s in runs}
        e["script_only"] = {s: [k for k, r in sorted(base.items()) if (ty == "all" or k.startswith(ty + "-"))
                                and not r["pass"] and runs[s][k]["pass"]] for s in runs}
        out["per_type" if ty != "all" else "total"][ty] = e
    out["branches"] = {s: dict(sum((branches(r["calls"]) for r in runs[s].values()), Counter())) for s in runs}
    # row 14 labels with `script` in place of `vlm` (task 010): the script made the choices the VLM makes
    out["labels"] = {s: dict(Counter({"vlm": "script"}.get(r["label"], r["label"]) for r in runs[s].values() if not r["pass"]))
                     for s in runs}
    out["leak_hits"] = {s: sum(r["leak_hits"] for r in runs[s].values()) for s in runs}
    out["vlm_field"] = {s: dict(Counter(r["vlm"] for r in runs[s].values())) for s in runs}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({ty: {k: v[k] for k in ("vlm", "e3a", "e3b")} for ty, v in {**out["per_type"], **out["total"]}.items()}))


if __name__ == "__main__":
    main()
