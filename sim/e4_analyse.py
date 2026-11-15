"""Task 014 (E4) report numbers. Committed before the counted pass; reads result.json (and log.jsonl for the leak scan).

    python -m sim.e4_analyse --run <runs/014/pass1> --meta <sealed/e4_100.meta.json> --out <json>

<run>/verified_<H>.jsonl = the regenerated sealed tasks (the task list: review 014 B3), <run>/<task_id>/ = the
frozen VLM agent, <run>/e3b/<task_id>/, <run>/e3c/<task_id>/ = the scripts on the same tasks. A task without a
result.json in an arm counts as a FAIL there and is listed ("no result"). Asserted: the task count and the per
(house, category) counts equal the seal meta's; every result present is sealed and carries its house's verified
file sha256 (the same in all three arms).

Value rule (task 014; proportional form fixed on 2026-11-14 before the seal, as C1 was dropped and the
categories are 34 / 33 / 33; row 29): per category (both houses pooled), best script = the better of E3b and
E3c on that category (a tie -> E3c; the verdict does not depend on it);
- "adds value": VLM - best script >= 20 percentage points AND VLM >= 50 % in that category;
- "no clear difference": |VLM - best script| < 20 points;
- "script better": best script - VLM >= 20 points.
(A margin >= 20 points with the VLM below 50 % is reported as "no clear difference (margin met, VLM below 50 %)", a
gap the rule's text leaves.) Overall: "the VLM adds value on E4" if >= 2 of the kept categories say adds value and
none says script better.
Also: Wilson 95 % per cell; per house; the paired table (VLM pass & best script fail / the reverse) per category and
per house; labels; steps median; place_ref split by relation (on a surface / next to furniture: the "on" half can be
answered from locate's text, review S2); secondary measures fixed before the seal (review S1 / S3):
recovery = the target located again after the move (a later "found" locate of it), and the move destination per arm;
hint = the share of runs whose first navigate goes to the hinted room, and median steps; the C3 move log; E4-leak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from sim.e3_analyse import wilson

CATS = ("vague", "place_ref", "recovery", "hint")
ARMS = ("vlm", "e3b", "e3c")
COLOUR = re.compile(r"\b(red|blue|green|yellow)\b", re.I)
MISSING = {"pass": False, "label": "no result", "steps_used": None, "calls": [], "missing": True}


def cell(rows: list[dict]) -> dict:
    k = sum(r["pass"] for r in rows)
    return {"pass": k, "n": len(rows), "wilson95": wilson(k, len(rows)) if rows else None}


def verdict(vlm: int, best: int, n: int) -> str:
    """The proportional form (2026-11-14, before the seal; row 29), in percentage points of the category."""
    d = 100.0 * (vlm - best) / n
    if -d >= 20.0 - 1e-9:
        return "script better"
    if d >= 20.0 - 1e-9:
        return "adds value" if 100.0 * vlm / n >= 50.0 - 1e-9 else "no clear difference (margin met, VLM below 50 %)"
    return "no clear difference"


def paired(a: dict, b: dict, sel: list[str]) -> dict:
    return {"vlm_only": sum(a[i]["pass"] and not b[i]["pass"] for i in sel),
            "script_only": sum(b[i]["pass"] and not a[i]["pass"] for i in sel),
            "both": sum(a[i]["pass"] and b[i]["pass"] for i in sel),
            "neither": sum(not a[i]["pass"] and not b[i]["pass"] for i in sel)}


def med(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads(a.meta.read_text())
    tasks, vsha = {}, {}
    for f in sorted(a.run.glob("verified_*.jsonl")):
        vsha[f.stem.split("_")[1]] = hashlib.sha256(f.read_bytes()).hexdigest()
        for x in f.read_text().splitlines():
            if x.strip():
                t = json.loads(x)
                tasks[t["task_id"]] = t
    ids = sorted(tasks)
    counts = Counter(f"{t.get('house', 'A')}:{t['category']}" for t in tasks.values())
    assert len(ids) == meta["n"] and dict(counts) == meta["counts"], (len(ids), dict(counts), meta["counts"])
    base = {"vlm": a.run, "e3b": a.run / "e3b", "e3c": a.run / "e3c"}
    runs: dict[str, dict] = {s: {} for s in ARMS}
    for s, d in base.items():
        for i in ids:
            p = d / i / "result.json"
            if p.exists():
                r = json.loads(p.read_text())
                h = tasks[i].get("house", "A")
                assert r.get("sealed") and r.get("verified_sha256") == vsha[h], (s, i)
                runs[s][i] = r
            else:
                runs[s][i] = dict(MISSING)
    cat_of = {i: tasks[i]["category"] for i in ids}
    house_of = {i: tasks[i].get("house", "A") for i in ids}
    out: dict = {"n": len(ids), "no_result": {s: [i for i in ids if runs[s][i].get("missing")] for s in ARMS},
                 "per_category": {}, "per_house": {}}
    built = [c for c in CATS if any(cat_of[i] == c for i in ids)]
    for c in built + ["all"]:
        sel = [i for i in ids if c == "all" or cat_of[i] == c]
        e = {s: cell([runs[s][i] for i in sel]) for s in ARMS}
        e["houses"] = {h: {s: cell([runs[s][i] for i in sel if house_of[i] == h]) for s in ARMS} for h in ("A", "B")}
        if c != "all":
            best = max(("e3b", "e3c"), key=lambda s: (e[s]["pass"], s == "e3c"))
            e["best_script"] = best
            e["verdict"] = verdict(e["vlm"]["pass"], e[best]["pass"], len(sel))
            e["paired"] = paired(runs["vlm"], runs[best], sel)
            e["paired_houses"] = {h: paired(runs["vlm"], runs[best], [i for i in sel if house_of[i] == h])
                                  for h in ("A", "B")}
            e["vlm_labels"] = dict(Counter(runs["vlm"][i]["label"] for i in sel if not runs["vlm"][i]["pass"]))
            e["steps_median"] = {s: med([runs[s][i]["steps_used"] for i in sel]) for s in ARMS}
        out["per_category"][c] = e
    for h in ("A", "B"):
        sel = [i for i in ids if house_of[i] == h]
        out["per_house"][h] = {s: cell([runs[s][i] for i in sel]) for s in ARMS}
    v = [out["per_category"][c]["verdict"] for c in built]
    out["overall"] = ("the VLM adds value on E4" if v.count("adds value") >= 2 and "script better" not in v
                      else "the VLM does not clearly add value on E4")
    out["verdicts"] = dict(zip(built, v))
    # place_ref split by relation
    pr = [i for i in ids if cat_of[i] == "place_ref"]
    rel = lambda i: "on" if tasks[i]["anchor"] in ("coffee table", "nightstand") else "next"  # noqa: E731
    out["place_ref_by_relation"] = {r: {s: cell([runs[s][i] for i in pr if rel(i) == r]) for s in ARMS}
                                    for r in ("on", "next")}
    # secondary: recovery

    def relocated(r: dict, box: str) -> bool:
        """A found locate of the target in the room it was moved to (re-read: not a repeat in a stale photo set)."""
        to = (r.get("move") or {}).get("to")
        room = to[0].replace("_", " ") if to else None
        return any(c.startswith(f"locate({box})") and f"-> found: {box} in {room}," in c for c in r["calls"])

    rec = [i for i in ids if cat_of[i] == "recovery"]
    out["recovery_secondary"] = {
        s: {"moved": sum(bool((runs[s][i].get("move") or {}).get("moved")) for i in rec),
            "located_again_after_move": sum(relocated(runs[s][i], tasks[i]["targets"][0]["object"]) for i in rec
                                            if (runs[s][i].get("move") or {}).get("moved")),
            "no_candidate": sum(bool((runs[s][i].get("move") or {}).get("failed")) for i in rec)} for s in ARMS}
    out["recovery_same_destination_all_arms"] = sum(
        len({json.dumps((runs[s][i].get("move") or {}).get("to")) for s in ARMS}) == 1 for i in rec)
    # secondary: hint

    def first_nav_hint(r: dict, room: str) -> bool:
        """The first navigate's place is exactly the hinted room (not 'bedroom door' / 'kitchen table')."""
        nav = next((c for c in r["calls"] if c.startswith("navigate(")), "")
        place = nav.split(" -> ")[0].removeprefix("navigate(").removesuffix(")").split(", ")[0]
        return place == room

    hint = [i for i in ids if cat_of[i] == "hint"]
    out["hint_secondary"] = {s: {"first_navigate_to_hinted_room": sum(first_nav_hint(runs[s][i], tasks[i]["hint_room"])
                                                                      for i in hint),
                                 "n": len(hint), "steps_median": med([runs[s][i]["steps_used"] for i in hint])}
                             for s in ARMS}
    out["move"] = {s: dict(Counter("moved" if (runs[s][i].get("move") or {}).get("moved") else
                                   ("no candidate" if (runs[s][i].get("move") or {}).get("failed") else "not located")
                                   for i in rec)) for s in ARMS}
    from agent.leak import scan_run
    have = [i for i in ids if not runs["vlm"][i].get("missing")]
    leak = {i: len(scan_run(a.run / i)) for i in have}
    colour_text = [i for i in have if cat_of[i] in ("vague", "place_ref")
                   and COLOUR.search(json.loads((a.run / i / "log.jsonl").read_text().splitlines()[0])
                                     ["extra"]["packet"]["user"].splitlines()[0])]
    out["leak"] = {"scan_hits": sum(leak.values()), "runs_with_hits": [i for i, n in leak.items() if n],
                   "colour_in_task_line": colour_text,
                   "result_leak_hits": sum(runs["vlm"][i].get("leak_hits", 0) for i in have)}
    out["timing"] = {"vlm_median_s": med([runs["vlm"][i]["timing"]["vlm_median_s"] for i in have]),
                     "run_wall_median_s": med([runs["vlm"][i]["timing"]["run_wall_s"] for i in have])}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"verdicts": out["verdicts"], "overall": out["overall"], "no_result": out["no_result"],
                      "per_category": {c: {s: out["per_category"][c][s]["pass"] for s in ARMS} for c in built + ["all"]},
                      "leak": out["leak"]["scan_hits"]}))


if __name__ == "__main__":
    main()
