"""Phase 4 clips (task 005): selection per the pre-registered rule (2026-11-13-005-clip-rule), rendering by sim/clip_phase3.render.

    MUJOCO_GL=egl python -m sim.clip_phase4 [select|render] [task_id ...]
`select` writes runs/clip4/selection.txt. Replay must reproduce every result string and the verdict, else no clip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sim import clip_phase3 as c3

RUNS = Path("runs/005/count5a")
OUT = Path("runs/clip4")
TYPES = ("blocked", "push_or_detour", "two_box")
CONTEXT = "counted run 1: blocked 9/10, push-or-detour 8/10, two-box 10/10, fetch 10/10"


def select() -> tuple[dict[str, str], list[str]]:
    res = {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("*/result.json"))}
    L = [f"Selection per the pre-registered rule (2026-11-13-005-clip-rule), applied mechanically to {RUNS}"]
    main, fails = {}, []
    for t in TYPES:
        rs = {k: v for k, v in res.items() if k.startswith(t + "-")}
        passed = sorted((v["steps_used"], v["seed"], k) for k, v in rs.items() if v["pass"])
        i = (len(passed) - 1) // 2  # lower median
        med = passed[i][0]
        main[t] = min((s, k) for st, s, k in passed if st == med)[1]
        L += ["", f"{t}: {len(rs)} counted runs, {len(passed)} passed",
              f"  passed steps_used sorted: {[(st, k) for st, s, k in passed]}",
              f"  n={len(passed)} ({'odd' if len(passed) % 2 else 'even: lower median'}): element {i + 1} -> median {med}",
              f"  ties at {med}: {[k for st, s, k in passed if st == med]} -> lowest seed -> {main[t]}"]
        for k, v in sorted(rs.items()):
            if k == main[t]:
                why = "CHOSEN type clip"
            elif not v["pass"]:
                why = f"CHOSEN failure clip (failed {v['failed']}, label {v['label']})"
                fails.append(k)
            else:
                why = f"not chosen: steps_used {v['steps_used']}" + (f" = median, seed {v['seed']} not lowest" if v["steps_used"] == med else f" != median {med}")
            L.append(f"  {k}: pass={v['pass']} steps={v['steps_used']}  {why}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return main, fails


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    sel, fails = select()
    if mode == "select":
        return
    only = sys.argv[2:]
    jobs = [(k, Path(f"showcase/phase4_{t}.mp4")) for t, k in sel.items()] + [(k, Path(f"showcase/failures/{k}.mp4")) for k in fails]
    ok = True
    for k, out in jobs:
        if only and k not in only:
            continue
        t = k.rsplit("-", 1)[0]
        entry = next(e for e in json.loads(Path(f"tasksets/phase4_{t}_counted.json").read_text())["tasks"] if e["task_id"] == k)
        ok &= c3.render(RUNS / k, entry, out, None, CONTEXT, OUT)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
