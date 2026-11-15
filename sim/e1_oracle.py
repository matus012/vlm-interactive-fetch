"""Task 009 step 4, the E1 oracle split (after the counted pass is written): the Phase 2 correct script on each E1
task, run as sim/gate2.py's script_trial runs it but with truth poses (task 009 step 4; gate2's default is perceived). A label, not a score: "solvable by the skills" if the
script passes (push_or_detour: either branch, push_then_detour or detour).

    python -m sim.e1_oracle --verified <verified.jsonl> --task-id <id> --out <dir>   # <dir>/oracle.json (+ logs)

Easings off (set before any world). Runs on NVIDIA only (script_truth picks room points by rendered px). The task is
regenerated with make_task(..., sealed_draw=True) first and must equal the verified entry (TASK_KEYS, easings, and
the out-of-sight decision hall_px < PX_OUT_OF_SIGHT per target); the renderer must be NVIDIA. oracle.json records the
code commit (.git/HEAD, as hpc/seal_regen.py reads it) and the renderer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sim.world.generate import set_easings

set_easings(False)

from agent.runtime import Runtime  # noqa: E402 - after set_easings
from agent.scorer import score  # noqa: E402
from agent.scripts import CORRECT  # noqa: E402
from agent.skills import Skills  # noqa: E402
from sim.adapter import SimRobot  # noqa: E402
from hpc.seal_regen import head_commit  # noqa: E402
from sim.agent_run import TASK_KEYS  # noqa: E402
from sim.constants import PX_OUT_OF_SIGHT  # noqa: E402
from sim.seal import REPO, gl_renderer  # noqa: E402
from sim.tasks import make_task, script_truth  # noqa: E402


def commit() -> str:
    """HEAD sha without git (GPU nodes have none): detached in the PERUN clone; a branch ref read from .git here."""
    h = head_commit(REPO)
    return (REPO / ".git" / h[5:].strip()).read_text().strip() if h.startswith("ref:") else h


def trial(task: dict, branch: str | None, out: Path) -> dict:
    tdir = out / (branch or "script")
    tdir.mkdir(parents=True, exist_ok=False)
    w0 = time.time()
    truth = script_truth(task)
    robot = SimRobot(task["seed"], task["blocker"], poses="truth", sealed_draw=True)
    rt = Runtime(Skills(robot), robot, task, tdir / "log.jsonl")
    if branch:
        CORRECT[task["type"]](rt, task, truth, branch)
    else:
        CORRECT[task["type"]](rt, task, truth)
    steps_used = rt.steps_used
    sc = score(robot.snapshot(2.0), task, steps_used)
    end = rt.finish(sc)
    return {"branch": branch, "pass": sc["pass"], "failed": sc["failed"], "label": end["label"],
            "steps_used": steps_used, "calls": [f"{s['call']} -> {s['result']}" for s in rt.steps],
            "wall_s": round(time.time() - w0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = next(t for t in map(json.loads, Path(a.verified).read_text().splitlines()) if t["task_id"] == a.task_id)
    assert task.get("easings") is False, "not an E1 task"
    renderer = gl_renderer()
    assert "NVIDIA" in renderer, f"renderer {renderer!r} is not NVIDIA"
    # opens the test seed for script_truth's world() (sim.tasks._SEALED_OK) and re-checks the task against the draw
    again = make_task(task["seed"], task["type"], sealed_draw=True, want=task["slot"])
    assert all(again[k] == task[k] for k in (*TASK_KEYS, "easings")), "task differs from its regeneration"
    assert all((again["hall_px"][b] < PX_OUT_OF_SIGHT) == (v < PX_OUT_OF_SIGHT) for b, v in task["hall_px"].items())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    branches = ("push_then_detour", "detour") if task["type"] == "push_or_detour" else (None,)
    rows = [trial(task, b, out) for b in branches]
    res = {"task_id": task["task_id"], "solvable": any(r["pass"] for r in rows), "trials": rows,
           "commit": commit(), "renderer": renderer}
    tmp = out / ".oracle.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out / "oracle.json")
    print(json.dumps({"task_id": task["task_id"], "solvable": res["solvable"]}))


if __name__ == "__main__":
    main()
