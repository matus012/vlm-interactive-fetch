"""One Phase 3 run (task 004): a taskset entry -> SimRobot -> Skills -> Runtime(scripted=False) -> VLM loop -> score.

    python -m sim.agent_run --taskset tasksets/phase3_tuning.json --task-id fetch-4000 --out <run_dir>
        --base-url http://127.0.0.1:<port>/v1 [--model-name qwen3.8-27b] [--stub <calls.json> | --stub script]
    API key: env VLLM_API_KEY (not needed with --stub).
    Sealed run (row 20): --verified <hpc/seal_regen.py output .jsonl> in place of --taskset; SimRobot gets sealed_draw=True.
    E1 (task 009, row 22): --e1 = the generator's easings off (sim.world.generate.set_easings(False), before any world)
    and the rules text without line E (agent/context_e1.py); the task must carry "easings": False, and only with --e1.
    E2 (task 011): house B runs need NAV116_HOUSE=B in the environment; the task's "house" must match it.
    E3 (task 010): --script e3a|e3b plays agent/e3.py's fixed script in place of the VLM (same packets, runtime, scorer).

`--stub script` plays agent/scripts.py CORRECT[type] for this task through the VLM tool-call path: the script is run
once as a real scripted run on a throw-away sim, and the calls it made (blocked / push / detour reactions included)
become the stub's list. The list is saved as <run_dir>/stub_calls.json.

Writes <run_dir>/log.jsonl (one line per step + the end line), photos/, tools.json and, LAST, result.json (tmp +
rename): a run dir without result.json is incomplete. <run_dir> must not exist yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from agent.loop import end_run, run_agent
from agent.runtime import Runtime
from agent.scripts import CORRECT
from agent.skills import Skills
from agent.vlm import OpenAIVLM, StubVLM
from sim.adapter import SimRobot
from sim.tasks import make_task, script_truth

TASK_KEYS = ("task_id", "type", "seed", "blocker", "instruction", "targets", "step_limit")


def script_calls(task: dict) -> list[dict]:
    """The correct script's calls AS THEY RAN on a real scripted run of this task (blocked / detour tasks need the
    reactions to the sim's own `failed: blocked` etc., which a recorder whose calls all succeed cannot give)."""
    with tempfile.TemporaryDirectory() as td:
        robot = SimRobot(task["seed"], task["blocker"], photo_dir=Path(td) / "photos")
        rt = Runtime(Skills(robot), robot, task, Path(td) / "log.jsonl", scripted=True)
        CORRECT[task["type"]](rt, task, script_truth(task))
        rt.finish(None)
        return [{"name": s["call"], "arguments": s["args"]} for s in rt.steps]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--taskset")
    src.add_argument("--verified")  # row 20: hpc/seal_regen.py's verified task dicts (.jsonl, or .json {"tasks": [...]})
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--stub")
    ap.add_argument("--model-name")
    ap.add_argument("--poses", choices=("perceived", "truth"), default="perceived")  # 006: perceived = the measured system
    ap.add_argument("--e1", action="store_true")  # task 009 (row 22): easings off + rules text without E
    ap.add_argument("--script", choices=("e3a", "e3b", "e3c"))  # task 010: a fixed script in place of the VLM; 014: e3c
    a = ap.parse_args()
    if a.e1:
        import agent.loop
        from agent.context_e1 import build_packet_no_e
        from sim.world.generate import set_easings
        set_easings(False)
        agent.loop.build_packet = build_packet_no_e
    t0 = time.time()
    if a.verified:  # row 20: the dict was regenerated and checked at seal-005 (row 21); no make_task, no HEAD re-check
        raw = Path(a.verified).read_bytes()
        rows = ([json.loads(x) for x in raw.splitlines() if x.strip()] if a.verified.endswith(".jsonl")
                else json.loads(raw)["tasks"])
        task = next(t for t in rows if t["task_id"] == a.task_id)
    else:
        entry = next(t for t in json.loads(Path(a.taskset).read_text())["tasks"] if t["task_id"] == a.task_id)
        assert ("easings" in entry) == a.e1, "E1 task <-> --e1"
        task = make_task(entry["seed"], entry["type"], want=entry.get("slot"))  # Phase 4 sets: the sampler slot
        assert {k: task[k] for k in TASK_KEYS} == {k: entry[k] for k in TASK_KEYS}, (task, entry)
    assert ("easings" in task) == a.e1 and task.get("easings", False) is False, "E1 task <-> --e1"
    if task.get("labels"):  # task 014 (E4 vague): printed box labels, before the first world
        from sim.world.generate import set_labels
        set_labels(True)
    from sim.world.generate import HOUSE
    assert task.get("house", "A") == HOUSE, f"task house {task.get('house', 'A')} != NAV116_HOUSE {HOUSE} (task 011)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.script == "e3c":
        from agent.e3c import ScriptPolicy as ScriptPolicyC
        vlm = ScriptPolicyC(a.script)
    elif a.script:
        from agent.e3 import ScriptPolicy
        vlm = ScriptPolicy(a.script)
    elif a.stub == "script":
        calls = script_calls(task)
        (out / "stub_calls.json").write_text(json.dumps(calls, indent=1))
        vlm = StubVLM(calls)
    elif a.stub:
        vlm = StubVLM.from_file(a.stub)
    else:
        vlm = OpenAIVLM(a.base_url, os.environ["VLLM_API_KEY"], a.model_name)
    robot = SimRobot(task["seed"], task["blocker"], photo_dir=out / "photos", poses=a.poses,
                     sealed_draw=a.verified is not None)
    move = None
    if "move" in task:  # task 014 (E4 recovery): the harness moves the target once, after its first locate
        from sim.e4 import install_move
        move = install_move(robot, task)
    skills = Skills(robot)
    rt = Runtime(skills, robot, task, out / "log.jsonl", scripted=False)
    run_agent(robot, skills, rt, task, vlm, out)
    end = end_run(robot, rt, task)
    sc = end["score"]
    res = {"task_id": task["task_id"], "seed": task["seed"], "instruction": task["instruction"],
           "vlm": "stub" if a.stub else vlm.model, "poses": a.poses, "pass": sc["pass"], "label": end["label"], "failed": sc["failed"],
           "steps_used": end["steps_used"], "done_called": end["done_called"],
           "n_validation_errors": end["n_validation_errors"], "timing": end["timing"] | {"run_wall_s": round(time.time() - t0, 1)},
           "n_unseen_box_contacts": end["n_unseen_box_contacts"], "unseen_box_contacts": end["unseen_box_contacts"],
           "places_reached": end["places_reached"], "kitchen_east_calls": end["kitchen_east_calls"],
           "leak_hits": end["leak_hits"], "sim_ok": end["sim_ok"],
           "calls": [f"{s['call'] or '-'}({', '.join(map(str, s['args'].values())) if isinstance(s['args'], dict) else s['args']})"
                     f" -> {s['result']}" for s in rt.steps]}
    if a.e1:
        res |= {"e1": True, "rules": "no-E"}
    if task.get("e4"):
        res |= {"category": task["category"], "house": task.get("house", "A"), "move": move}
    if a.verified:
        res |= {"sealed": True, "verified_sha256": hashlib.sha256(raw).hexdigest()}
    tmp = out / ".result.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out / "result.json")
    print(json.dumps({k: res[k] for k in ("task_id", "pass", "label", "failed", "steps_used", "leak_hits")}))


if __name__ == "__main__":
    main()
