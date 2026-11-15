"""Agent runtime loop (task 004): packet -> VLM -> parse -> Runtime.call -> repeat, until `done` or the step limit.
Stdlib only.

    run_agent(robot, skills, runtime, task, vlm, run_dir)   the loop; writes <run_dir>/tools.json (schemas sent)
    end_run(robot, runtime, task) -> end line               snapshot(2 s) -> score(steps = runtime.steps_used) -> finish
    check_log(run_dir) -> [problem, ...]                    G3-log: [] = complete step log, one label iff failed
    python -m agent.loop <run_dir>...                       prints G3-log per dir; exit 1 if any is incomplete

A reply with zero or several tool calls is an error step ("error: call exactly one tool"): it counts as a step and is
the next packet's last result. Invalid JSON arguments, unknown skills and unknown names go through the runtime's own
validate_call / check_names (error result, counts as a step). Every step record's `extra` holds the packet (system,
user, photo paths, photo labels), the raw VLM reply (content, reasoning, tool calls, usage, latency, retries), the
parsed call and the call seed; the runtime adds validation, result, skill wall time and sim time.

End-line summary: validation errors, timing totals, unseen-box contacts during navigate (sim/gate2.py's
box_contacts_nav rule: a navigate contact key whose object is a fetch box, minus the box held at that step, as
gate2's bumper scan), places reached, `kitchen east` use, leak hits over this run's packets + tools.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from agent.context import build_packet
from agent.leak import scan
from agent.robot_api import RobotAPI
from agent.runtime import LABELS, Runtime
from agent.scorer import score
from agent.skills import Skills
from agent.tools import TOOLS
from agent.vlm import Reply

ONE_TOOL = "error: call exactly one tool"
STEP_KEYS = ("step", "call", "args", "validation", "result", "ok", "skill_wall_s", "photos", "extra")


def run_agent(robot: RobotAPI, skills: Skills, runtime: Runtime, task: dict, vlm: Callable[..., Reply],
              run_dir: str | Path) -> None:
    Path(run_dir, "tools.json").write_text(json.dumps(TOOLS, indent=1))
    while not runtime.over:
        pk = build_packet(robot, skills, runtime, task)
        seed = int(task["seed"]) * 100 + runtime.steps_used + 1
        reply = vlm(pk, seed)
        extra = {"packet": asdict(pk), "vlm": reply.as_log(), "seed": seed,
                 "parsed_call": reply.tool_calls[0] if len(reply.tool_calls) == 1 else None}
        if extra["parsed_call"] is None:
            runtime.error_step(ONE_TOOL, extra)
        else:
            runtime.call(extra["parsed_call"]["name"], extra["parsed_call"]["arguments"], extra)


def nav_box_contacts(steps: list[dict], boxes: set[str]) -> list[dict]:
    out, held = [], None
    for s in steps:
        if s["call"] in ("pick", "place") and s["ok"]:
            held = s["args"]["object"] if s["call"] == "pick" else None
        if s["call"] != "navigate":
            continue
        for k, v in (s.get("meta", {}).get("contacts") or {}).items():
            if k.split("|")[0] in boxes - {held}:
                out.append({"step": s["step"], "key": k, "pen_m": v, "result": s["result"]})
    return out


def summary(robot: RobotAPI, runtime: Runtime) -> dict:
    steps = runtime.steps
    lat = [s["extra"]["vlm"]["latency_s"] for s in steps if "extra" in s]
    usage = [s["extra"]["vlm"]["usage"] for s in steps if "extra" in s]
    texts = [json.dumps(TOOLS)] + [t for s in steps if "extra" in s for t in
                                   (s["extra"]["packet"]["system"], s["extra"]["packet"]["user"],
                                    *s["extra"]["packet"]["labels"])]
    boxes = {n for n, v in robot.objects().items() if v["class"] == "box"}
    contacts = nav_box_contacts(steps, boxes)
    nav = [s for s in steps if s["call"] == "navigate" and s["validation"] == "ok"]
    return {"n_validation_errors": sum(s["validation"] != "ok" for s in steps),
            "timing": {"wall_s": round(time.time() - runtime._t0, 2), "vlm_s": round(sum(lat), 2),
                       "vlm_median_s": round(statistics.median(lat), 3) if lat else None,
                       "vlm_max_s": max(lat, default=None), "skill_wall_s": round(sum(s["skill_wall_s"] for s in steps), 2),
                       "sim_s": round(sum(s.get("sim_dt", 0.0) for s in steps), 2),
                       "vlm_retries": sum(len(s["extra"]["vlm"]["retries"]) for s in steps if "extra" in s),
                       "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                       "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage)},
            "unseen_box_contacts": contacts, "n_unseen_box_contacts": len(contacts),
            "places_reached": [s["args"]["place"] for s in nav if s["result"] == "reached"],
            "kitchen_east_calls": sum("kitchen east" in (s["args"]["place"], s["args"].get("via")) for s in nav),
            "leak_hits": sum(len(scan(t)) for t in texts)}


def end_run(robot: RobotAPI, runtime: Runtime, task: dict) -> dict:
    sc = score(robot.snapshot(2.0), task, runtime.steps_used)
    return runtime.finish(sc, summary(robot, runtime))


def check_log(run_dir: str | Path) -> list[str]:
    d = Path(run_dir)
    if not (d / "log.jsonl").exists() or not (d / "result.json").exists():
        return ["missing log.jsonl or result.json"]
    lines = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if x.strip()]
    ends = [x for x in lines if x.get("event") == "end"]
    if len(ends) != 1 or lines[-1] is not ends[0]:
        return [f"{len(ends)} end lines (need exactly 1, last)"]
    steps, end, bad = lines[:-1], ends[0], []
    if [s.get("step") for s in steps] != list(range(1, len(steps) + 1)) or len(steps) != end["steps_logged"]:
        bad.append("step numbers not 1..n or != steps_logged")
    for s in steps:
        miss = [k for k in STEP_KEYS if k not in s] + (["sim_s"] if "sim_s" not in s else [])
        ex = s.get("extra", {})
        miss += [f"packet.{k}" for k in ("system", "user", "photos", "labels") if k not in ex.get("packet", {})]
        miss += [f"vlm.{k}" for k in ("content", "reasoning", "tool_calls", "latency_s") if k not in ex.get("vlm", {})]
        miss += ["parsed_call"] * ("parsed_call" not in ex)
        miss += [f"photo file {p}" for p in ex.get("packet", {}).get("photos", [])
                 if p and not (d / "photos" / Path(p).name).exists()]  # photos live in <run_dir>/photos (sim.agent_run)
        if miss:
            bad.append(f"step {s.get('step')}: missing {miss}")
    sc = end.get("score") or {}
    if "pass" not in sc:
        bad.append("no scorer verdict")
    elif not sc["pass"] and end.get("label") not in LABELS:
        bad.append(f"failed run label {end.get('label')!r} is not exactly one of {LABELS}")
    elif sc["pass"] and end.get("label") is not None:
        bad.append("passed run has a label")
    res = json.loads((d / "result.json").read_text())
    if (res.get("pass"), res.get("label"), res.get("steps_used")) != (sc.get("pass"), end.get("label"),
                                                                      end.get("steps_used")):
        bad.append("result.json disagrees with the end line")
    return bad


if __name__ == "__main__":
    n_bad = 0
    for rd in sys.argv[1:]:
        probs = check_log(rd)
        n_bad += bool(probs)
        print(f"{rd}: {'complete' if not probs else 'INCOMPLETE ' + '; '.join(probs)}")
    print(f"G3-log: {len(sys.argv) - 1 - n_bad}/{len(sys.argv) - 1} complete")
    sys.exit(1 if n_bad else 0)
