"""Video run (task 013): replay one logged run's tool calls through the same stack, with full state logging.

    NAV116_HOUSE=A|B MUJOCO_GL=egl python -m sim.video_run --run runs/007/pass1/blocked-100040 --out runs/013/blocked-100040

Not a counted run and not part of the frozen system: the sealed results stay as recorded. It is the only entry point
besides the counted runner that opens a sealed seed (SimRobot(sealed_draw=True)), and only for a task whose counted
run is already logged and scored (the verified dict next to it). Same path as sim/agent_run.py: verified task dict ->
SimRobot(poses as logged) -> Skills -> Runtime(scripted=False) -> run_agent with a StubVLM that returns the logged tool
calls of every step (sim/clip_phase3's replay); E1 runs (task carries "easings") get the easings off and the no-E
rules text, house B runs need NAV116_HOUSE=B. NVIDIA only: perceived locate is pixel-dependent.

Writes OUT/trace.npz (t, qpos at every 1/60 s of sim time), OUT/events.json (every step's call / args / result and
its sim-time span; every photo's sim time, file and heading), OUT/photos/, OUT/log.jsonl, OUT/match.txt and
OUT/video_result.json. Valid only if every step's (call, args, validation, result, ok) equals the logged one and the
final verdict (pass, failed, rules, label, steps_used) equals the counted one; exit 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

TRACE_HZ = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the counted run dir (log.jsonl, result.json; verified.jsonl beside it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d_run, out = Path(a.run), Path(a.out)
    task = next(t for f in sorted(d_run.parent.glob("verified*.jsonl")) for t in map(json.loads, f.read_text().splitlines())
                if t and t["task_id"] == d_run.name)  # E4 (task 015): one verified file per house
    if task.get("labels"):
        from sim.world.generate import set_labels
        set_labels(True)
    if "easings" in task:
        import agent.loop
        from agent.context_e1 import build_packet_no_e
        from sim.world.generate import set_easings
        set_easings(False)
        agent.loop.build_packet = build_packet_no_e
    from agent.loop import end_run, run_agent
    from agent.runtime import Runtime
    from agent.skills import Skills
    from agent.vlm import StubVLM
    from sim.adapter import SimRobot
    from sim.world.generate import HOUSE
    assert task.get("house", "A") == HOUSE, f"task house {task.get('house', 'A')} != NAV116_HOUSE {HOUSE}"
    from sim.seal import gl_renderer
    vendor = gl_renderer()
    assert "NVIDIA" in vendor, f"renderer {vendor}: video runs need NVIDIA (perceived locate is pixel-dependent)"

    lines = [json.loads(x) for x in (d_run / "log.jsonl").read_text().splitlines() if x.strip()]
    steps_log, end_log = lines[:-1], lines[-1]
    counted = json.loads((d_run / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    robot = SimRobot(task["seed"], task["blocker"], photo_dir=out / "photos", poses=counted.get("poses", "perceived"),
                     sealed_draw=bool(counted.get("sealed")))
    m, d = robot.m, robot.d
    if "move" in task:  # E4 recovery: the same harness move as the counted run
        from sim.e4 import install_move
        install_move(robot, task)
    rt = Runtime(Skills(robot), robot, task, out / "log.jsonl", scripted=False)
    ts: list[float] = []
    qs: list[np.ndarray] = []
    st: dict[str, Any] = {"next": 0.0}
    events: dict[str, list] = {"steps": [], "photos": []}

    def rec() -> None:
        ts.append(float(d.time))
        qs.append(d.qpos.copy())

    orig_step = mujoco.mj_step

    def step(mm: mujoco.MjModel, dd: mujoco.MjData, *x: Any, **kw: Any) -> None:
        orig_step(mm, dd, *x, **kw)
        if dd is d and dd.time >= st["next"]:
            rec()
            st["next"] = dd.time + 1.0 / TRACE_HZ

    orig_shoot = robot._shoot

    def shoot(r: Any, index: int, heading_deg: float) -> Any:
        ph = orig_shoot(r, index, heading_deg)
        rec()
        events["photos"].append({"t": float(d.time), "file": Path(ph.path).name, "heading_deg": heading_deg,
                                 "step": len(rt.steps) + 1})
        return ph

    robot._shoot = shoot

    def hook(orig):
        def wrapped(*x: Any, **kw: Any):
            t0 = float(d.time)
            rec()
            r = orig(*x, **kw)
            rec()
            events["steps"].append({"step": len(rt.steps), "t0": t0, "t1": float(d.time), "result": r.text})
            return r
        return wrapped

    rt.call, rt.error_step = hook(rt.call), hook(rt.error_step)
    rec()
    mujoco.mj_step = step
    try:
        run_agent(robot, rt.skills, rt, task, StubVLM([s["extra"]["vlm"]["tool_calls"] for s in steps_log]), out)
        end = end_run(robot, rt, task)
    finally:
        mujoco.mj_step = orig_step
    rec()
    sc = end["score"]
    np.savez_compressed(out / "trace.npz", t=np.array(ts), qpos=np.array(qs))
    for e, s in zip(events["steps"], rt.steps):
        e |= {"call": s["call"], "args": s["args"]}
    (out / "events.json").write_text(json.dumps(events, indent=1))

    key = lambda s: None if s is None else (s["call"], s["args"], s["validation"], s["result"], s["ok"])  # noqa: E731
    mine, rows, same = rt.steps, [], len(rt.steps) == len(steps_log)
    for i in range(max(len(steps_log), len(mine))):
        L = steps_log[i] if i < len(steps_log) else None
        R = mine[i] if i < len(mine) else None
        eq = key(L) == key(R)
        same &= eq
        dt = abs(L["sim_s"] - R["sim_s"]) if L and R else float("nan")
        rows.append(f"{i + 1:2d} {'MATCH' if eq else 'DIFF '}  {L['call'] if L else '-'} -> logged {L['result'] if L else '-'!r}"
                    f" | video run {R['result'] if R else '-'!r}  (sim_s diff {dt:.3f})")
    ls = end_log["score"]
    v_log = (ls["pass"], ls["failed"], ls["rules"], end_log["label"], end_log["steps_used"])
    v_me = (sc["pass"], sc["failed"], sc["rules"], end["label"], end["steps_used"])
    same &= v_log == v_me
    head = (f"video run {task['task_id']}: {'VALID' if same else 'INVALID'}; {len(steps_log)} logged steps, {len(mine)} replayed; "
            f"verdict counted pass={ls['pass']} label={end_log['label']} steps={end_log['steps_used']} | video run "
            f"pass={sc['pass']} label={end['label']} steps={end['steps_used']} ({'same' if v_log == v_me else 'DIFFERENT'}); "
            f"renderer {vendor}; sim {d.time:.1f} s, {len(ts)} trace rows")
    (out / "match.txt").write_text(head + "\n" + "\n".join(rows) + "\n")
    (out / "video_result.json").write_text(json.dumps(
        {"task_id": task["task_id"], "valid": same, "counted_pass": ls["pass"], "video_pass": sc["pass"],
         "targets": {k: v["surface"] for k, v in sc["detail"]["targets"].items()} if sc["pass"] else None,
         "house": task.get("house", "A"), "easings": task.get("easings", True), "blocker": task["blocker"],
         "seed": task["seed"], "instruction": task["instruction"], "sim_s": float(d.time)}, indent=1))
    print(head)
    print("\n".join(rows))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
