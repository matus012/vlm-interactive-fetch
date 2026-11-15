"""Rebuild a Phase 4 tuning (or counted) task's state by replaying the logged calls (runs/005/tune5a|tune5c/<task>/log.jsonl) through
SimRobot -> Skills -> Runtime(scripted=False), the agent path without the VLM. The calls are copied here (the logs live
outside the repo). Locally the replay matches the logged results up to the steps used by the tests (checked
2026-11-13); blocked-8217 diverges from step 10's bumper timing on, so its tests rebuild only steps 1-10.
A key "<task_id>@<run dir>" names another logged run of the same task (task 006: tune5d, count5a = the counted set,
tasksets/phase4_<type>_counted.json); each such prefix ends on the pick the pushed crate stopped. g5run1 = the 006 G5
counted run 1 (perceived mode, runs/006/g5run1, counted set): its prefixes end on the drive the crate stopped, the next
logged call being push(crate) (task 006 5d, row 17 addendum 4)."""
import json
from pathlib import Path

import mujoco

from agent.runtime import Runtime
from agent.skills import Skills
from sim.adapter import SimRobot
from sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
L, N, LOC, PK, PU = "look", "navigate", "locate", "pick", "push"
CALLS = {
    "blocked-8217": [(L, {}), (N, {"place": "bedroom"}), (L, {}), (N, {"place": "hall"}), (L, {}),
                     (N, {"place": "living room"}), (L, {}), (LOC, {"object": "red box"}), (PK, {"object": "red box"}),
                     (N, {"place": "storage"})],
    "blocked-8218": [(L, {}), (N, {"place": "storage"}), (L, {}), (N, {"place": "hall"}), (N, {"place": "bedroom"}),
                     (L, {}), (N, {"place": "living room"}), (L, {}), (N, {"place": "storage"}), (L, {}),
                     (N, {"place": "kitchen"}), (L, {}), (N, {"place": "storage"}), (L, {}), (N, {"place": "bedroom"}),
                     (L, {}), (LOC, {"object": "green box"})],
    "push_or_detour-8441": [(L, {}), (N, {"place": "bedroom"}), (L, {}), (N, {"place": "storage"}), (L, {}),
                            (N, {"place": "bedroom"}), (L, {}), (LOC, {"object": "green box"}),
                            (N, {"place": "kitchen"}), (L, {}), (LOC, {"object": "crate"}),
                            (LOC, {"object": "cabinet"}), (PU, {"object": "cabinet"}),
                            (N, {"place": "kitchen", "via": "kitchen south door"}), (L, {}),
                            (LOC, {"object": "green box"}), (PK, {"object": "green box"}), (N, {"place": "storage"}),
                            (N, {"place": "storage", "via": "hall"})],
    # tune5c (round 3), fix (d): each prefix ends on the drive the pushed crate stopped (the next call is push(crate))
    "blocked-8204": [(L, {}), (N, {"place": "bedroom"}), (L, {}), (N, {"place": "storage"}), (LOC, {"object": "crate"}),
                     (PU, {"object": "crate"}), (L, {}), (N, {"place": "living room"}), (L, {}),
                     (N, {"place": "kitchen"}), (L, {}), (N, {"place": "storage"}), (L, {}),
                     (LOC, {"object": "yellow box"}), (PK, {"object": "yellow box"})],
    "blocked-8208": [(L, {}), (N, {"place": "kitchen"}), (L, {}), (N, {"place": "hall"}), (L, {}),
                     (N, {"place": "bedroom"}), (L, {}), (N, {"place": "storage"}), (PU, {"object": "crate"}),
                     (N, {"place": "storage"}), (L, {}), (LOC, {"object": "yellow box"}), (PK, {"object": "yellow box"})],
    "blocked-8206": [(L, {}), (N, {"place": "bedroom"}), (L, {}), (N, {"place": "hall"}), (L, {}),
                     (N, {"place": "living room"}), (L, {}), (N, {"place": "kitchen"}), (L, {}),
                     (N, {"place": "storage"}), (LOC, {"object": "crate"}), (PU, {"object": "crate"}),
                     (N, {"place": "storage"}), (L, {}), (LOC, {"object": "red box"}), (PK, {"object": "red box"}),
                     (N, {"place": "kitchen table"})],
    "blocked-8204@tune5d": [(L, {}), (N, {"place": "living room"}), (L, {}), (N, {"place": "hall"}), (L, {}),
                            (N, {"place": "bedroom"}), (L, {}), (N, {"place": "storage"}), (LOC, {"object": "crate"}),
                            (N, {"place": "kitchen"}), (L, {}), (N, {"place": "hall"}), (PU, {"object": "crate"}),
                            (N, {"place": "storage"}), (L, {}), (LOC, {"object": "yellow box"}),
                            (PK, {"object": "yellow box"})],
    "blocked-8217@tune5b": [(L, {}), (N, {"place": "living room"}), (L, {}), (LOC, {"object": "red box"}),
                            (PK, {"object": "red box"}), ("place", {"surface": "storage bench"}), (L, {}),
                            (N, {"place": "storage door"}), (N, {"place": "storage bench"}), (LOC, {"object": "cabinet"}),
                            (L, {}), (LOC, {"object": "crate"}), (PU, {"object": "crate"}),
                            ("place", {"surface": "storage bench"}), (N, {"place": "storage bench"}), (L, {}),
                            (LOC, {"object": "crate"}), (PU, {"object": "crate"}), ("place", {"surface": "storage bench"})],
    "blocked-9200@count5a": [(L, {}), (N, {"place": "kitchen"}), (L, {}), (N, {"place": "storage"}),
                             (LOC, {"object": "crate"}), (PU, {"object": "crate"}), (N, {"place": "storage"}), (L, {}),
                             (LOC, {"object": "red box"}), (PK, {"object": "red box"})],
    "blocked-9204@g5run1": [(L, {}), (N, {"place": "bedroom"}), (L, {}), (LOC, {"object": "red box"}),
                            (N, {"place": "hall"}), (L, {}), (N, {"place": "living room"}), (L, {}),
                            (LOC, {"object": "red box"}), (PK, {"object": "red box"}), (N, {"place": "storage"})],
    "blocked-9225@g5run1": [(L, {}), (N, {"place": "living room"}), (L, {}), (N, {"place": "storage"}), (L, {})],
}


def replay(key: str, tmp: Path, poses: str = "truth") -> tuple[SimRobot, Runtime]:
    task_id, _, run = key.partition("@")
    ttype = task_id.rsplit("-", 1)[0]
    split = "counted" if run.startswith(("count", "g5")) else "tuning"
    entry = next(t for t in json.loads((ROOT / f"tasksets/phase4_{ttype}_{split}.json").read_text())["tasks"]
                 if t["task_id"] == task_id)
    task = make_task(entry["seed"], entry["type"], want=entry.get("slot"))
    # truth mode by default: these are replays of Phase 4 (truth-pose) tuning logs. poses="perceived" replays the same
    # calls in perceived mode (task 006 5c: the 9200 / 8204@tune5d prefixes reproduce there; a push that moves a
    # located box now drops the located poses, so a pick after it must locate again)
    robot = SimRobot(task["seed"], task["blocker"], photo_dir=None, poses=poses)
    rt = Runtime(Skills(robot), robot, task, tmp / "log.jsonl", scripted=False)
    for name, args in CALLS[key]:
        rt.call(name, args)
    return robot, rt


def set_xy(robot: SimRobot, body: str, xy: tuple[float, float]) -> None:
    """Test fixture: put a free body at xy (z and orientation kept), at rest."""
    j = robot.m.body(body).jntadr[0]
    robot.d.qpos[robot.m.jnt_qposadr[j]:robot.m.jnt_qposadr[j] + 2] = xy
    robot.d.qvel[robot.m.jnt_dofadr[j]:robot.m.jnt_dofadr[j] + 6] = 0.0
    mujoco.mj_forward(robot.m, robot.d)
