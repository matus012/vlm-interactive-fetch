"""Phase 2 gates (task 003) through the full stack: SimRobot -> Skills -> Runtime (JSONL per run) -> scorer.

    python -m sim.gate2 <gate> [--jobs N] [--out runs/gate2] [--poses perceived|truth]
    gates: pick place push-crate push-cabinet blocked carry-push script wrong carry-cover bumper

Every trial: its own dir runs/gate2/<gate>/<trial id>/ with log.jsonl (Runtime: one line per skill call + the end
line) and result.json; runs/gate2/<gate>/gate.json = bar, verdict, every trial row. photo_dir None (segmentation
visibility is still computed for every photo, so locate / seen / R4(b) work as in a real run). --poses (task 006):
the SimRobot pose mode of every trial (default perceived: locate / pick / avoid from the depth camera; truth = the
Phase 2-4 runs); recorded in gate.json. Trial rules, fixed before running:

- pick: seeds 2000, 2001, ...; seed s takes generate(s).boxes[(s - 2000) % 4]; a floor box is accepted while fewer than
  15 floor trials are in, a surface box always (>= 5 of 20 on a low surface); stop at 20. Robot spawned at the box
  room's room point (tasks.room_point_for: the point whose look sees the box most), places.yaml yaw. Skills look ->
  locate(box) -> pick(box). The adapter's pick_phase hook holds the lift for G2_HOLD_S in total (pick_box's own
  SETTLE_S + the rest) before pick_box measures lift / held, so the lift height in pick's detail is reading (a).
  (a) = ruling (binding): "holding <box>", both pads in contact at the end of the lift hold, box bottom >=
  G2_PICK_LIFT above its support after that hold, AND the box between both pads (adapter pad_touch() == 2) when pick returns
  in the carry pose. (b) = info: "holding <box>", box bottom >= G2_PICK_LIFT above its support when pick returns and
  2 s later, held (both pads) 2 s after the return.
- place: seeds 3000, ...; box (s - 3000) % 4, any support; trial i targets kitchen table if i even else storage bench (house A; layout place_surfaces order);
  same spawn + look -> locate -> pick, then place(surface) (the adapter drives to the approach point itself).
  Trials whose pick fails are not counted; the first 10 counted per surface form the gate. Pass: scorer.on_target on
  snapshot(2.0) for that box and surface (centre in rect - 2 cm, on top, upright <= 15 deg, still 2 s), snapshot
  gripper_open and arm_retracted.
- push-crate: seeds 1100-1109, generate(s, "crate"), robot at the hall spawn; crate jittered by
  push_check.jitter (rng [seed, 2]: +-3 cm along / across the passage, +-3 deg yaw, free-joint qpos write) right
  after SimRobot's start settle, then 0.5 s settle. navigate(<room behind the door>) -> expect failed: blocked ->
  push(crate) -> navigate(same). Pass: push "pushed", second navigate "reached", sim_ok.
- cabinet start jitter (task 005 row 12; push-cabinet and blocked cabinet): the robot spawns at the "living room" room
  point (house A; generally the room point of the cabinet door's non-cabinet room) moved by U(-CAB_JIT_ALONG, +) along
  the door normal (house A: x, toward the kitchen), U(-CAB_JIT_LAT, +) across it (y)
  and yawed by U(-CAB_JIT_DEG, +), rng default_rng([seed, 3]); the A* route into the doorway, and so the approach
  angle and the first-contact time / point, follow from the start. +-0.10 m / +-0.15 m / +-10 deg keep the start 2.15 m
  or more from the wall and >= 0.85 m inside the 1.0 m lateral gap to either door axis, so the robot still drives
  straight into the cabinet's doorway. Every trial records its start pose and its first robot-cabinet contact
  (d.time, world contact point, from the StepWatch over every physics step); gate.json counts distinct trials
  (distinct first-contact time to the ms or point to the cm).
- push-cabinet: seeds 1100-1109, generate(s, "cabinet"), robot spawned at the jittered "living room" room point (house A).
  navigate("kitchen" (house A), via=<cabinet's door>) -> push(cabinet). Pass (task table): push "failed: not movable", cabinet
  true-pose displacement over the whole trial <= G2_CABINET_MAX_MOVE, max robot-cabinet penetration over every physics
  step of the trial (mj_step wrapped) <= G2_CABINET_MAX_PEN, sim_ok. The first navigate's "failed: blocked" is
  recorded beside it.
- blocked: crate seeds 1200-1209 with the same jitter, from the hall spawn into the room behind the door; cabinet seeds
  1200-1209, from the jittered living room point, navigate("kitchen", via=door) (house A). Pass: "failed: blocked", detector fired,
  detection - first robot-blocker contact <= G2_BLOCKED_S, sim_ok.
- carry-push (G4-carry-push, task 005): seeds 1300-1309, generate(s, "crate"), crate jitter as push-crate. The box:
  the first of generate(s).boxes (generate order) whose room is not the one behind the crate door (reachable without
  the push). Robot spawned at that box's room point (tasks.room_point_for), look -> locate -> pick through Skills /
  Runtime (as the pick gate), then navigate(<room behind the door>) -> push(crate) -> navigate(same). Pass: pick
  "holding <box>", push "pushed", the box held with both pads (adapter pad_touch() == 2) right after the push, second navigate
  "reached", sim_ok. A failed pick counts as a failed trial. Info: the first navigate (expected "failed: blocked"), arm
  joint drift over the push (max |q_after - q_before|, rad), the push's footprint (robot_ext_m: base-frame rect, robot_radius_m).
- script: dev_tasks(type, 10) per type, run_script (push_or_detour: both branches on each task), scored by
  score(snapshot(2.0), task, steps_used). wrong: dev_tasks("fetch", 3) x the four wrong scripts.
  steps_used = len(rt.steps) + rt.n_refused (calls refused at the limit count: a run that hit the limit asked for more
  than the limit).
- carry-cover: G1-cover method (sim/cover_check.py: spots, yaws, room points, 100 px bar, base yaw set directly) for
  one room with the arm + a held box in the carry pose captured from a real pick (info, deviation 7).
- bumper: scans every log.jsonl under --out for navigate steps whose blocked detector fired on "bumper" while the
  robot touched a fetch box, and every navigate fetch-box contact (R4(b) unseen-box contacts).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from sim.push_check import behind_room, cabinet_axes, cabinet_rooms, room_place
from sim.world.generate import load_layout

OUT = Path("runs/gate2")
G2_HOLD_S = 2.0  # lift hold before the (a) measurement (the gate's "held 2 s")
PICK_SEED0, PLACE_SEED0, N_PICK, N_PICK_SURFACE, N_PLACE = 2000, 3000, 20, 5, 10
PUSH_SEEDS, BLOCKED_SEEDS, CARRY_PUSH_SEEDS = range(1100, 1110), range(1200, 1210), range(1300, 1310)
CAB_JIT_ALONG, CAB_JIT_LAT, CAB_JIT_DEG = 0.10, 0.15, 10.0  # cabinet start jitter (module doc)
POSES = "perceived"  # SimRobot pose mode of every trial (--poses; set in each worker by _pool's initializer)


def _set_poses(poses: str) -> None:
    global POSES
    POSES = poses
_LAY = load_layout()
CRATE_GOAL = {d: room_place(behind_room(_LAY, d)) for d in _LAY["blockers"]["crate"]["doors"]}  # house A: bedroom, storage
KT, SB = _LAY["place_surfaces"][:2]  # house A: kitchen table, storage bench


def _j(x: Any) -> Any:
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _setup_rt(robot: Any, tdir: Path, task: dict) -> tuple[Any, Any]:
    from agent.runtime import Runtime
    from agent.skills import Skills
    sk = Skills(robot)
    return sk, Runtime(sk, robot, task, tdir / "log.jsonl")


def _save(tdir: Path, row: dict) -> dict:
    row = _j(row)
    (tdir / "result.json").write_text(json.dumps(row, indent=1) + "\n")
    return row


def _pose(place: str) -> tuple[float, float, float]:
    from sim.world.generate import load_places
    p = next(q for q in load_places() if q["name"] == place)
    return float(p["pose"][0]), float(p["pose"][1]), math.radians(float(p["pose"][2]))


class StepWatch:
    """Wraps mujoco.mj_step (every caller uses the module attribute) to record the max penetration between two geom
    sets over every physics step while active, and the first contact (d.time, world contact point)."""

    def __init__(self, a: np.ndarray, b: np.ndarray) -> None:
        self.a, self.b, self.pen, self._orig = a, b, 0.0, None
        self.first: dict | None = None

    def __enter__(self) -> "StepWatch":
        self._orig = orig = mujoco.mj_step

        def step(m: mujoco.MjModel, d: mujoco.MjData, *args: Any, **kw: Any) -> None:
            orig(m, d, *args, **kw)
            if d.ncon:
                g = d.contact.geom[: d.ncon]
                hit = (np.isin(g[:, 0], self.a) & np.isin(g[:, 1], self.b)) | \
                      (np.isin(g[:, 1], self.a) & np.isin(g[:, 0], self.b))
                if hit.any():
                    self.pen = max(self.pen, float(-d.contact.dist[: d.ncon][hit].min()))
                    if self.first is None:
                        i = int(np.flatnonzero(hit)[0])
                        self.first = {"t": float(d.time), "pos": [float(v) for v in d.contact.pos[i]]}

        mujoco.mj_step = step
        return self

    def __exit__(self, *exc: Any) -> None:
        mujoco.mj_step = self._orig


# ---- pick / place ---------------------------------------------------------------------------------------------

def pick_list(n: int = N_PICK) -> list[tuple[int, str]]:
    from sim.world.generate import generate
    out, floor, s = [], 0, PICK_SEED0
    while len(out) < n:
        b = generate(seed=s, blocker="none").boxes[(s - PICK_SEED0) % 4]
        if b["surface"] != "floor" or floor < n - N_PICK_SURFACE:
            out.append((s, f"{b['colour']} box"))
            floor += b["surface"] == "floor"
        s += 1
    return out


def _pick(seed: int, box: str, tdir: Path, task: dict) -> tuple[dict, Any, Any, Any]:
    """Spawn at the box's room point, look -> locate -> pick through Skills / Runtime. Returns (row, robot, rt)."""
    from sim.body import run
    from sim.grasp import SETTLE_S, holding, pad_contacts
    from sim.snapshot import box_bottom
    from sim.adapter import SimRobot
    from sim.constants import G2_PICK_LIFT
    from sim.tasks import room_point_for
    rp = room_point_for(seed, "none", box)
    robot = SimRobot(seed, "none", start=_pose(rp), poses=POSES)
    body = robot.body_of[box]
    w = robot.world
    wb = next(b for b in w.boxes if b["name"] == body)
    lift: dict = {}

    def phase(name: str, d: mujoco.MjData) -> None:
        if name == "lift":
            run(robot.m, d, G2_HOLD_S - SETTLE_S)
            pc = pad_contacts(robot.m, d, body)
            lift.update(pads_both=bool(pc["left_pad"] and pc["right_pad"]), held=holding(robot.m, d, body),
                        t=float(d.time))

    robot.pick_phase = phase
    sk, rt = _setup_rt(robot, tdir, task)
    texts = []
    for name, args in (("look", {}), ("locate", {"object": box}), ("pick", {"object": box})):
        r = rt.call(name, args)
        texts.append(r.text)
        if not r.ok:
            break
    det = rt.steps[-1].get("meta", {}).get("detail", {}) if rt.steps[-1]["call"] == "pick" else {}
    g = det.get(f"grasp{det.get('attempts', 1)}", {})
    sup = g.get("support_z")
    text = texts[-1]
    held_ret = robot.pad_touch().get(box) == 2  # both pads (the gate's reading; held() is grasp.holding, row 18)
    h_ret = box_bottom(robot.d, body) - sup if sup is not None else None
    pc_ret = pad_contacts(robot.m, robot.d, body)
    row = {"seed": seed, "box": box, "room": wb["room"], "support": wb["surface"], "box_yaw": wb["yaw"],
           "room_point": rp, "texts": texts, "result": text, "lift_hold_s": G2_HOLD_S,
           "a_lift_above_support_m": g.get("lift_above_support_m"), "a_pads_both": lift.get("pads_both"),
           "a_held_after_hold": lift.get("held"), "ret_held": held_ret,
           "ret_pads_both": bool(pc_ret["left_pad"] and pc_ret["right_pad"]), "b_ret_above_support_m": h_ret,
           "attempts": det.get("attempts"), "pick_reason": rt.steps[-1]["result"]}
    ok_text = text == f"holding {box}"
    row["pass_a"] = bool(ok_text and lift.get("pads_both") and (g.get("lift_above_support_m") or 0.0) >= G2_PICK_LIFT
                         and held_ret)
    return row, robot, rt, (ok_text, sup, body)


def pick_trial(i: int, seed: int, box: str, out: Path) -> dict:
    from sim.body import run
    from sim.grasp import pad_contacts
    from sim.snapshot import box_bottom
    from sim.constants import G2_PICK_LIFT
    tdir = out / "pick" / f"{i:02d}_s{seed}_{box.split()[0]}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    row, robot, rt, (ok_text, sup, body) = _pick(seed, box, tdir, {"task_id": f"pick-{seed}", "step_limit": 20})
    t_ret = robot.sim_time()
    run(robot.m, robot.d, 2.0)
    pc = pad_contacts(robot.m, robot.d, body)
    h2 = box_bottom(robot.d, body) - sup if sup is not None else None
    row.update(i=i, b_after2s_above_support_m=h2, b_held_after2s=robot.pad_touch().get(box) == 2,
               b_pads_both_after2s=bool(pc["left_pad"] and pc["right_pad"]), carry_hold_s=robot.sim_time() - t_ret)
    row["pass_b"] = bool(ok_text and row["ret_held"] and min(row["b_ret_above_support_m"] or 0.0, h2 or 0.0)
                         >= G2_PICK_LIFT and row["b_held_after2s"] and row["b_pads_both_after2s"])
    row["sim_ok"] = robot.sim_ok()
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row["pass_a"], "pass_b": row["pass_b"]})
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


def place_trial(i: int, seed: int, box: str, surface: str, out: Path) -> dict:
    from agent.scorer import on_target
    tdir = out / "place" / f"{i:02d}_s{seed}_{box.split()[0]}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    row, robot, rt, (ok_text, _, _) = _pick(seed, box, tdir, {"task_id": f"place-{seed}", "step_limit": 20})
    row.update(i=i, surface=surface, counted=ok_text)
    if ok_text:
        r = rt.call("place", {"surface": surface})
        snap = robot.snapshot(2.0)
        ok, det = on_target(snap["boxes"][box], snap["surfaces"][surface], snap["settle_s"])
        row.update(place_result=r.text, on_target=ok, scorer_detail=det, gripper_open=snap["gripper_open"],
                   arm_retracted=snap["arm_retracted"], arm_track_err_rad=snap["arm_track_err_rad"],
                   held_end=snap["held"], sim_ok=snap["sim_ok"],
                   place_detail={k: v for k, v in rt.steps[-1].get("meta", {}).get("detail", {}).items()
                                 if k in ("go_to", "centre_margin_m", "timings")})
        row["pass"] = bool(ok and snap["gripper_open"] and snap["arm_retracted"])
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row.get("pass", False)} if ok_text else None)
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


# ---- push / blocked -------------------------------------------------------------------------------------------

def _jittered(seed: int, variant: str, start: tuple[float, float, float] | None) -> tuple[Any, list[float] | None]:
    from sim.adapter import SimRobot
    from sim.body import run
    from sim.push_check import SETTLE_S, door_dict, jitter
    robot = SimRobot(seed, variant, start=start, poses=POSES)
    jit = None
    if variant == "crate":
        jit = jitter(robot.m, robot.d, "crate", door_dict(robot.layout, robot.world.blocker["door"]), seed)
        run(robot.m, robot.d, SETTLE_S)
    return robot, jit


def cabinet_start(seed: int) -> tuple[float, float, float]:
    """The jittered start-room start for a cabinet trial (module doc; house A: living room). Rooms and axes come from
    the seed's own cabinet door (house A: both cabinet doors give the same ones; task 011: house B's may differ)."""
    from sim.world.generate import generate
    rng = np.random.default_rng([seed, 3])
    door = generate(seed=seed, blocker="cabinet").blocker["door"]
    ka, kc, sign = cabinet_axes(_LAY, door)
    p = list(_pose(room_place(cabinet_rooms(_LAY, door)[0])))
    p[ka] += sign * float(rng.uniform(-CAB_JIT_ALONG, CAB_JIT_ALONG))
    p[kc] += float(rng.uniform(-CAB_JIT_LAT, CAB_JIT_LAT))
    return p[0], p[1], p[2] + math.radians(float(rng.uniform(-CAB_JIT_DEG, CAB_JIT_DEG)))


def cabinet_goal(door: str) -> str:
    """The VLM place name of the cabinet room (navigate target of the cabinet trials; house A: kitchen)."""
    return room_place(cabinet_rooms(_LAY, door)[1])


def distinct_trials(rows: list[dict]) -> dict:
    """Distinct first robot-blocker contacts over trials: by time (ms), by point (cm), and by either."""
    f = [r["first_contact"] for r in rows if r.get("first_contact")]
    ts = {round(c["t"], 3) for c in f}
    ps = {tuple(round(v, 2) for v in c["pos"][:2]) for c in f}
    both = {(round(c["t"], 3), tuple(round(v, 2) for v in c["pos"][:2])) for c in f}
    return {"n_with_contact": len(f), "distinct_t": len(ts), "distinct_point": len(ps), "distinct_trials": len(both)}


def _blocker_contact(robot: Any, body: str) -> tuple[float | None, float]:
    names = {robot.m.geom(int(g)).name or str(int(g)) for g in robot._geoms[body]}
    g = robot.last_nav
    keys = {k: v for k, v in g["contacts"].items() if k.split("|")[0] in names}
    return min((g["first_contact_t"][k] for k in keys), default=None), max(keys.values(), default=0.0)


def push_crate_trial(seed: int, out: Path) -> dict:
    tdir = out / "push-crate" / f"s{seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    robot, jit = _jittered(seed, "crate", None)
    door = robot.world.blocker["door"]
    goal = CRATE_GOAL[door]
    p0 = robot.true_pose("crate")
    sk, rt = _setup_rt(robot, tdir, {"task_id": f"push-crate-{seed}", "step_limit": 20})
    r1 = rt.call("navigate", {"place": goal})
    nav1 = {k: robot.last_nav.get(k) for k in ("blocked", "max_lag_m", "max_bump_n")}
    r2 = rt.call("push", {"object": "crate"})
    r3 = rt.call("navigate", {"place": goal})
    p1 = robot.true_pose("crate")
    pdet = rt.steps[1].get("meta", {}).get("detail", {})
    row = {"seed": seed, "door": door, "goal": goal, "jitter_along_lat_m_yaw_deg": jit, "nav1": r1.text,
           "nav1_detail": rt.steps[0].get("meta", {}).get("detail"), "nav1_blocked": nav1, "push": r2.text,
           "nav2": r3.text, "nav2_detail": rt.steps[2].get("meta", {}).get("detail"),
           "nav2_contacts": rt.steps[2].get("meta", {}).get("contacts"),
           "crate_start": [p0.x, p0.y, p0.yaw], "crate_end": [p1.x, p1.y, p1.yaw],
           "push_detail": {k: pdet.get(k) for k in ("plan", "fmax_n", "pen_max_m", "tilt_deg", "corridor_clear",
                                                     "sim_s", "approach_reached", "return_reached")},
           "sim_ok": robot.sim_ok()}
    row["pass"] = bool(r2.text == "pushed" and r3.text == "reached" and row["sim_ok"])
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row["pass"]})
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


def push_cabinet_trial(seed: int, out: Path) -> dict:
    from sim.constants import G2_CABINET_MAX_MOVE, G2_CABINET_MAX_PEN
    from sim.look import robot_geoms
    from sim.tasks import DOOR_NAMES
    tdir = out / "push-cabinet" / f"s{seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    start = cabinet_start(seed)
    robot, _ = _jittered(seed, "cabinet", start)
    door = robot.world.blocker["door"]
    c0 = robot.true_pose("cabinet")
    sk, rt = _setup_rt(robot, tdir, {"task_id": f"push-cabinet-{seed}", "step_limit": 20})
    with StepWatch(robot_geoms(robot.m), np.asarray(robot._geoms["cabinet"])) as sw:
        r1 = rt.call("navigate", {"place": cabinet_goal(door), "via": DOOR_NAMES[door]})
        nav1 = {k: robot.last_nav.get(k) for k in ("blocked", "max_lag_m", "max_bump_n")}
        r2 = rt.call("push", {"object": "cabinet"})
    c1 = robot.true_pose("cabinet")
    move = math.hypot(c1.x - c0.x, c1.y - c0.y)
    move3 = math.sqrt((c1.x - c0.x) ** 2 + (c1.y - c0.y) ** 2 + (c1.z - c0.z) ** 2)
    pdet = rt.steps[1].get("meta", {}).get("detail", {})
    row = {"seed": seed, "door": door, "via": DOOR_NAMES[door], "start": start, "first_contact": sw.first,
           "nav1": r1.text,
           "nav1_detail": rt.steps[0].get("meta", {}).get("detail"), "nav1_blocked": nav1, "push": r2.text,
           "cabinet_move_m": move3, "cabinet_move_xy_m": move, "cabinet_yaw_change_deg": math.degrees(c1.yaw - c0.yaw),
           "pen_max_whole_run_m": sw.pen, "push_detail_pen_max_m": pdet.get("pen_max_m"), "sim_ok": robot.sim_ok()}
    row["pass"] = bool(r2.text == "failed: not movable" and move3 <= G2_CABINET_MAX_MOVE
                       and sw.pen <= G2_CABINET_MAX_PEN and row["sim_ok"])
    row["nav1_blocked_ok"] = r1.text == "failed: blocked"
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row["pass"]})
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


def blocked_trial(variant: str, seed: int, out: Path) -> dict:
    from sim.constants import G2_BLOCKED_S
    from sim.tasks import DOOR_NAMES
    tdir = out / "blocked" / f"{variant}_s{seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    from sim.look import robot_geoms
    start = cabinet_start(seed) if variant == "cabinet" else None
    robot, jit = _jittered(seed, variant, start)
    door, body = robot.world.blocker["door"], robot.world.blocker["body"]
    p0 = robot.true_pose(body)
    sk, rt = _setup_rt(robot, tdir, {"task_id": f"blocked-{variant}-{seed}", "step_limit": 20})
    args = {"place": CRATE_GOAL[door]} if variant == "crate" else {"place": cabinet_goal(door), "via": DOOR_NAMES[door]}
    with StepWatch(robot_geoms(robot.m), np.asarray(robot._geoms[body])) as sw:
        r = rt.call("navigate", args)
    t_contact, pen = _blocker_contact(robot, body)
    b = robot.last_nav["blocked"]
    dt = b["t"] - t_contact if b is not None and t_contact is not None else None
    p1 = robot.true_pose(body)
    row = {"variant": variant, "seed": seed, "door": door, "args": args, "jitter_along_lat_m_yaw_deg": jit,
           "start": start, "first_contact": sw.first,
           "result": r.text, "detail": rt.steps[0].get("meta", {}).get("detail"), "signal": (b or {}).get("why"),
           "first_contact_t": t_contact, "detect_t": (b or {}).get("t"), "detect_after_contact_s": dt,
           "max_pen_m": pen, "blocker_disp_m": math.hypot(p1.x - p0.x, p1.y - p0.y),
           "other_contacts": {k: v for k, v in robot.last_nav["contacts"].items()
                              if k.split("|")[0] not in {robot.m.geom(int(g)).name for g in robot._geoms[body]}},
           "sim_ok": robot.sim_ok()}
    row["pass"] = bool(r.text == "failed: blocked" and b is not None and dt is not None and dt <= G2_BLOCKED_S
                       and row["sim_ok"])
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row["pass"]})
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


def carry_push_box(seed: int) -> str:
    """The carry-push trial's box (module doc): first box not in the room behind the crate door."""
    from sim.world.generate import generate
    w = generate(seed=seed, blocker="crate")
    behind = behind_room(w.layout, w.blocker["door"])
    return next(f"{b['colour']} box" for b in w.boxes if b["room"] != behind)


def carry_push_trial(seed: int, out: Path) -> dict:
    from sim.robot import ARM_JOINTS
    from sim.tasks import room_point_for
    tdir = out / "carry-push" / f"s{seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    box = carry_push_box(seed)
    rp = room_point_for(seed, "crate", box)
    robot, jit = _jittered(seed, "crate", _pose(rp))
    door = robot.world.blocker["door"]
    goal = CRATE_GOAL[door]
    sk, rt = _setup_rt(robot, tdir, {"task_id": f"carry-push-{seed}", "step_limit": 20})
    texts = {}
    for name, args in (("look", {}), ("locate", {"object": box}), ("pick", {"object": box})):
        texts[name] = rt.call(name, args).text
    row: dict[str, Any] = {"seed": seed, "door": door, "goal": goal, "box": box, "room_point": rp,
                           "jitter_along_lat_m_yaw_deg": jit, "pick": texts["pick"]}
    if texts["pick"] == f"holding {box}":
        r1 = rt.call("navigate", {"place": goal})
        qa = [int(robot.m.joint(j).qposadr[0]) for j in ARM_JOINTS]
        q0 = robot.d.qpos[qa].copy()
        r2 = rt.call("push", {"object": "crate"})
        q1 = robot.d.qpos[qa].copy()
        held_after = robot.pad_touch().get(box) == 2
        r3 = rt.call("navigate", {"place": goal})
        pdet = next(s for s in rt.steps if s["call"] == "push").get("meta", {}).get("detail", {})
        row.update(nav1=r1.text, push=r2.text, held_after_push=held_after, nav2=r3.text,
                   held_end=robot.pad_touch().get(box) == 2, arm_drift_rad=float(np.abs(q1 - q0).max()),
                   push_detail={k: pdet.get(k) for k in ("plan", "why", "obj_moved_m", "robot_ext_m", "robot_radius_m",
                                                         "approach_lat", "obj_along0", "obj_lat0", "obj_along",
                                                         "obj_lat", "corridor_clear", "tilt_deg", "pen_max_m")})
    row["sim_ok"] = robot.sim_ok()
    row["pass"] = bool(row.get("push") == "pushed" and row.get("held_after_push") and row.get("nav2") == "reached"
                       and row["sim_ok"])
    rt.call("done", {"note": "gate trial"})
    rt.finish({"pass": row["pass"]})
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


# ---- scripts ----------------------------------------------------------------------------------------------------

def _nav_box_contacts(steps: list[dict], boxes: set[str]) -> list[dict]:
    out = []
    for s in steps:
        if s["call"] != "navigate":
            continue
        for k, v in (s.get("meta", {}).get("contacts") or {}).items():
            if k.split("|")[0] in boxes:
                out.append({"step": s["step"], "key": k, "pen_m": v, "result": s["result"],
                            "detail": s.get("meta", {}).get("detail", "")})
    return out


def script_trial(kind: str, ttype: str, seed: int, script: str, out: Path) -> dict:
    """kind 'script' (script = branch for push_or_detour, else '-') or 'wrong' (script = wrong-script name)."""
    from agent.scorer import score
    from agent.scripts import CORRECT, WRONG
    from sim.adapter import SimRobot
    from sim.tasks import make_task, script_truth
    tdir = out / kind / f"{ttype}_s{seed}_{script}"
    tdir.mkdir(parents=True, exist_ok=True)
    w0 = time.time()
    task = make_task(seed, ttype)
    truth = script_truth(task)
    robot = SimRobot(seed, task["blocker"], poses=POSES)
    sk, rt = _setup_rt(robot, tdir, task)
    if kind == "wrong":
        WRONG[script](rt, task, truth)
    elif ttype == "push_or_detour":
        CORRECT[ttype](rt, task, truth, script)
    else:
        CORRECT[ttype](rt, task, truth)
    steps_used = rt.steps_used
    snap = robot.snapshot(2.0)
    sc = score(snap, task, steps_used)
    end = rt.finish(sc)
    fail_step = next((s for s in rt.steps if not s["ok"]), None)
    boxes = set(robot.objects()) - {"crate", "cabinet"}
    contacts = _nav_box_contacts(rt.steps, boxes)
    row = {"kind": kind, "type": ttype, "seed": seed, "script": script, "task": task, "truth": truth,
           "pass": sc["pass"], "failed": sc["failed"], "label": end["label"], "steps_used": steps_used,
           "steps_logged": len(rt.steps), "n_refused": rt.n_refused, "done_called": end["done_called"],
           "fail_step": None if fail_step is None else {k: fail_step[k] for k in ("step", "call", "args", "result")}
           | {"detail": (fail_step.get("meta") or {}).get("detail")},
           "calls": [f"{s['call']}({', '.join(map(str, s['args'].values()))}) -> {s['result']}" for s in rt.steps],
           "score_detail": sc["detail"], "box_contacts_nav": contacts, "n_box_contacts_nav": len(contacts),
           "sim_s": robot.sim_time()}
    row["wall_s"] = time.time() - w0
    return _save(tdir, row)


# ---- carry cover (info) -------------------------------------------------------------------------------------------

def carry_cover(out: Path, room: str = cabinet_rooms(_LAY, _LAY["blockers"]["cabinet"]["doors"][0])[0]) -> dict:
    """G1-cover count for `room` with the arm + held box in the carry pose (captured from a real pick through the
    stack) vs LOOK_ARM_QPOS, same spots / yaws / room points / bar / renderer as sim/cover_check.py."""
    from sim.body import base_pose
    from sim.constants import G1_COVER_MIN_PX
    from sim.cover_check import room_cases
    from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, render_look, set_arm, set_yaw_direct
    from sim.robot import BASE_JOINTS, CAM_H, CAM_W
    from sim.world.generate import LAYOUT, generate, load_layout, load_places, room_points
    from sim.world.house import load_house, set_base
    # 1) carry state from a real pick: first pick-list-rule seed with a box in `room` that the stack picks
    s, cap = PICK_SEED0, None
    while cap is None:
        b = generate(seed=s, blocker="none").boxes[(s - PICK_SEED0) % 4]
        if b["room"] == room:
            tdir = out / "carry-cover" / f"pick_s{s}"
            tdir.mkdir(parents=True, exist_ok=True)
            row, robot, rt, (ok, _, body) = _pick(s, f"{b['colour']} box", tdir, {"task_id": "carry", "step_limit": 20})
            if ok and row["ret_held"]:
                m0, d0 = robot.m, robot.d
                bp = base_pose(m0, d0)
                c, sn = math.cos(-bp[2]), math.sin(-bp[2])
                rel = d0.body(body).xpos[:2] - bp[:2]
                yaw_b = math.atan2(float(d0.body(body).xmat[3]), float(d0.body(body).xmat[0]))
                cap = {"seed": s, "body": body, "joints": {m0.joint(j).name: d0.qpos[m0.joint(j).qposadr[0]: m0.joint(j).qposadr[0] + 1].tolist()
                                                           for j in range(m0.njnt) if m0.jnt_type[j] in (2, 3)
                                                           and m0.joint(j).name not in BASE_JOINTS},
                       "rel_xy": [c * rel[0] - sn * rel[1], sn * rel[0] + c * rel[1]], "z": float(d0.body(body).xpos[2]),
                       "rel_yaw": yaw_b - float(bp[2]), "base_yaw": float(bp[2]), "quat_world": d0.body(body).xquat.tolist(),
                       "tilt_from_up": float(d0.body(body).xmat[8])}
                rt.finish(None)
        s += 1
    # 2) cover world: seed 0, only the moved box + the held box
    held = cap["body"]
    moved = "box_red" if held != "box_red" else "box_blue"
    w = generate(LAYOUT, 0, "none")
    for b in w.boxes:
        if b["name"] not in (moved, held):
            w.xml = re.sub(rf'    <body name="{b["name"]}".*?</body>\n', "", w.xml, flags=re.S)
    m, d = load_house(w)
    gid = m.geom(moved).id
    qm = m.joint(m.body(moved).jntadr[0]).qposadr[0]
    qh = m.joint(m.body(held).jntadr[0]).qposadr[0]
    layout, places = load_layout(), load_places()
    spots, cases = room_cases(layout, room)
    pts = room_points(room, places)
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    res: dict = {"room": room, "capture": {k: v for k, v in cap.items() if k != "joints"}, "n_cases": len(cases)}
    try:
        for mode in ("look", "carry"):
            px = np.zeros((len(cases), len(HEADINGS_DEG)), dtype=int)
            for pt in pts:
                set_base(m, d, (pt[0], pt[1], 0.0))
                if mode == "look":
                    set_arm(m, d, LOOK_ARM_QPOS)
                    d.qpos[qh: qh + 3] = (0.0, 0.0, -5.0)  # held box out of the house (not in view)
                else:
                    for jn, v in cap["joints"].items():
                        a = m.joint(jn).qposadr[0]
                        d.qpos[a: a + len(v)] = v
                for j, h in enumerate(HEADINGS_DEG):
                    th = math.radians(h)
                    set_yaw_direct(m, d, th)
                    if mode == "carry":
                        c, sn = math.cos(th), math.sin(th)
                        x, y = cap["rel_xy"]
                        dq = np.array([math.cos((th - cap["base_yaw"]) / 2), 0, 0, math.sin((th - cap["base_yaw"]) / 2)])
                        qw = np.zeros(4)
                        mujoco.mju_mulQuat(qw, dq, np.asarray(cap["quat_world"]))
                        d.qpos[qh: qh + 7] = (pt[0] + c * x - sn * y, pt[1] + sn * x + c * y, cap["z"], *qw)
                    for i, (x, y, z, yw) in enumerate(cases):
                        d.qpos[qm: qm + 7] = (x, y, z, math.cos(yw / 2), 0, 0, math.sin(yw / 2))
                        mujoco.mj_forward(m, d)
                        px[i, j] = max(px[i, j], int((render_look(r, d)[2] == gid).sum()))
            mx = px.max(axis=1)
            res.setdefault("_mx", {})[mode] = mx
            res[mode] = {"n_pass": int((mx >= G1_COVER_MIN_PX).sum()), "pct": 100.0 * float((mx >= G1_COVER_MIN_PX).mean()),
                         "min_max_px": int(mx.min()),
                         "failing": [[*cases[i][:3], round(math.degrees(cases[i][3])), int(mx[i])]
                                     for i in np.flatnonzero(mx < G1_COVER_MIN_PX)]}
            print(mode, {k: v for k, v in res[mode].items() if k != "failing"}, flush=True)
            from PIL import Image
            set_yaw_direct(m, d, 0.0)
            mujoco.mj_forward(m, d)
            Image.fromarray(render_look(r, d)[0]).save(out / "carry-cover" / f"{mode}_h000.png")
    finally:
        r.close()
    lo, hi = res.pop("_mx").values()
    res["cases_px_lower_in_carry"] = int((hi < lo).sum())
    res["px_drop_max"] = int((lo - hi).max())
    return res


# ---- bumper / box contacts scan -------------------------------------------------------------------------------

def _base_geoms(blocker: str) -> set[str]:
    """Contact-key labels (name or id, as go_to writes them) of the base_link geoms the bumper reads, per world variant
    (geom ids depend only on the variant: every world has the same furniture and four boxes)."""
    from sim.world.generate import generate
    from sim.world.house import load_house
    m, _ = load_house(generate(seed=1000, blocker=blocker))
    b = m.body("base_link").id
    return {m.geom(g).name or str(g) for g in range(m.ngeom) if m.geom_bodyid[g] == b}


def bumper_scan(out: Path) -> dict:
    """Navigate steps where the blocked detector fired on "bumper" while a fetch box touched a base_link geom (the only
    geoms the bumper reads; the held box's pad / finger contacts are not counted), split by whether the blocker was
    touched in the same step; plus every navigate contact of a fetch box with any robot geom (R4(b))."""
    boxes = {f"{c} box" for c in ("red", "blue", "green", "yellow")}
    gate_blocker = {"push-crate": "crate", "push-cabinet": "cabinet", "pick": "none", "place": "none"}
    base = {v: _base_geoms(v) for v in ("none", "crate", "cabinet")}
    rows, contacts, unseen = [], [], []
    logs = sorted(out.glob("*/*/log.jsonl"))
    for f in logs:
        rf = f.parent / "result.json"
        res = json.loads(rf.read_text()) if rf.exists() else {}
        blk = res.get("task", {}).get("blocker") or res.get("variant") or gate_blocker.get(f.parent.parent.name, "none")
        steps = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        steps = [s for s in steps if "call" in s]
        run = str(f.parent.relative_to(out))
        for c in _nav_box_contacts(steps, boxes):
            c["run"] = run
            contacts.append(c)
        held = None
        for s in steps:
            if s["call"] in ("pick", "place") and s["ok"]:
                held = s["args"]["object"] if s["call"] == "pick" else None
            if s["call"] != "navigate":
                continue
            ck = s.get("meta", {}).get("contacts") or {}
            hit = sorted({k for k in ck if k.split("|")[0] in boxes - {held} and k.split("|")[1] in base[blk]})
            if not hit:
                continue
            ev = {"run": run, "step": s["step"], "result": s["result"], "detail": s.get("meta", {}).get("detail", ""),
                  "box_base_keys": hit, "blocker_touched": any(k.split("|")[0] in ("crate", "cabinet") for k in ck)}
            unseen.append(ev)
            if s["result"] == "failed: blocked" and str(ev["detail"]).startswith("bumper"):
                rows.append(ev)
    return {"bumper_fired_with_box_on_base": rows, "n_events": len(rows), "runs": sorted({r["run"] for r in rows}),
            "n_events_without_blocker_contact": sum(not r["blocker_touched"] for r in rows),
            "box_base_contacts_nav": unseen, "n_box_base_contacts_nav": len(unseen),
            "n_runs_scanned": len(logs), "n_nav_box_contacts_any_robot_geom": len(contacts),
            "push_note": "sim/push.py push_object calls go_to(detect=False) for its approach / return and drives "
                         "the pushes with drive_base / push_line (no detector): push() cannot trip the detector."}


# ---- driver -------------------------------------------------------------------------------------------------------

def _pool(fn: Callable, args: list[tuple], jobs: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"), initializer=_set_poses,
                             initargs=(POSES,)) as ex:
        futs = [ex.submit(fn, *a) for a in args]
        rows = []
        for a, f in zip(args, futs):
            rows.append(f.result())
            r = rows[-1]
            print(json.dumps({k: r.get(k) for k in ("seed", "box", "variant", "type", "script", "result", "pass",
                                                    "pass_a", "pass_b", "failed", "label", "wall_s") if k in r}),
                  flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=("pick", "place", "push-crate", "push-cabinet", "blocked", "carry-push", "script", "wrong",
                                     "carry-cover", "bumper"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--poses", choices=("perceived", "truth"), default="perceived")
    a = ap.parse_args()
    _set_poses(a.poses)
    out = a.out
    (out / a.gate).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res: dict[str, Any] = {"gate": a.gate, "mujoco": mujoco.__version__, "poses": a.poses}
    if a.gate == "pick":
        rows = _pool(pick_trial, [(i, s, b, out) for i, (s, b) in enumerate(pick_list())], a.jobs)
        n_a, n_b = sum(r["pass_a"] for r in rows), sum(r["pass_b"] for r in rows)
        res.update(bar="18/20", n=len(rows), surface_trials=sum(r["support"] != "floor" for r in rows),
                   pass_a=n_a, verdict_a=n_a >= 18, pass_b=n_b, verdict_b=n_b >= 18, trials=rows)
    elif a.gate == "place":
        from sim.world.generate import generate
        rows, i, s = [], 0, PLACE_SEED0
        while True:
            cnt = {sf: sum(r["counted"] for r in rows if r["surface"] == sf) for sf in (KT, SB)}
            if min(cnt.values()) >= N_PLACE:
                break
            batch = []
            for _ in range(2 * a.jobs):
                b = generate(seed=s, blocker="none").boxes[(s - PLACE_SEED0) % 4]
                batch.append((i, s, f"{b['colour']} box", KT if i % 2 == 0 else SB, out))
                i, s = i + 1, s + 1
            rows += _pool(place_trial, batch, a.jobs)
        gate = [r for sf in (KT, SB) for r in [x for x in rows if x["surface"] == sf and x["counted"]][:N_PLACE]]
        n = sum(r["pass"] for r in gate)
        res.update(bar="20/20", n=len(gate), passed=n, verdict=n == 2 * N_PLACE,
                   pick_failures_not_counted=[r["seed"] for r in rows if not r["counted"]],
                   gate_trials=[r["i"] for r in gate], trials=rows)
    elif a.gate == "push-crate":
        rows = _pool(push_crate_trial, [(s, out) for s in PUSH_SEEDS], a.jobs)
        n = sum(r["pass"] for r in rows)
        res.update(bar=">= 9/10", passed=n, n=len(rows), verdict=n >= 9,
                   doors={dr: sum(r["door"] == dr for r in rows) for dr in CRATE_GOAL}, trials=rows)
    elif a.gate == "push-cabinet":
        rows = _pool(push_cabinet_trial, [(s, out) for s in PUSH_SEEDS], a.jobs)
        n = sum(r["pass"] for r in rows)
        res.update(bar="10/10", passed=n, n=len(rows), verdict=n == len(rows),
                   max_move_m=max(r["cabinet_move_m"] for r in rows), max_pen_m=max(r["pen_max_whole_run_m"] for r in rows),
                   nav1_blocked=sum(r["nav1_blocked_ok"] for r in rows), distinct=distinct_trials(rows), trials=rows)
    elif a.gate == "carry-push":
        rows = _pool(carry_push_trial, [(s, out) for s in CARRY_PUSH_SEEDS], a.jobs)
        n = sum(r["pass"] for r in rows)
        res.update(bar=">= 9/10 pushed, box held (both pads) after the push, then reached", passed=n, n=len(rows),
                   verdict=n >= 9, doors={dr: sum(r["door"] == dr for r in rows) for dr in CRATE_GOAL},
                   picks_ok=sum(r["pick"].startswith("holding") for r in rows),
                   max_arm_drift_rad=max((r.get("arm_drift_rad") or 0.0) for r in rows), trials=rows)
    elif a.gate == "blocked":
        rows = _pool(blocked_trial, [(v, s, out) for v in ("crate", "cabinet") for s in BLOCKED_SEEDS], a.jobs)
        per = {}
        for v in ("crate", "cabinet"):
            vr = [r for r in rows if r["variant"] == v]
            per[v] = {"passed": sum(r["pass"] for r in vr), "n": len(vr),
                      "doors": {dr: sum(r["door"] == dr for r in vr) for dr in sorted({r["door"] for r in vr})},
                      "signals": {sg: sum(r["signal"] == sg for r in vr) for sg in sorted({str(r["signal"]) for r in vr})},
                      "max_detect_after_contact_s": max((r["detect_after_contact_s"] or 0.0) for r in vr),
                      "max_pen_m": max(r["max_pen_m"] for r in vr), "distinct": distinct_trials(vr)}
        res.update(bar="10/10 per variant", per_variant=per, verdict=all(p["passed"] == p["n"] for p in per.values()),
                   trials=rows)
    elif a.gate in ("script", "wrong"):
        from agent.scripts import WRONG
        from sim.tasks import dev_tasks
        if a.gate == "script":
            args = []
            for t in ("fetch", "blocked", "push_or_detour", "two_box"):
                for k in dev_tasks(t, 10):
                    for br in (("push_then_detour", "detour") if t == "push_or_detour" else ("-",)):
                        args.append(("script", t, k["seed"], br, out))
        else:
            args = [("wrong", "fetch", k["seed"], w, out) for k in dev_tasks("fetch", 3) for w in WRONG]
        rows = _pool(script_trial, args, a.jobs)
        per = {}
        for r in rows:
            key = r["type"] if a.gate == "script" and r["script"] == "-" else (
                f"{r['type']}:{r['script']}" if a.gate == "script" else r["script"])
            p = per.setdefault(key, {"passed": 0, "n": 0, "failing": []})
            p["n"] += 1
            p["passed"] += r["pass"]
            if a.gate == "script" and not r["pass"] or a.gate == "wrong":
                p["failing"].append({"seed": r["seed"], "failed": r["failed"], "label": r["label"],
                                     "fail_step": r["fail_step"]})
        if a.gate == "script":
            res.update(bar=">= 9/10 per type and per push_or_detour branch", per=per,
                       verdict=all(p["passed"] >= 9 for p in per.values()),
                       box_contacts_nav={f"{r['type']}_s{r['seed']}_{r['script']}": r["n_box_contacts_nav"] for r in rows})
        else:
            right = {"wrong_colour": {"surface", "wrong_box"}, "wrong_surface": {"surface"},
                     "box_in_gripper": {"held"}, "step_limit": {"steps"}}
            need = {"wrong_colour": {"surface"}, "wrong_surface": {"surface"}, "box_in_gripper": {"held"},
                    "step_limit": {"steps"}}
            for r in rows:
                r["right_rule"] = bool(not r["pass"] and need[r["script"]] <= set(r["failed"]))
                r["only_expected_rules"] = set(r["failed"]) <= right[r["script"]] | ({"surface"} if r["script"] in
                                                                                      ("box_in_gripper", "step_limit") else set())
            res.update(bar="100% scored fail with the right rule", per=per,
                       verdict=all(r["right_rule"] for r in rows),
                       rows_brief=[{k: r[k] for k in ("script", "seed", "pass", "failed", "right_rule",
                                                      "only_expected_rules", "label", "steps_used")} for r in rows])
        res["trials"] = rows
    elif a.gate == "carry-cover":
        res.update(carry_cover(out))
    elif a.gate == "bumper":
        res.update(bumper_scan(out))
    res["wall_s"] = time.time() - t0
    (out / a.gate / "gate.json").write_text(json.dumps(_j(res), indent=1) + "\n")
    print(json.dumps({k: v for k, v in _j(res).items() if k not in ("trials", "box_base_contacts_nav")},
                     indent=1)[:4000])


if __name__ == "__main__":
    main()
