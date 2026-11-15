"""E4 (task 014): tasks that need the VLM. Task generator for the four categories and the C3 mid-task move.

    make_task_e4(seed, cat, sealed_draw=False) -> task dict          (house = NAV116_HOUSE, as make_task)
    install_move(robot, task) -> log dict                             (C3 only; the runner calls it)

Categories (task "type" = "category"); every task: no blocker, one target, step limit 20, the scorer unchanged
(targets = the intended box, so a wrong box on the surface fails the wrong_box rule and the intended one is missing):
- vague (C1): the text names a use ("Bring something to drink to the kitchen table."); boxes carry printed labels
  (sim.world.generate.set_labels(True), the task carries "labels": True); exactly one box has the requested use;
  the target is that box. Built only if its gate passes (context row 24).
- place_ref (C2): the text names a place, not a colour ("Bring the box next to the sofa to the storage bench.").
  Anchors: the spawn surfaces (coffee table, nightstand: "on the ...") and sofa / bed / kitchen counter (a floor box
  whose footprint centre is <= NEAR_M from the anchor's footprint: "next to the sofa", "by the bed", "next to the
  kitchen counter"). Unique: every other box is >= d_target + 1.0 m from that anchor's footprint.
- recovery (C3): a fetch (make_task's fetch rule) whose target the harness moves once, right after the first
  successful locate of it (before any pick), to a floor spot in another room: the first of the task's "move"
  candidates (drawn at task time with the floor spawn rule) that is (a) not in the robot's current room, (b) out of
  view from the robot's current position (segmentation px < PX_OUT_OF_SIGHT at all 4 look headings, look arm), and
  (c) pickable (grasp.plan_pregrasp with the other boxes as avoid discs). The move is logged, never told to the agent.
- hint (C4): a fetch whose text adds the target's true room ("...; someone left it in the bedroom.").
Every target passes make_task's filter: out of sight from the hall spawn and a feasible pre-grasp (easings on).
Draw: numpy PCG64 seeded [seed, 14, category index]; move candidates [seed, 15]; same (seed, category, want) -> same
task. `want` (lead review 014 B2 / S2), stored in the task as "slot": recovery / hint pass it to make_task's fetch rule
(the balanced sampler's {"room", "support", "surface"} slot, sim.sampler.plan); place_ref takes {"relation": "on" |
"next"} (on = a spawn-surface anchor, next = a floor anchor) so the seal can cap the on-surface share.
"""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from sim import tasks as T
from sim.constants import BOX_HALF, PX_OUT_OF_SIGHT, STEP_LIMIT
from sim.world import generate as gen

CATS = ("vague", "place_ref", "recovery", "hint")
NEAR_M = 0.7
UNIQUE_M = 1.0
ANCHOR_TEXT = {"coffee table": "on the coffee table", "nightstand": "on the nightstand", "sofa": "next to the sofa",
               "bed": "by the bed", "kitchen_counter": "next to the kitchen counter"}
FLOOR_ANCHORS = ("sofa", "bed", "kitchen_counter")
REQUEST = {"drink": "something to drink", "wipe": "something to wipe a table with", "read": "something to read",
           "wash": "something to wash your hands with"}
MOVE_PER_ROOM = 6


def _base(seed: int, sealed_draw: bool) -> None:
    T.check_seed(seed, sealed_draw)
    if sealed_draw and seed in T.TEST_SEEDS:
        T._SEALED_OK.add(seed)


def _task(seed: int, cat: str, instruction: str, box: str, surface: str, px: dict[str, int], **extra: Any) -> dict:
    t = {"task_id": f"{cat}-{seed}", "type": cat, "category": cat, "seed": seed, "blocker": "none",
         "instruction": instruction, "targets": [{"object": box, "surface": surface}], "step_limit": STEP_LIMIT,
         "hall_px": {box: px[box]}, "e4": True} | extra
    return t if gen.HOUSE == "A" else t | {"house": gen.HOUSE}


def _eligible(seed: int) -> tuple[gen.World, dict[str, int], frozenset[str]]:
    w = T.world(seed, "none")
    px = T.hall_px(seed, "none")
    ok = T.pregrasp_ok(seed, "none")
    return w, px, frozenset(b for b in ok if px[b] < PX_OUT_OF_SIGHT)


def _place_ref(seed: int, rng: np.random.Generator, want: dict | None = None) -> dict:
    w, px, ok = _eligible(seed)
    furn = {f["id"]: f for f in w.layout["furniture"]}
    boxes = {T.box_name(b["colour"]): b for b in w.boxes}

    def dist(b: dict, anchor: str) -> float:
        if anchor in ("coffee table", "nightstand"):
            return 0.0 if b["surface"] == anchor else gen.rect_dist(b["x"], b["y"], gen.furniture_rect(
                next(f for f in w.layout["furniture"] if f.get("surface") == anchor)))
        return gen.rect_dist(b["x"], b["y"], gen.furniture_rect(furn[anchor]))

    for a in ("coffee table", "nightstand"):
        assert sum(f.get("surface") == a for f in w.layout["furniture"]) <= 1, a  # lead note 5: one furniture per surface
    anchors = [a for a in ("coffee table", "nightstand") if any(f.get("surface") == a for f in w.layout["furniture"])]
    anchors += [a for a in FLOOR_ANCHORS if a in furn]
    cand = []
    for name in sorted(ok):
        b = boxes[name]
        for a in anchors:
            on = a in ("coffee table", "nightstand")
            if on != (b["surface"] == a):
                continue
            d = dist(b, a)
            if not on and (b["surface"] != "floor" or d > NEAR_M):
                continue
            if all(dist(o, a) >= d + UNIQUE_M for n, o in boxes.items() if n != name):
                if want is None or want.get("relation") == ("on" if on else "next"):
                    cand.append((name, a))
    if not cand:
        raise T.NoTask(f"place_ref seed {seed}")
    box, anchor = cand[int(rng.integers(len(cand)))]
    surfaces = [s for s in w.layout["place_surfaces"] if s != boxes[box]["surface"]]
    surface = surfaces[int(rng.integers(len(surfaces)))]
    return _task(seed, "place_ref", f"Bring the box {ANCHOR_TEXT[anchor]} to the {surface}.", box, surface, px,
                 anchor=anchor, **({"slot": dict(want)} if want else {}))


def _vague(seed: int, rng: np.random.Generator) -> dict:
    assert getattr(gen, "LABELS", False), "vague tasks need set_labels(True) before the first world"
    w, px, ok = _eligible(seed)
    by_cat = {b["label"]["category"]: T.box_name(b["colour"]) for b in w.boxes}
    assert len(by_cat) == len(w.boxes), "one use category per box"
    cats = [c for c in sorted(by_cat) if by_cat[c] in ok]
    if not cats:
        raise T.NoTask(f"vague seed {seed}")
    want = cats[int(rng.integers(len(cats)))]
    box = by_cat[want]
    support = next(b["surface"] for b in w.boxes if T.box_name(b["colour"]) == box)
    surfaces = [s for s in w.layout["place_surfaces"] if s != support]
    surface = surfaces[int(rng.integers(len(surfaces)))]
    return _task(seed, "vague", f"Bring {REQUEST[want]} to the {surface}.", box, surface, px, labels=True, request=want)


def _move_candidates(seed: int, w: gen.World, box: str) -> list[list]:
    """Floor spots (generator's floor rule + route rule + MIN_BOX_SEP from the other boxes' start) in every other
    spawn room, rooms in a seeded order, MOVE_PER_ROOM per room."""
    rng = np.random.default_rng([seed, 15])
    room0 = next(b["room"] for b in w.boxes if T.box_name(b["colour"]) == box)
    others = [b for b in w.boxes if T.box_name(b["colour"]) != box]
    rooms = [r for r in w.layout["rooms"] if r in w.layout["spawn_regions"] and r != room0]
    rooms = [rooms[i] for i in rng.permutation(len(rooms))]
    segs = gen.route_segments(w.layout)
    places = gen.load_places()
    out = []
    for room in rooms:
        reg = w.layout["spawn_regions"][room]
        n = 0
        for _ in range(4000):
            x, y = float(rng.uniform(reg[0], reg[2])), float(rng.uniform(reg[1], reg[3]))
            if (gen.spot_ok(w.layout, room, x, y, places) and gen.route_clear(segs, x, y)
                    and all(math.hypot(x - o["x"], y - o["y"]) >= gen.MIN_BOX_SEP for o in others)):
                out.append([room, round(x, 6), round(y, 6), round(float(rng.uniform(0, math.pi)), 6)])
                n += 1
                if n == MOVE_PER_ROOM:
                    break
    return out


def make_task_e4(seed: int, cat: str, sealed_draw: bool = False, want: dict | None = None) -> dict:
    assert cat in CATS, cat
    _base(seed, sealed_draw)
    rng = np.random.default_rng([seed, 14, CATS.index(cat)])
    if cat == "place_ref":
        return _place_ref(seed, rng, want)
    if cat == "vague":
        return _vague(seed, rng)
    f = T.make_task(seed, "fetch", sealed_draw=sealed_draw, want=want)  # fetch rule (+ sampler slot); NoTask propagates
    slot = {"slot": dict(want)} if want else {}
    box, surface = f["targets"][0]["object"], f["targets"][0]["surface"]
    w = T.world(seed, "none")
    if cat == "hint":
        room = next(b["room"] for b in w.boxes if T.box_name(b["colour"]) == box).replace("_", " ")
        return _task(seed, cat, f"Bring the {box} to the {surface}; someone left it in the {room}.", box, surface,
                     f["hall_px"], hint_room=room, **slot)
    cands = _move_candidates(seed, w, box)
    if not cands:
        raise T.NoTask(f"recovery seed {seed}: no move spot")
    return _task(seed, cat, f"Bring the {box} to the {surface}.", box, surface, f["hall_px"], move=cands, **slot)


# ---- C3: the harness move ------------------------------------------------------------------------------------

def _view_px(robot: Any, body: str) -> int:
    """Max segmentation px of the body's geom over the 4 look headings from the robot's current base xy (look arm),
    on a copy of the state (the live state is not touched)."""
    from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, set_arm, set_yaw_direct
    from sim.robot import CAM_H, CAM_W, MAST_CAM
    m = robot.m
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, robot.d)
    set_arm(m, d, LOOK_ARM_QPOS)
    gid = m.geom(body).id
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    best = 0
    try:
        r.enable_segmentation_rendering()
        for h in HEADINGS_DEG:
            set_yaw_direct(m, d, math.radians(h))
            r.update_scene(d, camera=MAST_CAM)
            s = r.render()
            best = max(best, int(((s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (s[..., 0] == gid)).sum()))
    finally:
        r.close()
    return best


def _set_box(m: mujoco.MjModel, d: mujoco.MjData, body: str, x: float, y: float, yaw: float) -> None:
    j = m.body(body).jntadr[0]
    a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
    d.qpos[a:a + 7] = [x, y, BOX_HALF[2] + gen.SETTLE_DROP, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    d.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(m, d)


def install_move(robot: Any, task: dict) -> dict:
    """Wrap robot.locate: after the first successful locate of the target, move it (module doc). Returns the log
    dict the runner stores in result.json ("move")."""
    from sim import grasp
    box = task["targets"][0]["object"]
    body = robot.body_of[box]
    log: dict[str, Any] = {"moved": False, "tried": None}
    orig = robot.locate

    def locate(obj: str):
        res = orig(obj)
        if log["tried"] is not None or obj != box or res.pose is None:  # one attempt, at the first success
            return res
        m, d = robot.m, robot.d
        here = robot.robot_room()
        x0, y0 = (float(v) for v in d.body(body).xpos[:2])
        tried = []
        for room, x, y, yaw in task["move"]:
            if robot.room_name[room] == here:
                tried.append([room, "robot's room"])
                continue
            saved = d.qpos.copy(), d.qvel.copy()
            _set_box(m, d, body, x, y, yaw)
            px = _view_px(robot, body)
            avoid = [(float(d.body(robot.body_of[n]).xpos[0]), float(d.body(robot.body_of[n]).xpos[1]), gen.BOX_REACH)
                     for n, o in robot.objects().items() if o["class"] == "box" and n != box]
            pg = bool(grasp.plan_pregrasp(m, d, robot.grid, body, avoid)) if px < PX_OUT_OF_SIGHT else False
            if px < PX_OUT_OF_SIGHT and pg:
                others = [math.hypot(a[0] - x, a[1] - y) for a in avoid]
                log.update(moved=True, t=float(d.time), locate_obj=obj, robot_room=here, from_xy=[x0, y0],
                           to=[room, x, y, yaw], view_px=px, tried=tried,
                           min_box_dist_m=round(min(others), 3) if others else None)
                return res
            d.qpos[:], d.qvel[:] = saved
            mujoco.mj_forward(m, d)
            tried.append([room, f"px {px}" if px >= PX_OUT_OF_SIGHT else "no pre-grasp"])
        log.update(moved=False, failed="no candidate", tried=tried, t=float(d.time))
        return res

    robot.locate = locate
    return log
