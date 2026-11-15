"""Push a blocker out of a doorway with the force-limited base (task 003 `push`). No arm, no teleport, no weld.

    push_object(m, d, grid, body, door, cb=None) -> {"ok": bool, "reason": str, "detail": dict}

door: the adapter's shape {id, name, axis ("x"|"y": the wall normal), line (wall coordinate), center (coordinate along
the wall), width, between: [room, room], pose}; {name, centre: (x, y), axis, rooms} is accepted too.
Door frame: n = unit wall normal from the robot's side to the far side, t = n rotated +90 deg; `along` / `lat` are
coordinates from the door centre on n / t. n and t are world axes, so every footprint below is an axis-aligned square.
Strategy (all base moves through the base actuators; the arm stays in LOOK_ARM_QPOS, which go_to puts it in):
1. approach: `go_to` the stand-off pose on the robot's side, facing n, GAP from the object, centred on the object's
   lateral position clipped so the robot's sides stay CLIP_MARGIN inside the door opening: pushing an off-centre
   crate from the door axis turned it and let it drift sideways (003 blocked 1013: lat 0.19 -> 0.50 m, 0.27 m short).
2. plan (from the object's predicted path: it moves straight along n while pushed, measured lateral drift < 2 cm, and
   squares up against the bumper, so the plan uses its own half size, not its yawed extent: a navigate that stopped on
   the crate at an angle leaves it yawed 5-17 deg (G4-carry-push dev 1304 / 1306), and the through-push brings it back
   to <= 1.5 deg; the side manoeuvre then uses the extent measured after the through-push):
   "side" (sd = +-1, depth A, offset delta): through-push to along A, back off, turn on the door axis to face sd*t,
   side-step to the object's other side, close in to along A - delta, push the object to lat sd*SIDE_LAT (off the door
   -> room-point path), back off. Depths SIDE_PLANS: A 1.1 first; A 0.84 / delta 0.07 for the bedroom, where the bed
   leaves no room beside a crate pushed 1.1 m in; A 0.84 / delta 0.10 when carrying there (the held box's 3 cm bulge
   faces the bed on the sd +1 side-push: delta >= 0.082 m clears it, the wall behind keeps A >= 0.82 m).
   "long": push straight on to LONG_PAST beyond the far room point.
   Feasible = every object position on its path (5 cm samples, square half + OBJ_MARGIN) misses every static world
   box geom (walls, furniture: exact model geometry) and stays >= BOX_CLEAR + its half-diagonal from every fetch box
   centre (sim truth, Phase 2: a push must not shove a fetch box); the robot's route (footprint rect + ROBOT_MARGIN,
   + WIGGLE_M while pushing; circle radius + 1 cm where it turns) misses the static boxes and the object.
   Robot footprint (base-frame rect, radius): empty-handed the LOOK pose's (BASE_HALF square, ROBOT_RADIUS circle);
   holding a box the carry pose's, measured at the call from every robot geom point + the held box (the box sticks out
   3 cm on the base's -y side at z 0.35-0.47 m, crate height: y from -0.271 m). The rect is placed with the robot's
   heading on each leg (the route faces sd*t), so the bulge is checked on the side it is on; a symmetric 0.271 m
   square ruled out every side plan in 3 of 10 carry trials (dev run). The contact face stays BASE_HALF (the carry
   pose adds nothing in front of the bumper).
   Order: side A 1.1 (larger box clearance first), side A 0.84, long; none -> push anyway (movability probe), then
   "no room to clear".
3. return: `go_to` the stand-off pose (back through the doorway; detector off, the object's geoms ignored).
4. place-clearing push (task 005 row 17): when the object is outside every door corridor but its centre is within
   ROOM_PT_MIN of some place pose (at the call, or after steps 1-3: an off-centre long push let the crate slide 0.28 m
   sideways and park 0.23 m from the storage-bench approach, tuning blocked-8217), push it along the world axis u
   (+-x, +-y) by the smallest s (CLEAR_STEP .. CLEAR_MAX) whose end passes keepout_ok with every place counted and
   hw + OBJ_MARGIN, object path and robot sweep checked as in step 2 (`_clear_plan`). The robot `go_to`s a pre point
   behind the GAP stand-off (back PRE_BACK, across PRE_LAT: the first one with a free turning circle, a free straight
   leg in, and a planned path from the robot's pose: a crate parked in front of a door can seal its far side), facing
   u, on the map with the object's square marked (detector on), drives straight to the stand-off, pushes
   (`push_line`), backs off BACKOFF. No such push -> "no room to clear". Axis pushes only: the object squares up
   against the bumper (step 2), the bumper face (0.48 m) covers a crate face (0.45 m) whole.
5. stopped-path push (task 005 fix (d)): stop_path = the planned path (stop -> goal) of the last drive this object
   stopped (the adapter's last-stop memory). The object then counts as blocking whatever row 17's test says, and its
   end must also keep its footprint more than ROBOT_RADIUS + PATH_CLEAR from every segment of that path (`path_dist`).
   After steps 1-3 (if it was in a corridor), an object still near a place or on the path gets the step-4 push with
   that term added to its end test (square hw + OBJ_MARGIN, `_clear_plan(path=...)`). Before planning it, a base
   inside go_to's planner inflation of the object's square (ROBOT_RADIUS + CLEARANCE + DISC_MARGIN: a drive stopped
   against the object, and pick's drive has no back-off, tuning blocked-8208: 0.17 m from the crate square, no plan
   from there) first drives straight away from the square's nearest point to that distance + ROBOT_MARGIN, if that
   spot keeps the turning circle (rad + 1 cm) off every static box and rad + BOX_REACH from every fetch box not
   riding on the robot.
Room reachability (row 17 addendum 2, task 006): an object end (footprint: the actual yawed one for the verdict, the
planned square for a candidate) "keeps its room" if, with that footprint occupied on the known map, every place pose of
the room holding its centre (room look spots, surface approach poses, the door places of the room's doors) is reached
from that room's first door place by go_to's planner (plan_path, ROBOT_RADIUS, CLEARANCE + DISC_MARGIN), so all of them
are mutually reachable (`room_places_reachable`). Plan choice: side / long ends (step 2) and clearing-push ends (steps
4-5) must keep their room; after the door push an end that does not gets the clearing push like one by a place.
Not blocking (row 17): if at the call the object's footprint is outside every door corridor, its centre is
>= ROOM_PT_MIN from every place pose, it keeps its room's places reachable (row 17 addendum 3: an object that cuts a
room's places off is blocking and gets the clearing push), no stop_path is given and no map-blocked plan fails with
it (addendum 4), nothing is moved and the reason is "not blocking". All on the same-map footprints (below).
Same map (row 17 addendum 4, task 006): with `est` (the adapter's), every blocking / end / candidate test reads the
object as the drives' planner map has it, never the sim pose: est(detail) -> (x, y, yaw) = the adapter's map entry
(perceived: push's own contact geometry of the latest stroke that touched it, else its current entry; truth: the true
pose), None = not on the map and not touched yet (then the read pose, as before). Footprints at that pose: "own" = the
object's half size, "map" = half + map_margin (the adapter's map margin). The corridor (R4) and place-pose tests use
own (row 17 as ruled: the corridor is a rule on the object, not a planner cell; its effect on drives is the
reachability term); room reachability, the stopped path (step 5) and the map-blocked plan use map, on `grid` (the
adapter's map without this object's entry). Candidate ends (steps 2, 4-5) use their planned square + map_margin for
those three. detail["end_corridor_margined"] (report only) says whether map would also be in a corridor. Push's
physics (approach, stand-offs, strokes, moved, the fetch-box keep-off) still acts on the real object.
Map-blocked plan (row 17 addendum 4, analogue of fix (d)): need_plan(grid) -> bool is the remembered drive / pick plan
that failed on planning and succeeds with this object's map entry removed (adapter memory). If it fails with the
object's map footprint stamped, the object counts as blocking; after the door push (if in a corridor) it gets the
clearing push with need_plan on the candidate end added, and "pushed" also needs need_plan on the actual end's map
footprint ("blocks the failed plan" else, i.e. "stuck").
Step off the map (same map): with est, a push that acted ends with step 5's step-off from the map footprint's bounding
square, so the robot is outside the drives' planner inflation of the object's new entry (a clearing push backs off only
BACKOFF: 0.20 m from a 0.14 m-margined entry, inside the 0.375 m inflation, and the next drive had no path, G5 9225).
Last contact (row 17 addendum 3): detail["last_contact"] = {"base": the base's own (x, y, yaw) at the last step of the
latest push stroke (push_line) that touched the object, "u": that stroke's unit push direction, "off": the stroke's
PLANNED object-centre offset across u from the bumper centre (side push: n * delta; through / clearing push: 0)}; the
adapter derives
the object's end footprint from it in perceived mode (`contact_footprint`). Base pose = joint qpos (proprioception).
Progress check (every push): from the first robot-object contact, if after PROGRESS_S the object has moved less than
PROGRESS_M, release the setpoint at once (no base left pressing at the force limit), back off to the stand-off pose and
stop with "no progress"; a base force saturated SAT_MAX_S in a row (against anything, e.g. a fetch box in the way) stops
the push the same way with "saturated" (normal crate pushing saturates <= 0.21 s in a row, runs/2c).
Result (row 17): ok ("pushed") only from the object's ACTUAL end pose: its yawed footprint's bounding square outside
every door's R4 corridor (door +- 0.6 m along the passage, door width across), its centre >= ROOM_PT_MIN from every
place pose (door places included), with a stop_path its yawed footprint > ROBOT_RADIUS + PATH_CLEAR off it, its room
kept (above), moved >= PUSHED_MIN_M from the call, and the robot not stranded ("unreachable");
a plan step that stalled does not matter if the end passes. Otherwise internal why = the step's reason, or "still in
doorway" / "near a place" / "on the stopped path" / "cuts off a place" / "not moved".
Result reason (task 005 rows 10 and 17; the internal reason is detail["why"]): "" when ok; "not blocking" (above);
"not movable" only when the object was pushed against and never moved (a push step stopped on "no progress" or
"saturated" with the object < PROGRESS_M from its position at the call: the cabinet); every other failure (no feasible
plan / approach so nothing was pushed, moved then stalled, no room to clear, still in doorway, near a place, cuts off a
place, blocks the failed plan) is
"stuck" (row 10; ruling 2026-11-13: an object with no feasible plan is not known to be immovable).

Why the wiggle (runs/2c/wedge.json): the base's only contact geom at crate height is the bumper slab (z 0.027-0.082 m,
half-width 0.241 m). Pushing a crate at 4 cm height, the static friction on the crate face (MuJoCo elliptic cone with
the house's impratio 10) takes a large DOWNWARD share: measured floor normal 223 N vs crate weight 39 N, so the push
needs ~212 N and the crate creeps at ~0.03 m/s at the 212 N per-axis cap whatever the command speed. With impratio 1
the lock is gone (diagnostic only; the house option is not this track's). A +-2 cm lateral wiggle at 1 Hz keeps the face
contact sliding sideways, the friction cone is spent on that, and the crate follows the base (0.66 m of 0.66 m).
A diagonal push (robot turned 50 deg, corner first) was tried and dropped: the corner slid past the crate (0.17 m of
1.0 m moved, crate left on the door axis).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import mujoco
import numpy as np

from sim.body import StepCb, _ids, base_pose, drive_base, run
from sim.constants import ROBOT_RADIUS
from sim.look import geom_points, robot_geoms
from sim.nav.astar import plan_path
from sim.nav.follow import CLEARANCE, DISC_MARGIN, go_to
from sim.robot import BASE_JOINTS
from sim.world.generate import BOX_REACH, ROOM_PT_MIN, load_layout, load_places, rect_dist, room_point
from sim.world.occupancy import Grid

BASE_HALF = 0.241  # bumper slab half-width (Menagerie base mesh, z 0.027-0.082 m), the face that touches the object
GAP = 0.05  # stand-off gap, bumper face to object face
PUSH_V = 0.10  # m/s setpoint speed while pushing (min-jerk drives elsewhere keep the 0.5 m/s cap)
WIGGLE_M, WIGGLE_HZ = 0.02, 1.0  # lateral setpoint wiggle while pushing (see module doc)
PROGRESS_S, PROGRESS_M = 1.5, 0.03  # not movable: < 3 cm object motion 1.5 s after the first contact
SAT_MAX_S = 2.0  # stuck: base force saturated this long in a row (crate pushing: <= 0.21 s; > PROGRESS_S so a
# pressed, unmoved object is reported "not movable" first)
SIDE_PLANS = ((1.1, 0.0), (0.84, 0.07), (0.84, 0.10))  # (through depth A, close-in offset delta) for the side plans, in order
SIDE_LAT = 0.75  # object centre this far off the door axis after the side-push
LONG_PAST = ROOM_PT_MIN + 0.05  # long push: object centre this far past the far room point along n (robot turns
# there); 0.70 until 2j fix 4, whose 0.80 m place keep-out then rejected every long plan
BOX_CLEAR = 0.15  # fetch box centre >= object half-diagonal + this from the object path (box circumradius 0.104)
OBJ_MARGIN, ROBOT_MARGIN = 0.03, 0.01  # square inflation; robot xy extent in LOOK pose = the 0.2407 m bumper square
# (all robot geom points, arm included: runs/2c/robot_extent.json), ROBOT_MARGIN = servo error allowance
CLIP_MARGIN = 0.03  # stand-off lat: robot side >= this inside the door opening
BACKOFF = 0.10
CORRIDOR_ALONG = 0.6  # R4: door +- 0.6 m along the passage
DOOR_HALF_W = 0.5  # layout.yaml door_width 1.0 m: the corridor's lateral half-width when the door has no width
SETTLE = 0.3
PUSHED_MIN_M = 0.01  # row 17: a push that moved the object less than this never says "pushed"
CLEAR_STEP, CLEAR_MAX = 0.05, 1.5  # place-clearing push (module doc, step 4): distances tried along each axis
# place-clearing push: go_to's goal (pre) is the stand-off moved back by a PRE_BACK and across by a PRE_LAT (first free
# one, _clear_plan): the planner's inflation (ROBOT_RADIUS + CLEARANCE + DISC_MARGIN = 0.375 m) rules out a stand-off
# 0.29 m from the crate face; a crate parked 0.67 m from the storage door jamb (tuning blocked-8217) needs a pre point
# back and to the side, then a straight drive in
PRE_BACK = (0.02, 0.10, 0.20, 0.30)
PRE_LAT = (0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20, 0.30, -0.30, 0.40, -0.40)
PATH_CLEAR = 0.05  # fix (d): object footprint > ROBOT_RADIUS + this from every segment of the stopped drive's path


def _yaw(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def _footprint(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> np.ndarray:
    """xy corners (4 x 2, world) of the body's box geom (crate and cabinet are one box each), yaw-only footprint."""
    g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id)
    R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
    return np.array([(c + R @ (h * [sx, sy, 0.0]))[:2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def footprint(m: mujoco.MjModel, d: mujoco.MjData, held_geoms: tuple[int, ...] = ()) -> tuple[np.ndarray, float]:
    """(base-frame rect [xmin, xmax, ymin, ymax], circle radius) of the robot: LOOK pose (BASE_HALF square,
    ROBOT_RADIUS) empty-handed, else measured from every robot geom point + the held box (carry pose, module doc),
    never smaller than the LOOK pose's."""
    sq = np.array([-BASE_HALF, BASE_HALF, -BASE_HALF, BASE_HALF])
    if not held_geoms:
        return sq, ROBOT_RADIUS
    p = geom_points(m, d, np.append(robot_geoms(m), held_geoms).astype(int))[:, :2] - d.body("base_link").xpos[:2]
    b = p @ d.body("base_link").xmat.reshape(3, 3)[:2, :2]  # base-frame xy (base yaw only)
    ext = np.array([min(sq[0], b[:, 0].min()), max(sq[1], b[:, 0].max()), min(sq[2], b[:, 1].min()),
                    max(sq[3], b[:, 1].max())])
    return ext, max(ROBOT_RADIUS, float(np.hypot(b[:, 0], b[:, 1]).max()))


def _robot_free(c: np.ndarray, ext: np.ndarray, u: np.ndarray, margin: float,
                rects: list[tuple[float, float, float, float]]) -> bool:
    """The footprint rect ext (base frame), centred at c with heading u (an axis), + margin, misses every rect."""
    w = np.array([-u[1], u[0]])
    pts = np.array([c + bx * u + by * w for bx in ext[:2] for by in ext[2:]])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    return all(hi[0] <= r[0] or lo[0] >= r[2] or hi[1] <= r[1] or lo[1] >= r[3] for r in rects)


def _tilt_deg(d: mujoco.MjData, body: str) -> float:
    return math.degrees(math.acos(float(np.clip(d.body(body).xmat[8], -1.0, 1.0))))


def _static_rects(m: mujoco.MjModel, d: mujoco.MjData, skip_body: int) -> list[tuple[float, float, float, float]]:
    """xy AABBs of every colliding box geom on a joint-less world body (walls, furniture, a welded cabinet), except
    skip_body and the robot."""
    root = m.body("base_link").id
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if (m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or b in (0, skip_body) or m.body_rootid[b] == root
                or m.body_jntnum[b] or not (m.geom_contype[g] or m.geom_conaffinity[g])):
            continue
        h = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
        c = d.geom_xpos[g]
        out.append((c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1]))
    return out


def _sq_free(c: np.ndarray, h: float, rects: list[tuple[float, float, float, float]]) -> bool:
    return all(c[0] + h <= r[0] or c[0] - h >= r[2] or c[1] + h <= r[1] or c[1] - h >= r[3] for r in rects)


def _samples(p0: np.ndarray, p1: np.ndarray) -> list[np.ndarray]:
    return [p0 + (p1 - p0) * f for f in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(p1 - p0) / 0.05) + 1))]


def _rect_clear(pts: list[tuple[float, float]], half_along: float, half_lat: float) -> bool:
    """True if the convex quad with these (along, lat) vertices, in order, misses the rect |along| <= half_along,
    |lat| <= half_lat (separating-axis test on the rect axes and the quad's edge normals)."""
    P = np.array(pts)
    R = np.array([[-half_along, -half_lat], [half_along, -half_lat], [half_along, half_lat], [-half_along, half_lat]])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    axes += [np.array([-(P[(i + 1) % 4] - P[i])[1], (P[(i + 1) % 4] - P[i])[0]]) for i in range(4)]
    return any((P @ ax).max() < (R @ ax).min() or (R @ ax).max() < (P @ ax).min() for ax in axes)


def push_line(m: mujoco.MjModel, d: mujoco.MjData, body: str, u: np.ndarray, dist: float,
              cb: StepCb = None) -> dict[str, Any]:
    """Move the base setpoint dist m along unit u at PUSH_V with the lateral wiggle; yaw setpoint held. Ends on the
    nominal (wiggle-free) setpoint. Stops early (stalled) on the progress check and releases the setpoint."""
    qa, _, act = _ids(m, BASE_JOINTS)
    rg = robot_geoms(m)
    og = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id])
    w = np.array([-u[1], u[0]])
    c0 = d.ctrl[act[:2]].copy()
    p_obj0 = d.body(body).xpos[:2].copy()
    dt = m.opt.timestep
    n = round(dist / PUSH_V / dt)
    out: dict[str, Any] = {"stalled": False, "why": None, "contact_t0": None, "fmax_n": 0.0, "pen_max_m": 0.0, "sat_steps": 0,
                           "sat_run_max_s": 0.0, "contact_base": None, "u": [float(u[0]), float(u[1])]}
    p_c, run_n = None, 0
    for i in range(1, n + 1):
        t = i * dt
        d.ctrl[act[:2]] = c0 + u * PUSH_V * t + w * WIGGLE_M * math.sin(2 * math.pi * WIGGLE_HZ * t)
        mujoco.mj_step(m, d)
        if cb is not None:
            cb(d)
        f = d.actuator_force[act[:2]]
        out["fmax_n"] = max(out["fmax_n"], float(math.hypot(f[0], f[1])))
        sat = bool(np.any(np.abs(f) >= m.actuator_forcerange[act[:2], 1] - 1e-6))
        out["sat_steps"] += int(sat)
        run_n = run_n + 1 if sat else 0
        out["sat_run_max_s"] = max(out["sat_run_max_s"], run_n * dt)
        g = d.contact.geom[: d.ncon]
        hit = (np.isin(g[:, 0], rg) & np.isin(g[:, 1], og)) | (np.isin(g[:, 1], rg) & np.isin(g[:, 0], og))
        if hit.any():
            out["pen_max_m"] = max(out["pen_max_m"], float(-d.contact.dist[: d.ncon][hit].min()))
            out["contact_base"] = [*map(float, base_pose(m, d))]  # the base's own pose at the last bumper contact
            if p_c is None:
                p_c, out["contact_t0"] = d.body(body).xpos[:2].copy(), t
        no_prog = p_c is not None and t - out["contact_t0"] >= PROGRESS_S \
            and np.linalg.norm(d.body(body).xpos[:2] - p_c) < PROGRESS_M
        if no_prog or run_n * dt >= SAT_MAX_S:  # saturated that long against anything (e.g. a fetch box): stop too
            out["stalled"], out["why"] = True, "no progress" if no_prog else "saturated"
            d.ctrl[act[:2]] = d.qpos[qa[:2]]  # release: setpoint = where the base is, force drops at once
            run(m, d, SETTLE, cb)
            break
    if not out["stalled"]:
        x, y = c0 + u * dist
        drive_base(m, d, (float(x), float(y), float(d.ctrl[act[2]])), settle=SETTLE, cb=cb)
    out["obj_disp_m"] = float(np.linalg.norm(d.body(body).xpos[:2] - p_obj0))
    out["sim_s"] = n * dt if not out["stalled"] else None
    return out


def push_object(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, body: str, door: dict[str, Any],
                cb: StepCb = None, held_geoms: tuple[int, ...] = (), stop_path: np.ndarray | None = None,
                est: Callable[[dict[str, Any]], tuple[float, float, float] | None] | None = None, map_margin: float = 0.0,
                need_plan: Callable[[Grid], bool] | None = None) -> dict[str, Any]:
    """Push `body` out of `door` and off the door -> far-room path, then return to the stand-off pose on the robot's
    side; or, outside every corridor, away from the place pose it is too close to, or off stop_path (the path of the
    last drive it stopped, step 5). See the module doc. held_geoms:
    geoms of a box held in the carry pose; the arm then stays where it is (no move to LOOK_ARM_QPOS with a box in the
    fingers, 2i) and those geoms are ignored like the object's. est / map_margin / need_plan: module doc, same map and
    map-blocked plan."""
    t_start = d.time
    k = 0 if door["axis"] == "x" else 1
    if "centre" in door:
        C = np.asarray(door["centre"], dtype=float)
    else:  # adapter shape: line = wall coordinate, center = coordinate along the wall
        C = np.array([door["line"], door["center"]] if k == 0 else [door["center"], door["line"]], dtype=float)
    rooms = door.get("rooms") or door["between"]
    half_w = float(door.get("width", 2 * DOOR_HALF_W)) / 2.0
    og = tuple(int(g) for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id) + tuple(held_geoms)
    look = not held_geoms
    q0 = base_pose(m, d)
    nv = np.zeros(2)
    nv[k] = 1.0 if C[k] > q0[k] else -1.0
    tv = np.array([-nv[1], nv[0]])
    yaw_n = _yaw(nv)

    def frame(p: np.ndarray) -> tuple[float, float]:
        return float((p - C) @ nv), float((p - C) @ tv)

    def world(a: float, l: float) -> np.ndarray:
        return C + nv * a + tv * l

    corners = _footprint(m, d, body)
    a_o, l_o = frame(d.body(body).xpos[:2])
    near = min(frame(c)[0] for c in corners)  # object's near face along n
    hw = max(abs(frame(c)[1] - l_o) for c in corners)  # object half extent across (~ along: crate and cabinet faces)
    g_o = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id)
    hw_p = float(m.geom_size[g_o][:2].max())  # plan: the object squared up against the bumper (module doc, step 2)
    half = m.geom_size[g_o][:2]
    ext, rad = footprint(m, d, tuple(held_geoms))
    # approach / through-push lat, facing n (base +y = t): sides CLIP_MARGIN inside the opening
    l_a = float(np.clip(l_o, -half_w + CLIP_MARGIN - ext[2], half_w - CLIP_MARGIN - ext[3]))
    detail: dict[str, Any] = {"door": door["name"], "obj_start": [*map(float, d.body(body).xpos[:2])],
                              "obj_along0": a_o, "obj_lat0": l_o, "robot_ext_m": ext.tolist(), "robot_radius_m": rad,
                              "approach_lat": l_a, "pen_max_m": 0.0, "fmax_n": 0.0}
    p_start = d.body(body).xpos[:2].copy()
    standoff = world(near - BASE_HALF - GAP, l_a)
    sp = (float(standoff[0]), float(standoff[1]), yaw_n)

    def log(stage: str, res: dict[str, Any], off: np.ndarray = np.zeros(2)) -> None:
        detail[stage] = {k_: res[k_] for k_ in ("stalled", "why", "obj_disp_m", "fmax_n", "pen_max_m", "sat_steps", "sat_run_max_s")}
        detail["pen_max_m"] = max(detail["pen_max_m"], res["pen_max_m"])
        detail["fmax_n"] = max(detail["fmax_n"], res["fmax_n"])
        if res.get("contact_base") is not None:  # the latest push stroke that touched the object (map footprint)
            detail["last_contact"] = {"base": res["contact_base"], "u": res["u"], "off": [*map(float, off)]}

    def drive(a: float, l: float, yaw: float) -> None:
        p = world(a, l)
        drive_base(m, d, (float(p[0]), float(p[1]), yaw), settle=SETTLE, cb=cb)

    def back_home(ok: bool, reason: str) -> tuple[bool, str]:
        r_ = go_to(m, d, grid, sp, cb=cb, detect=False, ignore_geoms=og, arm_look=look)
        detail["return_reached"] = r_["reached"]
        detail["return_contacts"] = {k_: v for k_, v in r_["contacts"].items() if body in k_}
        if ok and not r_["reached"]:
            ok, reason = False, "unreachable"
        return ok, reason

    def door_push() -> tuple[bool, str]:
        r = go_to(m, d, grid, sp, cb=cb, detect=False, ignore_geoms=og, arm_look=look)
        detail["approach_reached"] = r["reached"]
        if not r["reached"]:
            return False, "unreachable"
        # plan, then the through-push
        far = next(r_ for r_ in rooms if frame(np.array(room_point(r_)))[0] > 0.0)
        a_long = frame(np.array(room_point(far)))[0] + LONG_PAST
        plan = _choose_plan(m, d, grid, body, world, l_o, hw_p, a_long, door.get("id"), l_a, ext, rad, map_margin)
        detail["plan"] = plan
        a_thr = plan[2] if plan[0] == "side" else (a_long if plan[0] == "long" else SIDE_PLANS[0][0])
        res = push_line(m, d, body, nv, GAP + max(0.0, a_thr - a_o) + 0.02, cb)
        log("through", res)
        if res["stalled"]:
            drive_base(m, d, sp, settle=SETTLE, cb=cb)
            return False, res["why"]
        if plan[0] == "none":
            return back_home(False, "no room to clear")
        a_c, l_c = frame(d.body(body).xpos[:2])
        a_r, l_r = frame(base_pose(m, d)[:2])
        hw_c = max(abs(frame(c)[1] - l_c) for c in _footprint(m, d, body))  # after the through-push
        if plan[0] == "long":
            drive(a_r - BACKOFF, l_r, yaw_n)
            return back_home(True, "")
        _, sd, _, delta = plan
        yaw_u = _yaw(tv * sd)
        a_t = a_c - hw_c - rad - 0.01  # turn here: footprint circle clears the object face
        l_s = l_c - sd * (hw_c + BASE_HALF + GAP)  # robot centre lat on the object's other side
        drive(a_t, l_r, yaw_n)  # back off
        drive(a_t, l_r, yaw_u)  # turn on the door axis
        drive(a_t, l_s, yaw_u)  # side-step (backwards)
        drive(a_c - delta, l_s, yaw_u)  # close in beside the object
        res = push_line(m, d, body, tv * sd, GAP + max(0.0, SIDE_LAT - sd * l_c) + 0.02, cb)
        log("side", res, nv * delta)  # planned: the bumper centre delta short of the object centre along n
        q = base_pose(m, d)[:2] - tv * sd * BACKOFF
        drive_base(m, d, (float(q[0]), float(q[1]), yaw_u), settle=SETTLE, cb=cb)
        return back_home(not res["stalled"], res["why"] or "")

    def step_off(c: np.ndarray, h: float, key: str = "step_off_m") -> None:
        """Module doc, step 5: out of the planner's inflation of the object's square, straight away from it."""
        q = base_pose(m, d)
        near_pt = np.clip(q[:2], c - h, c + h)
        gap = float(np.linalg.norm(q[:2] - near_pt))
        need = ROBOT_RADIUS + CLEARANCE + DISC_MARGIN + ROBOT_MARGIN - gap
        if need <= 0.0 or gap < 1e-6:
            return
        p = q[:2] + (q[:2] - near_pt) / gap * need
        boxes = [b_ for b_ in _box_xy(m, d) if np.linalg.norm(b_ - q[:2]) > rad]  # a held box rides on the robot
        if all(rect_dist(float(p[0]), float(p[1]), r_) >= rad + 0.01 for r_ in _static_rects(m, d, m.body(body).id)) \
                and all(np.linalg.norm(b_ - p) >= rad + BOX_REACH for b_ in boxes):
            drive_base(m, d, (float(p[0]), float(p[1]), float(q[2])), settle=SETTLE, cb=cb)
            detail[key] = need

    def clear_place() -> tuple[bool, str]:
        """Outside every corridor but too close to a place pose: the shortest feasible axis push (_clear_plan) that
        takes the object to a keep-out-passing spot. go_to its pre point (on the map with the object's square marked;
        detector on), straight to the stand-off, push, back off BACKOFF along -u."""
        c, h = _aabb(m, d, body)
        if stop_path is not None:
            step_off(c, h)
        cp = _clear_plan(m, d, grid, body, c, h, ext, rad, stop_path, map_margin, need_plan)
        detail["clear_plan"] = None if cp is None else {"u": cp[0].tolist(), "s": cp[1], "pre": cp[2].tolist()}
        if cp is None:
            return False, "no room to clear"
        u, s_, pre, stand = cp
        yaw_u = _yaw(u)
        r = go_to(m, d, clear_grid(grid, c, h), (float(pre[0]), float(pre[1]), yaw_u), cb=cb,
                  ignore_geoms=tuple(held_geoms), arm_look=look)
        detail["clear_approach"] = {k_: r.get(k_) for k_ in ("reached", "blocked", "pos_err_m", "yaw_err_deg", "path_len_m",
                                                              "contacts")}
        if not r["reached"]:
            return False, "unreachable"
        drive_base(m, d, (float(stand[0]), float(stand[1]), yaw_u), settle=SETTLE, cb=cb)
        res = push_line(m, d, body, u, GAP + s_ + 0.02, cb)
        log("clear", res)
        q = base_pose(m, d)[:2] - u * BACKOFF
        drive_base(m, d, (float(q[0]), float(q[1]), yaw_u), settle=SETTLE, cb=cb)
        return not res["stalled"], res["why"] or ""

    def mapped() -> tuple[np.ndarray, np.ndarray]:
        """(own, map) footprints of the object as the planner map has it now (module doc, same map)."""
        e = est(detail) if est is not None else None
        if e is None:
            f = _footprint(m, d, body)
            return f, f
        return _rect(*e, *half), _rect(*e, *(half + map_margin))

    def own_sq() -> tuple[np.ndarray, float]:
        f = mapped()[0]
        c = f.mean(0)
        return c, float(np.abs(f - c).max())

    def on_path() -> bool:
        return stop_path is not None and path_dist(mapped()[1], stop_path) <= ROBOT_RADIUS + PATH_CLEAR

    def cuts() -> bool:
        return not room_places_reachable(grid, mapped()[1])

    def plan_blocked() -> bool:
        return need_plan is not None and not need_plan(stamp_poly(grid, mapped()[1]))

    in_corr, near_place = _blocks(*own_sq())
    cut0, pb0 = cuts(), plan_blocked()
    detail.update(start_in_corridor=in_corr, start_near_place=near_place, stopped_last_drive=stop_path is not None,
                  start_on_path=on_path(), start_cuts_place=cut0, map_blocked=need_plan is not None,
                  start_blocks_plan=pb0)
    ok, reason = True, ""
    if not (in_corr or near_place or cut0 or pb0 or stop_path is not None):
        ok, reason = False, "not blocking"
    else:
        if in_corr:
            ok, reason = door_push()
            detail["door_result"] = [ok, reason]
        in_corr, near_place = _blocks(*own_sq())
        if not in_corr and (near_place or on_path() or cuts() or plan_blocked()):  # out of the doorway but by a place
            # pose / on the stopped path / cutting its room's places off / still stopping the failed plan
            ok, reason = clear_place()

    if est is not None and reason != "not blocking":  # module doc, step off the map
        f_m = mapped()[1]
        step_off(f_m.mean(0), float(np.abs(f_m - f_m.mean(0)).max()), "end_step_off_m")

    # finish: "pushed" only on the ACTUAL end pose (outside every corridor, >= ROOM_PT_MIN from every place pose) and a
    # move >= PUSHED_MIN_M (row 17); the internal reason stays in detail["why"]
    a, l = frame(d.body(body).xpos[:2])
    own, mp = mapped()
    clear = _rect_clear([frame(c) for c in own], CORRIDOR_ALONG, half_w)
    in_corr, near_place = _blocks(*own_sq())
    path_hit = on_path()
    cut = cuts()
    pblk = plan_blocked()
    moved = float(np.linalg.norm(d.body(body).xpos[:2] - p_start))
    if stop_path is not None:
        detail["end_path_dist_m"] = path_dist(mp, stop_path)
    c_m = mp.mean(0)
    detail.update(end_blocks_plan=pblk, end_corridor_margined=not _corridors_free(c_m, float(np.abs(mp - c_m).max())),
                  end_on_path=path_hit, end_cuts_place=cut, obj_end=[*map(float, d.body(body).xpos[:3])], obj_along=a, obj_lat=l,
                  tilt_deg=_tilt_deg(d, body), sim_s=float(d.time - t_start), corridor_clear=clear,
                  end_in_corridor=in_corr, end_near_place=near_place, obj_moved_m=moved,
                  sim_ok=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))
    if reason != "not blocking":  # the actual end decides, whatever the plan said; a robot that cannot get back fails
        ok = reason != "unreachable" and not (in_corr or near_place or path_hit or cut or pblk) and moved >= PUSHED_MIN_M
        if not ok and reason == "":
            reason = ("still in doorway" if in_corr else "near a place" if near_place else
                      "on the stopped path" if path_hit else "cuts off a place" if cut else
                      "blocks the failed plan" if pblk else "not moved")
    detail["why"] = reason
    if not ok and reason != "not blocking":  # VLM-facing label (module doc): pushed against, never moved -> "not movable"
        reason = "not movable" if moved < PROGRESS_M and reason in ("no progress", "saturated") else "stuck"
    return {"ok": ok, "reason": "" if ok else reason, "detail": detail}


def contact_footprint(last_contact: dict[str, Any], half: tuple[float, float]) -> tuple[float, float, float]:
    """(x, y, yaw) of an object of xy half size `half` squared up against the bumper face (BASE_HALF ahead of the base
    centre along the push direction u) at the last contact base pose: centre = base + u (BASE_HALF + half[0]) + the
    stroke's planned offset across u ("off"), yaw = the direction of u (the object squares up against the bumper,
    module doc step 2). Unplanned slip along the face and the wiggle are not known here: the caller's margin covers
    them (sim.adapter PUSHED_MAP_MARGIN)."""
    b, u = np.asarray(last_contact["base"][:2], float), np.asarray(last_contact["u"], float)
    c = b + u * (BASE_HALF + float(half[0])) + np.asarray(last_contact["off"], float)
    return float(c[0]), float(c[1]), _yaw(u)


def _aabb(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> tuple[np.ndarray, float]:
    """(centre xy, half side) of the square that holds the body's actual (yawed) footprint."""
    f = _footprint(m, d, body)
    c = d.body(body).xpos[:2].copy()
    return c, float(np.abs(f - c).max())


def _blocks(c: np.ndarray, hw: float) -> tuple[bool, bool]:
    """(square footprint in some door's R4 corridor, centre < ROOM_PT_MIN from some place pose), every door and place
    counted (keepout_ok with no door excused)."""
    return not _corridors_free(c, hw), not _places_free(c, None)


def keepout_ok(c: np.ndarray, hw: float, door_id: str | None) -> bool:
    """Object centre c (xy), half extent hw: >= ROOM_PT_MIN from every place pose but door_id's own door place, and its
    square footprint outside every door's R4 corridor (module doc, step 2)."""
    return _places_free(c, door_id) and _corridors_free(c, hw)


def _places_free(c: np.ndarray, door_id: str | None) -> bool:
    return all(math.hypot(c[0] - p["pose"][0], c[1] - p["pose"][1]) >= ROOM_PT_MIN for p in load_places()
               if not (p["kind"] == "door" and p["door"] == door_id))


def _corridors_free(c: np.ndarray, hw: float) -> bool:
    lay = load_layout()
    w = lay["door_width"] / 2.0
    for dd in lay["doors"].values():
        (axis, line), = dd["wall"].items()
        ha, hl = (CORRIDOR_ALONG, w) if axis == "x" else (w, CORRIDOR_ALONG)
        cx, cy = (line, dd["center"]) if axis == "x" else (dd["center"], line)
        if abs(c[0] - cx) < ha + hw and abs(c[1] - cy) < hl + hw:
            return False
    return True


def path_dist(poly: np.ndarray, path: np.ndarray) -> float:
    """Min distance between the convex polygon poly (k x 2, in order) and the polyline path (n x 2); 0 if they meet."""
    def seg_pts(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        return float(np.hypot(*(a + t[:, None] * ab - p).T).min())

    path = path if len(path) > 1 else np.vstack([path, path])
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    best = math.inf
    for a, b in zip(path[:-1], path[1:]):
        seg = np.array([a, b])
        axes = [np.array([-(q - p)[1], (q - p)[0]]) for p, q in edges] + [np.array([-(b - a)[1], (b - a)[0]])]
        if all(not ((poly @ ax).max() < (seg @ ax).min() or (seg @ ax).max() < (poly @ ax).min()) for ax in axes
               if float(ax @ ax) > 1e-18):
            return 0.0
        best = min(best, seg_pts(poly, a, b), *(seg_pts(seg, p, q) for p, q in edges))
    return best


def _sq(c: np.ndarray, h: float) -> np.ndarray:
    return np.array([c + [sx * h, sy * h] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _rect(x: float, y: float, yaw: float, hx: float, hy: float) -> np.ndarray:
    """Corners (4 x 2, counter-clockwise as _footprint) of the rectangle centred (x, y), yaw, half sides hx, hy."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(x + c * sx * hx - s * sy * hy, y + s * sx * hx + c * sy * hy)
                     for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def stamp_poly(grid: Grid, poly: np.ndarray) -> Grid:
    """The grid with every cell whose centre is inside the convex counter-clockwise poly occupied."""
    ny, nx = grid.occ.shape
    X, Y = np.meshgrid(grid.origin[0] + (np.arange(nx) + 0.5) * grid.res, grid.origin[1] + (np.arange(ny) + 0.5) * grid.res)
    inside = np.ones_like(grid.occ)
    for a, b in zip(poly, np.roll(poly, -1, 0)):  # left of every edge
        inside &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0.0
    return Grid(grid.origin, grid.res, grid.occ | inside)


def room_places_reachable(grid: Grid, poly: np.ndarray) -> bool:
    """Module doc, room reachability: poly = convex footprint (k x 2, counter-clockwise as _footprint / _sq give it).
    False if its centre is in no room."""
    lay = load_layout()
    c = poly.mean(0)
    room = next((k for k, r in lay["rooms"].items()
                 if r["rect"][0] <= c[0] < r["rect"][2] and r["rect"][1] <= c[1] < r["rect"][3]), None)
    if room is None:
        return False
    places = load_places()
    pts = [p["pose"][:2] for p in places if p["kind"] == "door" and room in lay["doors"][p["door"]]["between"]]
    pts += [p["pose"][:2] for p in places if p["kind"] != "door" and p["room"] == room]
    occ = stamp_poly(grid, poly).occ
    s = (float(pts[0][0]), float(pts[0][1]))
    return all(plan_path(occ, grid.origin, grid.res, s, (float(p[0]), float(p[1])), ROBOT_RADIUS, CLEARANCE + DISC_MARGIN)
               is not None for p in pts[1:])


def _clear_plan(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, body: str, c: np.ndarray, hw: float,
                ext: np.ndarray, rad: float, path: np.ndarray | None = None, map_margin: float = 0.0,
                need: Callable[[Grid], bool] | None = None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
    """(u, s, pre, stand): push along world axis u by s (CLEAR_STEP steps up to CLEAR_MAX), the smallest s over the
    four axes whose end passes keepout_ok with hw + OBJ_MARGIN (every place counted), with `path` (step 5) keeps the
    square hw + OBJ_MARGIN more than ROBOT_RADIUS + PATH_CLEAR off it, whose square hw keeps its room (module doc), and
    whose object path is free (_obj_clear); the end's square checks for path / room / need (the map-blocked plan) are grown
    by map_margin (module doc, same map; the path term by at least OBJ_MARGIN); stand = the GAP stand-off behind the object, its footprint (+ ROBOT_MARGIN + WIGGLE_M) free over the
    push sweep; pre = where go_to ends and the robot turns to face u: the first of stand - a u + b w (a in PRE_BACK,
    then b in PRE_LAT; w = u turned +90 deg) whose turning circle (rad + 1 cm) misses every static box and the object,
    whose straight leg in to stand keeps the footprint (+ ROBOT_MARGIN) off them, and that go_to's planner reaches
    from the robot's pose on `grid` with the object's square marked (`clear_grid`). None: no such push."""
    rects = _static_rects(m, d, m.body(body).id)
    obj = [(c[0] - hw, c[1] - hw, c[0] + hw, c[1] + hw)]
    boxes = _box_xy(m, d)
    occ = clear_grid(grid, c, hw).occ
    q = base_pose(m, d)

    def reachable(p: np.ndarray) -> bool:
        return plan_path(occ, grid.origin, grid.res, (float(q[0]), float(q[1])), (float(p[0]), float(p[1])),
                         ROBOT_RADIUS, CLEARANCE + DISC_MARGIN) is not None

    best: tuple[np.ndarray, float, np.ndarray, np.ndarray] | None = None
    for u in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])):
        w = np.array([-u[1], u[0]])
        stand = c - u * (hw + BASE_HALF + GAP)
        s_u = next((float(s) for s in np.arange(CLEAR_STEP, CLEAR_MAX + 1e-9, CLEAR_STEP)
                    if (best is None or s < best[1]) and keepout_ok(c + u * s, hw + OBJ_MARGIN, None)
                    and (path is None or path_dist(_sq(c + u * s, hw + max(OBJ_MARGIN, map_margin)), path)
                         > ROBOT_RADIUS + PATH_CLEAR)
                    and _obj_clear(rects, boxes, hw, c, c + u * s) > 0.0
                    and all(_robot_free(p, ext, u, ROBOT_MARGIN + WIGGLE_M, rects) for p in _samples(stand, stand + u * s))
                    and room_places_reachable(grid, _sq(c + u * s, hw + map_margin))
                    and (need is None or need(stamp_poly(grid, _sq(c + u * s, hw + map_margin))))),
                   None)
        if s_u is None:
            continue
        pre = next((p for p in (stand - u * a + w * b for a in PRE_BACK for b in PRE_LAT)
                    if all(rect_dist(float(p[0]), float(p[1]), r_) >= rad + 0.01 for r_ in rects + obj)
                    and all(_robot_free(q_, ext, u, ROBOT_MARGIN, rects + obj) for q_ in _samples(p, stand))
                    and reachable(p)), None)
        if pre is not None:
            best = (u, s_u, pre, stand)
    return best


def clear_grid(grid: Grid, c: np.ndarray, hw: float) -> Grid:
    """The grid with every cell whose centre is inside the object's square (centre c, half side hw) occupied."""
    ny, nx = grid.occ.shape
    xs = grid.origin[0] + (np.arange(nx) + 0.5) * grid.res
    ys = grid.origin[1] + (np.arange(ny) + 0.5) * grid.res
    sq = (np.abs(xs[None, :] - c[0]) <= hw) & (np.abs(ys[:, None] - c[1]) <= hw)
    return Grid(grid.origin, grid.res, grid.occ | sq)


def _box_xy(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    return np.array([d.body(i).xpos[:2] for i in range(m.nbody) if m.body(i).name.startswith("box_")]).reshape(-1, 2)


def _obj_clear(rects: list[tuple[float, float, float, float]], boxes: np.ndarray, hw: float, p0: np.ndarray,
               p1: np.ndarray) -> float:
    """min fetch-box clearance along the object path p0 -> p1, -1 if a sample (square hw + OBJ_MARGIN) hits a static
    box or comes closer than hw * sqrt 2 + BOX_CLEAR to a box."""
    pts = _samples(p0, p1)
    if not all(_sq_free(p, hw + OBJ_MARGIN, rects) for p in pts):
        return -1.0
    clr = min(float(np.linalg.norm(boxes - p, axis=1).min()) if len(boxes) else 1e9 for p in pts)
    return clr if clr >= hw * math.sqrt(2.0) + BOX_CLEAR else -1.0


def _choose_plan(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, body: str, world: Any, l_o: float, hw: float,
                 a_long: float, door_id: str | None, l_r: float, ext: np.ndarray, rad: float,
                 map_margin: float = 0.0) -> tuple:
    """("side", sd, A, delta) | ("long",) | ("none",). l_r: the robot's lat on the through-push; ext / rad: its
    footprint (`footprint`). Feasibility and order: module doc, step 2; every end also keeps its room (module doc), its
    square grown by map_margin (same map)."""
    rects = _static_rects(m, d, m.body(body).id)
    boxes = _box_xy(m, d)

    def obj_clear(p0: np.ndarray, p1: np.ndarray) -> float:
        return _obj_clear(rects, boxes, hw, p0, p1)

    for a_side, delta in SIDE_PLANS:
        best: tuple | None = None
        for sd in (1, -1):
            clr = obj_clear(world(a_side, l_o), world(a_side, sd * SIDE_LAT))
            obj_now = [(world(a_side, l_o)[0] - hw, world(a_side, l_o)[1] - hw,
                        world(a_side, l_o)[0] + hw, world(a_side, l_o)[1] + hw)]
            a_t = a_side - hw - rad - 0.01
            l_s = l_o - sd * (hw + BASE_HALF + GAP)
            turn = world(a_t, l_r)
            u = (world(0.0, sd) - world(0.0, 0.0))  # heading on the side legs: sd * t
            route = _samples(turn, world(a_t, l_s)) + _samples(world(a_t, l_s), world(a_side - delta, l_s))
            sweep = _samples(world(a_side - delta, l_s), world(a_side - delta, sd * SIDE_LAT - sd * (hw + BASE_HALF)))
            ok = (clr > 0.0 and keepout_ok(world(a_side, sd * SIDE_LAT), hw, door_id)
                  and all(rect_dist(float(turn[0]), float(turn[1]), r_) >= rad + 0.01 for r_ in rects)
                  and all(_robot_free(p, ext, u, ROBOT_MARGIN, rects + obj_now) for p in route)
                  and all(_robot_free(p, ext, u, ROBOT_MARGIN + WIGGLE_M, rects) for p in sweep)
                  and room_places_reachable(grid, _sq(world(a_side, sd * SIDE_LAT), hw + map_margin)))
            if ok and (best is None or clr > best[0]):
                best = (clr, ("side", sd, a_side, delta))
        if best is not None:
            return best[1]
    if obj_clear(world(SIDE_PLANS[0][0], l_o), world(a_long, l_o)) > 0.0 and keepout_ok(world(a_long, l_o), hw, door_id) \
            and room_places_reachable(grid, _sq(world(a_long, l_o), hw + map_margin)):
        return ("long",)
    return ("none",)
