"""Path follower: plan on the known map, drive the force-limited base through the setpoints, finish at the goal pose.

Rule (task 002 item 3): `plan_path` on the known map inflated by ROBOT_RADIUS + CLEARANCE + DISC_MARGIN,
`segment_headings` with
the current yaw held while translating; each intermediate waypoint is one `body.drive_base` call with no settle (the
min-jerk setpoint stops at every waypoint, so the base never cuts a corner the planner did not check); at the last
waypoint the base first arrives with the held yaw, then turns in place to the goal yaw and settles SETTLE_S. The arm is
put in LOOK_ARM_QPOS by its actuators (`move_arm`) if it is not there (unless `arm_look=False`: carrying a box). The
base moves only through its actuators.

Blocked detector (task 003 R6; on in every go_to, so G1-reach runs with it in the loop). Three signals, checked every
physics step; the first one that fires stops the drive, the base setpoint is set to the current base pose (it holds
where it is and stops pushing), HOLD_S of physics runs, and go_to returns reached False with `blocked` filled:
- "tracking": servo lag |base xy setpoint - base xy| > LAG_MAX_M. Free drives: lag = kv/kp x speed = 0.05 s x
  0.5 m/s = 25 mm max (measured 22.7 mm, hall -> storage); 002: crate push <= 24 mm, welded cabinet 0.04 -> 1.4 m in
  2 s. 0.10 m = 4x the free-drive maximum. Independent of the base force limit as long as a free drive does not
  saturate it (it needs ~28 N at the speed cap; 300 N or 212 N per axis both leave it unsaturated, so the lag stays
  kv/kp x v).
- "no_progress": over the last STALL_S the setpoint advanced >= STALL_SP_M along the planned path (arc length of
  its closest point on the polyline) but the base advanced < STALL_M (sliding sideways along an obstacle face is no
  progress). Backup for the other two: on the cabinet with lag and bumper both disabled it is what stops the base.
- "bumper": the mean over BUMP_WIN_S of the total contact normal force on the base body (base_link geoms: bumper,
  plates, body; not arm / gripper, not the floor, not `ignore_geoms`, not fetch boxes: walls, furniture, crate,
  cabinet) > BUMP_N. This is the base contact force the
  task lists, NOT actuator force saturation (R6: the force saturates in normal crate pushing). Needed because a free
  4 kg crate is pushed along with lag <= 24 mm, so lag / progress alone never fire on it (measured: hall -> storage
  with the crate in d_sto pushes it 2 m into the room and "reaches"). Data (scratch probes, 2a): pushing an unseen
  0.3 kg fetch box on the route, 3 placements: max 1 s mean 12.6-13.9 N (impact + ~3 N friction); crate: impact +
  ~39 N friction (4 kg, mu 1), 1 s mean > 30 N 0.26 s after first contact, max 249 N. BUMP_N = 30 N: 2.2x above the
  box maximum, i.e. "the base is pushing something heavier than ~3 kg". Bounded by friction, not by the base force
  limit: unchanged at a 212 N per-axis limit (still >> 30 N).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from sim.body import StepCb, _ids, base_pose, drive_base, wrap
from sim.constants import BOX_COLOURS, GRID_RES, G1_REACH_POS_TOL, G1_REACH_YAW_TOL_DEG, ROBOT_RADIUS
from sim.look import LOOK_ARM_QPOS, move_arm, robot_geoms
from sim.nav.astar import Point, path_length, plan_path, segment_headings
from sim.robot import ARM_JOINTS, BASE_JOINTS
from sim.world.occupancy import Grid

CLEARANCE = 0.02  # true clearance, robot footprint radius to the nearest wall / furniture edge
# plan_path inflates cell centre to cell centre and its line-of-sight test accepts any point of a free cell, so the
# continuous clearance it guarantees is (inflation radius - res*sqrt(2)): half a cell diagonal on the free side plus
# half on the occupied side. Measured (G1-reach round 1, runs/1g/reach_r1.json): with only +0.02 the polylines passed
# door jambs at 0.2828 m < ROBOT_RADIUS 0.2842 and the base mesh grazed 3 walls (<= 0.33 mm). Adding res*sqrt(2)
# makes the planned polyline >= ROBOT_RADIUS + CLEARANCE from every occupied cell; the inflated map stays one component
# and every place pose stays free.
DISC_MARGIN = GRID_RES * math.sqrt(2.0)
SETTLE_S = 1.0
# Blocked detector (module docstring for the data behind each number).
LAG_MAX_M = 0.10
STALL_S, STALL_SP_M, STALL_M = 2.0, 0.10, 0.02
BUMP_WIN_S, BUMP_N = 1.0, 30.0
HOLD_S = 0.5
BACK_OFF_S = 2.5  # back-off drive duration (0.25 m: 0.1 m/s mean, the push speed)


class _Blocked(Exception):
    pass


def _arc(p: np.ndarray, a: np.ndarray, b: np.ndarray, s0: np.ndarray) -> float:
    """Arc length along the polyline (segments a[i] -> b[i], segment start arc s0[i]) of the point closest to p."""
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
    k = int(np.argmin(np.hypot(*(a + t[:, None] * ab - p).T)))
    return float(s0[k] + t[k] * math.sqrt(L2[k]))


def _dist_to_path(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Min distance from point p to the polyline with segments a[i] -> b[i]."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(a + t[:, None] * ab - p).T).min())


def go_to(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, goal_pose: tuple[float, float, float],
          cb: StepCb = None, via: np.ndarray | None = None, via_xy: Point | None = None, arm_look: bool = True,
          detect: bool = True, ignore_geoms: tuple[int, ...] = (), back_off: float = 0.0) -> dict[str, Any]:
    """Drive the base to goal_pose (x, y, yaw rad). Returns reached (path found, not blocked, and final errors within
    the G1-reach tolerances after settling), pos_err_m, yaw_err_deg, path_len_m, waypoints, sim_s, max_track_err_m
    (base xy to the planned polyline), max_lag_m (setpoint to base), max_bump_n (largest BUMP_WIN_S mean base contact
    force), contacts {"otherGeom|robotGeom": max penetration m} for robot vs non-robot geoms, first_contact_t
    {same keys: absolute d.time of the first contact}, blocked None | {"why": tracking|no_progress|bumper, "t": d.time
    at detection, "lag_m", "bump_n", "pose": [x, y, yaw]}. `via` = room cell mask, `via_xy` = a point the path must
    pass (door centre), see plan_path. `detect=False` turns the blocked detector off. back_off > 0: once blocked, the
    base first drives back_off m back along the planned path (its point back_off of arc length behind the base's
    closest point), slowly, detector off, contacts recorded, then holds; blocked["t"] stays the detection time and
    blocked["back_off_m"] is the distance actually driven back."""
    t0 = d.time
    _, _, arm_act = _ids(m, ARM_JOINTS)
    if arm_look and not np.allclose(d.ctrl[arm_act], LOOK_ARM_QPOS, atol=1e-6):
        move_arm(m, d, LOOK_ARM_QPOS, cb)
    q0 = base_pose(m, d)
    gx, gy, gyaw = goal_pose
    path = plan_path(grid.occ, grid.origin, grid.res, (float(q0[0]), float(q0[1])), (gx, gy),
                     ROBOT_RADIUS, CLEARANCE + DISC_MARGIN, via_cells=via, via_xy=via_xy)
    out: dict[str, Any] = {"reached": False, "waypoints": path, "path_len_m": None, "max_track_err_m": 0.0,
                           "max_lag_m": 0.0, "max_bump_n": 0.0, "contacts": {}, "first_contact_t": {},
                           "blocked": None}
    if path is None:
        q = base_pose(m, d)
        out.update(pos_err_m=float(math.hypot(q[0] - gx, q[1] - gy)), sim_s=0.0,
                   yaw_err_deg=abs(math.degrees(wrap(q[2] - gyaw))))
        return out
    out["path_len_m"] = path_length(path)
    pts = np.array(path)
    a, b = (pts[:-1], pts[1:]) if len(pts) > 1 else (pts, pts)
    rg = robot_geoms(m)
    base_b = m.body("base_link").id
    bump = set(int(g) for g in rg if m.geom_bodyid[g] == base_b)
    # bumper: non-box geoms only (2j fix 3: an unseen 0.3 kg box on the route tripped it at 30.1 N, blocked seed 1204);
    # box contacts stay in `contacts` (R4b); a box jammed against something still trips tracking / no_progress
    boxes = {m.body(f"box_{c}").id for c in BOX_COLOURS if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"box_{c}") >= 0}
    skip = {m.geom("floor").id, *ignore_geoms, *(g for g in range(m.ngeom) if m.geom_bodyid[g] in boxes)}
    qx, _, ax = _ids(m, BASE_JOINTS)
    qx, ax = qx[:2], ax[:2]
    contacts: dict[str, float] = out["contacts"]
    first: dict[str, float] = out["first_contact_t"]
    n_bump, n_stall = int(round(BUMP_WIN_S / m.opt.timestep)), int(round(STALL_S / m.opt.timestep))
    fwin: deque[float] = deque(maxlen=n_bump)
    hist: deque[tuple[float, float]] = deque(maxlen=n_stall)
    s0 = np.concatenate([[0.0], np.cumsum(np.hypot(*(b - a).T))[:-1]])
    f6 = np.zeros(6)
    fsum = [0.0]

    def step_cb(dd: mujoco.MjData) -> None:
        out["max_track_err_m"] = max(out["max_track_err_m"], _dist_to_path(dd.qpos[qx], a, b))
        fc = 0.0
        if dd.ncon:
            g = dd.contact.geom[: dd.ncon]
            r1, r2 = np.isin(g[:, 0], rg), np.isin(g[:, 1], rg)
            for k in np.nonzero(r1 ^ r2)[0]:
                gr, go = (g[k, 0], g[k, 1]) if r1[k] else (g[k, 1], g[k, 0])
                key = f"{m.geom(go).name or go}|{m.geom(gr).name or gr}"
                contacts[key] = max(contacts.get(key, 0.0), float(-dd.contact.dist[k]))
                first.setdefault(key, float(dd.time))
                if detect and gr in bump and go not in skip:
                    mujoco.mj_contactForce(m, dd, int(k), f6)
                    fc += abs(f6[0])
        if cb is not None:
            cb(dd)
        if not detect:
            return
        if len(fwin) == n_bump:
            fsum[0] -= fwin[0]
        fwin.append(fc)
        fsum[0] += fc
        sp, q = dd.ctrl[ax].copy(), dd.qpos[qx].copy()
        hist.append((_arc(sp, a, b, s0), _arc(q, a, b, s0)))
        lag = float(math.hypot(*(sp - q)))
        bmean = fsum[0] / n_bump
        out["max_lag_m"] = max(out["max_lag_m"], lag)
        out["max_bump_n"] = max(out["max_bump_n"], bmean)
        why = None
        if lag > LAG_MAX_M:
            why = "tracking"
        elif bmean > BUMP_N:
            why = "bumper"
        elif len(hist) == n_stall:
            (sp0, q0_), (sp1, q1) = hist[0], hist[-1]
            if sp1 - sp0 >= STALL_SP_M and q1 - q0_ < STALL_M:
                why = "no_progress"
        if why is not None:
            qq = base_pose(m, dd)
            out["blocked"] = {"why": why, "t": float(dd.time), "lag_m": lag, "bump_n": float(bmean),
                              "pose": [float(v) for v in qq]}
            raise _Blocked

    sp = segment_headings(path, gyaw, start_yaw=float(q0[2]))
    hold = float(q0[2])
    try:
        for x, y, _ in sp:  # sp[0] is the start (or its snapped free cell): a no-op unless the start was snapped
            drive_base(m, d, (x, y, hold), settle=0.0, cb=step_cb)
        drive_base(m, d, (gx, gy, gyaw), settle=SETTLE_S, cb=step_cb)  # turn in place at the goal, then settle
    except _Blocked:
        _, _, act = _ids(m, BASE_JOINTS)
        detect = False  # read by step_cb (closure): contacts keep being recorded while backing off / holding
        if back_off > 0.0 and len(pts) > 1:
            # 2j fix 5: back off along the path so the base does not stay pressed on (or scrape along) the blocker
            qb = base_pose(m, d)
            d.ctrl[act] = qb
            s_back = max(0.0, _arc(qb[:2], a, b, s0) - back_off)
            k = max(0, int(np.searchsorted(s0, s_back, side="right")) - 1)
            seg = b[k] - a[k]
            tgt = a[k] + seg * min(1.0, (s_back - s0[k]) / max(float(np.hypot(*seg)), 1e-12))
            drive_base(m, d, (float(tgt[0]), float(tgt[1]), float(qb[2])), seconds=BACK_OFF_S, settle=0.0, cb=step_cb)
            out["blocked"]["back_off_m"] = float(math.hypot(*(base_pose(m, d)[:2] - qb[:2])))
        # hold here: setpoint = current pose EVERY step (a setpoint frozen at detection was pressed 1.7 mm back by the
        # contact and the servo then pushed the blocker at the 212 N cap through the hold and into the next navigate,
        # whose bumper fired at once: 2i, push_or_detour detour branch)
        for _ in range(int(round(HOLD_S / m.opt.timestep))):
            d.ctrl[act] = base_pose(m, d)
            mujoco.mj_step(m, d)
            step_cb(d)
        d.ctrl[act] = base_pose(m, d)
    q = base_pose(m, d)
    out["pos_err_m"] = float(math.hypot(q[0] - gx, q[1] - gy))
    out["yaw_err_deg"] = abs(math.degrees(wrap(q[2] - gyaw)))
    xb, xm = d.body("base_link").xpos, d.body("base_link").xmat  # cross-check in the body frame
    out["body_pos_err_m"] = float(math.hypot(xb[0] - gx, xb[1] - gy))
    out["body_yaw_err_deg"] = abs(math.degrees(wrap(math.atan2(xm[3], xm[0]) - gyaw)))
    out["sim_s"] = float(d.time - t0)
    out["reached"] = bool(out["blocked"] is None
                          and max(out["pos_err_m"], out["body_pos_err_m"]) <= G1_REACH_POS_TOL
                          and max(out["yaw_err_deg"], out["body_yaw_err_deg"]) <= G1_REACH_YAW_TOL_DEG)
    return out
