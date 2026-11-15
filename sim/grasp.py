"""Pick and place of a fetch box in the house (task 003, track 2b). Plain physics: Robotiq 2F-85 actuator, 0.3 kg box,
nothing welded, base and box never teleported (the base moves only through `go_to` / `drive_base`).

pick_box: pre-grasp base pose planned from the box pose the caller gives (`pose_fn() -> (centre xyz, short-axis yaw)`:
the adapter passes the perceived pose in perceived mode, `true_box_pose` in truth mode; task 006): candidates at REACH distances x N_ANGLES around the
box, facing it, whose footprint (TidyBot square, half side FOOT_HALF, corners cut at ROBOT_RADIUS) + NEAR_MARGIN is free
on the known map and clear of the `avoid` discs, with the arm IK feasible for pre-grasp / grasp / lift. `go_to` drives
to a stand-off on the same ray that is free for the planner, then one straight `drive_base` closes in (furniture tops
sit closer than the planner's inflation allows). Arm: LOOK -> retract -> above the box -> straight down -> slow close
-> straight up -> CARRY (box over the base's back-left quarter, under the mast camera's frustum).
place_box: `go_to` the surface approach point with the arm kept in CARRY, straight in to PLACE_GAP from the near
edge, box over a target inside the edge, straight down until the box touches (landing height from the in-hand pose),
open, up, arm to retract -> LOOK, straight back to the approach point.

Grip depth (finding, 2b): the 2F-85 palm and spring links stop the hand when the box top is ~3.3 cm above
pinch_site, i.e. pinch ~2.7 cm above the box centre, whatever depth is commanded (sweep -2..+3 cm: the in-hand offset at
close is +2.6..+3.0 cm in every case). Phase 0's +2 cm command already sat at that stop; a deeper grip is not possible
top-down. GRIP_DZ commands just past the stop so the palm seats on the box top.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import mujoco
import numpy as np

import sim.nav.follow as follow
from sim.body import MINJERK_PEAK, StepCb, _ids, base_pose, drive_base, ik, run, wrap
from sim.constants import (BOX_HALF, G2_PICK_LIFT, ROBOT_RADIUS, SCORE_ON_TOP_TOL, SCORE_SURFACE_MARGIN,
                           SCORE_UPRIGHT_DEG)
from sim.look import LOOK_ARM_QPOS, move_arm, robot_geoms
from sim.robot import ARM_JOINTS, BASE_JOINTS, GRIPPER_ACT
from sim.world.occupancy import Grid

PINCH = "pinch_site"
GRIP_OPEN, GRIP_CLOSED = 0.0, 255.0
RETRACT_QPOS = (0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633)  # Menagerie `retract`
GRIP_DZ = 0.02  # commanded pinch height over the box centre; the palm stops the hand at ~+0.027
PRE_DZ = 0.10  # pre-grasp: pinch this far above the grasp point
LIFT_DZ = 0.20  # lift: pinch this far above the grasp point (box ends ~0.19 above its support)
REACH = (0.55, 0.50, 0.60, 0.45, 0.65)  # base centre to box centre (m), in order of preference
N_ANGLES = 24
FOOT_HALF = 0.241  # TidyBot base: 0.482 m square, corners rounded to ROBOT_RADIUS
NEAR_MARGIN = 0.03  # footprint clearance to walls / furniture at the pre-grasp pose and on the straight close-in
AVOID_MARGIN = 0.05  # footprint radius + this to an avoid disc
STANDOFF_MAX = 0.8
IK_TOL = 2e-3
KNOT_STEP = 0.05  # rad, max joint change between Cartesian knots
ARM_VMAX = 0.5  # rad/s, peak joint speed along knot paths (the box hangs from the pads and swings when hurried)
MAX_GOTO = 3  # candidates tried with go_to before "unreachable"
BLOCKERS = ("crate", "cabinet")  # blocker geom names (generate.py names each blocker's one box geom after its body)
CLOSE_S, SETTLE_S = 1.5, 0.5
# CARRY: pinch over the base's back-left quarter (base frame), closing axis at CARRY_YAW to the heading, arm joints
# near CARRY_SEED. Picked by a rendered search (2b, runs/2b/cv/carry_search.json: 5x4x2 pinch points x 4 yaws x 12
# IK seeds; kept: robot + box inside the footprint radius, no contacts, >= 1 cm from the mast pole, reachable by a
# continuous top-down transit) for the fewest robot + box pixels in the mast image (box itself 0 px; the elbow
# shows). No top-down carry with the box clear of the base top hides as little as LOOK_ARM_QPOS: 13.6k px vs 7.3k.
CARRY_PINCH = (-0.04, -0.18, 0.46)
CARRY_YAW = math.radians(45.0)
CARRY_SEED = (2.674, 0.302, 2.092, -2.568, 0.579, -0.49, 1.907)
TRANSIT_N, TRANSIT_ARC, TRANSIT_STEP, TRANSIT_S = 40, 0.05, 0.15, 6.0
RETRACT_PINCH = (0.13, 0.0, 0.55)  # base frame, ~ pinch_site at RETRACT_QPOS (0.130, 0.001, 0.544)
OVER_BASE = (0.20, 0.0, 0.60)  # base frame: pinch over the front of the base, box long side across (clear of the shoulder)
PLACE_GAP = 0.28  # base centre to the surface near edge when placing (front face 4 cm from the edge): top-down reach
# at the kitchen table (pinch ~0.88 m high) ends at ~0.40 m from the base centre
ROLL_FRACTIONS = (1.0, 0.5, 0.0)  # of the levelling roll, tried in this order (outer loop over PLACE_INSETS)
PLACE_INSETS = (0.04, 0.03, 0.02)  # box footprint this far inside the near edge, first with a reachable path (the
# scorer needs the centre >= 2 cm inside; the box half depth adds 3 cm)
RETREAT_UP = 0.10  # straight-up retreat after release
RETREAT_LIFTS = (0.02, 0.01, 0.005)  # else lift this much (palm off the box top), then slide sideways
RETREAT_PAST = 0.05  # slide until the pinch is this far past the box's end (pad half width ~1 cm + margin)
RISE_GAP = 0.01  # box far face this far short of the near edge at the top of the rise
JOINT_MARGIN = 0.03  # rad: every planned place / retreat knot this far from the arm's joint limits
PLACE_ABOVE = 0.01  # box bottom this far above top_z before the final descent (2j: 0.02 put the kitchen-table crossing
# at the arm's wrist limit; 1 cm costs ~0.035 rad of joint_6 margin there)
PLACE_TRAVEL = 0.06  # box bottom up to this much higher still while moving over the edge (lowered while out of reach)
DESCENT_V = 0.05  # m/s, final descent
CLEARANCE_FOLLOW = follow.CLEARANCE + follow.DISC_MARGIN  # go_to's inflation over ROBOT_RADIUS

Phase = Callable[[str, mujoco.MjData], None] | None


# ---- state ------------------------------------------------------------------------------------------------------

def box_yaw(xmat: np.ndarray) -> float:
    r = np.asarray(xmat).reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


def tilt_deg(xmat: np.ndarray) -> float:
    """Angle of the box's z (12 cm) axis from world z."""
    return math.degrees(math.acos(float(np.clip(np.asarray(xmat).reshape(3, 3)[2, 2], -1.0, 1.0))))


def pad_contacts(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> dict[str, Any]:
    """Normal force of each pad on `body`, other robot bodies touching it, non-robot bodies touching it."""
    b = m.body(body).id
    pads = {"left": 0.0, "right": 0.0}
    robot: set[str] = set()
    world: set[str] = set()
    root = m.body("base_link").id
    f6 = np.zeros(6)
    for i, c in enumerate(d.contact[: d.ncon]):
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b not in (b1, b2):
            continue
        o = b2 if b1 == b else b1
        name = m.body(o).name
        mujoco.mj_contactForce(m, d, i, f6)
        side = name.removesuffix("_pad")
        if side in pads:
            pads[side] += float(f6[0])
        elif m.body_rootid[o] == root:
            robot.add(name)
        else:
            world.add(name)
    return {"left_pad": pads["left"] > 0.0, "right_pad": pads["right"] > 0.0, "left_N": pads["left"],
            "right_N": pads["right"], "other_robot": sorted(robot), "world": sorted(world)}


def holding(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> bool:
    """The box is between the pads: its centre within 6 cm of pinch_site and at least one pad pressing on it (a box
    hanging on one pad still counts, 005 tuning 8217). THE "held" of SimRobot.held() (deviation 18); the scorer adds
    any pad contact on top (SimRobot.held_any)."""
    c = pad_contacts(m, d, body)
    return bool((c["left_pad"] or c["right_pad"]) and np.linalg.norm(d.body(body).xpos - d.site(PINCH).xpos) < 0.06)


def in_hand(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> dict[str, Any]:
    """Box pose in the pinch_site frame (site z = approach direction): offset (m), quat (w, x, y, z), yaw about site z."""
    rp = d.site(PINCH).xmat.reshape(3, 3)
    off = rp.T @ (d.body(body).xpos - d.site(PINCH).xpos)
    rr = rp.T @ d.body(body).xmat.reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rr.flatten())
    return {"offset": off.tolist(), "quat": q.tolist(), "yaw": math.atan2(rr[1, 0], rr[0, 0])}


def gripper_open(m: mujoco.MjModel, d: mujoco.MjData) -> bool:
    return bool(d.ctrl[m.actuator(GRIPPER_ACT).id] <= GRIP_OPEN + 1e-9)


def arm_at(m: mujoco.MjModel, d: mujoco.MjData, q: tuple[float, ...], tol: float = 0.05) -> bool:
    """Arm joint positions (not only the targets) within tol rad of q."""
    return bool(np.abs(d.qpos[_ids(m, ARM_JOINTS)[0]] - np.asarray(q)).max() <= tol)


# ---- base footprint on the known map ---------------------------------------------------------------------------

def mark_discs(grid: Grid, discs: list[tuple[float, float, float]]) -> Grid:
    """Copy of the grid with every cell whose centre is within r of (x, y) occupied."""
    occ = grid.occ.copy()
    ny, nx = occ.shape
    xs = grid.origin[0] + (np.arange(nx) + 0.5) * grid.res
    ys = grid.origin[1] + (np.arange(ny) + 0.5) * grid.res
    for x, y, r in discs:
        occ |= (xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2 <= r * r
    return Grid(grid.origin, grid.res, occ)


_FOOT_PTS: dict[float, np.ndarray] = {}


def _foot_pts(margin: float) -> np.ndarray:
    """Sample points (2 cm) of the footprint grown by margin, base frame."""
    if margin not in _FOOT_PTS:
        h, r = FOOT_HALF + margin, ROBOT_RADIUS + margin
        g = np.arange(-h, h + 1e-9, 0.02)
        p = np.array([(x, y) for x in np.append(g, h) for y in np.append(g, h)])
        _FOOT_PTS[margin] = p[np.hypot(p[:, 0], p[:, 1]) <= r + 1e-9]
    return _FOOT_PTS[margin]


def foot_free(grid: Grid, x: float, y: float, yaw: float, margin: float = NEAR_MARGIN) -> bool:
    """Footprint + margin at (x, y, yaw) touches no occupied cell (cells over-cover walls and furniture)."""
    p = _foot_pts(margin)
    c, s = math.cos(yaw), math.sin(yaw)
    wx, wy = x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]
    ix = np.floor((wx - grid.origin[0]) / grid.res).astype(int)
    iy = np.floor((wy - grid.origin[1]) / grid.res).astype(int)
    ny, nx = grid.occ.shape
    if ix.min() < 0 or iy.min() < 0 or ix.max() >= nx or iy.max() >= ny:
        return False
    return not bool(grid.occ[iy, ix].any())


def segment_free(grid: Grid, a: tuple[float, float], b: tuple[float, float], yaw: float, margin: float) -> bool:
    n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.02)))
    return all(foot_free(grid, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, yaw, margin)
               for t in np.linspace(0.0, 1.0, n + 1))


# ---- arm --------------------------------------------------------------------------------------------------------

def top_down_rot(closing_yaw: float, roll: float = 0.0) -> np.ndarray:
    """Site frame: z (approach) = world down, x (finger closing axis) = horizontal at closing_yaw; then turned by roll
    about the closing axis (place: levels a box that pivoted in the pads)."""
    x = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    c, s = math.cos(roll), math.sin(roll)
    return np.column_stack([x, np.cross(z, x), z]) @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ik_at(m: mujoco.MjModel, d: mujoco.MjData, q_arm: np.ndarray, pos: np.ndarray, rot: np.ndarray,
          base: tuple[float, float, float] | None = None) -> tuple[np.ndarray, float, float]:
    """`body.ik` seeded at q_arm, optionally with the base at `base` instead of its current pose."""
    s = mujoco.MjData(m)
    s.qpos[:] = d.qpos
    s.qpos[_ids(m, ARM_JOINTS)[0]] = q_arm
    if base is not None:
        s.qpos[_ids(m, BASE_JOINTS)[0]] = base
    return ik(m, s, pos, rot)


def ik_mid(m: mujoco.MjModel, d: mujoco.MjData, q_arm: np.ndarray, pos: np.ndarray, rot: np.ndarray,
           iters: int = 100, k: float = 0.5) -> tuple[np.ndarray, float, float]:
    """`ik_at`, then the 7th (redundant) DOF used to move the limited joints towards mid-range while the pinch pose is
    held (damped least squares + null-space gradient). Place only (2j fix 2: wrist margin at the kitchen table)."""
    q, _, _ = ik_at(m, d, q_arm, pos, rot)
    qadr, dadr, _ = _ids(m, ARM_JOINTS)
    ids = [m.joint(j).id for j in ARM_JOINTS]
    lim = m.jnt_limited[ids].astype(bool)
    lo, hi = m.jnt_range[ids].T
    s = mujoco.MjData(m)
    s.qpos[:] = d.qpos
    sid = m.site(PINCH).id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        s.qpos[qadr] = q
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        cur = s.site_xmat[sid].reshape(3, 3)
        e = np.concatenate([pos - s.site_xpos[sid], 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))])
        mujoco.mj_jacSite(m, s, jp, jr, sid)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
        grad = np.zeros(len(q))
        grad[lim] = ((lo + hi)[lim] / 2 - q[lim]) / ((hi - lo)[lim] / 2) ** 2
        q = q + np.clip(Jp @ e + (np.eye(len(q)) - Jp @ J) @ (k * grad), -0.2, 0.2)
        q[lim] = np.clip(q[lim], lo[lim], hi[lim])
    s.qpos[qadr] = q
    mujoco.mj_kinematics(m, s)
    cur = s.site_xmat[sid].reshape(3, 3)
    er = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
    return q, float(np.linalg.norm(pos - s.site_xpos[sid])), float(np.linalg.norm(er))


def arm_ctrl(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    return d.ctrl[_ids(m, ARM_JOINTS)[2]].copy()


def cart_knots(m: mujoco.MjModel, d: mujoco.MjData, q0: np.ndarray, p0: np.ndarray, p1: np.ndarray,
               yaw0: float, yaw1: float, n: int = 8, roll0: float = 0.0,
               roll1: float | None = None, mid: bool = False) -> tuple[list[np.ndarray], float]:
    """IK knots along the straight top-down pinch line p0 -> p1, closing yaw interpolated yaw0 -> yaw1 (roll about the
    closing axis roll0 -> roll1, default constant), each seeded by the previous one; the knot count doubles (up to 16x) until no joint moves more than KNOT_STEP between knots, so
    linear joint interpolation between knots stays on the line. Returns (knots, max position / rotation error)."""
    for nn in (n, 2 * n, 4 * n, 8 * n, 16 * n):
        knots, q, err = [], q0, 0.0
        for s in np.linspace(0.0, 1.0, nn + 1)[1:]:
            r1 = roll0 if roll1 is None else roll1
            q, ep, er = (ik_mid if mid else ik_at)(m, d, q, p0 + (p1 - p0) * s,
                                                   top_down_rot(yaw0 + wrap(yaw1 - yaw0) * s, roll0 + (r1 - roll0) * s))
            knots.append(q)
            err = max(err, ep, er)
        if np.abs(np.diff(np.vstack([q0] + knots), axis=0)).max() <= KNOT_STEP:
            break
    return knots, err


def follow_knots(m: mujoco.MjModel, d: mujoco.MjData, knots: list[np.ndarray], seconds: float,
                 cb: StepCb = None, stop: Callable[[mujoco.MjData], bool] | None = None) -> bool:
    """Arm targets along the knot polyline from the current targets, min-jerk timing over the whole path, at least
    long enough that the peak joint speed stays <= ARM_VMAX.
    Stops early (targets frozen at the current joint positions) when stop(d) is true; returns whether it stopped."""
    qadr, _, act = _ids(m, ARM_JOINTS)
    pts = np.vstack([d.ctrl[act]] + list(knots))
    seconds = max(seconds, MINJERK_PEAK * float(np.abs(np.diff(pts, axis=0)).max(1).sum()) / ARM_VMAX)
    n = max(1, int(round(seconds / m.opt.timestep)))
    k = len(pts) - 1
    for i in range(1, n + 1):
        s = i / n
        u = (10 * s**3 - 15 * s**4 + 6 * s**5) * k
        j = min(int(u), k - 1)
        d.ctrl[act] = pts[j] + (pts[j + 1] - pts[j]) * (u - j)
        mujoco.mj_step(m, d)
        if cb is not None:
            cb(d)
        if stop is not None and stop(d):
            d.ctrl[act] = d.qpos[qadr]
            return True
    return False


def set_gripper(m: mujoco.MjModel, d: mujoco.MjData, target: float, seconds: float, cb: StepCb = None) -> None:
    """Ramp the gripper command linearly to target over `seconds` (slow close / open)."""
    g = m.actuator(GRIPPER_ACT).id
    c0 = float(d.ctrl[g])
    n = max(1, int(round(seconds / m.opt.timestep)))
    for i in range(1, n + 1):
        d.ctrl[g] = c0 + (target - c0) * i / n
        mujoco.mj_step(m, d)
        if cb is not None:
            cb(d)


def to_world(m: mujoco.MjModel, d: mujoco.MjData, p_base: tuple[float, float, float]) -> np.ndarray:
    x, y, th = base_pose(m, d)
    c, s = math.cos(th), math.sin(th)
    return np.array([x + c * p_base[0] - s * p_base[1], y + s * p_base[0] + c * p_base[1], p_base[2]])


def closing_yaw_near(axis_yaw: float, ref: float) -> float:
    """axis_yaw or axis_yaw + pi (the gripper is symmetric), whichever is nearer ref."""
    return min((axis_yaw, axis_yaw + math.pi), key=lambda a: abs(wrap(a - ref)))


def blend_knots(m: mujoco.MjModel, d: mujoco.MjData, qa: np.ndarray, qb: np.ndarray,
                at: Callable[[float], np.ndarray], yaw_at: Callable[[float], float],
                n: int = TRANSIT_N) -> tuple[list[np.ndarray], float] | None:
    """Top-down pinch path at(s), closing yaw yaw_at(s), s in [0, 1]: knot k is IK seeded at the joint-space blend
    qa + (qb - qa) k/n, so the path can cross between IK branches through the arm's redundancy. None unless it is
    continuous (knot step <= TRANSIT_STEP rad) with error <= IK_TOL. Returns (knots incl. s = 0, max error)."""
    knots, err = [], 0.0
    for k in range(n + 1):
        s = k / n
        q, ep, er = ik_at(m, d, qa + (qb - qa) * s, at(s), top_down_rot(yaw_at(s)))
        if knots and np.abs(q - knots[-1]).max() > TRANSIT_STEP:
            return None
        knots.append(q)
        err = max(err, ep, er)
    return (knots, err) if err <= IK_TOL else None


def transit_knots(m: mujoco.MjModel, d: mujoco.MjData, y0: float) -> tuple[list[np.ndarray], float] | None:
    """Top-down pinch path OVER_BASE (closing yaw y0, front-reach IK branch, retract seed) -> CARRY_PINCH (folded-back
    branch, CARRY_SEED): an arc around the arm's shoulder axis through the base's -y side (the mast is on the 180 deg
    side), radius and height blended, raised TRANSIT_ARC mid-way, closing yaw blended to CARRY_YAW (either sign).
    Base-frame geometry only, so the result does not depend on where the robot stands."""
    th = float(base_pose(m, d)[2])
    r0, r1 = math.hypot(*OVER_BASE[:2]), math.hypot(*CARRY_PINCH[:2])
    phi1 = math.atan2(CARRY_PINCH[1], CARRY_PINCH[0])

    def at(s: float) -> np.ndarray:
        ph, r = phi1 * s, r0 + (r1 - r0) * s
        z = OVER_BASE[2] + (CARRY_PINCH[2] - OVER_BASE[2]) * s + TRANSIT_ARC * math.sin(math.pi * s)
        return to_world(m, d, (r * math.cos(ph), r * math.sin(ph), z))

    q0 = ik_at(m, d, np.array(RETRACT_QPOS), at(0.0), top_down_rot(y0))[0]
    for y1 in sorted((th + CARRY_YAW, th + CARRY_YAW + math.pi), key=lambda a: abs(wrap(a - y0))):
        q1, e1, r_1 = ik_at(m, d, np.array(CARRY_SEED), at(1.0), top_down_rot(y1))
        if max(e1, r_1) > IK_TOL:
            continue
        tk = blend_knots(m, d, q0, q1, at, lambda s: y0 + wrap(y1 - y0) * s)
        if tk is not None:
            return tk
    return None


def to_carry(m: mujoco.MjModel, d: mujoco.MjData, yaw: float, cb: StepCb = None) -> float | None:
    """From the lift: straight up, straight to OVER_BASE (closing axis turned to the heading), a self-motion at that
    pose onto the transit's start configuration (the straight part may end elsewhere in the arm's null space), then the
    transit to CARRY. Returns the max IK error, None (arm not moved) if no continuous path exists."""
    th = float(base_pose(m, d)[2])
    p_now = d.site(PINCH).xpos.copy()
    p1 = to_world(m, d, OVER_BASE)
    up = np.array([p_now[0], p_now[1], max(p1[2], p_now[2])])  # straight up first: clear the base's front edge
    kn, e1 = cart_knots(m, d, arm_ctrl(m, d), p_now, up, yaw, yaw, n=4)
    for y0 in sorted((th, th + math.pi), key=lambda a: abs(wrap(a - yaw))):
        kn2, e2 = cart_knots(m, d, kn[-1], up, p1, yaw, y0)
        tk = transit_knots(m, d, y0)
        if tk is None:
            continue
        sm = blend_knots(m, d, kn2[-1], tk[0][0], lambda s: p1, lambda s: y0, n=30)
        if sm is None:
            continue
        follow_knots(m, d, kn + kn2, 4.0, cb)
        follow_knots(m, d, sm[0][1:], 3.0, cb)
        follow_knots(m, d, tk[0][1:], TRANSIT_S, cb)
        return max(e1, e2, sm[1], tk[1])
    return None


def from_carry(m: mujoco.MjModel, d: mujoco.MjData, y0: float, cb: StepCb = None) -> float | None:
    """CARRY -> OVER_BASE with closing yaw y0 (the transit reversed). Returns the max IK error, None (arm not moved)
    if no path starts at the current arm targets."""
    tk = transit_knots(m, d, y0)
    if tk is None or np.abs(tk[0][-1] - arm_ctrl(m, d)).max() > TRANSIT_STEP:
        return None
    follow_knots(m, d, tk[0][::-1][1:], TRANSIT_S, cb)
    return tk[1]


def stopped_by_blocker(nav: dict[str, Any]) -> bool:
    """The drive was stopped by the blocked detector (tracking / no_progress / bumper) with a crate or cabinet contact:
    pick / place then say "blocked" (deviation 16), every other drive failure stays "unreachable"."""
    return nav["blocked"] is not None and any(k.split("|")[0] in BLOCKERS for k in nav["contacts"])


def go_to_keep_arm(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, pose: tuple[float, float, float],
                   cb: StepCb = None, held_body: str | None = None) -> dict[str, Any]:
    """follow.go_to with the arm held at its current targets (carry); the held box's geoms are ignored by the
    blocked detector's bumper (it rides on the robot)."""
    ign = tuple(i for i in range(m.ngeom) if held_body and m.geom_bodyid[i] == m.body(held_body).id)
    return follow.go_to(m, d, grid, pose, cb, arm_look=False, ignore_geoms=ign)


def robot_xy_radius(m: mujoco.MjModel, d: mujoco.MjData, body: str | None = None) -> float:
    """Max xy distance from the base centre of any robot geom point (+ the held box's corners)."""
    from sim.look import geom_points

    g = robot_geoms(m)
    if body is not None:
        g = np.append(g, [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(body).id])
    p = geom_points(m, d, g) - d.body("base_link").xpos
    return float(np.hypot(p[:, 0], p[:, 1]).max())


# ---- pick -------------------------------------------------------------------------------------------------------

PoseFn = Callable[[], tuple[np.ndarray, float]]  # () -> (box centre xyz, yaw of its short (body x) axis)


def true_box_pose(d: mujoco.MjData, body: str) -> tuple[np.ndarray, float]:
    """TRUTH: the box body's centre and short-axis yaw (truth mode, the task generator's pre-grasp check, tests)."""
    return d.body(body).xpos.copy(), box_yaw(d.body(body).xmat)


def _support_z(d: mujoco.MjData, body: str) -> float:
    """TRUTH, outcome measurement only: the box's support height before the grasp, for lift_above_support_m."""
    return float(d.body(body).xpos[2]) - BOX_HALF[2]


def _tilted(d: mujoco.MjData, body: str) -> bool:
    """TRUTH, outcome measurement after contact: the box tipped over (> 10 deg) when the gripper let it drop."""
    return tilt_deg(d.body(body).xmat) > 10.0


def plan_pregrasp(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, body: str,
                  avoid: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    """Truth wrapper (sim.tasks' pre-grasp check, pregrasp_survey): plan_pregrasp_at on the box's true pose."""
    return plan_pregrasp_at(m, d, grid, *true_box_pose(d, body), avoid)


def plan_pregrasp_at(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, bp: np.ndarray, axis: float,
                     avoid: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    """Feasible pre-grasp base poses for a box at centre bp with short-axis yaw `axis`, best first: dict(pose,
    standoff, reach, q_pre, q_grasp, yaw)."""
    bp = np.asarray(bp, dtype=float).copy()
    gp = bp + [0.0, 0.0, GRIP_DZ]
    avoid = [a for a in avoid if math.hypot(a[0] - bp[0], a[1] - bp[1]) > 0.05]
    g_near = mark_discs(grid, [(x, y, r + AVOID_MARGIN) for x, y, r in avoid])
    g_plan = grid.inflate(ROBOT_RADIUS + CLEARANCE_FOLLOW)
    g_plan = mark_discs(g_plan, [(x, y, r + ROBOT_RADIUS + CLEARANCE_FOLLOW) for x, y, r in avoid])
    q0 = base_pose(m, d)
    cands = []
    for ri, r in enumerate(REACH):
        for k in range(N_ANGLES):
            a = 2 * math.pi * k / N_ANGLES
            x, y = bp[0] - r * math.cos(a), bp[1] - r * math.sin(a)
            if not foot_free(g_near, x, y, a):
                continue
            so = None
            for back in np.arange(0.0, STANDOFF_MAX + 1e-9, 0.05):
                sx, sy = x - back * math.cos(a), y - back * math.sin(a)
                if not foot_free(g_near, sx, sy, a):
                    break
                if g_plan.free(sx, sy):
                    so = (sx, sy)
                    break
            if so is None or not segment_free(g_near, so, (x, y), a, NEAR_MARGIN):
                continue
            cost = math.hypot(so[0] - q0[0], so[1] - q0[1]) + 0.5 * ri
            cands.append((cost, r, a, (x, y), so))
    out = []
    for cost, r, a, (x, y), so in sorted(cands):
        pose = (x, y, a)
        # both closing-axis signs; keep the IK solution nearest retract (the other may sit on a flipped shoulder
        # branch from which no continuous top-down path reaches CARRY)
        sols = []
        for yaw in (axis, axis + math.pi):
            qp, e1, e2 = ik_at(m, d, np.array(RETRACT_QPOS), gp + [0, 0, PRE_DZ], top_down_rot(yaw), base=pose)
            if max(e1, e2) <= IK_TOL:
                sols.append((float(np.abs(qp - np.array(RETRACT_QPOS)).max()), yaw, qp))
        if not sols:
            continue
        _, yaw, qp = min(sols, key=lambda v: v[0])
        ql, e1, e2 = ik_at(m, d, qp, gp + [0, 0, LIFT_DZ], top_down_rot(yaw), base=pose)
        qg, e3, e4 = ik_at(m, d, qp, gp, top_down_rot(yaw), base=pose)
        if max(e1, e2, e3, e4) > IK_TOL:
            continue
        out.append({"pose": pose, "standoff": (so[0], so[1], a), "reach": r, "q_pre": qp, "yaw": yaw})
        if len(out) >= MAX_GOTO:
            break
    return out


def _grasp_once(m: mujoco.MjModel, d: mujoco.MjData, body: str, bp: np.ndarray, yaw: float, cb: StepCb,
                t: dict[str, float], phase: Phase = None) -> dict[str, Any]:
    """From the arm above the box at bp (or anywhere): above -> down -> close -> up. Returns the lift measurements
    (`body` is read only by the outcome measurements: support height, in-hand pose, pads, holding)."""
    bp = np.asarray(bp, dtype=float).copy()
    sup = _support_z(d, body)
    gp = bp + [0.0, 0.0, GRIP_DZ]
    pre = gp + [0, 0, PRE_DZ]
    t0 = d.time
    qp, ep, er = ik_at(m, d, arm_ctrl(m, d), pre, top_down_rot(yaw))
    move_arm(m, d, tuple(qp), cb)
    if phase:
        phase("pregrasp", d)
    kn, e_down = cart_knots(m, d, qp, pre, gp, yaw, yaw)
    follow_knots(m, d, kn, 2.0, cb)
    run(m, d, 0.3, cb)
    t["descend_s"] = t.get("descend_s", 0.0) + d.time - t0
    t0 = d.time
    set_gripper(m, d, GRIP_CLOSED, CLOSE_S, cb)
    run(m, d, SETTLE_S, cb)
    t["close_s"] = t.get("close_s", 0.0) + d.time - t0
    if phase:
        phase("grasp", d)
    at_close = in_hand(m, d, body)
    pads_close = pad_contacts(m, d, body)
    t0 = d.time
    kn, e_up = cart_knots(m, d, kn[-1], gp, gp + [0, 0, LIFT_DZ], yaw, yaw)
    follow_knots(m, d, kn, 2.5, cb)
    run(m, d, SETTLE_S, cb)
    t["lift_s"] = t.get("lift_s", 0.0) + d.time - t0
    if phase:
        phase("lift", d)
    return {"closing_yaw": yaw, "support_z": sup, "lift_above_support_m": _support_z(d, body) - sup, "in_hand_close": at_close,
            "pads_close": pads_close, "ik_err": max(ep, er, e_down, e_up), "held": holding(m, d, body)}


def pick_box(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, body: str,
             avoid: list[tuple[float, float, float]], cb: StepCb = None, phase: Phase = None, *,
             pose_fn: PoseFn) -> dict[str, Any]:
    """Pick the box `body` from wherever the robot is. avoid = other known boxes (x, y, radius) for base planning.
    pose_fn() = the box pose every pre-contact decision uses (pre-grasp plan, drive grid, grasp point, closing yaw);
    it is called when planning and again before each grasp. Truth read here after that: only the outcome measurements
    (_support_z, holding, in_hand, pad_contacts, _tilted, robot_xy_radius, _end), never a pose for a decision before
    contact. The "box not upright" pre-check is the adapter's (truth mode).
    phase(name, d) is called at "pregrasp", "grasp", "lift", "carry" (frames for strips / clips).
    Returns {ok, reason ("" | "blocked" | "unreachable" | "missed" | "slipped"), detail}: "blocked" when every
    candidate's drive stopped on a blocker (stopped_by_blocker)."""
    t: dict[str, float] = {}
    w0, s0 = time.perf_counter(), d.time
    detail: dict[str, Any] = {"timings": t, "attempts": 0}
    bp, axis = pose_fn()
    cands = plan_pregrasp_at(m, d, grid, bp, axis, avoid)
    t["plan_wall_s"] = time.perf_counter() - w0
    detail["n_candidates"] = len(cands)
    g_drive = mark_discs(grid, [(x, y, r) for x, y, r in avoid if math.hypot(x - bp[0], y - bp[1]) > 0.05])
    chosen = None
    on_blocker = []
    for c in cands:
        t0 = d.time
        nav = follow.go_to(m, d, g_drive, c["standoff"], cb)
        t["drive_s"] = t.get("drive_s", 0.0) + d.time - t0
        detail.setdefault("go_to", []).append({k: nav.get(k) for k in ("reached", "pos_err_m", "yaw_err_deg",
                                                                       "path_len_m", "sim_s", "contacts", "blocked",
                                                                       "waypoints")})
        if nav["reached"]:
            chosen = c
            break
        on_blocker.append(stopped_by_blocker(nav))
    if chosen is None:
        reason = "blocked" if on_blocker and all(on_blocker) else "unreachable"
        return {"ok": False, "reason": reason, "detail": {**detail, "why": "no pre-grasp pose / path"}}
    t0 = d.time
    if math.hypot(chosen["pose"][0] - chosen["standoff"][0], chosen["pose"][1] - chosen["standoff"][1]) > 1e-6:
        detail["close_in"] = drive_base(m, d, chosen["pose"], settle=0.5, cb=cb)
    t["close_in_s"] = d.time - t0
    detail["pregrasp"] = {"pose": list(chosen["pose"]), "standoff": list(chosen["standoff"]),
                          "reach": chosen["reach"], "closing_yaw": chosen["yaw"],
                          "base_actual": base_pose(m, d).tolist()}
    t0 = d.time
    move_arm(m, d, RETRACT_QPOS, cb)
    t["arm_out_s"] = d.time - t0
    yaw = chosen["yaw"]
    for attempt in (1, 2):  # one regrasp after a failed lift
        detail["attempts"] = attempt
        bp, axis = pose_fn()
        g = _grasp_once(m, d, body, bp, closing_yaw_near(axis, yaw), cb, t, phase)
        detail[f"grasp{attempt}"] = g
        if g["held"] and g["lift_above_support_m"] >= G2_PICK_LIFT:
            break
        set_gripper(m, d, GRIP_OPEN, 0.5, cb)
        run(m, d, 1.0, cb)  # the box drops back onto its support (or is still on it)
        if _tilted(d, body):
            break
    else:
        attempt = 3
    g = detail[f"grasp{detail['attempts']}"]
    if not (g["held"] and g["lift_above_support_m"] >= G2_PICK_LIFT):
        move_arm(m, d, RETRACT_QPOS, cb)
        move_arm(m, d, LOOK_ARM_QPOS, cb)
        pc = g["pads_close"]
        reason = "missed" if not (pc["left_pad"] and pc["right_pad"]) else "slipped"
        return {"ok": False, "reason": reason, "detail": {**detail, **_end(m, d, body, s0, t)}}
    detail["in_hand"] = in_hand(m, d, body)  # after lift: what place uses
    detail["lift_above_support_m"] = g["lift_above_support_m"]
    detail["pads_lift"] = pad_contacts(m, d, body)
    t0 = d.time
    detail["carry_ik_err"] = to_carry(m, d, g["closing_yaw"], cb)
    if detail["carry_ik_err"] is None:
        return {"ok": False, "reason": "unreachable", "detail": {**detail, "why": "no carry transit", **_end(m, d, body, s0, t)}}
    run(m, d, SETTLE_S, cb)
    t["to_carry_s"] = d.time - t0
    if phase:
        phase("carry", d)
    held = holding(m, d, body)
    detail["in_hand_carry"] = in_hand(m, d, body)
    detail["carry_xy_radius_m"] = robot_xy_radius(m, d, body)
    detail.update(_end(m, d, body, s0, t))
    return {"ok": held, "reason": "" if held else "slipped", "detail": detail}


def _end(m: mujoco.MjModel, d: mujoco.MjData, body: str, s0: float, t: dict[str, float]) -> dict[str, Any]:
    t["total_s"] = d.time - s0
    return {"pads": pad_contacts(m, d, body), "box_pos": d.body(body).xpos.tolist(),
            "box_tilt_deg": tilt_deg(d.body(body).xmat), "held": holding(m, d, body)}


# ---- place ------------------------------------------------------------------------------------------------------

def _edge_dist(x: float, y: float, yaw: float, rect: list[float]) -> float:
    """Distance along heading yaw from (x, y) to the first edge of rect (inf if the ray misses it)."""
    x0, y0, x1, y1 = rect
    c, s = math.cos(yaw), math.sin(yaw)
    lo, hi = 0.0, math.inf
    for p, v, a, b in ((x, c, x0, x1), (y, s, y0, y1)):
        if abs(v) < 1e-9:
            if not a <= p <= b:
                return math.inf
            continue
        t1, t2 = sorted(((a - p) / v, (b - p) / v))
        lo, hi = max(lo, t1), min(hi, t2)
    return lo if lo <= hi else math.inf


def joint_margin(m: mujoco.MjModel, qs: list[np.ndarray]) -> float:
    """Smallest distance (rad) of a limited arm joint to its limit over the arm configurations qs."""
    ids = [m.joint(j).id for j in ARM_JOINTS]
    lim = m.jnt_limited[ids].astype(bool)
    lo, hi = m.jnt_range[ids].T
    return float(min(np.minimum(q - lo, hi - q)[lim].min() for q in qs))


def path_ok(m: mujoco.MjModel, knots: list[np.ndarray], err: float) -> bool:
    """IK error within IK_TOL and every limited joint >= JOINT_MARGIN from its limit (round 1, place 3010: the wrist at
    its limit while moving over, the box swung 15 deg in the pads)."""
    return err <= IK_TOL and joint_margin(m, knots) >= JOINT_MARGIN


def retreat_knots(m: mujoco.MjModel, d: mujoco.MjData, q: np.ndarray, p: np.ndarray, box_xy: np.ndarray,
                  y0: float, roll: float = 0.0) -> tuple[list[np.ndarray], float, str] | None:
    """Open hand at pinch p (arm q, closing axis y0 along the heading: the pads stand off the box's near and far faces;
    box centre at box_xy) -> clear of the box -> back over the base -> RETRACT_PINCH, every knot `path_ok`; None if no
    candidate is. Candidates in order: straight up RETREAT_UP then back at that height (the storage bench); lift
    RETREAT_LIFTS then slide sideways along the box's long side until the pinch is RETREAT_PAST beyond the box's end
    (the short way first), then back at that height (the kitchen table, where the top-down workspace ends ~2 cm above
    the release pinch)."""
    th = float(base_pose(m, d)[2])
    lat = np.array([-math.sin(th), math.cos(th), 0.0])
    low = to_world(m, d, RETRACT_PINCH)
    off = float((p[:2] - np.asarray(box_xy)[:2]) @ lat[:2])  # pinch beside the box centre along its long side
    sides = sorted((1, -1), key=lambda sd: -sd * off)
    cands = [("up", [p + [0, 0, RETREAT_UP]])]
    cands += [(f"lift {lift:.3f} slide {sd:+d}",
               [p + [0, 0, lift], p + [0, 0, lift] + (sd * (BOX_HALF[1] + RETREAT_PAST) - off) * lat])
              for lift in RETREAT_LIFTS for sd in sides]
    for name, pts in cands:
        last = pts[-1]
        rel_y = float((last - d.body("base_link").xpos) @ lat)
        back = to_world(m, d, (OVER_BASE[0], rel_y, 0.0))
        back[2] = last[2]
        knots, err, qq, prev = [], 0.0, q, p
        for nxt, n in [(w, 4) for w in pts] + [(back, 8), (low, 4)]:
            kk, e = cart_knots(m, d, qq, prev, nxt, y0, y0, n=n, roll0=roll, mid=True)
            knots += kk
            err, qq, prev = max(err, e), kk[-1], nxt
            if not path_ok(m, knots, err):
                break
        else:
            return knots, err, name
    return None


def place_box(m: mujoco.MjModel, d: mujoco.MjData, grid: Grid, surface: dict[str, Any], held_body: str,
              in_hand_pose: dict[str, Any], cb: StepCb = None, phase: Phase = None) -> dict[str, Any]:
    """Put the held box down upright on surface = {name, rect [x0, y0, x1, y1], top_z, approach (x, y, yaw rad)}.
    in_hand_pose (pick's) is recorded in detail; the target uses the in-hand pose measured after the carry -> over-base
    transit. The whole arm path (over, down, release, retreat, retract) is planned before moving over: every knot within
    IK_TOL and JOINT_MARGIN of the joint limits, else the next PLACE_INSETS target, else "unreachable" with the box
    still held. phase(name, d) is called at "over", "released", "retracted".
    Returns {ok, reason ("" | "nothing held" | "blocked" | "unreachable" | "dropped" | "missed"), detail}: "blocked"
    (box still held) when the drive to the approach point stopped on a blocker (stopped_by_blocker)."""
    t: dict[str, float] = {}
    s0 = d.time
    detail: dict[str, Any] = {"timings": t}
    if not holding(m, d, held_body):
        return {"ok": False, "reason": "nothing held", "detail": detail}
    rect, top = [float(v) for v in surface["rect"]], float(surface["top_z"])
    ax, ay, ayaw = surface["approach"]
    t0 = d.time
    nav = go_to_keep_arm(m, d, grid, (ax, ay, ayaw), cb, held_body)
    t["drive_s"] = d.time - t0
    detail["go_to"] = {k: nav.get(k) for k in ("reached", "pos_err_m", "yaw_err_deg", "path_len_m", "contacts", "blocked",
                                                "waypoints")}
    if not nav["reached"]:
        return {"ok": False, "reason": "blocked" if stopped_by_blocker(nav) else "unreachable", "detail": detail}
    detail["in_hand_arrive"] = in_hand(m, d, held_body)
    if not holding(m, d, held_body):
        return {"ok": False, "reason": "dropped", "detail": {**detail, **_end(m, d, held_body, s0, t)}}
    # straight in towards the near edge
    bx, by, th = (float(v) for v in base_pose(m, d))
    e = _edge_dist(bx, by, th, rect)
    t0 = d.time
    if math.isfinite(e) and e > PLACE_GAP:  # as close as the footprint (+ 2 cm) allows on the map, >= PLACE_GAP
        for g in np.arange(e - PLACE_GAP, 0.0, -0.01):
            goal = (bx + g * math.cos(th), by + g * math.sin(th), th)
            if segment_free(grid, (bx, by), goal[:2], th, 0.02):
                drive_base(m, d, goal, settle=0.5, cb=cb)
                break
    t["close_in_s"] = d.time - t0
    bx, by, th = (float(v) for v in base_pose(m, d))
    e = _edge_dist(bx, by, th, rect)
    # arm: carry -> OVER_BASE with the closing axis along the heading (box 6 cm side across the edge)
    t0 = d.time
    y0 = None
    for cand in (th, th + math.pi):
        e_tr = from_carry(m, d, cand, cb)
        if e_tr is not None:
            y0 = cand
            break
    if y0 is None:
        return {"ok": False, "reason": "unreachable", "detail": {**detail, "why": "no transit", **_end(m, d, held_body, s0, t)}}
    # the box pivots a little in the pads during the transit (round 2, place 3010: 9.9 deg at the end of pick, 17 deg
    # here): land and cross the edge with the pose measured now, not the one handed over from pick
    ih = in_hand(m, d, held_body)
    detail["in_hand_place"], detail["in_hand_pick"] = ih, in_hand_pose
    detail["tilt_in_hand_deg"] = tilt_deg(d.body(held_body).xmat)
    rr = np.zeros(9)
    mujoco.mju_quat2Mat(rr, np.asarray(ih["quat"]))
    bz = rr.reshape(3, 3)[:, 2]  # box z axis in the site frame
    level = math.atan(bz[1] / bz[2])  # hand roll about the closing axis (the pads' hinge) that hangs the box level
    detail["level_deg"] = math.degrees(level)
    c, s = math.cos(th), math.sin(th)
    q_seed = arm_ctrl(m, d)
    p_now = d.site(PINCH).xpos.copy()
    # target: the first inset (box footprint inside the near edge) whose whole arm path is reachable: over -> down ->
    # release -> retreat -> retract, every IK knot within IK_TOL, planned BEFORE moving over (2j fix 2: at the
    # kitchen table the top-down workspace ends ~2 cm above the release pinch, round 1 followed 0.56 m IK errors)
    plan = None
    for frac, inset in [(f, i) for f in ROLL_FRACTIONS for i in PLACE_INSETS]:
        roll = frac * level
        rot = top_down_rot(y0, roll)
        half_w = np.abs(rot @ rr.reshape(3, 3)) @ np.asarray(BOX_HALF)  # world half extents of the box as it lands
        centre_off = rot @ np.asarray(ih["offset"])  # box centre minus pinch, world
        along = e + inset + abs(c) * half_w[0] + abs(s) * half_w[1]
        tx, ty = bx + along * c, by + along * s
        lat = abs(s) * half_w[0] + abs(c) * half_w[1] + inset  # lateral position clamped into the rect
        if abs(c) > abs(s):
            ty = float(np.clip(ty, rect[1] + lat, rect[3] - lat))
        else:
            tx = float(np.clip(tx, rect[0] + lat, rect[2] - lat))
        target = np.array([tx - centre_off[0], ty - centre_off[1], top + half_w[2] - centre_off[2]])
        above = target + [0, 0, PLACE_ABOVE]
        # up first (the hand must clear the top's edge), across at the highest reachable height <= PLACE_TRAVEL over
        # `above` (the box swings a little in the fingers while moving), then straight down to `above`
        for dz in np.arange(PLACE_TRAVEL, -1e-9, -0.01):
            high = above + [0, 0, dz]
            qh, eh, rh = ik_mid(m, d, q_seed, high, rot)  # rot includes the roll
            if path_ok(m, [qh], max(eh, rh)):
                break
        # rise on a line out towards the edge: the box's far face stops RISE_GAP short of the near edge (over the
        # base the top-down workspace ends lower than out at the edge)
        hd = np.array([c, s, 0.0])
        r_now = float((p_now - [bx, by, 0.0]) @ hd)
        r_rise = max(r_now, e - abs(c) * half_w[0] - abs(s) * half_w[1] - RISE_GAP - float(centre_off @ hd))
        rise = p_now + (r_rise - r_now) * hd
        rise[2] = max(p_now[2], high[2])
        kn1, e1 = cart_knots(m, d, q_seed, p_now, rise, y0, y0, n=4, roll1=roll, mid=True)
        kn2, e2 = cart_knots(m, d, kn1[-1], rise, high, y0, y0, n=10, roll0=roll, mid=True)
        kn3, e3 = cart_knots(m, d, kn2[-1], high, above, y0, y0, n=4, roll0=roll, mid=True)
        down = target - [0, 0, 0.02]  # aim 2 cm below the landing height; the descent stops at first touch
        kn4, e4 = cart_knots(m, d, kn3[-1], above, down, y0, y0, roll0=roll, mid=True)
        q_t, ep, er = ik_mid(m, d, kn4[-1], target, rot)
        ret = retreat_knots(m, d, q_t, target, np.array([tx, ty]), y0, roll)
        e_over = max(e1, e2, e3, e4, ep, er)
        margin = [joint_margin(m, k) for k in (kn1, kn2, kn3, kn4, [q_t])]  # rise, across, to above, down, release
        detail.setdefault("target_tries", []).append({"inset": inset, "roll_deg": math.degrees(roll),
                                                      "ik_err_over": e_over, "joint_margin": margin,
                                                      "ik_err": [e1, e2, e3, e4, max(ep, er)], "travel_dz": float(dz),
                                                      "retreat": None if ret is None else ret[2]})
        if path_ok(m, kn1 + kn2 + kn3 + kn4 + [q_t], e_over) and ret is not None:
            plan = (inset, roll, target, above, float(dz), kn1 + kn2 + kn3, kn4, centre_off)
            break
    if plan is None:
        return {"ok": False, "reason": "unreachable", "detail": {**detail, "why": "no reachable place + retreat path",
                                                               **_end(m, d, held_body, s0, t)}}
    inset, roll, target, above, dz, kn2, kn, centre_off = plan
    detail.update(target_box_xy=[float(target[0] + centre_off[0]), float(target[1] + centre_off[1])],
                  inset=inset, roll_deg=math.degrees(roll), travel_dz=dz, ik_err_over=max(detail["target_tries"][-1]["ik_err_over"], e_tr))
    follow_knots(m, d, kn2, 5.0, cb)
    run(m, d, SETTLE_S, cb)
    t["over_s"] = d.time - t0
    if phase:
        phase("over", d)
    detail["in_hand_over"] = in_hand(m, d, held_body)
    t0 = d.time
    bid = m.body(held_body).id
    rg = set(robot_geoms(m).tolist())

    def touched(dd: mujoco.MjData) -> bool:
        for cc in dd.contact[: dd.ncon]:
            b1, b2 = m.geom_bodyid[cc.geom1], m.geom_bodyid[cc.geom2]
            if bid in (b1, b2) and cc.geom1 not in rg and cc.geom2 not in rg:
                return True
        return False

    stopped = follow_knots(m, d, kn, 1.875 * (PLACE_ABOVE + 0.02) / DESCENT_V, cb, stop=touched)
    detail["touched"] = stopped
    run(m, d, 0.3, cb)
    t["lower_s"] = d.time - t0
    t0 = d.time
    set_gripper(m, d, GRIP_OPEN, 0.8, cb)
    run(m, d, SETTLE_S, cb)
    if phase:
        phase("released", d)
    # re-plan the retreat from where the hand and the box actually are (the descent stops at first touch)
    ret = retreat_knots(m, d, arm_ctrl(m, d), d.site(PINCH).xpos.copy(), d.body(held_body).xpos.copy(), y0, roll)
    if ret is None:
        return {"ok": False, "reason": "unreachable", "detail": {**detail, "why": "no retreat path after release",
                                                               **_end(m, d, held_body, s0, t)}}
    detail["retreat"], detail["ik_err_retreat"] = ret[2], ret[1]
    follow_knots(m, d, ret[0], 4.0, cb)
    move_arm(m, d, RETRACT_QPOS, cb)
    move_arm(m, d, LOOK_ARM_QPOS, cb)
    t["release_s"] = d.time - t0
    t0 = d.time
    q = base_pose(m, d)
    if math.hypot(q[0] - ax, q[1] - ay) > 0.01:
        drive_base(m, d, (ax, ay, ayaw), settle=0.5, cb=cb)
    t["back_off_s"] = d.time - t0
    run(m, d, 0.5, cb)
    if phase:
        phase("retracted", d)
    bp = d.body(held_body).xpos
    margin = min(bp[0] - rect[0], rect[2] - bp[0], bp[1] - rect[1], rect[3] - bp[1])
    tilt = tilt_deg(d.body(held_body).xmat)
    bottom = float(bp[2]) - float((np.abs(d.body(held_body).xmat.reshape(3, 3)) @ np.asarray(BOX_HALF))[2])
    detail.update(_end(m, d, held_body, s0, t))
    detail.update(final_pos=bp.tolist(), tilt_deg=tilt, centre_margin_m=float(margin),
                  bottom_above_top_m=bottom - top, gripper_open=gripper_open(m, d),
                  arm_retracted=arm_at(m, d, LOOK_ARM_QPOS))
    pc = pad_contacts(m, d, held_body)
    released = not (pc["left_pad"] or pc["right_pad"])
    on = margin >= SCORE_SURFACE_MARGIN and abs(bottom - top) <= SCORE_ON_TOP_TOL and tilt <= SCORE_UPRIGHT_DEG
    ok = bool(released and on)
    return {"ok": ok, "reason": "" if ok else ("missed" if released else "still held"), "detail": detail}
