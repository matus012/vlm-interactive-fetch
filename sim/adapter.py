"""The robot adapter (task 003): `SimRobot` implements `agent.robot_api.RobotAPI` on the seeded MuJoCo house.

The only module the agent side (skills / runtime / scorer) talks to; it owns the MjModel / MjData and every physics
step. All motion goes through actuators (no teleports): navigate = `sim.nav.follow.go_to`, look = `drive_yaw` turns.
The world is the known map (walls + furniture); boxes and blockers are not on it.

navigate(place, via, avoid):
- `via` = a room place name ("living room", ...): path forced through that room's cells (Phase 1 room mask), or a
  door place name ("kitchen south door", ...): the planning grid gets every OTHER door between the same two rooms
  closed (its wall gap stamped occupied) and the path is start -> door centre -> goal (`plan_path(via_xy=...)`,
  two legs). With the other door shut the only connection between those two rooms is the named door, and the door
  centre is a waypoint, so the route goes through it whatever the start side.
- `avoid` = object names: their footprint (rotated box, + AVOID_MARGIN; perceived pose in perceived mode, true pose
  in truth mode) is stamped occupied for this plan only.
  Objects not listed are not on the map; any robot contact with them is in NavResult.contacts.
- Never re-plans around a blocker: the follower's blocked detector (follow.py docstring) stops the base and holds it;
  navigate returns status "blocked" with the photo at that heading. A drive that ends outside the G1-reach
  tolerance without the detector firing is also "blocked" (detail says so). No path -> "no_path".
- Navigating to a door place stops in the doorway centre, yaw across the door.

Contract extensions (documented here, robot_api.py untouched):
- `places()` rows carry no pose (no coordinates leave the adapter). Door rows: kind "door", "room" = the first room
  of the door's `between`, plus "door" (layout id) and "between" (both room names). Room names are VLM room names
  ("living room"), not layout keys.
- `sim_time()`: current sim time d.time (s), for the runtime log (track 2e).
- `support_of` returns "floor", a surface name, or "none" (in the gripper / in the air).
- `snapshot()["arm_retracted"]`: the arm joint targets are exactly LOOK_ARM_QPOS or the Menagerie `retract`
  pose (sim.snapshot; commanded state, no tolerance); "gripper_open": gripper actuator target == GRIP_OPEN.
- `self.seen`: every object that had >= PX_VISIBLE px in any photo this adapter took (pick passes the other seen fetch
  boxes to `pick_box` as `avoid` discs (x, y, BOX_REACH); place marks the same discs on the grid its drive to the
  approach point plans on, R4(b)). `_avoid_discs` gives those discs in the adapter's pose mode; `_seen_discs` is
  the truth version (sim.tasks' pre-grasp check and pregrasp_survey call it; never on a perceived-mode path).
- `SimRobot(seed, ...)` refuses a test-space seed (sim.constants.TEST_SEEDS) unless `sealed_draw=True`.
- One "held" (deviation 18, task 006): `SimRobot.held()` (RobotAPI: skills' preconditions, place, the runtime log) is
  `sim.grasp.holding` (box between the pads). The scorer stays conservative: snapshot["held"] = `held_any()` = a box
  with grasp.holding OR any pad contact, so "robot holds nothing" fails under either meaning; snapshot["holding"] =
  `held()` after the settle (the labeller's lost-box rule reads it). `pad_touch()` = pads in contact per box (the G2
  gates' "both pads" readings, unchanged).
- `SimRobot(..., start=(x, y, yaw rad))`: spawn pose (default layout.spawn); `pick_phase`: optional pick_box phase hook
  (the G2-pick harness, sim/gate2.py, holds the lift there). `self.last_nav`: the raw go_to dict of the last navigate (waypoints, blocked, ...).

Last-stop memory (task 005 fix (d), L1): `self._last_stop` = {"body", "path"} of the last drive (navigate, pick's or
place's approach drive) that the blocked detector stopped with crate / cabinet contact (`stopped_by_blocker`): the
blocker's body and that drive's planned path from where it stopped (the base's closest point on the polyline) to its
goal. Forgotten when a later drive of any skill reaches its goal, or when a push moves that body (>= PUSHED_MIN_M).
Read ONLY by `push` (passed to push_object as stop_path: the body then counts as blocking and must end clear of the
path); navigate never reads it (it never re-plans around a blocker) and it never reaches a result text or the packet
(tests/test_last_stop.py checks both).

Pushed-object map (row 17 addendum 3, task 006): after a push that returns ok ("pushed"), or fails having moved the
object >= PUSHED_MAP_MIN_M (push's own obj_moved_m, the measurement its verdict uses), the object's end footprint is
stamped on `self.grid`, the planner map of every later drive (navigate, pick's and place's drives, push's own
approach), replacing any earlier entry for it; push of that object plans on the map without its own entry. Nothing
else goes on it: a blocker never pushed (or a cabinet, which never moves) stays off, navigate never routes around it.
Footprint: truth mode the true pose + AVOID_MARGIN; perceived mode never the sim pose: sim.push.contact_footprint
(the base's own pose at the last bumper contact of the last push stroke + that stroke's direction: the object sits
squared up in front of the bumper face) at the known object size + PUSHED_MAP_MARGIN. The margin is measured: the
regression's pushes' cover_m (`_push_eval`, report only: how far the true footprint pokes out of the unmargined
estimate) -> worst case, rounded up (value and data in PUSHED_MAP_MARGIN's comment). No contact stroke -> no entry.
Same map for push (row 17 addendum 4, task 006): push_object gets est = `_map_entry` (the pose the map entry has or
would get: truth mode the true pose; perceived push's own contact geometry, else the current entry; never truth), the
map margin of the mode, and runs every blocking / end / reachability test on that footprint (sim.push module doc);
`_map_pushed` stamps the same entry.
Map-blocked memory (row 17 addendum 4, analogue of fix (d)): `self._map_block` = {"plan", "bodies"}, set when a drive
or pick fails on planning: navigate "no_path", pick "unreachable" with no pre-grasp pose / path, place "unreachable"
with no path. "plan"(grid) -> bool re-runs that plan request (navigate / place: plan_path from the base's pose at the
failure to the goal with the request's own extra cells (via door closures, avoid footprints, seen-box discs) and via;
pick: plan_pregrasp_at for the box at the pose pick planned from (located pose in perceived mode) and a planned path
from the base's pose to one of its stand-offs) on a given map; "bodies" = the pushed objects whose map entry alone
stops it (plan fails on the map, succeeds on the map without that entry). Cleared by any drive that reaches its goal
and any successful pick. Read ONLY by `push` (need_plan for an object in bodies; sim.push module doc); never by
navigate, never in a result text or the packet (tests/test_map_block.py).
Re-locate (task 006, 5b finding): in perceived mode a push that moved the object >= PUSHED_MIN_M drops EVERY located
box pose (adapter `_located`; the result's detail["dropped_located"] tells Skills to drop its `located` too), so pick
says "failed: not located" until the box is located again: a pushed crate can shove a located box (005 8204 / 8208:
0.28 m). All boxes, not only those near the push sweep: simplest, and a stale pose is never used.

Pose mode (task 006, Phase 5): `SimRobot(..., poses="perceived" | "truth")`, default "perceived".
- Every photo (look / navigate's photo) is one render: RGB + depth + segmentation, plus the mast camera pose.
- perceived: a fetch box's px in `Photo.visible` (and so `seen`) = its HSV colour-mask px (sim.perception); after each
  photo set (a look's 4, or navigate's 1) `sim.perception.locate_photos` runs on the set for every fetch box. A box
  in view (>= IN_VIEW_PX mask px in one photo and a fit) gets its perceived pose: `locate` returns it (room from the
  perceived xy, support from the perceived bottom z vs the known surface tops: sim.snapshot.support_of), `pick` plans
  and grasps from it (grasp.pick_box pose_fn), and the seen-box avoid discs / navigate avoid footprints use the latest
  perceived pose of each box (a picked box is forgotten until it is seen again).
- Truth still read in perceived mode, and why: `_locate_eval` (truth pose + segmentation px of the located box, into
  the step meta for the report only: never a result string or the packet); crate / cabinet (not fetch boxes, not
  perceived in the brief): their visibility is segmentation px and `locate` on them is the truth path
  (`_locate_unperceived`), `push` reads the blocker's true pose for its door (sim.push); pick_box's outcome
  measurements after contact (lift height, holding, in-hand pose: grasp.pick_box doc); `sim_ok` (simulator health).
  tests/test_truthfree.py checks this statically (call graph from locate / pick / navigate / look / photo) and with
  truth sabotaged at run time.
- truth: the Phase 2-4 behaviour (segmentation px, true poses), kept for the Phase 6 fallback the brief allows.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from agent.robot_api import PX_VISIBLE, ActResult, Located, NavResult, Photo, Pose
from sim.body import _ids, base_pose, run
from sim.constants import BOX_HALF, CABINET_HALF, CRATE_HALF, ROBOT_RADIUS, refuse_test_seed
from sim.look import HEADINGS_DEG, LOOK_ARM_QPOS, drive_yaw, move_arm, render_look, set_arm
from sim.nav.astar import plan_path
from sim.nav.follow import CLEARANCE, DISC_MARGIN, go_to
from sim.perception import colour_mask, locate_photos
from sim.robot import ARM_JOINTS, CAM_H, CAM_W, GRIPPER_ACT, MAST_CAM, camera_intrinsics, robot_spec
from sim.snapshot import snapshot as _snapshot
from sim.snapshot import support_of as snapshot_support_of
from sim.world.generate import BOX_REACH, World, furniture_rect, generate, load_places
from sim.world.house import GRIP_OPEN, load_house
from sim.world.occupancy import Grid, known_map

START_SETTLE_S = 0.5
AVOID_MARGIN = 0.05  # m around an avoided object's footprint (on top of the planner's robot inflation)
CLOSED_DOOR_DEPTH = max(CABINET_HALF[0], CRATE_HALF[0])  # m along the passage, each side of the wall line
PUSHED_MAP_MIN_M = 0.03  # row 17 addendum 3: a "stuck" push that moved the object this far also maps it
PUSHED_MAP_MARGIN = 0.14  # m around the perceived-mode pushed footprint (module doc). Measured 2026-11-13 (track 5c,
# mfpc, gate2 push-crate + carry-push, 20 pushes, all "pushed"): cover_m (true footprint outside the unmargined
# estimate) median 0.015, worst 0.134 (carry-push 1303: side push A 0.84 / delta 0.10, crate turned 9.6 deg against the
# bumper and slid 0.10 m along it), next 0.112 (1309, same plan); through / long / side A 1.1 pushes <= 0.027 -> 0.14
NAV_BACK_OFF = 0.25  # m: navigate backs off this far along its path once blocked (2j fix 5; not a re-plan)
BLOWUP_V = 50.0  # m/s or rad/s: any qvel above this = blow-up
BLOWUP_POS = 2.0  # m outside the house rectangle = blow-up


def _yaw_of(xmat: np.ndarray) -> float:
    return math.atan2(float(xmat[3]), float(xmat[0]))


class SimRobot:
    def __init__(self, seed: int, blocker: str = "none", photo_dir: str | Path | None = None,
                 world: World | None = None, start: tuple[float, float, float] | None = None, *,
                 sealed_draw: bool = False, poses: str = "perceived") -> None:
        refuse_test_seed(seed, sealed_draw)
        assert poses in ("perceived", "truth"), poses
        self.poses = poses
        self.world = world if world is not None else generate(seed=seed, blocker=blocker, sealed_draw=sealed_draw)
        self.layout = self.world.layout
        self.m, self.d = load_house(self.world, start)
        set_arm(self.m, self.d, LOOK_ARM_QPOS)  # start state: look arm pose (qpos + ctrl), then settle
        self.d.ctrl[self.m.actuator(GRIPPER_ACT).id] = GRIP_OPEN
        run(self.m, self.d, START_SETTLE_S)
        self._known = known_map(self.layout)  # walls + furniture
        self._pushed: dict[str, tuple[float, float, float, float, float]] = {}  # module doc: pushed-object map
        self.grid = self._known  # the planner map of every drive: known map + pushed objects' footprints
        self.photo_dir = Path(photo_dir) if photo_dir is not None else None
        if self.photo_dir is not None:
            self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.n_photos = 0
        self.seen: set[str] = set()
        self._frames: list[dict[str, Any]] = []  # current photo set (last look / photo): rgb, depth, K, camera pose
        self._cur_seg: list[dict[str, int]] = []  # segmentation px per object, per photo of the current set (truth)
        self._cur: dict[str, dict[str, Any]] = {}  # perceived mode: locate_photos result per fetch box, current set
        self._seen_pose: dict[str, tuple[float, float, float]] = {}  # perceived (x, y, yaw) of each box last in view
        self._located: dict[str, Pose] = {}  # pose of each box at its last successful locate (what pick uses)
        self.in_hand: dict | None = None
        self.held_body: str | None = None
        self.last_nav: dict[str, Any] = {}
        self._last_stop: dict[str, Any] | None = None  # module doc: read only by push
        self._map_block: dict[str, Any] | None = None  # module doc: read only by push
        self.pick_phase: Any = None  # optional pick_box phase(name, d) hook (gate harness: hold / measure at "lift")
        self._bad = False
        # names: VLM name <-> body
        self.body_of: dict[str, str] = {}
        self._objects: dict[str, dict] = {}
        for b in self.world.boxes:
            n = f"{b['colour']} box"
            self.body_of[n] = b["name"]
            self._objects[n] = {"class": "box", "colour": b["colour"]}
        if self.world.blocker["variant"] != "none":
            n = self.world.blocker["body"]
            self.body_of[n] = n
            self._objects[n] = {"class": n, "colour": self.world.labels[n]["colour"]}
        self.name_of = {v: k for k, v in self.body_of.items()}
        self._geoms = {n: np.nonzero(self.m.geom_bodyid == self.m.body(b).id)[0] for n, b in self.body_of.items()}
        self._seg_lut = np.full(self.m.ngeom + 1, -1, np.int32)  # geom id -> object index (index -1 -> none)
        self._names = list(self.body_of)
        for i, n in enumerate(self._names):
            self._seg_lut[self._geoms[n]] = i
        # places / rooms
        self._places = {p["name"]: p for p in load_places()}
        self.room_name = {}  # layout room key -> VLM room name (its first room point's place name)
        for p in self._places.values():
            if p["kind"] == "room":
                self.room_name.setdefault(p["room"], p["name"])
        self._surfaces = {f["surface"]: f for f in self.layout["furniture"] if f.get("surface")}
        robot = robot_spec()
        key, joints = list(robot.key("retract").qpos), [j.name for j in robot.joints]
        self._retract = np.array([key[joints.index(j)] for j in ARM_JOINTS])
        self._arm_act = _ids(self.m, ARM_JOINTS)[2]
        self._K = camera_intrinsics(self.m)
        self._cam = self.m.camera(MAST_CAM).id

    # ---- world facts --------------------------------------------------------------------------------------------
    def places(self) -> dict[str, dict]:
        out = {}
        for n, p in self._places.items():
            row = {"kind": p["kind"], "room": self.room_name[p["room"]]}
            if p["kind"] == "door":
                row["door"] = p["door"]
                row["between"] = [self.room_name[r] for r in self.layout["doors"][p["door"]]["between"]]
            out[n] = row
        return out

    def objects(self) -> dict[str, dict]:
        return {n: dict(v) for n, v in self._objects.items()}

    def sim_time(self) -> float:
        return float(self.d.time)

    # ---- sim health ---------------------------------------------------------------------------------------------
    def sim_ok(self) -> bool:
        d = self.d
        if self._bad:
            return False
        xs = [v for r in self.layout["rooms"].values() for v in (r["rect"][0], r["rect"][2])]
        ys = [v for r in self.layout["rooms"].values() for v in (r["rect"][1], r["rect"][3])]
        p = d.xpos[1:]
        bad = (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
               or float(np.abs(d.qvel).max()) > BLOWUP_V
               or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
               or (p[:, 0] < min(xs) - BLOWUP_POS).any() or (p[:, 0] > max(xs) + BLOWUP_POS).any()
               or (p[:, 1] < min(ys) - BLOWUP_POS).any() or (p[:, 1] > max(ys) + BLOWUP_POS).any())
        self._bad = bool(bad)
        return not self._bad

    # ---- photos -------------------------------------------------------------------------------------------------
    def _seg_px(self, seg: np.ndarray) -> dict[str, int]:
        """TRUTH: segmentation px per object (> 0 only)."""
        ids = self._seg_lut[np.where(seg >= 0, seg, self.m.ngeom)]
        cnt = np.bincount(ids[ids >= 0], minlength=len(self._names))
        return {self._names[i]: int(c) for i, c in enumerate(cnt) if c > 0}

    def _shoot(self, r: mujoco.Renderer, index: int, heading_deg: float) -> Photo:
        rgb, depth, seg = render_look(r, self.d)
        seg_px = self._seg_px(seg)  # truth mode: the visibility; perceived: eval log + crate / cabinet (module doc)
        self._cur_seg.append(seg_px)
        if self.poses == "truth":
            visible = seg_px
        else:
            visible = {n: px for n, px in seg_px.items() if self._objects[n]["class"] != "box"}
            for n, o in self._objects.items():
                if o["class"] == "box" and (px := int(colour_mask(rgb, o["colour"]).sum())) > 0:
                    visible[n] = px
            self._frames.append({"rgb": rgb, "depth": depth, "K": self._K, "cam_pos": self.d.cam_xpos[self._cam].copy(),
                                 "cam_mat": self.d.cam_xmat[self._cam].reshape(3, 3).copy()})
        self.seen |= {n for n, c in visible.items() if c >= PX_VISIBLE}
        path = None
        if self.photo_dir is not None:
            from PIL import Image

            self.n_photos += 1
            path = str(self.photo_dir / f"{self.n_photos:04d}_h{int(round(heading_deg)) % 360:03d}.png")
            Image.fromarray(rgb).save(path)
        return Photo(index=index, heading_deg=heading_deg, path=path, visible=visible)

    def _heading_deg(self) -> float:
        return round(math.degrees(float(base_pose(self.m, self.d)[2])) % 360.0, 1)

    def _new_set(self) -> None:
        self._frames, self._cur_seg, self._cur = [], [], {}

    def _perceive(self) -> None:
        """Perceived mode, after a photo set: locate_photos for every fetch box; the in-view ones update _seen_pose."""
        if self.poses == "truth":
            return
        for n, o in self._objects.items():
            if o["class"] == "box":
                self._cur[n] = res = locate_photos(self._frames, o["colour"])
                if res["in_view"]:
                    self._seen_pose[n] = (float(res["pos"][0]), float(res["pos"][1]), float(res["yaw"]))

    def photo(self) -> Photo:
        """One photo at the current heading; it becomes the current photo set."""
        self._new_set()
        r = mujoco.Renderer(self.m, CAM_H, CAM_W)
        try:
            ph = self._shoot(r, 1, self._heading_deg())
        finally:
            r.close()
        self._perceive()
        return ph

    def look(self) -> list[Photo]:
        """Look arm pose (unless a box is held: the carry pose stays), then turn to each absolute heading by driving
        the base yaw (drive_yaw: speed-capped joint_th setpoint + settle) and take one photo there."""
        if self.held() is None and not np.allclose(self.d.ctrl[self._arm_act], LOOK_ARM_QPOS, atol=1e-6):
            move_arm(self.m, self.d, LOOK_ARM_QPOS)
        out = []
        self._new_set()
        r = mujoco.Renderer(self.m, CAM_H, CAM_W)
        try:
            for i, h in enumerate(HEADINGS_DEG):
                drive_yaw(self.m, self.d, math.radians(h))
                out.append(self._shoot(r, i + 1, h))
        finally:
            r.close()
        self._perceive()
        self.sim_ok()
        return out

    # ---- locate -------------------------------------------------------------------------------------------------
    def locate(self, obj: str) -> Located:
        """The object in the current photo set (module doc). Perceived mode, fetch box: sim.perception's pose."""
        if self.poses == "truth":
            return self._locate_truth(obj)
        if self._objects[obj]["class"] != "box":
            return self._locate_unperceived(obj)
        res = self._cur.get(obj, {"in_view": False, "mask_px": 0})
        meta: dict[str, Any] = {"max_px": int(res["mask_px"]), "poses": "perceived", "perceived": None,
                                "eval": self._locate_eval(obj, res)}
        if not res["in_view"]:
            return Located(None, meta=meta)
        x, y, z = (float(v) for v in res["pos"])
        pose = Pose(x, y, z, float(res["yaw"]))
        self._located[obj] = pose
        meta["perceived"] = {"pos": [x, y, z], "yaw": pose.yaw, "size": [float(v) for v in res["size"]],
                             "n_inliers": int(res["n_inliers"]), "photo": int(res["photo"])}
        support = snapshot_support_of(self.world, x, y, z - BOX_HALF[2]) or "none"
        return Located(pose, self._room_at(x, y), support, meta)

    def _locate_truth(self, obj: str) -> Located:
        """Truth mode (Phases 2-4): in view = >= PX_VISIBLE segmentation px in a current photo; the true pose."""
        px = max((s.get(obj, 0) for s in self._cur_seg), default=0)
        if px < PX_VISIBLE:
            return Located(None, meta={"max_px": px})
        self._located[obj] = pose = self.true_pose(obj)
        return Located(pose, self.room_of(obj).replace("_", " "), self.support_of(obj), {"max_px": px})

    def _locate_unperceived(self, obj: str) -> Located:
        """Crate / cabinet in perceived mode: not fetch boxes, not perceived in the brief -> the truth path."""
        return self._locate_truth(obj)

    def _locate_eval(self, obj: str, res: dict[str, Any]) -> dict[str, Any]:
        """EVALUATION ONLY (step meta for the report; never a result string or the packet): the true pose, the
        segmentation px of the current set, and the perceived pose's error (yaw mod 180 deg)."""
        t = self.true_pose(obj)
        out: dict[str, Any] = {"truth": {"pos": [t.x, t.y, t.z], "yaw": t.yaw},
                               "seg_px": max((s.get(obj, 0) for s in self._cur_seg), default=0),
                               "room": self.room_of(obj), "support": self.support_of(obj)}
        if res.get("pos") is not None:
            p = np.asarray(res["pos"], float)
            e = (float(res["yaw"]) - t.yaw) % math.pi
            cam = self._frames[int(res["photo"])]["cam_pos"]
            out.update(pos_err_m=float(np.linalg.norm(p - [t.x, t.y, t.z])), xy_err_m=math.hypot(p[0] - t.x, p[1] - t.y),
                       yaw_err_deg=math.degrees(min(e, math.pi - e)), dist_m=math.hypot(t.x - cam[0], t.y - cam[1]))
        return out

    # ---- navigate -----------------------------------------------------------------------------------------------
    def _door_gap(self, door_id: str, depth: float = 0.0) -> tuple[float, float, float, float]:
        """The door's wall gap; depth > 0: grown to +-depth along the passage (at least the wall's half thickness)."""
        dd = self.layout["doors"][door_id]
        (axis, c), = dd["wall"].items()
        hw, w = max(self.layout["wall"]["thickness"] / 2, depth), self.layout["door_width"] / 2
        return (c - hw, dd["center"] - w, c + hw, dd["center"] + w) if axis == "x" else \
            (dd["center"] - w, c - hw, dd["center"] + w, c + hw)

    def _stamp_rect(self, occ: np.ndarray, r: tuple[float, float, float, float]) -> None:
        g = self.grid
        i0, j0 = g.world_to_cell(r[0] + 1e-9, r[1] + 1e-9)
        i1, j1 = g.world_to_cell(r[2] - 1e-9, r[3] - 1e-9)
        occ[max(j0, 0): j1 + 1, max(i0, 0): i1 + 1] = True

    def _stamp_object(self, occ: np.ndarray, name: str) -> None:
        """Cells whose centre lies inside the object's rotated xy footprint grown by AVOID_MARGIN (perceived mode:
        the box's latest perceived pose; none yet -> nothing stamped)."""
        hx, hy = (float(v) for v in self._half(name)[:2])
        if self.poses == "truth":
            b = self.d.body(self.body_of[name])
            x, y, yaw = float(b.xpos[0]), float(b.xpos[1]), _yaw_of(b.xmat)
        elif name in self._seen_pose:
            x, y, yaw = self._seen_pose[name]
        else:
            return
        self._stamp_rot(occ, x, y, yaw, hx + AVOID_MARGIN, hy + AVOID_MARGIN)

    def _stamp_rot(self, occ: np.ndarray, x: float, y: float, yaw: float, hx: float, hy: float) -> None:
        """Cells whose centre lies inside the rectangle centred (x, y), yaw, half sides hx, hy."""
        g = self.grid
        ny, nx = occ.shape
        cx = g.origin[0] + (np.arange(nx) + 0.5) * g.res
        cy = g.origin[1] + (np.arange(ny) + 0.5) * g.res
        X, Y = np.meshgrid(cx - x, cy - y)
        u, v = X * math.cos(yaw) + Y * math.sin(yaw), -X * math.sin(yaw) + Y * math.cos(yaw)
        occ |= (np.abs(u) <= hx) & (np.abs(v) <= hy)

    def _map_grid(self, skip: str | None = None) -> Grid:
        """The known map + every pushed object's footprint (module doc) but skip's."""
        occ = self._known.occ.copy()
        for n, f in self._pushed.items():
            if n != skip:
                self._stamp_rot(occ, *f)
        return Grid(self._known.origin, self._known.res, occ)

    def _map_entry(self, obj: str, det: dict[str, Any]) -> tuple[float, float, float] | None:
        """(x, y, yaw) of obj's map entry after push detail det (module doc). Truth mode: the true pose. Perceived:
        push's own geometry (sim.push.contact_footprint: last bumper contact base pose + push direction, known size),
        else its current entry, else None (never truth)."""
        from sim.push import contact_footprint

        if self.poses == "truth":
            t = self.true_pose(obj)
            return t.x, t.y, t.yaw
        if det.get("last_contact") is not None:
            return contact_footprint(det["last_contact"], tuple(float(v) for v in self._half(obj)[:2]))
        e = self._pushed.get(obj)
        return None if e is None else e[:3]

    def _map_margin(self) -> float:
        return AVOID_MARGIN if self.poses == "truth" else PUSHED_MAP_MARGIN

    def _map_pushed(self, obj: str, det: dict[str, Any]) -> None:
        """Pushed-object map (module doc): put obj's end footprint on the planner map (replacing its entry): the
        `_map_entry` pose, the known size + `_map_margin` (perceived PUSHED_MAP_MARGIN, truth AVOID_MARGIN); no entry
        (perceived, no contact stroke, not mapped) -> nothing added."""
        e = self._map_entry(obj, det)
        if e is None:
            return
        hx, hy = (float(v) for v in self._half(obj)[:2])
        (x, y, yaw), mg = e, self._map_margin()
        self._pushed[obj] = (x, y, yaw, hx + mg, hy + mg)
        self.grid = self._map_grid()
        det["map_footprint"] = {"pose": [x, y, yaw], "half": [hx + mg, hy + mg]}

    def _half(self, name: str) -> np.ndarray:
        g = int(self._geoms[name][0])
        return self.m.geom_size[g]

    def _nav_grid(self, via: str | None, avoid: list[str] | tuple[str, ...]) -> tuple[Grid, Any, Any]:
        occ = self.grid.occ.copy()
        via_cells, via_xy = None, None
        if via is not None:
            p = self._places[via]
            if p["kind"] == "door":
                rooms = set(self.layout["doors"][p["door"]]["between"])
                for k, dd in self.layout["doors"].items():
                    if k != p["door"] and set(dd["between"]) == rooms:
                        # closed as deep as a blocker standing in it (2j fix 5b: the detour path hugged the wall
                        # past the cabinet, which stands 0.20 m out of each wall face, and scraped it)
                        self._stamp_rect(occ, self._door_gap(k, CLOSED_DOOR_DEPTH))
                via_xy = (float(p["pose"][0]), float(p["pose"][1]))
            elif p["kind"] == "room":
                via_cells = np.zeros_like(occ)
                self._stamp_rect(via_cells, tuple(self.layout["rooms"][p["room"]]["rect"]))
            else:
                raise ValueError(f"via must be a room or a door: {via}")
        for n in avoid:
            self._stamp_object(occ, n)
        return Grid(self.grid.origin, self.grid.res, occ), via_cells, via_xy

    def _vlm_key(self, key: str) -> str:
        other, robot = key.split("|", 1)
        for n, gs in self._geoms.items():
            if other.isdigit() and int(other) in gs or other == self.body_of[n]:
                return f"{n}|{robot}"
        return key

    def navigate(self, place: str, via: str | None = None, avoid: list[str] | tuple[str, ...] = ()) -> NavResult:
        if not self.sim_ok():
            return NavResult("blocked", None, 0.0, None, {}, "sim not ok before navigate")
        p = self._places[place]
        grid, via_cells, via_xy = self._nav_grid(via, avoid)
        goal = (float(p["pose"][0]), float(p["pose"][1]), math.radians(float(p["pose"][2])))
        held = self.held()
        ign = tuple(int(g) for g in self._geoms[held]) if held else ()
        q0 = base_pose(self.m, self.d)
        r = go_to(self.m, self.d, grid, goal, via=via_cells, via_xy=via_xy, arm_look=held is None, ignore_geoms=ign,
                  back_off=NAV_BACK_OFF)
        self.last_nav = r
        self._note_drive(r)
        if r["waypoints"] is None:
            self._note_plan(self._path_plan(q0, goal, grid.occ & ~self.grid.occ, via_cells, via_xy))
        contacts = {self._vlm_key(k): v for k, v in r["contacts"].items()}
        ok = self.sim_ok()
        if r["waypoints"] is None:
            status, detail = "no_path", "no path on the known map"
        elif r["blocked"] is not None:
            b = r["blocked"]
            status, detail = "blocked", f"{b['why']}: lag {b['lag_m']:.3f} m, base contact {b['bump_n']:.1f} N"
        elif not r["reached"]:
            status, detail = "blocked", f"goal not reached: pos {r['pos_err_m']:.3f} m, yaw {r['yaw_err_deg']:.1f} deg"
        else:
            status, detail = "reached", ""
        if not ok:
            status, detail = "blocked", "sim not ok (NaN / blow-up) " + detail
        return NavResult(status=status, photo=self.photo() if ok else None, sim_s=float(r["sim_s"]),
                         path_len_m=r["path_len_m"], contacts=contacts, detail=detail)

    def _note_drive(self, nav: dict[str, Any]) -> None:
        """Update the last-stop memory (module doc) from one go_to result: reached -> forget; stopped by the
        detector on a crate / cabinet -> remember it and the path from the stop to the goal; else unchanged."""
        from sim.grasp import BLOCKERS, stopped_by_blocker

        if nav.get("reached"):
            self._last_stop = self._map_block = None
        elif nav.get("waypoints") and stopped_by_blocker(nav):
            pts = np.array(nav["waypoints"], dtype=float)
            q = np.array(nav["blocked"]["pose"][:2])
            a, b = (pts[:-1], pts[1:]) if len(pts) > 1 else (pts, pts)
            ab = b - a
            t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
            k = int(np.argmin(np.hypot(*(a + t[:, None] * ab - q).T)))
            body = next(k_.split("|")[0] for k_ in nav["contacts"] if k_.split("|")[0] in BLOCKERS)
            self._last_stop = {"body": body, "path": np.vstack([a[k] + t[k] * ab[k], b[k:]])}

    def _path_plan(self, start: Any, goal: Any, extra: np.ndarray, via: np.ndarray | None = None,
                   via_xy: Any = None) -> Callable[[Grid], bool]:
        """A drive's plan request (module doc, map-blocked memory): grid -> plan_path start -> goal (go_to's planner) on
        the grid + the request's extra cells exists."""
        s, g = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
        return lambda grid: plan_path(grid.occ | extra, grid.origin, grid.res, s, g, ROBOT_RADIUS, CLEARANCE + DISC_MARGIN,
                                      via_cells=via, via_xy=via_xy) is not None

    def _note_plan(self, plan: Callable[[Grid], bool]) -> None:
        """Map-blocked memory (module doc): a drive / pick failed on planning; remember its plan request and the pushed
        objects whose map entry alone stops it (none if the plan passes on the map: not a map failure)."""
        self._map_block = None if plan(self.grid) else \
            {"plan": plan, "bodies": {n for n in self._pushed if plan(self._map_grid(n))}}

    # ---- manipulation (other tracks' modules, imported lazily) --------------------------------------------------
    def _door_dict(self, x: float, y: float) -> dict:
        """The layout door nearest (x, y), with its place name and pose."""
        best = None
        for p in self._places.values():
            if p["kind"] == "door":
                dist = math.hypot(p["pose"][0] - x, p["pose"][1] - y)
                if best is None or dist < best[0]:
                    best = (dist, p)
        p = best[1]
        dd = self.layout["doors"][p["door"]]
        (axis, c), = dd["wall"].items()
        return {"id": p["door"], "name": p["name"], "axis": axis, "line": float(c), "center": float(dd["center"]),
                "width": float(self.layout["door_width"]), "between": list(dd["between"]),
                "pose": (float(p["pose"][0]), float(p["pose"][1]), math.radians(float(p["pose"][2])))}

    def _act(self, res: dict) -> ActResult:
        if not self.sim_ok():
            return ActResult(False, "sim error", dict(res.get("detail", {})))
        return ActResult(bool(res["ok"]), str(res.get("reason", "")), dict(res.get("detail", {})))

    def _avoid_discs(self, skip: set[str]) -> list[tuple[float, float, float]]:
        """pick_box / mark_discs avoid discs in the adapter's pose mode (perceived: each box's latest perceived pose)."""
        if self.poses == "truth":
            return self._seen_discs(skip)
        return [(x, y, BOX_REACH) for n, (x, y, _) in sorted(self._seen_pose.items()) if n not in skip]

    def _seen_discs(self, skip: set[str]) -> list[tuple[float, float, float]]:
        """TRUTH: (x, y, footprint radius BOX_REACH) of every seen fetch box not in skip (truth mode; sim.tasks)."""
        out = []
        for n in sorted(self.seen - skip):
            if self._objects[n]["class"] == "box":
                b = self.d.body(self.body_of[n])
                out.append((float(b.xpos[0]), float(b.xpos[1]), BOX_REACH))
        return out

    def pick(self, obj: str) -> ActResult:
        if not self.sim_ok():
            return ActResult(False, "sim error")
        if obj not in self.body_of or self._objects[obj]["class"] != "box":
            return ActResult(False, "not a box")
        if self.held() is not None:
            return ActResult(False, "hand full")
        from sim.grasp import mark_discs, pick_box, plan_pregrasp_at, tilt_deg, true_box_pose

        body = self.body_of[obj]
        if self.poses == "truth":
            if tilt_deg(self.d.body(body).xmat) > 10.0:
                return ActResult(False, "unreachable", {"why": "box not upright"})
            pose_fn = lambda: true_box_pose(self.d, body)  # noqa: E731
        else:
            if obj not in self._located:
                return ActResult(False, "not located")
            p = self._located[obj]
            pose_fn = lambda: (np.array([p.x, p.y, p.z]), p.yaw)  # noqa: E731
        q0, avoid = base_pose(self.m, self.d), self._avoid_discs({obj})
        res = pick_box(self.m, self.d, self.grid, body, avoid, phase=self.pick_phase, pose_fn=pose_fn)
        for nav in res.get("detail", {}).get("go_to", []):
            self._note_drive(nav)
        out = self._act(res)
        if out.reason == "unreachable" and res["detail"].get("why") == "no pre-grasp pose / path":
            bp, axis = pose_fn()  # the pick target as pick planned it (perceived: the located pose)
            far = [a for a in avoid if math.hypot(a[0] - bp[0], a[1] - bp[1]) > 0.05]
            extra = mark_discs(self.grid, far).occ & ~self.grid.occ  # pick's drive grid: the other boxes' discs
            self._note_plan(lambda g: any(self._path_plan(q0, c["standoff"], extra)(g)
                                          for c in plan_pregrasp_at(self.m, self.d, g, bp, axis, avoid)))
        if out.ok:
            self._map_block = None
            det = res.get("detail", {})
            self.in_hand = det.get("in_hand_carry") or det.get("in_hand")  # pose actually in hand when place starts (2b)
            self.held_body = self.body_of[obj]
            self._seen_pose.pop(obj, None)
            self._located.pop(obj, None)
        return out

    def place(self, surface: str) -> ActResult:
        if not self.sim_ok():
            return ActResult(False, "sim error")
        held = self.held()
        if held is None:
            return ActResult(False, "nothing held")
        if surface not in self.layout["place_surfaces"]:
            return ActResult(False, "not a place surface")
        try:
            from sim.grasp import mark_discs, place_box
        except ImportError:
            return ActResult(False, "not implemented")
        f, p = self._surfaces[surface], self._places[surface]
        surf = {"name": surface, "rect": list(furniture_rect(f)), "top_z": float(f["box"][4]),
                "approach": (float(p["pose"][0]), float(p["pose"][1]), math.radians(float(p["pose"][2])))}
        grid = mark_discs(self.grid, self._avoid_discs({held}))  # R4(b): the drive to the approach avoids seen boxes
        q0 = base_pose(self.m, self.d)
        res = place_box(self.m, self.d, grid, surf, self.body_of[held], self.in_hand or {})
        if "go_to" in res.get("detail", {}):
            self._note_drive(res["detail"]["go_to"])
            if res["detail"]["go_to"]["waypoints"] is None:
                self._note_plan(self._path_plan(q0, surf["approach"], grid.occ & ~self.grid.occ))
        out = self._act(res)
        if out.ok:
            self.in_hand, self.held_body = None, None
        return out

    def push(self, obj: str) -> ActResult:
        if not self.sim_ok():
            return ActResult(False, "sim error")
        if obj not in self.body_of:
            return ActResult(False, "unknown object")
        try:
            from sim.push import push_object
        except ImportError:
            return ActResult(False, "not implemented")
        from sim.push import PUSHED_MIN_M

        b = self.d.body(self.body_of[obj])
        held = self.held()
        hg = tuple(int(g) for g in self._geoms[held]) if held else ()
        last = self._last_stop
        mine = last is not None and last["body"] == self.body_of[obj]
        mb = self._map_block
        res = push_object(self.m, self.d, self._map_grid(obj), self.body_of[obj],
                          self._door_dict(float(b.xpos[0]), float(b.xpos[1])), held_geoms=hg,
                          stop_path=last["path"] if mine else None,
                          est=lambda det: self._map_entry(obj, det) if obj in self._pushed or "last_contact" in det else None,
                          map_margin=self._map_margin(), need_plan=mb["plan"] if mb is not None and obj in mb["bodies"] else None)
        det = res["detail"]
        moved = det.get("obj_moved_m", 0.0)
        if mine and moved >= PUSHED_MIN_M:
            self._last_stop = None
        if res["ok"] or moved >= PUSHED_MAP_MIN_M:
            self._map_pushed(obj, det)
            if "map_footprint" in det:
                det["map_eval"] = self._push_eval(obj, det["map_footprint"])
        if moved >= PUSHED_MIN_M and self.poses != "truth":  # it moved things: located poses are stale (module doc)
            det["dropped_located"] = sorted(self._located)
            self._located.clear()
        return self._act(res)

    def _push_eval(self, obj: str, fp: dict[str, Any]) -> dict[str, Any]:
        """EVALUATION ONLY (push step meta for the report, never a result string, the map or the packet): the true end
        pose and how far the true footprint pokes out of the mapped rectangle without its margin (cover_m: the margin
        the map entry needed to hold it; <= 0 = held)."""
        t = self.true_pose(obj)
        (x, y, yaw), (hx, hy) = fp["pose"], self._half(obj)[:2]
        tc = np.array([[t.x + c * sx * hx - s_ * sy * hy, t.y + s_ * sx * hx + c * sy * hy]
                       for c, s_ in [(math.cos(t.yaw), math.sin(t.yaw))] for sx in (-1, 1) for sy in (-1, 1)])
        rel = tc - [x, y]
        u, v = rel @ [math.cos(yaw), math.sin(yaw)], rel @ [-math.sin(yaw), math.cos(yaw)]
        return {"truth": [t.x, t.y, t.yaw], "xy_err_m": math.hypot(t.x - x, t.y - y),
                "cover_m": float(max((np.abs(u) - hx).max(), (np.abs(v) - hy).max()))}

    def pad_touch(self) -> dict[str, int]:
        """Fetch box -> number of finger pads (left_pad / right_pad bodies) with any contact on it (boxes touching
        none left out)."""
        lp, rp = self.m.body("left_pad").id, self.m.body("right_pad").id
        touch: dict[int, set[int]] = {}
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            for pad, other in ((b1, b2), (b2, b1)):
                if pad in (lp, rp):
                    touch.setdefault(other, set()).add(pad)
        return {n: len(touch[self.m.body(b).id]) for n, b in self.body_of.items()
                if self._objects[n]["class"] == "box" and self.m.body(b).id in touch}

    def held(self) -> str | None:
        """The box sim.grasp.holding says is between the pads, else None (deviation 18)."""
        from sim.grasp import holding

        return next((n for n, b in self.body_of.items() if self._objects[n]["class"] == "box"
                     and holding(self.m, self.d, b)), None)

    def held_any(self) -> str | None:
        """Scorer's conservative "held": held(), else a box any pad touches (deviation 18)."""
        return self.held() or next(iter(self.pad_touch()), None)

    # ---- truth --------------------------------------------------------------------------------------------------
    def true_pose(self, obj: str) -> Pose:
        b = self.d.body(self.body_of[obj])
        return Pose(float(b.xpos[0]), float(b.xpos[1]), float(b.xpos[2]), _yaw_of(b.xmat))

    def room_of(self, obj: str) -> str:
        p = self.d.body(self.body_of[obj]).xpos
        return self._room_at(float(p[0]), float(p[1]))

    def _room_at(self, x: float, y: float) -> str:
        for k, r in self.layout["rooms"].items():
            x0, y0, x1, y1 = r["rect"]
            if x0 <= x < x1 and y0 <= y < y1:
                return self.room_name[k]
        return "none"

    def _bottom_z(self, obj: str) -> float:
        b = self.d.body(self.body_of[obj])
        return float(b.xpos[2] - np.abs(b.xmat.reshape(3, 3)[2]) @ self._half(obj))

    def support_of(self, obj: str) -> str:
        """sim.snapshot.support_of on the object's lowest point and centre; "none" if held, in the air or leaning."""
        if self.held() == obj:
            return "none"
        x, y = (float(v) for v in self.d.body(self.body_of[obj]).xpos[:2])
        return snapshot_support_of(self.world, x, y, self._bottom_z(obj)) or "none"

    # ---- scorer input -------------------------------------------------------------------------------------------
    def snapshot(self, settle_s: float = 2.0) -> dict:
        """Scorer input (schema: agent/robot_api.py). One implementation: sim.snapshot.snapshot (track 2d); the
        adapter supplies the conservative `held_any` as "held", adds "holding" = held() (deviation 18) and ANDs its own
        sim_ok (NaN / blow-up / out-of-house checks)."""
        snap = _snapshot(self.m, self.d, self.world, lambda m, d: self.held_any(), settle_s)
        snap["holding"] = self.held()
        snap["sim_ok"] = bool(snap["sim_ok"] and self.sim_ok())
        return snap

    # ---- packet fact (task 004) ---------------------------------------------------------------------------------
    def robot_room(self) -> str:
        """VLM room name of the room rect holding the base centre (half-open rects as room_of, so a base centred in a
        doorway counts as the room its centre is in); outside every rect: the room whose rect is nearest."""
        x, y = (float(v) for v in base_pose(self.m, self.d)[:2])

        def dist(r: list[float]) -> float:
            return math.hypot(max(r[0] - x, 0.0, x - r[2]), max(r[1] - y, 0.0, y - r[3]))

        rooms = self.layout["rooms"]
        for k, r in rooms.items():
            if r["rect"][0] <= x < r["rect"][2] and r["rect"][1] <= y < r["rect"][3]:
                return self.room_name[k]
        return self.room_name[min(rooms, key=lambda k: dist(rooms[k]["rect"]))]
