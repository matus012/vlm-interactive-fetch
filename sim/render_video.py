"""Render a video run (sim/video_run.py output) at any sim time: third-person 3D view + the live mast camera (task 013).

    from sim.render_video import VideoScene
    vs = VideoScene(Path("runs/013/blocked-100040"))     # NAV116_HOUSE / easings must match the run, as in video_run
    third = vs.third(t, w, h)                           # follows the base, walls cut low (this view only)
    mast = vs.mast(t)                                   # the robot's own camera (MAST_CAM), full walls

The world is rebuilt from the verified task dict exactly as the video run built it (SimRobot); qpos comes from the
trace (60 Hz), linearly interpolated, free-joint quaternions renormalised. Nothing is simulated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from sim.robot import CAM_H, CAM_W, MAST_CAM

WALL_CUT = 0.30  # wall height (m) in the third-person view only, so the robot is never hidden


class VideoScene:
    def __init__(self, run_dir: Path) -> None:
        from sim.adapter import SimRobot
        self.dir = run_dir
        vr = json.loads((run_dir / "video_result.json").read_text())
        assert vr["valid"], run_dir
        if vr["easings"] is False:
            from sim.world.generate import set_easings
            set_easings(False)
        from sim.world.generate import HOUSE
        assert vr["house"] == HOUSE, (vr["house"], HOUSE)
        robot = SimRobot(vr["seed"], vr["blocker"], sealed_draw=True)
        self.m, self.d = robot.m, robot.d
        tr = np.load(run_dir / "trace.npz")
        t, q = tr["t"], tr["qpos"]
        keep = np.concatenate([[True], np.diff(t) > 1e-9])  # drop duplicate stamps (records at call edges)
        self.t, self.q = t[keep], q[keep]
        self.quat = [int(self.m.jnt_qposadr[j]) + 3 for j in range(self.m.njnt)
                     if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.bx = int(self.m.joint("joint_x").qposadr[0])
        self.walls = [i for i in range(self.m.ngeom) if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("wall")]
        self.wall_size = self.m.geom_size[self.walls].copy()
        self.r3: dict[tuple[int, int], mujoco.Renderer] = {}
        self.rm = mujoco.Renderer(self.m, CAM_H, CAM_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.look: np.ndarray | None = None
        self.vr = vr

    def set(self, t: float) -> None:
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        t0, t1 = self.t[i - 1], self.t[i]
        u = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        q = self.q[i - 1] * (1 - u) + self.q[i] * u
        for a in self.quat:
            q[a:a + 4] /= np.linalg.norm(q[a:a + 4])
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)

    def base(self, t: float) -> tuple[float, float, float]:
        self.set(t)
        return tuple(float(v) for v in self.d.qpos[self.bx:self.bx + 3])

    def third(self, t: float, w: int, h: int, az: float = 90.0, elev: float = -52.0, dist: float = 3.6,
              smooth: float = 0.12, z: float = 0.45) -> np.ndarray:
        x, y, _ = self.base(t)
        p = np.array([x, y, z])
        self.look = p if self.look is None else self.look + smooth * (p - self.look)
        self.cam.lookat[:] = self.look
        self.cam.azimuth, self.cam.elevation, self.cam.distance = az, elev, dist
        zs = self.d.geom_xpos[self.walls, 2].copy()
        self.m.geom_size[self.walls, 2] = WALL_CUT / 2  # the scene reads the size from the model and the centre from data
        self.d.geom_xpos[self.walls, 2] = WALL_CUT / 2
        try:
            r = self.r3.get((w, h)) or self.r3.setdefault((w, h), mujoco.Renderer(self.m, h, w))
            r.update_scene(self.d, camera=self.cam)
            return r.render()
        finally:
            self.m.geom_size[self.walls] = self.wall_size
            self.d.geom_xpos[self.walls, 2] = zs

    def mast(self, t: float | None = None) -> np.ndarray:
        if t is not None:
            self.set(t)
        self.rm.update_scene(self.d, camera=MAST_CAM)
        return self.rm.render()

    def reset_follow(self) -> None:
        self.look = None

    def close(self) -> None:
        for r in self.r3.values():
            r.close()
        self.rm.close()
