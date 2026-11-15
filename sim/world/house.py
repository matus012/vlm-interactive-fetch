"""World MJCF + the TidyBot -> one compiled house model.

The robot (`sim.robot.robot_spec()`, mast camera included) is attached with MjSpec at the world origin through a
frame on the worldbody, with an empty prefix: robot joint / actuator / camera names are unchanged, so Phase 0 helpers
(`sim.body`) work on the house model as they are. Base joints joint_x/y/th are then the world pose (x, y, yaw).
Keyframes are not relied on (the world's free bodies change qpos size): the arm is set from the robot spec's own
`retract` key by joint name.
"""
from __future__ import annotations

import math
import warnings

import mujoco

from sim.robot import ARM_JOINTS, BASE_JOINTS, GRIPPER_ACT, robot_spec
from sim.world.generate import World

GRIP_OPEN = 0.0


def house_spec(world: World) -> tuple[mujoco.MjSpec, dict[str, float]]:
    """World spec with the robot attached; returns (spec, retract arm qpos by joint name)."""
    spec = mujoco.MjSpec.from_string(world.xml)
    robot = robot_spec()
    key = list(robot.key("retract").qpos)
    joints = [j.name for j in robot.joints]
    retract = {j: float(key[joints.index(j)]) for j in ARM_JOINTS}  # qpos order = joint order (all 1-dof here)
    assert joints[: len(BASE_JOINTS) + len(ARM_JOINTS)] == list(BASE_JOINTS + ARM_JOINTS), joints
    frame = spec.worldbody.add_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "empty prefix may prevent saving": we never save the combined XML
        frame.attach_body(robot.body("base_link"), "", "")
    return spec, retract


def set_base(m: mujoco.MjModel, d: mujoco.MjData, pose: tuple[float, float, float] | list[float]) -> None:
    """Base qpos and ctrl (position actuators) to pose (x, y, yaw rad); no mj_forward."""
    for j, v in zip(BASE_JOINTS, pose):
        d.qpos[m.joint(j).qposadr[0]] = v
        d.ctrl[m.actuator(j).id] = v


def load_house(world: World, robot_pose: tuple[float, float, float] | None = None
               ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compiled house + robot. Base at robot_pose (x, y, yaw rad; default layout.spawn with yaw in degrees),
    arm at Menagerie `retract` (qpos and ctrl), gripper open, mj_forward done."""
    spec, retract = house_spec(world)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = spec.compile()
    d = mujoco.MjData(m)
    if robot_pose is None:
        x, y, yaw_deg = world.layout["spawn"]
        robot_pose = (x, y, math.radians(yaw_deg))
    set_base(m, d, robot_pose)
    for j, v in retract.items():
        d.qpos[m.joint(j).qposadr[0]] = v
        d.ctrl[m.actuator(j).id] = v
    d.ctrl[m.actuator(GRIPPER_ACT).id] = GRIP_OPEN
    mujoco.mj_forward(m, d)
    return m, d


def main() -> None:
    """Look renders (not a gate): mast photos at every room point x headings 0/90/180/270 deg + the overview."""
    import argparse
    from pathlib import Path

    from PIL import Image

    from sim.robot import render_rgbd
    from sim.world.generate import LAYOUT, generate, load_places

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocker", default="none")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = generate(LAYOUT, a.seed, a.blocker)
    m, d = load_house(w)
    rgb, _ = render_rgbd(m, d, cam="overview", width=1280, height=960)
    Image.fromarray(rgb).save(a.out / "overview.png")
    for p in load_places():
        if p["kind"] != "room":
            continue
        for h in (0, 90, 180, 270):
            set_base(m, d, (p["pose"][0], p["pose"][1], math.radians(h)))
            mujoco.mj_forward(m, d)
            rgb, _ = render_rgbd(m, d)
            Image.fromarray(rgb).save(a.out / f"mast_{p['room']}_{h:03d}.png")
    print(f"wrote {a.out}/overview.png + mast_<room>_<heading>.png")


if __name__ == "__main__":
    main()
