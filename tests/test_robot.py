"""G0a-cam: the mast camera is 640x480 with fx = fy = 320 +- 0.5 px; it renders finite RGB + depth."""
import numpy as np

from sim.robot import MAST_CAM, camera_intrinsics, load_tidybot, render_rgbd


def test_mast_camera_intrinsics() -> None:
    m, _ = load_tidybot()
    assert tuple(m.cam_resolution[m.camera(MAST_CAM).id]) == (640, 480)
    K = camera_intrinsics(m)
    assert abs(K[0, 0] - 320.0) <= 0.5 and abs(K[1, 1] - 320.0) <= 0.5


def test_mast_camera_renders() -> None:
    m, d = load_tidybot()
    rgb, depth = render_rgbd(m, d)
    assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
    assert np.isfinite(depth).all() and depth.min() > 0.05


def test_mast_projection_matches_render() -> None:
    """Non-circular G0a-cam check: K + camera pose project known world points onto the pixels MuJoCo renders them at."""
    import mujoco

    from sim.robot import tidybot_spec

    spec = tidybot_spec()
    pts = [(1.0, 0.0, 0.3), (1.5, 0.6, 0.2), (2.0, -1.2, 0.75), (1.2, 0.9, 0.9)]
    for i, p in enumerate(pts):
        b = spec.worldbody.add_body(name=f"probe{i}", pos=list(p))
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], contype=0, conaffinity=0)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("retract").id)
    mujoco.mj_forward(m, d)
    K = camera_intrinsics(m)
    cid = m.camera(MAST_CAM).id
    R, c = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    r = mujoco.Renderer(m, 480, 640)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=MAST_CAM)
        seg = r.render().copy()
    finally:
        r.close()
    for i, p in enumerate(pts):
        pc = R.T @ (np.asarray(p) - c)  # camera frame: x right, y up, looking along -z
        u = K[0, 0] * pc[0] / -pc[2] + K[0, 2]
        v = K[1, 1] * -pc[1] / -pc[2] + K[1, 2]
        gid = m.body(f"probe{i}").geomadr[0]
        mask = (seg[..., 0] == gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        assert mask.sum() > 0, f"probe{i} not visible"
        ys, xs = np.nonzero(mask)
        assert abs(xs.mean() + 0.5 - u) < 1.0 and abs(ys.mean() + 0.5 - v) < 1.0, (i, u, v, xs.mean(), ys.mean())
