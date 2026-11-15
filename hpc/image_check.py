"""Inside nav116.sif on a GPU node: EGL renderer string + one 640x480 RGB+depth Menagerie TidyBot frame."""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from mujoco.egl import GLContext
from OpenGL import GL
from PIL import Image

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
c = GLContext(64, 64); c.make_current()
print("GL_RENDERER", GL.glGetString(GL.GL_RENDERER), flush=True)
print("mujoco", mujoco.__version__, "numpy", np.__version__)
m = mujoco.MjModel.from_xml_path(os.environ["PROJ"] + "/ext/mujoco_menagerie/stanford_tidybot/scene.xml")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
cam.lookat[:] = [0, 0, 0.5]; cam.distance = 3.0; cam.azimuth = 135; cam.elevation = -20
r.update_scene(d, cam); rgb = r.render()
r.enable_depth_rendering(); r.update_scene(d, cam); dep = r.render(); r.disable_depth_rendering()
r.close()
Image.fromarray(rgb).save(out / "rgb.png")
v = np.clip(dep / max(dep.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
Image.fromarray(v).save(out / "depth.png")
print("rgb", rgb.shape, rgb.dtype, rgb.std(), "depth", dep.shape, float(dep.min()), float(dep.max()))
assert rgb.std() > 5 and dep.max() > dep.min()
print("IMAGE_CHECK_OK")
