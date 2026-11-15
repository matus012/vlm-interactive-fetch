# nav116.sif

`$PROJECT_DIR/images/nav116.sif`, 273 MB, sha256 `17e0a292f1f068bee9f1c466e6720611442b9dea543b3b776b391ddd2357104e`.
Ubuntu 24.04, Python 3.12.3 (`/opt/venv`), MuJoCo 3.17.0 (= uv.lock), numpy 2.5.4, Pillow 13.0.0, openai 3.50.0, PyYAML 6.0.3, ffmpeg 6.1.1.
Full freeze: `images/nav116.versions.txt`, `hpc/image_requirements.txt`. Rebuild: sync code to <login-node>, `nohup bash hpc/build_image.sh`.
~4.5 min build on <login-node>. Falls back to the g1.sif MuJoCo image if ever broken.

## Use in a job
```
apptainer exec --nv --cleanenv --env PROJ=$PROJ -B $PROJ $PROJ/images/nav116.sif python <script>
```
`python` = `/opt/venv/bin/python` (PATH set by the image), `MUJOCO_GL=egl` set by the image. The NVIDIA EGL driver comes from `--nv`
via the glvnd vendor json baked into the image. Close renderers/GL contexts explicitly (a harmless `EGLError` is printed at exit otherwise).
No project code is inside; bind the clone and run it from there. Check: `sbatch hpc/image_check.sbatch` (job 158110: `NVIDIA H200/PCIe/SSE2`).
