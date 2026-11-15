"""Sim process for the 0b job (runs inside g1.sif): render the mast photo, then keep stepping MuJoCo until STOP exists.

    python hpc/sim_beside.py <out_dir>      writes photo.png, sim_steps.jsonl (epoch time, cumulative steps, every 2 s)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.photo import load_test_scene, render_photo  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
render_photo(out / "photo.png")
print(f"photo rendered in {time.time() - t0:.1f}s", flush=True)
m, d = load_test_scene()
steps, last = 0, time.time()
with open(out / "sim_steps.jsonl", "w") as f:
    while not (out / "STOP").exists():
        if (out / "render_now").exists() and not (out / "photo_after.png").exists():  # EGL render beside vLLM
            t1 = time.time()
            render_photo(out / "photo_after.tmp.png")
            (out / "photo_after.tmp.png").rename(out / "photo_after.png")
            (out / "render_after.json").write_text(json.dumps({"render_s": round(time.time() - t1, 2)}))
        for _ in range(200):
            mujoco.mj_step(m, d)
        steps += 200
        now = time.time()
        if now - last >= 2.0:
            f.write(json.dumps({"t": now, "steps": steps}) + "\n")
            f.flush()
            last = now
print(f"stopped after {steps} steps", flush=True)
