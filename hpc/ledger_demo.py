"""The work for the ledger test: one MuJoCo episode (TidyBot, no rendering) per item, ~item_seconds of wall.
usage: ledger_demo.py <ledger> <scene.xml> [item_seconds=30]"""
from __future__ import annotations

import hashlib
import sys
import time

import mujoco
import numpy as np

import ledger

ledger_dir, scene = sys.argv[1], sys.argv[2]
item_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
model = mujoco.MjModel.from_xml_path(scene)
data = mujoco.MjData(model)


def episode(item: str, heartbeat) -> dict:
    rng = np.random.default_rng(int(item[4:]))
    mujoco.mj_resetData(model, data)
    steps, t0 = 0, time.time()
    while time.time() - t0 < item_seconds:
        data.ctrl[:] = rng.uniform(-1, 1, model.nu) * (model.actuator_ctrlrange[:, 1] if model.nu else 0)
        for _ in range(200):
            mujoco.mj_step(model, data)
        steps += 200
        heartbeat()
    return {"steps": steps, "steps_per_s": round(steps / (time.time() - t0)),
            "qpos_hash": hashlib.sha256(np.round(data.qpos, 6).tobytes()).hexdigest()[:16]}


sys.exit(ledger.run(ledger_dir, episode))
