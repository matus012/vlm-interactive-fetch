"""Smoke tests for the Phase 2 gate harness (sim/gate2.py). The gate runs themselves live in runs/gate2/."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from sim import gate2


def test_pick_list_rule() -> None:
    lst = gate2.pick_list()
    assert len(lst) == gate2.N_PICK and len({s for s, _ in lst}) == gate2.N_PICK
    from sim.world.generate import generate
    sup = [next(b for b in generate(seed=s, blocker="none").boxes if f"{b['colour']} box" == n)["surface"] for s, n in lst]
    assert sum(x != "floor" for x in sup) >= gate2.N_PICK_SURFACE


def test_stepwatch_restores_mj_step() -> None:
    orig = mujoco.mj_step
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="f" type="plane" size="1 1 .1"/>'
                                       '<body pos="0 0 .05"><freejoint/><geom type="sphere" size=".1"/></body>'
                                       '</worldbody></mujoco>')
    d = mujoco.MjData(m)
    with gate2.StepWatch(np.array([1]), np.array([0])) as sw:
        for _ in range(50):
            mujoco.mj_step(m, d)
    assert mujoco.mj_step is orig and sw.pen > 0.0  # sphere starts 5 cm into the plane


def test_bumper_scan(tmp_path: Path) -> None:
    run = tmp_path / "script" / "fetch_s1_-"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"task": {"blocker": "none"}}))
    b = sorted(gate2._base_geoms("none"))[0]
    steps = [{"step": 1, "call": "navigate", "args": {"place": "hall"}, "result": "failed: blocked", "ok": False,
              "meta": {"contacts": {f"red box|{b}": 0.002, "red box|left_pad1": 0.001},
                       "detail": "bumper: lag 0.02 m, base contact 31 N"}},
             {"step": 2, "call": "navigate", "args": {"place": "hall"}, "result": "reached", "ok": True,
              "meta": {"contacts": {"blue box|left_pad1": 0.001, "crate|49": 0.003}, "detail": ""}},
             {"event": "end"}]
    (run / "log.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")
    r = gate2.bumper_scan(tmp_path)
    assert r["n_events"] == 1 and r["runs"] == ["script/fetch_s1_-"] and r["n_events_without_blocker_contact"] == 1
    assert r["n_box_base_contacts_nav"] == 1 and r["n_nav_box_contacts_any_robot_geom"] == 3


def test_cabinet_start_jitter() -> None:
    """005 row 12: every cabinet seed gets its own start, within the documented bounds, free in the inflated map."""
    from sim.constants import ROBOT_RADIUS
    from sim.world.generate import load_layout
    from sim.world.occupancy import known_map
    g = known_map(load_layout()).inflate(ROBOT_RADIUS + 0.02)
    x0, y0, yaw0 = gate2._pose("living room")
    starts = [gate2.cabinet_start(s) for s in [*gate2.PUSH_SEEDS, *gate2.BLOCKED_SEEDS]]
    assert len(set(starts)) == len(starts) and gate2.cabinet_start(1100) == starts[0]
    for x, y, yaw in starts:
        assert abs(x - x0) <= gate2.CAB_JIT_ALONG and abs(y - y0) <= gate2.CAB_JIT_LAT
        assert abs(math.degrees(yaw - yaw0)) <= gate2.CAB_JIT_DEG and g.free(x, y)


def test_distinct_trials() -> None:
    rows = [{"first_contact": {"t": 6.658, "pos": [4.2, 2.5, 0.05]}}] * 3 + \
           [{"first_contact": {"t": 6.7, "pos": [4.2, 2.5, 0.05]}}, {"first_contact": None}]
    assert gate2.distinct_trials(rows) == {"n_with_contact": 4, "distinct_t": 2, "distinct_point": 1,
                                           "distinct_trials": 2}


def test_carry_push_box_reachable() -> None:
    from sim.world.generate import generate
    for s in gate2.CARRY_PUSH_SEEDS:
        w = generate(seed=s, blocker="crate")
        b = next(b for b in w.boxes if f"{b['colour']} box" == gate2.carry_push_box(s))
        assert b["room"] != {"d_bed": "bedroom", "d_sto": "storage"}[w.blocker["door"]]
