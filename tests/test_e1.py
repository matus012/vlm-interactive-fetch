"""Task 009 (E1, context row 22): the easings switch. (a) default unchanged: here the default-path invariants; the
seal-005 regeneration itself is `python -m sim.e1 --check-seal-005` (NVIDIA, report evidence). (b) easings off on dev
seeds produces the lifted cases. (c) the rules text differs by exactly line E. Easings off is process-wide, so every
easings-off check runs in a subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDS = range(9000, 9050)


def run_py(code: str, *args: str, timeout: int = 900) -> str:
    env = os.environ | {"PYTHONPATH": str(REPO), "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    p = subprocess.run([sys.executable, "-c", code, *args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=timeout)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout.strip().splitlines()[-1]


def test_rules_text_differs_by_line_e_only() -> None:
    from agent.context import SYSTEM
    from agent.context_e1 import RULE_E, SYSTEM_NO_E
    a, b = SYSTEM.splitlines(), SYSTEM_NO_E.splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 1 and diff[0][0].replace(RULE_E, "") == diff[0][1]
    assert "stays blocked" not in SYSTEM_NO_E and "stays blocked" in SYSTEM


def test_default_has_easings_on() -> None:
    from sim.tasks import make_task
    from sim.world import generate as gen
    assert gen.EASINGS is True
    assert "easings" not in make_task(4000, "fetch")


def test_default_worlds_break_no_eased_rule() -> None:
    """The flags measure the eased rules: with the default, no box of any dev world breaks one."""
    from sim.e1 import box_flags
    from sim.tasks import world
    assert not any(v for s in SEEDS for f in box_flags(world(s, "none")).values() for v in f.values())


def test_set_easings_refuses_after_a_draw() -> None:
    out = run_py("from sim.world.generate import generate, set_easings\n"
                 "generate(seed=4000)\nset_easings(True)\n"
                 "try:\n    set_easings(False)\nexcept RuntimeError:\n    print('refused')")
    assert out == "refused"


def test_easings_off_produces_lifted_cases() -> None:
    """(b) easings off on dev seeds 9000-9049: route-corridor boxes, back-half nightstand boxes, near-surface floor
    boxes all occur; coffee-table boxes still never pass the kept out-of-sight rule."""
    out = json.loads(run_py(
        "import json, sys\nfrom collections import Counter\n"
        "from sim.world.generate import set_easings\nset_easings(False)\n"
        "from sim.e1 import box_flags\nfrom sim.tasks import box_name, hall_px, world\n"
        "c = Counter()\n"
        "for s in range(int(sys.argv[1]), int(sys.argv[2])):\n"
        "    w = world(s, 'none'); px = hall_px(s, 'none')\n"
        "    srf = {b['name']: b['surface'] for b in w.boxes}\n"
        "    for k, f in box_flags(w).items():\n"
        "        c.update(x + ('_' + srf[k] if x == 'back_of_surface' else '') for x, v in f.items() if v)\n"
        "    c['coffee_out_of_sight'] += sum(b['surface'] == 'coffee table' and px[box_name(b['colour'])] < 10\n"
        "                                    for b in w.boxes)\n"
        "print(json.dumps(c))", str(SEEDS.start), str(SEEDS.stop)))
    assert out.get("on_route", 0) > 0 and out.get("back_of_surface_nightstand", 0) > 0
    assert out.get("near_surface_floor", 0) > 0
    assert out.get("coffee_out_of_sight", 0) == 0


def test_easings_off_task_skips_pregrasp_filter_and_carries_key() -> None:
    out = run_py("import sim.tasks as T\nfrom sim.world.generate import set_easings\nset_easings(False)\n"
                 "def no(*a):\n    raise AssertionError('pregrasp filter used with easings off')\n"
                 "T.pregrasp_ok = no\n"
                 "print(T.make_task(T.dev_tasks('fetch', 1, 9000)[0]['seed'], 'fetch').get('easings'))")
    assert out == "False"


@pytest.mark.parametrize("flag", [True, False])
def test_runner_e1_needs_e1_task(tmp_path: Path, flag: bool) -> None:
    """--e1 runs an easings-off task end to end (stub = the correct script) with the no-E rules text in every packet;
    a default task with --e1 (or an E1 task without it) is refused."""
    ts = tmp_path / "ts.json"
    run_py("import json, sys\nfrom sim.world.generate import set_easings\n"
           f"set_easings({not flag})\n"
           "from sim.tasks import dev_tasks\n"
           "t = dev_tasks('fetch', 1, 9000)[0]\n"
           "open(sys.argv[1], 'w').write(json.dumps({'tasks': [t]}))\nprint(t['task_id'])", str(ts))
    tid = json.loads(ts.read_text())["tasks"][0]["task_id"]
    env = os.environ | {"PYTHONPATH": str(REPO)}
    p = subprocess.run([sys.executable, "-m", "sim.agent_run", "--taskset", str(ts), "--task-id", tid,
                        "--out", str(tmp_path / "run"), "--stub", "script", "--e1"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    if not flag:
        assert p.returncode != 0 and "E1 task <-> --e1" in p.stderr
        return
    assert p.returncode == 0, p.stderr[-3000:]
    from agent.context_e1 import SYSTEM_NO_E
    res = json.loads((tmp_path / "run" / "result.json").read_text())
    assert res["e1"] is True and res["rules"] == "no-E"
    steps = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    systems = {s["extra"]["packet"]["system"] for s in steps if "extra" in s}
    assert systems == {SYSTEM_NO_E}
