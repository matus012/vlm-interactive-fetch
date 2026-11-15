"""Task 007 harness: H1 (row 20) the runner passes sealed_draw=True only with --verified; H3 (row 21) seal_regen's
trap guard. No sim, no sealed file: SimRobot and make_task are mocked, the guard runs on tmp trees."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import sim.agent_run as ar

REPO = Path(__file__).resolve().parents[1]
TASK = {"task_id": "fetch-1000", "type": "fetch", "seed": 1000, "blocker": "none", "instruction": "x",
        "targets": [{"object": "red box", "surface": "desk"}], "step_limit": 20}


class Stop(Exception):
    pass


def run_main(monkeypatch, tmp_path, argv: list[str]) -> tuple[dict, list]:
    seen: dict = {}
    made: list = []

    def fake_robot(seed, blocker, **kw):
        seen.update(kw, seed=seed, blocker=blocker)
        raise Stop

    def fake_make_task(seed, ttype, **kw):
        made.append((seed, ttype, kw))
        return dict(TASK)

    monkeypatch.setattr(ar, "SimRobot", fake_robot)
    monkeypatch.setattr(ar, "make_task", fake_make_task)
    monkeypatch.setattr(sys, "argv", ["agent_run", *argv, "--task-id", "fetch-1000", "--out", str(tmp_path / "run"),
                                      "--stub", str(tmp_path / "calls.json")])
    (tmp_path / "calls.json").write_text("[]")
    with pytest.raises(Stop):
        ar.main()
    return seen, made


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_verified_mode_passes_sealed_draw_and_skips_make_task(monkeypatch, tmp_path, suffix):
    f = tmp_path / f"verified{suffix}"
    other = TASK | {"task_id": "fetch-1001", "seed": 1001}
    f.write_text("".join(json.dumps(t) + "\n" for t in (other, TASK)) if suffix == ".jsonl"
                 else json.dumps({"tasks": [other, TASK]}))
    seen, made = run_main(monkeypatch, tmp_path, ["--verified", str(f)])
    assert seen["sealed_draw"] is True and seen["seed"] == 1000
    assert made == []


def test_taskset_mode_never_passes_sealed_draw(monkeypatch, tmp_path):
    f = tmp_path / "set.json"
    f.write_text(json.dumps({"tasks": [TASK]}))
    seen, made = run_main(monkeypatch, tmp_path, ["--taskset", str(f)])
    assert seen["sealed_draw"] is False
    assert made == [(1000, "fetch", {"want": None})]


def test_taskset_and_verified_are_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["agent_run", "--taskset", "a", "--verified", "b", "--task-id", "x", "--out", "o"])
    with pytest.raises(SystemExit):
        ar.main()


def load_seal_regen():
    spec = importlib.util.spec_from_file_location("seal_regen", REPO / "hpc" / "seal_regen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_head_commit_reads_clone_and_worktree(tmp_path):
    sr = load_seal_regen()
    (tmp_path / "clone" / ".git").mkdir(parents=True)
    (tmp_path / "clone" / ".git" / "HEAD").write_text(sr.SEAL_COMMIT + "\n")
    (tmp_path / "gd").mkdir()
    (tmp_path / "gd" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text(f"gitdir: {tmp_path / 'gd'}\n")
    assert sr.head_commit(tmp_path / "clone") == sr.SEAL_COMMIT
    assert sr.head_commit(tmp_path / "wt") == "ref: refs/heads/main"


def test_trap_guard_fires_when_sim_is_not_from_the_worktree(tmp_path):
    """The trap: sim already imported from this tree (HEAD), worktree elsewhere -> refuse."""
    sr = load_seal_regen()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(sr.SEAL_COMMIT)
    with pytest.raises(AssertionError, match="trap: sim from"):
        sr.guard(tmp_path)


def test_trap_guard_fires_on_a_tree_not_at_seal_005():
    """sim / agent do come from this tree, but its HEAD is not ba7bfff."""
    sr = load_seal_regen()
    if sr.head_commit(REPO) == sr.SEAL_COMMIT:
        pytest.skip("this checkout is at seal-005")
    with pytest.raises(AssertionError, match="is not the seal commit ba7bfff"):
        sr.guard(REPO)


def test_verified_end_to_end_on_a_patched_dev_seed(tmp_path, monkeypatch):
    """R3 (review 007): a dev task run with --taskset --stub script, then the same dict with --verified --stub <its
    calls> while the dev seed is patched into TEST_SEEDS: the whole verified run + scoring path opens no other
    seed-guarded entry and gives the same verdict. Control: under the patch the --taskset run (and a SimRobot without
    sealed_draw) raises PermissionError."""
    import hashlib

    import sim.constants
    import sim.tasks
    from sim.adapter import SimRobot
    from sim.tasks import make_task

    task = make_task(1000, "fetch")
    (tmp_path / "set.json").write_text(json.dumps({"tasks": [task]}))
    vf = tmp_path / "verified.jsonl"
    vf.write_text(json.dumps(task, sort_keys=True) + "\n")

    def main(*argv: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["agent_run", *argv, "--task-id", task["task_id"]])
        ar.main()
        return json.loads((Path(argv[argv.index("--out") + 1]) / "result.json").read_text())

    a = main("--taskset", str(tmp_path / "set.json"), "--out", str(tmp_path / "a"), "--stub", "script")
    calls = str(tmp_path / "a" / "stub_calls.json")
    patched = frozenset(sim.constants.TEST_SEEDS) | {task["seed"]}
    for mod in (sim.constants, sim.tasks):
        monkeypatch.setattr(mod, "TEST_SEEDS", patched)
    b = main("--verified", str(vf), "--out", str(tmp_path / "b"), "--stub", calls)
    keys = ("pass", "label", "failed", "steps_used", "done_called", "calls")
    assert a["pass"] and {k: b[k] for k in keys} == {k: a[k] for k in keys}
    assert "sealed" not in a and b["sealed"] is True
    assert b["verified_sha256"] == hashlib.sha256(vf.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        main("--taskset", str(tmp_path / "set.json"), "--out", str(tmp_path / "c"), "--stub", calls)
    with pytest.raises(PermissionError):
        SimRobot(task["seed"], task["blocker"])
