"""Task sets (tasksets/*.json). Historical sets (drawn by rules since changed: Phase 3, before task 005's deviations
13 and 15) are checked against their stored sha256_core only, never regenerated. Every other set regenerates from its
`rule` (a sim.sampler.draw_set call) on the current code to the same tasks (no drift)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import sim.sampler

SETS = Path(__file__).resolve().parents[1] / "tasksets"
HISTORICAL = ["phase3_tuning", "phase3_counted"]
SMOKE = ["e2_smoke"]  # E2 house B plumbing smoke: 3 hand-listed dev tasks, no draw rule or sha256_core to check
CURRENT = sorted(p.stem for p in SETS.glob("*.json") if p.stem not in HISTORICAL + SMOKE)


def regenerate(rule: str) -> list[dict]:
    call = ast.parse(rule, mode="eval").body
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "sim.sampler.draw_set"
    args = [ast.literal_eval(a) for a in call.args]
    return sim.sampler.draw_set(*args, **{k.arg: ast.literal_eval(k.value) for k in call.keywords})


def core_hash(tasks: list[dict]) -> str:
    core = [{k: t[k] for k in ("task_id", "seed", "instruction", "targets")} for t in tasks]
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("name", HISTORICAL)
def test_historical_set_hash(name: str) -> None:
    stored = json.loads((SETS / f"{name}.json").read_text())
    assert core_hash(stored["tasks"]) == stored["sha256_core"]


def test_phase4_sets_present() -> None:
    want = {f"phase4_{t}_{k}" for t in ("blocked", "push_or_detour", "two_box") for k in ("tuning", "counted")}
    assert want | {"phase4_fetch_counted"} <= set(CURRENT)


@pytest.mark.parametrize("name", CURRENT)
def test_taskset_regenerates(name: str) -> None:
    stored = json.loads((SETS / f"{name}.json").read_text())
    assert core_hash(stored["tasks"]) == stored["sha256_core"]
    tasks = regenerate(stored["rule"])
    assert core_hash(tasks) == stored["sha256_core"]
    assert tasks == stored["tasks"]
