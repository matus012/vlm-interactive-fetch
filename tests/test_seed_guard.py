"""Seed guard: generate() and SimRobot refuse test-space seeds (100000-199999) unless sealed_draw=True."""
from __future__ import annotations

import pytest

from sim.adapter import SimRobot
from sim.constants import TEST_SEEDS
from sim.tasks import make_task
from sim.world.generate import generate

SEEDS = [TEST_SEEDS.start, 150123, TEST_SEEDS.stop - 1]


@pytest.mark.parametrize("seed", SEEDS)
def test_generate_refuses_test_seed(seed: int) -> None:
    with pytest.raises(PermissionError):
        generate(seed=seed)
    assert generate(seed=seed, sealed_draw=True).seed == seed


@pytest.mark.parametrize("seed", [TEST_SEEDS.start, TEST_SEEDS.stop - 1])
def test_simrobot_refuses_test_seed(seed: int) -> None:
    with pytest.raises(PermissionError):
        SimRobot(seed)
    SimRobot(seed, sealed_draw=True)


@pytest.mark.parametrize("seed", [0, 999, 1000, 9999, 99999, 200000])
def test_other_seeds_unaffected(seed: int) -> None:
    assert generate(seed=seed).seed == seed


def test_dev_simrobot_and_make_task() -> None:
    SimRobot(1000)
    with pytest.raises(PermissionError):
        make_task(150123, "fetch")
    assert make_task(1000, "fetch")["seed"] == 1000


@pytest.mark.parametrize("helper", ["world", "hall_px", "script_truth_world"])
def test_task_helpers_refuse_test_seed(helper: str) -> None:
    """004 review: tasks.world() opened the test space without the flag; every helper goes through it.
    Seed 150123 only (already opened by the tests above; the sealed draw excludes the seeds the tests open)."""
    from sim import tasks as T
    T.world.cache_clear(), T.hall_px.cache_clear()
    with pytest.raises(PermissionError):
        if helper == "world":
            T.world(150123, "none")
        elif helper == "hall_px":
            T.hall_px(150123, "none")
        else:
            T.room_point_for(150123, "none", "red box")
