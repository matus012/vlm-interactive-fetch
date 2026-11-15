"""One home for the numbers the world, skills and gates share (Phase 0 finding 10). Metres, kilograms, radians."""
from __future__ import annotations

# Fetch boxes: 6 x 20 x 12 cm (the 6 cm side fits the 85 mm Robotiq 2F-85), ~0.3 kg, matte, saturated colours.
BOX_HALF = (0.03, 0.10, 0.06)
BOX_MASS = 0.3
BOX_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.05, 0.05, 1.0),
    "blue": (0.05, 0.15, 0.85, 1.0),
    "green": (0.05, 0.65, 0.10, 1.0),
    "yellow": (0.95, 0.85, 0.05, 1.0),
}

# Blockers. The crate is pushable (Phase 2), its colour is used by no fetch box; the cabinet is welded to the floor.
CRATE_HALF = (0.225, 0.225, 0.225)
CRATE_MASS = 4.0
CRATE_RGBA = (0.45, 0.20, 0.55, 1.0)  # purple
CABINET_HALF = (0.25, 0.45, 0.80)  # 0.5 deep, 0.9 wide, 1.6 tall: fills a 1.0 m doorway for the robot
CABINET_RGBA = (0.55, 0.55, 0.58, 1.0)

# Robot footprint (Menagerie TidyBot base, measured 2026-11-13: 0.482 m square, corner radius 0.284 m).
ROBOT_RADIUS = 0.2842
# Deviation 9 (task 003 fix round 3, L1 ruling 2026-11-13): no fetch box within the robot's swept corridor (footprint
# + ROUTE_MARGIN) of any planned route between two places (sim.world.generate.route_clear).
ROUTE_MARGIN = 0.05

# Seed spaces. Dev seeds are free; test seeds are drawn only by Phase 4's sealing step and Phase 6 (sealed_draw=True).
DEV_SEEDS = range(1000, 10000)
TEST_SEEDS = range(100000, 200000)
# Pre-work 9 (task 005): every test-space seed a test touches (tests/test_seed_guard.py, tests/test_tasks.py). The
# sealed draw skips them; tests/test_sampler.py fails if a test file uses a test-space seed missing here.
SEALED_EXCLUDE = frozenset({100000, 150000, 150123, 199999})


def refuse_test_seed(seed: int, sealed_draw: bool) -> None:
    """The one guard: a test-space seed needs an explicit sealed_draw=True (generate, SimRobot, make_task)."""
    if seed in TEST_SEEDS and not sealed_draw:
        raise PermissionError(f"seed {seed} is in the sealed test space; pass sealed_draw=True (Phase 4 sealing / Phase 6 only)")


# Known map.
GRID_RES = 0.05

# Gate tolerances (brief Phase 0 / Phase 1; fixed before running).
G0A_FX = 320.0
G0A_FX_TOL = 0.5
G0A_GOAL_POS_TOL = 0.02
G0A_GOAL_YAW_TOL_DEG = 2.0
G0A_LIFT_MIN = 0.18
G1_REACH_POS_TOL = 0.05
G1_REACH_YAW_TOL_DEG = 3.0
G1_REACH_TRIES = 10
G1_COVER_MIN_PX = 100
G1_COVER_GRID = 0.25
G1_COVER_YAWS_DEG = (0.0, 45.0, 90.0, 135.0)
G1_REST_SIM_S = 600.0
G1_REST_MAX_DRIFT = 0.001

# Phase 2 (task 003) — scorer pass rules (brief §One run) and skill gates, fixed before running.
SCORE_SURFACE_MARGIN = 0.02  # box centre xy inside the surface rect shrunk by 2 cm on every side
SCORE_UPRIGHT_DEG = 15.0  # box 12 cm axis within 15 deg of world z
SCORE_STILL_S = 2.0  # settle window at the end of a run (sim time)
SCORE_STILL_DISP = 0.005  # max displacement in that window (m)
SCORE_STILL_SPEED = 0.02  # max linear speed in that window (m/s)
SCORE_ON_TOP_TOL = 0.015  # box bottom within 1.5 cm of the surface top counts as resting on it
STEP_LIMIT = 20
# Deviation 13 (task 005, L1 2026-11-13): a target is out of sight from the spawn only if it shows < 10 segmentation px
# in every hall look photo (was PX_VISIBLE = 100, the locate bar, which stays 100 for locate / seen).
PX_OUT_OF_SIGHT = 10
# Deviation 13 (task 005): balanced sampler, at least this many target boxes starting on a spawn surface per set,
# where the task allows: 2 of 10 in a counted set, 5 of 25 per type in the sealed set.
MIN_SURFACE_COUNTED = 2
MIN_SURFACE_SEALED = 5
STEP_LIMIT_TWO_BOX = 30
G2_PICK_LIFT = 0.10  # box >= 10 cm above its support after pick, held 2 s
G2_BLOCKED_S = 10.0  # failed: blocked within 10 s sim time of first blocker contact
G2_CABINET_MAX_MOVE = 0.01
G2_CABINET_MAX_PEN = 0.01
