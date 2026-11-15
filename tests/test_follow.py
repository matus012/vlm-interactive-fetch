"""Path follower (task 002 item 3): go_to reaches a place in the house within the G1-reach tolerances, no wall
contacts; a mutated follower (last segment + final turn skipped) is caught by the same checks."""
import math

import pytest

import sim.nav.follow as follow
from sim.constants import G1_REACH_POS_TOL, G1_REACH_YAW_TOL_DEG
from sim.reach_check import compile_house, pose_of, start_state
from sim.world.generate import generate, load_places
from sim.world.occupancy import known_map

PLACES = {p["name"]: p for p in load_places()}


def _hall_to_bench() -> dict:
    w = generate(seed=0, blocker="none")
    m = compile_house(w)
    d = start_state(m, pose_of(PLACES["hall"]))
    return follow.go_to(m, d, known_map(w.layout), pose_of(PLACES["storage bench"]))


def _ok(r: dict) -> bool:
    return (r["pos_err_m"] <= G1_REACH_POS_TOL and r["body_pos_err_m"] <= G1_REACH_POS_TOL
            and r["yaw_err_deg"] <= G1_REACH_YAW_TOL_DEG and r["body_yaw_err_deg"] <= G1_REACH_YAW_TOL_DEG)


def test_go_to_hall_to_storage_bench() -> None:
    r = _hall_to_bench()  # hall (yaw 0) -> storage bench (yaw 90 deg), through the hall-storage door
    assert r["reached"] and _ok(r)
    assert len(r["waypoints"]) >= 3 and r["path_len_m"] > 3.0
    assert not [k for k in r["contacts"] if "wall" in k]
    assert r["contacts"] == {}  # seed 0 has no box in the hall or storage
    assert r["max_track_err_m"] < 0.01
    assert r["sim_s"] > r["path_len_m"] / 0.5  # driven at <= BASE_VMAX, not teleported


@pytest.mark.parametrize("drop_last_segment", [False, True])
def test_mutant_follower_fails(monkeypatch: pytest.MonkeyPatch, drop_last_segment: bool) -> None:
    """Mutant: the final arrive-and-turn drive is skipped (and optionally the last waypoint segment too)."""
    real_drive, real_sh = follow.drive_base, follow.segment_headings

    def drive(m, d, goal, settle=1.0, cb=None):  # type: ignore[no-untyped-def]
        if settle != follow.SETTLE_S:
            real_drive(m, d, goal, settle=settle, cb=cb)

    monkeypatch.setattr(follow, "drive_base", drive)
    if drop_last_segment:
        monkeypatch.setattr(follow, "segment_headings", lambda *a, **k: real_sh(*a, **k)[:-1])
    r = _hall_to_bench()
    assert not r["reached"] and not _ok(r)
    assert r["yaw_err_deg"] > 80.0  # held yaw 0 instead of the bench's 90 deg
    if drop_last_segment:
        assert r["pos_err_m"] > G1_REACH_POS_TOL


@pytest.mark.parametrize("margin_on", [True, False])
def test_door_jamb_clearance(monkeypatch: pytest.MonkeyPatch, margin_on: bool) -> None:
    """kitchen -> bedroom passes the d_bed jamb. With DISC_MARGIN the base clears it; without it (G1-reach round 1)
    the planned polyline is 0.2828 m from the jamb (< ROBOT_RADIUS) and the base mesh touches wall01."""
    if not margin_on:
        monkeypatch.setattr(follow, "DISC_MARGIN", 0.0)
    w = generate(seed=0, blocker="none")
    m = compile_house(w)
    d = start_state(m, pose_of(PLACES["kitchen"]))
    r = follow.go_to(m, d, known_map(w.layout), pose_of(PLACES["bedroom"]))
    assert r["reached"]
    walls = [k for k in r["contacts"] if "wall" in k]
    assert (walls == []) is margin_on, r["contacts"]


def test_bumper_ignores_box_but_jam_still_blocks() -> None:
    """2j fix 3: fetch-box contacts do not count for the bumper (recorded in contacts); a box jammed between the base
    and the welded cabinet still stops the drive (tracking / no_progress)."""
    import math

    import mujoco

    from sim.nav.follow import go_to
    from sim.world.generate import generate
    from sim.world.house import load_house
    from sim.world.occupancy import known_map

    seed = next(s for s in range(1200, 1300) if generate(seed=s, blocker="cabinet").blocker["door"] == "d_lk1")
    w = generate(seed=seed, blocker="cabinet")
    m, d = load_house(w, (2.25, 2.75, 0.0))
    a = m.jnt_qposadr[m.body("box_red").jntadr[0]]
    d.qpos[a:a + 7] = (4.25 - 0.10 - 0.002, 2.75, 0.0605, math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))
    for b in ("box_blue", "box_green", "box_yellow"):  # out of the way (hall floor)
        a = m.jnt_qposadr[m.body(b).jntadr[0]]
        d.qpos[a:a + 3] = (3.5 + 0.6 * ("box_blue", "box_green", "box_yellow").index(b), 6.6, 0.0605)
    mujoco.mj_forward(m, d)
    r = go_to(m, d, known_map(w.layout), (6.75, 2.75, 0.0))
    assert not r["reached"] and r["blocked"]["why"] in ("tracking", "no_progress"), r["blocked"]
    assert any(k.startswith("box_red|") for k in r["contacts"])


def test_back_off_after_block() -> None:
    """2j fix 5: once blocked (cabinet in d_lk1), the base backs off ~0.25 m along its path, then holds; blocked["t"]
    is still the detection time (the back-off and hold come after it)."""
    import mujoco

    from sim.nav.follow import BACK_OFF_S, HOLD_S, go_to
    from sim.world.generate import generate
    from sim.world.house import load_house
    from sim.world.occupancy import known_map

    seed = next(s for s in range(1200, 1300) if generate(seed=s, blocker="cabinet").blocker["door"] == "d_lk1")
    w = generate(seed=seed, blocker="cabinet")
    m, d = load_house(w, (2.25, 2.75, 0.0))
    for i, b in enumerate(("box_red", "box_blue", "box_green", "box_yellow")):  # out of the way (hall floor)
        a = m.jnt_qposadr[m.body(b).jntadr[0]]
        d.qpos[a:a + 3] = (3.5 + 0.6 * i, 6.6, 0.0605)
    mujoco.mj_forward(m, d)
    r = go_to(m, d, known_map(w.layout), (6.75, 2.75, 0.0), back_off=0.25)
    b = r["blocked"]
    assert b is not None and 0.2 <= b["back_off_m"] <= 0.3, b
    assert d.time - b["t"] >= BACK_OFF_S + HOLD_S - 1e-6
    cab = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("cabinet").id}
    assert not any(c.geom1 in cab or c.geom2 in cab for c in d.contact[: d.ncon])  # off the cabinet at the end
