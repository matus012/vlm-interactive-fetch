"""House-A invariance of the layout-driven gate harness (task 011): start poses / goals / vias / jitter of the push,
blocked and crate trials equal the values the pre-refactor house-A literals gave (captured before the refactor)."""
from __future__ import annotations

import pytest

from sim import gate2, push_check as pc
from sim.world.generate import HOUSE, generate

pytestmark = pytest.mark.skipif(HOUSE != "A", reason="house-A literals")

# seed -> [door, goal place, far room, start_pose(hall side)]
CRATE = {
    1100: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1101: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1102: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1103: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1104: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1105: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1106: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1107: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1108: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1109: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1200: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1201: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1202: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1203: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
    1204: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1205: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1206: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1207: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1208: ['d_bed', 'bedroom', 'bedroom', [4.2, 5.25, 3.141592654]],
    1209: ['d_sto', 'storage', 'storage', [4.8, 5.25, 0.0]],
}
# seed -> [door, start_pose(living_room side)]
CAB = {
    1100: ['d_lk2', [3.3, 0.75, 0.0]],
    1101: ['d_lk1', [3.3, 2.75, 0.0]],
    1102: ['d_lk1', [3.3, 2.75, 0.0]],
    1103: ['d_lk1', [3.3, 2.75, 0.0]],
    1104: ['d_lk1', [3.3, 2.75, 0.0]],
    1105: ['d_lk1', [3.3, 2.75, 0.0]],
    1106: ['d_lk1', [3.3, 2.75, 0.0]],
    1107: ['d_lk2', [3.3, 0.75, 0.0]],
    1108: ['d_lk2', [3.3, 0.75, 0.0]],
    1109: ['d_lk2', [3.3, 0.75, 0.0]],
    1200: ['d_lk1', [3.3, 2.75, 0.0]],
    1201: ['d_lk1', [3.3, 2.75, 0.0]],
    1202: ['d_lk2', [3.3, 0.75, 0.0]],
    1203: ['d_lk2', [3.3, 0.75, 0.0]],
    1204: ['d_lk1', [3.3, 2.75, 0.0]],
    1205: ['d_lk1', [3.3, 2.75, 0.0]],
    1206: ['d_lk1', [3.3, 2.75, 0.0]],
    1207: ['d_lk1', [3.3, 2.75, 0.0]],
    1208: ['d_lk1', [3.3, 2.75, 0.0]],
    1209: ['d_lk2', [3.3, 0.75, 0.0]],
}
# seed -> cabinet_start(seed) (jittered living-room start)
CABSTART = {
    1100: [2.27264905, 1.661806649, 0.137263884],
    1101: [2.313623241, 1.607642127, -0.049010665],
    1102: [2.225821497, 1.823805412, 0.144031482],
    1103: [2.334135692, 1.621688812, 0.133263958],
    1104: [2.24305292, 1.878577064, -0.008258229],
    1105: [2.22497956, 1.820452193, -0.104894798],
    1106: [2.343278636, 1.721376976, -0.14511501],
    1107: [2.338014387, 1.641265731, 0.012959939],
    1108: [2.196568731, 1.730705722, -0.163573783],
    1109: [2.347731918, 1.701673636, -0.044899984],
    1200: [2.227685757, 1.689414884, -0.047405668],
    1201: [2.250863712, 1.86729284, 0.156739999],
    1202: [2.264433041, 1.744292229, 0.104966293],
    1203: [2.318046012, 1.891364799, -0.097868919],
    1204: [2.333200026, 1.687107723, 0.112030589],
    1205: [2.169959999, 1.802986792, -0.047686033],
    1206: [2.326143977, 1.875266005, -0.013113455],
    1207: [2.221381401, 1.899306265, -0.119680522],
    1208: [2.16211078, 1.60915151, 0.121122636],
    1209: [2.286625919, 1.715097621, -0.058930581],
}


def test_mappings() -> None:
    assert gate2.CRATE_GOAL == {"d_bed": "bedroom", "d_sto": "storage"}
    assert (gate2.KT, gate2.SB) == ("kitchen table", "storage bench")
    assert gate2.cabinet_goal("d_lk1") == gate2.cabinet_goal("d_lk2") == "kitchen"
    from sim.cover_check import ROOMS
    assert ROOMS == ("bedroom", "storage", "living_room", "kitchen")


@pytest.mark.parametrize("seed", sorted(CRATE))
def test_crate(seed: int) -> None:
    w = generate(seed=seed, blocker="crate")
    dn = w.blocker["door"]
    door = pc.door_dict(w.layout, dn)
    far = pc.behind_room(w.layout, dn)
    want = CRATE[seed]
    assert (dn, gate2.CRATE_GOAL[dn], far) == tuple(want[:3])
    assert pc.start_pose(door, "hall", w.layout) == pytest.approx(want[3], abs=1e-9)
    assert pc.room_place(far) == gate2.CRATE_GOAL[dn]


@pytest.mark.parametrize("seed", sorted(CAB))
def test_cabinet(seed: int) -> None:
    w = generate(seed=seed, blocker="cabinet")
    dn = w.blocker["door"]
    door = pc.door_dict(w.layout, dn)
    assert dn == CAB[seed][0]
    start_room, goal_room = pc.cabinet_rooms(w.layout, dn)
    assert (start_room, goal_room) == ("living_room", "kitchen")
    assert pc.cabinet_axes(w.layout, dn) == (0, 1, 1.0)
    assert pc.start_pose(door, start_room, w.layout) == pytest.approx(CAB[seed][1], abs=1e-9)
    assert gate2.cabinet_start(seed) == pytest.approx(CABSTART[seed], abs=1e-9)


def test_house_a_detour_is_the_twin_door() -> None:
    """Task 011 row 23: script_truth's detour stays the twin kitchen door in house A (room via only without a twin)."""
    import os

    from sim.tasks import dev_tasks, script_truth
    if os.environ.get("NAV116_HOUSE", "A") != "A":
        pytest.skip("house A only")
    vias = {script_truth(t)["detour_via"] for t in dev_tasks("push_or_detour", 4, 1000)}
    assert vias <= {"kitchen north door", "kitchen south door"} and vias, vias
