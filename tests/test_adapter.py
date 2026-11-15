"""Robot adapter (task 003): look, navigate (rooms, via doors, avoid), blocked detector, snapshot."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

import sim.nav.follow as follow
from sim.adapter import SimRobot
from sim.constants import ROBOT_RADIUS


def _put_box(r: SimRobot, name: str, x: float, y: float, yaw: float = 0.0) -> None:
    """Test setup only: start state with the box moved onto a route."""
    adr = r.m.joint(r.m.body(r.body_of[name]).jntadr[0]).qposadr[0]
    r.d.qpos[adr: adr + 7] = [x, y, 0.0605, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(r.m, r.d)


def _x_crossing(path: list[tuple[float, float]], x: float) -> list[float]:
    out = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if (x0 - x) * (x1 - x) < 0 or (x1 == x and x0 != x):
            out.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return out


def test_look_four_photos_with_visibility(tmp_path: Path) -> None:
    r = SimRobot(1000, photo_dir=tmp_path)
    ph = r.look()
    assert [p.heading_deg for p in ph] == [0.0, 90.0, 180.0, 270.0] and [p.index for p in ph] == [1, 2, 3, 4]
    assert all(Path(p.path).is_file() for p in ph)
    assert sum(len(p.visible) for p in ph) > 0  # seed 1000: the hall sees boxes in the bedroom and the living room
    assert all(set(p.visible) <= set(r.objects()) and min(p.visible.values(), default=1) > 0 for p in ph)
    yaw = math.degrees(float(r.d.qpos[r.m.joint("joint_th").qposadr[0]])) % 360
    assert abs(yaw - 270.0) < 1.0  # turned by driving the yaw, last heading kept
    assert r.sim_time() > 4 * 1.0  # each turn settles 1 s: driven, not set


def test_navigate_reaches_room_and_free_drive_not_blocked() -> None:
    r = SimRobot(0)
    n = r.navigate("storage bench")
    assert n.status == "reached" and n.photo is not None and n.path_len_m > 3.0
    g = r.last_nav
    assert g["blocked"] is None and g["max_lag_m"] < follow.LAG_MAX_M / 2 and g["max_bump_n"] < follow.BUMP_N / 2
    n = r.navigate("kitchen", via="living room")
    assert n.status == "reached" and r.last_nav["blocked"] is None


@pytest.mark.parametrize("door,y", [("kitchen north door", 2.75), ("kitchen south door", 0.75)])
def test_navigate_via_kitchen_door(door: str, y: float) -> None:
    r = SimRobot(0)
    n = r.navigate("kitchen", via=door)
    assert n.status == "reached", n.detail
    ys = _x_crossing(r.last_nav["waypoints"], 4.5)
    assert len(ys) == 1 and abs(ys[0] - y) < 0.5 - ROBOT_RADIUS, ys


def test_navigate_to_door_place() -> None:
    r = SimRobot(0)
    assert r.navigate("storage door").status == "reached"
    x, y = (float(r.d.qpos[r.m.joint(j).qposadr[0]]) for j in ("joint_x", "joint_y"))
    assert math.hypot(x - 6.0, y - 5.25) < 0.05


def test_navigate_avoid_plans_around_box() -> None:
    hits = {}
    for avoid in ([], ["red box"]):
        r = SimRobot(0)
        _put_box(r, "red box", 5.1, 5.25, math.pi / 2)  # on the straight hall -> storage route
        r.look()  # perceived mode: avoid stamps the box's perceived pose, so it must have been seen
        assert "red box" in r._seen_pose
        n = r.navigate("storage", avoid=avoid)
        assert n.status == "reached"
        hits[bool(avoid)] = [k for k in n.contacts if k.startswith("red box|")]
        if avoid:
            p = np.array(r.last_nav["waypoints"])
            t = np.linspace(0, 1, 200)[:, None]
            dense = np.vstack([a + t * (b - a) for a, b in zip(p[:-1], p[1:])])
            assert np.hypot(*(dense - [5.1, 5.25]).T).min() > ROBOT_RADIUS + 0.10
    assert hits[False] and not hits[True]  # unseen: touched and reported; avoided: no contact


def test_blocked_on_welded_cabinet(monkeypatch: pytest.MonkeyPatch) -> None:
    for bump_on in (True, False):  # bumper on (default), then tracking error alone must still fire
        if not bump_on:
            monkeypatch.setattr(follow, "BUMP_N", float("inf"))
        r = SimRobot(1000, "cabinet")
        assert r.navigate("living room").status == "reached"
        door = {"d_lk1": "kitchen north door", "d_lk2": "kitchen south door"}[r.world.blocker["door"]]
        n = r.navigate("kitchen", via=door)
        g = r.last_nav
        assert n.status == "blocked" and n.photo is not None and n.photo.visible.get("cabinet", 0) > 1000
        assert g["blocked"]["why"] == ("bumper" if bump_on else "tracking")
        t_contact = min(t for k, t in g["first_contact_t"].items() if k.startswith("cabinet|"))
        assert 0 <= g["blocked"]["t"] - t_contact < 1.0
        assert max(v for k, v in g["contacts"].items() if k.startswith("cabinet|")) < 0.01
        q = r.d.qpos[[r.m.joint(j).qposadr[0] for j in ("joint_x", "joint_y")]].copy()
        r.snapshot(1.0)  # held where it stopped: no more pushing
        assert np.hypot(*(r.d.qpos[[r.m.joint(j).qposadr[0] for j in ("joint_x", "joint_y")]] - q)) < 0.005
        assert r.sim_ok()


def test_snapshot_schema_and_truth() -> None:
    r = SimRobot(1000)
    s = r.snapshot(0.5)
    assert {"boxes", "surfaces", "held", "gripper_open", "arm_retracted", "sim_ok", "settle_s"} <= set(s)  # + 2d extras
    assert set(s["boxes"]) == {"red box", "blue box", "green box", "yellow box"}
    assert set(s["surfaces"]) == {"kitchen table", "storage bench"}
    assert s["held"] is None and s["gripper_open"] and s["arm_retracted"] and s["sim_ok"]
    for n, b in s["boxes"].items():
        assert b["tilt_deg"] < 1.0 and b["max_disp_settle_m"] < 1e-3 and b["support"] == r.support_of(n)
        assert r.room_of(n) in {p["room"] for p in r.places().values()}
    assert r.pick("crate").ok is False


def test_detour_closes_other_door_as_deep_as_a_blocker() -> None:
    """2j fix 5b: navigate via one kitchen door stamps the other one closed CLOSED_DOOR_DEPTH out of each wall face side,
    so the detour path keeps clear of a cabinet standing in it (round 1/2: scraped the cabinet face, bumper)."""
    from sim.adapter import CLOSED_DOOR_DEPTH

    r = SimRobot(1200, "cabinet")
    occ = r._nav_grid("kitchen south door", ())[0].occ
    for x in (4.5 - CLOSED_DOOR_DEPTH + 0.03, 4.5 + CLOSED_DOOR_DEPTH - 0.03):
        i, j = r.grid.world_to_cell(x, 2.75)
        assert occ[j, i] and not r.grid.occ[j, i]
    i, j = r.grid.world_to_cell(4.5, 0.75)
    assert not occ[j, i]  # the via door stays open
