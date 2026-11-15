"""Task 005 row 16: pick / place whose own drive stops on the crate / cabinet say "failed: blocked" (labelled vlm);
every other drive failure stays "failed: unreachable" (labelled skill)."""
from types import SimpleNamespace

import numpy as np

import sim.grasp as G
from agent.runtime import label_failure
from tune_replay import replay


def nav(contacts: dict, why: str | None = "bumper") -> dict:
    return {"reached": False, "pos_err_m": 1.0, "yaw_err_deg": 0.0, "path_len_m": 2.0, "sim_s": 1.0,
            "contacts": contacts, "blocked": None if why is None else {"why": why}}


def test_stopped_by_blocker() -> None:
    assert G.stopped_by_blocker(nav({"crate|49": 0.002}))
    assert G.stopped_by_blocker(nav({"box_red|left_pad1": 1e-4, "cabinet|49": 0.002}, "tracking"))
    assert not G.stopped_by_blocker(nav({"crate|49": 0.002}, None))  # contact, detector never fired
    assert not G.stopped_by_blocker(nav({"wall07|49": 0.002}))  # stopped on a wall
    assert not G.stopped_by_blocker(nav({"box_red|left_pad1": 1e-4}, "no_progress"))


def _place(monkeypatch, n: dict) -> dict:
    monkeypatch.setattr(G, "holding", lambda *a: True)
    monkeypatch.setattr(G, "go_to_keep_arm", lambda *a, **k: n)
    surf = {"name": "storage bench", "rect": [6.9, 6.475, 8.1, 6.925], "top_z": 0.45, "approach": (7.5, 5.95, 1.57)}
    return G.place_box(None, SimpleNamespace(time=0.0), None, surf, "box_red", {})


def _pick(monkeypatch, navs: list[dict]) -> dict:
    it = iter(navs)
    monkeypatch.setattr(G, "tilt_deg", lambda *a: 0.0)
    monkeypatch.setattr(G, "plan_pregrasp_at", lambda *a: [{"standoff": (0.0, 0.0, 0.0)}] * len(navs))
    monkeypatch.setattr(G, "mark_discs", lambda g, discs: g)
    monkeypatch.setattr(G.follow, "go_to", lambda *a, **k: next(it))
    d = SimpleNamespace(time=0.0, body=lambda b: SimpleNamespace(xmat=np.eye(3).ravel(), xpos=np.zeros(3)))
    return G.pick_box(None, d, None, "box_green", [], pose_fn=lambda: (np.zeros(3), 0.0))


def test_place_and_pick_reasons(monkeypatch) -> None:
    assert _place(monkeypatch, nav({"cabinet|49": 0.002}))["reason"] == "blocked"
    assert _place(monkeypatch, nav({"wall07|49": 0.002}))["reason"] == "unreachable"
    assert _place(monkeypatch, nav({}, None))["reason"] == "unreachable"  # goal tolerance missed, no detector
    assert _pick(monkeypatch, [nav({"crate|49": 0.003})] * 3)["reason"] == "blocked"
    assert _pick(monkeypatch, [nav({"crate|49": 0.003}), nav({"wall07|49": 0.001}), nav({"crate|49": 0.001})]
                 )["reason"] == "unreachable"  # one candidate failed for another reason


def test_labels() -> None:
    S = lambda c, r: {"call": c, "result": r, "ok": False}  # noqa: E731
    for c in ("pick", "place"):
        assert label_failure([S(c, "failed: blocked")], {"sim_ok": True}, False) == "vlm"
        assert label_failure([S(c, "failed: unreachable")], {"sim_ok": True}, False) == "skill"


def test_place_through_cabinet_door_is_blocked_box_held(tmp_path) -> None:
    """Tuning push_or_detour-8441 steps 1-19 (holding the green box in the kitchen, cabinet in d_lk1), then step 20
    place(storage bench): the drive to the approach point stops on the cabinet -> failed: blocked, box still held."""
    robot, rt = replay("push_or_detour-8441", tmp_path)
    assert robot.held() == "green box" and rt.steps[-1]["result"] == "failed: blocked"
    r = rt.call("place", {"surface": "storage bench"})
    assert r.text == "failed: blocked", r.text
    g = r.meta["detail"]["go_to"]
    assert g["blocked"] is not None and any(k.startswith("cabinet|") for k in g["contacts"])
    assert robot.held() == "green box"


def test_pick_behind_crate_is_blocked(tmp_path) -> None:
    """Tuning blocked-8218 steps 1-17 (green box located in the bedroom behind the never-pushed crate), then step 18
    pick: every candidate's drive stops on the crate -> failed: blocked."""
    robot, rt = replay("blocked-8218", tmp_path)
    r = rt.call("pick", {"object": "green box"})
    assert r.text == "failed: blocked", r.text
    gs = r.meta["detail"]["go_to"]
    assert gs and all(g["blocked"] is not None and any(k.startswith("crate|") for k in g["contacts"]) for g in gs)
    assert robot.held() is None
