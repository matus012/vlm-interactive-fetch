import json

import pytest

from agent.fake_robot import FakeRobot
from agent.runtime import Runtime, label_failure
from agent.scripts import CORRECT, WRONG, run_script
from agent.skills import Skills


def snapshot_score(snap: dict, pairs: list[dict]) -> dict:
    """Minimal scorer stand-in (the real one is agent/scorer.py)."""
    rules = []
    for p in pairs:
        b = snap["boxes"][p["box"]]
        x0, y0, x1, y1 = snap["surfaces"][p["surface"]]["rect"]
        if not (x0 + .02 <= b["pos"][0] <= x1 - .02 and y0 + .02 <= b["pos"][1] <= y1 - .02) or b["support"] != p["surface"]:
            rules.append(f"{p['box']} not on {p['surface']}")
        if b["tilt_deg"] > 15:
            rules.append("tilt")
    for n, b in snap["boxes"].items():
        for s in snap["surfaces"]:
            if b["support"] == s and not any(p["box"] == n and p["surface"] == s for p in pairs):
                rules.append(f"wrong box {n} on {s}")
    if snap["held"]:
        rules.append("held at end")
    return {"pass": not rules, "failed_rules": rules}


TASKS = {
    "fetch": dict(boxes={"red box": ("bedroom", 2.5, 5.0, "floor"), "blue box": ("storage", 8.0, 5.0, "floor"),
                         "green box": ("living room", 1.0, 1.0, "floor")}, blockers={},
                  pairs=[("red box", "bedroom", "kitchen table")], blocker=None),
    "blocked": dict(boxes={"red box": ("storage", 8.0, 5.0, "floor"), "blue box": ("bedroom", 2.5, 5.0, "floor")},
                    blockers={"crate": "storage door"}, pairs=[("red box", "storage", "kitchen table")], blocker="crate"),
    "push_or_detour": dict(boxes={"red box": ("living room", 1.0, 1.0, "floor"), "blue box": ("bedroom", 2.5, 5.0, "floor")},
                           blockers={"cabinet": "kitchen north door"}, pairs=[("red box", "living room", "kitchen table")],
                           blocker="cabinet"),
    "two_box": dict(boxes={"red box": ("bedroom", 2.5, 5.0, "floor"), "blue box": ("storage", 8.0, 5.0, "floor"),
                           "green box": ("living room", 1.0, 1.0, "floor")}, blockers={},
                    pairs=[("red box", "bedroom", "kitchen table"), ("blue box", "storage", "storage bench")], blocker=None),
}


def setup(kind, tmp_path):
    t = TASKS[kind]
    robot = FakeRobot(t["boxes"], t["blockers"])
    pairs = [dict(box=b, room_point=rp, surface=s) for b, rp, s in t["pairs"]]
    other = next(b for b in t["boxes"] if b not in [p["box"] for p in pairs] and t["boxes"][b][0] != "storage")
    truth = {"pairs": pairs, "blocker": t["blocker"], "detour_via": "kitchen south door",
             "wrong_box": {"box": other, "room_point": t["boxes"][other][0]},
             "wrong_surface": "storage bench" if pairs[0]["surface"] == "kitchen table" else "kitchen table"}
    task = {"type": kind, "step_limit": 30 if kind == "two_box" else 20}
    rt = Runtime(Skills(robot), robot, task, tmp_path / "run.jsonl")
    return robot, rt, task, truth


@pytest.mark.parametrize("kind", list(TASKS))
def test_correct_scripts_pass(kind, tmp_path):
    robot, rt, task, truth = setup(kind, tmp_path)
    n = run_script(rt, task, truth)
    sc = snapshot_score(robot.snapshot(), truth["pairs"])
    end = rt.finish(sc)
    assert sc["pass"], sc
    assert end["label"] is None and end["done_called"] and end["steps_used"] == n <= task["step_limit"]


@pytest.mark.parametrize("branch", ["push_then_detour", "detour"])
def test_push_or_detour_branches(branch, tmp_path):
    robot, rt, task, truth = setup("push_or_detour", tmp_path)
    CORRECT["push_or_detour"](rt, task, truth, branch)
    calls = [(s["call"], s["result"]) for s in rt.steps]
    assert ("push", "failed: not movable") in calls if branch == "push_then_detour" else ("push", "failed: not movable") not in calls
    assert ("navigate", "failed: blocked") in calls
    assert snapshot_score(robot.snapshot(), truth["pairs"])["pass"]


def test_blocked_sequence(tmp_path):
    robot, rt, task, truth = setup("blocked", tmp_path)
    run_script(rt, task, truth)
    assert [(s["call"], s["result"]) for s in rt.steps][:4] == [
        ("navigate", "failed: blocked"), ("push", "pushed"), ("navigate", "reached"), ("look", "4 photos")]


EXPECT = {"wrong_colour": "not on", "wrong_surface": "not on", "box_in_gripper": "held at end"}


@pytest.mark.parametrize("kind", ["fetch", "blocked", "push_or_detour"])
@pytest.mark.parametrize("wrong", list(WRONG))
def test_wrong_scripts_fail_for_their_reason(kind, wrong, tmp_path):
    robot, rt, task, truth = setup(kind, tmp_path)
    n = WRONG[wrong](rt, task, truth)
    sc = snapshot_score(robot.snapshot(), truth["pairs"])
    end = rt.finish(sc)
    if wrong == "step_limit":
        assert n == task["step_limit"] and end["step_limit_hit"] and not end["done_called"]
        assert not sc["pass"]
        assert rt.call("look", {}).text == "error: step limit reached" and len(rt.steps) == task["step_limit"]
    else:
        assert not sc["pass"] and any(EXPECT[wrong] in r for r in sc["failed_rules"]), sc
    assert end["label"] is not None


def test_name_checks_count_as_steps(tmp_path):
    robot, rt, task, truth = setup("fetch", tmp_path)
    assert rt.call("navigate", {"place": "x=1.2, y=0.5"}).text == "error: unknown place 'x=1.2, y=0.5'"
    assert rt.call("navigate", {"place": "kitchen", "via": "(1,2)"}).text == "error: unknown place '(1,2)'"
    assert rt.call("pick", {"object": "purple box"}).text == "error: unknown object 'purple box'"
    assert rt.call("place", {"surface": "kitchen"}).text == "error: unknown surface 'kitchen'"
    assert rt.call("fly", {}).text.startswith("error: unknown skill")
    assert rt.call("look", {"x": "1"}).text.startswith("error: look: unexpected argument")
    assert rt.call("navigate", {}).text.startswith("error: navigate: missing")
    assert len(rt.steps) == 7 and not any(s["ok"] for s in rt.steps)


def test_step_limit_and_done(tmp_path):
    robot, rt, task, truth = setup("fetch", tmp_path)
    for _ in range(20):
        assert rt.call("look", {}).ok
    assert rt.over
    assert rt.call("look", {}).text == "error: step limit reached"
    assert len(rt.steps) == 20
    end = rt.finish({"pass": False})
    assert end["step_limit_hit"] and end["label"] == "skill"  # scripted run hitting the limit is never vlm
    _, rt2, _, _ = setup("fetch", tmp_path)
    rt2.call("done", {"note": "n"})
    assert rt2.call("look", {}).text == "error: run ended"


def test_done_as_step_20_passes_the_steps_rule(tmp_path):
    """Fix round 2: `done` arriving exactly as step 20 of a 20-step run is within the limit: steps_used 20
    (gate2's steps_used = len(steps) + refused calls), not step_limit_hit, scorer steps rule True."""
    from agent.scorer import score

    robot, rt, task, truth = setup("fetch", tmp_path)
    for _ in range(19):
        assert rt.call("look", {}).ok
    assert not rt.over and rt.call("done", {"note": "n"}).ok
    assert len(rt.steps) == 20 and rt.n_refused == 0
    end = rt.finish({"pass": True})
    assert end["done_called"] and not end["step_limit_hit"] and end["steps_used"] == 20
    snap = {"boxes": {}, "surfaces": {}, "settle_s": 2.0, "held": None, "sim_ok": True}
    sc = score(snap, {"targets": [], "step_limit": task["step_limit"]}, len(rt.steps) + rt.n_refused)
    assert sc["rules"]["steps"] and sc["pass"]
    assert not score(snap, {"targets": [], "step_limit": task["step_limit"]}, 21)["rules"]["steps"]


def test_jsonl_contents(tmp_path):
    robot, rt, task, truth = setup("blocked", tmp_path)
    run_script(rt, task, truth)
    rt.finish({"pass": True})
    lines = [json.loads(x) for x in (tmp_path / "run.jsonl").read_text().splitlines()]
    steps, end = lines[:-1], lines[-1]
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    for s in steps:
        assert {"step", "t_wall", "call", "args", "result", "ok", "photos", "sim_s"} <= s.keys()
    look = next(s for s in steps if s["call"] == "look")
    assert len(look["photos"]) == 4 and all("visible" in p and "path" in p for p in look["photos"])
    assert end["event"] == "end" and end["steps_used"] == len(steps) and end["done_called"] and end["score"] == {"pass": True}
    assert end["label"] is None


def S(call, result):
    return {"call": call, "result": result, "ok": not result.startswith(("failed", "error"))}


def test_label_rule():
    base = {"sim_ok": True}
    assert label_failure([S("look", "4 photos")], {"sim_ok": False}, True) == "sim"
    assert label_failure([S("navigate", "failed: blocked")], base, True) == "nav"
    assert label_failure([S("navigate", "failed: blocked"), S("push", "pushed"), S("navigate", "reached")], base, True) == "scorer"
    assert label_failure([S("locate", "failed: not in view")], base, True) == "perception"
    assert label_failure([S("pick", "failed: missed")], base, True) == "skill"
    assert label_failure([S("place", "failed: slipped")], base, True) == "skill"
    assert label_failure([S("push", "failed: not movable"), S("navigate", "reached")], base, True) == "scorer"
    assert label_failure([S("push", "failed: not movable")], base, True) == "skill"
    assert label_failure([S("pick", "failed: missed"), S("pick", "holding x")], base, True) == "scorer"
    assert label_failure([S("pick", "error: unknown object 'x'")], base, True) == "skill"
    assert label_failure([S("pick", "error: unknown object 'x'")], base, False) == "vlm"
    assert label_failure([S("look", "4 photos")], {**base, "step_limit_hit": True}, False) == "vlm"
    assert label_failure([S("done", "done")], base, False) == "vlm"


def A(call, result, **args):
    return {**S(call, result), "args": args}


def test_agent_label_rule():
    base = {"sim_ok": True}
    lab = lambda steps, end=base: label_failure(steps, end, False)  # noqa: E731
    # correct failures, run then fails: the VLM
    for r in ("failed: not in view", "failed: blocked", "failed: not movable", "failed: stuck", "failed: not blocking",
              "failed: not located", "failed: nothing held", "failed: already holding something", "failed: not a box"):
        for call in ("navigate", "locate", "push", "pick", "place"):
            assert lab([S(call, r)]) == "vlm", (call, r)
    assert lab([S("navigate", "failed: blocked"), S("push", "failed: stuck")]) == "vlm"
    # wrong results
    assert lab([S("navigate", "failed: no path")]) == "nav"
    for r in ("failed: missed", "failed: slipped", "failed: unreachable"):
        assert lab([S("pick", r)]) == "skill"
    assert lab([S("place", "failed: dropped")]) == "skill"
    assert lab([S("place", "failed: unreachable")]) == "skill"
    # recovered wrong failures are not blamed
    assert lab([S("pick", "failed: missed"), A("pick", "holding x", object="x")]) == "vlm"
    assert lab([S("navigate", "failed: no path"), S("navigate", "reached")]) == "vlm"
    # last unrecovered wrong failure wins; a later correct failure does not hide it
    assert lab([S("navigate", "failed: no path"), S("pick", "failed: missed"), S("locate", "failed: not in view")]) == "skill"
    # errors, step limit, wrong sequence, never done
    assert lab([S("pick", "error: unknown object 'x'")]) == "vlm"
    assert lab([S("look", "4 photos")], {**base, "step_limit_hit": True}) == "vlm"
    assert lab([S("done", "done")]) == "vlm"
    # sim wins
    assert lab([S("navigate", "failed: no path")], {"sim_ok": False}) == "sim"
    # placed ok on the target surface but the scorer says that box is not on it -> skill
    steps = [A("pick", "holding red box", object="red box"), A("place", "placed", surface="table")]
    sc = lambda ok, surf="table": {**base, "score": {"detail": {"targets": {"red box": {"ok": ok, "surface": surf}}}}}  # noqa: E731
    assert lab(steps, sc(False)) == "skill"
    assert lab(steps, sc(False, "bed")) == "vlm"  # placed on another surface: the VLM's choice
    assert lab(steps[:1], sc(False)) == "vlm"  # never placed
    # two boxes: one placed fine, the other never placed -> vlm; the other placed and not on surface -> skill
    two = {"detail": {"targets": {"red box": {"ok": True, "surface": "table"}, "blue box": {"ok": False, "surface": "desk"}}}}
    s2 = steps + [A("pick", "failed: slipped", object="blue box")]
    assert lab(steps, {**base, "score": two}) == "vlm"
    s3 = steps + [A("pick", "holding blue box", object="blue box"), A("place", "placed", surface="desk")]
    assert lab(s3, {**base, "score": two}) == "skill"
    assert lab(s2, {**base, "score": two}) == "skill"  # unrecovered wrong pick


def test_scripted_label_unchanged_by_agent_rule():
    base = {"sim_ok": True}
    assert label_failure([S("navigate", "failed: blocked")], base, True) == "nav"
    assert label_failure([S("locate", "failed: not in view")], base, True) == "perception"
    assert label_failure([S("push", "failed: stuck")], base, True) == "skill"
    assert label_failure([S("place", "placed")], {**base, "score": {"detail": {"targets": {"a": {"ok": False, "surface": "t"}}}}}, True) == "scorer"


def test_rules_text_clean():
    from agent.context import SYSTEM
    from agent.leak import scan
    assert scan(SYSTEM) == []
    assert "navigate to the surface first" not in SYSTEM


def test_agent_label_lost_box_is_skill() -> None:
    """Ruling 2026-11-13 (row 14): a held box lost without any skill saying so is a wrong result -> skill (tuning
    blocked-8230: push 'stuck' while carrying, then place -> 'failed: nothing held')."""
    from agent.runtime import label_failure

    def st(call, args, result, ok):
        return {"call": call, "args": args, "result": result, "ok": ok}
    base = [st("pick", {"object": "yellow box"}, "holding yellow box", True),
            st("place", {"surface": "storage bench"}, "failed: blocked", False),
            st("push", {"object": "crate"}, "failed: stuck", False)]
    end = {"sim_ok": True, "score": {"pass": False, "detail": {"held": None, "targets": {}}}}
    assert label_failure(base + [st("place", {"surface": "storage bench"}, "failed: nothing held", False)], end,
                         scripted=False) == "skill"
    assert label_failure(base, end, scripted=False) == "skill"  # ends not held, never released
    held_end = {"sim_ok": True, "score": {"pass": False, "detail": {"held": "yellow box", "targets": {}}}}
    assert label_failure(base, held_end, scripted=False) == "vlm"  # still held: correct failures only
