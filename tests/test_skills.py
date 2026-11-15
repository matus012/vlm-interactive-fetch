import pytest

from agent.fake_robot import FakeRobot
from agent.skills import Skills


def world(**kw):
    r = FakeRobot({"red box": ("bedroom", 2.5, 5.0, "floor"), "blue box": ("storage", 8.0, 5.0, "floor")}, **kw)
    return r, Skills(r)


def test_navigate_look_locate_pick_place_strings():
    r, s = world()
    n = s.navigate("bedroom")
    assert n.text == "reached" and n.ok and len(n.photos) == 1
    assert s.locate("blue box").text == "failed: not in view"
    lk = s.look()
    assert lk.text == "4 photos" and len(lk.photos) == 4
    assert "red box" in s.seen
    assert s.locate("red box").text == "found: red box in bedroom, on floor"
    assert s.pick("red box").text == "holding red box"
    assert s.navigate("kitchen table").text == "reached"
    assert s.place("kitchen table").text == "placed"
    assert s.done("x").text == "done"


def test_failures_and_names():
    r, s = world(blockers={"crate": "storage door", "cabinet": "kitchen north door"})
    assert s.pick("red box").text == "failed: not located"
    assert s.place("kitchen table").text == "failed: nothing held"
    assert s.pick("crate").text == "failed: not a box"
    assert s.navigate("storage").text == "failed: blocked"
    assert s.push("cabinet").text == "failed: not movable"
    assert s.push("red box").text == "failed: not movable"
    assert s.push("crate").text == "pushed"
    assert s.navigate("storage").text == "reached"
    assert s.navigate("nowhere").text == "error: unknown place 'nowhere'"
    assert s.navigate("kitchen", via="oven").text == "error: unknown place 'oven'"
    assert s.navigate("kitchen", via="kitchen south door").ok
    assert s.locate("pink box").text == "error: unknown object 'pink box'"
    assert s.place("kitchen").text == "error: unknown surface 'kitchen'"


def test_pick_failure_reason_and_second_pick():
    r, s = world()
    s.navigate("bedroom"); s.look(); s.locate("red box")
    r.pick_fail.add("red box")
    assert s.pick("red box").text == "failed: missed"
    r.pick_fail.clear()
    assert s.pick("red box").text == "holding red box"
    s.locate("red box")
    assert s.pick("red box").text in ("failed: not located", "failed: already holding something")


def test_navigate_avoids_seen_unheld_boxes_only():
    seen_avoid = []
    r, s = world()
    orig = r.navigate
    r.navigate = lambda p, v, a: (seen_avoid.append(a), orig(p, v, a))[1]
    s.navigate("bedroom"); s.look(); s.locate("red box"); s.pick("red box")
    s.navigate("hall")
    assert seen_avoid[0] == [] and seen_avoid[-1] == []  # held box not avoided


def test_no_poses_in_text():
    r, s = world()
    s.navigate("bedroom"); s.look()
    t = s.locate("red box").text
    assert not any(c.isdigit() for c in t)
