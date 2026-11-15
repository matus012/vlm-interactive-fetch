"""Task 014 (E4): generator rules on dev seeds, the scorer's success rule per category (mutants caught), the text leak.

The scorer is unchanged (agent/scorer.py byte-identical to freeze-006): an E4 task's targets name the intended box, so
"the intended box on the surface, nothing held, done within the limit" is exactly surface + wrong_box + held + steps.
The per-category mutant check: for every category a snapshot with the intended box placed passes, and each mutant
(a different box placed instead, the intended box left on the floor, both placed, intended box still held) fails.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

import agent.scorer as scorer
from agent.leak import scan
from tests.test_scorer import KT, box, snap

COLOURS = ("red", "blue", "green", "yellow")
CATS = ("vague", "place_ref", "recovery", "hint")


def e4_task(cat: str, target: str = "green box") -> dict:
    text = {"vague": "Bring something to drink to the kitchen table.",
            "place_ref": "Bring the box next to the sofa to the kitchen table.",
            "recovery": f"Bring the {target} to the kitchen table.",
            "hint": f"Bring the {target} to the kitchen table; someone left it in the bedroom."}[cat]
    return {"task_id": f"{cat}-1000", "type": cat, "category": cat, "seed": 1000, "blocker": "none",
            "instruction": text, "targets": [{"object": target, "surface": KT}], "step_limit": 20, "e4": True}


def placed(name: str) -> dict:
    return box(0.5, 0.5, support=KT) if name else {}


@pytest.mark.parametrize("cat", CATS)
def test_success_rule_per_category(cat: str) -> None:
    t = e4_task(cat)
    floor = lambda x: box(x, 2.0, 0.06, support="floor")  # noqa: E731
    good = snap(**{"red box": floor(2.0), "green box": box(0.5, 0.5, support=KT)})
    assert scorer.score(good, t, 9)["pass"]
    mutants = {
        "other box placed": snap(**{"red box": box(0.5, 0.5, support=KT), "green box": floor(5.0)}),
        "intended left on floor": snap(**{"red box": floor(2.0), "green box": floor(5.0)}),
        "both placed": snap(**{"red box": box(0.3, 0.3, support=KT), "green box": box(0.7, 0.7, support=KT)}),
    }
    held = copy.deepcopy(good)
    held["held"] = "green box"
    mutants["intended still held"] = held
    for name, s in mutants.items():
        assert not scorer.score(s, t, 9)["pass"], (cat, name)
    assert not scorer.score(good, t, 21)["pass"], (cat, "over the step limit")


def _has_freeze_tag() -> bool:
    import subprocess
    root = Path(scorer.__file__).resolve().parents[1]
    return subprocess.run(["git", "-C", str(root), "rev-parse", "-q", "--verify", "freeze-006^{commit}"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_freeze_tag(), reason="needs the development history (tag freeze-006), not in the "
                    "public one-commit snapshot")
def test_scorer_unchanged_since_freeze() -> None:
    import subprocess
    root = Path(scorer.__file__).resolve().parents[1]
    frozen = subprocess.run(["git", "-C", str(root), "show", "freeze-006:agent/scorer.py"], capture_output=True).stdout
    assert frozen and hashlib.sha256(frozen).digest() == hashlib.sha256(Path(scorer.__file__).read_bytes()).digest()


@pytest.fixture(scope="module")
def dev_tasks() -> list[dict]:
    from sim.e4 import make_task_e4
    from sim.tasks import NoTask
    out = []
    for cat in ("place_ref", "recovery", "hint"):
        n = 0
        for seed in range(2000, 2100):
            try:
                out.append(make_task_e4(seed, cat))
                n += 1
            except NoTask:
                continue
            if n == 4:
                break
    return out


def test_dev_tasks_rules(dev_tasks: list[dict]) -> None:
    from sim.e4 import ANCHOR_TEXT, NEAR_M, UNIQUE_M, make_task_e4
    from sim import tasks as T
    from sim.world import generate as gen
    for t in dev_tasks:
        assert t == make_task_e4(t["seed"], t["category"])  # deterministic
        assert t["blocker"] == "none" and t["step_limit"] == 20 and len(t["targets"]) == 1
        box = t["targets"][0]["object"]
        w = T.world(t["seed"], "none")
        b = next(x for x in w.boxes if T.box_name(x["colour"]) == box)
        assert T.hall_px(t["seed"], "none")[box] < 10 and box in T.pregrasp_ok(t["seed"], "none")
        text = t["instruction"]
        assert not scan(text), text
        if t["category"] == "place_ref":
            assert not any(c in text.lower() for c in COLOURS), text
            assert ANCHOR_TEXT[t["anchor"]] in text
            a = t["anchor"]
            if a in ("coffee table", "nightstand"):
                assert b["surface"] == a
            else:
                f = next(f for f in w.layout["furniture"] if f["id"] == a)
                d = gen.rect_dist(b["x"], b["y"], gen.furniture_rect(f))
                assert d <= NEAR_M
                for o in w.boxes:
                    if o is not b:
                        assert gen.rect_dist(o["x"], o["y"], gen.furniture_rect(f)) >= d + UNIQUE_M
        if t["category"] == "hint":
            assert text.endswith(f"someone left it in the {b['room'].replace('_', ' ')}.")
        if t["category"] == "recovery":
            assert t["move"] and all(m[0] != b["room"] for m in t["move"])
            for room, x, y, _ in t["move"]:
                assert gen.spot_ok(w.layout, room, x, y)


def test_packet_text_no_colour_leak_for_colourless_categories(dev_tasks: list[dict]) -> None:
    """E4-leak (static half): the Task line of a place-referenced / vague task names no colour; no digits."""
    for t in dev_tasks:
        if t["category"] in ("place_ref", "vague"):
            assert not re.search(r"\b(red|blue|green|yellow)\b", t["instruction"], re.I)
        assert not re.search(r"\d", t["instruction"])


def test_want_slots_honoured() -> None:
    """review 014 B2 / S2: recovery / hint honour the sampler slot; place_ref the relation; the slot is stored."""
    from sim import tasks as T
    from sim.e4 import make_task_e4
    from sim.sampler import plan
    for cat in ("recovery", "hint"):
        for want in plan("fetch", 4, 3100):
            for seed in range(3100, 3300):
                try:
                    t = make_task_e4(seed, cat, want=want)
                except T.NoTask:
                    continue
                room, sup = T.start_of(seed, "none", t["targets"][0]["object"])
                assert (room, sup, t["targets"][0]["surface"]) == (want["room"], want["support"], want["surface"])
                assert t["slot"] == want and t == make_task_e4(seed, cat, want=want)
                break
    for rel in ("on", "next"):
        for seed in range(3100, 3300):
            try:
                t = make_task_e4(seed, "place_ref", want={"relation": rel})
            except T.NoTask:
                continue
            assert (t["anchor"] in ("coffee table", "nightstand")) == (rel == "on") and t["slot"] == {"relation": rel}
            break
