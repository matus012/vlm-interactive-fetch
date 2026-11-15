"""Task 004: one step count, packet, leak scanner, VLM loop (stub VLM, FakeRobot), G3-log check, robot_room."""
import json
import re
from pathlib import Path

import pytest

from agent.context import OBJECTS, SYSTEM, build_packet
from agent.fake_robot import FakeRobot
from agent.leak import scan, scan_run
from agent.loop import ONE_TOOL, check_log, end_run, run_agent
from agent.runtime import Runtime
from agent.scorer import score
from agent.skills import Skills
from agent.vlm import StubVLM

BOXES = {"red box": ("bedroom", 2.5, 5.0, "floor"), "blue box": ("storage", 8.0, 5.0, "floor"),
         "green box": ("living room", 1.0, 1.0, "floor")}
TASK = {"task_id": "fetch-4000", "type": "fetch", "seed": 4000, "blocker": "none", "step_limit": 20,
        "instruction": "Bring the red box to the kitchen table.",
        "targets": [{"object": "red box", "surface": "kitchen table"}]}


def C(name, **args):
    return {"name": name, "arguments": args}


CORRECT = [C("navigate", place="bedroom"), C("look"), C("locate", object="red box"), C("pick", object="red box"),
           C("navigate", place="kitchen table"), C("place", surface="kitchen table"), C("done", note="finished")]


def run(tmp_path, items, boxes=BOXES):
    robot = FakeRobot(boxes)
    sk = Skills(robot)
    rt = Runtime(sk, robot, TASK, tmp_path / "log.jsonl", scripted=False)
    run_agent(robot, sk, rt, TASK, StubVLM(items), tmp_path)
    end = end_run(robot, rt, TASK)
    (tmp_path / "result.json").write_text(json.dumps({"pass": end["score"]["pass"], "label": end["label"],
                                                      "steps_used": end["steps_used"]}))
    return robot, sk, rt, end


def steps_of(tmp_path):
    return [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()][:-1]


# ---- one step count -------------------------------------------------------------------------------------------

def test_done_as_step_20_of_20_passes(tmp_path):
    _, _, rt, end = run(tmp_path, [C("look")] * 19 + [C("done", note="n")])
    assert rt.steps_used == end["steps_used"] == end["score"]["detail"]["steps_used"] == 20
    assert end["score"]["rules"]["steps"] and not end["step_limit_hit"]


def test_no_done_by_step_20_counts_21_and_fails(tmp_path):
    _, _, rt, end = run(tmp_path, [C("look")] * 25)
    assert len(rt.steps) == 20 and rt.n_refused == 0
    assert rt.steps_used == end["steps_used"] == end["score"]["detail"]["steps_used"] == 21
    assert not end["score"]["rules"]["steps"] and end["step_limit_hit"] and end["label"] == "vlm"


def test_refused_call_counts_once(tmp_path):
    robot = FakeRobot(BOXES)
    rt = Runtime(Skills(robot), robot, TASK, tmp_path / "l.jsonl")
    for _ in range(20):
        rt.call("look")
    assert rt.call("done", {"note": "late"}).text == "error: step limit reached"
    assert rt.steps_used == 21
    assert rt.finish(score(robot.snapshot(), TASK, rt.steps_used))["steps_used"] == 21


def test_finish_refuses_a_scorer_with_another_count(tmp_path):
    robot = FakeRobot(BOXES)
    rt = Runtime(Skills(robot), robot, TASK, tmp_path / "l.jsonl")
    rt.call("done", {"note": "n"})
    with pytest.raises(AssertionError):
        rt.finish(score(robot.snapshot(), TASK, rt.steps_used + 1))


# ---- loop ------------------------------------------------------------------------------------------------------

def test_correct_stub_passes(tmp_path):
    _, _, rt, end = run(tmp_path, CORRECT)
    assert end["score"]["pass"], end["score"]
    assert end["label"] is None and end["steps_used"] == 7 and end["n_validation_errors"] == 0
    assert end["places_reached"] == ["bedroom", "kitchen table"] and end["kitchen_east_calls"] == 0
    assert end["leak_hits"] == 0


def test_malformed_calls_are_error_steps(tmp_path):
    bad = [[], [C("look"), C("look")], {"name": "navigate", "arguments": "{bad json"}, C("fly"),
           C("navigate", place="attic"), C("pick", object="red box")]
    _, _, rt, end = run(tmp_path, bad + CORRECT)
    res = [s["result"] for s in rt.steps]
    assert res[:2] == [ONE_TOOL, ONE_TOOL]
    assert res[2].startswith("error: arguments are not valid JSON") and res[3].startswith("error: unknown skill")
    assert res[4] == "error: unknown place 'attic'" and res[5] == "failed: not located"
    assert end["n_validation_errors"] == 5 and end["steps_used"] == 13 and end["score"]["pass"]
    st = steps_of(tmp_path)
    assert st[1]["extra"]["vlm"]["tool_calls"] == [{"name": "look", "arguments": "{}"}] * 2
    assert st[1]["extra"]["parsed_call"] is None and st[1]["call"] == ""
    # each error is the next packet's last result and is in its history
    assert f"Last result: {ONE_TOOL}" in st[1]["extra"]["packet"]["user"]
    assert "1. no single tool call -> error: call exactly one tool" in st[1]["extra"]["packet"]["user"]
    assert "Last result: error: unknown place 'attic'" in st[5]["extra"]["packet"]["user"]
    # the JSON decoder's line / column numbers never reach the packet
    assert "line # column #" in st[3]["extra"]["packet"]["user"] and end["leak_hits"] == 0


def test_step_limit_ends_the_loop(tmp_path):
    stub = StubVLM([C("look")] * 30)
    robot = FakeRobot(BOXES)
    sk = Skills(robot)
    rt = Runtime(sk, robot, TASK, tmp_path / "log.jsonl", scripted=False)
    run_agent(robot, sk, rt, TASK, stub, tmp_path)
    assert stub.n == 20 and len(rt.steps) == 20 and rt.over and not rt.done_called


def test_seed_per_step(tmp_path):
    run(tmp_path, CORRECT)
    assert [s["extra"]["seed"] for s in steps_of(tmp_path)] == [400001 + i for i in range(7)]


# ---- packet ----------------------------------------------------------------------------------------------------

def test_packet_contents(tmp_path):
    robot = FakeRobot(BOXES)
    sk = Skills(robot)
    rt = Runtime(sk, robot, TASK, tmp_path / "l.jsonl", scripted=False)
    p0 = build_packet(robot, sk, rt, TASK)
    assert p0.photos == [] and "No photos yet" in p0.user and "History:\n(none)" in p0.user
    rt.call("navigate", {"place": "bedroom"})
    p1 = build_packet(robot, sk, rt, TASK)
    assert p1.labels == ["Photo 1"] and "from your last navigate" in p1.user
    rt.call("look")
    p = build_packet(robot, sk, rt, TASK)
    u = p.user
    assert u.startswith("Task: Bring the red box to the kitchen table.\nStep 3 of 20.\nYou are in: bedroom.")
    for line in ("- kitchen: room look spot, in kitchen", "- kitchen east: room look spot, in kitchen",
                 "- kitchen table: surface, in kitchen", "- storage door: door, in storage",
                 "Surfaces (for place): kitchen table, storage bench.", f"Objects: {', '.join(OBJECTS)}.",
                 "1. navigate(place='bedroom') -> reached", "2. look() -> 4 photos", "Last result: 4 photos",
                 "Current photos (Photo 1 to Photo 4, from your last look)"):
        assert line in u, line
    assert p.labels == [f"Photo {i}" for i in range(1, 5)] and len(p.photos) == 4
    assert p.system == SYSTEM
    for rule in ("exactly one tool", "only if it is visible in the current photos", "Needs that box located",
                 "Needs a held box", "one new photo", "four new photos", "two look spots", "Call done",
                 "counts as a failure"):
        assert rule in SYSTEM, rule


def test_first_packet_does_not_depend_on_where_the_boxes_are(tmp_path):
    other = {"red box": ("storage", 8.0, 5.0, "floor"), "yellow box": ("kitchen", 6.0, 1.0, "floor")}
    texts = []
    for i, boxes in enumerate((BOXES, other)):
        robot = FakeRobot(boxes)
        sk = Skills(robot)
        rt = Runtime(sk, robot, TASK, tmp_path / f"l{i}.jsonl", scripted=False)
        texts.append(build_packet(robot, sk, rt, TASK))
    assert texts[0] == texts[1]


def test_vlm_echo_digits_masked(tmp_path):
    robot = FakeRobot(BOXES)
    sk = Skills(robot)
    rt = Runtime(sk, robot, TASK, tmp_path / "l.jsonl", scripted=False)
    rt.call("navigate", {"place": "spot 12"})
    u = build_packet(robot, sk, rt, TASK).user
    assert "navigate(place='spot ##') -> error: unknown place 'spot ##'" in u and not scan(u)


# ---- leak scanner ----------------------------------------------------------------------------------------------

def test_clean_run_has_no_hits(tmp_path):
    run(tmp_path, CORRECT)
    assert scan_run(tmp_path) == []


@pytest.mark.parametrize("cls,text", [
    ("coord", "the red box is at (2.5, 5.0)"), ("coord", "goal x=2 y=3"), ("coord", "pose [2, 5]"),
    ("decimal", "the box is 0.75 away"), ("word", "move 3 metres"), ("word", "table 75 cm high"),
    ("word", "arm joint angles"), ("word", "the occupancy map"), ("word", "your pose"), ("word", "yaw of the box"),
    ("brand", "you are a TidyBot"), ("brand", "Kinova Gen3 arm"), ("brand", "Robotiq 2F-85 gripper"),
    ("brand", "simulated in MuJoCo"), ("digit", "the box is 3 steps away"),
])
def test_planted_leak_caught(cls, text):
    assert cls in {c for c, _ in scan(text)}, scan(text)


def test_planted_leak_in_a_run_log(tmp_path):
    run(tmp_path, CORRECT)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    s = json.loads(lines[2])
    s["extra"]["packet"]["user"] += "\nred box at (2.5, 5.0)"
    lines[2] = json.dumps(s)
    (tmp_path / "log.jsonl").write_text("\n".join(lines) + "\n")
    hits = scan_run(tmp_path)
    assert {c for _, c, _ in hits} == {"coord", "decimal"} and all(w == "step 3 user" for w, _, _ in hits)


def test_allowlisted_counters_not_flagged():
    text = "Step 12 of 20.\nHistory:\n1. look() -> 4 photos\n12. done(note='x') -> done\nPhoto 1 to Photo 4"
    assert scan(text) == []
    assert scan("Step 12 of 20, then 7") == [("digit", "7")]


def test_cli_exit_codes(tmp_path):
    from agent.leak import main
    run(tmp_path, CORRECT)
    assert main([str(tmp_path)]) == 0
    (tmp_path / "tools.json").write_text('{"description": "a TidyBot"}')
    assert main([str(tmp_path)]) == 1


# ---- G3-log ----------------------------------------------------------------------------------------------------

def test_log_check_complete_and_failed_label(tmp_path):
    run(tmp_path / "a", CORRECT)
    assert check_log(tmp_path / "a") == []
    _, _, _, end = run(tmp_path / "b", [C("look")] * 20)
    assert end["label"] == "vlm" and check_log(tmp_path / "b") == []


def test_log_check_catches_gaps(tmp_path):
    run(tmp_path, CORRECT)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    s = json.loads(lines[1])
    del s["extra"]["vlm"]["reasoning"]
    end = json.loads(lines[-1])
    end["score"]["pass"], end["label"] = False, None
    (tmp_path / "log.jsonl").write_text("\n".join([lines[0], json.dumps(s), *lines[2:-1], json.dumps(end)]) + "\n")
    probs = check_log(tmp_path)
    assert any("vlm.reasoning" in p for p in probs) and any("label" in p for p in probs)
    (tmp_path / "result.json").unlink()
    assert check_log(tmp_path) == ["missing log.jsonl or result.json"]


# ---- robot_room ------------------------------------------------------------------------------------------------

def test_fake_robot_room():
    robot = FakeRobot(BOXES)
    assert robot.robot_room() == "hall"
    robot.navigate("kitchen east", None, [])
    assert robot.robot_room() == "kitchen"


def test_sim_robot_room():
    from sim.adapter import SimRobot
    from sim.world.house import set_base
    r = SimRobot(4000)
    assert r.robot_room() == "hall"
    for xy, room in (((1.5, 5.25), "bedroom"), ((8.5, 2.0), "kitchen"), ((3.0, 5.25), "hall"),  # bedroom door centre
                     ((4.5, 2.75), "kitchen"), ((3.75, 3.5), "hall"), ((-0.3, 1.0), "living room"),
                     ((9.4, 6.0), "storage")):
        set_base(r.m, r.d, (*xy, 0.0))
        assert r.robot_room() == room, (xy, room)
    assert not re.search(r"\d", r.robot_room())


# ---- OpenAI client (offline: create() replaced) ----------------------------------------------------------------

def test_openai_vlm_messages_retry_and_parse(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    import httpx2 as httpx
    import openai

    from agent.context import Packet
    from agent.vlm import OpenAIVLM, messages
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG fake")
    pk = Packet("sys", "user text", [str(png)], ["Photo 1"])
    m = messages(pk)
    assert m[0] == {"role": "system", "content": "sys"} and [c["type"] for c in m[1]["content"]] == ["text", "text", "image_url"]
    assert m[1]["content"][1]["text"] == "Photo 1" and m[1]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    v = OpenAIVLM("http://127.0.0.1:9/v1", "k", model="m")
    seen, fails = [], [openai.APIConnectionError(request=httpx.Request("POST", "http://x"))] * 2

    def create(**kw):
        seen.append(kw)
        if fails:
            raise fails.pop()
        msg = NS(content=None, reasoning="think", tool_calls=[NS(function=NS(name="look", arguments="{}"))])
        return NS(choices=[NS(message=msg)], usage=NS(model_dump=lambda: {"prompt_tokens": 5, "completion_tokens": 2}))

    v.client = NS(chat=NS(completions=NS(create=create)))
    import agent.vlm
    monkeypatch.setattr(agent.vlm.time, "sleep", lambda s: None)
    r = v(pk, 400003)
    assert len(seen) == 3 and len(r.retries) == 2 and r.reasoning == "think"
    assert r.tool_calls == [{"name": "look", "arguments": "{}"}] and r.usage["prompt_tokens"] == 5
    kw = seen[-1]
    assert kw["seed"] == 400003 and kw["extra_body"] == {"top_k": 20} and kw["temperature"] == 0.6
    assert kw["top_p"] == 0.95 and kw["max_tokens"] == 8192 and kw["tool_choice"] == "auto"
    fails[:] = [openai.APIConnectionError(request=httpx.Request("POST", "http://x"))] * 4
    with pytest.raises(openai.APIConnectionError):
        v(pk, 1)
