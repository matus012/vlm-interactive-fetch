import json

import pytest

from agent.tools import TOOLS
from agent.validate import validate_call


def test_six_skills_plus_done():
    assert [t["function"]["name"] for t in TOOLS] == ["navigate", "look", "locate", "pick", "place", "push", "done"]
    json.dumps(TOOLS)


@pytest.mark.parametrize("name,args", [
    ("navigate", '{"place": "kitchen table"}'),
    ("navigate", '{"place": "kitchen", "via": "living room"}'),
    ("look", "{}"),
    ("look", ""),
    ("locate", {"object": "red box"}),
    ("pick", '{"object": "red box"}'),
    ("place", '{"surface": "kitchen table"}'),
    ("push", '{"object": "crate"}'),
    ("done", '{"note": "finished"}'),
])
def test_valid(name, args):
    assert validate_call(name, args) == (True, "ok")


@pytest.mark.parametrize("name,args,frag", [
    ("fly", "{}", "unknown skill"),
    ("navigate", "{bad json", "not valid JSON"),
    ("navigate", "[1]", "JSON object"),
    ("navigate", "{}", "missing required"),
    ("navigate", '{"place": "kitchen", "speed": "fast"}', "unexpected argument"),
    ("navigate", '{"place": [1.0, 2.0]}', "must be a string"),
    ("navigate", '{"place": 3}', "must be a string"),
    ("navigate", '{"place": ""}', "empty"),
    ("look", '{"heading": "north"}', "unexpected argument"),
    ("pick", '{"object": null}', "must be a string"),
    ("place", '{"object": "red box"}', "unexpected argument"),
])
def test_invalid(name, args, frag):
    ok, why = validate_call(name, args)
    assert not ok and frag in why
