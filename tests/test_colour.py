"""G1-colour selection / parsing / scoring (no VLM, no render)."""
from sim.colour_photos import COLOURS, PER_COLOUR, parse_answer, score, select, status


def rec(i: str, seed: int, room: str, **px: int) -> dict:
    return {"id": i, "seed": seed, "room": room, "status": {c: status(px.get(c, 0)) for c in COLOURS}}


def test_status_thresholds() -> None:
    assert (status(0), status(1), status(99), status(100)) == ("absent", "ambiguous", "ambiguous", "present")


def test_select_first_three_distinct_rooms() -> None:
    recs = [rec("a0", 0, "hall", red=500), rec("a1", 0, "hall", red=900), rec("b0", 0, "kitchen", red=300, blue=100),
            rec("c0", 1, "hall", red=100), rec("d0", 1, "hall", red=800), rec("e0", 2, "kitchen", red=100, blue=400)]
    s = select(recs)
    assert s["red"] == ["a0", "b0", "c0"]       # a1 skipped (same seed+room as a0), d0 skipped, e0 not needed
    assert s["blue"] == ["b0", "e0"]            # shared photo b0 counts for both colours
    assert s["green"] == [] and PER_COLOUR == 3


def test_parse_answer() -> None:
    assert parse_answer('["Red", "blue"]') == ["red", "blue"]
    assert parse_answer('Sure: [\n "green" ] done') == ["green"]
    assert parse_answer("[]") == []
    assert parse_answer("no list") is None and parse_answer(None) is None and parse_answer("[red]") is None


def test_score_pass_and_fail() -> None:
    abs_ = {c: "absent" for c in COLOURS}
    truth = {"p1": {**abs_, "red": "present", "blue": "present"}, "p2": {**abs_, "red": "present", "green": "ambiguous"},
             "p3": {**abs_, "red": "present"}, "p4": {**abs_, "blue": "present"}, "p5": {**abs_, "blue": "present"}}
    sel = {"red": ["p1", "p2", "p3"], "blue": ["p1", "p4", "p5"], "green": [], "yellow": []}
    ans = {"p1": ["red", "blue"], "p2": ["red", "green"],      # green is ambiguous: not forbidden
           "p3": ["red", "yellow"],                             # yellow absent: fail
           "p4": ["blue"], "p5": None}                          # unparseable: fail
    s = score(truth, sel, ans)
    assert [r["pass"] for r in s["red"]["photos"]] == [True, True, False] and not s["red"]["gate"]
    assert [r["pass"] for r in s["blue"]["photos"]] == [True, True, False] and s["blue"]["n_pass"] == 2
    assert not s["all_pass"]
    ok = score(truth, {"red": ["p1", "p2"] + ["p1"], "blue": ["p1", "p4", "p1"], "green": ["p2"] * 3, "yellow": []},
               {"p1": ["red", "blue"], "p2": ["red"], "p4": ["blue"]})
    assert ok["red"]["gate"] and ok["blue"]["gate"] and not ok["green"]["gate"]   # green's true colour never named
