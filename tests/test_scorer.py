"""Scorer (agent/scorer.py) on hand-built snapshots + G2-mut (mutation test of the scorer).

Every check is a function of the scorer module `sc`, so the same checks run on the real module (one pytest case
each) and on every source-level mutant (test_g2_mut). A mutant is killed when at least one check fails on it. The
mutants are executed from the mutated text (compile + exec into a fresh module), so no file on disk is touched and
no stale bytecode can stand in for a mutant (mutation-test-before-freeze). A pattern that matches 0 or >= 2 times
is BAD_PATTERN and fails the test.

    PYTHONPATH=. uv run --frozen python -c "from tests.test_scorer import table; table()"   # the mutant table
"""
from __future__ import annotations

import copy
import math
import types
from pathlib import Path
from typing import Callable

import pytest

import agent.scorer as real

SRC = Path(real.__file__).read_text()
EPS = 1e-4  # metres / m/s / s
EPS_DEG = 1e-3
TOP = 0.5
# test surfaces: exact float edges (0 + 0.02 == 0.02, 1 - 0.02 == 0.98)
KT, SB = "kitchen table", "storage bench"
SURFACES = {KT: {"rect": [0.0, 0.0, 1.0, 1.0], "top_z": TOP}, SB: {"rect": [3.0, 0.0, 4.0, 1.0], "top_z": TOP}}
ZUP = TOP + 0.06  # centre z of an upright box resting on a top


def box(x: float, y: float, z: float = ZUP, tilt: float = 1.0, disp: float = 1e-4, speed: float = 1e-3,
        support: str | None = KT) -> dict:
    return {"pos": [x, y, z], "tilt_deg": tilt, "max_disp_settle_m": disp, "max_speed_settle": speed,
            "support": support}


def snap(**over: dict) -> dict:
    boxes = {"red box": box(0.5, 0.5), "blue box": box(2.0, 2.0, 0.06, support="floor"),
             "green box": box(5.0, 5.0, 0.06, support="floor"), "yellow box": box(6.0, 5.0, 0.06, support="floor")}
    boxes.update(over)
    return {"boxes": boxes, "surfaces": copy.deepcopy(SURFACES), "held": None, "gripper_open": True,
            "arm_retracted": True, "sim_ok": True, "settle_s": 2.0}


FETCH = {"task_id": "fetch-1000", "type": "fetch", "seed": 1000, "blocker": "none",
         "instruction": "Bring the red box to the kitchen table.",
         "targets": [{"object": "red box", "surface": KT}], "step_limit": 20}
TWO = {"task_id": "two_box-1000", "type": "two_box", "seed": 1000, "blocker": "none",
       "instruction": "Red box to the kitchen table, blue box to the storage bench.",
       "targets": [{"object": "red box", "surface": KT}, {"object": "blue box", "surface": SB}], "step_limit": 30}


def two_snap(**over: dict) -> dict:
    s = snap(**{"blue box": box(3.5, 0.5, support=SB)})
    s["boxes"].update(over)
    return s


def failed(sc: types.ModuleType, s: dict, task: dict = FETCH, steps: int = 5) -> list[str]:
    r = sc.score(s, task, steps)
    assert r["pass"] == (not r["failed"])
    assert r["failed"] == [k for k in sc.RULES if not r["rules"][k]]
    return r["failed"]


# ---- checks ---------------------------------------------------------------------------------------------------

def c_pass(sc: types.ModuleType) -> None:
    r = sc.score(snap(), FETCH, 5)
    assert r["pass"] is True and r["failed"] == [] and all(r["rules"].values())
    assert r["detail"]["targets"]["red box"]["edge_m"] == pytest.approx(0.5)
    assert r["detail"]["targets"]["red box"]["bottom_gap_m"] == pytest.approx(0.0, abs=1e-12)
    assert failed(sc, two_snap(), TWO, 30) == []


def c_rules_order(sc: types.ModuleType) -> None:
    assert tuple(sc.RULES) == ("surface", "wrong_box", "held", "steps", "sim")
    s = snap(**{"red box": box(2.0, 2.0, 0.06, support="floor"), "blue box": box(0.5, 0.5)})
    s["held"], s["sim_ok"] = "green box", False
    assert failed(sc, s, FETCH, 21) == ["surface", "wrong_box", "held", "steps", "sim"]
    assert sc.score(s, FETCH, 21)["pass"] is False


def c_margin_edges(sc: types.ModuleType) -> None:
    m = 0.02
    for x, y in ((m, 0.5), (1 - m, 0.5), (0.5, m), (0.5, 1 - m)):  # exactly on the shrunk edge: inside
        assert failed(sc, snap(**{"red box": box(x, y)})) == [], (x, y)
    for x, y in ((m + EPS, 0.5), (1 - m - EPS, 0.5), (0.5, m + EPS), (0.5, 1 - m - EPS)):
        assert failed(sc, snap(**{"red box": box(x, y)})) == [], (x, y)
    for x, y in ((m - EPS, 0.5), (1 - m + EPS, 0.5), (0.5, m - EPS), (0.5, 1 - m + EPS)):
        assert failed(sc, snap(**{"red box": box(x, y)})) == ["surface"], (x, y)


def c_off_surface(sc: types.ModuleType) -> None:
    assert failed(sc, snap(**{"red box": box(2.0, 2.0, 0.06, support="floor")})) == ["surface"]
    assert failed(sc, snap(**{"red box": box(3.5, 0.5, support=SB)})) == ["surface"]  # other surface


def c_on_top(sc: types.ModuleType) -> None:
    tol = 0.015
    for dz in (tol - EPS, -tol + EPS):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, ZUP + dz)})) == [], dz
    for dz in (tol + EPS, -tol - EPS, 0.2, -0.3):
        f = failed(sc, snap(**{"red box": box(0.5, 0.5, ZUP + dz, support=None)}))
        assert f == ["surface"], dz


def c_upright(sc: types.ModuleType) -> None:
    for t in (15.0, 15.0 - EPS_DEG, 0.0):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, tilt=t)})) == [], t
    for t in (15.0 + EPS_DEG, 30.0, 90.0):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, tilt=t)})) == ["surface"], t


def c_still(sc: types.ModuleType) -> None:
    for d in (0.005, 0.005 - EPS):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, disp=d)})) == [], d
    for d in (0.005 + EPS, 0.0075, 0.009, 0.5):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, disp=d)})) == ["surface"], d
    for v in (0.02, 0.02 - EPS):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, speed=v)})) == [], v
    for v in (0.02 + EPS, 0.03, 0.039, 1.0):
        assert failed(sc, snap(**{"red box": box(0.5, 0.5, speed=v)})) == ["surface"], v


def c_settle_window(sc: types.ModuleType) -> None:
    for st, want in ((2.0, []), (3.0, []), (4.5, []), (2.0 - EPS, ["surface"]), (1.5, ["surface"]),
                     (0.0, ["surface"])):
        s = snap()
        s["settle_s"] = st
        assert failed(sc, s) == want, st


def c_wrong_box(sc: types.ModuleType) -> None:
    on_kt = box(0.2, 0.8)
    assert failed(sc, snap(**{"blue box": on_kt})) == ["wrong_box"]
    # support clause alone (geometry says floor height elsewhere)
    assert failed(sc, snap(**{"blue box": box(2.0, 2.0, 0.06, support=KT)})) == ["wrong_box"]
    # geometry clause alone: lying on its side on the table, support not detected
    assert failed(sc, snap(**{"blue box": box(0.2, 0.8, TOP + 0.03, tilt=90.0, support=None)})) == ["wrong_box"]
    # rect edges are inclusive; just outside is not on it
    for x, y in ((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0)):
        assert failed(sc, snap(**{"blue box": box(x, y, support=None)})) == ["wrong_box"], (x, y)
    for x, y in ((-EPS, 0.5), (1 + EPS, 0.5), (0.5, -EPS), (0.5, 1 + EPS)):
        assert failed(sc, snap(**{"blue box": box(x, y, support=None)})) == [], (x, y)
    # under the table top (centre at or below top_z) is not on it
    for z in (TOP, TOP - EPS, 0.06):
        assert failed(sc, snap(**{"blue box": box(0.5, 0.2, z, support="floor")})) == [], z
    assert failed(sc, snap(**{"blue box": box(0.5, 0.2, TOP + EPS, support=None)})) == ["wrong_box"]
    # a box on a non-target surface is fine
    assert failed(sc, snap(**{"blue box": box(3.5, 0.5, support=SB)})) == []
    d = sc.score(snap(**{"blue box": on_kt}), FETCH, 5)["detail"]
    assert d["wrong_boxes"] == [f"blue box@{KT}"]


def c_two_box(sc: types.ModuleType) -> None:
    assert failed(sc, two_snap(), TWO, 30) == []
    # each target must hold: first target off, then second target off
    assert failed(sc, two_snap(**{"red box": box(2.0, 2.0, 0.06, support="floor")}), TWO, 30) == ["surface"]
    assert failed(sc, two_snap(**{"blue box": box(2.0, 2.0, 0.06, support="floor")}), TWO, 30) == ["surface"]
    # wrong box on the second target surface
    assert failed(sc, two_snap(**{"green box": box(3.2, 0.2, support=SB)}), TWO, 30) == ["wrong_box"]
    assert failed(sc, two_snap(**{"green box": box(0.2, 0.2)}), TWO, 30) == ["wrong_box"]
    # the other target's box on the wrong surface: swapped boxes fail both rules
    sw = two_snap(**{"red box": box(3.5, 0.5, support=SB), "blue box": box(0.5, 0.5)})
    assert failed(sc, sw, TWO, 30) == ["surface", "wrong_box"]
    # both boxes on the kitchen table: blue is wrong there
    both = two_snap(**{"blue box": box(0.2, 0.8)})
    assert failed(sc, both, TWO, 30) == ["surface", "wrong_box"]
    assert sc.score(both, TWO, 30)["detail"]["wrong_boxes"] == [f"blue box@{KT}"]


def c_held(sc: types.ModuleType) -> None:
    for h in ("red box", "green box"):
        s = snap()
        s["held"] = h
        assert failed(sc, s) == ["held"], h


def c_steps(sc: types.ModuleType) -> None:
    assert failed(sc, snap(), FETCH, 20) == []
    assert failed(sc, snap(), FETCH, 0) == []
    assert failed(sc, snap(), FETCH, 21) == ["steps"]
    assert failed(sc, two_snap(), TWO, 30) == []
    assert failed(sc, two_snap(), TWO, 31) == ["steps"]


def c_sim(sc: types.ModuleType) -> None:
    s = snap()
    s["sim_ok"] = False
    assert failed(sc, s) == ["sim"]
    for field in ("x", "y", "z", "tilt_deg", "max_disp_settle_m", "max_speed_settle"):
        for bad in (math.nan, math.inf):
            b = box(5.0, 5.0, 0.06, support="floor")
            if field in "xyz":
                b["pos"]["xyz".index(field)] = bad
            else:
                b[field] = bad
            assert failed(sc, snap(**{"green box": b})) == ["sim"], (field, bad)


def c_missing_target_raises(sc: types.ModuleType) -> None:
    s = snap()
    del s["boxes"]["red box"]
    with pytest.raises(KeyError):
        sc.score(s, FETCH, 5)


CHECKS: list[Callable[[types.ModuleType], None]] = [
    c_pass, c_rules_order, c_margin_edges, c_off_surface, c_on_top, c_upright, c_still, c_settle_window,
    c_wrong_box, c_two_box, c_held, c_steps, c_sim, c_missing_target_raises]


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_scorer(check: Callable[[types.ModuleType], None]) -> None:
    check(real)


# ---- G2-mut ---------------------------------------------------------------------------------------------------

# (name, old, new): old must occur exactly once in agent/scorer.py
MUTANTS: list[tuple[str, str, str]] = [
    # drop each rule
    ("drop rule surface", '"surface": surface_ok,', '"surface": True,'),
    ("drop rule wrong_box", '"wrong_box": not wrong,', '"wrong_box": True,'),
    ("drop rule held", '"held": snapshot["held"] is None,', '"held": True,'),
    ("drop rule steps", '"steps": steps_used <= task["step_limit"],', '"steps": True,'),
    ("drop rule sim", '"sim": bool(snapshot["sim_ok"]) and finite}', '"sim": True}'),
    ("sim: ignore sim_ok", '"sim": bool(snapshot["sim_ok"]) and finite}', '"sim": finite}'),
    ("sim: ignore NaN", '"sim": bool(snapshot["sim_ok"]) and finite}', '"sim": bool(snapshot["sim_ok"])}'),
    ("sim: NaN pos not checked", 'for v in (*b["pos"], b["tilt_deg"],', 'for v in (b["tilt_deg"],'),
    ("sim: NaN tilt not checked", '(*b["pos"], b["tilt_deg"], b["max_disp', '(*b["pos"], b["max_disp'),
    ("sim: NaN disp not checked", 'b["max_disp_settle_m"], b["max_speed_settle"]))',
     'b["max_speed_settle"]))'),
    ("sim: NaN speed not checked", 'b["max_disp_settle_m"], b["max_speed_settle"]))', 'b["max_disp_settle_m"]))'),
    ("held: inverted", 'snapshot["held"] is None,', 'snapshot["held"] is not None,'),
    ("steps: < instead of <=", 'steps_used <= task["step_limit"]', 'steps_used < task["step_limit"]'),
    ("steps: off by one", 'steps_used <= task["step_limit"]', 'steps_used <= task["step_limit"] + 1'),
    ("pass ignores rules but surface", '"pass": not failed,', '"pass": surface_ok,'),
    ("failed order sorted", 'failed = [r for r in RULES if not rules[r]]',
     'failed = sorted(r for r in RULES if not rules[r])'),
    ("any target instead of all", 'surface_ok = surface_ok and ok', 'surface_ok = surface_ok or ok'),
    ("only last target counts", 'surface_ok = surface_ok and ok', 'surface_ok = ok'),
    # margin
    ("ignore margin", "m = SCORE_SURFACE_MARGIN", "m = 0.0"),
    ("margin x2", "m = SCORE_SURFACE_MARGIN", "m = SCORE_SURFACE_MARGIN * 2"),
    ("margin x0.5", "m = SCORE_SURFACE_MARGIN", "m = SCORE_SURFACE_MARGIN * 0.5"),
    ("margin sign x0", "x0 + m <= x", "x0 - m <= x"),
    ("margin sign x1", "x <= x1 - m", "x <= x1 + m"),
    ("margin sign y0", "y0 + m <= y", "y0 - m <= y"),
    ("margin sign y1", "y <= y1 - m", "y <= y1 + m"),
    ("x0 strict", "x0 + m <= x", "x0 + m < x"),
    ("x1 strict", "x <= x1 - m", "x < x1 - m"),
    ("y0 strict", "y0 + m <= y", "y0 + m < y"),
    ("y1 strict", "y <= y1 - m", "y < y1 - m"),
    ("ignore x", "inside = x0 + m <= x <= x1 - m and ", "inside = "),
    ("ignore y", " and y0 + m <= y <= y1 - m", ""),
    ("x uses y bounds", "x0 + m <= x <= x1 - m", "y0 + m <= x <= y1 - m"),
    # on top
    ("ignore on_top", "on_top = abs(gap) <= SCORE_ON_TOP_TOL", "on_top = True"),
    ("on_top no abs", "abs(gap) <= SCORE_ON_TOP_TOL", "gap <= SCORE_ON_TOP_TOL"),
    ("on_top tol x2", "abs(gap) <= SCORE_ON_TOP_TOL", "abs(gap) <= SCORE_ON_TOP_TOL * 2"),
    ("on_top tol x0.5", "abs(gap) <= SCORE_ON_TOP_TOL", "abs(gap) <= SCORE_ON_TOP_TOL * 0.5"),
    ("bottom = centre", "gap = z - BOX_HALF_Z - surf", "gap = z - surf"),
    ("box half z x2", "BOX_HALF_Z = BOX_HALF[2]", "BOX_HALF_Z = BOX_HALF[2] * 2"),
    ("box half z wrong axis", "BOX_HALF_Z = BOX_HALF[2]", "BOX_HALF_Z = BOX_HALF[1]"),
    # tilt
    ("ignore tilt", 'upright = box["tilt_deg"] <= SCORE_UPRIGHT_DEG', "upright = True"),
    ("tilt strict", 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG', 'box["tilt_deg"] < SCORE_UPRIGHT_DEG'),
    ("tilt x2", 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG', 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG * 2'),
    ("tilt x0.5", 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG', 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG * 0.5'),
    ("tilt flipped", 'box["tilt_deg"] <= SCORE_UPRIGHT_DEG', 'box["tilt_deg"] >= SCORE_UPRIGHT_DEG'),
    # stillness
    ("ignore stillness", 'still = (box["max_disp_settle_m"] <= SCORE_STILL_DISP and box["max_speed_settle"] <= '
     'SCORE_STILL_SPEED\n             and settle_s >= SCORE_STILL_S)', "still = True"),
    ("ignore disp", 'box["max_disp_settle_m"] <= SCORE_STILL_DISP and ', ""),
    ("ignore speed", ' and box["max_speed_settle"] <= SCORE_STILL_SPEED', ""),
    ("ignore settle window", "\n             and settle_s >= SCORE_STILL_S)", ")"),
    ("disp strict", '"max_disp_settle_m"] <= SCORE_STILL_DISP', '"max_disp_settle_m"] < SCORE_STILL_DISP'),
    ("disp x2", '"max_disp_settle_m"] <= SCORE_STILL_DISP', '"max_disp_settle_m"] <= SCORE_STILL_DISP * 2'),
    ("disp x0.5", '"max_disp_settle_m"] <= SCORE_STILL_DISP', '"max_disp_settle_m"] <= SCORE_STILL_DISP * 0.5'),
    ("speed strict", '"max_speed_settle"] <= SCORE_STILL_SPEED', '"max_speed_settle"] < SCORE_STILL_SPEED'),
    ("speed x2", '"max_speed_settle"] <= SCORE_STILL_SPEED', '"max_speed_settle"] <= SCORE_STILL_SPEED * 2'),
    ("speed x0.5", '"max_speed_settle"] <= SCORE_STILL_SPEED', '"max_speed_settle"] <= SCORE_STILL_SPEED * 0.5'),
    ("settle strict", "settle_s >= SCORE_STILL_S)", "settle_s > SCORE_STILL_S)"),
    ("settle x2", "settle_s >= SCORE_STILL_S)", "settle_s >= SCORE_STILL_S * 2)"),
    ("settle x0.5", "settle_s >= SCORE_STILL_S)", "settle_s >= SCORE_STILL_S * 0.5)"),
    ("disp vs speed swapped", 'box["max_disp_settle_m"] <= SCORE_STILL_DISP and box["max_speed_settle"] <= '
     'SCORE_STILL_SPEED', 'box["max_disp_settle_m"] <= SCORE_STILL_SPEED and box["max_speed_settle"] <= '
     'SCORE_STILL_DISP'),
    # wrong box
    ("wrong box: first target surface only", 'for s in dict.fromkeys(t["surface"] for t in task["targets"]):',
     'for s in [task["targets"][0]["surface"]]:'),
    ("wrong box: any task box allowed anywhere",
     'meant = {t["object"] for t in task["targets"] if t["surface"] == s}', 'meant = {t["object"] for t in task["targets"]}'),
    ("wrong box: ignore support", 'return box.get("support") == name or (', 'return False or ('),
    ("wrong box: ignore geometry",
     'or (x0 <= x <= x1 and y0 <= y <= y1 and z > surf["top_z"])', "or False"),
    ("wrong box: ignore height", ' and z > surf["top_z"])', ")"),
    ("wrong box: height >=", 'z > surf["top_z"])', 'z >= surf["top_z"])'),
    ("wrong box: x0 strict", "x0 <= x <= x1 and", "x0 < x <= x1 and"),
    ("wrong box: x1 strict", "x0 <= x <= x1 and", "x0 <= x < x1 and"),
    ("wrong box: y0 strict", "y0 <= y <= y1 and z", "y0 < y <= y1 and z"),
    ("wrong box: y1 strict", "y0 <= y <= y1 and z", "y0 <= y < y1 and z"),
    ("wrong box: shrunk rect", "x0 <= x <= x1 and y0 <= y <= y1 and z",
     "x0 + SCORE_SURFACE_MARGIN <= x <= x1 - SCORE_SURFACE_MARGIN and y0 <= y <= y1 and z"),
]


def run_mutants(src: str = SRC) -> list[tuple[str, str, str]]:
    """(name, verdict, killing check or error): verdict KILLED | SURVIVED | BAD_PATTERN | NO_COMPILE."""
    out = []
    for name, old, new in MUTANTS:
        n = src.count(old)
        if n != 1 or old == new:
            out.append((name, "BAD_PATTERN", f"matches {n}"))
            continue
        text = src.replace(old, new)
        mod = types.ModuleType("scorer_mutant")
        try:
            exec(compile(text, "scorer_mutant.py", "exec"), mod.__dict__)
        except SyntaxError as e:
            out.append((name, "NO_COMPILE", str(e)))
            continue
        killer = ""
        for c in CHECKS:
            try:
                c(mod)
            except Exception as e:  # an AssertionError, or the mutant crashing on valid input
                killer = f"{c.__name__} ({type(e).__name__})"
                break
        out.append((name, "KILLED" if killer else "SURVIVED", killer))
    return out


def test_g2_mut() -> None:
    control = types.ModuleType("scorer_control")  # the harness itself: unmutated text exec'd the same way passes
    exec(compile(SRC, "scorer_control.py", "exec"), control.__dict__)
    for c in CHECKS:
        c(control)
    res = run_mutants()
    bad = [r for r in res if r[1] != "KILLED"]
    assert not bad, bad
    assert len(res) == len(MUTANTS) >= 60


def table() -> None:
    for name, verdict, why in run_mutants():
        print(f"| {name} | {verdict} | {why} |")


# ---- sim/snapshot.py on a real house ---------------------------------------------------------------------------

def test_snapshot_box_on_kitchen_table() -> None:
    import mujoco

    from sim.snapshot import snapshot
    from sim.world.generate import LAYOUT, generate
    from sim.world.house import load_house

    w = generate(LAYOUT, 1017, "none")  # green on the nightstand, blue on the coffee table (fix round 2 spawn rule)
    m, d = load_house(w)

    def put(body: str, pos: tuple[float, float, float], quat: tuple[float, float, float, float]) -> None:
        a = m.jnt_qposadr[m.body(body).jntadr[0]]
        d.qpos[a: a + 7] = (*pos, *quat)

    put("box_red", (7.55, 1.0, 0.75 + 0.06 + 0.0005), (math.cos(0.15), 0, 0, math.sin(0.15)))
    put("box_yellow", (4.0, 6.5, 0.10 + 0.0005), (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))  # on its side
    mujoco.mj_forward(m, d)
    first = snapshot(m, d, w, lambda m_, d_: None, 1.0)  # the drop (0.5 mm) happens here
    assert first["boxes"]["red box"]["max_speed_settle"] > 0.0
    s = snapshot(m, d, w, lambda m_, d_: None, 2.0)
    assert s["settle_s"] == 2.0 and s["n_steps"] == 1000 and s["sim_ok"]
    assert s["held"] is None and s["gripper_open"] and s["arm_retracted"]
    assert set(s["surfaces"]) == {"kitchen table", "storage bench"}
    assert s["surfaces"]["kitchen table"] == {"rect": pytest.approx([7.0, 0.65, 8.1, 1.35]), "top_z": 0.75}
    red, yellow = s["boxes"]["red box"], s["boxes"]["yellow box"]
    assert red["support"] == "kitchen table" and red["tilt_deg"] < 1.0
    assert red["pos"][2] - 0.06 == pytest.approx(0.75, abs=0.003)
    assert red["max_disp_settle_m"] < 1e-3 and red["max_speed_settle"] < 0.01
    assert yellow["support"] == "floor" and yellow["tilt_deg"] == pytest.approx(90.0, abs=1.0)
    assert s["boxes"]["green box"]["support"] == "nightstand"
    assert s["boxes"]["blue box"]["support"] == "coffee table"
    task = dict(FETCH, targets=[{"object": "red box", "surface": "kitchen table"}])
    r = real.score(s, task, 4)
    assert r["pass"], r
    task_y = dict(FETCH, targets=[{"object": "yellow box", "surface": "kitchen table"}])
    assert real.score(s, task_y, 4)["failed"] == ["surface", "wrong_box"]
    held = snapshot(m, d, w, lambda m_, d_: "blue box", 0.1)
    assert held["held"] == "blue box" and real.score(held, task, 4)["failed"] == ["surface", "held"]  # 0.1 s window
