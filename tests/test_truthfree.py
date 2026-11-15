"""G5-truthfree (task 006 item 3): in perceived mode, `locate`, the skills, `pick`, `place`, `push` and navigate's
avoidance read no truth pose except the residuals of context row 19, (a)-(g), each on ALLOWED (the one list).

Static part: an AST call-graph scan across the modules below, from the perceived-mode roots (ROOTS). Edges: `self.m()`
inside a class -> that class's method; `self.robot.m()` in Skills -> SimRobot.m (the adapter); a module function,
an imported one (`from sim.x import f`, also inside a function) or `mod.f` (`import sim.x as mod`); any reference
counts, not only calls (a function passed as a value is followed); a nested function is its own unit (`outer.inner`),
reached with its outer one. A truth read is
- a body pose attribute (.xpos .xquat .xmat .xipos .ximat .geom_xpos .geom_xmat) on anything but `<x>.site(...)` or
  `<x>.body("<robot body>")` (the robot's own kinematics), e.g. `d.body(box).xpos`, `b.xmat`, `d.xpos[1:]`;
- segmentation: an identifier / attribute / string / argument named seg / *_seg / seg_* / *segmentation*;
- anything named or keyed *truth* (true_pose's callers are caught by descending into true_pose itself).
The truth-mode branch is not scanned: the body of `if self.poses == "truth":` (and the IfExp form); its `else` is.
A `self.poses` comparison itself (the mode name) is not a read.
ALLOWED is the one list of perceived-mode truth reads in the tests: (row-19 letters, reason, reads). reads None = the
whole function is a residual (not descended); else only those reads (`ast.unparse` of the pose attribute, or the
identifier) are allowed in that function, which is still scanned and descended. Every entry must be reached.
NOT_TRUTH: identifiers that match the seg pattern but are path segments (no truth, no residual).

Dynamic part: a perceived-mode run (look -> locate -> pick -> navigate with avoid) with the truth pose functions
raising and the segmentation blanked; it must give the same perceived results.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("agent/skills.py", "sim/adapter.py", "sim/grasp.py", "sim/perception.py", "sim/nav/follow.py", "sim/body.py",
         "sim/look.py", "sim/nav/astar.py", "sim/world/occupancy.py", "sim/snapshot.py", "sim/robot.py", "sim/push.py")
ROOTS = [("agent/skills.py", f"Skills.{m}") for m in ("locate", "pick", "navigate", "look", "place", "push")] + \
        [("sim/adapter.py", f"SimRobot.{m}") for m in ("locate", "pick", "navigate", "look", "photo", "place", "push",
                                                      "_avoid_discs", "_map_pushed", "_map_grid", "_map_entry",
                                                      "_note_plan", "_path_plan")] + \
        [("sim/push.py", "push_object.mapped")]
# place (Skills.place -> SimRobot.place -> grasp.place_box) and push (Skills.push -> SimRobot.push -> push.push_object,
# its mapped() = the map-entry consumption of row 17 add. 4, and the map-blocked plan it gets) are scanned since task 006
# fix 2 (review phase 5: U1-U3); their truth reads are the row-19 residuals on ALLOWED.
# _map_pushed / _map_grid: the pushed-object map (row 17 addendum 3, task 006 5c): the perceived footprint comes from
# push's own contact geometry. _map_entry / _note_plan / _path_plan (row 17 addendum 4, task 006 5d): the map footprint
# push's tests use (est) and the map-blocked memory's plan requests (navigate / pick / place).
POSE_ATTRS = {"xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat"}
ROBOT_BODIES = {"base_link", "left_pad", "right_pad", "gen3/base_link"}
SEG = re.compile(r"(^|_)seg($|_)|segmentation", re.I)
ROW19 = set("abcdefg")  # row 19: the ONE list of perceived-mode sim-truth residuals
ALLOWED: dict[tuple[str, str], tuple[str, str, frozenset[str] | None]] = {
    # (c) eval-only log fields
    ("sim/adapter.py", "SimRobot._locate_eval"): ("c", "true pose + seg px into the locate step meta.eval", None),
    ("sim/adapter.py", "SimRobot._push_eval"): ("c", "true end pose + cover_m into the push detail map_eval", None),
    # (d)
    ("sim/adapter.py", "SimRobot.sim_ok"): ("d", "simulator health (NaN / blow-up), not a perception input", None),
    # (g) crate / cabinet: in-view test = segmentation px, locate = true pose; seg px also feed (c) meta.eval
    ("sim/adapter.py", "SimRobot._locate_unperceived"): ("g", "crate / cabinet locate returns the true pose", None),
    ("sim/adapter.py", "SimRobot._seg_px"): ("cg", "segmentation px: crate / cabinet visibility; eval log", None),
    ("sim/look.py", "render_look"): ("cg", "the one render pass: segmentation -> _seg_px only", None),
    ("sim/adapter.py", "SimRobot._shoot"): ("cg", "segmentation of the render -> _seg_px (crate / cabinet visible)",
                                            frozenset({"seg", "seg_px", "_seg_px", "_cur_seg"})),
    ("sim/adapter.py", "SimRobot._new_set"): ("cg", "the photo set's segmentation store", frozenset({"_cur_seg"})),
    # (b) pick's post-contact checks stand in for gripper sensing
    ("sim/grasp.py", "_support_z"): ("b", "lift height: support height for lift_above_support_m", None),
    ("sim/grasp.py", "_tilted"): ("b", "tilt after a drop", None),
    ("sim/grasp.py", "holding"): ("b", "holding: box between the pads", None),
    ("sim/grasp.py", "in_hand"): ("be", "in-hand pose (pick's check; place's landing input, (e))", None),
    ("sim/grasp.py", "robot_xy_radius"): ("b", "carry footprint: robot + the held box (in-hand)", None),
    ("sim/look.py", "geom_points"): ("b", "robot geoms + the held box (in-hand) for push's carry footprint", None),
    ("sim/grasp.py", "_end"): ("be", "pick / place end state (pads, box pose, held) into detail", None),
    # (e) U1 place: landing target, retreat plan, placed-vs-missed read the held box's true pose
    ("sim/grasp.py", "place_box"): ("e", "held box true pose: tilt in hand, retreat from its xpos, the verdict",
                                    frozenset({"d.body(held_body).xpos", "d.body(held_body).xmat"})),
    # (a) push reads the crate / cabinet pose for its plan (and its physics / verdict on the real object)
    ("sim/adapter.py", "SimRobot.push"): ("a", "the pushed object's true xy -> its door", frozenset({"b.xpos"})),
    ("sim/push.py", "push_object"): ("a", "the pushed object's true pose: frame, start, moved, end",
                                     frozenset({"d.body(body).xpos"})),
    ("sim/push.py", "push_object.door_push"): ("a", "the pushed object's lat after the through-push",
                                               frozenset({"d.body(body).xpos"})),
    ("sim/push.py", "push_line"): ("a", "the pushed object's progress / stall", frozenset({"d.body(body).xpos"})),
    ("sim/push.py", "_footprint"): ("a", "crate / cabinet geom footprint (mapped()'s fallback when not on the map)",
                                    None),
    ("sim/push.py", "_aabb"): ("a", "crate / cabinet AABB for the clearing push", None),
    ("sim/push.py", "_tilt_deg"): ("a", "the pushed object's tilt (detail)", None),
    ("sim/push.py", "_static_rects"): ("a", "joint-less box geoms: a welded cabinet; walls / furniture (known map)",
                                       frozenset({"d.geom_xpos", "d.geom_xmat"})),
    # (f) U2 push: _box_xy reads every fetch box's true xy for plan feasibility and the step-off
    ("sim/push.py", "_box_xy"): ("f", "every fetch box's true xy (BOX_CLEAR feasibility, step-off)", None),
}
NOT_TRUTH = {
    ("sim/nav/follow.py", "go_to"): {"seg"},  # a path segment vector (back-off), not segmentation
    ("sim/push.py", "path_dist"): {"seg", "seg_pts"},  # a path segment, not segmentation
}


class Module:
    def __init__(self, rel: str, src: str) -> None:
        self.rel, self.tree = rel, ast.parse(src)
        self.defs: dict[str, ast.AST] = {}  # qualname -> FunctionDef (nested: outer.inner)
        self.nested: dict[str, list[str]] = {}  # qualname -> its directly nested functions' qualnames
        todo = [(n.name, n) for n in self.tree.body if isinstance(n, ast.FunctionDef)]
        todo += [(f"{n.name}.{f.name}", f) for n in self.tree.body if isinstance(n, ast.ClassDef)
                 for f in n.body if isinstance(f, ast.FunctionDef)]
        while todo:
            q, f = todo.pop()
            self.defs[q] = f
            inner = [(f"{q}.{n.name}", n) for n in _nodes(f) if isinstance(n, ast.FunctionDef)]
            self.nested[q] = [iq for iq, _ in inner]
            todo += inner
        self.names: dict[str, tuple[str, str]] = {}  # local name -> (module rel, function name)
        self.mods: dict[str, str] = {}  # alias -> module rel
        for n in ast.walk(self.tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                for a in n.names:
                    self.names[a.asname or a.name] = (n.module.replace(".", "/") + ".py", a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    self.mods[a.asname or a.name] = a.name.replace(".", "/") + ".py"


def _truth_branch(test: ast.AST) -> bool:
    """`self.poses == "truth"`."""
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Attribute) and test.left.attr == "poses"
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "truth")


def _nodes(fn: ast.AST):
    """Every node of the function except the truth-mode branches (their test and body), `self.poses` comparisons,
    docstrings and the bodies of nested functions (yielded, not entered: their own unit)."""
    todo = list(ast.iter_child_nodes(fn))
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.If, ast.IfExp)) and _truth_branch(n.test):
            todo += n.orelse if isinstance(n, ast.If) else [n.orelse]
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):  # docstrings / bare strings
            continue
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Attribute) and n.left.attr == "poses":
            continue
        yield n
        if isinstance(n, ast.FunctionDef):
            continue
        todo += list(ast.iter_child_nodes(n))


def _robot_kinematics(v: ast.AST) -> bool:
    if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)):
        return False
    if v.func.attr == "site":
        return True
    return (v.func.attr == "body" and len(v.args) == 1 and isinstance(v.args[0], ast.Constant)
            and v.args[0].value in ROBOT_BODIES)


def scan(sources: dict[str, str] | None = None) -> tuple[set[tuple[str, str]], list[str], set[tuple[str, str]]]:
    """(functions scanned, truth reads found, allowed functions reached)."""
    mods = {rel: Module(rel, (sources or {}).get(rel) or (ROOT / rel).read_text()) for rel in FILES}
    seen: set[tuple[str, str]] = set()
    hits: list[str] = []
    allowed: set[tuple[str, str]] = set()
    todo = list(ROOTS)
    while todo:
        key = todo.pop()
        if key in seen or key[0] not in mods or key[1] not in mods[key[0]].defs:
            continue
        reads: frozenset[str] = frozenset()
        if key in ALLOWED:
            allowed.add(key)
            if ALLOWED[key][2] is None:
                continue
            reads = ALLOWED[key][2]
        seen.add(key)
        mod, qual = mods[key[0]], key[1]
        cls = qual.split(".")[0] if "." in qual else None
        ok_tokens = reads | NOT_TRUTH.get(key, set())
        todo += [(key[0], q) for q in mod.nested[qual]]
        for n in _nodes(mod.defs[qual]):
            word = None
            if isinstance(n, ast.Attribute):
                word = n.attr
                v = n.value
                if n.attr in POSE_ATTRS and not _robot_kinematics(v) and ast.unparse(n) not in reads:
                    hits.append(f"{key[0]}:{qual}:{n.lineno}: pose attribute {ast.unparse(n)}")
                if isinstance(v, ast.Name) and v.id == "self" and cls:
                    todo.append((key[0], f"{cls}.{n.attr}"))
                elif (isinstance(v, ast.Attribute) and v.attr == "robot" and isinstance(v.value, ast.Name)
                      and v.value.id == "self"):
                    todo.append(("sim/adapter.py", f"SimRobot.{n.attr}"))
                elif isinstance(v, ast.Name) and v.id in mod.mods:
                    todo.append((mod.mods[v.id], n.attr))
                elif isinstance(v, ast.Name) and v.id in mod.names:  # `import sim.nav.follow as follow` via from-import
                    todo.append((mod.names[v.id][0][:-3] + "/" + mod.names[v.id][1] + ".py", n.attr))
            elif isinstance(n, ast.Name):
                word = n.id
                if n.id in mod.defs:
                    todo.append((key[0], n.id))
                elif n.id in mod.names:
                    todo.append(mod.names[n.id])
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                word = n.value
            elif isinstance(n, ast.arg):
                word = n.arg
            if word is not None and word not in ok_tokens and (SEG.search(word) or "truth" in word.lower()):
                hits.append(f"{key[0]}:{qual}:{getattr(n, 'lineno', '?')}: '{word}'")
    return seen, hits, allowed


def test_perceived_paths_read_no_truth() -> None:
    seen, hits, allowed = scan()
    # not vacuous: the perceived pipeline, the pick planner and grasp, navigate's grid are all in the call graph
    need = {("agent/skills.py", "Skills.locate"), ("sim/adapter.py", "SimRobot.locate"),
            ("sim/adapter.py", "SimRobot._perceive"), ("sim/adapter.py", "SimRobot._shoot"),
            ("sim/adapter.py", "SimRobot._stamp_object"), ("sim/adapter.py", "SimRobot._nav_grid"),
            ("sim/perception.py", "locate_photos"), ("sim/perception.py", "fit_box"), ("sim/perception.py", "colour_mask"),
            ("sim/grasp.py", "pick_box"), ("sim/grasp.py", "plan_pregrasp_at"), ("sim/grasp.py", "_grasp_once"),
            ("sim/nav/follow.py", "go_to"), ("sim/snapshot.py", "support_of"),
            ("sim/adapter.py", "SimRobot._map_pushed"), ("sim/adapter.py", "SimRobot._stamp_rot"),
            ("sim/push.py", "contact_footprint"), ("sim/adapter.py", "SimRobot._map_entry"),
            ("sim/adapter.py", "SimRobot._note_plan"), ("sim/adapter.py", "SimRobot._path_plan"),
            ("sim/adapter.py", "SimRobot._map_margin"), ("sim/nav/astar.py", "plan_path"),
            # place: skill -> adapter -> grasp; push: skill -> adapter -> push_object, its nested tests and planners
            ("agent/skills.py", "Skills.place"), ("sim/adapter.py", "SimRobot.place"), ("sim/grasp.py", "place_box"),
            ("sim/grasp.py", "mark_discs"), ("agent/skills.py", "Skills.push"), ("sim/adapter.py", "SimRobot.push"),
            ("sim/push.py", "push_object"), ("sim/push.py", "push_object.mapped"), ("sim/push.py", "push_object.own_sq"),
            ("sim/push.py", "push_object.plan_blocked"), ("sim/push.py", "push_object.clear_place"),
            ("sim/push.py", "push_object.step_off"), ("sim/push.py", "_choose_plan"), ("sim/push.py", "_clear_plan"),
            ("sim/push.py", "room_places_reachable"), ("sim/push.py", "stamp_poly")}
    assert need <= seen, need - seen
    # the truth-mode / truth-only functions are not reached from a perceived path
    for k in (("sim/adapter.py", "SimRobot.true_pose"), ("sim/adapter.py", "SimRobot.room_of"),
              ("sim/adapter.py", "SimRobot.support_of"), ("sim/adapter.py", "SimRobot._seen_discs"),
              ("sim/adapter.py", "SimRobot._locate_truth"), ("sim/grasp.py", "true_box_pose"),
              ("sim/grasp.py", "plan_pregrasp"), ("sim/adapter.py", "SimRobot._push_eval")):
        assert k not in seen, k
    # ALLOWED is the one residual list: every entry cites row 19's letters and is on a scanned path (none stale)
    assert all(v[0] and set(v[0]) <= ROW19 and v[1] for v in ALLOWED.values()), ALLOWED
    assert allowed == set(ALLOWED), set(ALLOWED) - allowed
    assert hits == []


MUTANTS = {  # file, old (unique), new: each leaks truth into a perceived-mode path
    "locate: true pose for the perceived box": (
        "sim/adapter.py", "        pose = Pose(x, y, z, float(res[\"yaw\"]))\n",
        "        pose = self.true_pose(obj)\n"),
    "navigate avoid: true box pose in the perceived branch": (
        "sim/adapter.py", "        elif name in self._seen_pose:\n            x, y, yaw = self._seen_pose[name]\n",
        "        elif name in self._seen_pose:\n            b = self.d.body(self.body_of[name])\n"
        "            x, y, yaw = float(b.xpos[0]), float(b.xpos[1]), _yaw_of(b.xmat)\n"),
    "pick: true pose as pose_fn in perceived mode": (
        "sim/adapter.py", "            pose_fn = lambda: (np.array([p.x, p.y, p.z]), p.yaw)  # noqa: E731\n",
        "            pose_fn = lambda: true_box_pose(self.d, body)  # noqa: E731\n"),
    "seen discs: truth discs in perceived mode": (
        "sim/adapter.py", "        return [(x, y, BOX_REACH) for n, (x, y, _) in sorted(self._seen_pose.items()) if n not in skip]\n",
        "        return self._seen_discs(skip)\n"),
    "locate: segmentation px as the in-view test": (
        "sim/adapter.py", "        if not res[\"in_view\"]:\n            return Located(None, meta=meta)\n",
        "        if max(s.get(obj, 0) for s in self._cur_seg) < PX_VISIBLE:\n            return Located(None, meta=meta)\n"),
    "grasp: true yaw in the pre-grasp planner": (
        "sim/grasp.py", "    bp = np.asarray(bp, dtype=float).copy()\n    gp = bp + [0.0, 0.0, GRIP_DZ]\n    avoid =",
        "    bp = np.asarray(bp, dtype=float).copy()\n    gp = bp + [0.0, 0.0, GRIP_DZ]\n"
        "    axis = box_yaw(d.body('box_red').xmat)\n    avoid ="),
    "pushed map / push's map tests: true pose for the perceived footprint": (
        "sim/adapter.py",
        "            return contact_footprint(det[\"last_contact\"], tuple(float(v) for v in self._half(obj)[:2]))\n",
        "            t = self.true_pose(obj)\n            return t.x, t.y, t.yaw\n"),
    "push's map tests: true pose for an unmapped object's entry": (
        "sim/adapter.py", "        return None if e is None else e[:3]\n",
        "        return tuple(self.d.body(self.body_of[obj]).xpos[:2]) + (0.0,)\n"),
    "map-blocked memory: true pose for the pick target": (
        "sim/adapter.py", "            bp, axis = pose_fn()  # the pick target as pick planned it (perceived: the located pose)\n",
        "            bp, axis = true_box_pose(self.d, body)\n"),
    "pushed map: body pose in the contact geometry": (
        "sim/push.py", "    b, u = np.asarray(last_contact[\"base\"][:2], float), np.asarray(last_contact[\"u\"], float)\n",
        "    b, u = np.asarray(last_contact[\"d\"].body(\"crate\").xpos[:2], float), np.asarray(last_contact[\"u\"], float)\n"),
    # review phase 5 (probe_missed_mutants.py): the three that slipped through before place / push were roots
    "push mapped(): truth footprint instead of the map entry": (
        "sim/push.py", "        e = est(detail) if est is not None else None\n",
        "        e = (float(d.body(body).xpos[0]), float(d.body(body).xpos[1]), 0.0)\n"),
    "adapter.place: avoid discs from truth": (
        "sim/adapter.py", "        grid = mark_discs(self.grid, self._avoid_discs({held}))",
        "        grid = mark_discs(self.grid, self._seen_discs({held}))"),
    "skills.place: true_pose": (
        "agent/skills.py", "        r = self.robot.place(surface)\n",
        "        r = self.robot.place(surface); _ = self.robot.true_pose('red box')\n"),
    "skills: room from the truth API": (
        "agent/skills.py", "        return _res(f\"found: {object} in {loc.room}, on {loc.support}\", **loc.meta)\n",
        "        return _res(f\"found: {object} in {self.robot.room_of(object)}, on {loc.support}\", **loc.meta)\n"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_truth_leak_mutant_is_caught(name: str) -> None:
    rel, old, new = MUTANTS[name]
    src = (ROOT / rel).read_text()
    assert src.count(old) == 1, name
    _, hits, _ = scan({rel: src.replace(old, new)})
    assert hits, name


def test_perceived_run_with_truth_sabotaged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev seed 2001 (pick gate's 2nd trial: blue box, kitchen floor) from its room point, perceived mode, with every
    truth pose function raising and the segmentation blanked (geom id -1 everywhere): look -> locate -> pick ->
    navigate (seen boxes as avoid) work, and the located pose is the perceived one (within G5-pose's 2 cm)."""
    import numpy as np

    import sim.adapter as A
    import sim.grasp as G
    from agent.skills import Skills
    from sim.gate2 import _pose
    from sim.tasks import room_point_for

    r = A.SimRobot(2001, "none", start=_pose(room_point_for(2001, "none", "blue box")))
    truth = r.true_pose("blue box")

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("truth read in perceived mode")

    for f in ("true_pose", "room_of", "support_of", "_seen_discs", "_bottom_z", "_locate_truth"):
        monkeypatch.setattr(A.SimRobot, f, boom)
    monkeypatch.setattr(A.SimRobot, "_locate_eval", lambda self, obj, res: {})
    monkeypatch.setattr(G, "true_box_pose", boom)
    monkeypatch.setattr(G, "plan_pregrasp", boom)
    render = A.render_look
    monkeypatch.setattr(A, "render_look", lambda rr, d: (*render(rr, d)[:2], np.full((A.CAM_H, A.CAM_W), -1, np.int32)))
    s = Skills(r)
    s.look()
    loc = s.locate("blue box")
    assert loc.text == "found: blue box in kitchen, on floor", loc.text
    p = s.located["blue box"]
    assert np.linalg.norm(np.array([p.x - truth.x, p.y - truth.y, p.z - truth.z])) <= 0.02
    assert s.pick("blue box").text == "holding blue box"
    assert s.navigate("kitchen").text in ("reached", "failed: blocked")
