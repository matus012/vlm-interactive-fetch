"""Runtime: validate -> name-check -> step limit -> skill -> one JSONL line per step. Stdlib only.

Failure label (`label_failure`, assigned only to a failed run, from the log records alone). One of
{vlm, skill, nav, perception, sim, scorer}. First matching rule wins.

SCRIPTED runs (Phase 2, G2 gates; unchanged):
  1. sim_ok False in the end line                          -> sim
  2. the last UNRECOVERED failed skill result (a `failed:` line with no later ok line of the same skill; a failed
     push is unrecovered only if no later navigate succeeded):
       navigate -> nav;  locate -> perception;  pick / place / push -> skill
  3. no unrecovered failure: `error:` lines or the step limit -> skill (interface / budget mismatch of the script);
     otherwise every skill returned ok yet the scorer says fail -> scorer

AGENT runs (Phase 3+, row 14: the label is the part that returned a WRONG result; a skill that correctly reported
that the call cannot work is not the failing part, the VLM chose the call):
  CORRECT failures (never blame the skill): `failed: not in view`, `blocked` (navigate; pick / place whose own drive
  stopped on the crate / cabinet, row 16), `not movable`, `stuck`, `not blocking` (push, row 17), `not located`,
  `nothing held`, `already holding something`, `not a box`.
  WRONG results the runtime can see without the scorer: `navigate` -> `failed: no path` (nav); `pick` -> any other
  failure (missed / slipped / unreachable: the box was located, so it was named, in view and not held) (skill);
  `place` -> any failure but `nothing held` / `blocked` (skill). A wrong failure counts only if UNRECOVERED (a later ok result of
  the same skill).
  1. sim_ok False                                          -> sim
  1b. tripwire (row 14 addendum, task 006): a push that said `pushed`, then a later `failed: blocked` (any skill) whose
     logged drive contacts include that object                       -> skill (the push's end did not clear it)
  1c. lost box (`_lost_box`)                                -> skill
  2. last unrecovered WRONG failure                         -> nav (navigate) / skill (pick, place)
  3. a target box whose `place` returned ok on its own target surface while the scorer's per-target check says it is
     not on that surface (not upright / not still / off the rect)       -> skill
  4. anything else (errors, step limit, correct failures not acted on, wrong order, wrong box / surface) -> vlm
`perception` is never assigned to an agent run: "locate said found but the box was not there" needs the sim's truth,
which the log does not hold; the scorer's wrong_box / surface outcome of a located box is the VLM's choice.
Scripts that are deliberately WRONG (G2-wrong) are checked on the scorer's named rule, not on the label.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.robot_api import RobotAPI
from agent.skills import Result, Skills, check_names
from agent.validate import validate_call

LABELS = ("vlm", "skill", "nav", "perception", "sim", "scorer")
CORRECT_FAIL = {"failed: not in view", "failed: blocked", "failed: not movable", "failed: stuck", "failed: not blocking",
                "failed: not located", "failed: nothing held", "failed: already holding something", "failed: not a box"}
_FAIL = {"navigate": "nav", "locate": "perception", "pick": "skill", "place": "skill", "push": "skill"}


def _unrecovered(steps: list[dict], scripted: bool) -> list[dict]:
    out = []
    for i, s in enumerate(steps):
        if s["result"].startswith("failed"):
            later = steps[i + 1:]
            if scripted and s["call"] == "push":
                rec = any(t["call"] == "navigate" and t["ok"] for t in later)
            else:
                rec = any(t["call"] == s["call"] and t["ok"] for t in later)
            if not rec:
                out.append(s)
    return out


def _agent_wrong(s: dict) -> str | None:
    r, c = s["result"], s["call"]
    if c == "navigate" and r == "failed: no path":
        return "nav"
    if c == "pick" and r.startswith("failed") and r not in CORRECT_FAIL:
        return "skill"
    if c == "place" and r.startswith("failed") and r not in CORRECT_FAIL:
        return "skill"
    return None


def _placed_not_on_surface(steps: list[dict], end: dict) -> bool:
    targets = ((end.get("score") or {}).get("detail") or {}).get("targets", {})
    held = None
    last_place: dict[str, str] = {}  # box -> surface of its last ok place
    for s in steps:
        if not s["ok"]:
            continue
        if s["call"] == "pick":
            held = s["args"]["object"]
        elif s["call"] == "place" and held is not None:
            last_place[held], held = s["args"]["surface"], None
    return any(t["ok"] is False and last_place.get(b) == t["surface"] for b, t in targets.items())


def _lost_box(steps: list[dict], end: dict) -> bool:
    """A held box was lost without any skill reporting it (ruling 2026-11-13, row 14: a wrong result -> `skill`): after an
    ok pick and before an ok place, a later result shows nothing held (place -> "failed: nothing held"), or the run ends
    with nothing held while the log never released the box. "Held" = grasp.holding (deviation 18): place's precondition,
    and at the end the scorer detail's "holding" (logs before task 006 have only the scorer's "held")."""
    held = None
    for s in steps:
        if s["call"] == "pick" and s["ok"]:
            held = s["args"]["object"]
        elif s["call"] == "place" and s["ok"]:
            held = None
        elif s["call"] == "place" and held is not None and s["result"] == "failed: nothing held":
            return True
    det = (end.get("score") or {}).get("detail") or {}
    key = "holding" if "holding" in det else "held"
    return held is not None and key in det and det[key] is None


def _contacts(s: dict) -> set[str]:
    """Object names in a step's logged drive contacts ("<object>|<robot geom>" keys): navigate's meta.contacts, pick's /
    place's meta.detail.go_to (a list of drives for pick, one drive for place)."""
    meta = s.get("meta", {})
    det = meta.get("detail")
    drives = (det.get("go_to") if isinstance(det, dict) else None) or []  # navigate's detail is a text
    drives = [drives] if isinstance(drives, dict) else drives
    keys = [*meta.get("contacts", {}), *(k for g in drives for k in (g.get("contacts") or {}))]
    return {k.split("|")[0] for k in keys}


def _pushed_then_blocked(steps: list[dict]) -> bool:
    pushed: set[str] = set()
    for s in steps:
        if s["call"] == "push" and s["result"] == "pushed":
            pushed.add(s["args"]["object"])
        elif s["result"] == "failed: blocked" and pushed & _contacts(s):
            return True
    return False


def label_failure(steps: list[dict], end: dict, scripted: bool) -> str:
    if end.get("sim_ok") is False:
        return "sim"
    bad = _unrecovered(steps, scripted)
    if scripted:
        if bad:
            return _FAIL[bad[-1]["call"]]
        if end.get("step_limit_hit") or any(s["result"].startswith("error") for s in steps):
            return "skill"
        return "scorer"
    if _pushed_then_blocked(steps) or _lost_box(steps, end):
        return "skill"
    wrong = [w for s in bad if (w := _agent_wrong(s))]
    if wrong:
        return wrong[-1]
    return "skill" if _placed_not_on_surface(steps, end) else "vlm"


class Runtime:
    def __init__(self, skills: Skills, robot: RobotAPI, task: dict, log_path: str | Path, scripted: bool = True) -> None:
        self.skills, self.robot, self.task, self.scripted = skills, robot, task, scripted
        self.limit = int(task["step_limit"])
        self.steps: list[dict] = []
        self.over = False  # step limit reached or done called: no more calls accepted
        self.done_called = False
        self.n_refused = 0  # calls refused after the step limit (the run asked for more steps than the limit)
        self._t0 = time.time()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.log_path.open("w")

    def _write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def call(self, name: str, args: dict | str | None = None, extra: dict | None = None) -> Result:
        """One step. `args` may be the raw JSON string a VLM tool call carries (invalid JSON -> an error step).
        `extra` (agent loop: packet, VLM output, ...) is stored in the step record as is."""
        args = args or {}
        if self.over:
            self.n_refused += not self.done_called
            return Result("error: run ended" if self.done_called else "error: step limit reached", ok=False)
        ok, why = validate_call(name, args)
        if ok and isinstance(args, str):
            args = json.loads(args)
        err = None if ok else f"error: {why}"
        if ok and (e := check_names(self.robot, name, args)):
            err = e
        return self._step(name, args, err, extra)

    def error_step(self, text: str, extra: dict | None = None) -> Result:
        """A step with no single valid call (the agent loop: zero or several tool calls). Counts as a step."""
        assert text.startswith("error: ") and not self.over
        return self._step("", {}, text, extra)

    def _step(self, name: str, args: dict | str, err: str | None, extra: dict | None) -> Result:
        sim_t = getattr(self.robot, "sim_time", None)
        sim0 = sim_t() if callable(sim_t) else None
        w0 = time.time()
        res = Result(err, ok=False) if err else getattr(self.skills, name)(**args)
        if name == "done" and not err:
            self.over = self.done_called = True
        step = {"step": len(self.steps) + 1, "t_wall": round(time.time() - self._t0, 3), "call": name, "args": args,
                "validation": err or "ok", "result": res.text, "ok": res.ok, "skill_wall_s": round(time.time() - w0, 3),
                "photos": [{"path": p.path, "heading_deg": p.heading_deg, "visible": p.visible} for p in res.photos]}
        if sim0 is not None:
            step["sim_s"] = sim_t()
            step["sim_dt"] = round(step["sim_s"] - sim0, 4)
        if res.meta:
            step["meta"] = res.meta
        if extra:
            step["extra"] = extra
        self.steps.append(step)
        self._write(step)
        if len(self.steps) >= self.limit:
            self.over = True  # the run ends here; any further call is refused
        return res

    @property
    def steps_used(self) -> int:
        """THE step count (end line and scorer): steps logged + calls refused after the limit. A run that reached the
        limit without `done` asked for more than the limit, so it counts at least one refused call even when the
        caller stopped asking (the agent loop stops at the limit): `done` as step 20 of 20 -> 20, no `done` -> 21."""
        unfinished = len(self.steps) >= self.limit and not self.done_called
        return len(self.steps) + max(self.n_refused, int(unfinished))

    def finish(self, score: dict | None, extra: dict | None = None) -> dict:
        """Write the end line. `score` = scorer output (its detail.steps_used must be `self.steps_used`) or a gate's
        {'pass': bool}; `extra` = more end-line fields (agent loop summary). Returns the end line."""
        if score is not None and "steps_used" in score.get("detail", {}):
            assert score["detail"]["steps_used"] == self.steps_used, (score["detail"]["steps_used"], self.steps_used)
        passed = None if score is None else bool(score.get("pass", score.get("passed")))
        self.step_limit_hit = len(self.steps) >= self.limit and not self.done_called
        end = {"event": "end", "steps_used": self.steps_used, "steps_logged": len(self.steps), "n_refused": self.n_refused,
               "done_called": self.done_called, "step_limit_hit": self.step_limit_hit, "sim_ok": self.robot.sim_ok(),
               "score": score, "label": None}
        if passed is False or (score is None and not self.done_called):
            end["label"] = label_failure(self.steps, end, self.scripted)
        end |= extra or {}
        self._write(end)
        self._f.close()
        return end
