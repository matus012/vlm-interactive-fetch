"""Hand-written scripts that stand in for the VLM (Phase 2). They read `truth` to decide where to go but call the
skills in the brief's order through the runtime. Stdlib only. Each returns the steps used.

truth = {"pairs": [{"box", "room_point", "surface"}, ...],      # two_box has two, the rest one
         "blocker": None | "crate" | "cabinet", "detour_via": door name (the other kitchen door, cabinet task),
         "wrong_box": {"box", "room_point"}, "wrong_surface": surface name}

Correct scripts: fetch, blocked, push_or_detour (branch "push_then_detour" | "detour"), two_box.
Wrong scripts: wrong_colour, wrong_surface, box_in_gripper, step_limit.
"""
from __future__ import annotations

from collections.abc import Callable

from agent.runtime import Runtime


class _Abort(Exception):
    pass


def _go(rt: Runtime, place: str, truth: dict, branch: str = "push_then_detour") -> None:
    """navigate; on `failed: blocked` react as the task's blocker demands (never re-plans by itself, the script does)."""
    r = rt.call("navigate", {"place": place})
    if r.text != "failed: blocked":
        if not r.ok:
            raise _Abort(r.text)
        return
    blocker = truth.get("blocker")
    if blocker is None:
        raise _Abort(r.text)
    if blocker == "cabinet" and branch == "detour":
        r2 = rt.call("navigate", {"place": place, "via": truth["detour_via"]})
    else:
        p = rt.call("push", {"object": blocker})
        if p.ok:
            r2 = rt.call("navigate", {"place": place})
        elif truth.get("detour_via") is None:  # crate: no other route (was navigate(via=None) -> a validator error)
            raise _Abort(p.text)
        else:
            r2 = rt.call("navigate", {"place": place, "via": truth["detour_via"]})
    if not r2.ok:
        raise _Abort(r2.text)


def _fetch(rt: Runtime, pair: dict, truth: dict, branch: str, place_it: bool = True) -> None:
    box = pair["box"]
    _go(rt, pair["room_point"], truth, branch)
    for name, args in (("look", {}), ("locate", {"object": box}), ("pick", {"object": box})):
        if not (r := rt.call(name, args)).ok:
            raise _Abort(r.text)
    _go(rt, pair["surface"], truth, branch)
    if place_it and not (r := rt.call("place", {"surface": pair["surface"]})).ok:
        raise _Abort(r.text)


def _run(rt: Runtime, body: Callable[[], None]) -> int:
    try:
        body()
        rt.call("done", {"note": "finished"})
    except _Abort as e:
        rt.call("done", {"note": f"aborted: {e}"})
    return len(rt.steps)


def run_fetch(rt: Runtime, task: dict, truth: dict, branch: str = "push_then_detour") -> int:
    return _run(rt, lambda: [_fetch(rt, p, truth, branch) for p in truth["pairs"]])


run_blocked = run_two_box = run_fetch


def run_push_or_detour(rt: Runtime, task: dict, truth: dict, branch: str = "push_then_detour") -> int:
    return run_fetch(rt, task, truth, branch)


def run_wrong_colour(rt: Runtime, task: dict, truth: dict) -> int:
    w = truth["wrong_box"]
    return _run(rt, lambda: _fetch(rt, {**w, "surface": truth["pairs"][0]["surface"]}, truth, "push_then_detour"))


def run_wrong_surface(rt: Runtime, task: dict, truth: dict) -> int:
    p = truth["pairs"][0]
    return _run(rt, lambda: _fetch(rt, {**p, "surface": truth["wrong_surface"]}, truth, "push_then_detour"))


def run_box_in_gripper(rt: Runtime, task: dict, truth: dict) -> int:
    return _run(rt, lambda: _fetch(rt, truth["pairs"][0], truth, "push_then_detour", place_it=False))


def run_step_limit(rt: Runtime, task: dict, truth: dict) -> int:
    """Keeps calling past the limit (the runtime refuses the call after the last allowed step: rt.n_refused 1)."""
    while not rt.call("look", {}).text.startswith("error: step limit"):
        pass
    return len(rt.steps)


CORRECT = {"fetch": run_fetch, "blocked": run_blocked, "push_or_detour": run_push_or_detour, "two_box": run_two_box}
WRONG = {"wrong_colour": run_wrong_colour, "wrong_surface": run_wrong_surface, "box_in_gripper": run_box_in_gripper,
         "step_limit": run_step_limit}


def run_script(rt: Runtime, task: dict, truth: dict) -> int:
    """Dispatch on task['type'] to the correct script."""
    return CORRECT[task["type"]](rt, task, truth)
