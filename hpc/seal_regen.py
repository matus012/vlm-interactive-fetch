"""H3 (task 007, context row 21): regenerate every sealed task with the seal-005 generator and write the verified
task dicts that `sim.agent_run --verified` runs (row 20). Runs inside the PERUN job (NVIDIA), never by hand.

    python hpc/seal_regen.py --worktree <tree at seal-005> --sealed <test_100.jsonl> --meta <test_100.meta.json>
        --out <verified.jsonl>

The seal's dependency files differ at freeze-006 (sim/grasp.py, sim/adapter.py, agent/robot_api.py), so by
sim/seal.py's Phase 6 rule the tasks are regenerated from the seal tag's code, never from HEAD. Trap guard: `sim` and
`agent` must import from the worktree, whose HEAD must be ba7bfff (read from .git/HEAD: GPU nodes have no git;
hpc/sync_code.sh leaves a detached sha there), the tree's 17 seal DEP_FILES must hash as in the seal's meta (binds
the bytes, not just the ref; the meta holds no tasks), the sealed file's sha256 must be the seal's (and the meta's),
the renderer NVIDIA. The summary records the MuJoCo version (the seal's meta: 3.17.0).
Each entry: make_task(seed, type, want=slot, sealed_draw=True) | {"slot": slot} must equal it on TASK_KEYS + slot.
Writes <out> only if every task is equal. Prints counts and the verified sha256 only, never task contents.
Exit 0 = all equal; 1 = a mismatch or error; an AssertionError (exit 1) = a guard failed.
E1 (task 009, row 22): `--seal-commit <seal-009 sha> --sha256 <E1 seal sha256> --e1` regenerates the E1 seal from its
own tag's tree with the easings off (set before any world is drawn); "easings" joins the compared keys.
E2 (task 011): with NAV116_HOUSE=B (set in the job's environment) and `--seal-commit <seal-011> --sha256 <E2 sha>`:
the meta's "house" must be B and "house" joins the compared keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

SEAL_COMMIT = "ba7bfff3d9060f3c050d0fb41966d9b20522fbf6"  # tag seal-005
SEALED_SHA256 = "8f691844eb8312f05c1a40f386b73faf515be656948b42fe76a510bceede542d"


def head_commit(tree: Path) -> str:
    """HEAD of a git tree without git: .git is a directory (sync_code.sh clone, detached) or a worktree's gitdir file."""
    g = tree / ".git"
    if g.is_file():
        g = Path(g.read_text().split("gitdir:", 1)[1].strip())
    return (g / "HEAD").read_text().strip()


def guard(tree: Path, seal_commit: str = SEAL_COMMIT) -> None:
    import agent
    import sim
    for m in (sim, agent):
        assert Path(m.__file__).resolve().is_relative_to(tree), f"trap: {m.__name__} from {m.__file__}, not {tree}"
    head = head_commit(tree)
    assert head == seal_commit, f"trap: {tree} HEAD {head!r} is not the seal commit {seal_commit}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--sealed", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="the seal's meta.json (dep_sha256, sha256; no tasks)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha256", default=SEALED_SHA256, help="expected sha256 of --sealed (dev tests; E1)")
    ap.add_argument("--seal-commit", default=SEAL_COMMIT, help="the seal tag's commit (E1: seal-009)")
    ap.add_argument("--e1", action="store_true", help="task 009: easings off")
    a = ap.parse_args()
    tree = a.worktree.resolve()
    sys.path.insert(0, str(tree))
    guard(tree, a.seal_commit)
    if a.e1:
        from sim.world.generate import set_easings
        set_easings(False)
    raw = a.sealed.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == a.sha256, "sealed file sha256 differs from the seal"
    meta = json.loads(a.meta.read_text())
    assert meta["sha256"] == a.sha256, "meta sha256 differs from the sealed file's"
    assert meta["commit"] == a.seal_commit, "meta commit differs from --seal-commit"
    assert meta.get("easings", True) is not a.e1, "meta easings flag does not match --e1"
    from sim.world.generate import HOUSE
    assert meta.get("house", "A") == HOUSE, f"meta house {meta.get('house', 'A')} != NAV116_HOUSE {HOUSE} (task 011)"
    import mujoco
    from sim.agent_run import TASK_KEYS
    from sim.seal import DEP_FILES, gl_renderer
    bad = [f for f in DEP_FILES if hashlib.sha256((tree / f).read_bytes()).hexdigest() != meta["dep_sha256"][f]]
    assert set(DEP_FILES) == set(meta["dep_sha256"]) and not bad, f"trap: seal dependency files differ from the meta: {bad}"
    from sim.tasks import make_task
    renderer = gl_renderer()
    assert "NVIDIA" in renderer, f"renderer {renderer!r} is not NVIDIA"
    keys = (*TASK_KEYS, "slot", *(("easings",) if a.e1 else ()), *(("house",) if HOUSE != "A" else ()))
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    out, n_equal, errors = [], 0, Counter()
    for e in rows:
        try:
            t = make_task(e["seed"], e["type"], want=e["slot"], sealed_draw=True) | {"slot": dict(e["slot"])}
        except Exception as ex:  # noqa: BLE001 - count by class only: a traceback could print task contents
            errors[type(ex).__name__] += 1
            continue
        n_equal += {k: t[k] for k in keys} == {k: e[k] for k in keys}
        out.append(t)
    ok = n_equal == len(rows) > 0
    summary = {"n": len(rows), "n_per_type": dict(Counter(e["type"] for e in rows)), "n_equal": n_equal,
               "n_mismatch": len(out) - n_equal, "errors": dict(errors), "seal_commit": a.seal_commit, "e1": a.e1, "house": HOUSE,
               "renderer": renderer, "mujoco": mujoco.__version__, "sealed_sha256": a.sha256}
    if ok:
        data = "".join(json.dumps(t, sort_keys=True, separators=(",", ":")) + "\n" for t in out).encode()
        tmp = a.out.with_name(a.out.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, a.out)
        summary["verified_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(summary))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
