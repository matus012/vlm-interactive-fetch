"""Seal the 100-task test set (task 005 §Sealing, gate G4-seal). Run ONCE, at the frozen commit, after the gates pass.

    MUJOCO_GL=egl python -m sim.seal              # sealed: test-space seeds -> $NAV116_SEALED_DIR (default tasksets/sealed)/test_100.jsonl
    MUJOCO_GL=egl python -m sim.seal --dry-run --out DIR/x.jsonl   # same code path on DEV seeds (DRY_START..), any tree
    MUJOCO_GL=egl python -m sim.seal --e1 [--dry-run --out DIR/x.jsonl]   # task 009 E1 seal -> sealed/e1_100.jsonl

25 tasks per type (TYPES order), sim.sampler.draw_set(type, 25, start, sealed_draw=True,
exclude=SEALED_EXCLUDE | seeds already used by earlier types, min_surface=MIN_SURFACE_SEALED): fresh seeds, no seed
twice, none a test touches. Refuses a dirty tree (sealed mode) and an existing output file (both modes). Writes the
tasks as JSON lines plus <out>.meta.json (commit, rule, counts, sha256; no task content). Prints only the counts per
type, the sha256 of the file and the generator commit; never the tasks.

Guards added after the 005 review (S1, S2):
- Renderer (S1): the draw depends on rendered segmentation px (hall_px < 10, look_px), and Mesa llvmpipe px differ
  from NVIDIA EGL (005 finding: hall_px seed 1000 red 138 vs 130). Sealed mode refuses unless GL_RENDERER names NVIDIA;
  the renderer string and the MuJoCo version go into meta.json.
- Frozen dependencies (S2): task generation also runs the pre-grasp planner, A* and follower margins, the layout and
  the places. meta.json records the sha256 of every file in DEP_FILES, and sealed mode requires a local tag
  `seal-*` on HEAD (the generator commit).
- Phase 6 task-check rule: Phase 6 regenerates each sealed task with sim.agent_run's make_task(seed, type,
  want=slot, sealed_draw=True) check. If any DEP_FILES hash at the Phase 6 commit differs from meta.json, the task
  regeneration runs from a worktree at the seal tag (the frozen generator), never from HEAD; any task that then fails
  its equality check voids the seal (log it, redraw with new fresh seeds per task 005).
E2 (task 011): with NAV116_HOUSE=B the same rule draws house B's seal: seeds from E2_START, tag `seal-011`,
sealed/e2_100.jsonl, house B's two world files added to DEP_FILES, "house": "B" in meta.
E1 (task 009, context row 22): `--e1` draws with the generator's easings off (set_easings(False) before any world),
from E1_START (fresh test seeds above every seal-005 seed, 100001-100140, and above every SEALED_EXCLUDE seed but
199999, which stays excluded), requires the tag `seal-009` on HEAD and writes sealed/e1_100.jsonl; same rule
otherwise. The sampler's surface slots stay nightstand-only: the kept out-of-sight rule rejects every coffee-table box
(task 009 report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from sim.constants import MIN_SURFACE_SEALED, SEALED_EXCLUDE, TEST_SEEDS
from sim.world.generate import HOUSE
from sim.sampler import draw_set
from sim.tasks import TYPES

DEP_FILES = ("sim/world/generate.py", "sim/world/layout.yaml", "sim/world/places.yaml", "sim/world/house.py",
             "sim/world/occupancy.py", "sim/tasks.py", "sim/sampler.py", "sim/constants.py", "sim/grasp.py",
             "sim/body.py", "sim/robot.py", "sim/look.py", "sim/photo.py", "sim/nav/astar.py", "sim/nav/follow.py",
             "sim/adapter.py", "agent/robot_api.py") + (
    ("sim/world/layout_b.yaml", "sim/world/places_b.yaml") if HOUSE == "B" else ())  # E2 (task 011): house B's files
N_PER_TYPE = 25
DRY_START, DRY_STOP = 1400, 2000  # dev seeds for --dry-run (outside every Phase 2-4 dev range)
E1_START = 160000  # task 009: E1 seal seeds start here
E2_START = 170000  # task 011: house B seal seeds (seal-005 100001-100140, seal-009 160000-160116 below)
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def sealed_path(e1: bool = False) -> Path:
    name = "e2_100.jsonl" if HOUSE == "B" else "e1_100.jsonl" if e1 else "test_100.jsonl"
    return Path(os.environ.get("NAV116_SEALED_DIR", "tasksets/sealed")) / name


def gl_renderer() -> str:
    """GL_RENDERER of the context MuJoCo renders with (MUJOCO_GL=egl)."""
    from mujoco.gl_context import GLContext
    from OpenGL import GL
    c = GLContext(64, 64)
    c.make_current()
    r = GL.glGetString(GL.GL_RENDERER)
    c.free()
    return r.decode() if isinstance(r, bytes) else str(r)


def dep_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in DEP_FILES}


def seal(out: Path, dry_run: bool, e1: bool = False) -> dict:
    import mujoco
    if e1:
        from sim.world.generate import set_easings
        set_easings(False)
    if not dry_run and git("status", "--porcelain"):
        raise SystemExit("refusing: dirty tree (commit or stash first; the seal records the generator commit)")
    renderer = gl_renderer()
    if not dry_run and "NVIDIA" not in renderer:
        raise SystemExit(f"refusing: GL_RENDERER {renderer!r} is not NVIDIA (llvmpipe px differ; S1)")
    tags = [t for t in git("tag", "--points-at", "HEAD").split() if t.startswith("seal-")]
    if not dry_run and e1 and "seal-009" not in tags:
        raise SystemExit("refusing: tag HEAD `seal-009` first (E1)")
    if not dry_run and HOUSE == "B" and ("seal-011" not in tags or e1):
        raise SystemExit("refusing: house B seals need the tag `seal-011` on HEAD and the easings on (E2)")
    if not dry_run and not tags:
        raise SystemExit("refusing: tag HEAD `seal-<name>` first (the frozen generator commit; S2)")
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (a seal is never overwritten; any redraw uses new fresh seeds)")
    start, stop = (DRY_START, DRY_STOP) if dry_run else (
        E2_START if HOUSE == "B" else E1_START if e1 else TEST_SEEDS.start, TEST_SEEDS.stop)
    used: set[int] = set()
    tasks = []
    for t in TYPES:
        ts = draw_set(t, N_PER_TYPE, start, stop=stop, exclude=SEALED_EXCLUDE | used, sealed_draw=not dry_run,
                      min_surface=MIN_SURFACE_SEALED)
        used |= {x["seed"] for x in ts}
        tasks += ts
    data = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in tasks).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as f:
        f.write(data)
    meta = {"file": out.name, "sha256": hashlib.sha256(data).hexdigest(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "dry_run": dry_run, "tags": tags, "gl_renderer": renderer,
            "mujoco": mujoco.__version__, "dep_sha256": dep_hashes(), "seed_space": [start, stop],
            "exclude": sorted(SEALED_EXCLUDE), "counts": {t: sum(x["type"] == t for x in tasks) for t in TYPES},
            **({"e1": True, "easings": False} if e1 else {}), **({"house": "B"} if HOUSE == "B" else {}),
            "rule": f"sim.seal: per type in {list(TYPES)}, sim.sampler.draw_set(type, {N_PER_TYPE}, {start}, "
                    f"stop={stop}, exclude=SEALED_EXCLUDE | earlier types' seeds, sealed_draw={not dry_run}, "
                    f"min_surface={MIN_SURFACE_SEALED})" + (", easings off (set_easings(False))" if e1 else "")}
    with open(out.with_suffix(".meta.json"), "x") as f:
        json.dump(meta, f, indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, help="dry-run output (required with --dry-run, refused without)")
    ap.add_argument("--e1", action="store_true", help="task 009: easings off, E1 seeds, tag seal-009")
    a = ap.parse_args()
    if a.dry_run != (a.out is not None):
        raise SystemExit("--out goes with --dry-run only; the sealed file has one fixed path")
    meta = seal(a.out if a.dry_run else sealed_path(a.e1), a.dry_run, a.e1)
    print(json.dumps({k: meta[k] for k in ("counts", "sha256", "commit", "dry_run")}))


if __name__ == "__main__":
    main()
