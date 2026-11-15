"""E4 (task 014) seal: 100 fresh test-space tasks, both houses, drawn with sim.e4.make_task_e4; and its in-job regen.

    NAV116_HOUSE=A python -m sim.e4_seal draw --spec place_ref:17,recovery:17,hint:16 --part P_A.jsonl [--dry-run]
    NAV116_HOUSE=B python -m sim.e4_seal draw --spec ... --part P_B.jsonl [--dry-run]
    (vague tasks: their own process, --labels: the generator's set_labels(True) before any world)
    python -m sim.e4_seal merge --parts P_A.jsonl P_B.jsonl [...] [--out X.jsonl --dry-run]
        -> $NAV116_SEALED_DIR (default tasksets/sealed)/e4_100.jsonl + .meta.json (prints counts, sha256, commit only)
    NAV116_HOUSE=A python -m sim.e4_seal regen --sealed S --meta M --out V_A.jsonl [--labels]   (the PERUN job)

Seeds: per (house, category) its own start, SEED_START below (fresh: above every seal-005 / -009 / -011 seed, outside
SEALED_EXCLUDE), scanned in order; each seed fills the first open slot it can (draw_slots: balanced fetch slots for
recovery / hint, an on / next split for place_ref), at most one task per seed; no seed used twice. Sealed (not --dry-run) draws refuse a dirty tree,
a non-NVIDIA renderer (segmentation px decide out-of-sight), a HEAD without the tag `seal-014`, an existing output.
Meta: commit, tags, renderer, MuJoCo version, sha256 of every dependency file (sim.seal.DEP_FILES + house B's files +
sim/e4.py), seed starts, counts per house x category. Regen: each task of the given house (and labels flag) is drawn
again with make_task_e4(seed, category, sealed_draw=True) and must equal the sealed line; the dependency hashes of
the running code must equal the meta's; any difference -> exit 1, nothing written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from sim.constants import SEALED_EXCLUDE, TEST_SEEDS
from sim.seal import DEP_FILES as BASE_DEPS, REPO, git, gl_renderer

SEED_START = {("A", "vague"): 180000, ("A", "place_ref"): 181000, ("A", "recovery"): 182000, ("A", "hint"): 183000,
              ("B", "vague"): 185000, ("B", "place_ref"): 186000, ("B", "recovery"): 187000, ("B", "hint"): 188000}
DEPS = tuple(dict.fromkeys(BASE_DEPS + ("sim/world/layout_b.yaml", "sim/world/places_b.yaml", "sim/e4.py")))
TAG = "seal-014"


def line(t: dict) -> str:
    return json.dumps(t, sort_keys=True, separators=(",", ":"))


def dep_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in DEPS}


def sealed_path() -> Path:
    return Path(os.environ.get("NAV116_SEALED_DIR", "tasksets/sealed")) / "e4_100.jsonl"


def guards(dry_run: bool) -> str:
    renderer = gl_renderer()
    if dry_run:
        return renderer
    if git("status", "--porcelain"):
        raise SystemExit("refusing: dirty tree")
    if "NVIDIA" not in renderer:
        raise SystemExit(f"refusing: GL_RENDERER {renderer!r} is not NVIDIA")
    if TAG not in git("tag", "--points-at", "HEAD").split():
        raise SystemExit(f"refusing: tag HEAD `{TAG}` first")
    return renderer


def draw_slots(cat: str, n: int, start: int) -> list[dict | None]:
    """review 014 B2 / S2. recovery / hint: the balanced sampler's fetch slots (sim.sampler.plan: room uniform,
    support with the sealed set's surface minimum scaled to n, surface uniform), as the 86/100 condition. place_ref:
    n - n // 2 "next" (floor anchor) and n // 2 "on" (spawn-surface anchor) slots, interleaved. vague: no slot."""
    from sim.constants import MIN_SURFACE_SEALED
    from sim.sampler import plan
    if cat in ("recovery", "hint"):
        return plan("fetch", n, start, min_surface=MIN_SURFACE_SEALED * n // 25)
    if cat == "place_ref":
        return [{"relation": "next" if i % 2 == 0 else "on"} for i in range(n)]
    return [None] * n


def draw(spec: str, part: Path, dry_run: bool, labels: bool) -> None:
    if labels:
        from sim.world.generate import set_labels
        set_labels(True)
    from sim.e4 import make_task_e4
    from sim.tasks import NoTask
    from sim.world.generate import HOUSE
    renderer = guards(dry_run)
    if part.exists():
        raise SystemExit(f"refusing: {part} exists")
    out, counts = [], {}
    for item in spec.split(","):
        cat, n = item.split(":")
        n = int(n)
        assert (cat == "vague") == labels, "vague tasks are drawn in their own --labels process, and only they"
        # dry run: the same code path on dev seeds (3000 + 500 per (house, category) slot)
        start = 3000 + 500 * list(SEED_START).index((HOUSE, cat)) if dry_run else SEED_START[(HOUSE, cat)]
        slots = draw_slots(cat, n, start)
        got: list[dict | None] = [None] * n
        seed = start
        while not all(got):
            if seed in SEALED_EXCLUDE:
                seed += 1
                continue
            assert (seed < start + 900) and (dry_run or seed in TEST_SEEDS), f"{cat}: slots unfilled up to seed {seed}"
            for i, want in enumerate(slots):  # each seed fills the first open slot it can (sim.sampler.draw_set's rule)
                if got[i] is None:
                    try:
                        got[i] = make_task_e4(seed, cat, sealed_draw=not dry_run, want=want)
                        break
                    except NoTask:
                        continue
            seed += 1
        out += got
        counts[f"{HOUSE}:{cat}"] = n
    with open(part, "x") as f:
        f.write("".join(line(t) + "\n" for t in out))
    print(json.dumps({"house": HOUSE, "counts": counts, "renderer": renderer, "dry_run": dry_run}))


def merge(parts: list[Path], out: Path, dry_run: bool) -> None:
    import mujoco
    renderer = guards(dry_run)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (a seal is never overwritten)")
    tasks = [json.loads(x) for p in parts for x in p.read_text().splitlines() if x.strip()]
    assert len({t["seed"] for t in tasks}) == len(tasks), "a seed twice"
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    data = "".join(line(t) + "\n" for t in tasks).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as f:
        f.write(data)
    counts: dict[str, int] = {}
    for t in tasks:
        k = f"{t.get('house', 'A')}:{t['category']}"
        counts[k] = counts.get(k, 0) + 1
    meta = {"file": out.name, "sha256": hashlib.sha256(data).hexdigest(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "dry_run": dry_run,
            "tags": [t for t in git("tag", "--points-at", "HEAD").split() if t.startswith("seal-")],
            "gl_renderer": renderer, "mujoco": mujoco.__version__, "dep_sha256": dep_hashes(),
            "seed_start": {f"{h}:{c}": s for (h, c), s in SEED_START.items()}, "counts": counts, "n": len(tasks),
            "rule": "sim.e4_seal draw: per (house, category) seeds from SEED_START in order, each filling the first "
                    "open draw_slots slot via make_task_e4(seed, category, sealed_draw=True, want=slot); SEALED_EXCLUDE "
                    "skipped"}
    with open(out.with_suffix(".meta.json"), "x") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("counts", "n", "sha256", "commit", "dry_run")}))


def regen(sealed: Path, meta_p: Path, out: Path, labels: bool) -> int:
    if labels:
        from sim.world.generate import set_labels
        set_labels(True)
    from sim.e4 import make_task_e4
    from sim.world.generate import HOUSE
    meta = json.loads(meta_p.read_text())
    data = sealed.read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        print("sealed file sha256 != meta", file=sys.stderr)
        return 1
    bad = [f for f, h in meta["dep_sha256"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    if bad:
        print(f"dependency files differ from the seal: {bad}", file=sys.stderr)
        return 1
    if "NVIDIA" not in gl_renderer():
        print("renderer is not NVIDIA", file=sys.stderr)
        return 1
    mine = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
    mine = [t for t in mine if t.get("house", "A") == HOUSE and bool(t.get("labels")) == labels]
    ok = []
    for t in mine:
        r = make_task_e4(t["seed"], t["category"], sealed_draw=True, want=t.get("slot"))
        if line(r) != line(t):
            print(f"mismatch at sealed line of category {t['category']}", file=sys.stderr)
            return 1
        ok.append(t)
    with open(out, "x") as f:
        f.write("".join(line(t) + "\n" for t in ok))
    print(json.dumps({"house": HOUSE, "labels": labels, "regenerated_equal": len(ok)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--spec", required=True)
    d.add_argument("--part", type=Path, required=True)
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--labels", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--parts", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path)
    m.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("regen")
    r.add_argument("--sealed", type=Path, required=True)
    r.add_argument("--meta", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    if a.cmd == "draw":
        draw(a.spec, a.part, a.dry_run, a.labels)
    elif a.cmd == "merge":
        if a.dry_run != (a.out is not None):
            raise SystemExit("--out goes with --dry-run only")
        merge(a.parts, a.out or sealed_path(), a.dry_run)
    else:
        return regen(a.sealed, a.meta, a.out, a.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
