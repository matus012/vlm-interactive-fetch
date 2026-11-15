"""Task 005 pre-work 4, 5, 6, 9: balanced sampler plan, < 10 px rule, pre-grasp filter, sealed-draw exclusion list,
seal script (dry-run only: no test-space seed is opened here)."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import sim.seal as seal
import sim.tasks as T
from sim.constants import MIN_SURFACE_COUNTED, PX_OUT_OF_SIGHT, SEALED_EXCLUDE, TEST_SEEDS
from sim.sampler import draw_set, plan, slot_ok, table
from sim.world.generate import load_layout

TESTS = Path(__file__).resolve().parent


def seed_literals(path: Path) -> set[int]:
    """Test-space seeds a test file names: int literals in TEST_SEEDS and TEST_SEEDS.start/.stop +- a literal."""
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        expr = node if isinstance(node, (ast.Constant, ast.Attribute, ast.BinOp)) else None
        if expr is None:
            continue
        src = ast.unparse(expr)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int) or "TEST_SEEDS." in src:
            try:
                v = eval(src.replace("T.TEST_SEEDS", "TEST_SEEDS"), {"TEST_SEEDS": TEST_SEEDS})
            except NameError:  # an expression with other names in it is not a seed literal
                continue
            if isinstance(v, int) and v in TEST_SEEDS:
                out.add(v)
    return out


def test_exclusion_covers_every_test_seed() -> None:
    used = set().union(*(seed_literals(p) for p in TESTS.glob("*.py")))
    assert {100000, 150123, 199999} <= used  # the scan finds the known ones (TEST_SEEDS.start / .stop - 1 too)
    assert used <= SEALED_EXCLUDE, sorted(used - SEALED_EXCLUDE)


def test_sealed_draw_refuses_without_exclusion() -> None:
    with pytest.raises(AssertionError):
        draw_set("fetch", 1, TEST_SEEDS.start, sealed_draw=True)


@pytest.mark.parametrize("ttype", T.TYPES)
@pytest.mark.parametrize("n,m", [(10, MIN_SURFACE_COUNTED), (20, 0), (25, 5)])
def test_plan_balanced(ttype: str, n: int, m: int) -> None:
    p = plan(ttype, n, 8000, m)
    assert p == plan(ttype, n, 8000, m)
    tab = table([{"slot": s} for s in p])
    rooms = list(load_layout()["spawn_regions"])
    assert sorted(tab["room"]) == sorted(rooms) and max(tab["room"].values()) - min(tab["room"].values()) <= 1
    assert n - tab["support"]["floor"] >= m
    # surface uniform as far as the type rule allows (push_or_detour: follows the room; blocked: kitchen / living-room
    # boxes only to the storage bench): free slots fill the surface below quota
    layout = load_layout()
    places = layout["place_surfaces"]
    forced = Counter(o[0] for sl in p if len(o := [x for x in places if slot_ok(layout, ttype, sl["room"], x)]) == 1)
    free = n - sum(forced.values())
    diff = abs(tab["surface"].get(places[0], 0) - tab["surface"].get(places[1], 0))
    assert diff <= max(1, abs(forced[places[0]] - forced[places[1]]) - free)


def test_out_of_sight_is_10px() -> None:
    assert PX_OUT_OF_SIGHT == 10
    for s in range(1000, 1010):
        try:
            t = T.make_task(s, "fetch")
        except T.NoTask:
            continue
        assert all(v < PX_OUT_OF_SIGHT for v in t["hall_px"].values())


def test_pregrasp_filter_4016() -> None:
    """004 tuning fetch-4016: green box, its only pre-grasp side blocked by yellow's avoid disc -> never a target."""
    assert "green box" not in T.pregrasp_ok(4016, "none")
    assert all(c[0][0] != "green box" for c in T.candidates(4016, "fetch"))


def test_want_matches_slot() -> None:
    ts = draw_set("fetch", 4, 1400, stop=1500, min_surface=1)
    for t in ts:
        room, sup = T.start_of(t["seed"], "none", t["targets"][0]["object"])
        assert {"room": room, "support": sup, "surface": t["targets"][0]["surface"]} == t["slot"]
    assert len({t["seed"] for t in ts}) == 4


def test_seal_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "N_PER_TYPE", 4)  # one slot per room
    monkeypatch.setattr(seal, "MIN_SURFACE_SEALED", 1)
    out = tmp_path / "dry.jsonl"
    meta = seal.seal(out, dry_run=True)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert meta["counts"] == dict.fromkeys(T.TYPES, 4) and meta["dry_run"] and meta["seed_space"] == [1400, 2000]
    assert meta["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    seeds = [x["seed"] for x in lines]
    assert len(set(seeds)) == 16 and all(s in range(1400, 2000) for s in seeds)
    with pytest.raises(SystemExit, match="exists"):
        seal.seal(out, dry_run=True)


def test_seal_refuses_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "git", lambda *a: " M sim/tasks.py")
    monkeypatch.setattr(seal, "draw_set", lambda *a, **k: pytest.fail("drew on a dirty tree"))
    with pytest.raises(SystemExit, match="dirty"):
        seal.seal(tmp_path / "x.jsonl", dry_run=False)


def test_seal_refuses_non_nvidia_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """005 review S1: llvmpipe px differ from NVIDIA; the sealed draw refuses before drawing."""
    monkeypatch.setattr(seal, "git", lambda *a: "")
    monkeypatch.setattr(seal, "gl_renderer", lambda: "llvmpipe (LLVM 17.0.6, 256 bits)")
    monkeypatch.setattr(seal, "draw_set", lambda *a, **k: pytest.fail("drew with llvmpipe"))
    with pytest.raises(SystemExit, match="NVIDIA"):
        seal.seal(tmp_path / "x.jsonl", dry_run=False)


def test_seal_refuses_untagged_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """005 review S2: the frozen generator commit carries a local seal-* tag."""
    monkeypatch.setattr(seal, "git", lambda *a: "")
    monkeypatch.setattr(seal, "gl_renderer", lambda: "NVIDIA H200/PCIe/SSE2")
    monkeypatch.setattr(seal, "draw_set", lambda *a, **k: pytest.fail("drew untagged"))
    with pytest.raises(SystemExit, match="tag"):
        seal.seal(tmp_path / "x.jsonl", dry_run=False)


def test_seal_meta_records_renderer_and_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "N_PER_TYPE", 4)
    monkeypatch.setattr(seal, "MIN_SURFACE_SEALED", 1)
    meta = seal.seal(tmp_path / "dry.jsonl", dry_run=True)
    assert "NVIDIA" in meta["gl_renderer"] and meta["mujoco"]
    assert set(meta["dep_sha256"]) == set(seal.DEP_FILES) and all(len(h) == 64 for h in meta["dep_sha256"].values())
