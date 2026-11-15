"""Task 007 Procedure 1-2 (H2 freeze check + seal dependency check), run locally in the checkout of the run commit.

    python hpc/freeze_check.py <run commit> --reviewed <harness commit the review read>
        -> runs/007/freeze_check.json, runs/007/truthfree_v.txt

- interface sha256 at the run commit (rule of reports/2026-11-13-006-interface-freeze.json) = the freeze-006 value;
- `git diff --name-only freeze-006 <run> -- sim agent hpc tasksets`, each file classified allowed (H1 runner, sim/clip_*,
  tests/, hpc/*.md, the task-007 harness files) or not allowed (`*` never crosses `/`);
- reviewed bytes (R1): `git diff --quiet <reviewed> <run> -- sim agent hpc tasksets` minus hpc/*.md and sim/clip_*
  must pass, so the allowed files are the bytes the review read, not just their names;
- clean tree, HEAD = the run commit;
- the 17 dependency sha256 in the seal's meta (sealed/test_100.meta.json: no task content) vs the run commit: which
  differ (any -> the job regenerates from seal-005, row 21);
- `pytest -v tests/test_truthfree.py` at the run commit, named output saved.
Task 011 (E2): `--task 011 --seal-sha <E2 sha> --seal-commit <seal-011>` = ALLOWED_011 (house B world files, layout
plumbing, gate harness, e2.sbatch), the seal-011 meta (sealed/e2_100.meta.json; seal deps must equal the run commit's,
house B's files included), and house B's world files byte-equal to the tag freeze-011-world. Run with NAV116_HOUSE=B.
Task 014 (E4): `--task 014 --seal-sha <E4 sha> --seal-commit <seal-014>` = ALLOWED_014, sealed/e4_100.meta.json
(deps incl. sim/e4.py must equal the run commit's), house B world files = freeze-011-world.
Task 010 (E3): `--task 010` = the seal-005 checks with ALLOWED_010 (+ agent/e3.py, hpc/e3.sbatch), output runs/010/.
Task 009 (E1, row 22): `--task 009 --seal-sha <E1 seal sha256> --seal-commit <seal-009 sha>` adds the E1 switch files
to the allowed list (ALLOWED_009), checks the E1 seal's meta (sealed/e1_100.meta.json) and writes to runs/009/.
Task 011 note: run it with NAV116_HOUSE unset (house A): test_truthfree pins house A fixtures; the 011 checks read git objects.
Exit 1 if the interface hash differs, a not-allowed file differs, the reviewed bytes differ, the tree is dirty / not
at the run commit, the seal meta's sha256 or commit is not the seal's, or the truthfree tests fail. Never reads the sealed task file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

FREEZE_TAG = "freeze-006"
INTERFACE_FILES = ("agent/tools.py", "agent/robot_api.py", "agent/skills.py", "agent/validate.py", "agent/context.py")
INTERFACE_SHA256 = "806c70f4452a10ce6f1f8b2908aabf951cc5ec39c9be319c338208a1989c329e"
SEALED_SHA256 = "8f691844eb8312f05c1a40f386b73faf515be656948b42fe76a510bceede542d"
SEAL_COMMIT = "ba7bfff3d9060f3c050d0fb41966d9b20522fbf6"
ALLOWED = {"sim/agent_run.py": "H1 runner change (row 20, reviewed diff)", "sim/clip_*": "clip script",
           "tests/*": "tests", "hpc/*.md": "ledger / docs",
           **{f"hpc/{f}": "task 007 harness" for f in ("seal_regen.py", "phase6.sbatch", "phase6_batch.sh",
                                                        "freeze_check.py")}}
ALLOWED_009 = {**ALLOWED, "sim/agent_run.py": "H1 runner change (row 20) + task 009 --e1", "hpc/e1.sbatch": "task 009 harness",
               "sim/world/generate.py": "task 009 E1 switch (EASINGS, rows 8 / 9; default = freeze-006)",
               "sim/tasks.py": "task 009 E1 switch (row 15; easings key)", "sim/e1.py": "task 009 E1 helpers",
               "sim/e1_oracle.py": "task 009 oracle split (after the counted pass)",
               "hpc/e1_oracle_batch.sh": "task 009 harness",
               "sim/seal.py": "task 009 E1 seal mode (--e1)",
               "agent/context_e1.py": "task 009 rules text without line E"}
ALLOWED_010 = {**ALLOWED_009, "sim/agent_run.py": "H1 runner change (row 20) + task 009 --e1 + task 010 --script",
               "agent/e3.py": "task 010 fixed scripts (E3a / E3b)", "hpc/e3.sbatch": "task 010 harness",
               "sim/e1_analyse.py": "task 009 report numbers (imported by no runner)",
               "sim/e3_analyse.py": "task 010 report numbers (imported by no runner)"}
ALLOWED_011 = {**ALLOWED_010, "hpc/e2.sbatch": "task 011 harness", "hpc/e2_smoke.sbatch": "task 011 harness",
               "tasksets/e2_smoke.json": "task 011 step 2 VLM smoke taskset (3 house B dev tasks)",
               "sim/world/generate.py": "task 009 E1 switch + task 011 NAV116_HOUSE switch (house A default)",
               "sim/tasks.py": "task 009 row 15 / easings key + task 011 layout-driven detour, house key, row 23",
               "sim/seal.py": "task 009 --e1 + task 011 house B seal mode",
               "hpc/phase3_batch.sh": "task 011: NAV116_HOUSE pass-through (default A)",
               "sim/e2_analyse.py": "task 011 report numbers (imported by no runner)",
               "sim/world/layout_b.yaml": "task 011 house B world (frozen at freeze-011-world)",
               "sim/world/places_b.yaml": "task 011 house B world (frozen at freeze-011-world)",
               "sim/gate2.py": "task 011 gate harness, layout-driven (house A trial inputs unchanged)",
               "sim/push_check.py": "task 011 gate harness, layout-driven", "sim/cover_check.py": "task 011 gate harness",
               "sim/reach_check.py": "task 011 gate harness",
               "sim/sampler.py": "task 011 layout plumbing (target_spawn_surfaces; house A default)"}
ALLOWED_014 = {**ALLOWED_011,
               "sim/agent_run.py": "row 20 runner + 009 --e1 + 010 --script + 014 (labels switch, C3 move hook, e3c)",
               "sim/world/generate.py": "009 E1 switch + 011 house switch + 014 LABELS switch (default off = unchanged)",
               "sim/e4.py": "task 014 E4 task generator + C3 harness move", "sim/e4_seal.py": "task 014 seal / regen",
               "sim/e4_gates.py": "task 014 world gates (dev)", "sim/e4_c1_gate.py": "task 014 C1 build gate (dev)",
               "sim/e4_analyse.py": "task 014 report numbers (imported by no runner)",
               "agent/e3c.py": "task 014 baseline script E3c (E3b + 2 rules)", "hpc/e4.sbatch": "task 014 harness",
               "sim/video_run.py": "task 013 video runs (not imported by any runner)",
               "sim/render_video.py": "task 013 video renderer (not imported by any runner)",
               "sim/plan_png.py": "task 011 house plan image (reports / clips; imported by no runner)"}
WORLD_B = ("sim/world/layout_b.yaml", "sim/world/places_b.yaml")
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def allowed(f: str, table: dict[str, str] = ALLOWED) -> str | None:
    """Why f may differ from freeze-006 (ALLOWED pattern, `*` = any run of non-`/` chars), or None."""
    return next((w for p, w in table.items() if re.fullmatch(re.escape(p).replace(r"\*", "[^/]*"), f)), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reviewed", required=True, help="the commit whose harness bytes the review read")
    ap.add_argument("--task", choices=("007", "009", "010", "011", "014"), default="007")  # 010: seal-005, E3 files allowed
    ap.add_argument("--seal-sha", default=SEALED_SHA256)
    ap.add_argument("--seal-commit", default=SEAL_COMMIT)
    a = ap.parse_args()
    e1 = a.task == "009"
    e2 = a.task == "011"
    e4 = a.task == "014"
    run = git("rev-parse", "--verify", a.run + "^{commit}").decode().strip()
    reviewed = git("rev-parse", "--verify", a.reviewed + "^{commit}").decode().strip()
    spec = ["--", "sim", "agent", "hpc", "tasksets", ":(exclude,glob)hpc/*.md", ":(exclude,glob)sim/clip_*"]
    rev_clean = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", reviewed, run, *spec]).returncode == 0
    rev_diff = git("diff", "--name-only", reviewed, run, *spec).decode().split()
    h = hashlib.sha256()
    for f in INTERFACE_FILES:
        h.update(f.encode() + b"\0" + git("show", f"{run}:{f}"))
    diff = git("diff", "--name-only", FREEZE_TAG, run, "--", "sim", "agent", "hpc", "tasksets").decode().split()
    cls = {f: allowed(f, {"007": ALLOWED, "009": ALLOWED_009, "010": ALLOWED_010, "011": ALLOWED_011, "014": ALLOWED_014}[a.task])
           for f in diff}
    meta_path = (Path(os.environ.get("NAV116_SEALED_DIR", "tasksets/sealed"))
                 / ("e1_100.meta.json" if e1 else "e2_100.meta.json" if e2 else "e4_100.meta.json" if e4
                    else "test_100.meta.json"))
    meta = json.loads(meta_path.read_text())
    dep = {f: hashlib.sha256(git("show", f"{run}:{f}")).hexdigest() for f in meta["dep_sha256"]}
    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain").decode()
    out = REPO / "runs" / a.task
    out.mkdir(parents=True, exist_ok=True)
    pt = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests/test_truthfree.py"],
                        cwd=REPO, capture_output=True, text=True)
    (out / "truthfree_v.txt").write_text(pt.stdout + pt.stderr)
    extra_rc = 0
    if e4:  # review 014 S6: E3c's truth-free AST test, the E4 generator / scorer tests, the labels switch
        pe = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests/test_e3c.py",
                             "tests/test_e4.py", "tests/test_e4_labels.py", "tests/test_e4_move.py"],
                            cwd=REPO, capture_output=True, text=True)
        (out / "e4_tests_v.txt").write_text(pe.stdout + pe.stderr)
        extra_rc = pe.returncode
    rec = {
        "run_commit": run, "freeze_commit": git("rev-parse", FREEZE_TAG + "^{commit}").decode().strip(),
        "interface_sha256": h.hexdigest(), "interface_ok": h.hexdigest() == INTERFACE_SHA256,
        "diff_vs_freeze": {"allowed": {f: w for f, w in cls.items() if w},
                           "not_allowed": [f for f, w in cls.items() if not w]},
        "agent_run_sha256": hashlib.sha256(git("show", f"{run}:sim/agent_run.py")).hexdigest(),
        "reviewed": {"commit": reviewed, "bytes_equal": rev_clean, "differ": rev_diff},
        "head": head, "head_is_run_commit": head == run, "clean_tree": status == "", "dirty": status.splitlines(),
        "seal_meta": {"file": str(meta_path), "sha256_ok": meta["sha256"] == a.seal_sha,
                      "commit_ok": meta["commit"] == a.seal_commit, "n_dep": len(dep),
                      "dep_differ": sorted(f for f in dep if dep[f] != meta["dep_sha256"][f]),
                      "regenerate_from_seal_tag": any(dep[f] != meta["dep_sha256"][f] for f in dep)},
        "e4_tests": {"rc": extra_rc} if e4 else None,
        "locate_mode": "perceived (sim.agent_run --poses default, hpc/phase6_batch.sh passes perceived)",
        "truthfree": {"rc": pt.returncode, "summary": (pt.stdout.strip().splitlines() or [""])[-1],
                      "file": str(out / "truthfree_v.txt")},
    }
    if e2 or e4:  # task 011 (014 too): house B's world files at the run commit = the freeze-011-world tag's bytes
        rec["world_b_frozen"] = {f: git("show", f"{run}:{f}") == git("show", f"freeze-011-world:{f}") for f in WORLD_B}
    rec["ok"] = (extra_rc == 0 and rec["interface_ok"] and not rec["diff_vs_freeze"]["not_allowed"] and rec["clean_tree"]
                 and rec["head_is_run_commit"] and pt.returncode == 0 and rec["seal_meta"]["sha256_ok"]
                 and rec["seal_meta"]["commit_ok"] and rev_clean
                 and not ((e1 or e2 or e4) and rec["seal_meta"]["dep_differ"])  # 009 / 011 / 014: runner's worlds = sealed
                 and all(rec.get("world_b_frozen", {}).values()))
    (out / "freeze_check.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
