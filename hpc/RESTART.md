# Restart pattern for PERUN jobs

PERUN has `PreemptMode=REQUEUE`, `JobRequeue=1`; gpu_short is capped at 2 days. A job may be requeued, preempted or hit
its limit, so every job's work is a list of items and progress lives in a ledger dir on the shared project filesystem, not in the job.

## Files
- `hpc/ledger.py` (stdlib, Py 3.9-clean; runs on the host python3 and inside `g1.sif`): `init`, `run`, `check`.
- `hpc/ledger_demo.py`: the test work, one TidyBot MuJoCo episode (30 s wall, no render) per item.
- `hpc/restart_check.sbatch <ledger> [requeue|successor] [n_items]`: the template batch script.
- `tests/test_ledger.py`: kill mid-item, requeue, two racing workers, stop file (local, pytest).

## How every later job must use the ledger
Make the job's work a list of independent item ids and call `ledger.init(dir, n)` (idempotent) then `ledger.run(dir, work)`,
where `work(item, heartbeat) -> result dict` does one item and calls `heartbeat()` at least every few seconds. `run` claims
an item (`claims/<id>`, O_EXCL; heartbeat = mtime; a claim is dead when its heartbeat is older than 45 s or its job id is
the same with another `SLURM_RESTART_COUNT`), and writes the result to `done/<id>.json` with write-tmp, fsync, `link(2)`
(never overwrites, so one done record per item). Nothing partial is ever visible: a killed item is simply redone from
scratch. Every attempt is appended to `attempts.log` (attempt_start, item_start, item_done, item_takeover, item_dup,
attempt_end), and `python3 hpc/ledger.py check <dir> [--min-attempts K]` prints per-segment items and a PASS/FAIL verdict.
In the batch script: `#SBATCH --requeue --open-mode=append --signal=B:USR1@90`, ledger dir as a **positional** arg (never
`--export`), pass `SLURM_JOB_ID` and `SLURM_RESTART_COUNT` into the container with `--env` (`--cleanenv` strips them),
run the worker in the background, and have `trap USR1` touch `<ledger>/stop/<job>_<restart>` (the worker finishes its
current item, exits 75), then `scontrol requeue $SLURM_JOB_ID` if the worker returned 75. Items must be much shorter than
the signal lead time (here 30 s vs 90 s). Keep the ledger in a project dir under `runs/`, and judge a job by `check`, not by the log.

## What was measured on PERUN (gpu_short, 1 GPU, 4 cpus, 16G, 10-min wall)
- `scontrol requeue` from inside the job at the USR1 signal works: same job id, `SLURM_RESTART_COUNT` +1, no human.
  The self-submitted successor (`sbatch --dependency=afterany`) is coded in the script but was not needed or tested.
- Gap from cut to restart: the requeued job sits PENDING with reason `BeginTime` for about 2 min (squeue shows
  BeginTime = requeue time + ~2 min), then starts on any node: 148 s (158106 r0 -> r1), 144 s (158107 r0 -> r1),
  123 s (158107 r1 -> r2, measured from worker end to attempt_start).
- G0c-limit (158106): cut at 8.5 min (USR1), 17 items done, requeued, resumed on another node, 13 items; PASS.
- G0c-requeue (158107): `scontrol requeue` at 00:13:40 with item004 in progress; item004 was taken over and redone;
  it was later cut again by the 10-min limit signal and resumed once more (3 segments); PASS.
- Without `--open-mode=append` the job log is truncated at every restart (found in the test; the ledger is the record).
- MuJoCo, TidyBot `scene.xml`, no render, one core: about 50,000 steps/s (min 34k, max 59k over 60 episodes; dt of the scene).
- A first round was thrown away: `--cleanenv` stripped `SLURM_JOB_ID`, so attempts were logged as `local`.
