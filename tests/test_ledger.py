"""Ledger tests: exactly-once under a kill mid-item, a requeue (new restart count) and two racing workers."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc"))
import ledger  # noqa: E402


def work(item, hb):
    time.sleep(0.05)
    return {"steps": 1}


def worker(d, job, restart, slow=0.0):
    os.environ["SLURM_JOB_ID"], os.environ["SLURM_RESTART_COUNT"] = job, str(restart)
    ledger.run(d, (lambda i, hb: (time.sleep(slow), {"steps": 1})[1]) if slow else work)


def test_race_two_workers(tmp_path):
    d = str(tmp_path)
    ledger.init(d, 20)
    ps = [mp.Process(target=worker, args=(d, j, 0)) for j in ("1", "2")]
    [p.start() for p in ps]
    [p.join(30) for p in ps]
    assert ledger.check(d)
    assert len(os.listdir(tmp_path / "done")) == 20


def test_kill_mid_item_then_requeue(tmp_path):
    d = str(tmp_path)
    ledger.init(d, 6)
    p = mp.Process(target=worker, args=(d, "7", 0, 30.0))
    p.start()
    time.sleep(1.0)
    p.kill()
    p.join()
    assert len(ledger.remaining(d)) == 6 and len(os.listdir(tmp_path / "claims")) == 1
    worker(d, "7", 1)  # same job id, restart 1: stale claim taken over immediately
    assert ledger.check(d, min_attempts=2)
    ev = [json.loads(x)["event"] for x in open(tmp_path / "attempts.log")]
    assert ev.count("item_takeover") == 1


def test_stop_file_then_resume(tmp_path):
    d = str(tmp_path)
    ledger.init(d, 4)
    os.environ["SLURM_JOB_ID"], os.environ["SLURM_RESTART_COUNT"] = "9", "0"
    (tmp_path / "stop" / "9_0").touch()
    assert ledger.run(d, work) == 75 and len(ledger.remaining(d)) == 4
    worker(d, "9", 1)
    assert ledger.check(d, min_attempts=2)


def test_done_never_overwritten(tmp_path):
    d = str(tmp_path)
    ledger.init(d, 1)
    worker(d, "1", 0)
    before = (tmp_path / "done" / "item000.json").read_text()
    # a second attempt that somehow ran the item anyway cannot replace the record
    tmp = tmp_path / "done" / "x.tmp"
    tmp.write_text("{}")
    try:
        os.link(tmp, tmp_path / "done" / "item000.json")
        raise AssertionError("overwrote")
    except FileExistsError:
        pass
    assert (tmp_path / "done" / "item000.json").read_text() == before
