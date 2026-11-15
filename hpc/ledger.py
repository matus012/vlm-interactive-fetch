"""Work-item ledger for PERUN jobs that are requeued, preempted or hit their time limit. Stdlib only, Py 3.9-clean.

Layout of a ledger dir (shared storage):
  items/<id>            one file per item, created at init
  claims/<id>           owner of the item right now (json; mtime = heartbeat)
  done/<id>.json        the result; exists <=> the item is done. Created with link(2): atomic and never overwrites.
  stop/<job>_<restart>  tells that attempt to finish its current item and exit
  attempts.log          append-only json lines: attempt_start, item_start, item_done, item_dup, item_takeover, attempt_end

CLI: init <dir> <n_items> | remaining <dir> | check <dir> [--min-attempts K]
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

STALE_S = 45.0  # a claim whose heartbeat is older than this belongs to a dead attempt
HEARTBEAT_S = 5.0


def attempt_id() -> Dict[str, object]:
    return {"job": os.environ.get("SLURM_JOB_ID", "local"),
            "restart": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "node": os.uname().nodename, "pid": os.getpid()}


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def log(ledger: str, event: str, **kw: object) -> None:
    line = json.dumps(dict(t=time.time(), event=event, **kw), sort_keys=True) + "\n"
    fd = os.open(os.path.join(ledger, "attempts.log"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())  # one write() per line: O_APPEND keeps concurrent writers' lines whole
        os.fsync(fd)
    finally:
        os.close(fd)


def init(ledger: str, n_items: int) -> None:
    for d in ("items", "claims", "done", "stop"):
        os.makedirs(os.path.join(ledger, d), exist_ok=True)
    for i in range(n_items):
        p = os.path.join(ledger, "items", "item%03d" % i)
        try:
            os.close(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        except FileExistsError:
            pass


def items(ledger: str) -> List[str]:
    return sorted(os.listdir(os.path.join(ledger, "items")))


def is_done(ledger: str, item: str) -> bool:
    return os.path.exists(os.path.join(ledger, "done", item + ".json"))


def remaining(ledger: str) -> List[str]:
    return [i for i in items(ledger) if not is_done(ledger, i)]


def _claim_is_stale(claim_path: str, me: Dict[str, object]) -> bool:
    try:
        with open(claim_path) as f:
            owner = json.load(f)
        age = time.time() - os.stat(claim_path).st_mtime
    except (OSError, ValueError):
        return True  # vanished or half-written and abandoned
    if owner.get("job") == me["job"] and owner.get("restart") != me["restart"]:
        return True  # same job id, other restart count: that attempt was requeued, so it is dead
    return age > STALE_S


def try_claim(ledger: str, item: str, me: Dict[str, object]) -> bool:
    path = os.path.join(ledger, "claims", item)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if not _claim_is_stale(path, me):
                return False
            grave = "%s.dead.%d.%s" % (path, os.getpid(), me["job"])
            try:
                os.rename(path, grave)  # only one racer's rename succeeds
            except OSError:
                return False
            log(ledger, "item_takeover", item=item, **me)
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(me, f)
        return True
    return False


def run(ledger: str, work: Callable[[str, Callable[[], None]], Dict[str, object]]) -> int:
    """Claim and do items until none is left. work(item, heartbeat) returns the result dict.
    Returns 0 if every item is done, 75 if stopped early (stop file) with items left."""
    me = attempt_id()
    stop = os.path.join(ledger, "stop", "%s_%s" % (me["job"], me["restart"]))
    log(ledger, "attempt_start", remaining=len(remaining(ledger)), **me)
    rc = 0
    while True:
        todo = remaining(ledger)
        if not todo:
            break
        if os.path.exists(stop):
            rc = 75
            break
        got = next((i for i in todo if try_claim(ledger, i, me)), None)
        if got is None:
            time.sleep(5)  # everything left is claimed by someone alive; wait for them or for their claim to go stale
            continue
        claim = os.path.join(ledger, "claims", got)
        last = [time.time()]

        def heartbeat() -> None:
            if time.time() - last[0] >= HEARTBEAT_S:
                os.utime(claim, None)
                last[0] = time.time()

        log(ledger, "item_start", item=got, **me)
        t0 = time.time()
        result = work(got, heartbeat)
        result.update(item=got, job=me["job"], restart=me["restart"], node=me["node"], wall_s=round(time.time() - t0, 2))
        tmp = os.path.join(ledger, "done", ".%s.%s.%s.tmp" % (got, me["job"], me["pid"]))
        with open(tmp, "w") as f:
            json.dump(result, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, os.path.join(ledger, "done", got + ".json"))  # fails if a done record exists: never two
            _fsync_dir(os.path.join(ledger, "done"))
            log(ledger, "item_done", item=got, **me)
        except FileExistsError:
            log(ledger, "item_dup", item=got, **me)  # lost a race: our result is discarded
        os.unlink(tmp)
        try:
            os.unlink(claim)
        except OSError:
            pass
    log(ledger, "attempt_end", rc=rc, remaining=len(remaining(ledger)), **me)
    return rc


def check(ledger: str, min_attempts: int = 1) -> bool:
    ok = True
    its = items(ledger)
    done_dir = os.path.join(ledger, "done")
    done = sorted(f for f in os.listdir(done_dir) if f.endswith(".json"))
    stray = [f for f in os.listdir(done_dir) if f.endswith(".tmp")]
    events = [json.loads(l) for l in open(os.path.join(ledger, "attempts.log")) if l.strip()]
    finished: Dict[str, int] = {}
    seg_items: Dict[str, List[str]] = {}
    seg_t: Dict[str, List[float]] = {}
    for e in events:
        seg = "%s.r%s" % (e["job"], e["restart"])
        seg_t.setdefault(seg, []).append(e["t"])
        if e["event"] == "item_done":
            finished[e["item"]] = finished.get(e["item"], 0) + 1
            seg_items.setdefault(seg, []).append(e["item"])
    missing = [i for i in its if i + ".json" not in done]
    twice = [i for i, n in finished.items() if n != 1]
    unlogged = [i for i in its if finished.get(i, 0) == 0]
    mismatch = []
    for f in done:
        r = json.load(open(os.path.join(done_dir, f)))
        seg = "%s.r%s" % (r["job"], r["restart"])
        if r["item"] + ".json" != f or r["item"] not in seg_items.get(seg, []):
            mismatch.append(f)
    print("items=%d done_files=%d" % (len(its), len(done)))
    for seg in sorted(seg_t, key=lambda s: seg_t[s][0]):
        print("  segment %-12s %s -> %s  items_done=%d %s" % (
            seg, time.strftime("%H:%M:%S", time.localtime(seg_t[seg][0])),
            time.strftime("%H:%M:%S", time.localtime(seg_t[seg][-1])), len(seg_items.get(seg, [])),
            ",".join(i.replace("item", "") for i in seg_items.get(seg, []))))
    dups = [e for e in events if e["event"] == "item_dup"]
    takeovers = [e for e in events if e["event"] == "item_takeover"]
    print("takeovers (items started by a dead attempt, redone): %d %s" % (len(takeovers), sorted(set(e["item"] for e in takeovers))))
    print("lost-race duplicates discarded: %d" % len(dups))
    for name, bad in (("missing done file", missing), ("done logged !=1 time", twice), ("done file without log", unlogged),
                      ("done file not from its logging attempt", mismatch), ("stray tmp", stray)):
        if bad:
            ok = False
            print("FAIL %s: %s" % (name, bad))
    if len(seg_t) < min_attempts:
        ok = False
        print("FAIL only %d attempt segment(s), expected >= %d" % (len(seg_t), min_attempts))
    print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        init(sys.argv[2], int(sys.argv[3]))
    elif cmd == "remaining":
        print(len(remaining(sys.argv[2])))
    elif cmd == "check":
        k = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--min-attempts" else 1
        sys.exit(0 if check(sys.argv[2], k) else 1)
    else:
        sys.exit("usage: ledger.py init <dir> <n> | remaining <dir> | check <dir> [--min-attempts K]")
