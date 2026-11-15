"""G3-leak scanner (task 004): every text the VLM receives must carry no coordinates, poses, map, joint values,
units or robot identity, and no number except the allowlisted counters. Stdlib only.

    scan(text) -> [(class, offending substring), ...]
    scan_run(run_dir) -> hits over every packet in <run_dir>/log.jsonl (system + user + photo labels per step) and
                         the tool schemas sent (<run_dir>/tools.json, once per run)
    python -m agent.leak <run_dir>...   prints the hit count (and each hit); exit 1 on any hit

Allowlist (removed before the digit / number checks): "Step <k> of <N>", "Photo <k>", a history line number
("<n>. " at a line start), "<k> photos" (the look result). Classes, checked on the text as sent:
  brand     robot / body / sim names (tidybot, kinova, gen3, robotiq, 2f-85, mujoco, menagerie, stanford)
  word      pose / coordinate / map / joint / angle / unit words
  coord     "(a, b)" number pairs, "x=" style assignments, "[" + number
  decimal   a number with a decimal point
  digit     any other digit
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BRANDS = ("tidybot", "kinova", "gen3", "robotiq", "2f-85", "mujoco", "menagerie", "stanford")
WORDS = (r"poses?", r"coordinates?", r"maps?", r"occupancy", r"joints?", r"angles?", r"radians?", r"rad", r"degrees?",
         r"deg", r"yaw", r"quaternions?", r"metres?", r"meters?", r"centimet(?:re|er)s?", r"millimet(?:re|er)s?",
         r"cm", r"mm", r"km", r"kg", r"m/s")
ALLOW = (r"Step \d+ of \d+", r"Photo \d+", r"(?m)^\d+\. ", r"\b\d+ photos\b")
CHECKS = (
    ("brand", re.compile("|".join(re.escape(b) for b in BRANDS), re.I)),
    ("word", re.compile(r"\b(?:" + "|".join(WORDS) + r")(?![\w/])", re.I)),
    ("coord", re.compile(r"\(\s*-?\d[\d.]*\s*,\s*-?\d[\d.]*|\b[xyz]\s*[=:]|\[\s*-?\d")),
)
NUMBERS = (("decimal", re.compile(r"\d+\.\d+")), ("digit", re.compile(r"\d+")))


def scan(text: str) -> list[tuple[str, str]]:
    hits = [(c, m.group()) for c, rx in CHECKS for m in rx.finditer(text)]
    rest = text
    for a in ALLOW:
        rest = re.sub(a, " ", rest)
    for c, rx in NUMBERS:
        hits += [(c, m.group()) for m in rx.finditer(rest)]
        rest = rx.sub(" ", rest)
    return hits


def run_texts(run_dir: str | Path) -> list[tuple[str, str]]:
    """(where, text) for every text the run sent to the VLM."""
    d = Path(run_dir)
    out = [("tools.json", (d / "tools.json").read_text())]
    for line in (d / "log.jsonl").read_text().splitlines():
        s = json.loads(line)
        if "extra" in s and "packet" in s["extra"]:
            p = s["extra"]["packet"]
            out += [(f"step {s['step']} {k}", p[k]) for k in ("system", "user")]
            out += [(f"step {s['step']} label", t) for t in p["labels"]]
    return out


def scan_run(run_dir: str | Path) -> list[tuple[str, str, str]]:
    return [(w, c, h) for w, t in run_texts(run_dir) for c, h in scan(t)]


def main(argv: list[str]) -> int:
    total = 0
    for d in argv:
        hits = scan_run(d)
        total += len(hits)
        print(f"{d}: {len(hits)} hits, {len(run_texts(d))} texts")
        for h in hits:
            print("   ", *h)
    print(f"total hits: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
