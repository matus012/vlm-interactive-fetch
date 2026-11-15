"""Deviation 9 feasibility (task 003 fix round 3): floor spawn spots per room before / after the route-corridor rule
on the 0.25 m (G1-cover) and 0.05 m grids, surface spots inside a corridor, and box draws over seeds 0-999 with the
rule off ("before") and on ("after"): rejected samples per seed and seeds that cannot place their boxes.

    python -m sim.world.route_feasibility runs/gate2r3/feasibility.json
"""
import collections
import json
import sys
import time

import sim.world.generate as G
from sim.constants import G1_COVER_GRID
L = G.load_layout(); P = G.load_places()
t = time.time(); segs = G.route_segments(L); print("routes", len(segs[0]), "segments", f"{time.time()-t:.1f}s")
out = {"spots": {}, "surface_conflicts": {}, "draws": {}}
for step in (G1_COVER_GRID, 0.05):
    for room in L["spawn_regions"]:
        reg = L["spawn_regions"][room]
        before = after = 0
        n = lambda a, b: [a + i * step for i in range(int((b - a) / step + 1e-9) + 1)]
        for y in n(reg[1], reg[3]):
            for x in n(reg[0], reg[2]):
                if G.spot_ok(L, room, x, y, P):
                    before += 1; after += G.route_clear(segs, x, y)
        sp = G.spawn_spots(L, room, step)
        surf = [s for s in sp if s[3] != "floor"]
        bad = [s for s in surf if not G.route_clear(segs, s[0], s[1])]
        out["spots"][f"{room}@{step}"] = {"floor_before": before, "floor_after": after, "spawn_spots_floor": sum(s[3]=="floor" for s in sp),
                                          "surface": len(surf), "surface_in_corridor": len(bad)}
print(json.dumps(out["spots"], indent=1))
# draws
cnt = collections.Counter()
orig_spot, orig_surf, orig_rc = G.spot_ok, G.surface_spot_ok, G.route_clear
def so(*a, **k): cnt["att"] += 1; return orig_spot(*a, **k)
def ss(*a, **k): cnt["att"] += 1; return orig_surf(*a, **k)
G.spot_ok, G.surface_spot_ok = so, ss
for mode in ("before", "after"):
    G.route_clear = orig_rc if mode == "after" else (lambda s, x, y: True)
    per, fails, rooms = [], [], collections.Counter()
    for s in range(1000):
        cnt.clear()
        try:
            bx = G._draw_boxes(L, s, P)
            per.append(cnt["att"] - 4); rooms.update(b["room"] + "/" + b["surface"] for b in bx)
        except RuntimeError as e:
            fails.append((s, str(e)))
    per_s = sorted(per)
    out["draws"][mode] = {"seeds": 1000, "failures": fails, "retries_total": sum(per), "retries_mean": sum(per)/len(per),
                          "retries_median": per_s[len(per_s)//2], "retries_max": max(per), "boxes_by_room": dict(sorted(rooms.items()))}
print(json.dumps(out["draws"], indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
