# Deviations

The numbered changes to the original design, task rules, skills or measurement made during the project, in the order
they were fixed (world-only fixes to house B during its world checks are listed at the end). Each entry gives the date, what changed, why, and which result it was fixed before. No threshold was moved after
a result to make a check pass. Phase order: Phase 0 (components alone), Phase 1 (world), Phase 2 (scripted skills),
Phase 3 (VLM fetch), Phase 4 (all task types; the base sealed set drawn at its end), Phase 5 (depth-camera poses;
skill interface frozen as freeze-006), Phase 6 (base sealed test); then the extensions E1, E3, E2 and E4.

## 1. Base force limit and speed cap moved into Phase 1 (2026-11-13)
- **Change:** the base `forcerange` and speed cap move from Phase 2 into Phase 1; the blocked detector stays in
  Phase 2.
- **Why:** the Phase 0 base had no force limit (1.5 MN into a cabinet), so a Phase 1 reach check on it would pass
  trivially and have to be re-measured once Phase 2 adds the limit.
- **Fixed before:** the Phase 1 world checks.

## 2. Second kitchen room point; spawn keep-out per room point (2026-11-13)
- **Change:** a second kitchen room point `kitchen east` (8.5, 2.0) in `places.yaml`; the spawn keep-out (≥ 0.80 m)
  now applies to every room point of a room (kitchen floor spots 68 → 53 at the 0.25 m grid; the 7 failing spots are
  kept).
- **Why:** G1-cover round 1 gave kitchen 265/272: the kitchen table top hides the floor behind it from (6.75, 1.75),
  and no single kitchen point covers all 272 (7 candidates tried). The second point was the pre-planned fallback. The
  keep-out ensures boxes do not spawn under a room point.
- **Fixed before:** the Phase 1 G1-cover re-run (Phase 1 passed after 2 cover rounds).

## 3. `via` takes a room or a named door (2026-11-13)
- **Change:** `navigate(place, via)`: `via` takes a room or a named door; doors are named in `places.yaml`.
- **Why:** the two kitchen–living doors join the same rooms, so a room-level `via` cannot express the detour.
- **Fixed before:** the Phase 2 skill checks.

## 4. Spawn keep-outs and seen-box avoidance (2026-11-13)
- **Change:** spawn keep-outs at approach points and doorway corridors (± 0.6 m); `navigate` avoids boxes the runtime
  has seen (truth poses in Phases 2–4, perceived poses in Phase 5); unseen-box contacts are counted.
- **Why:** in Phase 1, boxes lay on routes in 44/100 seeds and under approach points in ~5 %.
- **Fixed before:** the Phase 2 skill checks.

## 5. Base planar force capped at 300 N (2026-11-13)
- **Change:** the base planar force norm is capped at 300 N.
- **Why:** the per-axis limit allowed 424 N on a diagonal (Phase 1 finding).
- **Fixed before:** the Phase 2 skill checks.

## 6. Blocked detector on tracking error (2026-11-13)
- **Change:** the blocked detector uses tracking error / no progress, not force saturation.
- **Why:** force saturates in normal crate pushing (Phase 1 finding).
- **Fixed before:** the Phase 2 skill checks.

## 7. Pick height read after the hold; carry-view miss; bumper (2026-11-13)
- **Change:** the G2-pick height requirement is read at the end of the 2 s hold (before carry), plus a "still held in
  carry" check; the carry pose may hide more of the mast view than the look pose for Phase 2 (logged as a miss against
  the Phase 2 specification); the blocked detector may use a base-contact bumper (30 N, 1 s mean).
- **Why:** carry holds surface-picked boxes below the surface top; tracking error alone missed the crate (shoved 2 m,
  "reached").
- **Fixed before:** the Phase 2 skill checks.

## 8. Spawn rules: surface edge band and floor distance (2026-11-13)
- **Change:** surface spawn spots only within 0.24 m of a surface edge with standable floor in front (first set at
  0.30 m, tightened to 0.24 m by a pick survey: pick works at 0.245 m, fails from 0.255 m); floor spots ≥ 0.60 m from
  spawn-surface footprints (was 0.30 m).
- **Why:** Phase 2 round 1: nightstand boxes in its back half were unreachable (bed + wall leave only the south side;
  16 failures), and one floor box occupied the only standing spot in front of the nightstand. Same rationale as the
  floor rule (the gripper must reach it), not a camera fit; G1-cover re-run as regression.
- **Fixed before:** the checks of Phase 2 fix round 2. Lifted in E1.

## 9. No fetch box on a route corridor (2026-11-13)
- **Change:** no fetch box within the robot's swept corridor (footprint + 5 cm) of any planned route between any two
  places (room points, approach points, doors).
- **Why:** the Phase 2 round-2 G2-blocked cabinet check failed on seed 1204: an unseen box on the
  living-room-to-kitchen-north-door route was pushed into the cabinet, the robot never touched the cabinet, and the
  check scored a fail. Also seed 1012 (target box tipped en route by navigate); in Phase 1, 251 / 2,100 routes over 100
  seeds passed over a floor box. The round-2 fail stands as recorded; the G2-blocked requirement and definition are
  unchanged. The alternative of counting contact through a pushed box as blocker contact was rejected, because it
  would redefine the check after the result.
- **Fixed before:** Phase 2 fix round 3. Lifted in E1.

## 10. Push: `stuck` vs `not movable` (2026-11-13)
- **Change:** push returns `failed: stuck` when the object moved then stalled; `failed: not movable` only when it
  never moved.
- **Why:** Phase 2 seed 1013: the crate moved 0.69 m, then push answered "not movable".
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set.

## 11. Push while carrying uses the carry footprint (2026-11-13)
- **Change:** push-while-carrying feasibility uses the carry footprint; a scripted check G4-carry-push is added.
- **Why:** in Phase 2, push succeeded 6/7 with a box in hand vs 19/19 empty.
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set.

## 12. Start / approach jitter on cabinet trials (2026-11-13)
- **Change:** start / approach jitter on every cabinet trial.
- **Why:** the 10 cabinet trials of Phase 2 were only 2 distinct ones.
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set.

## 13. Balanced task sampler; out-of-sight rule; nightstand-only surface targets (2026-11-13)
- **Change:** task sampler: target room uniform, support at natural share with a surface minimum, surface uniform;
  "out of sight from spawn" = < 10 px in every hall look photo; surface target slots = nightstand only. The coffee
  table is visible from the hall (≥ 52 px for all 130 coffee-table boxes, seeds 9000–9199), so coffee-table boxes stay
  in the house as non-target distractors; no exemption was made.
- **Why:** the Phase 3 counted run had 0 kitchen boxes and 1 surface pick; 4/30 targets were 69–83 px from the hall.
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set. The nightstand-only part is lifted
  in E1 (no effect in practice: the out-of-sight rule still rejects every coffee-table box).

## 14. Failure label rule (2026-11-13)
- **Change:** failure label = the part that returned a wrong result; an unrecovered correct failure result is `vlm`.
  Addendum (2026-11-13, 14:11 CET): `pushed` followed by `failed: blocked` on the same object in one run → label
  `skill`.
- **Why:** in Phase 3, correct "not in view" / "blocked" results were labelled perception / nav. The addendum follows
  Phase 4 counted run 9200 and tuning runs 8204/8206/8208.
- **Fixed before:** the Phase 4 counted run (rule); the addendum before the Phase 5 freeze and the Phase 6 sealed run.

## 15. Reject task draws without a feasible pre-grasp pose (2026-11-13)
- **Change:** task draws are rejected when the target box has no feasible pre-grasp pose.
- **Why:** Phase 3 tuning fetch-4016, 1 in 1,600 boxes.
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set. Lifted in E1.

## 16. pick / place return `failed: blocked` (2026-11-13)
- **Change:** when pick's or place's own approach drive stops on blocked detection with crate/cabinet contact, they
  return `failed: blocked` (a correct failure, `vlm` if unrecovered); every other drive failure stays
  `failed: unreachable` / `skill`.
- **Why:** Phase 4 tuning runs 8441, 8426 (place through the cabinet door) and 8218 (pick behind the unpushed crate)
  returned "unreachable" and were labelled `skill`.
- **Fixed before:** the Phase 4 counted run and the drawing of the base sealed set.

## 17. Push: when an object counts as `pushed`; planner map after a push (2026-11-13)
- **Change (first form):** push returns `pushed` only when the object's actual end pose is outside every door
  corridor and ≥ 0.8 m from every place pose, else it keeps pushing along the plan or returns `failed: stuck`; < 1 cm
  moved is never `pushed`; an object already blocking no door corridor / place pose → `failed: not blocking` (a new
  result string).
- **Amendment withdrawn (2026-11-13, 11:30 → 11:35 CET):** a route-corridor term (footprint within ROBOT_RADIUS +
  0.05 m of any cached place-to-place route) was added and then withdrawn on measurement: the crate in scripted run 1010
  lay ≥ 0.40 m off every cached route (limit 0.334 m); the hit was the carry path from the pick pose (1.71, 4.52), not
  a cached route, and the route term would have removed every living-room push end. The first form stands; run 1010
  is a known limit.
- **Addendum (d), 2026-11-13:** triggered by 3 tuning runs (8204, 8206, 8208) showing "not blocking" right after
  "blocked" on the same object. The adapter remembers the last drive (navigate, pick / place approach) that stopped on
  crate/cabinet contact and its planned path; push on that object treats it as blocking and moves it clear of that
  path (footprint > ROBOT_RADIUS + 0.05 m) while still meeting the first form, else `failed: stuck`. The memory clears
  after any successful drive or a push that moves the object; it is read only by push, never by navigate or the VLM
  packet (tested). When the robot stopped closer than the planner's clearance (tuning 8208), push first steps straight
  back if free, invisible to the VLM. Label fix (with deviation 10): no feasible plan, nothing pushed →
  `failed: stuck`; `not movable` only for an object pushed against that never moved.
- **Addendum 2 (2026-11-13, 14:11 CET):** `pushed` only if, with the object at its actual end pose added to the known
  map, every place pose in that room stays reachable from its door(s) and from each other (same planner and
  inflation); else keep pushing or `failed: stuck`. Cause: Phase 4 counted run 9200 and tuning 8204/8206/8208 (crate
  parked in the storage SW corner).
- **Addendum 3 (2026-11-13, 15:06 CET):** after `pushed`, or `stuck` that moved the object ≥ 3 cm, the adapter adds
  the object's end footprint to the planner map for every later drive in the run (the robot avoids what it knowingly
  moved, as in deviation 4); navigate never routes around a blocker it has not pushed; a cabinet or an unpushed crate
  stays off the map. In perceived mode the footprint comes from push's own geometry (base pose + contact side at the
  push end) or a crate-colour mask/depth fit plus a stated uncertainty margin, never simulator truth (covered by
  G5-truthfree with a truth-reading mutant); truth mode may use the truth pose. An object that cuts a room's places
  off counts as blocking in the `not blocking` test (9200, tuning 8204: the carry-out path ran into the parked crate).
  After a push that moved its object, the located pose is cleared, so pick returns `failed: not located` until the
  VLM locates again (a push can shove a located box; 8204/8208, 0.28 m).
- **Addendum 4 (2026-11-13, 16:18 CET):** push's blocking, end-pose and reachability tests use the drives' planner map
  (mapped footprint + margin). Map-blocked analogue of (d): if the last drive / pick failed on planning (`no path` /
  pick `unreachable`) and the plan succeeds with a pushed object's map entry removed, that object counts as blocking
  and push moves it until the plan succeeds on the updated map, else `failed: stuck`; read only by push; margin
  unchanged. The margin applies to reachability, the stopped path and the map-blocked plan, while door-corridor and
  place-pose tests use the object's real size at the mapped pose (measured: with the 0.365 m margined half-size, 11 of
  21 regression `pushed` ends fall in a corridor, push-crate 5/10; passability is guaranteed by the margined
  reachability test). A push that acted ends with a step-off out of the planner inflation around the object's new map
  entry. Cause: Phase 5 counted run 1, blocked 9204 (nav) and 9225 (skill); root cause: robot 0.20 m from its own
  margined entry.
- **Why (first form):** Phase 4 tuning 8217: a long push stopped at corridor-clear after 0.39 m, the crate was left by
  the storage-bench approach, place failed 4×; later pushes returned "pushed" with 3e-5 m moved.
- **Fixed before:** first form and addendum (d) before the Phase 4 counted run and the drawing of the base sealed set;
  addenda 2–4 before Phase 5 counted run 2, the freeze-006 interface freeze and the Phase 6 sealed run.

## 18. One meaning of "held" for place and the labeller; conservative scorer (2026-11-13)
- **Change:** place's precondition and the run labeller use one "held": `grasp.holding` (box between the pads). The
  scorer stays conservative: "robot holds nothing" fails if the box is held by either meaning (between the pads or
  any pad contact).
- **Why:** Phase 4 tuning 8217: the pad-contact `held()` read empty while the box hung on one pad, and place said
  "nothing held"; `grasp.holding` alone in the scorer would have loosened the pass rule.
- **Fixed before:** Phase 5 counted run 2 (final form, 16:30 CET), the freeze-006 interface freeze and the Phase 6
  sealed run.

## 19. The single list of simulator-truth residuals in perceived mode (2026-11-13)
- **Change:** one list of the remaining simulator-truth reads in perceived mode (supersedes earlier copies):
  (a) push reads the welded cabinet's and the crate's pose, for push's plan, its stall / moved / tilt checks and its
  verdict (walls and furniture come from the known map, not a residual); (b) the grip check and the held box's in-hand
  pose (pick's post-contact checks: holding, in-hand, lift height, tilt after a drop, standing in for gripper sensing;
  and the held box's live in-hand pose in push's carry footprint; its size is a known constant); (c) evaluation-only
  log fields (meta.eval, map_eval); (d) sim_ok; (e) place: the landing target, retreat plan and placed-vs-missed
  decision read the held box's true pose; (f) push: `_box_xy` reads the true xy of every fetch box (seen or not) for
  plan feasibility and the step-off; (g) crate / cabinet: their in-view test is segmentation px and `locate` on them
  returns the true pose. In plain words: the camera finds the boxes; the crate/cabinet positions, place's landing
  check, push's knowledge of nearby boxes and the grip check still come from the simulator. No other truth read is
  allowed in perceived mode; G5-truthfree covers place and push with each residual on its allowed list.
- **Why:** a Phase 5 code check found three unlisted truth reads on perceived decision paths (e, f, g: place gave 0
  "missed" in 104 calls; 4 crate/cabinet locates in Phase 5 counted run 2).
- **Fixed before:** the Phase 6 sealed run.

## 20. Sealed-run input mode of the runner (2026-11-13)
- **Change:** the runner `sim/agent_run.py` passes `sealed_draw=True` only when given the sealed-run input (verified
  task dicts from the regeneration of deviation 21); nothing else in it changes; in that mode it takes the verified
  dict and does not re-check it against the current `make_task`. Dev-seed stub test; diff reviewed.
- **Why:** the frozen runner raised PermissionError on every test seed.
- **Fixed before:** the Phase 6 sealed run.

## 21. Regeneration of the base sealed set from its own tree (2026-11-13)
- **Change:** the regeneration script imports from a worktree at the `seal-005` tag (guard: `sim.__file__` under that
  worktree, its HEAD = ba7bfff), runs inside the cluster job (NVIDIA renderer), regenerates each sealed task with
  `make_task(seed, type, want=slot, sealed_draw=True)`, asserts equality with the sealed entry, and hands the runner
  the verified dicts; it prints counts only.
- **Why:** the seal dependencies `sim/grasp.py`, `sim/adapter.py` and `agent/robot_api.py` differ between freeze-006
  and the sealed draw.
- **Fixed before:** the Phase 6 sealed run.

## 22. Scope of the extensions E1, E3, E2 (2026-11-13)
- **Change:** E1 = deviations 8, 9, 13 (nightstand-only) and 15 and rules-text line E lifted for one sealed set of
  100 (the out-of-sight rule and the deviation 2 / 4 keep-outs kept); E3 = fixed scripts on the base sealed set (E3a
  "search in order, always push"; E3b adds the go-round); E2 = house B, frozen agent, easings on. Order E1 → E3 → E2;
  no fixes in any extension. Later on 2026-11-14 the E3 scripts were also run byte-unchanged in house B, after house B
  was frozen.
- **Why:** to test whether the base result depends on the easings, on the VLM at all, or on house A.
- **Fixed before:** all extension runs (pre-registered on 2026-11-13).

## 23. House B world check: scripted detour via a room; door-via limit named (2026-11-14)
- **Change:** in the E2 world check, the scripted detour's `via` is the far-side room when no twin door exists (house
  A unchanged). The door-via behaviour of `navigate` is named as a frozen-skill limit and not fixed: `navigate`'s "via
  a door" only forces that door when a second door joins the same two rooms (true in house A, not in house B); "via a
  room" works in both. A world fix was also logged: the house B bed moved off the bedroom door's push lane (push-crate
  round 1: crate stuck on the bed 4/4 in that door). Redesigning house B with twin doors was rejected, because it
  would shape the house to the agent.
- **Why:** house B Phase 2 truth regression round 1: push_or_detour 0/10 and 2/10; a door via was planned back through
  the blocked doorway (`sim/tasks.script_truth`; `adapter._nav_grid` closes only twin doors).
- **Fixed before:** the house B world freeze and the E2 sealed run.

## 24. E4 design and value rule (2026-11-14)
- **Change:** E4 (tasks that need the VLM): categories C1 vague request (labelled boxes; build check or drop), C2
  place-referenced, C3 target moved mid-task (harness event), C4 out of place with a language hint; "inside a cabinet
  or drawer" dropped (no open skill in the frozen tool set); frozen agent unchanged; new script E3c (E3b + first-box
  rule + one re-look) as the fair baseline beside E3b. Value rule: per category, VLM − best script ≥ 5/25 and
  VLM ≥ 13/25 = adds value.
- **Why:** to measure the VLM on tasks designed to need its judgement, against the best fixed script.
- **Fixed before:** the E4 seal and run (pre-registered on 2026-11-14).

## 25. E4 task generator (2026-11-14)
- **Change:** E4 task generator `sim/e4.py`: categories place_ref (anchor relation, uniqueness 1.0 m), recovery (the
  harness moves the target once after its first successful locate, to the first precomputed floor spot in another
  room that is not the robot's room, out of view from it and pickable), hint (true room named in the text); no
  blockers in E4; targets keep the filter of the base sealed test (out of sight from the hall, pre-grasp feasible);
  seeds house A 180000–183xxx, house B 185000–188xxx; the full rules were written down before the seal.
- **Why:** implementation of the E4 categories of deviation 24.
- **Fixed before:** the E4 seal draw (pre-registered on 2026-11-14).

## 26. E4 printed labels on boxes (2026-11-14)
- **Change:** a generator LABELS switch (printed use-words on box faces) for C1 vague tasks only; default off =
  byte-identical worlds; C1 kept only if its build checks pass (legibility; G5-pose with labels).
- **Why:** the C1 category needs a visible cue that identifies the requested box.
- **Fixed before:** the E4 seal draw (pre-registered on 2026-11-14).

## 27. E4 baseline E3c and runner flags (2026-11-14)
- **Change:** baseline E3c (`agent/e3c.py`: E3b + first-box rule for colourless tasks + one re-look after a failed
  pick); runner flags `--script e3c`, task keys labels / move; result.json carries category / house / move.
- **Why:** E3b cannot parse colourless tasks, so a fairer script baseline was needed.
- **Fixed before:** the E4 seal draw (pre-registered on 2026-11-14).

## 28. E4 C1 (vague request) dropped (2026-11-14)
- **Change:** C1 dropped. Build check (a), legibility, failed: 12 px letters from the room point in 127/800 (15.9 %,
  house A) and 122/800 (15.3 %, house B) of dev placements against the 90 % requirement (letters are width-bound at
  30–56 mm on a 20 cm face; median camera distance 1.35 / 1.67 m); check (b), G5-pose with labels, passed 49/50. C1's
  25 tasks re-split: place_ref 34 (17/17), recovery 33 (A 17 / B 16), hint 33 (A 16 / B 17). The LABELS switch stays
  in the generator, off by default (worlds byte-identical).
- **Why:** the drop-and-re-split fallback was written before the build, for the case that the C1 build check fails.
- **Fixed before:** the E4 seal draw.

## 29. E4 value rule in proportional form (2026-11-14)
- **Change:** with C1 dropped and categories of 34 / 33 / 33: "adds value" = VLM − best script ≥ 20 percentage points
  AND VLM ≥ 50 % in that category; "no clear difference" = |VLM − best script| < 20 points; "script better" = best
  script − VLM ≥ 20 points; overall unchanged (≥ 2 of the 3 kept categories say adds value, none says script better).
  Margin met with VLM < 50 % is labelled "no clear difference (margin met, VLM below 50 %)"; a best-script tie goes to
  E3c (verdict unaffected); a task without a result after the job's restarts counts as a fail ("no result").
  Implemented in `sim/e4_analyse.py`.
- **Why:** C1 dropped (deviation 28); the proportional thresholds carry over the count form (5/25 = 20 points,
  13/25 ≈ 50 %).
- **Fixed before:** the E4 seal draw and run (pre-registered on 2026-11-14).

## World-only fixes to house B (2026-11-14)
During the house B world checks (before house B was frozen and before any VLM or script run in it) two world-only
fixes were made: the bed (deviation 23) and, after round 2 (push-then-detour 5/10: a not-movable push from the kitchen
side had no room to step off, the counter being 0.8 m behind the hall-kitchen door), the kitchen counter moved north
(round 3). No rule or threshold changed.
