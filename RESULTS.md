# Results

An off-the-shelf vision-language model (Qwen3.8-27B) chooses scripted skills (navigate, optionally via a door or
room; look, locate, pick, place, push, done) from camera photos and object names, and drives a TidyBot fetch
robot in a simulated MuJoCo house. All results below are from **sealed tests**: the task list was drawn by a
fixed rule and its sha256 hash recorded before any run of the frozen system; the tasks were first opened inside the
counted job, regenerated there and checked byte-equal against the sealed copy. Each arm (the VLM agent, each fixed
script) had exactly one counted pass on a sealed set, with no retries, nothing re-scored and no fixes afterwards;
diagnostic runs after a counted pass (the E1 oracle split, replays of failed runs on a second machine) are named
as such and change no counted result. Each task counts as passed only if the intended box ends on the target
surface, nothing is held, and `done` is called within the step limit (20 steps; 30 for two-box tasks). Ranges are
Wilson 95 % intervals. With 25 tasks per type the intervals are wide; per-type numbers are point estimates, not
established rates. The VLM samples at temperature 0.6, so its results are one stochastic draw; the fixed scripts are
deterministic. A 70 % per-type success level was set as the target before the project.

Experiments:
- **Base sealed test (Phase 6):** house A, frozen system, 100 tasks (25 per type).
- **E1:** house A with the task-generator easings lifted (deviations 8, 9, 13 nightstand-only, 15) and rules-text
  line E removed (line E tells the VLM to name the other door in `via` to go round an immovable cabinet).
- **E2:** house B, a new floor plan, frozen system unchanged.
- **E3:** the VLM replaced by fixed scripts that read only the VLM's packet text. **E3a**: search rooms in order,
  always push. **E3b**: E3a plus "if push fails, go round through another door that joins the same two rooms".
- **E4:** tasks designed to need the VLM, both houses, VLM against E3b and a new baseline **E3c** (E3b plus a
  first-box rule for colourless tasks and one re-look after a failed pick).

## 1. Base sealed test (Phase 6)

Frozen system, 100 sealed tasks in house A, one counted pass: **86/100**.

| type | passed | % | 95 % CI (Wilson) | 70 % level |
|---|---|---|---|---|
| fetch | 24/25 | 96 % | 80–99 % | met |
| blocked door (push the crate) | 19/25 | 76 % | 57–89 % | met |
| push-or-detour (the cabinet does not move) | 20/25 | 80 % | 61–91 % | met |
| two boxes | 23/25 | 92 % | 75–98 % | met |
| **total** | **86/100** (78–91) | | | |

The lower bounds for blocked and push-or-detour are below 70 %. 3 of the 14 failures had the box on its target but ran
out of steps. The camera finds the boxes; the crate and cabinet positions, place's landing check, push's knowledge of
nearby boxes and the grip check still come from the simulator (deviation 19). Depth is exact (no noise model).

### 1.1 Failure breakdown (14 failures, labels from the frozen labeller)

Failure label = the part that returned a wrong result; an unrecovered correct failure result is labelled `vlm`
(deviation 14). Labels: 7 `vlm`, 7 `skill`. 5 of the 7 `skill` labels are perception-caused; the labeller never
assigns `perception` in agent runs, so the label alone does not show this.

| task | label | what went wrong | cause class |
|---|---|---|---|
| fetch-100005 | skill | nightstand box located from 1.8 m with an 8.5 cm / 72° pose error; 7 picks from the same wrong pose failed | perception: nightstand pose (ii) |
| blocked-100024 | vlm | explored before pushing; box on the table at step 20, no step left for `done` | steps (box on target) |
| blocked-100025 | vlm | same, plus 2 malformed calls; box on target, no `done` | steps (box on target) |
| blocked-100031 | vlm | 15 exploration steps before the push, then out of steps | steps / search |
| blocked-100042 | vlm | `locate` 12 times in a row; the box sat just under the visibility bar (97 / 100 px) | search |
| blocked-100046 | vlm | never searched the kitchen; the box was barely visible (85 px) from the living room | search |
| blocked-100054 | skill | box located from 4.3 m; the first pick tipped it onto its side, then every pick slipped | perception: far pose + no tipped-box check (iii) |
| push_or_detour-100043 | skill | same as 100054, from 4.2 m | perception: far pose + (iii) |
| push_or_detour-100058 | vlm | 11 exploration steps; picked at step 20, ended holding the box | steps / search |
| push_or_detour-100065 | skill (lost box) | carrying, the robot drove into the cabinet; the impact knocked the box out of the grip | new: carried box lost on a cabinet stop |
| push_or_detour-100090 | skill (lost box) | same drop, twice; the VLM shares the blame (routed through the blocked north door after already using the south detour) | new + VLM |
| push_or_detour-100101 | vlm | box on target at step 20, `done` never called | steps (box on target) |
| two_box-100070 | skill | red box located from 4.1 m; the second pick reused that far pose and tipped it; 7 more picks slipped | perception: far pose + (iii) |
| two_box-100100 | skill | 7.1 cm pose error at the nightstand; the box fell, was picked upside down and placed upside down, so the 15° upright rule failed it; the VLM's `done` note wrongly claimed success | perception (ii) → (iii) |

Known limits, revisited: (i) stale photos decided no failure; (ii) nightstand pose error: 2 runs; (iii) no tipped-box
check, combined with far poses: 3 runs; new: a carried box knocked loose when the drive is stopped by the cabinet:
2 runs.

### 1.2 First-pick success against the distance of the last `locate` (all 100 runs)

| distance | first picks that succeeded |
|---|---|
| < 1.6 m | 66/66 |
| 1.6–2.4 m | 41/47 |
| > 2.4 m | 6/11 (3 of the 11 ended with the box tipped) |

### 1.3 Other measures

| measure | value |
|---|---|
| median steps | fetch 9, blocked 14, push-or-detour 13, two-box 17 |
| push-or-detour choices (25 runs) | detoured without pushing 12; pushed the cabinet (`not movable`), then detoured 10; pushed only 1; neither 2 |
| `navigate via` calls (push-or-detour) | 24 reached, 6 blocked (the VLM routed via the blocked door) |
| push on the crate (blocked) | `pushed` 22×, `stuck` 1× |
| VLM latency | median 2.8 s, p90 14.7 s, max 176 s (1,429 calls) |
| wall per run | median 107 s, max 656 s (16 runs in parallel beside vLLM) |
| validation errors / unseen-box contacts | 8 / 0 |
| leak scan (100 runs) | 0 hits |

## 2. Summary across experiments

| test (100 sealed tasks each, one counted pass, no fixes) | fetch | blocked | push-or-detour | two-box | all |
|---|---|---|---|---|---|
| base: house A, frozen VLM agent | 24 | 19 | 20 | 23 | **86** |
| E1: house A, easings lifted, rule line E removed | 21 | 22 | 20 | **14** | **77** (77/84 where the arm can reach) |
| E2: house B (new plan), frozen VLM agent | 25 | 19 | **17** | 25 | **86** |
| E3a: script "search in order, always push", house A / B | 21 / 24 | 23 / 23 | 2 / 2 | 23 / 21 | 69 / 70 |
| E3b: script + "go round if push fails", house A / B | 21 / 24 | 23 / 23 | 21 / 2 | 23 / 21 | 88 / 70 |

Per type out of 25. Bold cells in the type columns mark a VLM-agent result below the 70 % level (script cells are
not marked).

## 3. E1: easings lifted, rules line E removed

100 fresh sealed tasks, house A, frozen agent without rules line E: **77/100** (68–84). Base sealed test: 86/100
(78–91).

| type | E1 | 95 % CI | base sealed test | E1, oracle-solvable | E1, target reachable (pre-grasp pose exists) |
|---|---|---|---|---|---|
| fetch | 21/25 | 65–94 % | 24/25 (80–99) | 21/21 | 21/21 |
| blocked door | 22/25 | 70–96 % | 19/25 (57–89) | 19/20 | 22/23 |
| push-or-detour | 20/25 | 61–91 % | 20/25 (61–91) | 20/23 | 20/23 |
| two boxes | 14/25 | 37–73 % | 23/25 (75–98) | 13/16 | 14/17 |
| all | 77/100 | 68–84 % | 86/100 (78–91) | 73/80 (83–96) | 77/84 (84–96) |

Against the 70 % level: fetch (84 %), blocked (88 %) and push-or-detour (80 %) meet it; two boxes (56 %) does not.
8 of the 11 two-box failures had a target the arm cannot reach; a two-box task draws two targets, so it is about
twice as likely to contain one. On the 84 tasks whose target is reachable, the agent passed 77 (92 %, CI 84–96).

### 3.1 Eased cases in the E1 set (including the 16 unreachable targets)

"Target" means the task's target boxes; "any" means any of the 4 boxes.

| case | tasks | E1 pass |
|---|---|---|
| target has no pre-grasp pose (deviation 15) | 16 | **0/16** |
| target on the back of the nightstand (deviation 8 edge rule) | 21 | 5/21 |
| any box on the back of the nightstand / coffee table | 36 / 15 | 19/36 / 7/15 |
| target on a route corridor (deviation 9) | 25 | 17/25 |
| any box on a route corridor | 66 | 53/66 |
| target floor box near a spawn surface (deviation 8 floor rule) | 4 | 3/4 |
| any box near a spawn surface | 9 | 8/9 |
| no eased case at all | 15 | 15/15 |

All 16 unreachable targets are deviation-8 cases (back of a surface, or a floor box near one). Their 16 failures are
all "pick → failed: unreachable", repeated until the steps ran out or the VLM gave up; they are labelled `skill`, and
no agent could have passed them. Excluding those 16 tasks, tasks with a box on a route pass 53/59 (80–95 %); tasks
without pass 24/25 (80–99 %).

Oracle split (Phase 2 correct script, truth poses, run after the counted pass): 80 of 100 tasks are solvable by the
scripted skills (the 16 unreachable targets plus 4 others are not). The agent passed those 4 others. The oracle is a
lower bound on solvability. On the solvable 80 the agent passed 73.

### 3.2 Push-or-detour without rules line E (cabinet tasks, 25 each)

| | E1 (no line E) | base (with line E) |
|---|---|---|
| blocked by the cabinet at least once | 24 | 23 |
| tried push(cabinet) | 12 | 10 |
| routed with via | 23 | 22 |
| push then via / via without push | 11 / 12 | 9 / 13 |
| a via through the cabinet's own (blocked) door | 7 runs | 4 runs |
| pass | 20 | 20 |

E1 failures (23): `skill` 17 (16 unreachable targets + one two-box run with slipped and missed picks), `vlm` 6.
Median steps: fetch 15, blocked 14, push-or-detour 16, two-box 21 (base: 9, 14, 13, 17). Leak scan: 0 hits over 100
runs.

## 4. E2: new house (house B), frozen agent

House B: different floor plan, same rooms, furniture, boxes, crate and cabinet. It was designed without access to the
E3 scripts, passed the Phase 1/2 world checks after two logged world fixes, and was frozen before any VLM or script
run. 100 fresh sealed tasks, one counted pass: **86/100**.

| type | house A (base sealed test) | house B | 95 % CI (B) | E3a A / B | E3b A / B |
|---|---|---|---|---|---|
| fetch | 24/25 | 25/25 | 87–100 % | 21 / 24 | 21 / 24 |
| blocked door | 19/25 | 19/25 | 57–89 % | 23 / 23 | 23 / 23 |
| push-or-detour | 20/25 | 17/25 | 48–83 % | 2 / 2 | 21 / 2 |
| two boxes | 23/25 | 25/25 | 87–100 % | 23 / 21 | 23 / 21 |
| **all** | **86/100** (78–91) | **86/100** (78–91) | | 69 / 70 | 88 / 70 |

Against the 70 % level: fetch, blocked and two boxes meet it; push-or-detour (68 %) misses by one task.

### 4.1 House B world checks (round 3, the frozen files)

| check | result |
|---|---|
| G1-reach | 12 places × 10 = 120/120, 0 wall / furniture contacts |
| G1-cover | every spawn spot seen from its room point: storage 172/172, living room 120/120, bedroom 188/188, kitchen 92/92 (min max-px 178) |
| G1-rest | pass (seeds 0–3 × 3 variants, ≤ 1 mm) |
| G1-det | 20 seeds × 3 variants, 3 processes with different hash seeds: one sha256 |
| G2 truth regression | pick 20/20, place 20/20, push-crate 10/10, push-cabinet 10/10, blocked pass; scripted tasks 10/10 per type and per push-or-detour branch |

### 4.2 Push-or-detour in house A vs house B (25 cabinet tasks each)

`navigate`'s "via a door" forces that door only when a second door joins the same two rooms: true in house A, not in
house B (deviation 23). "Via a room" works in both.

| | house A | house B |
|---|---|---|
| runs with a door via | 22 (18 passed) | 5 (1 passed) |
| door-via calls | 24 reached / 4 blocked | 0 reached / 7 blocked |
| runs with room via only | 0 | 18 (15 passed) |
| room-via calls | 2 blocked | 18 reached / 7 blocked |
| replies with no single tool call (all 100 tasks) | 4 | 19 (18 in the 8 failed cabinet runs) |

House B VLM failures: 14 = `vlm` 11 (push-or-detour 8, blocked 3), `skill` 3. Leak scan: 0 hits.

Median steps:

| | A | B |
|---|---|---|
| fetch | 9 | 10 |
| blocked | 14 | 13 |
| push-or-detour | 13 | 13 |
| two-box | 17 | 17 |

E3 scripts in house B: pre-registered on 2026-11-14 (prediction written before the house B counted pass): E3b's
go-round set (doors that share their two rooms with another door) is empty in house B, so E3b behaves as E3a on every
cabinet task. The prediction held: `agent/e3.py` byte-unchanged, 70/100 each, push-or-detour 2/25 (19 runs "blocked,
push failed", 4 target not found).

## 5. E3: fixed-script baselines (house A, base sealed tasks)

Same 100 sealed tasks as the base sealed test, same skills, scorer, step limits and perceived mode; the VLM replaced
by a fixed script reading only the packet text. Both scripts hashed and committed before the sealed regeneration. One
counted pass per script.

| type | VLM (base sealed test) | E3a "search in order, always push" | E3b "+ go round" |
|---|---|---|---|
| fetch | 24/25 (80–99) | 21/25 (65–94) | 21/25 (65–94) |
| blocked door | 19/25 (57–89) | 23/25 (75–98) | 23/25 (75–98) |
| push-or-detour | 20/25 (61–91) | **2/25** (2–25) | 21/25 (65–94) |
| two boxes | 23/25 (75–98) | 23/25 (75–98) | 23/25 (75–98) |
| **all** | **86/100** (78–91) | **69/100** (59–77) | **88/100** (80–93) |

House B numbers for E3a / E3b are in sections 2 and 4.

Median steps used:

| | VLM | E3a | E3b |
|---|---|---|---|
| fetch | 9 | 13 | 13 |
| blocked | 14 | 14 | 14 |
| push-or-detour | 13 | 13 | 16 |
| two-box | 17 | 21 | 21 |
| all | 13.5 | 15 | 16 |

Where each script fails (house A):

| | E3a | E3b |
|---|---|---|
| push failed (cabinet), nothing else to try | 19 | 0 |
| pick missed | 5 | 5 |
| pick slipped | 4 | 4 |
| navigate "no path" | 1 | 1 |
| step limit | 1 | 2 |
| target not found | 1 | 0 |
| labels (script failures remapped from `vlm` to `script`) | script 21, skill 9, nav 1 | script 2, skill 9, nav 1 |

The VLM passed 24 tasks E3a failed (18 of them push-or-detour) and 7 that E3b failed. E3b passed 9 tasks the VLM
failed: 5 blocked, 2 push-or-detour, 1 fetch and 1 two-box.

Branches hit in the sealed pass:

| branch | E3a | E3b |
|---|---|---|
| target found | 123 | 124 |
| failed: blocked | 46 | 66 |
| crate pushed, step retried | 24 | 24 |
| no crate → push cabinet | 21 | 20 |
| push failed | 22 | 21 |
| search moved on after a failed push | 2 | – |
| search moved on after a failed navigate | – | 1 |
| not found anywhere → done | 1 | – |
| go round (via) | – | 21 |
| second box delivered | 23 | 23 |

## 6. E4: tasks that need the VLM (VLM vs E3b / E3c)

Same frozen agent, 100 new sealed tasks (50 per house), no blockers, one target, step limit 20, one counted pass, the
VLM vs E3b and E3c on the same tasks. Categories: place-referenced ("the box next to the sofa", no colour named),
recovery (the target is moved once, out of view, right after the first `locate` that finds it), hint (the text names
the target's true room). The vague-request category (C1, printed box labels) was dropped before the seal: labels were
legible from the room point in 15.9 % (house A) / 15.3 % (house B) of dev placements against a 90 % requirement
(deviation 28). "Inside a cabinet or drawer" was not built (no open skill in the frozen tool set; deviation 24).

### 6.1 Value rule (pre-registered on 2026-11-14, before the seal)

Count form (deviation 24), per category of 25: adds value = VLM − best script ≥ 5 tasks and VLM ≥ 13/25.
Proportional form used, because C1 was dropped and categories are 34 / 33 / 33 (deviation 29):

| verdict | condition (best script = the better of E3b and E3c in that category) |
|---|---|
| adds value | VLM − best script ≥ 20 percentage points AND VLM ≥ 50 % |
| script better | best script − VLM ≥ 20 points |
| no clear difference | otherwise (\|VLM − best script\| < 20 points) |
| overall "the VLM adds value on E4" | ≥ 2 categories say "adds value" and none says "script better" |

### 6.2 Per category, pooled over both houses

| category (n) | VLM | E3b | E3c | best script | VLM − best | verdict |
|---|---|---|---|---|---|---|
| place-referenced (34) | **31** (91 %, 77–97) | 0 (0–10) | 1 (1–15) | E3c 1 | +88 pts | **adds value** |
| recovery, box moved mid-task (33) | **0** (0 %, 0–10) | 0 (0–10) | 8 (24 %, 13–41) | E3c 8 | −24 pts | **script better** |
| hint, "someone left it in the …" (33) | **33** (100 %, 90–100) | 29 (73–95) | 31 (80–98) | E3c 31 | +6 pts | **no clear difference** |
| all (100) | 64 (54–73) | 29 (21–39) | 40 (31–50) | | | |

Wilson 95 % ranges in brackets. **Overall verdict by the pre-registered rule: the VLM does not clearly add value on
E4** (the recovery category says "script better").

### 6.3 Per house (VLM / E3b / E3c)

| category | house A | house B |
|---|---|---|
| place-referenced | 14 / 0 / 1 of 17 | 17 / 0 / 0 of 17 |
| recovery | 0 / 0 / 5 of 17 | 0 / 0 / 3 of 16 |
| hint | 16 / 13 / 14 of 16 | 17 / 16 / 17 of 17 |

### 6.4 Paired, VLM against E3c

| category | VLM pass, E3c fail | E3c pass, VLM fail |
|---|---|---|
| place-referenced | 30 | 0 |
| recovery | 0 | 8 |
| hint | 2 | 0 |

### 6.5 Recovery behaviour (33 tasks; the move happened in 33/33 for every arm, same destination in all three arms)

| | VLM | E3c | E3b |
|---|---|---|---|
| a stale "found" locate after a failed pick | 29 / 33 | 0 | 0 |
| looked again in place right after the first failed pick | 14 / 33 | 33 / 33 | 0 |
| later drove to the room the box was moved to | 0 / 33 | 15 / 33 | 0 |
| found the box again in its new room (pre-registered) | 0 / 33 | 17 / 33 | 0 |
| failed picks, total | 190 | 34 | 33 |
| passed | 0 | 8 | 0 |

The frozen `locate` reads the last photo set, so after the move it keeps answering "found" until the robot looks
again. Labels: 32 `skill`, 1 `vlm`.

### 6.6 Place-referenced and hint, secondary measures

Place-referenced, VLM split by relation: "on" a surface 16 / 16; "next to" furniture 15 / 18 (all three misses
were "next to the kitchen counter"). E3c took the right
box 3 times in 34 and passed once.

Hint (pre-registered secondary measures):

| | VLM | E3b | E3c |
|---|---|---|---|
| first navigate goes to the hinted room | 33 / 33 | 9 / 33 | 9 / 33 |
| median steps | 7 | 12 | 12 |

Leak scan over all 100 VLM runs: 0 hits; no colour word in any place-referenced task line.

## 7. Compute

| experiment | H200 GPU-hours |
|---|---|
| Phases 0–6 (of which the base sealed test 0.40) | 4.36 |
| E1 (counted pass 0.85, oracle 0.05) | 0.89 |
| E3 (house A) | 0.15 |
| E2 (house B world checks, smoke, counted pass + E3 in house B) | 0.85 |
| E4 | 0.75 |
| **total** | **7.0** |

<!-- sources (development write-ups and analysis outputs, not part of this snapshot):
Intro / sealed-test definition: 2026-11-13-007-phase6.md (Result, Integrity record); 2026-11-14-final-extensions.md (Why).
Section 1 base per-type table: 2026-11-13-007-phase6.md (Result); 2026-11-13-final-interactive-fetch.md (Result). Total CI 78-91: 2026-11-14-009-e1.md (Result).
Section 1.1 failure breakdown: 2026-11-13-007-phase6.md (Why: failures; Known limits, revisited).
Section 1.2 first-pick vs distance: 2026-11-13-007-phase6.md (Why: main perception finding).
Section 1.3 other measures: 2026-11-13-007-phase6.md (Why: also listed measures; G6-leak row).
Section 2 summary table: 2026-11-14-final-extensions.md (Result).
Section 3 E1 main table: 2026-11-14-009-e1.md (Result).
Section 3.1 eased cases + oracle split: 2026-11-14-009-e1.md (Why: the eased cases in the E1 set; oracle split).
Section 3.2 push-or-detour without line E, failures, steps: 2026-11-14-009-e1.md (Why: push or detour without rules line E; failures, steps, latency, wall).
Section 4 E2 main table: 2026-11-14-011-e2.md (Result).
Section 4.1 house B world checks: 2026-11-14-011-e2.md (Result: house B world gates, round 3).
Section 4.2 door/room via, labels, steps: 2026-11-14-011-e2.md (Why: push or detour in house B; labels, steps, latency, wall).
Section 4 E3 in house B prediction and outcome: 2026-11-14-011-e3b-prediction.md; 2026-11-14-011-e2.md (Why: E3 scripts in house B); 2026-11-14-final-extensions.md (Result).
Section 5 E3 tables (main, steps, failures, branches): 2026-11-14-010-e3.md (Result; Why: where each script fails; branches hit).
Section 6.1 value rule: 2026-11-14-014-e4-rules.md (The value rule; proportional form); 2026-11-15-014-e4.md (Result: the rule).
Section 6.2-6.4 per category, per house, paired: 2026-11-15-014-e4.md (Result); 2026-11-14-final-extensions.md (Addendum E4).
Section 6.5 recovery behaviour: 2026-11-15-014-e4.md (Why: recovery).
Section 6.6 place-referenced and hint: 2026-11-15-014-e4.md (Why: place-referenced; hint); the split by relation is taken from
the analysis output e4_analysis.json (place_ref_by_relation: on 16/16, next 15/18), recounted from results/e4_vlm.jsonl,
because the E4 write-up gave the denominators as 17 / 17 by mistake.
Section 7 compute: 2026-11-13-007-phase6.md (Result: cost); 2026-11-14-009-e1.md (Evidence); 2026-11-14-010-e3.md (Integrity and cost); 2026-11-14-011-e2.md (Evidence); 2026-11-15-014-e4.md (Evidence); 2026-11-14-final-extensions.md (Result; Addendum E4).
-->
