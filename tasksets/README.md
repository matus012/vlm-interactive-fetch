# Phase 3 task sets (task 004), drawn 2026-11-13 before any VLM call

- `phase3_tuning.json`: `dev_tasks('fetch', 20, start=4000)`, seeds 4000-4021 (4006, 4007 have no valid fetch task).
  Free to re-run for prompt / packet work.
- `phase3_counted.json`: `dev_tasks('fetch', 10, start=6000)`, seeds 6000-6009, drawn once.
  sha256_core `8beda5d541ab8df64817d5ac971fde74ae92245d88794c804ea6a9cdf4f408d4`.
  At most 3 counted runs on exactly these 10. Never tune on its logs beyond reading failure labels.
- Both drawn on main 27367da (round-3 spawn rules, deviation 9). Disjoint from Phase 2's seeds (1000-1019, 1100-1109,
  1200-1209, 2000-2039, 3000-3023) and from the untuned probe of Phase 3 (5000+). Test space untouched.
- Re-check: every later commit must regenerate the same tasks (`tests/test_tasksets.py`).

# Phase 4 task sets (task 005), drawn 2026-11-13 on code commit f769a6e, NVIDIA renderer, before any VLM call

- Rule (deviation 13, 15): `sim.sampler.draw_set(type, n, start, stop=start+200, min_surface=m)`: target room uniform
  over bedroom / storage / living_room / kitchen; support at the generator's natural share (SURFACE_P 0.25 in a room
  with a target-able spawn surface), counted sets topped up to >= 2 of 10 on a surface; target surface uniform where
  the type rule allows (push_or_detour: kitchen box -> storage bench, else kitchen table); out of sight = < 10 px in
  every hall photo; pre-grasp feasible with every other box avoided. The coffee table is never a target start: the
  < 10 px rule rejects every coffee-table box (hall px min 52 over 130 boxes, seeds 9000-9199), so surface = nightstand.
- Seed ranges (disjoint, avoid 1000-1019, 1100-1109, 1200-1209, 1300-1309, 2000-2039, 3000-3023, 4000-4021, 5000-5100,
  6000-6009, 7000-7299; seal dry-run uses 1400-1999): tuning fetch 8000, blocked 8200, push_or_detour 8400,
  two_box 8600; counted fetch 9000, blocked 9200, push_or_detour 9400, two_box 9600 (each scans at most 200 seeds).
- Counted sets: at most 3 counted runs per type on exactly these tasks. Tuning sets: free for prompt / packet work.
- Re-check: every later commit regenerates the same tasks (`tests/test_tasksets.py`); the Phase 3 sets above are
  historical (old rules) and are checked against their stored hash only.

| set | seeds used | rooms (bed/kit/liv/sto) | support floor / nightstand | surface kitchen table / storage bench | sha256_core |
|---|---|---|---|---|---|
| `phase4_fetch_counted.json` (10) | 9000-9009 | 3/3/2/2 | 8 / 2 | 5 / 5 | `4a07b7bff33ce39c39142645c2d76623b043024f5a5c57221f124cddea031a4c` |
| `phase4_blocked_counted.json` (10) | 9200-9225 | 2/2/3/3 | 8 / 2 | 5 / 5 | `d0ba389925f4e862fa6e4899481afd5b86ca25b14a3280dc8742a544f4864730` |
| `phase4_push_or_detour_counted.json` (10) | 9400-9427 | 2/3/3/2 | 8 / 2 | 7 / 3 | `4c547846c0cb06a092e8084f32615e13f97d910a24ab97cb6bbd6976d9c60e87` |
| `phase4_two_box_counted.json` (10) | 9600-9615 | 2/2/3/3 | 8 / 2 | 5 / 5 | `2adf1b870787481e14f321367c0c45172daab759dfc3a3f03c0bf69ac2a5a155` |
| `phase4_fetch_tuning.json` (10) | 8000-8011 | 2/2/3/3 | 9 / 1 | 5 / 5 | `96de0fb072ec65aaed0c5148b72fd0bc891a57a432a40aa7cad59a0f7ddab0e4` |
| `phase4_blocked_tuning.json` (20) | 8204-8251 | 5/5/5/5 | 19 / 1 | 10 / 10 | `0838154fb8650bff25656806812ab1d0cdadbf37bc97ab2a277d0c5f49d882cf` |
| `phase4_push_or_detour_tuning.json` (20) | 8401-8443 | 5/5/5/5 | 19 / 1 | 15 / 5 | `4827a27a8cb54c9d6a81818000815d7be1c691a20119e0719dd26c8b8906f659` |
| `phase4_two_box_tuning.json` (20) | 8600-8631 | 5/5/5/5 | 19 / 1 | 10 / 10 | `d6bf49d5a849807e002fbf400053a5d0407f7faf1f7b7dafb96128b27bccea0d` |
