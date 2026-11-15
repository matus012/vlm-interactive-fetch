# Per-task results

One JSON line per task of each sealed test, produced from the counted runs' `result.json` files.

| file | test | arm |
|---|---|---|
| `base_vlm_houseA.jsonl` | base sealed test (Phase 6), house A | VLM agent |
| `e1_vlm_houseA_no_easings.jsonl` | E1, house A, generator easings off | VLM agent |
| `e2_vlm_houseB.jsonl` | E2, house B | VLM agent |
| `e3a_houseA.jsonl`, `e3b_houseA.jsonl` | E3, the base test's tasks | fixed scripts E3a / E3b |
| `e3a_houseB.jsonl`, `e3b_houseB.jsonl` | E3, the E2 test's tasks | fixed scripts E3a / E3b |
| `e4_vlm.jsonl`, `e4_e3b.jsonl`, `e4_e3c.jsonl` | E4, both houses | VLM agent / E3b / E3c |

Fields: `task_id`, `house`, `type` (fetch / blocked / push_or_detour / two_box; E4: place_ref / recovery / hint),
`instruction`, `arm`, `pass` (the scorer's verdict), `steps_used`, `step_limit`, `failure_label` (the failure class
of a failed run: perception / skill / vlm / ...; null on a pass), `failed_checks` (the scorer's failed checks),
`calls` (the skill-call sequence, each `call(args) -> result`; for the VLM the argument of `done` is its own text),
`result_missing` (always false here: every task has a result).

The full VLM transcripts (prompts, thinking text and replies, without images) total about 46 MB across the four
VLM tests, above the 25 MB set for this snapshot, so only these summaries are published.
