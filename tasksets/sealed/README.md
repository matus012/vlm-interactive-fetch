# Sealed task lists

Each list was drawn once by a seeded generator, hashed, and kept unread until its single counted run. The tests are
over, so the lists are published here to let anyone re-run the exact test. `SHA256SUMS` holds the hashes; each
`.meta.json` holds the draw rule, seed space, per-type counts, MuJoCo version and renderer, and the sha256 of every
generator file the draw depended on (`dep_sha256`). The `commit` and `tags` fields name commits in the development
history, which is not part of this one-commit snapshot.

| file | test | tasks | sha256 |
|---|---|---|---|
| `test_100.jsonl` | base sealed test (Phase 6, house A) and E3 house A | 25 each of fetch / blocked / push_or_detour / two_box | `8f691844eb8312f05c1a40f386b73faf515be656948b42fe76a510bceede542d` |
| `e1_100.jsonl` | E1, house A with the generator easings off | 25 per type | `b4d64652c978f8d4f37fc31c65ebb786b3a2d9fbaffc0aa31937e880beaea3c6` |
| `e2_100.jsonl` | E2 (house B) and E3 house B | 25 per type | `f7dd2a7f30f57f54b0b8a87771dd8d371f2ad44bdb88d1ad139ef5615a0ecc53` |
| `e4_100.jsonl` | E4, both houses | place_ref 34, recovery 33, hint 33 | `42d7ada535d3d6f2831f374d5a211c6336dfff58a60d7954e7b74bc569bce260` |

How the hash was checked before each run: the batch job first compared the sealed file's sha256 with the value in its
meta and in the job script, then regenerated every task from its seed with the generator code at the seal commit
(`hpc/seal_regen.py`; for E4 `sim/e4_seal.py regen`), after checking that the generator files hash as in
`dep_sha256`. Only if every regenerated task equalled the sealed one did it write `verified.jsonl`, record that file's
sha256, and start the runs; a mismatch stopped the job before the VLM server started.

Check the files: `sha256sum -c SHA256SUMS`.
