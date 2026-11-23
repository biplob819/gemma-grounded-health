# V3.4 acceptance criteria

**RLX-01** — v3's main-run outputs are frozen and every dev summary recomputes from its replies; `ggh rl-freeze` applies the plan's selection again, refuses any other checkpoint, a run not started from v2's frozen adapter or not on the pinned recipe, copies and hashes only the two adapter files, and writes `train_selection.json` and an `RLRun` record.
**RLX-02** — v3 (`ft3`) has its own adapter, merged, GGUF and measurement paths and serve port; the merge record and the GGUF manifest name the frozen adapter hash, and the GGUF sizes equal v2's.
**RLX-03** — v3's dev parity (Q4_K_M against merged bf16, 500 prompts) is recorded under X3.1's bars with its every-25th sample; a miss is reported to the owner before scoring.
