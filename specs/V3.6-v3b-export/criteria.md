# V3.6 acceptance criteria

**RBX-01** — v3b's main-run outputs are frozen and every dev summary recomputes from its replies with the real checker (CI checks files and held-out rows); `ggh rl-freeze --model ft3b` applies the v3b plan's rule (300 held-out rows, gold-"no" slack 0.10, copy share at most v2's plus 0.10), refuses any other checkpoint, and writes `train_selection.json` and an `RLRun` record; a candidate that breaks the copy guard is never frozen.
**RBX-02** — v3b (`ft3b`) has its own adapter, merged, GGUF and measurement paths and serve port; the merge record and the GGUF manifest name the frozen adapter hash, and the GGUF sizes equal v3's.
**RBX-03** — v3b's dev parity (Q4_K_M against merged bf16, 500 prompts) is recorded under X3.1's bars with its every-25th sample; a miss is reported to the owner before scoring.
