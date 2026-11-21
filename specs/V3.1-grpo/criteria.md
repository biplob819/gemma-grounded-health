# V3.1 acceptance criteria

**RLR-01** — The reward is programmatic and deterministic, uses EvalKit's answer parser, and pays +1 for the gold label, +0.5 for grounded citations, −0.5 for an aim citation and −1 when unparsed; its constants are pinned by a test.
**RLR-02** — Planted hacks (all-"maybe", aim citation, citing everything, terse or empty answers, valid-but-unrelated citations) earn less than honest answers; each is a named test.
**RLR-03** — The GRPO recipe continues v2's adapter, pinned by Hub revision and TrainRun v2's adapter hash, and takes the base, LoRA shape and dtype from the v2 recipe; a bad recipe is refused.
**RLR-04** — `ggh grpo` logs the reward by component (summing to the reward) and the hack probes, refuses an adapter that is not v2's, and writes a report with the reward trace.
**RLR-05** — `ggh rl-dev` scores v2 and every checkpoint on dev prompts with the reward, label mix, grounded and aim shares, parse failures and answer length.
**RLR-06** — The Kaggle notebook is rendered from code, pinned to a full commit, installs the pinned GRPO packages, replays the hashed data and targets, and reads no secret.
**RLR-07** — The v3 plan is final and hashed before any training step, and pins the reward and recipe by hash, the primary test, guardrails, stretch, selection rule, test set, judge and the single scoring run.
