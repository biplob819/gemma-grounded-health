# V3.7 acceptance criteria

**RBS-01** — The main_v3b run uses the plan's models, judge, extractor, seed, task and 1,000 cases, with v3b served from the frozen Q4_K_M (and Q8_0 as the owner's exploratory extra); base and v2 raw answers are identical to the V2.2 run's on every case; the run starts after the plan, the V3.6 hashes and this node's config are committed; its spend is in the ledger and within the owner's $5 cap.
**RBS-02** — The primary test, guardrails, stretch goal, non-copy headline and the plan's other reported signals are computed from the committed run with the plan's method, and the verdict follows the plan's rule; the Q8_0 comparisons are labelled exploratory.
**RBS-03** — v3b's Q8_0 dev parity check rebuilds from its committed generations, reuses the registered bf16 replies on the same prompts, and is labelled exploratory beside the registered Q4_K_M parity.
