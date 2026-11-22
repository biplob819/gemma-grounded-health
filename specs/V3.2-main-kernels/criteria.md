# V3.2 acceptance criteria

**RLM-01** — The trial's outputs are committed, every summary recomputes from the committed replies, the plan's four stop rules pass on them, and each rule is shown able to stop the plan.
**RLM-02** — The main run is two kernels rendered from code at one full commit: a train kernel (`ggh grpo` only) and a score kernel (`ggh rl-dev --n 500` on the train kernel's output attached as a kernel source); both are private T4 kernels reading no secret; the trial notebook renders unchanged.
**RLM-03** — The dev checkpoint selection follows the plan: 500 rows, the seven guards against v2, the highest mean reward among candidates keeping them, ties to the earlier step, no selection when none keeps them, and whether the pick beats v2 reported.
