# V3.3 acceptance criteria

**RLB-01** — The checker is two local models at pinned revisions, averaged, on the cited sentences, with threshold 0.65; the committed evaluation shows it is the best of the compared checkers against Sol and ranks base above v2 above v1; the 200 tuning cases are listed.
**RLB-02** — The v3b reward pays 0.25 for the label and 1.0 more for a right, structurally sound, checker-supported answer, −0.5 for an aim citation and −1 when unparsed; planted hacks earn less than an honest policy with a third wrong; the checker runs once per batch and only on eligible replies.
**RLB-03** — The v3b trainer's components sum to the reward with one checker call per batch; its trainer arguments are v3's but for the reward weights; the dev check scores each adapter in its own process on dev rows outside the tuning set and reports every selection guard.
**RLB-04** — The v3b recipe differs from v3's only in checkpoint spacing; three v3b kernels are rendered at one commit with their metadata; v3's notebooks are unchanged.
**RLB-05** — The v3b plan is final and hashed before any v3b training, pins everything that shapes training, has v3's primary test, guardrails and stretch, selects on held-out dev with the hack guards, and is recorded before the v3b notebooks.
