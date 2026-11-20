# V2.2 acceptance criteria

**VEV-01** — v2 merged bf16 and Q4_K_M, Q8_0 GGUFs exist with hashes in a committed manifest, built by the X3.1 code; template parity holds (the V2.1 smoke: label first, no thought channel).
**VEV-02** — v2 dev parity is measured with X3.1's method and bars and recorded; a miss is disclosed (OWNER D-55).
**VEV-03** — `configs/eval_plan_v2.yaml` is final, names the primary pairs, outcomes, judge, extractor and rule, and its sha256 is committed before every V2.2 run started (checked by tests/results/test_preregistration.py).
**VEV-04** — The main run scores base, v1 and v2 on the same 1,000 PQA-L cases with one extractor and one judge (OWNER D-57); base and v1 answers come from the cache and equal E3.2's; the six primary comparisons are reported with Holm.
**VEV-05** — Off-scope, paraphrase-probe and judged parity runs for v2 are complete and committed.
**VEV-06** — Every V2.2 run's cost is in the spend ledger, total within $25, each row labelled built after v1's test results were seen.
