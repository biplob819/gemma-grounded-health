# V2.1 acceptance criteria

**VTG-01** — TGT-R10 classifies aim, objective, study-description and method sentences (fixtures both ways), excludes them from answer citations, and counts the drops.
**VTG-02** — Targets are rebuilt with citations chosen by per-sentence support (OWNER D-48); every example reconstructs from its row and rule ids; the share of answers citing an aim- or method-like sentence and the label mix are reported for v1 and each v2 build.
**VTG-03** — The dev grounding check runs v1 first; the v2 bars (citation precision, aim-citation rate, headline) are committed before v2 is measured, and v2 is judged against them.
**VTG-04** — One clean Kaggle Run-All trains v2; log, wall clock and GPU ledger stored.
**VTG-05** — The v2 checkpoint is selected by validation loss, the §5.2 bars and the dev grounding bars; `TrainRun` v2 is frozen with its adapter hash before any PQA-L use.
**VTG-06** — Read the sample: 20 v2 dev answers read against their cited sentences and noted.
**VTG-07** — A per-sentence audit of 50 seeded v2 examples passes AUD-R4 before the retrain; earlier failed audits stay recorded.

**Owner waivers.** VTG-05's dev-grounding part and VTG-07 were not met as written: v2 step 340 missed the headline and accuracy bars and no other checkpoint was tried (OWNER D-55); the round-3 audit missed AUD-R4 (OWNER D-50). VTG-03's citation bars were judged by GPT-6.1 Sol alone, not both judges (OWNER D-54). All are listed in docs/known-issues.md.
