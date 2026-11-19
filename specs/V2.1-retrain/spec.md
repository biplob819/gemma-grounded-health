# V2.1 — Citation-fixed targets and retrain (v2)

**Wave 3 (v2 iteration, OWNER D-43). Depends on:** E3.2. **Unblocks:** V2.2.

## Goal
Build training targets whose citations support their answers sentence by sentence, prove it on a per-sentence audit, retrain once on Kaggle, and choose the checkpoint on dev data only, with a grounding bar fixed before v2 is measured (D-4, OWNER D-43..D-48).

## Design
- **TGT-R10 (VTG-01).** Answer citations never include an aim, objective, study-description or pure method sentence (v2, tightened after OWNER D-46).
- **Citations by support (VTG-02, OWNER D-48).** Each cited sentence must, on its own, entail part of the answer under the pinned NLI model and report a result; candidates are ranked by that entailment, not by word overlap; an answer whose cited set does not entail it is dropped. The v1, v2 and v2.x targets stay reproducible from their rule ids.
- **Dev grounding check (VTG-03, OWNER D-44).** EvalKit at ee59699 with `citation_support` runs the 200-case `dev_ground` task. v1's baseline is measured first (`docs/results/dev_ground_v1/`); the v2 bars are fixed and committed before v2 is measured.
- **Retrain (VTG-04, VTG-05).** The T2.2 notebook and recipe on the new targets, one clean Kaggle session; selection by validation loss, the §5.2 bars and the dev grounding bars; `TrainRun` v2 frozen before any PQA-L use.
- **Audits (VTG-07, OWNER D-45..D-47).** A per-sentence audit of 50 seeded examples must pass before the retrain. The first v2 audit (owner) and the v2.1 re-audit (blind Claude reviewer) both failed and stay recorded.
- **Read the sample (VTG-06).** 20 v2 dev answers read against their cited sentences.
