# V3.8 — v3b release preparation: evidence, bundle, card and Hub dry run (GGH v1.1.0)

**Wave 7 (OWNER D-66, D-70; rulings OWNER D-75; mechanics D-76). Depends on:** V3.7. **Ends at:** a Hugging Face dry run. The push and the tag are asked of the owner separately.

## Why
v3b is the first fine-tune above base on the headline (0.503 at Q4_K_M against base's 0.404 and v2's 0.329; V3.7). The owner chose to prepare it as GGH v1.1.0, shipping Q4_K_M and Q8_0 with Q8_0 recommended (OWNER D-75). v1.0.0's release (v2) carried evidence v3b lacks: off-scope answers reviewed blind (B8), paraphrase probes, a laptop bench and lineage records.

## Design
- **Evidence (RBR-01).** `configs/eval_plan_v3b_release.yaml`, hashed before the runs: offscope_v3b (v3b at Q4_K_M and Q8_0, 20 prompts), probe_original_v3b and probe_paraphrase_v3b (Q4_K_M, 100 cases each). Caps $0.30, $0.35 and $0.35, the owner's $1 in all (OWNER D-75, D-76). B8 blind review of the 40 off-scope answers under configs/offscope_review_prompt.md (OWNER D-61); the main_v3b sample read for the bundle's B5.
- **Records (RBR-02).** `ggh export-records --model ft3b`: export and lineage manifests (train_v3b, export_ft3b, parity_v3b) whose chain runs parity → export → GRPO training → v2's training → v2's targets → data; every record names adapter 0537933d….
- **Bundle (RBR-03).** A v3b entry in configs/results.yaml; the schemas accept a GRPO run record (`RLRun`) as a release's training record. Before anything is regenerated, v1.0.0's published bundle and card are frozen as byte copies under docs/results/releases/v1.0.0/ with a test that the v1+v2 build still reproduces them (D-76). With v3b in the bundle its spend counts every ledger row (OWNER D-75).
- **Card, README, notes (RBR-04).** The card leads with v3b's registered Q4_K_M result, recommends Q8_0 where 5 GB fits (Q8_0 parity 0.990, headline 0.541, exploratory), discloses Q4_K_M's parity miss (OWNER D-73), describes the GRPO training, and carries V3.7's caveats; audits are v2's (D-76).
- **Bench (RBR-05).** `ggh bench --model ft3b` for Q4_K_M and Q8_0.
- **Hub dry run (RBR-06).** `ggh release --release v3b` dry run: both GGUFs, the adapter, the card, records; tag v3b.0.0; nothing pushed.
