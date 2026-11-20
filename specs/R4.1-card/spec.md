# R4.1 — Model card

**Wave 5 (release). Depends on:** E3.3, X3.2. **Unblocks:** R4.2.

## Goal
Generate `MODEL_CARD.md` for the shipped release (v2) from `docs/results/results_bundle.json` and committed evidence only, with every §10.1 section, every number equal to the bundle, the stamps and disclosures word for word, and Hub metadata that validates (FR7, US4, D-20, C5).

## Design
- **Generated, never typed (CRD-01, CRD-02).** `ggh card` renders the template from the bundle and its sources; a test checks every number in the card against the bundle. The shipped release's own inputs (recipe_v2, targets_v2 lineage, v2 laptop bench) are used.
- **Stamps (CRD-03, OWNER D-59, D-60, D-61, D-62).** The card prints the shipped release's per-judge calibration stamp as the bundle words it (GPT-6.1 Sol: agreement with a Claude Opus 5.5 reference labeller; Opus reported, never stamped), the judge mode, that B5 and B8 are blind Claude Opus 5.5 reviews, and the D-62 aim-rule caveat (no v2-beats-base citation claim). Earlier releases appear only as published, never compared; no line may claim v1 beats Luna under Sol.
- **Hub metadata (CRD-04).** YAML front matter: `license`, `base_model`, `datasets`, `language`, `pipeline_tag`, `tags`, `model-index` with the v2 results; validated with `huggingface_hub`'s model-card metadata parser.
- **Licence and credits (CRD-05, CRD-06).** Apache-2.0, the Gemma NOTICE and terms re-read and recorded, PubMedQA MIT credit, EvalKit credit; "Research only, not medical advice" in the summary and the footer.
