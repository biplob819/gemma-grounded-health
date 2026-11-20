# R4.2 — Hugging Face release

**Wave 5 (release). Depends on:** R4.1. **Unblocks:** R4.3, R4.4.

## Goal
Publish v2 (`gemma-e2b-grounded-v2-q4`) on the Hub as two repos (D-13), only after lineage verifies, with every file checked back by sha256 and the GGUF answering through Ollama (FR8, NFR5, US7). Pushed with the owner's token and go-ahead (given in session, 2026-11-20).

## Design
- **Plan first (HUB-01).** `ggh release --owner biplob123` verifies the v2 lineage chain (parity_v2 → export_ft2 → train_v2 → targets_v2 → data) and plans every file with size and sha256; the plan is committed (`docs/measurements/release_plan_v2.json`). Any problem stops the push.
- **Main repo (HUB-02).** `biplob123/gemma-4-e2b-grounded-health`: merged bf16 safetensors with config, tokenizer and chat template; the LoRA adapter under `adapter/`; `MODEL_CARD.md` as `README.md`; `results_bundle.json`; `lineage.json`; LICENSE and NOTICE.
- **GGUF repo (HUB-03).** `biplob123/gemma-4-e2b-grounded-health-GGUF`: Q4_K_M and Q8_0 named for Ollama's `:Q4_K_M` tag, a short card, LICENSE and NOTICE.
- **Download back (HUB-04).** Every file is downloaded from the tag and its sha256 compared; the record is committed (`docs/measurements/release_record_v2.json`).
- **Ollama (HUB-05).** `ollama run hf.co/biplob123/gemma-4-e2b-grounded-health-GGUF:Q4_K_M` answers one demo sample; the answer is recorded.
- **Visual.** A screenshot of the Hub model page with the card (`docs/assets/screenshots/hf_model_page.png`).
