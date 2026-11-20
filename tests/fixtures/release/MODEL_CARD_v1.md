---
license: apache-2.0
base_model: google/gemma-4-E2B-it
datasets:
- qiaojin/PubMedQA
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- gemma
- medical
- biomedical
- pubmedqa
- grounded-generation
- citations
- lora
model-index:
- name: gemma-4-e2b-grounded-health
  results:
  - task:
      type: text-generation
      name: Grounded biomedical question answering
    dataset:
      type: qiaojin/PubMedQA
      name: PubMedQA (expert-labelled)
      config: pqa_labeled
      split: train
      revision: 9001f2853fb87cab8d220904e0de81ac6973b318
    metrics:
    - type: accuracy
      name: Label accuracy
      value: 0.728
    - type: grounded_accuracy
      name: Correct and fully grounded (claude-opus-5-5)
      value: 0.388
    - type: grounded_accuracy
      name: Correct and fully grounded (gpt-6.1-sol)
      value: 0.33
    - type: citation_validity
      name: Citation validity
      value: 1.0
    source:
      name: EvalKit run r-f0f3b432a413
      url: https://github.com/biplob819/gemma-grounded-health/blob/main/docs/results/results_bundle.json
---

# Gemma 4 E2B Grounded Health

> **Research only, not medical advice.** This model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

## Summary

`google/gemma-4-E2B-it` fine-tuned with LoRA to answer a research question from a supplied abstract only, as JSON with a label, a short answer and the numbers of the sentences it relied on. It is trained on PubMedQA's artificial set and scored with EvalKit on the expert set, against its own base model and two API models. On 1000 expert questions the fine-tune (`gemma-e2b-grounded-q4`) has accuracy 0.728 [0.700, 0.755] against 0.501 [0.469, 0.532] for the base model (`gemma-e2b-base-q4`); the full table, with the judges' headline, is under Evaluation. Research only, not medical advice.

## Intended use

- Answering a research question about one biomedical abstract, from that abstract only, with a yes, no or maybe label, a short answer and the numbers of the sentences that support it.
- Studying grounded answering and citation behaviour in small open models, offline, on a laptop (GGUF) or a GPU (merged weights).
- Reproducing and extending the benchmark with EvalKit.

## Out-of-scope uses

- Clinical use of any kind: diagnosis, treatment, triage or dosing.
- Personal medical advice, for the user or anyone else.
- Languages other than English.
- Sources other than research abstracts: full papers, guidelines, web pages, patient records or open-ended chat without a supplied abstract.

## Training data

PubMedQA `qiaojin/PubMedQA` at revision `9001f2853fb87cab8d220904e0de81ac6973b318`: the artificial set (PQA-A) for training and dev, never the expert set (PQA-L), which is the benchmark. Rows overlapping PQA-L by PMID, question or context were removed (overlap_with_pqa_l:context 0, overlap_with_pqa_l:pmid 0, overlap_with_pqa_l:question 0).

| Split | Rows by reference label |
|---|---|
| dev_eval | no 43, yes 457 |
| offscope_eval | maybe 20 |
| train_pool | no 14942, yes 193827 |
| val | no 140, yes 1860 |

Targets: 8436 training and 843 validation examples from 35538 scored rows. Each answer cites only sentences that support it (support ≥ 0.500, entailment ≥ 0.500, at most 4 citations, at most 80 words).

Label mix:

| Split | Examples by kind |
|---|---|
| train | insufficient 844, no 3003, offscope 84, yes 4505 |
| val | insufficient 84, no 300, offscope 8, yes 451 |

Filters and their drop counts:

| Filter | Dropped |
|---|---|
| no:no_supported_sentence | 11779 |
| sentence:citation_cap | 233 |
| sentence:low_entailment | 27090 |
| sentence:low_support | 19682 |
| sentence:no_content_words | 15 |
| sentence:recommendation | 4422 |
| sentence:zero_overlap | 737 |
| yes:no_supported_sentence | 14270 |
| yes:over_80_words | 1 |

Hand audit (Tech Spec §4.6), v1 (N1.3, AUD-R2): 50 sampled training examples, 1.000 judged ok overall; the audit passed its bar.

## Training procedure

|  | Value |
|---|---|
| Base model | `google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37` |
| Method | QLoRA (4-bit base) with unsloth |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate, schedule, warm-up | 0.0002, cosine, 0.03 |
| Batch | 4 × 4 gradient accumulation |
| Max sequence length | 1024 |
| Epochs | 1.0 |
| Loss | completion only (prompt masked) |
| Seed | 7 |
| GPU | Tesla T4 (Kaggle kernel version 1) |
| Wall-clock | 6525.8 s |
| Peak VRAM | 9.84 GB |
| Steps (selected) | 527 (527) |
| Repo commit | `16ba41f43aa9b8a3d21219a504f1bff7938e7f0e` |

Dev checks on 500 held-out PQA-A prompts (deterministic, no judge): schema valid 1.000 constrained and 1.000 unconstrained, citation valid 1.000, label accuracy no 0.837, yes 0.947, mean answer length 15.1 words.

## Evaluation

- Judge mode: **dual**.
- Calibration of v1: **uncalibrated**. The stamp covers the judges v1's pre-registered comparisons rely on (`claude-opus-5-5` and `gpt-6.1-sol`) and is given only when each is calibrated (OWNER D-59). Per judge, as the results bundle records it:
  - claude-opus-5-5: never stamped calibrated: the reference labels are its own model's (Claude Opus 5.5), so its held-out kappa 0.397 [0.263, 0.511] on 231 pubmedqa claims is the same model agreeing with itself (OWNER D-60), and its lower bound is under the 0.4 bar; reported only, its numbers stay published.
  - gpt-6.1-sol: calibrated, held-out kappa 0.598 [0.468, 0.711] on 231 pubmedqa claims, lower bound at least 0.4 (OWNER D-59): agreement with a Claude Opus 5.5 reference labeller (a model's labels, not a human expert's; OWNER D-60). Gemma-family check (40 claims from main_v2): agreement 0.950, kappa 0.893 [0.722, 1.000].
- Reference labeller consistency: Claude Opus 5.5 relabelled 20 answers (88 claims) under `configs/labeller_prompt.md`: 81 labels identical, kappa 0.807 [0.662, 0.928] (OWNER D-60).
- ⚠ judge not calibrated for this task (see docs/calibrate-your-task.md)
- `offscope`: exempt: task ggh_offscope is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.
- `probe_original`: exempt: task ggh_probe_original is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.
- `probe_paraphrase`: exempt: task ggh_probe_paraphrase is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.

### Benchmark

EvalKit task `pubmedqa`, run `r-f0f3b432a413`. 95% bootstrap intervals in square brackets. The headline counts an answer only when its label is correct and every claim is supported by the sentences it cites.

| Model | n | Accuracy | Correct and grounded (claude-opus-5-5) | Correct and grounded (gpt-6.1-sol) | Citation validity |
|---|---|---|---|---|---|
| gemma-e2b-base-q4 | 1000 | 0.501 [0.469, 0.532] | 0.418 [0.387, 0.448] | 0.394 [0.364, 0.424] | 0.995 [0.990, 0.999] |
| gemma-e2b-grounded-q4 | 1000 | 0.728 [0.700, 0.755] | 0.388 [0.358, 0.419] | 0.330 [0.301, 0.359] | 1.000 [1.000, 1.000] |
| claude-haiku-5-5 | 1000 | 0.685 [0.656, 0.714] | 0.345 [0.316, 0.374] | 0.231 [0.205, 0.257] | 1.000 [1.000, 1.000] |
| gpt-6-luna | 1000 | 0.579 [0.548, 0.611] | 0.431 [0.401, 0.462] | 0.290 [0.262, 0.319] | 0.999 [0.997, 1.000] |

### Pre-registered comparisons

From `configs/eval_plan.yaml` (final). Differences with 95% paired bootstrap intervals on shared cases; a gap is separated only under the plan's Holm-corrected rule, otherwise it is a tie (Tech Spec §7.4).

| Comparison | Outcome | Judge | Difference | p | Verdict |
|---|---|---|---|---|---|
| gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 | headline | claude-opus-5-5 | -0.030 [-0.067, 0.007] | 0.119 | tie (not separated) |
| gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 | headline | gpt-6.1-sol | -0.064 [-0.101, -0.025] | 0.001 | gemma-e2b-base-q4 better |
| gemma-e2b-grounded-q4 minus claude-haiku-5-5 | headline | claude-opus-5-5 | 0.043 [0.005, 0.081] | 0.026 | tie (not separated) |
| gemma-e2b-grounded-q4 minus claude-haiku-5-5 | headline | gpt-6.1-sol | 0.099 [0.063, 0.136] | 0.000 | gemma-e2b-grounded-q4 better |
| gemma-e2b-grounded-q4 minus gpt-6-luna | headline | claude-opus-5-5 | -0.043 [-0.082, -0.005] | 0.027 | tie (not separated) |
| gemma-e2b-grounded-q4 minus gpt-6-luna | headline | gpt-6.1-sol | 0.040 [0.002, 0.079] | 0.044 | gemma-e2b-grounded-q4 better |

### Secondary: label accuracy

Label accuracy (yes, no or maybe matches the reference), secondary: differences with 95% paired bootstrap intervals, separated under EvalKit's run-wide Holm rule.

| Comparison | Accuracy difference | Separated |
|---|---|---|
| gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 | 0.227 [0.193, 0.263] | yes |
| gemma-e2b-grounded-q4 minus claude-haiku-5-5 | 0.043 [0.011, 0.075] | yes |
| gemma-e2b-grounded-q4 minus gpt-6-luna | 0.149 [0.112, 0.187] | yes |

### Behaviour checks (B1–B8)

| Rule | Model | Metric | Value | n | Source |
|---|---|---|---|---|---|
| B1 | gemma-e2b-base-q4 | schema_failures_or_zero_citations | 0.005 | 1000 | `ggh:b1` |
| B2 | gemma-e2b-base-q4 | headline:claude-opus-5-5 | 0.418 [0.387, 0.448] | 1000 | `evalkit:headline` |
| B2 | gemma-e2b-base-q4 | groundedness_mean:claude-opus-5-5 | 0.863 [0.844, 0.881] | 1000 | `evalkit:groundedness` |
| B2 | gemma-e2b-base-q4 | headline:gpt-6.1-sol | 0.394 [0.364, 0.424] | 1000 | `evalkit:headline` |
| B2 | gemma-e2b-base-q4 | groundedness_mean:gpt-6.1-sol | 0.819 [0.798, 0.839] | 1000 | `evalkit:groundedness` |
| B3 | gemma-e2b-base-q4 | maybe_recall | 0.491 [0.398, 0.586] | 110 | `ggh:b3` |
| B3 | gemma-e2b-base-q4 | maybe_precision | 0.127 [0.095, 0.161] | 424 | `ggh:b3` |
| B4 | gemma-e2b-base-q4 | over_80_words_share | 0.013 | 1000 | `ggh:b4` |
| B6 | gemma-e2b-base-q4 | unsupported_claim_share:claude-opus-5-5 | 0.117 | 2342 | `evalkit:verdicts` |
| B6 | gemma-e2b-base-q4 | unsupported_claim_share:gpt-6.1-sol | 0.156 | 2342 | `evalkit:verdicts` |
| B7 | gemma-e2b-base-q4 | citation_validity | 0.995 [0.990, 0.999] | 1000 | `evalkit:citation_valid` |
| B7 | gemma-e2b-base-q4 | citation_precision:claude-opus-5-5 | 0.725 [0.698, 0.752] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-base-q4 | citation_precision:gpt-6.1-sol | 0.746 [0.721, 0.771] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-base-q4 | aim_citation_rate:claude-opus-5-5 | 0.057 [0.043, 0.072] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-base-q4 | aim_citation_rate:gpt-6.1-sol | 0.142 [0.120, 0.164] | 1000 | `evalkit:citation_support` |
| B1 | gemma-e2b-grounded-q4 | schema_failures_or_zero_citations | 0.000 | 1000 | `ggh:b1` |
| B2 | gemma-e2b-grounded-q4 | headline:claude-opus-5-5 | 0.388 [0.358, 0.419] | 1000 | `evalkit:headline` |
| B2 | gemma-e2b-grounded-q4 | groundedness_mean:claude-opus-5-5 | 0.519 [0.489, 0.550] | 1000 | `evalkit:groundedness` |
| B2 | gemma-e2b-grounded-q4 | headline:gpt-6.1-sol | 0.330 [0.301, 0.359] | 1000 | `evalkit:headline` |
| B2 | gemma-e2b-grounded-q4 | groundedness_mean:gpt-6.1-sol | 0.444 [0.414, 0.475] | 1000 | `evalkit:groundedness` |
| B3 | gemma-e2b-grounded-q4 | maybe_recall | 0.045 [0.010, 0.086] | 110 | `ggh:b3` |
| B3 | gemma-e2b-grounded-q4 | maybe_precision | 0.227 [0.059, 0.409] | 22 | `ggh:b3` |
| B4 | gemma-e2b-grounded-q4 | over_80_words_share | 0.000 | 1000 | `ggh:b4` |
| B6 | gemma-e2b-grounded-q4 | unsupported_claim_share:claude-opus-5-5 | 0.447 | 1174 | `evalkit:verdicts` |
| B6 | gemma-e2b-grounded-q4 | unsupported_claim_share:gpt-6.1-sol | 0.524 | 1174 | `evalkit:verdicts` |
| B7 | gemma-e2b-grounded-q4 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `evalkit:citation_valid` |
| B7 | gemma-e2b-grounded-q4 | citation_precision:claude-opus-5-5 | 0.338 [0.319, 0.357] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-grounded-q4 | citation_precision:gpt-6.1-sol | 0.359 [0.340, 0.379] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-grounded-q4 | aim_citation_rate:claude-opus-5-5 | 0.557 [0.525, 0.588] | 1000 | `evalkit:citation_support` |
| B7 | gemma-e2b-grounded-q4 | aim_citation_rate:gpt-6.1-sol | 0.567 [0.536, 0.598] | 1000 | `evalkit:citation_support` |
| B1 | claude-haiku-5-5 | schema_failures_or_zero_citations | 0.000 | 1000 | `ggh:b1` |
| B2 | claude-haiku-5-5 | headline:claude-opus-5-5 | 0.345 [0.316, 0.374] | 1000 | `evalkit:headline` |
| B2 | claude-haiku-5-5 | groundedness_mean:claude-opus-5-5 | 0.817 [0.802, 0.831] | 1000 | `evalkit:groundedness` |
| B2 | claude-haiku-5-5 | headline:gpt-6.1-sol | 0.231 [0.205, 0.257] | 1000 | `evalkit:headline` |
| B2 | claude-haiku-5-5 | groundedness_mean:gpt-6.1-sol | 0.737 [0.722, 0.753] | 1000 | `evalkit:groundedness` |
| B3 | claude-haiku-5-5 | maybe_recall | 0.527 [0.430, 0.620] | 110 | `ggh:b3` |
| B3 | claude-haiku-5-5 | maybe_precision | 0.205 [0.158, 0.254] | 283 | `ggh:b3` |
| B4 | claude-haiku-5-5 | over_80_words_share | 0.143 | 1000 | `ggh:b4` |
| B6 | claude-haiku-5-5 | unsupported_claim_share:claude-opus-5-5 | 0.159 | 5138 | `evalkit:verdicts` |
| B6 | claude-haiku-5-5 | unsupported_claim_share:gpt-6.1-sol | 0.238 | 5139 | `evalkit:verdicts` |
| B7 | claude-haiku-5-5 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `evalkit:citation_valid` |
| B7 | claude-haiku-5-5 | citation_precision:claude-opus-5-5 | 0.848 [0.836, 0.860] | 1000 | `evalkit:citation_support` |
| B7 | claude-haiku-5-5 | citation_precision:gpt-6.1-sol | 0.882 [0.872, 0.893] | 1000 | `evalkit:citation_support` |
| B7 | claude-haiku-5-5 | aim_citation_rate:claude-opus-5-5 | 0.186 [0.162, 0.210] | 1000 | `evalkit:citation_support` |
| B7 | claude-haiku-5-5 | aim_citation_rate:gpt-6.1-sol | 0.225 [0.199, 0.251] | 1000 | `evalkit:citation_support` |
| B1 | gpt-6-luna | schema_failures_or_zero_citations | 0.001 | 1000 | `ggh:b1` |
| B2 | gpt-6-luna | headline:claude-opus-5-5 | 0.431 [0.401, 0.462] | 1000 | `evalkit:headline` |
| B2 | gpt-6-luna | groundedness_mean:claude-opus-5-5 | 0.852 [0.837, 0.867] | 1000 | `evalkit:groundedness` |
| B2 | gpt-6-luna | headline:gpt-6.1-sol | 0.290 [0.262, 0.319] | 1000 | `evalkit:headline` |
| B2 | gpt-6-luna | groundedness_mean:gpt-6.1-sol | 0.768 [0.750, 0.786] | 1000 | `evalkit:groundedness` |
| B3 | gpt-6-luna | maybe_recall | 0.545 [0.450, 0.640] | 110 | `ggh:b3` |
| B3 | gpt-6-luna | maybe_precision | 0.152 [0.117, 0.188] | 394 | `ggh:b3` |
| B4 | gpt-6-luna | over_80_words_share | 0.000 | 1000 | `ggh:b4` |
| B6 | gpt-6-luna | unsupported_claim_share:claude-opus-5-5 | 0.127 | 3734 | `evalkit:verdicts` |
| B6 | gpt-6-luna | unsupported_claim_share:gpt-6.1-sol | 0.207 | 3734 | `evalkit:verdicts` |
| B7 | gpt-6-luna | citation_validity | 0.999 [0.997, 1.000] | 1000 | `evalkit:citation_valid` |
| B7 | gpt-6-luna | citation_precision:claude-opus-5-5 | 0.865 [0.851, 0.879] | 1000 | `evalkit:citation_support` |
| B7 | gpt-6-luna | citation_precision:gpt-6.1-sol | 0.894 [0.881, 0.905] | 1000 | `evalkit:citation_support` |
| B7 | gpt-6-luna | aim_citation_rate:claude-opus-5-5 | 0.094 [0.076, 0.113] | 1000 | `evalkit:citation_support` |
| B7 | gpt-6-luna | aim_citation_rate:gpt-6.1-sol | 0.116 [0.096, 0.136] | 1000 | `evalkit:citation_support` |
| B5 | gemma-e2b-base-q4 | sample_read_ok_share | 0.300 | 20 | `model_review` |
| B5 | gemma-e2b-grounded-q4 | sample_read_ok_share | 0.450 | 20 | `model_review` |
| B5 | claude-haiku-5-5 | sample_read_ok_share | 0.550 | 20 | `model_review` |
| B5 | gpt-6-luna | sample_read_ok_share | 0.950 | 20 | `model_review` |
| B8 | gemma-e2b-base-q4 | presort_flagged_share | 0.100 | 20 | `ggh:b8` |
| B8 | gemma-e2b-base-q4 | gives_advice_count | 0 | 20 | `model_review` |
| B8 | gemma-e2b-grounded-q4 | presort_flagged_share | 0.000 | 20 | `ggh:b8` |
| B8 | gemma-e2b-grounded-q4 | gives_advice_count | 2 | 20 | `model_review` |
| B8 | claude-haiku-5-5 | presort_flagged_share | 0.250 | 20 | `ggh:b8` |
| B8 | claude-haiku-5-5 | gives_advice_count | 1 | 20 | `model_review` |
| B8 | gpt-6-luna | presort_flagged_share | 0.400 | 20 | `ggh:b8` |
| B8 | gpt-6-luna | gives_advice_count | 0 | 20 | `model_review` |

### Parity: demo weights against benchmark weights

The demo runs merged bf16 weights; the benchmark scores the Q4_K_M GGUF. On 500 dev prompts, greedy, the two agree on the label for 0.956 of prompts and cite exactly the same sentences for 0.500.

| Weights | Schema valid |
|---|---|
| bf16 | 1.000 |
| q4_k_m | 1.000 |

Judged headline difference (bf16 minus Q4_K_M) on 200 PubMedQA expert questions from the main run, 95% interval:

| Judge | Difference |
|---|---|
| claude-opus-5-5 | 0.025 [-0.035, 0.085] |
| gpt-6.1-sol | 0.015 [-0.045, 0.075] |

### Laptop speed

v1's GGUF (`gemma-e2b-grounded-q4`), variant `ft-Q4_K_M` in `docs/measurements/laptop_bench.json`, served by llama.cpp on a laptop:

|  | Value |
|---|---|
| Machine | Apple M3 Max, 64 GB, macOS 26.5.2, ac power |
| llama.cpp | b12421 |
| Weights | Q4_K_M, 3269.1 MB |
| Prompts | 50 × 3 repetitions |
| Prompt processing (p50) | 3829.9 tokens/s |
| Generation (p50 / p95) | 105.1 / 107.6 tokens/s |
| Time to first token (p50) | 137 ms |
| Answer latency (p50 / p95) | 604 / 759 ms |
| Peak memory | 4462.2 MB |
| llama-bench pp512 | 7018.0 tokens/s |
| llama-bench tg128 | 124.9 tokens/s |

## Limits

- One benchmark's question style: PubMedQA questions are mostly titles turned into yes or no questions, so other phrasings may do worse.
- English only, and abstracts only.
- The `maybe` label was learned from 928 constructed examples (an abstract with its deciding sentences removed, OWNER D-15), not from real maybe-labelled questions.
- Contamination probe `ggh_probe_original` (run `r-f98be6681636`): accuracy gemma-e2b-base-q4 0.560 [0.460, 0.660], gemma-e2b-grounded-q4 0.790 [0.710, 0.860].
- Contamination probe `ggh_probe_paraphrase` (run `r-5bd60094cf2d`): accuracy gemma-e2b-base-q4 0.590 [0.490, 0.690], gemma-e2b-grounded-q4 0.840 [0.760, 0.910].
- Paraphrase drop (original minus paraphrased accuracy, 100 questions): -0.050 [-0.100, 0.000]; contamination signal: no.

## Licence and credits

- Weights and code: Apache-2.0 (`LICENSE`). The repo's `NOTICE` is shipped with the weights.
- Base model: Google DeepMind's Gemma 4 E2B (`google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37`). Its model card declares `license: apache-2.0` and links its terms at https://ai.google.dev/gemma/docs/gemma_4_license. The snapshot at that revision ships no NOTICE file of its own, so the repo's `NOTICE` carries the Gemma credit. Read on 2026-11-20 from the model card at that revision (sha256 `3e0608a6be80e3eb040a5aa6c76707809503deb85be6360184a118fb6156526c`), recorded in `docs/licence_check.json`. Re-read on 2026-11-20: that link resolves to the Apache License 2.0 text (https://gemma.ai/resources/apache-2/), and the Gemma Terms of Use (https://gemma.ai/resources/terms/, last modified 2026-04-01) say they do not cover Gemma 4. Although those terms do not cover Gemma 4, this release voluntarily keeps to Google's Gemma Prohibited Use Policy (https://gemma.ai/resources/prohibited-use/), which names no version and bars, among other things, the unlicensed practice of medicine and automated decisions affecting people's health; the model is for research only and is never to be used for either.
- Data: PubMedQA (Jin et al., 2019), MIT licence, https://pubmedqa.github.io/.
- Evaluation: EvalKit 06591f8de9b1, ee5969978c92 (MIT licence, https://github.com/biplob819/evalkit).

## Research only, not medical advice

These answers summarise a study abstract. They are not medical advice and must not be used to make decisions about anyone's health. Ask a clinician about your own situation.
