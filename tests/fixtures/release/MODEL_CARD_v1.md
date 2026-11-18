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
      url: results_bundle.json
---

# Gemma 4 E2B Grounded Health

> **Research only, not medical advice.** This model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

## Summary

`google/gemma-4-E2B-it` fine-tuned with LoRA to answer a research question from a supplied abstract only, as JSON with a label, a short answer and the numbers of the sentences it relied on. It is trained on PubMedQA's artificial set and scored with EvalKit on the expert set, against its own base model and two API models. On 1000 expert questions the fine-tune (`gemma-e2b-grounded-q4`) has accuracy 0.728 [0.700, 0.755] against 0.501 [0.469, 0.532] for the base model (`gemma-e2b-base-q4`); the full table, with the judges' headline, is under Evaluation.

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

Hand audit: 1.000 of the sampled training examples passed (Tech Spec §4.6).

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

- Judge mode: **dual**. Calibration: **uncalibrated**.
- ⚠ judge not calibrated for this task (see docs/calibrate-your-task.md)

### Benchmark

EvalKit task `pubmedqa`, run `r-f0f3b432a413`. 95% bootstrap intervals in square brackets. The headline counts an answer only when its label is correct and every claim is supported by the sentences it cites.

| Model | n | Accuracy | Correct and grounded (claude-opus-5-5) | Correct and grounded (gpt-6.1-sol) | Citation validity |
|---|---|---|---|---|---|
| gemma-e2b-base-q4 | 1000 | 0.501 [0.469, 0.532] | 0.418 [0.387, 0.448] | 0.394 [0.364, 0.424] | 0.995 [0.990, 0.999] |
| gemma-e2b-grounded-q4 | 1000 | 0.728 [0.700, 0.755] | 0.388 [0.358, 0.419] | 0.330 [0.301, 0.359] | 1.000 [1.000, 1.000] |
| claude-haiku-5-5 | 1000 | 0.685 [0.656, 0.714] | 0.345 [0.316, 0.374] | 0.231 [0.205, 0.257] | 1.000 [1.000, 1.000] |
| gpt-6-luna | 1000 | 0.579 [0.548, 0.611] | 0.431 [0.401, 0.462] | 0.290 [0.262, 0.319] | 0.999 [0.997, 1.000] |

### Pre-registered comparisons

Differences with 95% paired bootstrap intervals (Tech Spec §7.4).

| Comparison | Difference |
|---|---|
| accuracy: gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 | 0.227 [0.193, 0.263] |
| headline: gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 (claude-opus-5-5) | -0.030 [-0.067, 0.007] |
| headline: gemma-e2b-grounded-q4 minus gemma-e2b-base-q4 (gpt-6.1-sol) | -0.064 [-0.101, -0.025] |

### Behaviour checks (B1–B8)

| Rule | Model | Metric | Value | n | Source |
|---|---|---|---|---|---|
| B7 | gemma-e2b-base-q4 | citation_validity | 0.995 [0.990, 0.999] | 1000 | `evalkit:citation_validity` |
| B7 | gemma-e2b-grounded-q4 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `evalkit:citation_validity` |
| B7 | claude-haiku-5-5 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `evalkit:citation_validity` |
| B7 | gpt-6-luna | citation_validity | 0.999 [0.997, 1.000] | 1000 | `evalkit:citation_validity` |

### Parity: demo weights against benchmark weights

The demo runs merged bf16 weights; the benchmark scores the Q4_K_M GGUF. On 500 dev prompts, greedy, the two agree on the label for 0.956 of prompts and cite exactly the same sentences for 0.500.

| Weights | Schema valid |
|---|---|
| bf16 | 1.000 |
| q4_k_m | 1.000 |

Judged headline difference (bf16 minus Q4_K_M), 95% interval:

| Judge | Difference |
|---|---|
| claude-opus-5-5 | 0.025 [-0.035, 0.085] |
| gpt-6.1-sol | 0.015 [-0.045, 0.075] |

### Laptop speed

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

## Licence and credits

- Weights and code: Apache-2.0. The model is a derivative of Google DeepMind's Gemma 4 E2B (Apache-2.0); the Gemma NOTICE is reproduced with the weights (see `NOTICE`).
- Data: PubMedQA (Jin et al., 2019), MIT licence, https://pubmedqa.github.io/.
- Evaluation: EvalKit 0.1.0+06591f8.

## Research only, not medical advice

These answers summarise a study abstract. They are not medical advice and must not be used to make decisions about anyone's health. Ask a clinician about your own situation.
