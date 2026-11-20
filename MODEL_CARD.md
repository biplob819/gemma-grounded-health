---
license: apache-2.0
base_model: google/gemma-4-E2B-it
base_model_relation: finetune
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
      value: 0.678
    - type: grounded_accuracy
      name: Correct and fully grounded (gpt-6.1-sol)
      value: 0.329
    - type: citation_validity
      name: Citation validity
      value: 1.0
    - type: citation_precision
      name: Citation precision (gpt-6.1-sol)
      value: 0.7359223300970874
    - type: aim_citation_rate
      name: Aim-citation rate, lower is better (gpt-6.1-sol)
      value: 0.08433734939759036
    source:
      name: EvalKit run r-715fc213a8a5
      url: https://github.com/biplob819/gemma-grounded-health/blob/main/docs/results/results_bundle.json
---

# Gemma 4 E2B Grounded Health

> **Research only, not medical advice.** This model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

## Summary

`google/gemma-4-E2B-it` fine-tuned with LoRA to answer a research question from a supplied abstract only, as JSON with a label, a short answer and the numbers of the sentences it relied on. It is trained on PubMedQA's artificial set and scored with EvalKit on the expert set, against its own base model and v1 (`gemma-e2b-grounded-q4`), all scored in one run with the same claim extractor and judge set. The published v1 numbers are shown beside them, labelled with their own extractor and judges, and not compared. On 1000 expert questions the fine-tune (`gemma-e2b-grounded-v2-q4`) has accuracy 0.678 [0.649, 0.706] against 0.501 [0.469, 0.532] for the base model (`gemma-e2b-base-q4`) and 0.728 [0.700, 0.755] for `gemma-e2b-grounded-q4` (v2 is separated lower); the full table, with the judges' headline, is under Evaluation. Against v1 (`gemma-e2b-grounded-q4`), pre-registered: a tie on correct and fully grounded, better on citation precision, better on aim-citation rate. Against the base model (`gemma-e2b-base-q4`), pre-registered: worse on correct and fully grounded; citation precision and aim-citation rate depend on how the judge applies the citation check's aim rule, so v2 is not claimed to beat it on citations (OWNER D-62). Research only, not medical advice.

## Intended use

- Answering a research question about one biomedical abstract, from that abstract only, with a yes, no or maybe label, a short answer and the numbers of the sentences that support it.
- Studying grounded answering and citation behaviour in small open models, offline, on a laptop (GGUF) or a GPU (merged weights).
- Reproducing and extending the benchmark with EvalKit.

## Out-of-scope uses

- Clinical use of any kind: diagnosis, treatment, triage or dosing.
- Personal medical advice, for the user or anyone else.
- Languages other than English.
- Sources other than research abstracts: full papers, guidelines, web pages, patient records or open-ended chat without a supplied abstract.

## How to use

The model answers as JSON only when it gets EvalKit's PubMedQA prompt, the prompt it was trained and benchmarked on: the question and the abstract split into numbered sentences. A bare question gets free text. Send this as the user message (Gemma's chat template, thinking off), with the real question and one numbered line per sentence:

```text
You are answering a research question using only the numbered sentences from a
scientific abstract below. This is a research benchmark, not medical advice.

Question: <question>

Abstract sentences:
[1] <sentence 1>
[2] <sentence 2>
[3] <sentence 3>

Reply with JSON only, in this shape:
{"label": one of yes / no / maybe, "answer": "two to four sentences", "citations": [sentence numbers]}

Rules:
- Base every statement on the sentences above, not on outside knowledge.
- Cite the sentence numbers (1 to 3) that support your answer.
- Choose "maybe" when the sentences do not settle the question.
```

The reply is one JSON object, for example `{"label": "no", "answer": "...", "citations": [2]}`. The benchmark decoded greedily (temperature 0) with thinking off. Offline, the Q4_K_M GGUF runs with `ollama run --think=false hf.co/biplob123/gemma-4-e2b-grounded-health-GGUF:Q4_K_M`; the repository's `ggh space --local` demo and `ggh serve` build the prompt for you.

## Training data

PubMedQA `qiaojin/PubMedQA` at revision `9001f2853fb87cab8d220904e0de81ac6973b318`: the artificial set (PQA-A) for training and dev, never the expert set (PQA-L), which is the benchmark. Rows overlapping PQA-L by PMID, question or context were removed (overlap_with_pqa_l:context 0, overlap_with_pqa_l:pmid 0, overlap_with_pqa_l:question 0).

| Split | Rows by reference label |
|---|---|
| dev_eval | no 43, yes 457 |
| offscope_eval | maybe 20 |
| train_pool | no 14942, yes 193827 |
| val | no 140, yes 1860 |

Targets: 2722 training and 272 validation examples from 39252 scored rows. Citations are chosen by rule, not checked by hand: each cited sentence must on its own reach a minimum entailment of the claim under the pinned NLI model (a higher one unless it reports a result), the cited set must entail it (≥ 0.500) and share its content words (≥ 0.500), with at most 4 citations and at most 80 words. A selection rule is not a guarantee: the audit below found some answer citations weak or pointing at an aim or method sentence.

Label mix:

| Split | Examples by kind |
|---|---|
| train | insufficient 272, no 969, offscope 27, yes 1454 |
| val | insufficient 27, no 97, offscope 3, yes 145 |

Filters and their drop counts:

| Filter | Dropped |
|---|---|
| no:no_supported_sentence | 14016 |
| sentence:citation_cap | 9 |
| sentence:low_entailment | 117 |
| sentence:low_support | 3377 |
| sentence:no_content_words | 20 |
| sentence:no_supporting_sentence | 57445 |
| sentence:recommendation | 4834 |
| yes:no_supported_sentence | 22162 |
| yes:over_80_words | 1 |

Hand audit (Tech Spec §4.6), v2.2 round 3 (AUD-R4, blind Claude reviewer, OWNER D-47, D-50): 50 sampled training examples, 0.980 judged ok overall; the audit did not pass its bar (the failed checks are in `docs/measurements/audit_v2_result.json`), and the release trained on these targets with the miss disclosed in `DATA_CARD.md`.

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
| Epochs | 2.0 |
| Loss | completion only (prompt masked) |
| Seed | 7 |
| GPU | Tesla T4 (Kaggle kernel version 1) |
| Wall-clock | 4551.2 s |
| Peak VRAM | 9.80 GB |
| Steps (selected) | 340 (340) |
| Repo commit | `8ff61b977580700c592ee2c8443d46570e0126f1` |

Dev checks on 500 held-out PQA-A prompts (deterministic, no judge): schema valid 1.000 constrained and 1.000 unconstrained, citation valid 1.000, label accuracy no 0.860, yes 0.937, mean answer length 13.7 words.

## Evaluation

- Judge mode: **single**.
- v2 scoring: claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone.
- Calibration of v2: **calibrated**. The stamp covers the judges v2's pre-registered comparisons rely on (`gpt-6.1-sol`) and is given only when each is calibrated (OWNER D-59). Per judge, as the results bundle records it:
  - gpt-6.1-sol: calibrated, held-out kappa 0.598 [0.468, 0.711] on 231 pubmedqa claims, lower bound at least 0.4 (OWNER D-59): agreement with a Claude Opus 5.5 reference labeller (a model's labels, not a human expert's; OWNER D-60). Gemma-family check (40 claims from main_v2): agreement 0.950, kappa 0.893 [0.722, 1.000].
  - claude-opus-5-5: never stamped calibrated: the reference labels are its own model's (Claude Opus 5.5), so its held-out kappa 0.397 [0.263, 0.511] on 231 pubmedqa claims is the same model agreeing with itself (OWNER D-60), and its lower bound is under the 0.4 bar; reported only, its numbers stay published (not a judge of v2's runs).
- Reference labeller consistency: Claude Opus 5.5 relabelled 20 answers (88 claims) under `configs/labeller_prompt.md`: 81 labels identical, kappa 0.807 [0.662, 0.928] (OWNER D-60).
- ⚠ single judge: self-preference not measured
- The committed EvalKit summaries keep EvalKit's own per-run stamp ("judge not calibrated for this task"): they are never rewritten (OWNER D-59). The release's stamp above is GGH's, read from EvalKit's calibration run.
- `offscope_v2`: exempt: task ggh_offscope is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.
- `probe_original_v2`: exempt: task ggh_probe_original is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.
- `probe_paraphrase_v2`: exempt: task ggh_probe_paraphrase is not pubmedqa, so the pubmedqa calibration cannot cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp.

### Benchmark

v2: EvalKit task `pubmedqa`, run `r-715fc213a8a5`. Every model in this table was scored in this run: claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone. 95% bootstrap intervals in square brackets. The headline counts an answer only when its label is correct and every claim is supported by the sentences it cites.

| Model | n | Accuracy | Correct and grounded (gpt-6.1-sol) | Citation precision (gpt-6.1-sol) | Aim-citation rate (gpt-6.1-sol, lower is better) |
|---|---|---|---|---|---|
| gemma-e2b-base-q4 | 1000 | 0.501 [0.469, 0.532] | 0.404 [0.373, 0.434] | 0.747 [0.722, 0.772] | 0.138 [0.117, 0.161] |
| gemma-e2b-grounded-q4 | 1000 | 0.728 [0.700, 0.755] | 0.331 [0.302, 0.361] | 0.362 [0.343, 0.382] | 0.560 [0.529, 0.590] |
| **gemma-e2b-grounded-v2-q4** | 1000 | 0.678 [0.649, 0.706] | 0.329 [0.300, 0.358] | 0.736 [0.708, 0.763] | 0.084 [0.067, 0.102] |

### Published v1 numbers (not compared with v2)

These are v1's numbers as published under `configs/eval_plan.yaml`; they stand as they are and are shown for reference, not as a v2 comparison. They were produced with a different claim extractor and a different judge set from v2's run (claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone), so the two sets of numbers are not comparable. As answering models, `claude-haiku-5-5` and `gpt-6-luna` were not re-scored for v2, so v2 is not compared with them. `gemma-e2b-base-q4` and `gemma-e2b-grounded-q4` appear in both v2's table and here; each table's numbers come from its own run.

Run `main` (`r-f0f3b432a413`), claims extracted by `claude-haiku-5-5`, judged by `claude-opus-5-5` and `gpt-6.1-sol`:

| Model | n | Accuracy | Correct and grounded (claude-opus-5-5) | Correct and grounded (gpt-6.1-sol) |
|---|---|---|---|---|
| gemma-e2b-base-q4 | 1000 | 0.501 [0.469, 0.532] | 0.418 [0.387, 0.448] | 0.394 [0.364, 0.424] |
| **gemma-e2b-grounded-q4** | 1000 | 0.728 [0.700, 0.755] | 0.388 [0.358, 0.419] | 0.330 [0.301, 0.359] |
| claude-haiku-5-5 | 1000 | 0.685 [0.656, 0.714] | 0.345 [0.316, 0.374] | 0.231 [0.205, 0.257] |
| gpt-6-luna | 1000 | 0.579 [0.548, 0.611] | 0.431 [0.401, 0.462] | 0.290 [0.262, 0.319] |

Run `main_citation` (`r-edfcba37159e`), claims extracted by `claude-haiku-5-5`, judged by `claude-opus-5-5` and `gpt-6.1-sol`:

| Model | n | Accuracy | Correct and grounded (claude-opus-5-5) | Correct and grounded (gpt-6.1-sol) | Citation precision (claude-opus-5-5) | Citation precision (gpt-6.1-sol) | Aim-citation rate (claude-opus-5-5, lower is better) | Aim-citation rate (gpt-6.1-sol, lower is better) |
|---|---|---|---|---|---|---|---|---|
| gemma-e2b-base-q4 | 1000 | 0.501 [0.469, 0.532] | 0.418 [0.387, 0.448] | 0.394 [0.364, 0.424] | 0.725 [0.698, 0.752] | 0.746 [0.721, 0.771] | 0.057 [0.043, 0.072] | 0.142 [0.120, 0.164] |
| **gemma-e2b-grounded-q4** | 1000 | 0.728 [0.700, 0.755] | 0.388 [0.358, 0.419] | 0.330 [0.301, 0.359] | 0.338 [0.319, 0.357] | 0.359 [0.340, 0.379] | 0.557 [0.525, 0.588] | 0.567 [0.536, 0.598] |
| claude-haiku-5-5 | 1000 | 0.685 [0.656, 0.714] | 0.345 [0.316, 0.374] | 0.231 [0.205, 0.257] | 0.848 [0.836, 0.860] | 0.882 [0.872, 0.893] | 0.186 [0.162, 0.210] | 0.225 [0.199, 0.251] |
| gpt-6-luna | 1000 | 0.579 [0.548, 0.611] | 0.431 [0.401, 0.462] | 0.290 [0.262, 0.319] | 0.865 [0.851, 0.879] | 0.894 [0.881, 0.905] | 0.094 [0.076, 0.113] | 0.116 [0.096, 0.136] |

### Pre-registered comparisons

From `configs/eval_plan_v2.yaml` (final). Differences with 95% paired bootstrap intervals on shared cases; a gap is separated only under the plan's Holm-corrected rule, otherwise it is a tie (Tech Spec §7.4). Every model on both sides was scored the same way: run `main_v2`, claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone. Holm runs over the plan's 6 tests.

| Comparison | Outcome | Judge | Difference | p | Verdict |
|---|---|---|---|---|---|
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | headline | gpt-6.1-sol | -0.002 [-0.034, 0.030] | 0.927 | tie (not separated) |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | citation_precision | gpt-6.1-sol | 0.374 [0.344, 0.403] | <0.001 | gemma-e2b-grounded-v2-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | aim_citation_rate (lower is better) | gpt-6.1-sol | -0.476 [-0.508, -0.443] | <0.001 | gemma-e2b-grounded-v2-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | headline | gpt-6.1-sol | -0.075 [-0.113, -0.037] | <0.001 | gemma-e2b-base-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | citation_precision | gpt-6.1-sol | -0.011 [-0.046, 0.023] | 0.504 | tie (not separated; depends on the aim rule, OWNER D-62) |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | aim_citation_rate (lower is better) | gpt-6.1-sol | -0.054 [-0.080, -0.028] | <0.001 | gemma-e2b-grounded-v2-q4 better (depends on the aim rule, OWNER D-62) |

#### Citation results and the aim rule (OWNER D-62)

The v2-vs-base citation results depend on how the judge applies the citation check's aim rule (EvalKit citation rubric `be34ebe577bc96ae`: a sentence that only states an aim does not establish a claim). The judge applies it even to claims that only restate the study's aim, while the claim judge cites that sentence as evidence. This is not a code bug, and the pre-registered numbers above stay as measured (OWNER D-62).

A post-hoc count made after the results were read (`docs/measurements/aim_rule_check.json`) treats such aim sentences as supporting: under `gpt-6.1-sol` it moves `gemma-e2b-base-q4`'s aim-citation rate from 0.138 to 0.054 and its citation precision from 0.747 to 0.783, and `gemma-e2b-base-q4` is then better than `gemma-e2b-grounded-v2-q4` on citation precision and aim-citation rate. So this card does not claim that v2 beats base on citations. The results of `gemma-e2b-grounded-v2-q4` against `gemma-e2b-grounded-q4` hold under both counts.

| Comparison | Outcome | Judge | Pre-registered count | Post-hoc count |
|---|---|---|---|---|
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | headline | gpt-6.1-sol | -0.002 [-0.034, 0.030], tie (not separated) | -0.002 [-0.034, 0.030], tie (not separated) |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | citation_precision | gpt-6.1-sol | 0.374 [0.344, 0.403], gemma-e2b-grounded-v2-q4 better | 0.372 [0.341, 0.401], gemma-e2b-grounded-v2-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | aim_citation_rate (lower is better) | gpt-6.1-sol | -0.476 [-0.508, -0.443], gemma-e2b-grounded-v2-q4 better | -0.471 [-0.503, -0.438], gemma-e2b-grounded-v2-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | headline | gpt-6.1-sol | -0.075 [-0.113, -0.037], gemma-e2b-base-q4 better | -0.075 [-0.113, -0.037], gemma-e2b-base-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | citation_precision | gpt-6.1-sol | -0.011 [-0.046, 0.023], tie (not separated; depends on the aim rule, OWNER D-62) | -0.047 [-0.081, -0.012], gemma-e2b-base-q4 better |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | aim_citation_rate (lower is better) | gpt-6.1-sol | -0.054 [-0.080, -0.028], gemma-e2b-grounded-v2-q4 better (depends on the aim rule, OWNER D-62) | 0.030 [0.008, 0.051], gemma-e2b-base-q4 better |

### Secondary: label accuracy

Label accuracy (yes, no or maybe matches the reference), secondary: differences with 95% paired bootstrap intervals, separated under EvalKit's run-wide Holm rule.

| Comparison | Accuracy difference | Separated |
|---|---|---|
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-base-q4 | 0.177 [0.143, 0.211] | yes |
| gemma-e2b-grounded-v2-q4 minus gemma-e2b-grounded-q4 | -0.050 [-0.068, -0.032] | yes |

### Behaviour checks (B1–B8)

v2's runs (`main_v2`: claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone; `offscope_v2`: claims extracted by `gpt-6-luna`, judged by `gpt-6.1-sol` alone). The rows of v1 are in the results bundle under their own runs.

| Rule | Model | Metric | Value | n | Run | Source |
|---|---|---|---|---|---|---|
| B1 | gemma-e2b-base-q4 | schema_failures_or_zero_citations | 0.005 | 1000 | `main_v2` | `ggh:b1` |
| B2 | gemma-e2b-base-q4 | headline:gpt-6.1-sol | 0.404 [0.373, 0.434] | 1000 | `main_v2` | `evalkit:headline` |
| B2 | gemma-e2b-base-q4 | groundedness_mean:gpt-6.1-sol | 0.840 [0.820, 0.859] | 1000 | `main_v2` | `evalkit:groundedness` |
| B3 | gemma-e2b-base-q4 | maybe_recall | 0.491 [0.398, 0.586] | 110 | `main_v2` | `ggh:b3` |
| B3 | gemma-e2b-base-q4 | maybe_precision | 0.127 [0.095, 0.161] | 424 | `main_v2` | `ggh:b3` |
| B4 | gemma-e2b-base-q4 | over_80_words_share | 0.013 | 1000 | `main_v2` | `ggh:b4` |
| B6 | gemma-e2b-base-q4 | unsupported_claim_share:gpt-6.1-sol | 0.142 | 2481 | `main_v2` | `evalkit:verdicts` |
| B7 | gemma-e2b-base-q4 | citation_validity | 0.995 [0.990, 0.999] | 1000 | `main_v2` | `evalkit:citation_valid` |
| B7 | gemma-e2b-base-q4 | citation_precision:gpt-6.1-sol | 0.747 [0.722, 0.772] | 1000 | `main_v2` | `evalkit:citation_support` |
| B7 | gemma-e2b-base-q4 | aim_citation_rate:gpt-6.1-sol | 0.138 [0.117, 0.161] | 1000 | `main_v2` | `evalkit:citation_support` |
| B1 | gemma-e2b-grounded-q4 | schema_failures_or_zero_citations | 0.000 | 1000 | `main_v2` | `ggh:b1` |
| B2 | gemma-e2b-grounded-q4 | headline:gpt-6.1-sol | 0.331 [0.302, 0.361] | 1000 | `main_v2` | `evalkit:headline` |
| B2 | gemma-e2b-grounded-q4 | groundedness_mean:gpt-6.1-sol | 0.457 [0.427, 0.487] | 1000 | `main_v2` | `evalkit:groundedness` |
| B3 | gemma-e2b-grounded-q4 | maybe_recall | 0.045 [0.010, 0.086] | 110 | `main_v2` | `ggh:b3` |
| B3 | gemma-e2b-grounded-q4 | maybe_precision | 0.227 [0.059, 0.409] | 22 | `main_v2` | `ggh:b3` |
| B4 | gemma-e2b-grounded-q4 | over_80_words_share | 0.000 | 1000 | `main_v2` | `ggh:b4` |
| B6 | gemma-e2b-grounded-q4 | unsupported_claim_share:gpt-6.1-sol | 0.523 | 1199 | `main_v2` | `evalkit:verdicts` |
| B7 | gemma-e2b-grounded-q4 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `main_v2` | `evalkit:citation_valid` |
| B7 | gemma-e2b-grounded-q4 | citation_precision:gpt-6.1-sol | 0.362 [0.343, 0.382] | 1000 | `main_v2` | `evalkit:citation_support` |
| B7 | gemma-e2b-grounded-q4 | aim_citation_rate:gpt-6.1-sol | 0.560 [0.529, 0.590] | 1000 | `main_v2` | `evalkit:citation_support` |
| B1 | gemma-e2b-grounded-v2-q4 | schema_failures_or_zero_citations | 0.000 | 1000 | `main_v2` | `ggh:b1` |
| B2 | gemma-e2b-grounded-v2-q4 | headline:gpt-6.1-sol | 0.329 [0.300, 0.358] | 1000 | `main_v2` | `evalkit:headline` |
| B2 | gemma-e2b-grounded-v2-q4 | groundedness_mean:gpt-6.1-sol | 0.493 [0.463, 0.523] | 1000 | `main_v2` | `evalkit:groundedness` |
| B3 | gemma-e2b-grounded-v2-q4 | maybe_recall | 0.055 [0.018, 0.098] | 110 | `main_v2` | `ggh:b3` |
| B3 | gemma-e2b-grounded-v2-q4 | maybe_precision | 0.240 [0.087, 0.409] | 25 | `main_v2` | `ggh:b3` |
| B4 | gemma-e2b-grounded-v2-q4 | over_80_words_share | 0.000 | 1000 | `main_v2` | `ggh:b4` |
| B6 | gemma-e2b-grounded-v2-q4 | unsupported_claim_share:gpt-6.1-sol | 0.469 | 1209 | `main_v2` | `evalkit:verdicts` |
| B7 | gemma-e2b-grounded-v2-q4 | citation_validity | 1.000 [1.000, 1.000] | 1000 | `main_v2` | `evalkit:citation_valid` |
| B7 | gemma-e2b-grounded-v2-q4 | citation_precision:gpt-6.1-sol | 0.736 [0.708, 0.763] | 1000 | `main_v2` | `evalkit:citation_support` |
| B7 | gemma-e2b-grounded-v2-q4 | aim_citation_rate:gpt-6.1-sol | 0.084 [0.067, 0.102] | 1000 | `main_v2` | `evalkit:citation_support` |
| B5 | gemma-e2b-base-q4 | sample_read_ok_share | 0.250 | 20 | `main_v2` | `model_review` |
| B5 | gemma-e2b-grounded-q4 | sample_read_ok_share | 0.450 | 20 | `main_v2` | `model_review` |
| B5 | gemma-e2b-grounded-v2-q4 | sample_read_ok_share | 0.550 | 20 | `main_v2` | `model_review` |
| B8 | gemma-e2b-grounded-v2-q4 | presort_flagged_share | 0.000 | 20 | `offscope_v2` | `ggh:b8` |
| B8 | gemma-e2b-grounded-v2-q4 | gives_advice_count | 0 | 20 | `offscope_v2` | `model_review` |

Rows with source `model_review` are a blind review by Claude Opus 5.5, not the owner (OWNER D-61): B5 is the sample read of 20 answers per model from `main_v2` under `configs/sample_read_prompt.md`; B8 is every off-scope answer (20 per model scored in `offscope_v2`) under `configs/offscope_review_prompt.md`, the reviewers seeing the question and answer only. Each review ran as a fresh subagent with no model names. Reviewer as recorded: `claude-opus-5-5 (blind subagent, OWNER D-61)`; `claude-opus-5-5 (blind subagent, configs/sample_read_prompt.md, OWNER D-61)`.

Sample read findings (B5). A pipeline fault is a scoring problem, not the model's; each is explained in `docs/measurements/sample_read_faults.md`:

| Model | Run | Read | model fault | ok | pipeline fault |
|---|---|---|---|---|---|
| gemma-e2b-base-q4 | `main_v2` | 20 | 9 | 5 | 6 |
| gemma-e2b-grounded-q4 | `main_v2` | 20 | 10 | 9 | 1 |
| gemma-e2b-grounded-v2-q4 | `main_v2` | 20 | 8 | 11 | 1 |

### Parity: demo weights against benchmark weights

The published merged bf16 weights are for GPU use; the benchmark scores, and the local demo serves, the Q4_K_M GGUF (OWNER D-63). On 500 dev prompts, greedy, the two agree on the label for 0.920 of prompts and cite exactly the same sentences for 0.730. The label agreement is under the plan's bar of 0.950 (`configs/eval_plan_v2.yaml`, `dev_parity`); the release went ahead at Q4_K_M with the miss disclosed (OWNER D-55).

| Weights | Schema valid |
|---|---|
| bf16 | 1.000 |
| q4_k_m | 1.000 |

Judged headline difference (bf16 minus Q4_K_M) on 200 PubMedQA expert questions from the main run, 95% interval, claims extracted by `gpt-6-luna`:

| Judge | Difference |
|---|---|
| gpt-6.1-sol | -0.030 [-0.090, 0.030] |

### Laptop speed

v2's GGUF (`gemma-e2b-grounded-v2-q4`), variant `ft2-Q4_K_M` in `docs/measurements/v2/laptop_bench.json`, served by llama.cpp on a laptop:

|  | Value |
|---|---|
| Machine | Apple M3 Max, 64 GB, macOS 26.5.2, ac power |
| llama.cpp | b12421 |
| Weights | Q4_K_M, 3269.1 MB |
| Prompts | 50 × 3 repetitions |
| Prompt processing (p50) | 2868.2 tokens/s |
| Generation (p50 / p95) | 73.1 / 80.1 tokens/s |
| Time to first token (p50) | 185 ms |
| Answer latency (p50 / p95) | 840 / 1011 ms |
| Peak memory (server RSS) | 4506.6 MB |
| llama-bench pp512 | 7066.7 tokens/s |
| llama-bench tg128 | 129.2 tokens/s |

## Limits

- One benchmark's question style: PubMedQA questions are mostly titles turned into yes or no questions, so other phrasings may do worse.
- English only, and abstracts only.
- The `maybe` label was learned from 272 constructed training examples, 27 more held out for validation (an abstract with its deciding sentences removed, OWNER D-15), not from real maybe-labelled questions.
- Contamination probe `ggh_probe_original` (run `r-1ba7c0e6a17c`): accuracy gemma-e2b-grounded-v2-q4 0.760 [0.680, 0.840].
- Contamination probe `ggh_probe_paraphrase` (run `r-49fbcc60b2a1`): accuracy gemma-e2b-grounded-v2-q4 0.760 [0.680, 0.840].
- Paraphrase drop (original minus paraphrased accuracy, 100 questions): 0.000 [-0.030, 0.030]; contamination signal: no.
- Dev grounding check before the benchmark (`docs/results/dev_ground_v2_sol/summary.json`, run `r-4b7e3a0c3d19`, 200 dev questions, judged by `gpt-6.1-sol`, bars in `configs/v2_dev_bars.yaml`): passed on citation precision 0.768 (bar 0.600) and aim-citation rate 0.067 (bar 0.200); missed on headline 0.350 against v1's 0.355 (bar: no lower than v1) and label accuracy 0.795 (bar 0.800). The release went ahead with these misses disclosed (OWNER D-55); the bars name two judges, but only `gpt-6.1-sol` judged this run (OWNER D-54), and on an earlier partial dual-judge run the headline also missed under Claude Opus 5.5 (OWNER D-55).

## Licence and credits

- Weights and code: Apache-2.0 (`LICENSE`). The repo's `NOTICE` is shipped with the weights.
- Base model: Google DeepMind's Gemma 4 E2B (`google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37`). Its model card declares `license: apache-2.0` and links its terms at https://ai.google.dev/gemma/docs/gemma_4_license. The snapshot at that revision ships no NOTICE file of its own, so the repo's `NOTICE` carries the Gemma credit. Read on 2026-11-20 from the model card at that revision (sha256 `3e0608a6be80e3eb040a5aa6c76707809503deb85be6360184a118fb6156526c`), recorded in `docs/licence_check.json`. Re-read on 2026-11-20: that link resolves to the Apache License 2.0 text (https://gemma.ai/resources/apache-2/), and the Gemma Terms of Use (https://gemma.ai/resources/terms/, last modified 2026-04-01) say they do not cover Gemma 4. Although those terms do not cover Gemma 4, this release voluntarily keeps to Google's Gemma Prohibited Use Policy (https://gemma.ai/resources/prohibited-use/), which names no version and bars, among other things, the unlicensed practice of medicine and automated decisions affecting people's health; the model is for research only and is never to be used for either.
- Data: PubMedQA (Jin et al., 2019), MIT licence, https://pubmedqa.github.io/.
- Evaluation: EvalKit 06591f8de9b1, ee5969978c92, f9cbc8e5396c (MIT licence, https://github.com/biplob819/evalkit).

## Research only, not medical advice

These answers summarise a study abstract. They are not medical advice and must not be used to make decisions about anyone's health. Ask a clinician about your own situation.
