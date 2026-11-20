---
$front_matter
---

# Gemma 4 E2B Grounded Health

> **Research only, not medical advice.** This model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

## Summary

$summary

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

$training_data

## Training procedure

$training_procedure

## Evaluation

$stamps

### Benchmark

$model_table

### Pre-registered comparisons

$primary_table

### Secondary: label accuracy

$secondary_table

### Behaviour checks (B1–B8)

$behaviour_table

### Parity: demo weights against benchmark weights

$parity

### Laptop speed

$bench

## Limits

- One benchmark's question style: PubMedQA questions are mostly titles turned into yes or no questions, so other phrasings may do worse.
- English only, and abstracts only.
$limits

## Licence and credits

- Weights and code: Apache-2.0. The model is a derivative of Google DeepMind's Gemma 4 E2B (Apache-2.0); the Gemma NOTICE is reproduced with the weights (see `NOTICE`).
- Data: PubMedQA (Jin et al., 2019), MIT licence, https://pubmedqa.github.io/.
- Evaluation: EvalKit $evalkit_version.

## Research only, not medical advice

These answers summarise a study abstract. They are not medical advice and must not be used to make decisions about anyone's health. Ask a clinician about your own situation.
