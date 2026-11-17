## What I built

A small open model (Gemma 4 E2B, 4-bit, 3.3 GB) fine-tuned to answer a research question **only from the numbered sentences of a given abstract**, in a fixed JSON shape: a yes / no / maybe label, a short answer, and the sentence numbers it relied on. It runs offline on a laptop through llama.cpp, and it is scored with [EvalKit](https://github.com/biplob819/evalkit) against its own base model, Claude Haiku 5.5 and GPT-6 Luna on all 1,000 expert-labelled PubMedQA questions, with two judges from different model families.

Research only, not medical advice. The demo and the Hugging Face release come after the v2 run below.

## Decisions and trade-offs

- **Score on grounding, not just the label.** The headline metric is "correct and fully grounded": the label is right *and* every claim is supported by the sentences the answer cites, judged by Claude Opus 5.5 and GPT-6.1 Sol. Label accuracy alone would have called v1 a success.
- **Pre-register the comparisons, then live with them.** The three comparisons (fine-tuned against base, Haiku and Luna) were written down and hashed in the repo before the main run (`configs/eval_plan.yaml`, commit `9bc0f97`). When v1 lost on the headline, the result stood.
- **Fix it on dev data only.** v2 is chosen with a dev-set grounding check, never with the 1,000 test questions, and it is labelled as built after v1's test results were seen.

## What failed

**The fine-tuned model got much better at the label and worse at grounding, so it did not beat its own base model on the metric that matters.** Status: the failure and its cause are measured; the v2 fix is in progress, and its numbers will be added here from its own published run.

### The result (v1, main run `r-f0f3b432a413`, 1,000 PQA-L questions)

| Model | Label accuracy | Headline: correct and fully grounded (Opus / Sol) | Claims supported (Opus) |
| --- | --- | --- | --- |
| Gemma base | 0.501 | 0.418 / 0.394 | 0.863 |
| **Gemma fine-tuned (v1)** | **0.728** | 0.388 / 0.330 | **0.519** |
| Claude Haiku 5.5 | 0.685 | 0.345 / 0.231 | 0.817 |
| GPT-6 Luna | 0.579 | 0.431 / 0.290 | 0.852 |

Accuracy rose 22.7 points over the base (separated, paired bootstrap), but the headline fell 3.0 points (Opus judge, not separated) because 45 % of the fine-tuned model's claims were judged unsupported, against 12 % for the base. Source: `docs/results/main/summary.json`.

### How I found the cause, step by step

1. **Read the judges' reasons, not just the score.** A typical v1 answer stated the study's finding correctly but cited sentence 1, and both judges said the same thing: *"The sentence only states the research question being studied; it reports no result."*
2. **Count it.** Of the 525 fine-tuned claims the primary judge marked unsupported, 303 cite sentence 1, which is usually the study's aim (92 cite only sentence 1). The base model's unsupported claims cite sentence 1 in 31 of 273 cases.
3. **Check whether the warning signs were already there.** They were. Reading 20 dev answers after training (step D-18 of the process) I noted that 8 of 20 cited sentence 1, often the aim. Every automatic dev bar passed (valid JSON 100 %, valid citations 100 %), because "valid" only meant the cited numbers existed, not that they supported the claim.
4. **Trace it to the training data.** 43 % of the 7,508 training answers cite an aim-like sentence ("To investigate…", "We examined…", "The aim of this study…"). The model learned exactly what it was shown.
5. **Find the mechanism.** Each training answer is the abstract's conclusion, which is not one of the numbered sentences. Its citations were picked by word overlap with that conclusion, and an entailment model at a 0.5 threshold let them through. An aim sentence shares most of its words with the conclusion ("Does X improve Y?" against "X improves Y"), so it ranked high and passed.

### What it taught me

- **"Valid citation" is not "supporting citation".** A citation check that only asks whether the sentence exists will pass a model that cites the question.
- **Word overlap rewards restating the question.** Ranking evidence by lexical overlap favours the sentence that asks the question over the sentence that answers it.
- **The hand-read sample was the earliest warning, and I under-weighted it.** It saw the pattern on 20 answers before any money was spent. The fix is to turn what the sample shows into a measured dev bar, not a note.

### What I changed (v2, in progress)

1. Training answers never cite an aim, objective or study-description sentence; citations must come from result sentences that the entailment check confirms (rule tested on fixtures, drops counted).
2. A dev grounding bar, judged by the same EvalKit judges on 200 dev questions, measured for v1 first as the baseline, with the v2 bars fixed before v2 is measured.
3. One retrain on Kaggle, the checkpoint chosen by validation loss, the existing bars and the new grounding bar.
4. A second, separately pre-registered run on the same 1,000 questions (v2 against base, v1, Haiku and Luna). Only v2's answers are new; the other models come from EvalKit's cache.

| Measure | v1 | v2 |
| --- | --- | --- |
| Training answers citing an aim-like sentence | 43 % | pending |
| Dev claims supported (Opus) | pending | pending |
| Test label accuracy | 0.728 | pending |
| Test headline (Opus / Sol) | 0.388 / 0.330 | pending |
| Test claims supported (Opus) | 0.519 | pending |

## Cost and how to reproduce

- v1 main run (4,000 answers, extraction and dual batch judging): **$53.68** against a $63.65 estimate; off-scope run $1.24; paraphrase probe set $0.47. Training: one Kaggle T4 session, 2.22 GPU hours. Every dollar is in `docs/measurements/spend-ledger.jsonl`.
- Reproduce: `make data && make targets`, run the committed Kaggle notebook (`notebooks/train_kaggle.ipynb`), then `ggh serve --model base`, `ggh serve --model ft` and `evalkit run configs/evalkit.yaml --task pubmedqa --n 1000`.
