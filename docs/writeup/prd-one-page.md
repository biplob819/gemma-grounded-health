# The PRD on one page

A summary of the [Gemma Grounded Health PRD](../../Gemma%20Grounded%20Health%20PRD.md), written before the build. It records what the project set out to do, not what it found; the results are in the [write-up](how-i-built-it.md) and the [project page](project-page.md). Every number here is quoted from the PRD.

## Goal

Answer one question the open-model community cares about: can a small open model, tuned in a day on free hardware, give grounded answers close to frontier small models while running on a laptop? The answer is a published recipe, model and benchmark. Gemma 4 E2B was chosen because it is open (Apache 2.0) and small enough for an on-device story.

## Problem

Developers building health features want an on-device model that answers only from the source in front of it and shows where each claim came from. Small open models tend to answer from memory and cite loosely, and few public recipes measure this honestly.

## Users

- **Developer building an on-device health feature** (primary): wants a small open model that stays grounded and runs offline.
- **Open-model community member on Hugging Face**: tries the demo, reads the model card, may reuse the recipe.
- **Open-model team reviewer**: wants to see the recipe, release and measurement run end to end.

## Must-haves (the first release)

- **US1** Groundedness and citation validity for the fine-tune against the base model and two frontier small models (Claude Haiku 5.5 and GPT-6 Luna), on the same 1,000 expert-labelled questions, through EvalKit.
- **US2** Speed and memory on a laptop at the shipped quantisation.
- **US3** A training notebook that runs end to end on a free Kaggle GPU within one session.
- **US4** A model card with intended use, out-of-scope uses, evaluation and limits.
- **US5** A demo showing base and fine-tuned answers side by side, with citations.
- **US6** The fine-tuned model callable through EvalKit like any other model.

Hard rules: no Claude or OpenAI output is used as training data or rewards, only to evaluate (NFR1); the fine-tune fits one free Kaggle session of at most 12 hours (NFR2); the card and demo say "research only, not medical advice" (NFR4); seeds, configs and data hashes are published (NFR5).

Non-goals: clinical use, training on frontier-model outputs, the larger Gemma 4 models, and a phone app.

## Success metrics

| Metric | Target |
| --- | --- |
| Groundedness uplift (fine-tuned minus base, held-out set) | Reported with intervals |
| Gap to frontier small models | Reported, whichever way it falls |
| Citation validity (citations exist and support claims) | Reported per model |
| Laptop speed (tokens per second and peak memory at 4-bit) | Reported |
| Reproducibility (the notebook reruns on a free GPU) | Yes |
| Community signal in the first two weeks | Tracked, no target |

The PRD named the risk that came true: "heuristic citation targets teach weak citing", mitigated by checking 50 targets by hand before training. That check passed and was the wrong test for this failure; the write-up tells how it was found and fixed.

## What changed later

The PRD planned a ZeroGPU Space for the demo; the demo became a local app (OWNER D-63). The PRD's answer targets were the raw conclusions with overlap-picked citations (FR2); the build filtered them (D-3), then rebuilt the citations by per-sentence support for v2 (OWNER D-43, D-48). Every change is in [docs/decisions.md](../decisions.md), summarised in [Decisions that shaped the result](decisions-summary.md).
