# Gemma Grounded Health PRD

@Biplob Chakraborty · Design: [Gemma Grounded Health Tech Spec](Gemma%20Grounded%20Health%20Tech%20Spec.md)

A small open Gemma model, fine-tuned to answer health questions only from a given abstract and to cite its sentences, scored with EvalKit against Claude Haiku 5.5 and GPT-6 Luna, and released on Hugging Face with a training recipe and a demo.

## Overview and deliverables

The project asks a question the open-model ecosystem cares about: can a small open model, tuned in a day on free hardware, give grounded answers close to frontier small models while running on a laptop? You answer it with a published recipe, model and benchmark.

**Why Gemma 4.** Google DeepMind released Gemma 4 on 2 April 2026 under the Apache 2.0 licence, in four sizes: E2B (2.3 billion effective parameters), E4B (4.5 billion effective), a 26B mixture-of-experts and a 31B dense model ([DataNorth](https://datanorth.ai/news/google-releases-gemma-4-open-models)). E2B runs in under 1.5 GB at 2-bit quantisation and E4B runs well on Apple Silicon laptops, which makes them the right sizes for an on-device story.

**Problem.** Developers building health features want an on-device model that answers only from the source in front of it and shows where each claim came from. Small open models tend to answer from memory and cite loosely, and there are few public recipes that measure this honestly.

**Goals.**

1. Fine-tune Gemma 4 E2B with QLoRA on PubMedQA's artificial set to answer from a given abstract and cite sentence numbers.
2. Score base and fine-tuned models with EvalKit against Claude Haiku 5.5 and GPT-6 Luna on the 1,000 expert-labelled questions.
3. Measure on-device speed and memory on your laptop.
4. Publish the model, a model card, the training recipe and a Gradio demo on Hugging Face.
5. Add the model to EvalKit as an adapter, so anyone can compare it.

**Non-goals.** Clinical use; training on frontier-model outputs; the 26B or 31B models; a phone app (named as the next step via Android's AICore developer preview).

**Deliverables.**

| Deliverable | What it is | Where it lives |
| --- | --- | --- |
| GitHub repo `gemma-grounded-health` | Data prep, training notebook, evaluation, export, tests | GitHub, public |
| Model and model card | LoRA adapter and merged weights, with intended use, limits and evaluation results | Hugging Face Hub |
| Training recipe | A step-by-step notebook that runs on a free Kaggle GPU | Kaggle and the repo |
| Benchmark results | Base vs fine-tuned vs Haiku vs Luna, plus on-device speed | Project page and EvalKit dashboard |
| Demo | A Gradio Space on ZeroGPU comparing base and fine-tuned answers | Hugging Face Spaces |
| Project page and video | Case study; a 90-second video including the model running offline on a laptop | Project site |
| Supporting artifacts | One-page PRD, decision log, data card, "how I built it" note | Repo and project page |

## Personas, user stories and flows

Three personas drive eight stories; the six marked Must make up the first release.

**Personas.**

- **Developer building an on-device health feature (primary).** Wants a small open model that stays grounded and runs offline.
- **Open-model community member on Hugging Face.** Will try the demo, read the model card and maybe reuse the recipe.
- **Open-model team reviewer.** Owns open-weight models, harnesses and evaluation sandboxes; wants to see the open-model flywheel run end to end: recipe, release, measurement.

**User stories.**

| ID | As a | I want to | So that | Acceptance criteria | Priority |
| --- | --- | --- | --- | --- | --- |
| US1 | Developer | see the fine-tuned model's groundedness and citation validity against base and two frontier small models | I know if it is good enough | EvalKit results for four models on the same 1,000 questions | Must |
| US2 | Developer | know its speed and memory on a laptop | I can plan on-device use | Tokens per second and peak memory reported for E2B at the shipped quantisation | Must |
| US3 | Community member | reproduce the fine-tune on free hardware | I can adapt it | The notebook runs end to end on a free Kaggle GPU within one session | Must |
| US4 | Community member | read what the model is for and where it fails | I use it responsibly | Model card with intended use, out-of-scope uses, evaluation and limits | Must |
| US5 | Open-model reviewer | try base and fine-tuned side by side | I see the change myself | Gradio demo shows both answers with citations for the same input | Must |
| US6 | Developer | call the model through EvalKit like any other | comparisons stay one command | `gemma-e2b-grounded` adapter in EvalKit | Must |
| US7 | Community member | download a quantised file for llama.cpp or Ollama | I can run it offline today | GGUF export published with the model | Should |
| US8 | Developer | see a small preference step's effect | I learn whether it is worth it | Optional DPO run with rule-based pairs, reported only if done | Could |

**Flow A. Train and release (US1 to US4, US6).**

1. `make data` builds training examples from PubMedQA's artificial set and holds out the expert-labelled set.
2. The Kaggle notebook runs QLoRA fine-tuning on Gemma 4 E2B.
3. `make eval` runs EvalKit on base, fine-tuned, Haiku 5.5 and GPT-6 Luna.
4. `make export` merges the adapter, exports GGUF and measures laptop speed.
5. `make release` pushes the model, model card and results to Hugging Face.

**Flow B. Try the demo (US5).**

1. The visitor opens the Space and picks a sample question, or pastes an abstract and a question.
2. Base and fine-tuned answers appear side by side, with cited sentences highlighted.
3. A note shows the free daily GPU quota and links to the benchmark.

## Requirements

Ten functional requirements and six non-functional bars; the most important bar is that no Claude or OpenAI output is used to train the model.

**Functional.**

| ID | Requirement | Stories |
| --- | --- | --- |
| FR1 | Training examples from PubMedQA's artificial set (211,269 examples, MIT): question plus numbered abstract sentences in, JSON `{label, answer, citations}` out | US1 |
| FR2 | Answer targets taken from the author-written conclusion (`long_answer`); citation targets from the abstract sentences with the highest lexical overlap with it | US1 |
| FR3 | The 1,000 expert-labelled questions held out entirely for evaluation | US1 |
| FR4 | QLoRA fine-tuning of Gemma 4 E2B in a Kaggle notebook, with config, seed and loss curves saved | US3 |
| FR5 | EvalKit evaluation of base, fine-tuned, Claude Haiku 5.5 and GPT-6 Luna | US1 |
| FR6 | Laptop benchmark: tokens per second and peak memory for the shipped quantisation | US2 |
| FR7 | Model card with intended use, out-of-scope uses, training data, evaluation, limits and licence | US4 |
| FR8 | Merged weights and a GGUF export on Hugging Face | US7 |
| FR9 | A Gradio demo on ZeroGPU with side-by-side base and fine-tuned answers | US5 |
| FR10 | An EvalKit adapter for the fine-tuned model | US6 |

**Non-functional.**

| ID | Area | Bar |
| --- | --- | --- |
| NFR1 | Terms | No Claude or OpenAI outputs used as training data or rewards; they are used only to evaluate |
| NFR2 | Compute | Fine-tune fits one free Kaggle GPU session (12 hours maximum) |
| NFR3 | Licensing | Gemma 4 is Apache 2.0; PubMedQA is MIT; both credited in the model card |
| NFR4 | Safety | Model card and demo state "research only, not medical advice" |
| NFR5 | Reproducibility | Seeds, configs and data hashes published |
| NFR6 | Demo cost | ZeroGPU only; no paid hardware |

## Behavioral spec

The model answers only from the abstract it is given, cites the sentences it used, and says so when the abstract does not answer the question. EvalKit tests each rule on the held-out set.

| ID | Rule | Type | Eval cases |
| --- | --- | --- | --- |
| B1 | Return valid JSON with a label (yes, no or maybe), an answer and at least one citation | Must | All 1,000 questions, schema check |
| B2 | Base every claim on cited sentences | Must | Groundedness scorer |
| B3 | Answer "maybe" and say the abstract is insufficient when it does not settle the question | Must | Questions with reference label "maybe" |
| B4 | Keep answers under 80 words | Must | Length check |
| B5 | Quote short phrases from the abstract where it helps | May | Sampled answers |
| B6 | Add facts not in the abstract, even true ones | Never | Groundedness scorer |
| B7 | Cite sentence numbers that do not exist | Never | Citation validity scorer |
| B8 | Give personal medical or dosing advice | Never | 20 off-scope prompts added to the eval |

**Good output.** `{"label": "maybe", "answer": "The abstract reports an association in one cohort but does not test causation.", "citations": [3, 5]}`

**Bad output.** `{"label": "yes", "answer": "Yes, and patients should increase their dose.", "citations": [9]}`. It adds advice, breaks B6 and B8, and cites a sentence that may not exist.

## Trust architecture

An open health model can be shared responsibly when its training is clean, its limits are published and its behaviour is measured by someone else's yardstick.

| Layer | What it does | Risk it handles | Evidence |
| --- | --- | --- | --- |
| Clean training data | MIT-licensed PubMedQA only; no frontier-model outputs | Licence and terms breaches | Data card and NFR1 |
| Held-out evaluation | Expert set never seen in training | Inflated scores from leakage | Overlap check in the repo |
| Independent judge | EvalKit's judge, calibrated against hand labels | Self-graded quality | EvalKit calibration panel |
| Grounding format | Answers must cite numbered sentences | Answers from memory | Citation validity and groundedness |
| Model card | Intended use, out-of-scope uses, limits | Misuse as a medical tool | Model card on Hugging Face |
| Demo framing | "Research only, not medical advice" banner; sample abstracts by default | Harmful reliance | Demo screen |
| Reproducible recipe | Seeds, configs, data hashes | Unverifiable claims | Notebook reruns |

The model card names its limits plainly: trained on one benchmark's style of question, English only, and tested only on abstracts.

## Wireframe: Hugging Face demo

&#91;embedded content: wireframe · Hugging Face demo, base against fine-tuned\]

The demo shows the change the fine-tune makes on the same input, which is the comparison an open-model reviewer wants to see first. The benchmark link carries the frontier comparison, and the GGUF link turns a visitor into a user.

## Data flow

&#91;embedded content: data flow · training data to fine-tune to evaluation to release\]

The top row trains on MIT-licensed data only; the second row exports the model and measures it. The expert set and the frontier models feed EvalKit from below and never touch training, which keeps the comparison clean and the release within both providers' terms.

## Technical architecture

&#91;embedded content: technical architecture · training, evaluation, hosting\]

- **Training.** A Kaggle notebook with Hugging Face Transformers, TRL and PEFT (or Unsloth), bitsandbytes for 4-bit loading, and the Gemma chat template.
- **Evaluation.** EvalKit, pinned, with a new adapter that runs the GGUF through llama.cpp on your laptop; Claude Haiku 5.5 and GPT-6 Luna through their APIs; the judge batched as in EvalKit.
- **Release.** Merged weights, a 4-bit GGUF and a model card on the Hugging Face Hub; the Gradio Space on ZeroGPU loads the merged weights.
- **Hosting choice.** Hugging Face holds this project because open-model reviewers and users already work there.

## Data model, training recipe and interfaces

One training example format, one recipe file and the EvalKit adapter carry the project.

```json
{"id": "pqa-art-104233",
 "input": {"question": "...", "context": ["[1] ...", "[2] ...", "[3] ..."]},
 "target": {"label": "yes", "answer": "<author-written conclusion>", "citations": [2, 3]}}
```

**Recipe (starting point, tuned on day one).**

| Setting | Value | Why |
| --- | --- | --- |
| Base model | Gemma 4 E2B | Laptop-class size; Apache 2.0 |
| Method | QLoRA, 4-bit base, LoRA on attention and MLP projections | Fits a free 16 GB GPU |
| Training examples | 20,000 from the artificial set, plus 2,000 for validation | Fits one Kaggle session with room to spare |
| Epochs | 1 to 2, stopping on validation loss | Avoids overfitting the style |
| Prompt format | Gemma chat template, JSON-only answer | Matches EvalKit's task |
| Tools | Hugging Face TRL and PEFT, or Unsloth for speed | Standard, documented, free |
| Export | Merged 16-bit weights and a 4-bit GGUF | Hub release and laptop use |

**Free compute.** Kaggle notebooks offer T4 and P100-class GPUs with 12-hour sessions; the weekly quota floats and shows in your account settings ([GPU per Hour, 21 September 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)).

**Demo hosting.** Free Hugging Face accounts in good standing (verified email, account older than 30 days) can host up to 2 ZeroGPU Spaces; ZeroGPU works only with the Gradio SDK, and visitors with a free account get 5 minutes of GPU a day ([Hugging Face docs](https://huggingface.co/docs/hub/spaces-zerogpu.md)). If your account is newer than 30 days, record the laptop demo for the video and publish the Space later.

**Interfaces.**

- `make data`, `make eval`, `make export`, `make release` in the repo; training runs in the Kaggle notebook.
- EvalKit adapter `gemma-e2b-grounded`, running the GGUF locally through llama.cpp, or the merged weights through Transformers.
- The Gradio Space exposes `compare(question, abstract)` returning base and fine-tuned answers with citations.

**Cost.** Training and the demo are free. Evaluating four models on all 1,000 expert questions costs about $56, almost all of it extraction and dual judging (Tech Spec §11).

## Success metrics, milestones and risks

Success is a measured, honest gap: how close a short, free-hardware fine-tune gets to frontier small models on grounding, and what it costs to run offline.

**Success metrics.**

| Metric | Definition | Target |
| --- | --- | --- |
| Groundedness uplift | Fine-tuned minus base, on the held-out set | Reported with intervals |
| Gap to frontier small models | Fine-tuned against Haiku 5.5 and GPT-6 Luna | Reported, whichever way it falls |
| Citation validity | Share of answers whose citations exist and support claims | Reported per model |
| Laptop speed | Tokens per second and peak memory at 4-bit | Reported |
| Reproducibility | Notebook reruns end to end on a free GPU | Yes |
| Community signal | Downloads, likes, demo use in the first two weeks | Tracked, no target |

**Milestones.**

1. Data prep, leakage check and hand audit of training targets.
2. Training recipe and a pilot run on Kaggle, then the full fine-tune.
3. Export, laptop benchmark and evaluation with EvalKit on the held-out expert set, run once on the frozen model.
4. Model card and Hugging Face release.
5. Gradio Space and the EvalKit example.
6. Project page and video.

**Risks.**

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Fine-tuned model barely beats base | Medium | Medium | Publish the result anyway; the honest gap and recipe are the point |
| Heuristic citation targets teach weak citing | Medium | Medium | Check 50 targets by hand before training; tighten the overlap rule |
| Hugging Face account too new for ZeroGPU | Unknown | Low | Laptop demo in the video; Space later |
| Kaggle quota runs out | Low | Medium | Keep runs to one session; use E2B, not E4B |
| Reviewers ask about RL post-training | High | Medium | Document why SFT came first and what a DPO or RL step would add |

## Sources

EvalKit details are in the [EvalKit repo](https://github.com/biplob819/evalkit).

- [Google releases Gemma 4, DataNorth](https://datanorth.ai/news/google-releases-gemma-4-open-models)
- [Hugging Face ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu.md)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Free cloud GPUs in 2026, GPU per Hour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- [PubMedQA on Hugging Face](https://huggingface.co/datasets/qiaojin/PubMedQA/blob/main/README.md)
