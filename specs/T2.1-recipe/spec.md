# T2.1 — Training recipe and pilot

**Wave 2. Depends on:** N1.3. **Unblocks:** T2.2.

## Goal
Training code that is tested on CPU, plus a thin Kaggle notebook. A 300-step pilot on a Kaggle T4 measures throughput, peak VRAM and fp16 stability, so that D-9 (Unsloth, fp16, T4) is confirmed or replaced on evidence before the full run (T2.2).

## Design
- **The same data on Kaggle (D-10, PRV-R1).** The notebook rebuilds the N1.2 splits with `ggh data` and checks the committed data manifest. It then rebuilds the targets with `ggh targets --replay`, which reads the recorded NLI scores committed as `data/nli_scores.jsonl.gz` (gzip with a fixed mtime) instead of running the model. The targets are byte-identical to the local build, and the hashes of `train.jsonl` and `val.jsonl` are checked against `docs/measurements/targets_drop_report.json`. Re-running the NLI model on another GPU could change scores in the last digit and so change the data.
- **Training text.** The user turn is the example's `prompt`, which is EvalKit's rendering (D-2). The assistant turn is the target as compact JSON, `{"label": ..., "answer": ..., "citations": [...]}`, which is exactly what EvalKit's `parse_answer` accepts. Both go through the pinned tokenizer's chat template, with thinking off, as at serving time (E3.1).
- **Completion-only loss.** The labels are -100 for every token of the rendered user turn plus the generation prompt. The test checks that this prefix is token-identical to the tokenizer's rendering of the user turn alone with `add_generation_prompt=True`, so the model learns only the answer and its end-of-turn token.
- **Backends (D-9).** `tool: unsloth` uses Unsloth's `FastModel` (4-bit, LoRA). `tool: trl` uses transformers + PEFT (+ bitsandbytes 4-bit on CUDA). Both feed one shared, tested pre-tokenized dataset to `transformers.Trainer`, so masking and data order do not depend on the backend. The TRL fallback uses PEFT with the plain `Trainer` rather than TRL's `SFTTrainer`, because the masking is done here and tested. The config value stays `trl`, as in the frozen N1.1 schema.
- **Pilot measurements.** A callback records tokens per second (non-padding tokens), peak VRAM, every logged loss, and whether any loss or gradient norm was not finite. The pilot JSON goes to `docs/measurements/pilot_t4.json`, and its GPU hours go into the GPU ledger.
- **Dev metrics (RCP-05).** `dev_metrics(rows, texts)` is deterministic over the generated texts. Schema validity uses EvalKit's `parse_answer`. Citation validity is every citation within 1..n, and has-citation is at least one citation. It also reports label accuracy per reference label and mean words. Generation itself (greedy) is a thin function, tested on the tiny model. Constrained-decoding validity is measured through llama-server in X3.1, and T2.1 reports the unconstrained figure.
- **Kaggle runs (OWNER D-38).** The builder pushes the notebook through the Kaggle API with the owner's token. Nothing secret goes into cells; the base model is not gated. `notebooks/kernel-metadata.json` turns on the GPU and the internet.
- **Dependencies.** A `train` dependency group (torch, transformers, peft, accelerate) is used locally for `pytest -m recipe`. CI does not install it, and those tests skip there, as the `nli` group's do. On Kaggle the notebook installs pinned packages from `notebooks/requirements-kaggle.txt`.

## Outputs
`src/ggh/train/{format,run,dev}.py`, `ggh train`, `ggh targets --replay`, `ggh doctor --kaggle`, `data/nli_scores.jsonl.gz`, `notebooks/train_kaggle.ipynb`, `notebooks/kernel-metadata.json`, `notebooks/requirements-kaggle.txt`, `docs/measurements/pilot_t4.json`, a GPU ledger entry, and a D-9 decision entry (confirmed or replaced).
