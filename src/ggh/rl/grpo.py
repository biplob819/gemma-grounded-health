"""RLR-04, RLR-05: `ggh grpo` trains v3 from v2's adapter, and `ggh rl-dev` scores checkpoints on
dev prompts with the same reward (OWNER D-66).

The reward is logged by component so a hack shows up in the run log as it happens: the four
component functions below sum to `reward.score(...).reward` for every completion, and TRL logs
each one's mean per step next to the completion length. Two zero-weight probes log the share of
"maybe" labels and of aim citations without changing the reward.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ggh.rl import reward as R
from ggh.rl.config import GRPORecipe
from ggh.schemas import TrainExample
from ggh.schemas.train import RecipeConfig

# ---- the reward, by component (they sum to reward.score().reward) ----


def _scores(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any]):
    return [
        R.score(R._text(c), g, ctx)
        for c, g, ctx in zip(completions, gold_label, context, strict=True)
    ]


def r_parse(
    completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any], **_: Any
):
    return [0.0 if s.parsed else R.R_UNPARSED for s in _scores(completions, gold_label, context)]


def r_label(
    completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any], **_: Any
):
    return [R.R_LABEL * s.label_correct for s in _scores(completions, gold_label, context)]


def r_grounded(
    completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any], **_: Any
):
    scores = _scores(completions, gold_label, context)
    return [R.R_GROUNDED * (s.grounded and s.label_correct) for s in scores]


def r_aim(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any], **_: Any):
    return [-R.P_AIM * s.aim_cited for s in _scores(completions, gold_label, context)]


def probe_maybe(completions: Sequence[Any], **_: Any) -> list[float]:
    return [
        float((a := R.parse(R._text(c))) is not None and a.label == "maybe") for c in completions
    ]


def probe_words(completions: Sequence[Any], **_: Any) -> list[float]:
    return [
        float(len(a.answer.split())) if (a := R.parse(R._text(c))) else 0.0 for c in completions
    ]


REWARD_FUNCS = (r_parse, r_label, r_grounded, r_aim, probe_maybe, probe_words)
REWARD_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)


# ---- data ----


def prompt_rows(examples: Sequence[TrainExample]) -> list[dict[str, Any]]:
    """GRPO rows: the chat prompt and the columns the reward reads (gold label, sentences)."""
    return [
        {
            "id": e.id,
            "prompt": [{"role": "user", "content": e.prompt}],
            "gold_label": e.target.label,
            "context": list(e.context),
        }
        for e in examples
    ]


def label_mix(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(r["gold_label"] for r in rows).items()))


# ---- trainer arguments ----


def grpo_args(
    g: GRPORecipe, base: RecipeConfig, out: Path, max_steps: int | None
) -> dict[str, Any]:
    """Keyword arguments for TRL 1.15's `GRPOConfig` (every key a field of it or of transformers
    5.17's `TrainingArguments`, pinned by a test). TRL has no prompt-length cap: `run` drops
    prompts longer than `max_seq_len - max_completion_len` before training."""
    return {
        "output_dir": str(out),
        "learning_rate": g.learning_rate,
        "lr_scheduler_type": g.scheduler,
        "warmup_steps": g.warmup_ratio,  # transformers 5: a float below 1 is a ratio of the steps
        "beta": g.beta,
        "loss_type": g.loss_type,
        "scale_rewards": g.scale_rewards,
        "epsilon": g.epsilon,
        "num_iterations": g.num_iterations,
        "num_generations": g.num_generations,
        "temperature": g.temperature,
        "top_p": g.top_p,
        "max_completion_length": g.max_completion_len,
        "per_device_train_batch_size": g.per_device_batch,
        "gradient_accumulation_steps": g.grad_accum,
        "num_train_epochs": g.epochs,
        "max_steps": max_steps if max_steps else -1,
        "save_strategy": "steps",
        "save_steps": g.save_every_steps,
        "save_only_model": True,  # adapters only: keeps a long run's output under Kaggle's cap
        "logging_steps": 1,
        "mask_truncated_completions": g.mask_truncated_completions,
        "reward_weights": list(REWARD_WEIGHTS),
        "seed": g.seed,
        "fp16": base.compute_dtype == "float16",
        "bf16": base.compute_dtype == "bfloat16",
        "report_to": "none",
        "remove_unused_columns": False,
    }


class InitError(ValueError):
    pass


def check_init(folder: Path, want: str) -> str:
    """The downloaded v2 adapter must hash to TrainRun v2's adapter hash."""
    from ggh.train.select import adapter_hash

    got = adapter_hash(folder)
    if got != want:
        raise InitError(f"v2 adapter at {folder} hashes to {got[:16]}…, want {want[:16]}…")
    return got


def reward_trace(log_history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-step reward, its components, the probes and completion length from TRL's log."""
    keep = ("step", "reward", "reward_std", "completions/mean_length", "frac_reward_zero_std")
    rows = []
    for h in log_history:
        if "reward" not in h:
            continue
        row = {k: h[k] for k in keep if k in h}
        row.update({k.split("/")[1]: v for k, v in h.items() if k.startswith("rewards/")
                    and k.endswith("/mean")})  # fmt: skip
        rows.append(row)
    return rows


def run(
    config: Path, root: Path, out: Path, max_steps: int | None = None
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    import torch  # pyright: ignore[reportMissingImports]
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    from ggh.rl.config import load_grpo
    from ggh.train.run import check_source, config_hash, load_examples, load_unsloth, peak_vram_gb

    g, base = load_grpo(config, root)
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(snapshot_download(g.init_repo, revision=g.init_revision,
                                  allow_patterns=[f"{g.init_subfolder}/*"]))  # fmt: skip
    init_hash = check_init(snap / g.init_subfolder, g.init_adapter_sha256)

    loaded = load_unsloth(base)  # the pinned base and a fresh LoRA of v2's shape
    if loaded.source is not None:
        check_source(base, loaded.source)
    # imported after Unsloth has patched transformers and TRL (it must load first)
    from datasets import Dataset  # pyright: ignore[reportMissingImports]
    from peft import set_peft_model_state_dict  # pyright: ignore[reportMissingImports]
    from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
    from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportMissingImports]

    state = load_file(str(snap / g.init_subfolder / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(loaded.model, state)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise InitError(f"{len(unexpected)} v2 adapter tensors did not load, e.g. {unexpected[0]}")
    n_lora = sum(1 for k, _ in loaded.model.named_parameters() if "lora_" in k)
    if n_lora != len(state):
        raise InitError(f"the model has {n_lora} LoRA tensors, v2's adapter {len(state)}")

    examples = load_examples(root / "artifacts" / "targets", "train")
    tok = loaded.tokenizer
    limit = base.max_seq_len - g.max_completion_len
    rows, dropped = [], []
    for row in prompt_rows(examples):
        ids = tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        n = len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        (rows if n <= limit else dropped).append(row)
    torch.manual_seed(g.seed)
    trainer = GRPOTrainer(
        model=loaded.model,
        processing_class=tok,
        reward_funcs=list(REWARD_FUNCS),
        args=GRPOConfig(**grpo_args(g, base, out, max_steps)),
        train_dataset=Dataset.from_list(rows),
    )
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    trainer.save_model(str(out / "final"))
    trace = reward_trace(trainer.state.log_history)
    report = {
        "config_hash": config_hash(config),
        "init_adapter_sha256": init_hash,
        "prompts": len(rows),
        "prompts_dropped_too_long": [r["id"] for r in dropped],
        "label_mix": label_mix(rows),
        "steps": trainer.state.global_step,
        "max_steps": max_steps,
        "wall_clock_s": round(wall, 2),
        "peak_vram_gb": peak_vram_gb(),
        "trl_args": {
            k: v
            for k, v in trainer.args.to_dict().items()
            if k in grpo_args(g, base, out, max_steps)
        },
        "reward_trace": trace,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# ---- dev scoring ----


def dev_summary(texts: Sequence[str], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reward and hack signals over dev replies: mean reward, label accuracy and mix, grounded
    and aim-citation shares, parse failures and answer length."""
    if len(texts) != len(rows) or not rows:
        raise ValueError("one reply per dev row, and at least one row")
    scores = [R.score(t, r["label"], r["sentences"]) for t, r in zip(texts, rows, strict=True)]
    parsed = [R.parse(t) for t in texts]
    words = [len(a.answer.split()) for a in parsed if a is not None]
    n = len(rows)
    return {
        "n": n,
        "mean_reward": round(statistics.fmean(s.reward for s in scores), 4),
        "label_accuracy": round(sum(s.label_correct for s in scores) / n, 4),
        "grounded": round(sum(s.grounded for s in scores) / n, 4),
        "aim_cited": round(sum(s.aim_cited for s in scores) / n, 4),
        "unparsed": round(sum(not s.parsed for s in scores) / n, 4),
        "label_mix": dict(sorted(Counter(a.label for a in parsed if a is not None).items())),
        "mean_words": round(statistics.fmean(words), 2) if words else 0.0,
        "max_words": max(words, default=0),
    }


def rl_dev(
    config: Path, run_dir: Path, root: Path, out: Path, n: int
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    """Greedy replies on the first `n` dev prompts from the start adapter (v2, downloaded and
    hash-checked) and each saved checkpoint, scored with the reward."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    from ggh.rl.config import load_grpo
    from ggh.train.devrun import load_for_inference, release
    from ggh.train.generate import generate

    g, base = load_grpo(config, root)
    snap = Path(snapshot_download(g.init_repo, revision=g.init_revision,
                                  allow_patterns=[f"{g.init_subfolder}/*"]))  # fmt: skip
    init = snap / g.init_subfolder
    check_init(init, g.init_adapter_sha256)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    with open(root / "artifacts" / "data" / "dev_eval.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:n]
    prompts = [r["prompt"] for r in rows]
    adapters: list[tuple[str, Path]] = [("v2", init)]
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    adapters += [(p.name, p) for p in cks]
    if (run_dir / "final").is_dir():
        adapters.append(("final", run_dir / "final"))
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"run_steps": report["steps"], "dev_rows": n, "adapters": {}}
    for name, path in adapters:
        model, tok = load_for_inference(base, path)
        texts = generate(model, tok, prompts)
        release(model)
        results["adapters"][name] = dev_summary(texts, rows)
        with open(out / f"dev_{name}.jsonl", "w", encoding="utf-8") as f:
            for r, t in zip(rows, texts, strict=True):
                f.write(json.dumps({"pmid": r["pmid"], "label": r["label"], "reply": t}) + "\n")
        print(name, json.dumps(results["adapters"][name]), flush=True)
    (out / "rl_dev.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
