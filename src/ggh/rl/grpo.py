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
import sys
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


def train_prompts(root: Path) -> list[dict[str, Any]]:
    """The GRPO rows from the v2 train targets under the repository root `root`."""
    from ggh.train.run import load_examples  # it adds artifacts/targets itself

    return prompt_rows(load_examples(root, "train"))


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
    from ggh.train.run import check_source, config_hash, load_unsloth, peak_vram_gb

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

    tok = loaded.tokenizer
    limit = base.max_seq_len - g.max_completion_len
    rows, dropped = [], []
    for row in train_prompts(root):
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
    """Reward and the plan's hack guards over dev replies. Every share is over all `n` rows (an
    unparsed reply counts as no label). `grounded_paid` is the reward's grounding credit (right
    label and grounded citations); `citations_pass` counts replies whose citations pass the check
    whatever their label."""
    if len(texts) != len(rows) or not rows:
        raise ValueError("one reply per dev row, and at least one row")
    scores = [R.score(t, r["label"], r["sentences"]) for t, r in zip(texts, rows, strict=True)]
    parsed = [R.parse(t) for t in texts]
    labels = [a.label if a is not None else None for a in parsed]
    words = [len(a.answer.split()) for a in parsed if a is not None]
    n = len(rows)
    no_rows = [i for i, r in enumerate(rows) if r["label"] == "no"]

    def share(k: int) -> float:
        return round(k / n, 4)

    return {
        "n": n,
        "mean_reward": round(statistics.fmean(s.reward for s in scores), 4),
        "label_accuracy": share(sum(s.label_correct for s in scores)),
        "no_accuracy": round(sum(scores[i].label_correct for i in no_rows) / len(no_rows), 4)
        if no_rows
        else None,
        "n_gold_no": len(no_rows),
        "yes_share": share(labels.count("yes")),
        "maybe_share": share(labels.count("maybe")),
        "grounded_paid": share(sum(s.grounded and s.label_correct for s in scores)),
        "citations_pass": share(sum(s.grounded for s in scores)),
        "aim_cited": share(sum(s.aim_cited for s in scores)),
        "unparsed": share(sum(not s.parsed for s in scores)),
        "label_mix": dict(sorted(Counter(x for x in labels if x is not None).items())),
        "mean_words": round(statistics.fmean(words), 2) if words else 0.0,
        "max_words": max(words, default=0),
    }


def adapter_cmd(config: Path, adapter: Path, name: str, root: Path, out: Path, n: int) -> list[str]:
    """The command scoring one adapter in its own process, so each model's GPU memory is
    returned to the T4 when that process exits (a second model loaded in-process left 5.5 GB)."""
    return [
        sys.executable, "-m", "ggh.cli", "rl-dev-one", "--config", str(config),
        "--adapter", str(adapter), "--name", name, "--root", str(root), "--out", str(out),
        "--n", str(n),
    ]  # fmt: skip


def collect_dev(out: Path, names: Sequence[str], run_steps: int, n: int) -> dict[str, Any]:
    """rl_dev.json from each adapter's `dev_<name>.json`, in scoring order."""
    adapters = {}
    for name in names:
        summary = json.loads((out / f"dev_{name}.json").read_text(encoding="utf-8"))
        if summary.get("n") != n:
            raise ValueError(f"dev_{name}.json scored {summary.get('n')} rows, not {n}")
        adapters[name] = summary
    return {"run_steps": run_steps, "dev_rows": n, "adapters": adapters}


def dev_rows(root: Path, n: int) -> list[dict[str, Any]]:
    with open(root / "artifacts" / "data" / "dev_eval.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f][:n]


def rl_dev_one(
    config: Path, adapter: Path, name: str, root: Path, out: Path, n: int
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    """Greedy replies from one adapter on the first `n` dev prompts, scored with the reward;
    writes `dev_<name>.jsonl` (the replies) and `dev_<name>.json` (the summary)."""
    from ggh.rl.config import load_grpo
    from ggh.train.devrun import load_for_inference
    from ggh.train.generate import generate

    _, base = load_grpo(config, root)
    rows = dev_rows(root, n)
    model, tok = load_for_inference(base, adapter)
    texts = generate(model, tok, [r["prompt"] for r in rows])
    summary = dev_summary(texts, rows)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"dev_{name}.jsonl", "w", encoding="utf-8") as f:
        for r, t in zip(rows, texts, strict=True):
            f.write(json.dumps({"pmid": r["pmid"], "label": r["label"], "reply": t}) + "\n")
    (out / f"dev_{name}.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    print(name, json.dumps(summary), flush=True)
    return summary


def rl_dev(
    config: Path, run_dir: Path, root: Path, out: Path, n: int
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    """The start adapter (v2, downloaded and hash-checked), each saved checkpoint and the final
    adapter, each scored by `rl_dev_one` in its own process; writes rl_dev.json."""
    import subprocess

    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    from ggh.rl.config import load_grpo

    g, _ = load_grpo(config, root)
    snap = Path(snapshot_download(g.init_repo, revision=g.init_revision,
                                  allow_patterns=[f"{g.init_subfolder}/*"]))  # fmt: skip
    init = snap / g.init_subfolder
    check_init(init, g.init_adapter_sha256)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    adapters: list[tuple[str, Path]] = [("v2", init)]
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    adapters += [(p.name, p) for p in cks]
    if (run_dir / "final").is_dir():
        adapters.append(("final", run_dir / "final"))
    out.mkdir(parents=True, exist_ok=True)
    for name, path in adapters:
        done = subprocess.run(adapter_cmd(config, path, name, root, out, n), check=False)
        if done.returncode:
            raise ValueError(f"scoring {name} failed (exit {done.returncode})")
    results = collect_dev(out, [a for a, _ in adapters], report["steps"], n)
    (out / "rl_dev.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
