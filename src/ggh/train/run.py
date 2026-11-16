"""RCP-01, RCP-06: one recipe, two backends, one Trainer (Tech Spec §5.1, D-9).

`tool: unsloth` loads through Unsloth's `FastModel`; `tool: trl` through transformers + PEFT
(bitsandbytes 4-bit on CUDA). Either way the model trains on the same pre-tokenized,
completion-masked dataset (`ggh.train.format`) with the same `transformers.Trainer` settings,
so masking and data order never depend on the backend. Needs the `train` group locally, or
the Kaggle requirements; imports are inside functions so `ggh` itself stays light.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ggh.config import load_recipe
from ggh.data.build import read_jsonl
from ggh.schemas import RecipeConfig, TrainExample
from ggh.train import format as fmt

TARGETS = Path("artifacts") / "targets"
SKIP_TOWERS = ("vision", "audio")  # any path part naming these: train the language model only


class RecipeError(ValueError):
    pass


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_recipe(cfg: RecipeConfig) -> None:
    if not cfg.is_pinned:
        raise RecipeError("base_revision is unpinned; T2.1 trains only on a pinned Hub commit")


def lora_targets(module_names: Sequence[str], wanted: Sequence[str]) -> list[str]:
    """Full module names to adapt: projections named in the recipe, inside the language model
    only (a multimodal checkpoint also has vision and audio towers with `q_proj` layers)."""
    out = []
    for name in module_names:
        parts = name.split(".")
        if parts[-1] in wanted and not any(t in p for p in parts for t in SKIP_TOWERS):
            out.append(name)
    if not out:
        raise RecipeError(f"no module matches target_modules {list(wanted)}")
    return out


# ---- data ----


def load_examples(root: Path, split: str, limit: int | None = None) -> list[TrainExample]:
    rows = read_jsonl(root / TARGETS / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    return [TrainExample.model_validate(r) for r in rows]


class TorchRows:
    """A map-style dataset over encoded rows (what `Trainer` needs)."""

    def __init__(self, rows: Sequence[fmt.Encoded]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> fmt.Encoded:
        return self.rows[i]


def collator(pad_id: int) -> Callable[[list[fmt.Encoded]], dict[str, Any]]:
    import torch  # pyright: ignore[reportMissingImports]

    def collate(batch: list[fmt.Encoded]) -> dict[str, Any]:
        return {k: torch.tensor(v) for k, v in fmt.collate(batch, pad_id).items()}

    return collate


# ---- model loading ----


@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    backend: str
    source: tuple[str, str | None] | None = None  # (model id, commit) actually loaded


def loaded_source(model: Any) -> tuple[str, str | None]:
    """The Hub id and commit transformers recorded when it loaded the weights."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config
    return str(getattr(cfg, "_name_or_path", "")), getattr(cfg, "_commit_hash", None)


def check_source(cfg: RecipeConfig, source: tuple[str, str | None]) -> None:
    """D-1: "base" is the pinned Google checkpoint, never a mirror or another revision."""
    name, commit = source
    if name != cfg.base_model or commit != cfg.base_revision:
        raise RecipeError(
            f"loaded {name}@{commit}, but the recipe pins {cfg.base_model}@{cfg.base_revision}"
        )


def load_trl(cfg: RecipeConfig) -> Loaded:  # pragma: no cover - downloads the base model
    import torch  # pyright: ignore[reportMissingImports]
    from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # pyright: ignore[reportMissingImports]

    cuda = torch.cuda.is_available()
    dtype = getattr(torch, cfg.compute_dtype) if cuda else torch.float32
    quant = None
    if cfg.load_in_4bit and cuda:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        revision=cfg.base_revision,
        dtype=dtype,
        quantization_config=quant,
        device_map={"": 0} if cuda else None,
    )
    if quant is not None:
        # not peft's prepare_model_for_kbit_training: it casts every unquantized weight to
        # fp32, and Gemma 4 E2B's per-layer embedding tables (~2.3B parameters) then need
        # 8.75 GiB on their own (T2.1 pilot: out of memory on a 15 GB T4)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    peft_model = attach_lora(model, cfg, get_peft_model, LoraConfig)
    return Loaded(peft_model, tok, "trl", loaded_source(peft_model))


def attach_lora(model: Any, cfg: RecipeConfig, get_peft_model: Any, lora_config: Any) -> Any:
    names = [n for n, _ in model.named_modules()]
    lc = lora_config(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=lora_targets(names, cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


def load_unsloth(cfg: RecipeConfig) -> Loaded:  # pragma: no cover - CUDA and Kaggle only
    import torch  # pyright: ignore[reportMissingImports]
    from unsloth import FastModel  # pyright: ignore[reportMissingImports]

    model, tok = FastModel.from_pretrained(
        model_name=cfg.base_model,
        revision=cfg.base_revision,
        # without this Unsloth swaps in its own pre-quantized mirror at HEAD and ignores the
        # revision (T2.1 verifier round 1, from the first pilot's log): D-1 needs the pin
        use_exact_model_name=True,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit,
        dtype=getattr(torch, cfg.compute_dtype),
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules=list(cfg.target_modules),
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=cfg.seed,
    )
    tok = getattr(tok, "tokenizer", tok)  # Unsloth may return a processor for Gemma 4
    return Loaded(model, tok, "unsloth", loaded_source(model))


LOADERS: dict[str, Callable[[RecipeConfig], Loaded]] = {"trl": load_trl, "unsloth": load_unsloth}


# ---- training ----


def count_skipped(stats: Stats, trainer: Any) -> None:
    """One optimizer step: count it if accelerate's fp16 loss scaler skipped it (inf grads).
    An accelerator without the attribute is an error, so an API rename cannot silently
    report zero skipped steps forever."""
    if trainer is None or getattr(trainer, "accelerator", None) is None:
        raise RuntimeError("the skipped-step counter is not wired to a trainer with an accelerator")
    stats.skipped_steps += bool(trainer.accelerator.optimizer_step_was_skipped)


def non_finite(loss: Any) -> bool:
    import torch  # pyright: ignore[reportMissingImports]

    return not bool(torch.isfinite(loss.detach()).all())


@dataclass
class Stats:
    """What the pilot measures: throughput, memory, numerics."""

    tokens: int = 0  # non-padding training tokens seen
    started: float = 0.0
    seconds: float = 0.0
    losses: list[tuple[int, float]] = field(default_factory=list)
    eval_losses: list[tuple[int, float]] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    non_finite: int = 0  # logged losses and gradient norms that were NaN or inf
    non_finite_steps: int = 0  # micro-batch losses, checked every step (not only logged ones)
    skipped_steps: int = 0  # optimizer steps the fp16 loss scaler skipped for inf gradients

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    def log(self, step: int, logs: dict[str, float]) -> None:
        for key, store in (("loss", self.losses), ("eval_loss", self.eval_losses)):
            if key in logs:
                v = float(logs[key])
                store.append((step, v))
                self.non_finite += not math.isfinite(v)
        if "grad_norm" in logs:
            g = float(logs["grad_norm"])
            self.grad_norms.append(g)
            self.non_finite += not math.isfinite(g)


def training_args(
    cfg: RecipeConfig, out: Path, *, max_steps: int | None, cuda: bool, log_every: int = 10
) -> Any:
    from transformers import TrainingArguments  # pyright: ignore[reportMissingImports]

    return TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.scheduler,
        warmup_steps=cfg.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio
        num_train_epochs=cfg.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        eval_strategy="steps",
        eval_steps=cfg.eval_every_steps,
        save_strategy="steps",
        save_steps=cfg.eval_every_steps,
        save_total_limit=2,
        load_best_model_at_end=True,  # with save_total_limit=2: the best and the latest kept
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=log_every,
        logging_nan_inf_filter=False,  # never replace a NaN loss with the running mean
        fp16=cuda and cfg.compute_dtype == "float16",
        bf16=cuda and cfg.compute_dtype == "bfloat16",
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=not cuda,
    )


def make_trainer(cfg: RecipeConfig, loaded: Loaded, train: fmt.Dataset, val: fmt.Dataset,
                 out: Path, stats: Stats, *, max_steps: int | None,
                 log_every: int = 10) -> Any:  # fmt: skip
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import Trainer, TrainerCallback  # pyright: ignore[reportMissingImports]

    class Measure(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            stats.started = time.perf_counter()

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            stats.log(state.global_step, logs or {})

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            stats.seconds = time.perf_counter() - stats.started

    class Counting(Trainer):
        def training_step(self, model: Any, inputs: Any, *a: Any, **k: Any) -> Any:
            stats.tokens += int(inputs["attention_mask"].sum())
            loss = super().training_step(model, inputs, *a, **k)
            stats.non_finite_steps += non_finite(loss)
            return loss

    class Skipped(TrainerCallback):
        trainer: Any = None

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            count_skipped(stats, self.trainer)

    tok = loaded.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    skipped = Skipped()
    trainer = Counting(
        model=loaded.model,
        args=training_args(
            cfg, out, max_steps=max_steps, cuda=torch.cuda.is_available(), log_every=log_every
        ),
        train_dataset=TorchRows(train.rows),
        eval_dataset=TorchRows(val.rows),
        data_collator=collator(pad),
        callbacks=[Measure(), skipped],
    )
    skipped.trainer = trainer
    return trainer


def precision(model: Any, args: Any) -> dict[str, Any]:
    """What actually ran: parameter dtypes (trainable and frozen) and the autocast mode."""
    from collections import Counter

    trainable: Counter[str] = Counter()
    frozen: Counter[str] = Counter()
    for p in model.parameters():
        (trainable if p.requires_grad else frozen)[str(p.dtype).replace("torch.", "")] += p.numel()
    amp = "fp16" if args.fp16 else "bf16" if args.bf16 else None
    return {"amp": amp, "trainable": dict(trainable), "frozen": dict(frozen)}


def peak_vram_gb() -> float:
    import torch  # pyright: ignore[reportMissingImports]

    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def run(
    config_path: Path,
    root: Path,
    out: Path,
    *,
    max_steps: int | None = None,
    limit: int | None = None,
    load: Callable[[RecipeConfig], Loaded] | None = None,
    log_every: int = 10,
    tool: str | None = None,
) -> dict[str, Any]:
    """Train from the recipe and write `report.json` into `out`. `max_steps` makes a pilot;
    `limit` caps the examples read (tests)."""
    import torch  # pyright: ignore[reportMissingImports]

    cfg = load_recipe(config_path)
    if tool is not None:  # a pilot comparison; the report records which backend ran
        cfg = cfg.model_copy(update={"tool": tool})
    check_recipe(cfg)
    torch.manual_seed(cfg.seed)
    loaded = (load or LOADERS[cfg.tool])(cfg)
    if loaded.source is not None:
        check_source(cfg, loaded.source)
    train = fmt.build(loaded.tokenizer, load_examples(root, "train", limit), cfg.max_seq_len)
    val = fmt.build(loaded.tokenizer, load_examples(root, "val", limit), cfg.max_seq_len)
    stats = Stats()
    out.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(
        cfg, loaded, train, val, out, stats, max_steps=max_steps, log_every=log_every
    )
    trainer.train()
    report = {
        "backend": loaded.backend,
        "config_sha256": config_hash(config_path),
        "max_steps": max_steps,
        "steps": int(trainer.state.global_step),
        "train_examples": len(train.rows),
        "val_examples": len(val.rows),
        "eval_rows": len(trainer.eval_dataset),
        "epochs": trainer.args.num_train_epochs,
        "train_tokens": train.tokens,
        "dropped_too_long": {"train": train.dropped_too_long, "val": val.dropped_too_long},
        "tokens_seen": stats.tokens,
        "seconds": round(stats.seconds, 2),
        "tokens_per_s": round(stats.tokens_per_s, 1),
        "peak_vram_gb": round(peak_vram_gb(), 2),
        "losses": stats.losses,
        "eval_losses": stats.eval_losses,
        "non_finite": stats.non_finite,
        "non_finite_steps": stats.non_finite_steps,
        "skipped_steps": stats.skipped_steps,
        "base_loaded": list(loaded.source) if loaded.source else None,
        "precision": precision(loaded.model, trainer.args),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_class": type(
            loaded.model.get_base_model()
            if hasattr(loaded.model, "get_base_model")
            else loaded.model
        ).__name__,
        "trainable_params": sum(p.numel() for p in loaded.model.parameters() if p.requires_grad),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def full_run_hours(train_tokens: int, epochs: float, tokens_per_s: float) -> float:
    """RCP-07: wall-clock hours for the full run at a measured throughput."""
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be positive")
    return train_tokens * epochs / tokens_per_s / 3600
