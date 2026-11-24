"""Training schemas: the recipe, deterministic dev metrics and a training run (§3, §5)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StringConstraints, field_validator, model_validator

from ggh.schemas._base import (
    FrozenMap,
    GitSha,
    Model,
    NonBlank,
    NonNegFloat,
    PosInt,
    Prob,
    Sha256,
)

UNPINNED = "unpinned"
Revision = Annotated[str, StringConstraints(pattern=rf"^(?:[0-9a-f]{{40}}|{UNPINNED})$")]


class RecipeConfig(Model):
    base_model: NonBlank
    base_revision: Revision  # a 40-hex Hub commit; T2.1 refuses to train on "unpinned"
    tool: Literal["unsloth", "trl"]
    load_in_4bit: StrictBool
    compute_dtype: Literal["float16", "bfloat16"]
    lora_r: PosInt
    lora_alpha: PosInt
    lora_dropout: Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False, strict=True)]
    target_modules: tuple[NonBlank, ...]
    learning_rate: Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False, strict=True)]
    scheduler: NonBlank
    warmup_ratio: Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False, strict=True)]
    per_device_batch: PosInt
    grad_accum: PosInt
    max_seq_len: PosInt
    epochs: Annotated[float, Field(gt=0.0, le=3.0, allow_inf_nan=False, strict=True)]
    eval_every_steps: PosInt
    seed: Annotated[int, Field(ge=0, le=2**32 - 1, strict=True)]  # numpy's seed range
    completion_only_loss: StrictBool

    @field_validator("target_modules")
    @classmethod
    def _modules(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("target_modules needs at least one module")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def _multiple_of_128(cls, v: int) -> int:
        if v % 128:
            raise ValueError(f"max_seq_len {v} must be a multiple of 128")
        return v

    @property
    def is_pinned(self) -> bool:
        return self.base_revision != UNPINNED


class DevMetrics(Model):
    """Deterministic checks on the PQA-A dev split; never a model judge (D-4)."""

    n: PosInt
    schema_valid: Prob
    schema_valid_unconstrained: Prob
    citation_valid: Prob
    has_citation: Prob
    label_accuracy: FrozenMap[Prob]
    mean_words: NonNegFloat
    over_80_words: Prob


class TrainRun(Model):
    config_hash: Sha256
    data_manifest_hash: Sha256
    repo_commit: GitSha
    kaggle_kernel_version: NonBlank
    gpu: NonBlank
    wall_clock_s: NonNegFloat
    peak_vram_gb: NonNegFloat
    steps: PosInt
    selected_step: PosInt
    val_loss_curve: tuple[tuple[PosInt, NonNegFloat], ...]
    dev_metrics: DevMetrics

    @model_validator(mode="after")
    def _curve(self) -> TrainRun:
        steps = [s for s, _ in self.val_loss_curve]
        if not steps:
            raise ValueError("val_loss_curve needs at least one point")
        if steps != sorted(set(steps)):
            raise ValueError("val_loss_curve steps must be strictly increasing")
        if steps[-1] > self.steps:
            raise ValueError("val_loss_curve goes past the last training step")
        if self.selected_step not in steps:
            raise ValueError(f"selected_step {self.selected_step} is not on val_loss_curve")
        return self


class RLRun(Model):
    """The GRPO counterpart of `TrainRun` (V3.4, `ggh.rl.freeze`): no validation-loss curve, a
    dev reward instead. A release's training record is either (V3.8: v3b's is a GRPO run)."""

    reward: NonBlank
    config_hash: Sha256
    init_adapter_sha256: Sha256
    repo_commit: GitSha
    kaggle_kernel_version: NonBlank
    score_kernel_version: NonBlank
    gpu: NonBlank
    wall_clock_s: NonNegFloat
    peak_vram_gb: NonNegFloat
    steps: PosInt
    selected: NonBlank
    selected_step: PosInt
    adapter_sha256: Sha256
    dev_rows: PosInt
    dev_metrics: dict[str, Any]
    v2_dev_metrics: dict[str, Any]
