"""RLR-03: the v3 GRPO recipe (`configs/recipe_v3_grpo.yaml`, OWNER D-66).

v3 continues training v2's own LoRA adapter, so the base model, its revision, the LoRA shape,
the dtype and the sequence length are not restated: they come from the v2 recipe named by
`init_recipe`, and the adapter to start from is pinned by Hub repo, revision and hash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, StrictBool, ValidationError, model_validator

from ggh.config import ConfigError, _UniqueKeyLoader, load_recipe
from ggh.schemas._base import Model, NonBlank, PosInt
from ggh.schemas.train import RecipeConfig

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Unit = Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False, strict=True)]


class GRPORecipe(Model):
    init_recipe: NonBlank  # the v2 recipe: base, revision, LoRA shape, dtype, max_seq_len
    init_repo: NonBlank  # the Hub repo holding v2's adapter
    init_revision: Revision
    init_subfolder: NonBlank
    init_adapter_sha256: Sha256  # ggh.train.select.adapter_hash of the downloaded folder
    prompts: Literal["targets_train"]  # the v2 train targets' prompts and gold labels
    learning_rate: Annotated[float, Field(gt=0.0, lt=1e-3, allow_inf_nan=False, strict=True)]
    scheduler: NonBlank
    warmup_ratio: Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False, strict=True)]
    beta: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]
    loss_type: Literal["dapo", "grpo", "dr_grpo", "bnpo"]  # pinned, not left to TRL's default
    scale_rewards: Literal["group", "batch", "none"]
    epsilon: Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False, strict=True)]
    num_iterations: PosInt
    num_generations: Annotated[int, Field(ge=2, le=16, strict=True)]
    temperature: Annotated[float, Field(gt=0.0, le=2.0, allow_inf_nan=False, strict=True)]
    top_p: Unit
    max_completion_len: PosInt
    per_device_batch: PosInt  # completions per device step
    grad_accum: PosInt
    epochs: Annotated[float, Field(gt=0.0, le=3.0, allow_inf_nan=False, strict=True)]
    save_every_steps: PosInt
    mask_truncated_completions: StrictBool
    seed: Annotated[int, Field(ge=0, le=2**32 - 1, strict=True)]

    @model_validator(mode="after")
    def _shape(self) -> GRPORecipe:
        if self.per_device_batch % self.num_generations:
            raise ValueError("per_device_batch must be a multiple of num_generations")
        return self

    def prompts_per_step(self) -> int:
        return self.per_device_batch * self.grad_accum // self.num_generations


def load_grpo(path: Path, root: Path = Path(".")) -> tuple[GRPORecipe, RecipeConfig]:
    """The GRPO recipe and the v2 recipe it continues from."""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except (yaml.YAMLError, TypeError, UnicodeDecodeError, RecursionError) as exc:
        raise ConfigError(f"{path} is not valid YAML: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must hold a mapping of recipe settings")
    try:
        g = GRPORecipe.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    base = load_recipe(root / g.init_recipe)
    if g.max_completion_len >= base.max_seq_len:
        raise ConfigError("max_completion_len must leave room for the prompt in max_seq_len")
    return g, base
