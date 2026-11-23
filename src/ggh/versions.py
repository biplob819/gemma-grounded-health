"""The fine-tuned model versions and where each one's files live (V2.1).

v1 (`ft`, T2.2 and X3.1) keeps every path it was built with; v2 (`ft2`, the V2.1 retrain on
`configs/recipe_v2.yaml`) gets its own measurement folder, adapter, merged folder and GGUF
names, so building v2 never overwrites a v1 file. v3 (`ft3`, V3.4) is the GRPO checkpoint
selected under configs/eval_plan_v3.yaml, and v3b (`ft3b`, V3.6) the one selected under
configs/eval_plan_v3b.yaml, each with its own folders the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MEASURE = Path("docs") / "measurements"


@dataclass(frozen=True)
class Version:
    model: str  # the GGUF name and `ggh serve --model` value
    measurements: Path  # train_run.json, train_selection.json, … and the GGUF manifest copy
    adapter_dir: Path
    merged_dir: Path
    recipe: Path

    @property
    def selection(self) -> Path:
        return self.measurements / "train_selection.json"

    @property
    def train_run(self) -> Path:
        return self.measurements / "train_run.json"


VERSIONS = {
    "ft": Version(
        "ft",
        MEASURE,
        Path("artifacts") / "adapter",
        Path("artifacts") / "merged" / "ft-bf16",
        Path("configs") / "recipe.yaml",
    ),
    "ft2": Version(
        "ft2",
        MEASURE / "v2",
        Path("artifacts") / "adapter_v2",
        Path("artifacts") / "merged" / "ft2-bf16",
        Path("configs") / "recipe_v2.yaml",
    ),
    "ft3": Version(  # v3 (V3.4): GRPO from v2's adapter, the checkpoint the v3 plan selects
        "ft3",
        MEASURE / "v3",
        Path("artifacts") / "adapter_v3",
        Path("artifacts") / "merged" / "ft3-bf16",
        Path("configs") / "recipe_v3_grpo.yaml",
    ),
    "ft3b": Version(  # v3b (V3.6): GRPO from v2's adapter with the checker reward (OWNER D-70)
        "ft3b",
        MEASURE / "v3b",
        Path("artifacts") / "adapter_v3b",
        Path("artifacts") / "merged" / "ft3b-bf16",
        Path("configs") / "recipe_v3b_grpo.yaml",
    ),
}


def version(model: str) -> Version:
    if model not in VERSIONS:
        raise ValueError(f"model must be one of {sorted(VERSIONS)}, got {model!r}")
    return VERSIONS[model]


def measurements(name: str) -> Path:
    """The folder for a model's committed records; the base shares v1's."""
    return VERSIONS[name].measurements if name in VERSIONS else MEASURE
