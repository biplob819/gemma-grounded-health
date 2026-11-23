"""RLX-01 (V3.4): freeze a GRPO run's selected checkpoint for export.

The plan's selection rule (`ggh.rl.select.select_checkpoint`) is applied again to the committed
`rl_dev.json`; the downloaded checkpoint must be the one it picks, the run must have started from
v2's frozen adapter under the pinned recipe, and only the two adapter files are copied, so the
adapter hash is the same kind of value as v1's and v2's (`ggh.train.select.adapter_hash`).
`train_selection.json` carries that hash for `ggh merge`; `rl_run.json` is the run's record.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ggh.lineage import sha256_file
from ggh.rl.select import select_checkpoint, step_of
from ggh.schemas._base import GitSha, Model, NonBlank, NonNegFloat, PosInt, Sha256
from ggh.train.select import ADAPTER_FILES, adapter_hash


class FreezeError(ValueError):
    pass


class RLRun(Model):
    """The GRPO counterpart of `TrainRun`: no validation-loss curve, a dev reward instead."""

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


def freeze_rl(
    root: Path,
    rl_dev: dict[str, Any],
    report: dict[str, Any],
    checkpoint: Path,
    *,
    recipe: Path,
    init_adapter_sha256: str,
    adapter_dir: Path,
    measurements: Path,
    kernel_version: str,
    score_kernel_version: str,
    repo_commit: str,
    gpu: str,
    rows: int = 500,
    no_slack: float = 0.05,
    copy_slack: float | None = None,
) -> RLRun:
    sel = select_checkpoint(rl_dev, rows=rows, no_slack=no_slack, copy_slack=copy_slack)
    if sel.selected is None:
        raise FreezeError("no candidate keeps every guard; nothing to freeze")
    if checkpoint.name != sel.selected:
        raise FreezeError(f"the plan selects {sel.selected}, not {checkpoint.name}")
    if int(rl_dev["run_steps"]) != int(report["steps"]):
        raise FreezeError(f"rl_dev covers {rl_dev['run_steps']} steps, the run {report['steps']}")
    want = sha256_file(root / recipe)
    if report["config_hash"] != want:
        raise FreezeError(f"the run's config hash {report['config_hash'][:12]}… is not "
                          f"{recipe.name}'s {want[:12]}…")  # fmt: skip
    if report["init_adapter_sha256"] != init_adapter_sha256:
        raise FreezeError("the run did not start from the frozen v2 adapter")
    missing = [f for f in ADAPTER_FILES if not (checkpoint / f).is_file()]
    if missing:
        raise FreezeError(f"{checkpoint} is missing {missing}")
    dest = root / adapter_dir
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ADAPTER_FILES:
        shutil.copy2(checkpoint / f, dest / f)
    sha = adapter_hash(dest)
    a = rl_dev["adapters"]
    step = step_of(sel.selected, int(report["steps"]))
    run = RLRun(
        reward=str(report.get("reward", "v3")),  # v3's report predates the field (v3b sets it)
        config_hash=report["config_hash"],
        init_adapter_sha256=report["init_adapter_sha256"],
        repo_commit=repo_commit,
        kaggle_kernel_version=kernel_version,
        score_kernel_version=score_kernel_version,
        gpu=gpu,
        wall_clock_s=float(report["wall_clock_s"]),
        peak_vram_gb=float(report["peak_vram_gb"]),
        steps=int(report["steps"]),
        selected=sel.selected,
        selected_step=step,
        adapter_sha256=sha,
        dev_rows=rows,
        dev_metrics=a[sel.selected],
        v2_dev_metrics=a["v2"],
    )
    out = root / measurements
    out.mkdir(parents=True, exist_ok=True)
    selection = {
        "selected": sel.selected,
        "selected_step": step,
        "beats_v2": sel.beats_v2,
        "adapter_sha256": sha,
        "rule": {"rows": rows, "no_slack": no_slack, "copy_slack": copy_slack},
        "table": sel.table,
    }
    (out / "train_selection.json").write_text(json.dumps(selection, indent=2) + "\n", "utf-8")
    (out / "rl_run.json").write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run
