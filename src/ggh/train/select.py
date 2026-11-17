"""TRN-03, TRN-04: checkpoint selection by validation loss and the §5.2 dev bars, and the
frozen record of the choice (D-4).

Candidates are tried in order of validation loss (ties: the earlier step). The first whose
dev metrics pass every bar is selected; the others are never generated for. If none passes,
nothing is selected and the reasons are kept: §5.2 step 4 then needs a decision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ggh.schemas import DevMetrics
from ggh.train.dev import passes_selection

BAR = 0.98


@dataclass(frozen=True)
class Candidate:
    step: int
    eval_loss: float
    path: Path


def checkpoints(run_dir: Path) -> list[Candidate]:
    """The kept checkpoints and the eval loss each was saved at (from its trainer_state)."""
    out = []
    for d in sorted(run_dir.glob("checkpoint-*")):
        step = int(d.name.split("-")[1])
        state = json.loads((d / "trainer_state.json").read_text(encoding="utf-8"))
        losses = [h["eval_loss"] for h in state["log_history"] if h.get("step") == step
                  and "eval_loss" in h]  # fmt: skip
        if not losses:
            raise ValueError(f"{d.name} has no eval loss at step {step}")
        out.append(Candidate(step, float(losses[-1]), d))
    if not out:
        raise ValueError(f"no checkpoints under {run_dir}")
    return out


def ordered(cands: Sequence[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: (c.eval_loss, c.step))


def failed_bars(m: DevMetrics, bar: float = BAR) -> list[str]:
    return [
        f"{k} {getattr(m, k):.3f} < {bar}"
        for k in ("schema_valid", "has_citation", "citation_valid")
        if getattr(m, k) < bar
    ]


@dataclass(frozen=True)
class Tried:
    candidate: Candidate
    metrics: DevMetrics
    passed: bool
    failed: tuple[str, ...]


def select(
    cands: Sequence[Candidate], metrics_for: Callable[[Candidate], DevMetrics], bar: float = BAR
) -> tuple[Candidate | None, list[Tried]]:
    tried: list[Tried] = []
    for c in ordered(cands):
        m = metrics_for(c)
        ok = passes_selection(m, bar)
        tried.append(Tried(c, m, ok, tuple(failed_bars(m, bar))))
        if ok:
            return c, tried
    return None, tried


def adapter_hash(path: Path) -> str:
    """sha256 over the sorted relative file names and their bytes: one value for the adapter."""
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"{path} holds no files")
    for p in files:
        h.update(p.relative_to(path).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def selection_record(selected: Candidate | None, tried: Sequence[Tried], adapter: str | None
                     ) -> dict[str, Any]:  # fmt: skip
    return {
        "selected_step": selected.step if selected else None,
        "adapter_sha256": adapter,
        "tried": [
            {
                "step": t.candidate.step,
                "eval_loss": t.candidate.eval_loss,
                "passed": t.passed,
                "failed": list(t.failed),
                "metrics": t.metrics.model_dump(mode="json"),
            }
            for t in tried
        ],
    }
