"""RCP-04: `ggh doctor --kaggle`. What a Kaggle session must have before training starts: a
CUDA GPU (T4 expected, D-9), enough VRAM and disk, and internet for the pinned downloads."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

MIN_VRAM_GB = 14.0  # a T4 reports about 15 GB
MIN_DISK_GB = 20.0  # base model, data, checkpoints
PROBE = "https://huggingface.co/api/models/google/gemma-4-E2B-it"


def _gpu() -> tuple[str, float] | None:  # pragma: no cover - needs torch and a GPU
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return p.name, p.total_memory / 1e9


def _online(url: str = PROBE) -> bool:  # pragma: no cover - network
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status == 200
    except OSError:
        return False


def kaggle_report(
    work: Path,
    *,
    gpu: Callable[[], tuple[str, float] | None] = _gpu,
    disk: Callable[[Path], Any] = shutil.disk_usage,
    online: Callable[[], bool] = _online,
) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    g = gpu()
    if g is None:
        ok = False
        lines.append("[FAIL] no CUDA GPU: turn on a GPU accelerator (T4) for this session")
    else:
        name, vram = g
        good = vram >= MIN_VRAM_GB
        ok &= good
        lines.append(
            f"[{'ok' if good else 'FAIL'}] gpu {name}, {vram:.1f} GB (needs >= {MIN_VRAM_GB:.0f})"
        )
        if "T4" not in name:
            lines.append(f"[info] gpu is not a T4; D-9's fp16 plan assumes one ({name})")
    free = disk(work).free / 1e9
    good = free >= MIN_DISK_GB
    ok &= good
    need = f"needs >= {MIN_DISK_GB:.0f}"
    lines.append(f"[{'ok' if good else 'FAIL'}] disk {free:.0f} GB free at {work} ({need})")
    net = online()
    ok &= net
    state = "on" if net else "off: turn it on in the notebook settings"
    lines.append(f"[{'ok' if net else 'FAIL'}] internet {state}")
    return ok, lines
