"""RCP-04: the thin Kaggle notebook (D-10, Tech Spec §5.3), rendered from code so the
committed copy is checked against its template.

It clones this repo at a pinned commit, installs pinned packages, runs `ggh doctor --kaggle`,
rebuilds the N1.2 data and replays the N1.3 targets with every committed hash checked, then
calls `ggh train`. No secrets: the base model is not gated and the cells read no token.
"""

from __future__ import annotations

import json
from typing import Any

from ggh.data.build import EVALKIT_COMMIT as EVALKIT  # the D-28 pin

REPO = "https://github.com/biplob819/gemma-grounded-health"
WORK = "/kaggle/working"
CHECKOUT = f"{WORK}/ggh"
EVALKIT_REPO = "https://github.com/biplob819/evalkit"
VERSIONED = "torch|transformers|peft|accelerate|bitsandbytes|unsloth|unsloth.zoo|evalkit-harness"


def _code(src: str) -> dict[str, Any]:
    lines = src.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + lines[-1:],
    }


def _md(src: str) -> dict[str, Any]:
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + lines[-1:]}  # fmt: skip


def render(commit: str, tool: str, max_steps: int | None) -> dict[str, Any]:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("commit must be a full 40-character git sha")
    if tool not in ("unsloth", "trl"):
        raise ValueError(f"tool must be unsloth or trl, got {tool!r}")
    steps = f" --max-steps {max_steps}" if max_steps else ""
    out = f"{WORK}/run"
    cells = [
        _md(
            "# Gemma Grounded Health: training run\n\n"
            f"Thin runner (D-10): repo at `{commit[:12]}`, backend `{tool}`"
            + (f", {max_steps} steps (pilot)" if max_steps else ", full run")
            + ". All logic lives in the tested `ggh` package; any failing step stops the run."
        ),
        _code(
            "import os, subprocess, shutil, json\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one T4 (D-9), even on a T4 x2 machine\n'
            'os.environ["TOKENIZERS_PARALLELISM"] = "false"\n'
            "\n"
            "def sh(cmd):\n"
            '    """Run a shell command; a non-zero exit stops the notebook."""\n'
            '    print("$", cmd, flush=True)\n'
            "    subprocess.run(cmd, shell=True, check=True)\n"
            "\n"
            f'sh("git clone -q {REPO} {CHECKOUT}")\n'
            f'os.chdir("{CHECKOUT}")\n'
            f'sh("git checkout -q {commit}")\n'
            f'sh("pip install -q -r notebooks/requirements-kaggle-{tool}.txt")\n'
            'sh("pip install -q --no-deps -e . && pip install -q pydantic pyyaml pyarrow")\n'
            f"sh(\"pip install -q --no-deps 'evalkit-harness @ git+{EVALKIT_REPO}@{EVALKIT}'\")"
        ),
        _code(
            f"sh(\"pip list 2>/dev/null | grep -i -E '^({VERSIONED}) ' || true\")\n"
            f'sh("ggh doctor --kaggle --work {WORK}")'
        ),
        _code(
            "# rebuild N1.2 data and replay N1.3 targets; every committed hash is checked\n"
            'sh("ggh data")\n'
            'sh("git diff --exit-code docs/measurements/data_manifest.json")\n'
            'sh("ggh targets --replay")'
        ),
        _code(f'sh("ggh train --config configs/recipe.yaml --out {out} --tool {tool}{steps}")'),
        _code(
            f'report = json.load(open("{out}/report.json"))\n'
            f'shutil.copy("{out}/report.json", "{WORK}/report.json")\n'
            "short = {k: v for k, v in report.items() if k not in ('losses', 'eval_losses')}\n"
            "print(json.dumps(short, indent=2))"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def dumps(nb: dict[str, Any]) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=True) + "\n"


def kernel_metadata(slug: str, code_file: str, owner: str = "biplob819") -> dict[str, Any]:
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
