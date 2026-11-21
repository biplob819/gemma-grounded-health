"""RLR-06: the thin Kaggle notebook for the v3 GRPO runs (OWNER D-66), rendered from code like
the T2.2 one (`ggh.train.notebook`) so the committed copy is checked against its template.

It clones this repo at a pinned commit, installs the pinned GRPO packages, runs
`ggh doctor --kaggle`, rebuilds the N1.2 data and replays the v2 targets with every committed
hash checked, then runs `ggh grpo` (which downloads v2's adapter and checks its hash) and
`ggh rl-dev` on dev prompts. No secrets: the base model and v2 are public.
"""

from __future__ import annotations

from typing import Any

from ggh.evals.config import EVALKIT_REQUIREMENT
from ggh.train.notebook import CHECKOUT, REPO, WORK, _code, _md

VERSIONED = (
    "torch|transformers|trl|peft|accelerate|bitsandbytes|datasets|unsloth|unsloth.zoo|"
    "evalkit-harness"
)
CONFIG = "configs/recipe_v3_grpo.yaml"
TRIAL_STEPS = 40
TRIAL_DEV = 200
MAIN_DEV = 500


def render(
    commit: str, max_steps: int | None, evalkit: str = EVALKIT_REQUIREMENT
) -> dict[str, Any]:
    """The v3 runner for repo `commit`: a dev trial when `max_steps` is set, else the main run."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("commit must be a full 40-character git sha")
    if not evalkit.startswith("evalkit-harness") or "==" not in evalkit:
        raise ValueError(f"evalkit must pin evalkit-harness exactly, got {evalkit!r}")
    steps = f" --max-steps {max_steps}" if max_steps else ""
    n_dev = TRIAL_DEV if max_steps else MAIN_DEV
    run, dev = f"{WORK}/run", f"{WORK}/dev"
    kind = f"dev trial, {max_steps} steps" if max_steps else "main run"
    cells = [
        _md(
            "# Gemma Grounded Health v3: GRPO\n\n"
            f"Thin runner (D-10): repo at `{commit[:12]}`, recipe `{CONFIG}`, {kind}. v3 "
            "continues training v2's adapter with a programmatic reward (no judge, no API "
            "call). All logic lives in the tested `ggh` package; any failing step stops the run."
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
            'sh("pip install -q -r notebooks/requirements-kaggle-grpo.txt")\n'
            'sh("pip install -q --no-deps -e .")\n'
            f"sh(\"pip install -q --no-deps '{evalkit}'\")"
        ),
        _code(
            f"sh(\"pip list 2>/dev/null | grep -i -E '^({VERSIONED}) ' || true\")\n"
            f'sh("ggh doctor --kaggle --work {WORK}")'
        ),
        _code(
            "# rebuild N1.2 data and replay the v2 targets; every committed hash is checked\n"
            'sh("ggh data")\n'
            'sh("git diff --exit-code docs/measurements/data_manifest.json")\n'
            'sh("ggh targets --replay")'
        ),
        _code(f'sh("ggh grpo --config {CONFIG} --out {run}{steps}")'),
        _code(
            "# v2 and each checkpoint, greedy, on the first dev prompts, scored by the reward\n"
            f'sh("ggh rl-dev --config {CONFIG} --run {run} --out {dev} --n {n_dev}")'
        ),
        _code(
            f'shutil.copy("{run}/report.json", "{WORK}/report.json")\n'
            f'shutil.copy("{dev}/rl_dev.json", "{WORK}/rl_dev.json")\n'
            f'report = json.load(open("{run}/report.json"))\n'
            "short = {k: v for k, v in report.items() if k != 'reward_trace'}\n"
            "print(json.dumps(short, indent=2))\n"
            f'print(open("{dev}/rl_dev.json").read())'
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
