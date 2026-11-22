"""RLB-03: the v3b GRPO run and its dev check (OWNER D-70). It reuses v3's pinned trainer
pieces (`ggh.rl.grpo`: the v2 adapter download and hash check, the prompt rows, the TRL
arguments, the report) and swaps in v3b's reward (`ggh.rl.reward_b`) and its checker, so v3's
pinned files stay byte-identical.

The checker runs on the CPU during training (the T4 is full with GRPO, about 13.4 GB at peak) and
on the GPU in the dev check, where nothing else holds it. Each training step scores 32 replies;
only right-label yes/no replies with sound structure reach the checker.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ggh.rl import grpo as G
from ggh.rl import reward as R
from ggh.rl import reward_b as RB
from ggh.rl.checker import TUNING_IDS, Checker
from ggh.rl.config import GRPORecipe
from ggh.schemas.train import RecipeConfig

_CHECKER: Checker | None = None
_CACHE: dict[str, Any] = {"key": None, "scores": None}


def set_checker(checker: Checker | None) -> None:
    """Tests inject a fake; training and the dev check load the real one on first use."""
    global _CHECKER
    _CHECKER = checker
    _CACHE.update(key=None, scores=None)


def checker() -> Checker:  # pragma: no cover - the real checker downloads two models
    global _CHECKER
    if _CHECKER is None:
        from ggh.rl.checker import HFChecker

        _CHECKER = HFChecker(device=os.environ.get("GGH_CHECKER_DEVICE", "cpu"))
    return _CHECKER


def _scores(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any]):
    """TRL calls every reward function on the same batch: the checker runs once per batch."""
    texts = [R._text(c) for c in completions]
    key = json.dumps([texts, list(gold_label)])
    if _CACHE["key"] != key:
        _CACHE.update(key=key, scores=RB.score_batch(texts, gold_label, context, checker()))
    return _CACHE["scores"]


def r_parse(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any],
            **_: Any) -> list[float]:  # fmt: skip
    return [0.0 if s.parsed else R.R_UNPARSED for s in _scores(completions, gold_label, context)]


def r_label(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any],
            **_: Any) -> list[float]:  # fmt: skip
    return [RB.R_LABEL * s.label_correct for s in _scores(completions, gold_label, context)]


def r_grounded(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any],
               **_: Any) -> list[float]:  # fmt: skip
    scores = _scores(completions, gold_label, context)
    return [RB.R_GROUNDED * (s.grounded and s.label_correct) for s in scores]


def r_aim(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any],
          **_: Any) -> list[float]:  # fmt: skip
    return [-R.P_AIM * s.aim_cited for s in _scores(completions, gold_label, context)]


def probe_copied(completions: Sequence[Any], gold_label: Sequence[str], context: Sequence[Any],
                 **_: Any) -> list[float]:  # fmt: skip
    return [float(s.copied) for s in _scores(completions, gold_label, context)]


REWARD_FUNCS = (r_parse, r_label, r_grounded, r_aim, G.probe_maybe, G.probe_words, probe_copied)
REWARD_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def grpo_args(g: GRPORecipe, base: RecipeConfig, out: Path, max_steps: int | None
              ) -> dict[str, Any]:  # fmt: skip
    return {**G.grpo_args(g, base, out, max_steps), "reward_weights": list(REWARD_WEIGHTS)}


def run(
    config: Path, root: Path, out: Path, max_steps: int | None = None
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    """v3's `ggh.rl.grpo.run` with v3b's reward; the checker loads before the base model, so a
    failed download stops the run before any GPU work."""
    import torch  # pyright: ignore[reportMissingImports]
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    from ggh.rl.config import load_grpo
    from ggh.train.run import check_source, config_hash, load_unsloth, peak_vram_gb

    g, base = load_grpo(config, root)
    out.mkdir(parents=True, exist_ok=True)
    checker()
    snap = Path(snapshot_download(g.init_repo, revision=g.init_revision,
                                  allow_patterns=[f"{g.init_subfolder}/*"]))  # fmt: skip
    init_hash = G.check_init(snap / g.init_subfolder, g.init_adapter_sha256)
    loaded = load_unsloth(base)
    if loaded.source is not None:
        check_source(base, loaded.source)
    from datasets import Dataset  # pyright: ignore[reportMissingImports]
    from peft import set_peft_model_state_dict  # pyright: ignore[reportMissingImports]
    from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
    from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportMissingImports]

    state = load_file(str(snap / g.init_subfolder / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(loaded.model, state)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise G.InitError(
            f"{len(unexpected)} v2 adapter tensors did not load, e.g. {unexpected[0]}"
        )
    n_lora = sum(1 for k, _ in loaded.model.named_parameters() if "lora_" in k)
    if n_lora != len(state):
        raise G.InitError(f"the model has {n_lora} LoRA tensors, v2's adapter {len(state)}")
    tok = loaded.tokenizer
    limit = base.max_seq_len - g.max_completion_len
    rows, dropped = [], []
    for row in G.train_prompts(root):
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
    report = {
        "reward": "v3b",
        "config_hash": config_hash(config),
        "init_adapter_sha256": init_hash,
        "prompts": len(rows),
        "prompts_dropped_too_long": [r["id"] for r in dropped],
        "label_mix": G.label_mix(rows),
        "steps": trainer.state.global_step,
        "max_steps": max_steps,
        "wall_clock_s": round(wall, 2),
        "peak_vram_gb": peak_vram_gb(),
        "trl_args": {
            k: v
            for k, v in trainer.args.to_dict().items()
            if k in grpo_args(g, base, out, max_steps)
        },
        "reward_trace": G.reward_trace(trainer.state.log_history),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# ---- dev check ----


def tuning_ids(root: Path) -> set[str]:
    return set(json.loads((root / TUNING_IDS).read_text(encoding="utf-8")))


def dev_rows(root: Path, n: int) -> list[dict[str, Any]]:
    """The first `n` dev_eval rows the checker was not chosen on (300 remain of 500)."""
    held = tuning_ids(root)
    with open(root / "artifacts" / "data" / "dev_eval.jsonl", encoding="utf-8") as f:
        rows = [r for r in (json.loads(line) for line in f) if r["pmid"] not in held]
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} dev rows are outside the checker's tuning set")
    return rows[:n]


def dev_summary(texts: Sequence[str], rows: Sequence[dict[str, Any]], check: Checker
                ) -> dict[str, Any]:  # fmt: skip
    """v3b's reward and the plan's hack guards over dev replies, with the keys v3's `dev_summary`
    uses (so `ggh.rl.select` applies unchanged) plus the copy share and mean checker support."""
    if len(texts) != len(rows) or not rows:
        raise ValueError("one reply per dev row, and at least one row")
    scores = RB.score_batch(texts, [r["label"] for r in rows], [r["sentences"] for r in rows],
                            check)  # fmt: skip
    parsed = [R.parse(t) for t in texts]
    labels = [a.label if a is not None else None for a in parsed]
    words = [len(a.answer.split()) for a in parsed if a is not None]
    n = len(rows)
    no_rows = [i for i, r in enumerate(rows) if r["label"] == "no"]
    supports = [s.support for s in scores if s.support is not None]

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
        "aim_cited": share(sum(s.aim_cited for s in scores)),
        "copied": share(sum(s.copied for s in scores)),
        "unparsed": share(sum(not s.parsed for s in scores)),
        "mean_support": round(statistics.fmean(supports), 4) if supports else None,
        "label_mix": dict(sorted(Counter(x for x in labels if x is not None).items())),
        "mean_words": round(statistics.fmean(words), 2) if words else 0.0,
        "max_words": max(words, default=0),
    }


def adapter_cmd(config: Path, adapter: Path, name: str, root: Path, out: Path, n: int
                ) -> list[str]:  # fmt: skip
    return [
        sys.executable, "-m", "ggh.cli", "rl-dev-one-b", "--config", str(config),
        "--adapter", str(adapter), "--name", name, "--root", str(root), "--out", str(out),
        "--n", str(n),
    ]  # fmt: skip


def rl_dev_one(
    config: Path, adapter: Path, name: str, root: Path, out: Path, n: int
) -> dict[str, Any]:  # pragma: no cover - CUDA and Kaggle only
    from ggh.rl.config import load_grpo
    from ggh.train.devrun import load_for_inference
    from ggh.train.generate import generate

    _, base = load_grpo(config, root)
    rows = dev_rows(root, n)
    model, tok = load_for_inference(base, adapter)
    texts = generate(model, tok, [r["prompt"] for r in rows])
    del model, tok
    import gc

    import torch  # pyright: ignore[reportMissingImports]

    gc.collect()
    torch.cuda.empty_cache()
    os.environ.setdefault("GGH_CHECKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    summary = dev_summary(texts, rows, checker())
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
    """v2 (downloaded and hash-checked), each checkpoint and the final adapter, each scored by
    `rl_dev_one` in its own process on dev rows outside the checker's tuning set."""
    import subprocess

    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    from ggh.rl.config import load_grpo

    g, _ = load_grpo(config, root)
    snap = Path(snapshot_download(g.init_repo, revision=g.init_revision,
                                  allow_patterns=[f"{g.init_subfolder}/*"]))  # fmt: skip
    init = snap / g.init_subfolder
    G.check_init(init, g.init_adapter_sha256)
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
    results = G.collect_dev(out, [a for a, _ in adapters], report["steps"], n)
    (out / "rl_dev.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
