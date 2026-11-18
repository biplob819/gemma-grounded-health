"""TRN-03, TRN-04: dev evaluation on Kaggle and the local freeze (D-4, Tech Spec §5.2).

`dev_eval` (on Kaggle, after `ggh train`) generates greedy replies on the 500 `dev_eval`
prompts for each kept checkpoint in order of validation loss, constrained and not, until one
passes the §5.2 bars; it copies that adapter out and records its hash. `freeze` (locally,
before any PQA-L use) checks the downloaded adapter against that hash and writes `TrainRun`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ggh.config import load_recipe
from ggh.data.build import MANIFEST_PATH, OUT_DIR, read_jsonl
from ggh.schemas import DevMetrics, RecipeConfig, TrainRun
from ggh.train import select as sel
from ggh.train.dev import dev_metrics
from ggh.train.generate import compile_grammar, generate

DEV_EVAL = OUT_DIR / "dev_eval.jsonl"
TRAIN_RUN = Path("docs") / "measurements" / "train_run.json"


def load_for_inference(cfg: RecipeConfig, adapter: Path) -> tuple[Any, Any]:  # pragma: no cover
    """The pinned base through Unsloth (exact name, as in training) plus one adapter."""
    import torch  # pyright: ignore[reportMissingImports]
    from peft import PeftModel  # pyright: ignore[reportMissingImports]
    from unsloth import FastModel  # pyright: ignore[reportMissingImports]

    from ggh.train.run import check_source, loaded_source

    model, tok = FastModel.from_pretrained(
        model_name=cfg.base_model,
        revision=cfg.base_revision,
        use_exact_model_name=True,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit,
        dtype=getattr(torch, cfg.compute_dtype),
        full_finetuning=False,
    )
    check_source(cfg, loaded_source(model))
    model = PeftModel.from_pretrained(model, str(adapter))
    FastModel.for_inference(model)
    return model, getattr(tok, "tokenizer", tok)


def vocab_size(model: Any) -> int:
    return int(model.get_output_embeddings().weight.shape[0])


def release(model: Any) -> None:  # pragma: no cover - CUDA memory
    import gc

    import torch  # pyright: ignore[reportMissingImports]

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dev_eval(
    root: Path,
    run_dir: Path,
    out_dir: Path,
    config_path: Path,
    *,
    load: Callable[[RecipeConfig, Path], tuple[Any, Any]] = load_for_inference,
    gen: Callable[..., list[str]] = generate,
    grammar_for: Callable[[Any, int], Any] = compile_grammar,
    free: Callable[[Any], None] = release,
    limit: int | None = None,
) -> dict[str, Any]:
    cfg = load_recipe(config_path)
    rows = read_jsonl(root / DEV_EVAL)[:limit]
    prompts = [r["prompt"] for r in rows]
    replies: dict[int, tuple[list[str], list[str]]] = {}

    def metrics_for(c: sel.Candidate) -> DevMetrics:
        model, tok = load(cfg, c.path)
        grammar = grammar_for(tok, vocab_size(model))
        constrained = gen(model, tok, prompts, grammar=grammar)
        unconstrained = gen(model, tok, prompts)
        free(model)
        replies[c.step] = (constrained, unconstrained)
        return dev_metrics(rows, constrained, unconstrained)

    selected, tried = sel.select(sel.checkpoints(run_dir), metrics_for)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = None
    if selected is not None:
        dest = out_dir / "adapter"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        for name in sel.ADAPTER_FILES:
            shutil.copy2(selected.path / name, dest / name)
        digest = sel.adapter_hash(dest)
    record = sel.selection_record(selected, tried, digest)
    record["n_prompts"] = len(prompts)
    shown = selected.step if selected else tried[0].candidate.step
    con, unc = replies[shown]
    with (out_dir / "generations.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r, a, b in zip(rows, con, unc, strict=True):
            fh.write(json.dumps({"pmid": r["pmid"], "label": r["label"], "step": shown,
                                 "constrained": a, "unconstrained": b},
                                ensure_ascii=True) + "\n")  # fmt: skip
    (out_dir / "selection.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeError(ValueError):
    pass


def freeze(
    root: Path,
    report: dict[str, Any],
    selection: dict[str, Any],
    adapter: Path,
    *,
    kernel_version: str,
    repo_commit: str,
    out: Path = TRAIN_RUN,
    config: Path | None = None,
) -> TrainRun:
    """Check the downloaded adapter against the recorded hash (and, given the recipe, the
    report's config hash against it), then write `TrainRun` to `out` under `root`."""
    if selection.get("selected_step") is None:
        raise FreezeError("no checkpoint passed the §5.2 bars; nothing to freeze")
    got = sel.adapter_hash(adapter)
    if got != selection["adapter_sha256"]:
        raise FreezeError(f"adapter hash {got[:12]}… differs from the recorded "
                          f"{selection['adapter_sha256'][:12]}…")  # fmt: skip
    if config is not None:
        want = sha256_file(config)
        if report["config_sha256"] != want:
            raise FreezeError(f"the run's config hash {report['config_sha256'][:12]}… is not "
                              f"{config.name}'s {want[:12]}…")  # fmt: skip
    (chosen,) = [t for t in selection["tried"] if t["step"] == selection["selected_step"]]
    run = TrainRun(
        config_hash=report["config_sha256"],
        data_manifest_hash=sha256_file(root / MANIFEST_PATH),
        repo_commit=repo_commit,
        kaggle_kernel_version=kernel_version,
        gpu=report["gpu"],
        wall_clock_s=report["seconds"],
        peak_vram_gb=report["peak_vram_gb"],
        steps=report["steps"],
        selected_step=selection["selected_step"],
        val_loss_curve=tuple((int(s), float(v)) for s, v in report["eval_losses"]),
        dev_metrics=DevMetrics.model_validate(chosen["metrics"]),
    )
    path = root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run


SAMPLE_EVERY = 25  # the read sample: every 25th dev generation (20 of 500)


def write_records(
    folder: Path,
    report: dict[str, Any],
    selection: dict[str, Any],
    generations: Path,
    log: Path | None = None,
) -> list[Path]:
    """The committed run record beside `train_run.json`, as T2.2 kept v1's: the selection,
    the full report, all dev generations (gzip, no timestamp), every 25th of them as the read
    sample and, when downloaded, the Kaggle session log."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [x for x in generations.read_text(encoding="utf-8").splitlines() if x]
    if {json.loads(x)["step"] for x in lines} != {selection["selected_step"]}:
        raise FreezeError(f"{generations} is not from the selected step")
    every = "".join(x + "\n" for x in lines).encode()
    written = {
        "train_selection.json": (json.dumps(selection, indent=2) + "\n").encode(),
        "train_full.json": (json.dumps(report, indent=2) + "\n").encode(),
        "dev_generations.jsonl.gz": gzip.compress(every, mtime=0),
        "dev_generations_sample.jsonl": "".join(x + "\n" for x in lines[::SAMPLE_EVERY]).encode(),
    }
    if log is not None:
        written["train_full_log.json.gz"] = gzip.compress(log.read_bytes(), mtime=0)
    for name, data in written.items():
        (folder / name).write_bytes(data)
    return [folder / n for n in written]
