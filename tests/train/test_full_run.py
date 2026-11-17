"""TRN-01 to TRN-06: the recorded full Kaggle run, its selection, the frozen `TrainRun`, the
read sample and the GPU ledger. These check the committed evidence, not a new run."""

import gzip
import json
import math
from pathlib import Path

from ggh.config import load_recipe
from ggh.data.build import MANIFEST_PATH
from ggh.schemas import TrainRun
from ggh.train.devrun import sha256_file
from ggh.train.notebook import render
from ggh.train.run import config_hash
from ggh.train.select import BAR, adapter_hash, failed_bars

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "docs/measurements"
REPORT = json.loads((M / "train_full.json").read_text(encoding="utf-8"))
RUN = TrainRun.model_validate_json((M / "train_run.json").read_text(encoding="utf-8"))
LOG = json.loads(gzip.decompress((M / "train_full_log.json.gz").read_bytes()))
LOG_TEXT = "".join(x["data"] for x in LOG)
SAMPLE = [json.loads(x) for x in (M / "dev_generations_sample.jsonl")
          .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
COMMIT = "16ba41f43aa9b8a3d21219a504f1bff7938e7f0e"
RUN_STATE = (ROOT / "specs/T2.2-training/RUN-STATE.md").read_text(encoding="utf-8")


def ledger() -> list[dict]:
    return [json.loads(x) for x in (M / "gpu-ledger.jsonl").read_text(encoding="utf-8")
            .splitlines() if x]  # fmt: skip


# TRN-01


def test_one_session_ran_the_committed_notebook_at_the_recorded_commit():
    assert f"$ git checkout -q {COMMIT}\n" in LOG_TEXT and RUN.repo_commit == COMMIT
    assert RUN.kaggle_kernel_version == "1" and "Traceback" not in LOG_TEXT
    span_h = (LOG[-1]["time"] - LOG[0]["time"]) / 3600
    assert span_h < 12  # inside one session
    assert REPORT["seconds"] == RUN.wall_clock_s and REPORT["seconds"] / 3600 < span_h
    cfg = load_recipe(ROOT / "configs/recipe.yaml")
    nb = render(COMMIT, cfg.tool, None)
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ran = [x["data"][2:].rstrip("\n") for x in LOG if x["data"].startswith("$ ")]
    assert ran[-2:] == ["ggh train --config configs/recipe.yaml --out /kaggle/working/run "
                        "--tool unsloth", "ggh dev-eval --run /kaggle/working/run --out "
                        "/kaggle/working/dev"]  # fmt: skip
    assert len(ran) == 12 and all(cmd in code for cmd in ran)  # every step is the notebook's
    committed = json.loads((ROOT / "notebooks/train_kaggle.ipynb").read_text(encoding="utf-8"))
    assert committed["cells"] == nb["cells"]  # the committed notebook is that rendering


def test_the_whole_epoch_ran_and_the_wall_clock_matches_the_report():
    assert REPORT["max_steps"] is None and REPORT["epochs"] == 1.0
    assert REPORT["steps"] == RUN.steps == -(-REPORT["train_examples"] // 16) == 527
    assert REPORT["tokens_per_s"] == round(REPORT["tokens_seen"] / REPORT["seconds"], 1)
    assert 0 < REPORT["peak_vram_gb"] == RUN.peak_vram_gb < 15.6


# TRN-02


def test_pinned_base_and_every_loss_finite():
    cfg = load_recipe(ROOT / "configs/recipe.yaml")
    assert REPORT["base_loaded"] == [cfg.base_model, cfg.base_revision]
    assert "unsloth/gemma" not in LOG_TEXT and "bnb-4bit" not in LOG_TEXT  # no mirror swap
    losses = [v for _, v in REPORT["losses"]]
    assert REPORT["non_finite"] == REPORT["non_finite_steps"] == 0
    assert all(math.isfinite(v) for v in losses) and len(losses) == 52
    assert all(math.isfinite(g) for g in REPORT["grad_norms"]) and len(REPORT["grad_norms"]) == 52
    assert REPORT["skipped_steps"] <= 3  # the fp16 scaler calibrating at the start
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:3]) / 3


def test_config_seed_and_data_hashes_are_saved_with_the_run():
    assert REPORT["config_sha256"] == RUN.config_hash == config_hash(ROOT / "configs/recipe.yaml")
    assert load_recipe(ROOT / "configs/recipe.yaml").seed == 7  # covered by the config hash
    assert RUN.data_manifest_hash == sha256_file(ROOT / MANIFEST_PATH)
    assert [list(p) for p in RUN.val_loss_curve] == REPORT["eval_losses"]


# TRN-03


def test_the_lowest_validation_loss_was_tried_first_and_passed_every_bar():
    evals = dict(REPORT["eval_losses"])
    assert RUN.selected_step == min(evals, key=lambda s: (evals[s], s)) == 527
    m = RUN.dev_metrics
    assert m.n == 500 and failed_bars(m) == []
    assert min(m.schema_valid, m.has_citation, m.citation_valid) >= BAR


def test_dev_metrics_are_the_ones_recorded_in_spec_and_run_state():
    m = RUN.dev_metrics
    assert m.schema_valid == m.schema_valid_unconstrained == 1.0 and m.over_80_words == 0.0
    assert f"yes {m.label_accuracy['yes']:.3f}" in RUN_STATE
    assert f"no {m.label_accuracy['no']:.3f}" in RUN_STATE


# TRN-04


def test_trainrun_validates_and_round_trips():
    text = (M / "train_run.json").read_text(encoding="utf-8")
    assert RUN.model_dump_json(indent=2) + "\n" == text
    assert RUN.gpu == REPORT["gpu"] == "Tesla T4"


def test_adapter_hash_is_frozen_and_the_local_copy_matches_it():
    sel = json.loads((M / "train_selection.json").read_text(encoding="utf-8"))
    assert sel["selected_step"] == RUN.selected_step and sel["n_prompts"] == 500
    assert sel["tried"][0]["metrics"] == RUN.dev_metrics.model_dump(mode="json")
    assert sel["adapter_sha256"][:12] in RUN_STATE
    local = ROOT / "artifacts/adapter"  # gitignored; present where the run was downloaded
    if local.exists():
        assert adapter_hash(local) == sel["adapter_sha256"]


# TRN-05


def test_twenty_dev_generations_from_the_selected_step_are_kept():
    assert len(SAMPLE) == 20 and {r["step"] for r in SAMPLE} == {RUN.selected_step}
    assert len({r["pmid"] for r in SAMPLE}) == 20


def test_the_sample_reading_is_noted():
    assert "Read the sample" in RUN_STATE and "18/20" in RUN_STATE
    for pmid in ("23472862", "24896137"):  # the two wrong labels, named
        assert pmid in RUN_STATE


# TRN-06


def test_the_ledger_holds_the_session_no_less_than_its_log_span():
    (row,) = [r for r in ledger() if r.get("kernel") == "ggh-t22-train v1"]
    assert row["node"] == "T2.2" and row["kind"] == "train"
    assert row["gpu_hours"] >= round(LOG[-1]["time"] / 3600, 2)


def test_every_t22_kernel_is_in_the_ledger():
    kernels = {r["kernel"] for r in ledger() if r.get("node") == "T2.2"}
    assert kernels == {"ggh-t22-devsmoke v1", "ggh-t22-train v1"}
