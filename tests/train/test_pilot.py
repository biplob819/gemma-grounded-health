"""RCP-06, RCP-07: the recorded Kaggle pilot and the full-run estimate built from it."""

import json
import math
from pathlib import Path

from ggh.config import load_recipe
from ggh.train.run import config_hash, full_run_hours

ROOT = Path(__file__).resolve().parents[2]
PILOT = json.loads((ROOT / "docs/measurements/pilot_t4.json").read_text(encoding="utf-8"))
REPORT = PILOT["report"]


def test_pilot_ran_300_steps_of_the_committed_recipe_on_a_t4_with_unsloth():
    assert REPORT["steps"] == REPORT["max_steps"] == 300 and REPORT["gpu"] == "Tesla T4"
    cfg = load_recipe(ROOT / "configs/recipe.yaml")
    assert REPORT["base_loaded"] == [cfg.base_model, cfg.base_revision]  # the pin, not a mirror
    assert REPORT["backend"] == "unsloth" == load_recipe(ROOT / "configs/recipe.yaml").tool
    assert REPORT["config_sha256"] == config_hash(ROOT / "configs/recipe.yaml")


def test_fp16_numerics_stayed_finite_and_the_loss_fell():
    losses = [v for _, v in REPORT["losses"]]
    assert REPORT["non_finite"] == 0 and all(math.isfinite(v) for v in losses)
    assert REPORT["non_finite_steps"] == 0  # every micro-batch loss, not only logged ones
    assert REPORT["skipped_steps"] <= 3  # the fp16 loss scaler calibrating, not diverging
    assert len(losses) == 30  # one per 10 steps
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:3]) / 3
    prec = REPORT["precision"]
    assert prec["amp"] == "fp16" and set(prec["trainable"]) == {"float32"}
    assert "float16" in prec["frozen"] and "uint8" in prec["frozen"]  # 4-bit packed base
    evals = dict((s, v) for s, v in REPORT["eval_losses"])
    assert set(evals) == {250, 300} and evals[300] <= evals[250] and evals[300] < 1.0


def test_throughput_and_memory_are_recorded_and_consistent():
    assert REPORT["tokens_per_s"] == round(REPORT["tokens_seen"] / REPORT["seconds"], 1)
    assert 0 < REPORT["peak_vram_gb"] < 15.6  # fits the T4
    assert REPORT["train_examples"] + REPORT["dropped_too_long"]["train"] == 8436


def test_full_run_estimate_fits_one_session_and_is_in_the_spec():
    est = PILOT["full_run_estimate"]
    hours = full_run_hours(est["train_tokens"], est["epochs"], est["tokens_per_s"])
    assert est["train_hours"] == round(hours, 2) and est["train_tokens"] == REPORT["train_tokens"]
    assert est["optimizer_steps"] == -(-REPORT["train_examples"] // 16) == 527
    assert est["eval_passes"] == 3  # at 250, 500 and the end
    total = hours + est["eval_passes"] * est["eval_seconds_each"] / 3600 + est["setup_minutes"] / 60
    assert est["total_hours"] == round(total, 2)
    assert total * 2 < est["session_limit_hours"] == 12  # margin: under half a session
    spec = (ROOT / "Gemma Grounded Health Tech Spec.md").read_text(encoding="utf-8")
    assert f"**{hours:.1f} h** of training" in spec and f"**{total:.1f} h**" in spec
    assert f"{REPORT['tokens_per_s']:.0f} tokens/s" in spec


def test_every_pilot_session_is_in_the_gpu_ledger():
    rows = [json.loads(x) for x in (ROOT / "docs/measurements/gpu-ledger.jsonl")
            .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
    pilots = [r for r in rows if r.get("node") == "T2.1" and r["kind"] == "pilot"]
    assert {r["kernel"] for r in pilots} == {
        "ggh-t21-pilot-unsloth v1", "ggh-t21-pilot-trl v1", "ggh-t21-pilot-trl v2",
        "ggh-t21-pilot-unsloth v2",
    }  # fmt: skip
    assert all(r["gpu_hours"] > 0 for r in pilots)
    (v2,) = [r for r in pilots if r["kernel"] == "ggh-t21-pilot-unsloth v2"]
    assert v2["gpu_hours"] >= PILOT["session_hours"]  # never below the log span


def test_d40_records_the_decision_with_the_measured_numbers():
    text = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| D-40 |")]
    assert "D-9 confirmed" in row and f"{REPORT['tokens_per_s']:.0f} training tokens" in row
    assert f"{REPORT['peak_vram_gb']} GB" in row and "4adf157" in row
    assert "requested fp16 mixed precision" in row and "GradScaler was active" in row
    assert "not which ops computed in fp16" in row  # round 3: what the skip does not show
    assert "refers to the trainable weights" not in row  # round 2: that reading was wrong
    assert "in fp16 trained" not in row  # the first pilot's unverified wording is gone


def test_the_superseded_first_pilot_is_kept_and_labelled():
    sup = PILOT["superseded"]["pilot_1"]
    assert "mirror" in sup["why"] and sup["repo_tag"] == "t2.1-pilot-1"
    assert PILOT["repo_tag"] == "t2.1-pilot-2" and PILOT["report"] is REPORT
