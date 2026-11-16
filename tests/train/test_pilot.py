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
    assert REPORT["backend"] == "unsloth" == load_recipe(ROOT / "configs/recipe.yaml").tool
    assert REPORT["config_sha256"] == config_hash(ROOT / "configs/recipe.yaml")


def test_fp16_numerics_stayed_finite_and_the_loss_fell():
    losses = [v for _, v in REPORT["losses"]]
    assert REPORT["non_finite"] == 0 and all(math.isfinite(v) for v in losses)
    assert len(losses) == 30  # one per 10 steps
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:3]) / 3
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
    total = hours + est["eval_passes"] * est["eval_seconds_each"] / 3600
    assert est["total_hours_with_eval"] == round(total, 2)
    assert total * 2 < est["session_limit_hours"] == 12  # margin: under half a session
    spec = (ROOT / "Gemma Grounded Health Tech Spec.md").read_text(encoding="utf-8")
    assert (
        f"**{hours:.1f} h** of training" in spec
        and f"{REPORT['tokens_per_s']:.0f} tokens/s" in spec
    )


def test_every_pilot_session_is_in_the_gpu_ledger():
    rows = [json.loads(x) for x in (ROOT / "docs/measurements/gpu-ledger.jsonl")
            .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
    pilots = [r for r in rows if r.get("node") == "T2.1" and r["kind"] == "pilot"]
    assert {r["kernel"] for r in pilots} == {
        "ggh-t21-pilot-unsloth v1", "ggh-t21-pilot-trl v1", "ggh-t21-pilot-trl v2",
    }  # fmt: skip
    assert all(r["gpu_hours"] > 0 for r in pilots)


def test_d40_records_the_decision_with_the_measured_numbers():
    text = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| D-40 |")]
    assert "D-9 confirmed" in row and f"{REPORT['tokens_per_s']:.0f} training tokens" in row
    assert f"{REPORT['peak_vram_gb']} GB" in row
