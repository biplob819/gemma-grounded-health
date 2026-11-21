"""RLR-07: the v3 pre-registration pins the reward and the recipe by hash, names the primary test,
the guardrails and the dev selection rule, and is hashed before any training."""

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/eval_plan_v3.yaml"


def _plan() -> dict:
    return yaml.safe_load(PLAN.read_text(encoding="utf-8"))


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def test_the_plan_is_final_and_its_hash_is_committed():
    plan = _plan()
    assert plan["status"] == "final" and plan["node"] == "V3.1"
    want = (ROOT / "docs/measurements/eval_plan_v3.sha256").read_text(encoding="utf-8").strip()
    assert want == hashlib.sha256(PLAN.read_bytes()).hexdigest()


def test_the_reward_and_recipe_are_the_ones_registered():
    plan = _plan()
    assert _sha(plan["reward"]["module"]) == plan["reward"]["sha256"]
    assert _sha(plan["recipe"]["file"]) == plan["recipe"]["sha256"]


def test_the_primary_test_guardrails_and_stretch_are_as_briefed():
    plan = _plan()
    assert plan["primary"] == [["gemma-e2b-grounded-v3-q4", "gemma-e2b-grounded-v2-q4"]]
    assert plan["primary_metrics"] == [{"metric": "headline", "better": "higher"}]
    assert plan["guardrails"] == [
        {"metric": "citation_precision", "min": 0.70},
        {"metric": "aim_citation_rate", "max": 0.10},
    ]
    assert plan["stretch"] == [{"metric": "headline", "reach": 0.404}]
    assert plan["judges"] == ["gpt-6.1-sol"] and plan["extractor"] == "gpt-6-luna"
    assert plan["n_cases"] == 1000 and plan["seed"] == 7 and plan["quantization"] == "Q4_K_M"
    assert "excludes zero" in plan["rule"] and "0.329" in plan["rule"]


def test_the_registered_baselines_are_v2s_published_numbers():
    import json

    s = json.loads((ROOT / "docs/results/main_v2/summary.json").read_text(encoding="utf-8"))
    by = {m["model"]: m for m in s["models"]}
    v2, base = by["gemma-e2b-grounded-v2-q4"], by["gemma-e2b-base-q4"]
    assert round(v2["headline"]["gpt-6.1-sol"]["point"], 3) == 0.329
    assert round(base["headline"]["gpt-6.1-sol"]["point"], 3) == 0.404
    assert v2["citation_precision"]["gpt-6.1-sol"]["point"] >= 0.70
    assert v2["aim_citation_rate"]["gpt-6.1-sol"]["point"] <= 0.10


def test_selection_is_on_dev_with_hack_guards_and_scoring_runs_once():
    plan = _plan()
    sel = plan["checkpoint_selection"]
    for guard in ("parse failures", '"maybe" share', "answer words", "label accuracy", "aim"):
        assert guard in sel, guard
    assert "dev_eval" in sel and "test" not in plan["training_data"].replace("no dev or test", "")
    assert [r["name"] for r in plan["runs"]] == ["main_v3"] and plan["runs"][0]["n"] == 1000
    assert "owner" in plan["spend"] and "max_cost_usd" in plan["spend"]
