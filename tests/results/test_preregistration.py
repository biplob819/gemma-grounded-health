"""VEV-03 and RUN-01, data-driven: every plan in configs/eval_plan*.yaml is either a draft with no
hash or final with its committed hash, and every run a plan lists that has committed results
started after that hash was committed. A new plan or run is covered with no edit here."""

import hashlib
import json

import pytest
import yaml

from ggh.evals.config import load_run_config
from ggh.results import plan as P
from tests.results import fake

ROOT = fake.ROOT
PLANS = P.all_plans(ROOT)


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: p.path.name)
def test_every_plan_is_a_draft_without_a_hash_or_final_with_it(plan):
    assert P.plan_problems(ROOT, plan) == []


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: p.path.name)
def test_every_planned_run_with_results_started_after_its_plan_was_committed(plan):
    assert P.preregistration_problems(ROOT, plan) == []


def test_the_check_is_not_vacuous_today():
    """The E3.2 plan's main run has results, so the check above judges a real commit order."""
    v1 = P.load_plan(ROOT, "configs/eval_plan.yaml")
    assert [r.name for r in v1.runs] == ["main"] and P.run_started(ROOT, v1.runs[0])
    rows = P.preregistrations(ROOT)
    assert rows[0]["sha256"] == v1.sha256 and rows[0]["commit"].startswith("9bc0f97")
    run_state = (ROOT / "specs/E3.2-runs/RUN-STATE.md").read_text(encoding="utf-8")
    assert f"9bc0f97 (plan and its hash) committed at {rows[0]['committed_at']}" in run_state


# the v2 plan's content (holds in draft and once final)

V2 = P.load_plan(ROOT, "configs/eval_plan_v2.yaml")
V2_MODEL = "gemma-e2b-grounded-v2-q4"


def test_v2_plan_names_the_primary_comparisons_judge_metric_and_rule():
    v1 = P.load_plan(ROOT, "configs/eval_plan.yaml").data
    d = V2.data
    # OWNER D-57: OpenAI only; Sol judges alone and Luna extracts (v1's plan: Opus and Sol)
    assert d["metric"] == "headline" and d["judges"] == ["gpt-6.1-sol"]
    assert "gpt-6.1-sol" in v1["judges"] and d["extractor"] == "gpt-6-luna"
    assert d["provider"] == "openai_only"
    assert d["n_cases"] == v1["n_cases"] == 1000 and d["seed"] == v1["seed"] == 7
    # OWNER D-53: v2 vs v1 and v2 vs base only; three outcomes with directions; Holm over them
    assert V2.primary == [(V2_MODEL, "gemma-e2b-grounded-q4"), (V2_MODEL, "gemma-e2b-base-q4")]
    assert [(m["metric"], m["better"]) for m in d["primary_metrics"]] == [
        ("headline", "higher"),
        ("citation_precision", "higher"),
        ("aim_citation_rate", "lower"),
    ]
    n_tests = len(V2.primary) * len(d["primary_metrics"]) * len(d["judges"])
    assert n_tests == 6 and d["holm"] == "family" and "six primary tests" in d["rule"]
    assert "tie" in d["rule"]
    assert d["shown_not_compared"] == ["claude-haiku-5-5", "gpt-6-luna"]  # OWNER D-57
    caps = [load_run_config(ROOT / r.spec["config"]).max_cost_usd for r in V2.runs]
    assert sum(caps) == 25  # OWNER D-53's $25, split across the V2.2 runs (D-57)
    assert any(r.name == "parity_v2" for r in V2.runs)  # judged parity runs (OWNER D-53)
    assert d["built_after_test_seen"] is True  # OWNER D-43


def test_v2_plan_is_final_and_names_the_frozen_checkpoint_and_the_evalkit_release():
    d = V2.data
    assert V2.status == "final" and d["evalkit"].startswith("f9cbc8e")  # D-56
    manifest = json.loads(
        (ROOT / "docs/measurements/v2/gguf_ft2_manifest.json").read_text(encoding="utf-8")
    )
    assert d["checkpoint"]["gguf_q4_k_m_sha256"] == manifest["files"]["ft2-Q4_K_M.gguf"]["sha256"]
    assert (ROOT / d["checkpoint"]["train_run"]).is_file()
    assert d["checkpoint"]["adapter_sha256"].startswith("e5f97861693346623bf5")


def test_v2_main_run_rescores_base_and_v1_from_cached_answers_with_one_extractor_and_judge():
    main = next(r for r in V2.runs if r.name == "main_v2")
    cfg = load_run_config(ROOT / main.spec["config"])
    e32_gemma = ["gemma-e2b-base-q4", "gemma-e2b-grounded-q4"]
    assert main.spec["new_models"] == [V2_MODEL]
    assert main.spec["cached_models"] == main.spec["rescored_models"] == e32_gemma
    assert sorted(cfg.models) == sorted([*e32_gemma, V2_MODEL])
    assert list(cfg.judges) == V2.data["judges"] and cfg.extractor == V2.data["extractor"]
    e32 = load_run_config(ROOT / "configs/evalkit.yaml")
    # everything that keys a cached answer matches E3.2, so base and v1 answers are reused
    for field in ("structured_output", "max_output_tokens", "seed", "default_effort"):
        assert getattr(cfg, field) == getattr(e32, field), field
    for field in ("judge_effort", "extractor_effort"):
        assert getattr(cfg, field) == getattr(e32, field), field
    for m, url in e32.endpoints.items():
        assert cfg.endpoints[m] == url  # base and v1 served as in E3.2
    assert V2_MODEL in cfg.endpoints and cfg.endpoints[V2_MODEL] not in e32.endpoints.values()
    task = yaml.safe_load((ROOT / main.spec["task"]).read_text(encoding="utf-8"))
    assert task["name"] == "pubmedqa" and "citation_support" in task["scorers"]  # OWNER D-44
    assert (
        V2.data["cache"]["source_run"]
        == json.loads((ROOT / "docs/results/main_citation/run.json").read_text(encoding="utf-8"))[
            "run_id"
        ]
    )
    assert V2.data["cache"]["expected_hits"] == e32_gemma
    for r in V2.runs:  # every V2.2 run uses the same extractor and judge
        c = load_run_config(ROOT / r.spec["config"])
        assert list(c.judges) == ["gpt-6.1-sol"] and c.extractor == "gpt-6-luna", r.name


def test_v2_side_runs_are_v2_only_on_the_e32_tasks():
    want = {
        "offscope_v2": "offscope",
        "probe_original_v2": "probe_original",
        "probe_paraphrase_v2": "probe_paraphrase",
        "parity_v2": "pubmedqa_cit",
    }
    runs = {r.name: r.spec for r in V2.runs}
    assert set(runs) == {"main_v2", *want}
    for name, stem in want.items():
        s = runs[name]
        assert (ROOT / s["task"]).stem == stem and (ROOT / s["task"]).is_file()
        assert load_run_config(ROOT / s["config"]).models == tuple(s["new_models"])
        assert s["cached_models"] == [] and s["results"] == f"docs/results/{name}"
        assert s["command"].startswith(f"uv run evalkit run {s['config']} --task {s['task']}")


# the checker itself, on throwaway copies


def test_a_run_that_started_before_the_hash_was_committed_is_caught(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root, started="2026-11-29T00:00:00Z")  # the plan hash lands on 11-30
    plan = P.load_plan(root, "configs/eval_plan_v2.yaml")
    problems = P.preregistration_problems(root, plan)
    assert len(problems) == 5 and all("before the plan hash" in p for p in problems)


def test_results_under_a_draft_plan_are_caught(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root, finalise=False)
    plan = P.load_plan(root, "configs/eval_plan_v2.yaml")
    assert P.preregistration_problems(root, plan) == [
        "configs/eval_plan_v2.yaml: is draft but has results (main_v2, offscope_v2, "
        "probe_original_v2, probe_paraphrase_v2, parity_v2)"
    ]


def test_an_edited_plan_or_a_missing_record_is_caught(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root)
    rel = "configs/eval_plan_v2.yaml"
    assert P.preregistration_problems(root, P.load_plan(root, rel)) == []
    (root / rel).write_text((root / rel).read_text(encoding="utf-8") + "# edit\n", "utf-8")
    edited = P.load_plan(root, rel)
    assert "no longer match" in P.plan_problems(root, edited)[0]
    assert "no committed hash" in P.preregistration_problems(root, edited)[0]
    (root / "docs/measurements/preregistrations.jsonl").write_text(
        (ROOT / "docs/measurements/preregistrations.jsonl").read_text(encoding="utf-8"), "utf-8"
    )
    sha = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    (root / "docs/measurements/eval_plan_v2.sha256").write_text(sha, encoding="utf-8")
    assert "needs exactly one row" in P.preregistration_problems(root, P.load_plan(root, rel))[0]


def test_a_final_plan_without_a_hash_and_a_draft_with_one_are_caught(tmp_path):
    root = fake.copy_repo(tmp_path)
    (root / "docs/measurements/eval_plan_v2.sha256").write_text("x\n", encoding="utf-8")
    draft = P.load_plan(root, "configs/eval_plan_v2.yaml")
    assert "draft plan must not have" in P.plan_problems(root, draft)[0]
    (root / "docs/measurements/eval_plan.sha256").unlink()
    v1 = P.load_plan(root, "configs/eval_plan.yaml")
    assert "a final plan needs" in P.plan_problems(root, v1)[0]
