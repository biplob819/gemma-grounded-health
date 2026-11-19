"""V2.2's cache-aware estimate (RUN-02, D-22) from committed spend: no API calls."""

import pytest

from ggh.cli import main
from ggh.results import estimate as E
from ggh.results.plan import load_plan
from tests.results import fake

ROOT = fake.ROOT
PLAN = load_plan(ROOT, "configs/eval_plan_v2.yaml")


def test_units_come_from_the_paid_runs_and_backtest_within_fifteen_percent():
    u = E.units(ROOT)
    assert u["from"] == {"per_claim": "r-f0f3b432a413", "per_claim_citation": "r-edfcba37159e"}
    # main: $53.68 less $0.34 frontier generation over 12,389 claims; rescore: $48.24 / 12,389
    assert u["per_claim"] == pytest.approx((53.68107 - 0.336) / 12389, rel=1e-3)
    assert u["per_claim_citation"] == pytest.approx(48.239756 / 12389, rel=1e-4)
    b = u["backtest"]
    assert b["actual_usd"] == 5.5 and abs(b["predicted_usd"] / b["actual_usd"] - 1) < 0.15


def test_only_v2_answers_are_priced_and_the_main_run_is_under_the_approval_line():
    est = E.estimate(ROOT, PLAN)
    runs = {r["run"]: r for r in est["runs"]}
    main = runs["main_v2"]
    assert main["answers"] == 1000 and len(main["cached_models"]) == 4
    assert main["citation_support"] and runs["offscope_v2"]["citation_support"] is False
    assert 8 < main["usd"]["like_v1"] < 12 and main["usd"]["like_base"] < 25
    assert not any(r["over_approval_line"] for r in est["runs"])
    assert est["total_usd"]["like_v1"] < est["total_usd"]["like_base"] < 30


def test_claims_per_answer_override_is_linear():
    one = E.estimate(ROOT, PLAN, cpa=1.0)
    two = E.estimate(ROOT, PLAN, cpa=2.0)
    for a, b in zip(one["runs"], two["runs"], strict=True):
        assert b["usd"]["like_v1"] == pytest.approx(2 * a["usd"]["like_v1"], abs=0.011)


def test_cli_prints_the_estimate_and_the_dry_run_commands(capsys):
    assert main(["results", "estimate", "--root", str(ROOT), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"plan_status": "draft"' in out and "--n 1000 --dry-run" in out
