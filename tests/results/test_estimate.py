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
    assert u["from"]["per_claim"] == "r-f0f3b432a413"
    assert u["from"]["per_claim_citation"] == "r-edfcba37159e"
    # OWNER D-57's Sol-only unit: both v2 dev-check runs' spend over their 261 claims
    assert u["from"]["per_claim_sol"] == ["r-b118d6679b72", "r-4b7e3a0c3d19"]
    assert u["per_claim_sol"] == pytest.approx((0.298562 + 0.181817) / 261, rel=1e-3)
    # main: $53.68 less $0.34 frontier generation over 12,389 claims; rescore: $48.24 / 12,389
    assert u["per_claim"] == pytest.approx((53.68107 - 0.336) / 12389, rel=1e-3)
    assert u["per_claim_citation"] == pytest.approx(48.239756 / 12389, rel=1e-4)
    b = u["backtest"]
    assert b["actual_usd"] == 5.5 and abs(b["predicted_usd"] / b["actual_usd"] - 1) < 0.15


def test_v2_and_the_rescored_models_are_priced_and_v22_fits_the_cap():
    est = E.estimate(ROOT, PLAN)
    assert est["sol_only"] is True  # OWNER D-57
    runs = {r["run"]: r for r in est["runs"]}
    main = runs["main_v2"]
    assert main["answers"] == 1000 and len(main["cached_models"]) == 2
    # base 2,342 and v1 1,174 claims in E3.2 are re-extracted and re-judged
    assert main["rescored_models"] == ["gemma-e2b-base-q4", "gemma-e2b-grounded-q4"]
    assert main["rescored_claims"] == 3516
    assert main["citation_support"] and runs["offscope_v2"]["citation_support"] is False
    assert 8 < main["usd"]["like_v1"] < main["usd"]["like_base"] < 16  # main_v2's cap
    assert not any(r["over_approval_line"] for r in est["runs"])
    assert est["total_usd"]["like_v1"] < est["total_usd"]["like_base"] < 25


def test_claims_per_answer_override_is_linear_in_the_new_answers():
    one = E.estimate(ROOT, PLAN, cpa=1.0)
    two = E.estimate(ROOT, PLAN, cpa=2.0)
    unit = E.units(ROOT)["per_claim_sol"]
    for a, b in zip(one["runs"], two["runs"], strict=True):
        fixed = a["rescored_claims"] * unit  # the re-scored models' known claims
        assert b["usd"]["like_v1"] - fixed == pytest.approx(
            2 * (a["usd"]["like_v1"] - fixed), abs=0.02
        )


def test_cli_prints_the_estimate_and_the_dry_run_commands(capsys):
    assert main(["results", "estimate", "--root", str(ROOT), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"plan_status": "final"' in out and "--n 1000 --dry-run" in out
