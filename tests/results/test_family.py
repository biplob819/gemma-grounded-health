"""OWNER D-53: the per-case definitions behind the V2.2 primary family reproduce EvalKit's own
summary numbers, and the paired bootstrap matches EvalKit's interval on the headline."""

import pytest

from ggh.results import family as F
from ggh.results.runs import Run
from tests.results import fake

CIT = Run(fake.ROOT, "main_citation")
JUDGES = ("claude-opus-5-5", "gpt-6.1-sol")


@pytest.mark.parametrize("metric", [*F.RATIO_METRICS, "headline"])
def test_pooled_per_case_values_equal_evalkits_summary(metric):
    for model in CIT.models():
        recs = [r for r in CIT.raw_records if r["completion"]["model"] == model]
        for j in JUDGES:
            vals = [F.case_values(r, metric, j) for r in recs]
            got = F.pooled([n for n, _ in vals], [d for _, d in vals])
            want = CIT.model_summary(model)[metric][j]["point"]
            assert got == pytest.approx(want, abs=1e-9), (model, j)


def test_family_headline_interval_matches_evalkits_paired_comparison():
    plan = {
        "primary": [["gemma-e2b-grounded-q4", "gemma-e2b-base-q4"]],
        "primary_metrics": [{"metric": "headline", "better": "higher"}],
        "judges": list(JUDGES),
        "bootstrap": {"resamples": 10000, "seed": 7},
    }
    tests = F.family_tests(Run(fake.ROOT, "main"), plan)
    ek = {
        c["judge"]: c
        for c in Run(fake.ROOT, "main").summary["comparisons"]
        if c["metric"] == "headline" and {c["model_a"], c["model_b"]} == set(plan["primary"][0])
    }
    for t in tests:
        e = ek[t["judge"]]["diff"]  # EvalKit stores base minus v1
        assert t["diff"].point == pytest.approx(-e["point"], abs=1e-9)
        assert t["diff"].lo == pytest.approx(-e["hi"], abs=0.005)
        assert t["diff"].hi == pytest.approx(-e["lo"], abs=0.005)
        assert t["family_size"] == 2
