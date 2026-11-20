"""OWNER D-62: the post-hoc aim-rule check reproduces the published numbers on its published
branch, its committed output is what the code writes, and the rule flips only the sentences it
names."""

from typing import Any

import pytest

from ggh.results import aim_check as A
from ggh.results import family as F
from ggh.results.runs import Run
from tests.results import fake

SOL = "gpt-6.1-sol"


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    return A.build(fake.ROOT)


def test_published_branch_equals_main_v2_summary(built):
    run = Run(fake.ROOT, A.RUN)
    table = built["citation"][A.RUN]
    assert set(table) == set(run.models())
    for model, per in table.items():
        for metric in F.RATIO_METRICS:
            want = run.model_summary(model)[metric][SOL]["point"]
            assert per[SOL]["published"][metric] == round(want, 6), (model, metric)
    for t in built["family"][A.RUN]["published"]:
        a, b = run.model_summary(t["model_a"]), run.model_summary(t["model_b"])
        want = a[t["metric"]][SOL]["point"] - b[t["metric"]][SOL]["point"]
        assert t["diff"]["point"] == pytest.approx(want, abs=1e-6)


def test_committed_json_is_what_the_code_writes(built):
    assert (fake.ROOT / A.OUT).read_text(encoding="utf-8") == A.dumps(built)


def test_corrected_v2_minus_v1_stays_separated(built):
    rows = {
        (t["model_b"], t["metric"]): t
        for t in built["family"][A.RUN]["corrected"]
        if t["model_a"] == fake.V2
    }
    prec, aim = rows[(fake.V1, "citation_precision")], rows[(fake.V1, "aim_citation_rate")]
    assert prec["separated"] and prec["diff"]["lo"] > 0
    assert aim["separated"] and aim["diff"]["hi"] < 0
    headline = {(t["model_b"], t["metric"]): t for t in built["family"][A.RUN]["published"]}
    for k, t in rows.items():
        if k[1] == "headline":
            assert t["diff"] == headline[k]["diff"]


def _rec() -> dict[str, Any]:
    """One answer, two claims citing sentence 1 (aim) and 2 (result). Claim 0 restates the aim
    and the claim judge used sentence 1 as evidence; claim 1 is a finding that also cites the aim
    sentence, but the claim judge's evidence for it is sentence 2 only."""
    return {
        "claims": [
            {"id": "c0", "text": "The study assessed whether drug X lowers blood pressure."},
            {"id": "c1", "text": "Drug X lowered blood pressure by 10 mmHg."},
        ],
        "verdicts": [
            {"claim_id": "c0", "judge": SOL, "verdict": "supported", "evidence": [1]},
            {"claim_id": "c1", "judge": SOL, "verdict": "supported", "evidence": [2]},
        ],
        "citation_support": {
            SOL: [
                {"claim_id": "c0", "n": 1, "supports": False, "role": "aim"},
                {"claim_id": "c1", "n": 1, "supports": False, "role": "aim"},
                {"claim_id": "c1", "n": 2, "supports": True, "role": "result"},
            ]
        },
    }


def test_rule_flips_exactly_the_intended_entry():
    rec = _rec()
    fixed = A.corrected_record(rec, [SOL])
    before, after = rec["citation_support"][SOL], fixed["citation_support"][SOL]
    changed = [(a["claim_id"], a["n"]) for a, b in zip(after, before, strict=True) if a != b]
    assert changed == [("c0", 1)]
    assert rec["citation_support"][SOL][0]["supports"] is False  # the input is not mutated
    assert F.case_values(rec, "citation_precision", SOL) == (1.0, 2.0)
    assert F.case_values(fixed, "citation_precision", SOL) == (2.0, 2.0)
    assert F.case_values(rec, "aim_citation_rate", SOL) == (1.0, 1.0)
    assert F.case_values(fixed, "aim_citation_rate", SOL) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (lambda r: r["claims"][0].update(text="Drug X lowered blood pressure."), "no aim wording"),
        (lambda r: r["verdicts"][0].update(verdict="unsupported"), "claim not supported"),
        (lambda r: r["verdicts"][0].update(evidence=[2]), "sentence not in evidence"),
        (lambda r: r["citation_support"][SOL][0].update(role="method"), "role not aim"),
    ],
)
def test_rule_needs_every_condition(change, why):
    rec = _rec()
    change(rec)
    fixed = A.corrected_record(rec, [SOL])
    assert fixed["citation_support"][SOL] == rec["citation_support"][SOL], why


def test_label_only_pattern():
    assert all(A.LABEL_ONLY.match(t) for t in ("Yes.", "No.", "maybe", " No! "))
    assert not any(A.LABEL_ONLY.match(t) for t in ("No difference.", "Yes, it does.", "Nothing."))
