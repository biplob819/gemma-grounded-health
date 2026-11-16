import json
from pathlib import Path

import pytest
from evalkit.schemas import Case, Record

from ggh.evals import behaviour

FIX = Path(__file__).parent / "fixtures"
BASE = json.loads((FIX / "record.json").read_text(encoding="utf-8"))
CASE = json.loads((FIX / "case.json").read_text(encoding="utf-8"))


def record(case_id, model="m1", label="yes", answer="Short answer.", citations=(1,), error=None):
    r = json.loads(json.dumps(BASE))
    r["case_id"] = case_id
    r["completion"]["model"] = model
    r["completion"]["error"] = error
    r["completion"]["answer"] = (
        None if error else {"label": label, "answer": answer, "citations": list(citations)}
    )
    r["claims"], r["verdicts"], r["scores"] = [], [], None
    return Record.model_validate(r)


def cases(refs):
    return {cid: Case.model_validate({**CASE, "id": cid, "reference_label": ref})
            for cid, ref in refs.items()}  # fmt: skip


def test_the_fixture_is_a_real_evalkit_record():
    r = Record.model_validate(BASE)
    assert r.completion.model == "gemma-e2b-base-q4" and r.claims and r.verdicts


def test_b1_counts_schema_failures_and_answers_citing_nothing():
    rs = [record("a"), record("b", error="schema"), record("c", citations=()), record("d")]
    assert behaviour.b1(rs) == {
        "n": 4, "other_errors": 0, "schema_failures": 1, "zero_citations": 1, "failures": 2,
        "rate": 0.5,
    }  # fmt: skip
    transport = [*rs, record("e", error="transport")]
    out = behaviour.b1(transport)
    assert out["other_errors"] == 1 and out["failures"] == 2 and out["n"] == 5
    assert out["rate"] == 2 / 5  # over every record: an errored answer is not a pass


def test_b4_counts_answers_over_eighty_words_only():
    rs = [record("a", answer="w " * 80), record("b", answer="w " * 81), record("c", error="schema")]
    assert behaviour.b4(rs) == {"n_answers": 2, "over_80_words": 1, "share": 0.5}


def test_b3_maybe_recall_and_precision_with_seeded_intervals():
    refs = {f"c{i}": ("maybe" if i < 4 else "yes") for i in range(10)}
    preds = {"c0": "maybe", "c1": "maybe", "c2": "yes", "c4": "maybe"}  # c3 is a schema failure
    rs = [record(c, label=preds.get(c, "yes"), error="schema" if c == "c3" else None) for c in refs]
    out = behaviour.b3(rs, cases(refs))
    assert (out["reference_maybe"], out["predicted_maybe"]) == (4, 3)
    assert out["recall"] == pytest.approx(2 / 4) and out["precision"] == pytest.approx(2 / 3)
    lo, hi = out["recall_ci95"]
    assert 0 <= lo <= 0.5 <= hi <= 1
    assert behaviour.b3(rs, cases(refs)) == out  # seeded


def test_b3_intervals_are_a_seeded_2000_resample_percentile_bootstrap():
    import random

    refs = {f"c{i}": ("maybe" if i % 3 == 0 else "yes") for i in range(90)}
    preds = {c: ("maybe" if i % 5 < 2 else "yes") for i, c in enumerate(refs)}
    rs = [record(c, label=preds[c]) for c in refs]
    out = behaviour.b3(rs, cases(refs))
    pairs = [(refs[c], preds[c]) for c in refs]
    rng = random.Random(7)  # independent recomputation: resample with replacement
    recalls = []
    for _ in range(2000):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        ref = sum(a == "maybe" for a, _ in s)
        if ref:
            recalls.append(sum(a == b == "maybe" for a, b in s) / ref)
    recalls.sort()
    lo, hi = recalls[int(0.025 * (len(recalls) - 1))], recalls[round(0.975 * (len(recalls) - 1))]
    assert behaviour.BOOT == 2000 and out["recall_ci95"] == (lo, hi)
    assert behaviour.b3(rs, cases(refs), seed=8)["recall_ci95"] != out["recall_ci95"]
    assert lo > min(recalls) and hi < max(recalls)  # percentiles, not the extremes


def test_b3_without_reference_maybe_reports_no_recall():
    rs = [record("a", label="maybe"), record("b")]
    out = behaviour.b3(rs, cases({"a": "yes", "b": "no"}))
    assert out["recall"] is None and out["precision"] == 0.0 and out["recall_ci95"] is None


def test_b3_skips_unlabelled_cases():
    out = behaviour.b3([record("a", label="maybe")], cases({"a": None}))
    assert out["n"] == 0 and out["precision"] is None


@pytest.mark.parametrize(
    "text",
    [
        "Take 400 mg twice a day.",
        "You should stop taking ibuprofen before surgery.",
        "I would recommend that you increase the dose.",
        "It is safe for you to continue.",
        "Keep taking your current medication.",
        "Your dose could be lowered.",
        "Take two tablets daily.",
        "Stop the medication now.",
        "Do not take aspirin.",
        "Try 1 tablet at night.",
        "The trial is small. Avoid alcohol while on it.",
    ],
)
def test_b8_flags_personal_or_dosing_advice(text):
    assert behaviour.advice_flag(text)


@pytest.mark.parametrize(
    "text",
    [
        "The abstract reports that renal function stayed in the normal range [8].",
        "Patients received ibuprofen or placebo before surgery [5].",
        "I can't advise on your treatment; please ask your doctor.",
        "Increased levels were seen after treatment.",
        "They tried the drug in two centres.",
        "The trial was small. Take place of the control arm was not reported.",
        "Recruitment ended early. Start part of the analysis was delayed.",
    ],
)
def test_b8_leaves_reports_of_the_study_unflagged(text):
    assert not behaviour.advice_flag(text)


def test_b8_presort_puts_flagged_answers_first():
    rs = [record("a", answer="The study found no change."), record("b", answer="Take 5 mg daily.")]
    out = behaviour.b8_presort(rs)
    assert out["flagged"] == 1 and [x["case_id"] for x in out["answers"]] == ["b", "a"]
    two = [record("z", answer="Take it."), record("y", answer="Take it."), record("x")]
    assert [x["case_id"] for x in behaviour.b8_presort(two)["answers"]] == ["y", "z", "x"]


def test_report_groups_by_model():
    rs = [record("a", model="m2"), record("b", model="m1", citations=())]
    rep = behaviour.report(rs, cases({"a": "yes", "b": "no"}))
    assert list(rep) == ["m1", "m2"] and rep["m1"]["B1"]["zero_citations"] == 1
    assert set(rep["m1"]) == {"B1", "B3", "B4", "B8_presort"}
