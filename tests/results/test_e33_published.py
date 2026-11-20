"""RES-02, RES-05, RES-06, RES-07 on the committed E3.3 files: the B8 review, the sample read, and
the published bundle and behaviour table (OWNER D-61, D-62)."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from ggh.results import bundle as B
from ggh.schemas import ResultsBundle, SampleReadRow

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "docs/measurements"
RES = ROOT / "docs/results"
REVIEWER = "claude-opus-5-5"


def _records(run):
    raw = gzip.decompress((RES / run / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


@pytest.fixture(scope="module")
def built():
    return B.build(ROOT)


def test_every_offscope_answer_has_one_blind_model_review_under_the_committed_prompt():
    rows = [json.loads(x) for x in (M / "b8_review.jsonl").read_text("utf-8").splitlines() if x]
    want = {
        (r["completion"]["model"], r["case_id"]) for run in ("offscope", "offscope_v2")
        for r in _records(run)
    }  # fmt: skip
    assert len(rows) == 100 and {(r["model"], r["case_id"]) for r in rows} == want
    prompt = hashlib.sha256((ROOT / "configs/offscope_review_prompt.md").read_bytes()).hexdigest()
    assert all(r["prompt_sha256"] == prompt and REVIEWER in r["reviewed_by"] for r in rows)
    assert all(isinstance(r["gives_advice"], bool) and r["reason"].strip() for r in rows)


def test_the_sample_read_covers_20_per_model_and_every_fault_names_its_explanation():
    faults = (M / "sample_read_faults.md").read_text(encoding="utf-8")
    for run, models in (("main", 4), ("main_v2", 3)):
        path = M / f"sample_read_{run}.jsonl"
        rows = [SampleReadRow.model_validate_json(x) for x in path.read_text("utf-8").splitlines()]
        assert len(rows) == 20 * models and all(r.finding and r.read_by for r in rows)
        assert all(REVIEWER in (r.read_by or "") for r in rows)  # OWNER D-61
        for r in rows:
            if r.finding == "pipeline_fault":
                doc, _, rest = r.explanation.partition("#")
                section = rest.split(":", 1)[0]
                assert doc == "docs/measurements/sample_read_faults.md"
                assert f"## {section}\n" in faults and r.case_id in faults
    run_state = (ROOT / "specs/E3.3-results/RUN-STATE.md").read_text(encoding="utf-8")
    assert "builder read" in run_state  # D-18: the builder reads the sample too


def test_the_published_bundle_and_behaviour_table_are_what_the_generator_writes(built):
    committed = json.loads((RES / "results_bundle.json").read_text(encoding="utf-8"))
    assert committed == json.loads(built.bundle.model_dump_json())
    ResultsBundle.model_validate(committed)
    assert (RES / "behaviour_table.md").read_text(encoding="utf-8") == built.behaviour_md
    assert committed["lineage_ok"] and committed["publish_blockers"] == []


def test_every_place_that_reports_b5_b8_or_calibration_says_who_reviewed():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    issues = (ROOT / "docs/known-issues.md").read_text(encoding="utf-8")
    assert "a model's, not a human expert's" in readme  # D-60
    assert "B5 and B8 are model reviews (Claude Opus 5.5" in issues  # D-61
    assert "OWNER D-62" in readme and "does not beat base on citations" in readme
