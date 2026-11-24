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
    runs = ("offscope", "offscope_v2", "offscope_v3b")  # V3.8: v3b at Q4_K_M and Q8_0
    want = {
        (run, r["completion"]["model"], r["case_id"]) for run in runs for r in _records(run)
    }  # fmt: skip
    assert len(rows) == 140 and {(r["run"], r["model"], r["case_id"]) for r in rows} == want
    assert sum(r["run"] == "offscope_v3b" for r in rows) == 40
    prompt = hashlib.sha256((ROOT / "configs/offscope_review_prompt.md").read_bytes()).hexdigest()
    assert all(r["prompt_sha256"] == prompt and REVIEWER in r["reviewed_by"] for r in rows)
    assert all(isinstance(r["gives_advice"], bool) and r["reason"].strip() for r in rows)


def test_the_sample_read_covers_20_per_model_and_every_fault_names_its_explanation():
    faults = (M / "sample_read_faults.md").read_text(encoding="utf-8")
    for run, models in (("main", 4), ("main_v2", 3), ("main_v3b", 4)):
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
    assert "matches base's citation quality" not in readme + issues  # not under either count
    table = (RES / "behaviour_table.md").read_text(encoding="utf-8")
    # one note per reviewed run: main, offscope, main_v2, offscope_v2, main_v3b, offscope_v3b
    assert table.count("blind review by Claude Opus 5.5 subagents, not the owner") == 6


def test_the_blind_batches_named_no_answering_model_and_map_back_to_the_labels():
    d = M / "blind_reviews"
    models = ("gemma-e2b", "claude-haiku", "gpt-6-luna", "grounded", "pqa-", "off-")
    batches = ("b8_batch1.json", "b8_batch2.json", "b8_batch3.json")
    for name in (*batches, *(f"b5_batch{k}.json" for k in range(1, 7))):
        text = (d / name).read_text(encoding="utf-8")
        assert not any(m in text for m in models), name
    key = json.loads((d / "b8_key.json").read_text(encoding="utf-8"))
    out = [
        json.loads(x)
        for k in (1, 2)
        for x in (d / f"b8_out{k}.jsonl").read_text("utf-8").splitlines()
    ]
    rows = list(map(json.loads, (M / "b8_review.jsonl").read_text("utf-8").splitlines()))
    review = {
        (r["model"], r["case_id"]): r["gives_advice"]
        for r in rows
        if r["run"] in ("offscope", "offscope_v2")
    }
    assert {(key[o["id"]]["model"], key[o["id"]]["case_id"]): o["gives_advice"] for o in out} == (
        review
    )


def test_the_v3b_blind_batches_map_back_to_the_answers_and_the_labels():
    """V3.8: v3b's B8 (b8_batch3, its own key) and B5 (b5_batch5/6) were blind like v1's and
    v2's: no model named in a batch, each key entry pointing at the answer it hid (B8: the
    sha256 of the record's raw text; B5: the sample-read row's record sha256)."""
    d = M / "blind_reviews"
    key = json.loads((d / "b8_key_v3b.json").read_text(encoding="utf-8"))
    raw = {
        (r["completion"]["model"], r["case_id"]): r["completion"]["raw_text"]
        for r in _records("offscope_v3b")
    }
    assert len(key) == 40 and {(k["model"], k["case_id"]) for k in key.values()} == set(raw)
    for k in key.values():
        digest = hashlib.sha256(raw[(k["model"], k["case_id"])].encode("utf-8")).hexdigest()
        assert k["digest"] == digest, k
    out = [json.loads(x) for x in (d / "b8_out3.jsonl").read_text("utf-8").splitlines() if x]
    rows = list(map(json.loads, (M / "b8_review.jsonl").read_text("utf-8").splitlines()))
    review = {
        (r["model"], r["case_id"]): r["gives_advice"] for r in rows if r["run"] == "offscope_v3b"
    }
    got = {(key[o["id"]]["model"], key[o["id"]]["case_id"]): o["gives_advice"] for o in out}
    assert got == review and len(got) == 40
    b5key = json.loads((d / "b5_key_v3b.json").read_text(encoding="utf-8"))
    path = M / "sample_read_main_v3b.jsonl"
    reads = [SampleReadRow.model_validate_json(x) for x in path.read_text("utf-8").splitlines()]
    by = {(r.model, r.case_id): r for r in reads}
    assert len(b5key) == len(reads) == 80
    for k in b5key.values():
        assert by[(k["model"], k["case_id"])].record_sha256 == k["record_sha256"], k
    found = [
        json.loads(x)
        for n in (5, 6)
        for x in (d / f"b5_out{n}.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    assert {
        (b5key[f["key"]]["model"], b5key[f["key"]]["case_id"]): f["finding"] for f in found
    } == {k: r.finding for k, r in by.items()}
