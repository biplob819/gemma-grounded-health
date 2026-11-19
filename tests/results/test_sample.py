"""RES-05: the sample-read sheet (Tech Spec §7.6, D-18) and how the bundle reads it back."""

import json

import pytest
from pydantic import ValidationError

from ggh.cli import main
from ggh.results import bundle as B
from ggh.results import sample as S
from ggh.results.runs import Run
from ggh.schemas import SampleReadRow
from tests.results import fake


@pytest.fixture
def root(tmp_path):
    return fake.copy_repo(tmp_path)


def test_draw_is_seeded_20_per_model_with_cited_sentences_and_verdicts(root):
    run = Run(root, "main_citation")
    rows = S.draw(run)
    assert len(rows) == 80 and rows == S.draw(run)
    assert {r.model for r in rows} == set(run.models())
    r = next(x for x in rows if x.model == fake.V1 and x.claims)
    assert set(r.cited_sentences) == {str(n) for n in r.citations}
    assert set(r.claims[0].verdicts) == {"claude-opus-5-5", "gpt-6.1-sol"}
    assert r.claims[0].verdicts["claude-opus-5-5"].citation_support  # OWNER D-44 roles shown
    assert all(x.finding is None for x in rows)


def _fill(path, finding="ok", note=""):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.update(finding=finding, note=note, read_by="builder")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_a_filled_sheet_completes_b5_and_is_not_redrawn(root, capsys):
    assert main(["results", "sample", "--run", "main", "--root", str(root)]) == 0
    path = S.sheet_path(root, "main")
    _fill(path)
    assert main(["results", "sample", "--run", "main", "--root", str(root)]) == 2
    assert "already holds findings" in capsys.readouterr().out
    b = B.build(root).bundle
    sums = {s.model: s for s in b.sample_read}
    assert all(s.complete and s.read == 20 and s.findings == {"ok": 20} for s in sums.values())
    b5 = [r for r in b.behaviour if r.rule == "B5" and r.run == "main"]
    assert [r.value for r in b5] == [1.0] * len(b5)
    assert not any("B5" in x for x in b.publish_blockers)


def test_a_pipeline_fault_or_an_edited_record_blocks_publication(root):
    S.write_sheet(root, Run(root, "main"))
    path = S.sheet_path(root, "main")
    _fill(path, finding="pipeline_fault", note="claims split mid-sentence")
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["record_sha256"] = "0" * 64
    path.write_text("\n".join([json.dumps(first), *rows[1:]]) + "\n", encoding="utf-8")
    blockers = "\n".join(B.build(root).bundle.publish_blockers)
    assert "pipeline fault" in blockers and "changed since it was drawn" in blockers


def test_a_fault_needs_a_note_and_a_reader():
    d = json.loads((fake.ROOT / "tests/fixtures/schemas/SampleReadRow.json").read_text("utf-8"))
    SampleReadRow.model_validate(d)
    with pytest.raises(ValidationError, match="needs a note"):
        SampleReadRow.model_validate({**d, "note": " "})
    with pytest.raises(ValidationError, match="needs read_by"):
        SampleReadRow.model_validate({**d, "finding": "ok", "read_by": None})
