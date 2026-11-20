"""RES-03, OWNER D-24: the 40-claim Gemma-family check on the synthetic main_v2 (base, v1 and a
v2 copied from v1, Sol only; tests/results/fake.py). No network, no API calls."""

import json
import threading
import urllib.request

import pytest
from evalkit.pipeline.calibrate import LabelEntry, LabelFile
from evalkit.pipeline.labeller import save_labels

from ggh.cli import main
from ggh.results import family_check as F
from ggh.results.runs import ResultsError, Run
from ggh.schemas import ResultsBundle
from tests.results import fake

BASE, V1, V2 = "gemma-e2b-base-q4", fake.V1, fake.V2
SOL = "gpt-6.1-sol"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = fake.copy_repo(tmp_path_factory.mktemp("fam"))
    fake.add_v2(root)
    return root


@pytest.fixture
def root(tmp_path, results):
    """An empty checkout that reads main_v2 from `results` (--results-root), as the real one
    reads the uncommitted V2.2 run from the main checkout."""
    (tmp_path / "docs/measurements").mkdir(parents=True)
    return tmp_path


def _sheet(root, results):
    assert main(["calibrate", "sample", "--root", str(root), "--results-root", str(results)]) == 0
    return F.load_sheet(root)


def _label(root, sheet, verdict_of, flip=()):
    labels = {}
    for i in sheet.items:
        v = verdict_of[i.claim_id]
        if i.key in flip:
            v = "unsupported" if v == "supported" else "supported"
        labels[i.key] = LabelEntry(label=v)
    save_labels(root / F.LABELS, LabelFile(sheet_id=sheet.sheet_id, round="first", labels=labels))


def _sol(results):
    run = Run(results, "main_v2")
    return {v["claim_id"]: v["verdict"] for r in run.raw_records for v in r["verdicts"]}


def test_the_draw_is_seeded_20_base_10_v1_10_v2_one_claim_per_question(results):
    run = Run(results, "main_v2")
    a, b = F.draw(run), F.draw(run)
    assert a == b and len(a.items) == 40
    assert dict(a.allocation) == {BASE: 20, V1: 10, V2: 10}
    assert len({i.case_id for i in a.items}) == 40
    assert [i.model for i in a.items] != sorted(i.model for i in a.items)  # interleaved
    other = F.draw(run, seed=25)
    assert {i.claim_id for i in other.items} != {i.claim_id for i in a.items}


def test_the_sheet_holds_no_verdicts_and_the_page_shows_no_model_or_claim_id(results):
    sheet = F.draw(Run(results, "main_v2"))
    text = sheet.model_dump_json()
    assert "verdict" not in text and SOL not in text and "rationale" not in text
    from evalkit.pipeline.labeller import Labeller

    lab = Labeller(sheet.rows(), results / "unused.json", "first", sheet.seed)
    view = json.dumps(lab.view(0))
    assert "gemma" not in view and sheet.items[0].claim_id not in view
    assert lab.view(0)["claim"] == sheet.items[0].claim_text


def test_allocation_rules_and_overrides():
    assert F.allocation(["b", "v1", "v2"]) == {"b": 20, "v1": 10, "v2": 10}
    assert F.allocation(["b", "v1"]) == {"b": 20, "v1": 20}
    assert F.allocation(["b", "v1", "v2"], 41) == {"b": 20, "v1": 11, "v2": 10}
    assert F.parse_allocation("b=14, v1=13,v2=13") == {"b": 14, "v1": 13, "v2": 13}
    with pytest.raises(ResultsError):
        F.parse_allocation("b=x")


def test_a_bad_allocation_is_refused(results):
    run = Run(results, "main_v2")
    with pytest.raises(ResultsError, match="adds up"):
        F.draw(run, alloc={BASE: 10})
    with pytest.raises(ResultsError, match="no model"):
        F.draw(run, alloc={"gpt-6-luna": 40})


def test_a_labelled_sheet_is_never_redrawn(root, results, capsys):
    sheet = _sheet(root, results)
    args = ["calibrate", "sample", "--root", str(root), "--results-root", str(results)]
    assert main(args) == 2 and "--force" in capsys.readouterr().out
    assert main([*args, "--force"]) == 0
    _label(root, sheet, _sol(results))
    assert main([*args, "--force"]) == 2
    assert "holds labels" in capsys.readouterr().out


def test_evalkits_page_serves_the_sheet_on_localhost_and_autosaves(root, results):
    sheet = _sheet(root, results)
    srv = F.serve(root)
    assert srv.server_address[0] == "127.0.0.1"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        item = json.loads(urllib.request.urlopen(base + "/api/item").read())
        assert item["total"] == 40 and item["key"] == sheet.items[0].key
        req = urllib.request.Request(
            base + "/api/label",
            data=json.dumps({"key": item["key"], "label": "supported"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True}
    finally:
        srv.shutdown()
        srv.server_close()
    saved = F.load_labels(root)
    assert saved is not None and saved.sheet_id == sheet.sheet_id
    assert dict(saved.labels) == {item["key"]: LabelEntry(label="supported")}


def test_score_matches_sols_recorded_verdicts_and_feeds_the_bundle_map(root, results, capsys):
    sheet = _sheet(root, results)
    with pytest.raises(ResultsError, match="no first-round labels"):
        F.score(root, results_root=results)
    flip = {i.key for i in sheet.items[:4]}
    _label(root, sheet, _sol(results), flip=flip)
    assert main(["calibrate", "score", "--root", str(root), "--results-root", str(results)]) == 0
    assert "agreement 0.900 on 40 claims" in capsys.readouterr().out
    check = json.loads((root / F.CHECK).read_text(encoding="utf-8"))
    assert check == {SOL: 0.9}
    report = F.FamilyReport.model_validate_json((root / F.REPORT).read_text(encoding="utf-8"))
    (j,) = report.judges
    assert j.n_claims == 40 and j.coverage == 1.0 and sum(map(sum, j.confusion)) == 40
    assert set(j.by_model) == {BASE, V1, V2}
    # the shape ResultsBundle.gemma_family_check takes
    assert ResultsBundle.model_fields["gemma_family_check"].annotation is not None
    from pydantic import TypeAdapter

    TypeAdapter(ResultsBundle.model_fields["gemma_family_check"].annotation).validate_python(check)


def test_score_refuses_partial_labels_and_a_changed_run(root, results, tmp_path_factory):
    sheet = _sheet(root, results)
    sol = _sol(results)
    labels = {i.key: LabelEntry(label=sol[i.claim_id]) for i in sheet.items[:39]}
    save_labels(root / F.LABELS, LabelFile(sheet_id=sheet.sheet_id, round="first", labels=labels))
    with pytest.raises(ResultsError, match="1 of 40 claims are not labelled"):
        F.score(root, results_root=results)
    _label(root, sheet, sol)
    other = fake.copy_repo(tmp_path_factory.mktemp("other"))
    fake.add_v2(other)
    run = Run(other, "main_v2")
    path = run.folder / "run.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m["run_id"] = "r-another"
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ResultsError, match="not r-fake-main_v2"):
        F.score(root, results_root=other)
