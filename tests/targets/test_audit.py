import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ggh import cli
from ggh.schemas import AuditLabel, TrainExample
from ggh.targets import audit, make
from tests.targets.test_targets import FakeScorer, pool


@pytest.fixture
def repo(tmp_path):
    sel = make.select(pool(60, 10), pool(1100, 400, start=100), FakeScorer(), seed=7)
    sample = make.audit_sample(sel.train, 7)
    path = tmp_path / audit.SAMPLE
    path.parent.mkdir(parents=True)
    path.write_text("".join(e.model_dump_json() + "\n" for e in sample), encoding="utf-8")
    return tmp_path, sample


def label_all(root, sample, verdict_for):
    for e in sample:
        audit.save_label(
            root,
            AuditLabel(example_id=e.id, content_digest=audit.digest(e), verdict=verdict_for(e)),
        )


def test_sample_is_fifty_seeded_and_stratified(repo):
    _, sample = repo
    kinds = [(e.kind, e.source_label if e.kind == "answer" else None) for e in sample]
    assert len(sample) == 50 and len({e.id for e in sample}) == 50
    assert kinds.count(("answer", "yes")) == 18 and kinds.count(("answer", "no")) == 12
    assert kinds.count(("insufficient", None)) == 15 and kinds.count(("offscope", None)) == 5
    sel = make.select(pool(60, 10), pool(1100, 400, start=100), FakeScorer(), seed=7)
    assert [e.id for e in make.audit_sample(sel.train, 7)] == [e.id for e in sample]


def test_pass_bar(repo):
    root, sample = repo
    label_all(root, sample, lambda e: "ok")
    assert audit.score(root)["passed"]
    weak = {e.id for e in sample[:5]}
    label_all(root, sample, lambda e: "weak_citation" if e.id in weak else "ok")
    assert audit.score(root)["passed"]  # 45 ok
    weak = {e.id for e in sample[:6]}
    label_all(root, sample, lambda e: "weak_citation" if e.id in weak else "ok")
    assert not audit.score(root)["passed"]  # 44 ok
    first_answer = next(e.id for e in sample if e.kind == "answer")
    label_all(root, sample, lambda e: "unsupported_answer" if e.id == first_answer else "ok")
    r = audit.score(root)
    assert not r["passed"] and r["unsupported_answers"] == [first_answer]  # 49 ok, still fails
    first_ins = next(e.id for e in sample if e.kind == "insufficient")
    label_all(root, sample, lambda e: "unsupported_answer" if e.id == first_ins else "ok")
    assert audit.score(root)["passed"]  # the rule is about answer examples only


def test_labels_are_bound_to_the_text_shown(repo):
    root, sample = repo
    label_all(root, sample, lambda e: "ok")
    e = sample[0]
    with pytest.raises(ValueError, match="changed since it was shown"):
        audit.save_label(root, AuditLabel(example_id=e.id, content_digest="0" * 64, verdict="ok"))
    edited = TrainExample.model_validate(
        {**json.loads(e.model_dump_json()), "question": e.question + " (edited)"}
    )
    lines = [edited.model_dump_json() if x.id == e.id else x.model_dump_json() for x in sample]
    (root / audit.SAMPLE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = audit.score(root)
    assert r["stale"] == [e.id] and not r["passed"]  # a changed example voids its label
    with pytest.raises(ValueError, match="not in the audit sample"):
        audit.save_label(
            root, AuditLabel(example_id="pqa-a-1", content_digest=audit.digest(e), verdict="ok")
        )


def test_page_serves_on_localhost_and_autosaves(repo):
    root, sample = repo
    srv = audit.server(root, port=0)
    assert srv.server_address[0] == "127.0.0.1"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "O &middot; ok" in page and "unsupported answer" in page
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert len(items) == 50 and items[0]["digest"] == audit.digest(sample[0])
        body = json.dumps(
            {"example_id": items[0]["id"], "content_digest": items[0]["digest"], "verdict": "ok"}
        ).encode()
        req = urllib.request.Request(
            base + "/api/label", body, {"Content-Type": "application/json"}
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True}
        assert audit.load_labels(root)[items[0]["id"]].verdict == "ok"  # saved immediately
        bad = urllib.request.Request(base + "/api/label", body.replace(b'"ok"}', b'"great"}'), {})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_score_exit_codes(repo, capsys):
    root, sample = repo
    assert cli.main(["audit", "score", "--root", str(root)]) == cli.EXIT_ERROR
    label_all(root, sample, lambda e: "ok")
    assert cli.main(["audit", "score", "--root", str(root)]) == 0
    assert "audit: PASS" in capsys.readouterr().out


def test_data_card_reports_the_committed_audit_result():
    root = Path(__file__).resolve().parents[2]
    r = audit.score(root)
    card = (root / "DATA_CARD.md").read_text(encoding="utf-8")
    ok = r["verdicts"].get("ok", 0)
    assert r["passed"] and f"Result: {ok} of {r['sample']} `ok`" in card
