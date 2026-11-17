"""AUD-R4 (OWNER D-45, VTG-07): the v2 per-sentence audit. v1's audit (test_audit.py) is
untouched; these tests use the v2 files only."""

import datetime as dt
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ggh import cli
from ggh.data import build as data
from ggh.lineage import StageManifest, sha256_file, stage_key, write_manifest
from ggh.schemas import TrainExample
from ggh.targets import audit, audit_v2, make, rules
from tests.targets.test_targets import FakeScorer, pool

ROOT = Path(__file__).resolve().parents[2]
NOW = dt.datetime(2026, 11, 17, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A tmp repo with v2 targets built from fixture rows, large enough for 5 off-scope."""
    root = tmp_path_factory.mktemp("v2")
    d = root / data.OUT_DIR
    d.mkdir(parents=True)
    outputs = {}
    for name, rows in {"val.jsonl": pool(60, 10), "train_pool.jsonl": pool(1100, 400, 100)}.items():
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        outputs[(data.OUT_DIR / name).as_posix()] = sha256_file(d / name)
    write_manifest(
        d,
        StageManifest(
            stage="data", code_version="1", key=stage_key("data", "1", {}, {}, 7), inputs={},
            outputs=outputs, params={}, seed=7, parents=(), started_at=NOW, finished_at=NOW,
            host="test",
        ),
    )  # fmt: skip
    make.build(root, FakeScorer, seed=7)
    return root


@pytest.fixture
def repo(built, tmp_path):
    for rel in (make.OUT_DIR / "train.jsonl", make.REPORT_COPY):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((built / rel).read_bytes())
    return tmp_path, audit_v2.draw(tmp_path)


def label(e, sentences=None, overall="ok", **kw):
    if sentences is None:
        sentences = "supports"
    if isinstance(sentences, str):
        sentences = {str(c): sentences for c in e.target.citations}
    if not audit_v2.needs_sentences(e):
        sentences = {}
    return audit_v2.SentenceLabel(
        example_id=e.id, content_digest=audit.digest(e), sentences=sentences, overall=overall, **kw
    )


def label_all(root, sample, fn=lambda e: label(e)):
    for e in sample:
        audit_v2.save_label(root, fn(e))


def answers(sample):
    return [e for e in sample if e.kind == "answer"]


# ---- the draw ----


def test_sample_is_aud_r1s_draw_on_the_v2_train_split(repo):
    root, sample = repo
    kinds = [(e.kind, e.source_label if e.kind == "answer" else None) for e in sample]
    assert len(sample) == 50 and len({e.id for e in sample}) == 50
    assert kinds.count(("answer", "yes")) == 18 and kinds.count(("answer", "no")) == 12
    assert kinds.count(("insufficient", None)) == 15 and kinds.count(("offscope", None)) == 5
    train = [TrainExample.model_validate(r) for r in data.read_jsonl(root / audit_v2.TRAIN)]
    assert [e.id for e in sample] == [e.id for e in make.audit_sample(train, 7)]
    assert all(
        e.split == "train" and "TGT-R10" in e.rule_ids for e in sample if e.kind != "offscope"
    )
    assert audit_v2.load_sample(root) == sample
    assert audit_v2.SAMPLE != audit.SAMPLE and audit_v2.LABELS != audit.LABELS  # v1 untouched


def test_draw_refuses_a_train_split_the_report_does_not_record(repo):
    root, _ = repo
    p = root / audit_v2.TRAIN
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rerun ggh targets"):
        audit_v2.draw(root)


def test_draw_refuses_v1_targets(repo):
    root, _ = repo
    p = root / audit_v2.TRAIN
    lines = [
        json.dumps({**r, "rule_ids": [x for x in r["rule_ids"] if x != "TGT-R10"]})
        for r in data.read_jsonl(p)
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = json.loads((root / make.REPORT_COPY).read_text(encoding="utf-8"))
    report["files"][audit_v2.TRAIN.as_posix()] = sha256_file(p)
    (root / make.REPORT_COPY).write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="not v2"):
        audit_v2.draw(root)


# ---- labels: bound to the text shown, one verdict per cited sentence ----


def test_labels_are_bound_to_the_text_shown(repo):
    root, sample = repo
    label_all(root, sample)
    e = sample[0]
    bad = label(e).model_copy(update={"content_digest": "0" * 64})
    with pytest.raises(ValueError, match="changed since it was shown"):
        audit_v2.save_label(root, bad)
    edited = TrainExample.model_validate(
        {**json.loads(e.model_dump_json()), "question": e.question + " (edited)"}
    )
    lines = [edited.model_dump_json() if x.id == e.id else x.model_dump_json() for x in sample]
    (root / audit_v2.SAMPLE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = audit_v2.score(root)
    assert r["stale"] == [e.id] and not r["passed"]
    with pytest.raises(ValueError, match="bound to changed text"):
        audit_v2.write_result(root)
    assert not (root / audit_v2.RESULT).exists()


def test_a_label_judges_only_the_cited_sentences_and_findings_only_for_insufficient(repo):
    root, sample = repo
    a = answers(sample)[0]
    off = next(e for e in sample if e.kind == "offscope")
    uncited = next(i for i in range(1, len(a.context) + 1) if i not in a.target.citations)
    with pytest.raises(ValueError, match="not judged"):
        audit_v2.save_label(root, label(a, {str(uncited): "supports"}))
    with pytest.raises(ValueError, match="not judged"):
        audit_v2.save_label(root, label(off).model_copy(update={"sentences": {"1": "aim"}}))
    with pytest.raises(ValueError, match="insufficient examples only"):
        audit_v2.save_label(root, label(a, overall="claims_finding"))
    with pytest.raises(ValueError):
        audit_v2.SentenceLabel.model_validate(
            {"example_id": a.id, "content_digest": audit.digest(a), "sentences": {"1": "great"}}
        )
    with pytest.raises(ValueError, match="not in the v2 audit sample"):
        audit_v2.save_label(root, label(a).model_copy(update={"example_id": "pqa-a-1"}))


def test_partial_labels_save_but_count_as_missing(repo):
    root, sample = repo
    label_all(root, sample)
    a = answers(sample)[0]
    first = str(a.target.citations[0])
    audit_v2.save_label(root, label(a, {first: "supports"}, overall=None))
    assert audit_v2.load_labels(root)[a.id].sentences == {first: "supports"}
    r = audit_v2.score(root)  # no overall verdict yet, so incomplete
    assert r["missing"] == [a.id] and not r["passed"]
    with pytest.raises(ValueError, match="missing or incomplete"):
        audit_v2.write_result(root)


# ---- the bar ----


def test_all_supporting_passes_and_writes_the_result(repo):
    root, sample = repo
    label_all(root, sample)
    r = audit_v2.write_result(root)
    assert r["passed"] and r["overall"] == {"ok": 50}
    saved = json.loads((root / audit_v2.RESULT).read_text(encoding="utf-8"))
    assert saved == r and saved["sample_sha256"] == sha256_file(root / audit_v2.SAMPLE)


def test_one_aim_or_method_sentence_in_an_answer_fails(repo):
    root, sample = repo
    for bad in ("aim", "method"):
        target = answers(sample)[3]
        first = str(target.target.citations[0])

        def fn(e, bad=bad, target=target, first=first):
            if e.id != target.id:
                return label(e)
            return label(e, {**{str(c): "supports" for c in e.target.citations}, first: bad})

        label_all(root, sample, fn)
        r = audit_v2.score(root)
        assert not r["passed"] and r["answers_failing"] == [target.id], bad
        assert not r["bars"]["no_answer_cites_aim_or_method_or_two_weak"]


def one_weak(e):
    first = str(e.target.citations[0])
    return label(e, {**{str(c): "supports" for c in e.target.citations}, first: "weak"})


def test_at_most_three_answers_may_have_one_weak_sentence(repo):
    root, sample = repo
    weak3 = {e.id for e in answers(sample)[:3]}
    label_all(root, sample, lambda e: one_weak(e) if e.id in weak3 else label(e))
    r = audit_v2.score(root)
    assert r["passed"] and sorted(r["answers_with_one_weak"]) == sorted(weak3)
    weak4 = {e.id for e in answers(sample)[:4]}
    label_all(root, sample, lambda e: one_weak(e) if e.id in weak4 else label(e))
    assert not audit_v2.score(root)["passed"]


def test_two_weak_sentences_in_one_answer_fail(repo):
    root, sample = repo
    a = answers(sample)[0]  # the fixture rows cite one sentence: give this one two
    d = json.loads(a.model_dump_json())
    d["target"]["citations"] = [2, 3]
    two = TrainExample.model_validate(d)
    sample = [two if e.id == a.id else e for e in sample]
    (root / audit_v2.SAMPLE).write_text(
        "".join(e.model_dump_json() + "\n" for e in sample), encoding="utf-8"
    )
    label_all(root, sample, lambda e: label(e, "weak") if e.id == two.id else label(e))
    r = audit_v2.score(root)
    assert not r["passed"] and r["answers_failing"] == [two.id]


def test_insufficient_examples_may_cite_aims_but_must_not_claim_a_finding(repo):
    root, sample = repo
    ins = [e for e in sample if e.kind == "insufficient"]
    label_all(root, sample, lambda e: label(e, "aim") if e.kind == "insufficient" else label(e))
    r = audit_v2.score(root)
    assert r["passed"] and r["sentences"]["insufficient"] == {
        "aim": sum(len(e.target.citations) for e in ins)
    }  # OWNER D-45: the aim is the right citation for "no finding reported"
    label_all(root, sample, lambda e: label(e, overall="claims_finding") if e.id == ins[0].id
              else label(e))  # fmt: skip
    r = audit_v2.score(root)
    assert not r["passed"] and r["insufficient_claiming_a_finding"] == [ins[0].id]


def test_at_least_45_overall_ok(repo):
    root, sample = repo
    wrong5 = {e.id for e in sample[:5]}
    label_all(
        root, sample, lambda e: label(e, overall="wrong_label") if e.id in wrong5 else label(e)
    )
    assert audit_v2.score(root)["passed"]
    wrong6 = {e.id for e in sample[:6]}
    label_all(
        root, sample, lambda e: label(e, overall="wrong_label") if e.id in wrong6 else label(e)
    )
    r = audit_v2.score(root)
    assert not r["passed"] and r["overall"] == {"ok": 44, "wrong_label": 6}


def test_the_bar_is_pinned():
    assert (audit_v2.PASS_OK, audit_v2.MAX_WEAK_ANSWERS, audit_v2.DEFAULT_PORT) == (45, 3, 8766)
    assert audit_v2.JUDGED_KINDS == ("answer", "insufficient")


# ---- the page and the CLI ----


def test_page_serves_on_localhost_autosaves_and_refuses_a_busy_port(repo):
    root, sample = repo
    srv = audit_v2.server(root, port=0)
    assert srv.server_address[0] == "127.0.0.1"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "S supports" in page and "claims a finding" in page
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        a = next(x for x in items if x["kind"] == "answer")
        assert a["judge_sentences"] and a["digest"] == audit.digest(
            next(e for e in sample if e.id == a["id"])
        )
        assert not next(x for x in items if x["kind"] == "offscope")["judge_sentences"]
        c = str(a["citations"][0])
        body = {"example_id": a["id"], "content_digest": a["digest"], "sentences": {c: "aim"}}
        req = urllib.request.Request(
            base + "/api/label", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True}
        assert audit_v2.load_labels(root)[a["id"]].sentences == {c: "aim"}  # saved at once
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert next(x for x in items if x["id"] == a["id"])["sentences"] == {c: "aim"}
        body["sentences"] = {c: "great"}
        bad = urllib.request.Request(base + "/api/label", json.dumps(body).encode(), {})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
        with pytest.raises(OSError):
            audit_v2.server(root, port=port)  # never takes over a listening port
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_sample_and_score(repo, capsys):
    root, sample = repo
    assert cli.main(["audit", "sample", "--v2", "--root", str(root)]) == 0
    assert cli.main(["audit", "score", "--v2", "--root", str(root)]) == cli.EXIT_ERROR
    assert not (root / audit_v2.RESULT).exists()
    label_all(root, sample)
    assert cli.main(["audit", "score", "--v2", "--root", str(root)]) == 0
    assert "audit: PASS" in capsys.readouterr().out and (root / audit_v2.RESULT).exists()
    p = cli.build_parser().parse_args(["audit", "serve", "--v2"])
    assert p.v2 and p.port is None


def test_serve_with_a_busy_port_fails_cleanly(repo, capsys):
    root, _ = repo
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        rc = cli.main(["audit", "serve", "--v2", "--root", str(root), "--port", str(port)])
    assert rc == cli.EXIT_ERROR and "audit: FAIL" in capsys.readouterr().out


# ---- the committed v2 sample ----


@pytest.mark.skipif(not (ROOT / audit_v2.SAMPLE).exists(), reason="drawn once v2 targets exist")
def test_committed_v2_sample_is_stratified_and_its_answers_cite_no_aim():
    sample = audit_v2.load_sample(ROOT)
    kinds = [(e.kind, e.source_label if e.kind == "answer" else None) for e in sample]
    assert len(sample) == 50 and kinds.count(("answer", "yes")) == 18
    assert kinds.count(("answer", "no")) == 12 and kinds.count(("insufficient", None)) == 15
    for e in answers(sample):
        assert "TGT-R10" in e.rule_ids
        assert not any(rules.is_aim_sentence(e.context[i - 1]) for i in e.target.citations)
    train = ROOT / audit_v2.TRAIN
    report = json.loads((ROOT / make.REPORT_COPY).read_text(encoding="utf-8"))
    if train.exists() and sha256_file(train) == report["files"][audit_v2.TRAIN.as_posix()]:
        rows = [TrainExample.model_validate(r) for r in data.read_jsonl(train)]
        assert [e.id for e in make.audit_sample(rows, 7)] == [e.id for e in sample]
