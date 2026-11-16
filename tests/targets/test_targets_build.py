"""CI coverage for `build()`, the audit sample, `score()`, the audit server's edges and the
NLI wrappers (N1.3 verifier round 2: each test kills a mutation that survived)."""

import datetime as dt
import json
import random
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Literal

import pytest

from ggh.data import build as data
from ggh.lineage import StageManifest, read_manifest, sha256_file, stage_key, verify, write_manifest
from ggh.schemas import AuditLabel
from ggh.targets import audit, make, nli, rules
from tests.targets.test_targets import FakeScorer, pool, row

NOW = dt.datetime(2026, 11, 16, tzinfo=dt.UTC)


@pytest.fixture
def repo(tmp_path):
    """A tmp repo with N1.2-shaped outputs and their data-stage manifest."""
    d = tmp_path / data.OUT_DIR
    d.mkdir(parents=True)
    files = {"val.jsonl": pool(40, 6), "train_pool.jsonl": pool(500, 120, start=100)}
    outputs = {}
    for name, rows in files.items():
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
    return tmp_path


def test_build_writes_outputs_copies_and_a_manifest_that_replays(repo):
    report = make.build(repo, FakeScorer, seed=7, cap=200, chunk=100)
    out = repo / make.OUT_DIR
    assert (repo / make.REPORT_COPY).read_bytes() == (out / "drop_report.json").read_bytes()
    assert (repo / make.AUDIT_COPY).read_bytes() == (out / "audit_sample.jsonl").read_bytes()
    man = read_manifest(out / "manifest.json")
    assert man.params["nli_model"] == f"{nli.MODEL_ID}@{nli.MODEL_REVISION}"
    assert man.params["rules_sha256"] == make.rules_hash() == report["rules_sha256"]
    assert verify(out / "manifest.json", repo) == []
    table = {r["key"]: r["entail"] for r in data.read_jsonl(out / "nli_scores.jsonl")}
    assert table  # every score the build used is recorded
    rows = {r["pmid"]: r for f in ("val.jsonl", "train_pool.jsonl")
            for r in data.read_jsonl(repo / data.OUT_DIR / f)}  # fmt: skip
    replay = nli.ReplayScorer(table)
    for split in ("train", "val"):
        for r in data.read_jsonl(out / f"{split}.jsonl"):
            again = make.reconstruct(
                rows[r["pmid"]], r["rule_ids"], split=split, seed=7, scorer=replay
            )
            assert json.loads(again.model_dump_json()) == r


def test_build_refuses_inputs_that_differ_from_the_data_stage(repo):
    p = repo / data.OUT_DIR / "val.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rerun ggh data"):
        make.build(repo, FakeScorer, cap=200)
    assert not (repo / make.OUT_DIR).exists()


# ---- TGT-07: every results word removes its section ----


@pytest.mark.parametrize("label", ["CONCLUSIONS", "FINDINGS", "OUTCOMES", "RESULTS"])
def test_each_results_word_marks_a_results_section(label):
    assert rules.is_results_section(label)
    r = row(20, sections=["BACKGROUND", "METHODS", label, label])
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None and ex.context == tuple(r["sentences"][:2])
    assert not rules.is_results_section("METHODS")


# ---- TGT-07 / TGT-08: insufficient citations and off-scope picks ----


def test_insufficient_citations_are_ascending_when_the_better_match_comes_later():
    sents = [
        "Background on zeta in general.",  # shares "zeta"
        "Methods for older adults with zeta use.",  # shares "zeta older adults": the best
        "Results showed outcome omega improved.",
    ]
    r = row(
        31, long_answer="Outcome omega improved.", sentences=sents,
        sections=["BACKGROUND", "METHODS", "RESULTS"], question="Does zeta help older adults?",
    )  # fmt: skip
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None and ex.target.citations == (1, 2)


def test_offscope_question_and_answer_are_not_locked_together():
    pairs = set()
    for i in range(80):
        ex = make.offscope_example(row(i), "train", 7)
        pairs.add((rules.OFFSCOPE_QUESTIONS.index(ex.question),
                   rules.OFFSCOPE_ANSWERS.index(ex.target.answer)))  # fmt: skip
    assert any(a != q % 6 for q, a in pairs)
    assert len({a for _, a in pairs}) == 6 and len({q for q, _ in pairs}) == 12


# ---- TGT-10: the audit sample is a seeded random draw ----


def test_audit_sample_is_random_within_each_stratum_and_depends_on_the_seed():
    sel = make.select(pool(60, 10), pool(1100, 400, start=100), FakeScorer(), seed=7)
    a = make.audit_sample(sel.train, 7)
    b = make.audit_sample(sel.train, 8)
    assert [e.id for e in a] != [e.id for e in b]
    yes = [e for e in sel.train if e.kind == "answer" and e.source_label == "yes"]
    assert [e.id for e in a[:18]] != [e.id for e in yes[:18]]  # not the first n
    rng = random.Random(7)  # the exact draw, stratum by stratum
    expect = []
    for kind, label, n in make.AUDIT_STRATA:
        stratum = [
            e for e in sel.train if e.kind == kind and (label is None or e.source_label == label)
        ]
        expect += rng.sample(stratum, n)
    off = [e for e in sel.train if e.kind == "offscope"]
    expect += rng.sample(off, make.AUDIT_OFFSCOPE)
    assert [e.id for e in a] == [e.id for e in expect]
    assert [e.id for e in a[-5:]] != [e.id for e in off[:5]]


@pytest.fixture
def audit_repo(tmp_path):
    sel = make.select(pool(60, 10), pool(1100, 400, start=100), FakeScorer(), seed=7)
    sample = make.audit_sample(sel.train, 7)
    p = tmp_path / audit.SAMPLE
    p.parent.mkdir(parents=True)
    p.write_text("".join(e.model_dump_json() + "\n" for e in sample), encoding="utf-8")
    return tmp_path, sample


def _label(root, e, verdict: Literal["ok", "weak_citation"] = "ok"):
    audit.save_label(
        root, AuditLabel(example_id=e.id, content_digest=audit.digest(e), verdict=verdict)
    )


def test_score_needs_every_label_and_a_full_sample(audit_repo):
    root, sample = audit_repo
    for e in sample[:-1]:
        _label(root, e)
    r = audit.score(root)
    assert not r["passed"] and r["missing"] == [sample[-1].id] and r["labelled"] == 49
    short = sample[:45]
    (root / audit.SAMPLE).write_text(
        "".join(e.model_dump_json() + "\n" for e in short), encoding="utf-8"
    )
    r = audit.score(root)
    assert r["sample"] == 45 and r["missing"] == [] and not r["passed"]


def test_a_stale_label_is_not_counted_as_a_verdict(audit_repo):
    root, sample = audit_repo
    for e in sample:
        _label(root, e)
    lines = (root / audit.SAMPLE).read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["question"] += " (edited)"
    lines[0] = json.dumps(first)
    (root / audit.SAMPLE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = audit.score(root)
    assert r["stale"] == [sample[0].id] and r["labelled"] == 49 and r["verdicts"] == {"ok": 49}


def test_labels_are_saved_in_sample_order_and_atomically(audit_repo, monkeypatch):
    root, sample = audit_repo
    for e in reversed(sample[:3]):
        _label(root, e)
    ids = [
        json.loads(x)["example_id"]
        for x in (root / audit.LABELS).read_text(encoding="utf-8").splitlines()
    ]
    assert ids == [e.id for e in sample[:3]]
    before = (root / audit.LABELS).read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audit.os, "replace", boom)
    with pytest.raises(OSError):
        _label(root, sample[3])
    assert (root / audit.LABELS).read_bytes() == before  # the old file is intact


def test_server_404s_unknown_paths_and_hides_stale_verdicts(audit_repo):
    root, sample = audit_repo
    _label(root, sample[0])
    lines = (root / audit.SAMPLE).read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["question"] += " (edited)"
    lines[0] = json.dumps(first)
    (root / audit.SAMPLE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    srv = audit.server(root, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert items[0]["verdict"] is None  # its label is stale, so it is not shown
        for req in (
            urllib.request.Request(base + "/other"),
            urllib.request.Request(
                base + "/api/other", b"{}", {"Content-Type": "application/json"}
            ),
        ):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req)
            assert err.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ---- select and reconstruct edges ----


def test_selection_does_not_depend_on_input_order():
    v, p = pool(30, 4), pool(300, 40, start=100)
    a = make.select(v, p, FakeScorer(), seed=7)
    b = make.select(list(reversed(v)), random.Random(1).sample(p, len(p)), FakeScorer(), seed=7)
    assert [e.id for e in a.train] == [e.id for e in b.train]
    assert [e.id for e in a.val] == [e.id for e in b.val]


def test_unfillable_quotas_raise():
    val_rows = pool(5, 1)
    no_yes_pool = [row(200 + i, "no") for i in range(40)]  # nothing can fill the yes quota
    with pytest.raises(ValueError, match="could not be filled"):
        make.select(val_rows, no_yes_pool, FakeScorer(), seed=7)


def test_reconstruct_rejects_unknown_rule_ids_and_rows_without_an_insufficient_example():
    with pytest.raises(ValueError, match="unknown rule ids"):
        make.reconstruct(row(1), ("TGT-R1",), split="train", seed=7, scorer=FakeScorer())
    no_results = row(2, sections=["BACKGROUND", "METHODS", "METHODS", "METHODS"])
    with pytest.raises(ValueError, match="cannot make an insufficient"):
        make.reconstruct(
            no_results, rules.INSUFFICIENT_RULES, split="train", seed=7, scorer=FakeScorer()
        )


# ---- NLI wrappers ----


def test_a_multi_sentence_premise_joins_the_cited_sentences_with_a_space():
    sents = ["Alpha bravo.", "Charlie delta.", "Echo foxtrot.", "Golf hotel."]
    r = row(30, long_answer="Alpha bravo charlie delta.", sentences=sents, sections=["RESULTS"] * 4)

    class Spy:
        def __init__(self):
            self.premises = []

        def entail(self, pairs):
            self.premises += [p for p, _ in pairs]
            return [0.9 if "Charlie" in p else 0.1 for p, _ in pairs]

    spy = Spy()
    (t,) = make.targets_for([r], spy)
    assert isinstance(t, make.AnswerTarget) and t.citations == (1, 2)
    assert "Alpha bravo. Charlie delta." in spy.premises


def test_recording_scorer_asks_the_model_once_per_distinct_pair():
    class Count:
        n = 0

        def entail(self, pairs):
            self.n += len(pairs)
            return [0.5] * len(pairs)

    inner = Count()
    rec = nli.RecordingScorer(inner, 6)
    assert rec.entail([("p", "h"), ("p", "h"), ("q", "h")]) == [0.5, 0.5, 0.5]
    rec.entail([("p", "h")])
    assert inner.n == 2


def test_hf_scorer_returns_scores_in_the_callers_order():
    """HFScorer with a fake tokenizer and model: batches are length-sorted, but each score
    must come back at its pair's position (the score encodes the pair's length)."""

    class T:
        def __init__(self, rows):
            self.rows = rows

        def to(self, device):
            return self

        def softmax(self, dim):
            return self

        def __getitem__(self, key):
            return T([r[key[1]] for r in self.rows])

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class Enc(dict):
        def to(self, device):
            return self

    def tok(premises, hyps, **kw):
        return Enc(lens=[len(p) + len(h) for p, h in zip(premises, hyps, strict=True)])

    def model(lens):
        return SimpleNamespace(logits=T([[0.0, n / 1000, 0.0] for n in lens]))

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    s = object.__new__(nli.HFScorer)
    s.__dict__["_torch"] = SimpleNamespace(no_grad=NoGrad)
    s.tok, s.model, s.device, s.ent = tok, model, "cpu", 1
    pairs = [("x" * n, "y") for n in (70, 3, 50, 1, 90, 20) * 7]
    got = s.entail(pairs)
    assert got == [(len(p) + len(h)) / 1000 for p, h in pairs]
