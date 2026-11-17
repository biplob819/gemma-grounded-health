"""EXP-04, EXP-06: the parity report and its bar, from fixed texts; the request the Q4_K_M side
sends is EvalKit's constrained request; the run writes the report, all replies and a sample."""

import gzip
import json

import pytest

from ggh.export import parity as P
from ggh.schemas import ParityReport


def ans(label, cites=(1,), text="Short answer."):
    return json.dumps({"label": label, "answer": text, "citations": list(cites)})


def test_agreement_counts_only_pairs_that_both_parse_and_match():
    bf16 = [ans("yes"), ans("no"), ans("yes", (2,)), ans("maybe"), "not json"]
    q4 = [ans("yes"), ans("yes"), ans("yes", (3,)), "{bad", ans("no")]
    r = P.build_report(bf16, q4)
    assert r.n == 5
    assert r.label_agreement == 2 / 5  # rows 0 and 2; a parse failure on either side counts against
    assert r.citation_exact_match == 2 / 5  # rows 0 and 1
    assert dict(r.schema_valid) == {"bf16": 4 / 5, "q4_k_m": 4 / 5}


def test_parsing_is_evalkits_strict_answer_schema():
    assert P.parsed(ans("yes")) is not None
    assert P.parsed(json.dumps({"label": "yes", "answer": "x", "citations": ["1"]})) is None
    assert P.parsed(json.dumps({"label": "probably", "answer": "x", "citations": [1]})) is None


def _rep(agree, bf16=1.0, q4=1.0):
    return ParityReport(n=500, label_agreement=agree, citation_exact_match=0.5,
                        schema_valid={"bf16": bf16, "q4_k_m": q4})  # fmt: skip


def test_the_bar_is_inclusive_at_095_and_one_point():
    assert P.passes(_rep(0.95)) and P.failures(_rep(0.95)) == []
    assert not P.passes(_rep(0.948)) and "label agreement" in P.failures(_rep(0.948))[0]
    assert P.passes(_rep(0.99, 1.0, 0.99))  # exactly one point apart
    assert not P.passes(_rep(0.99, 1.0, 0.988))
    assert "schema validity" in P.failures(_rep(0.99, 0.97, 0.99))[0]  # either direction


def test_misaligned_or_empty_replies_are_errors():
    with pytest.raises(ValueError):
        P.build_report([ans("yes")], [])
    with pytest.raises(ValueError):
        P.build_report([], [])


def test_the_q4_side_sends_evalkits_constrained_request():
    seen = []

    class Done:
        raw_text = ans("no")

    def complete(req):
        seen.append(req)
        return Done()

    assert P.server_replies(["p1", "p2"], complete) == [ans("no")] * 2  # pyright: ignore[reportArgumentType]
    assert [r.prompt for r in seen] == ["p1", "p2"]
    assert all(r.json_schema is not None and r.model_id == P.Q4_NAME for r in seen)


def test_run_writes_the_report_every_reply_and_every_25th_as_the_sample(tmp_path):
    rows = [{"pmid": i, "label": "yes", "prompt": f"p{i}"} for i in range(60)]
    order = []

    def bf16_gen(prompts):
        order.append("bf16")
        return [ans("yes") for _ in prompts]

    def q4_gen(prompts):
        order.append("q4")
        return [ans("yes") if i % 10 else ans("no") for i, _ in enumerate(prompts)]

    rep = P.run(tmp_path, rows, bf16_gen=bf16_gen, q4_gen=q4_gen)
    assert order == ["bf16", "q4"]  # bf16 first: its memory is freed before the server starts
    assert rep.label_agreement == 54 / 60 and not P.passes(rep)
    saved = ParityReport.model_validate_json((tmp_path / P.REPORT).read_text(encoding="utf-8"))
    assert saved == rep
    raw = (tmp_path / P.GENERATIONS).read_bytes()
    assert raw[4:8] == b"\0\0\0\0"  # gzip mtime 0: reproducible bytes
    lines = gzip.decompress(raw).decode("utf-8").splitlines()
    assert len(lines) == 60 and json.loads(lines[10])["q4_k_m"] == ans("no")
    sample = (tmp_path / P.SAMPLE).read_text(encoding="utf-8").splitlines()
    assert sample == lines[::25] and len(sample) == 3
