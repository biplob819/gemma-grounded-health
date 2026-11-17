"""RUN-06, D-42: the probe set follows EvalKit's DAT-06 method — EvalKit's sampler at seed 7,
one paraphrase per question, only "same" kept, a reject replaced from the same label."""

import json

import pytest
from evalkit.sampling import stratified_sample
from evalkit.schemas import Case

from ggh.evals import probe as P


def cases(n_yes=552, n_no=338, n_maybe=110):  # PQA-L's label counts
    out = []
    for label, k in (("yes", n_yes), ("no", n_no), ("maybe", n_maybe)):
        for i in range(k):
            out.append(Case(id=f"{label}{i:03d}", question=f"Does {label} {i} work?",
                            context=(f"S1 {label}{i}.", "S2."), reference_label=label,
                            source="pubmedqa/pqa_l"))  # fmt: skip
    return out


def test_verdicts_parse_strictly():
    assert P.parse_verdict("SAME\nsame meaning") == P.Verdict(True, "same meaning")
    assert P.parse_verdict("**Different**.\nscope changed").same is False
    with pytest.raises(ValueError, match="SAME or DIFFERENT"):
        P.parse_verdict("Probably the same")


def test_first_draw_is_evalkits_sampler_and_spares_are_the_rest_by_label():
    cs = cases()
    first, spare = P.queues(cs, 100, 7)
    assert [c.id for c in first] == [c.id for c in stratified_sample(cs, 100, 7).cases]
    assert {k: sum(c.reference_label == k for c in first) for k in ("yes", "no", "maybe")} == {
        "yes": 55, "no": 34, "maybe": 11}  # fmt: skip
    assert all(c.reference_label == k for k, v in spare.items() for c in v)
    assert not {c.id for c in first} & {c.id for v in spare.values() for c in v}
    assert sum(len(v) for v in spare.values()) == len(cs) - 100


def test_rejects_are_replaced_from_the_same_label_until_100_are_kept():
    cs = cases()
    first, spare = P.queues(cs, 100, 7)
    bad = {first[0].id, first[1].id}  # these two get a "different" verdict

    def para(q):
        return "Is it true that " + q[0].lower() + q[1:]

    def check(case, p):
        return P.Verdict(case.id not in bad, "r")

    kept, rejected = P.build(cs, para, check)
    assert len(kept) == 100 and {r["pubid"] for r in rejected} == bad
    mix = {k: sum(r["reference_label"] == k for r in kept) for k in ("yes", "no", "maybe")}
    assert mix == {"yes": 55, "no": 34, "maybe": 11}  # replacements keep the label mix
    for b in bad:
        lab = next(c.reference_label for c in cs if c.id == b)
        assert spare[lab][0].id in {r["pubid"] for r in kept}  # the next of its label
    r = kept[0]
    assert r["paraphrase_model"] == "gpt-6.1-terra" and r["check_model"] == "claude-opus-5-5"
    assert r["paraphrase_prompt_hash"] == P.prompt_hash(P.PARAPHRASE_PROMPT)


def test_an_unchanged_paraphrase_is_rejected_even_if_judged_same():
    # every paraphrase is the question itself: all are rejected until a label runs out
    with pytest.raises(ValueError, match="no replacement"):
        P.build(cases(), lambda q: q, lambda c, p: P.Verdict(True, "r"))


def test_written_tasks_share_abstracts_and_differ_only_in_wording(tmp_path):
    cs = cases()
    kept, rejected = P.build(cs, lambda q: "Rephrased: " + q, lambda c, p: P.Verdict(True, "r"))
    P.write(tmp_path, kept, rejected, cs)
    orig = [json.loads(x) for x in (tmp_path / P.ORIGINAL_DATA).read_text().splitlines()]
    para = [json.loads(x) for x in (tmp_path / P.PARAPHRASE_DATA).read_text().splitlines()]
    assert len(orig) == len(para) == 100
    for a, b in zip(orig, para, strict=True):
        assert a["id"] == b["id"] and a["context"] == b["context"] and a["label"] == b["label"]
        assert b["question"] == "Rephrased: " + a["question"]
    for task in (P.ORIGINAL_TASK, P.PARAPHRASE_TASK):
        text = (tmp_path / task).read_text(encoding="utf-8")
        assert "name: ggh_probe_" in text and "prompt_template" in text
    assert (tmp_path / P.REJECTS).read_text(encoding="utf-8") == ""
