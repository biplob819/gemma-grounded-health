"""Boundary and wiring tests that run in CI without the real artifacts (N1.3 verifier round 1:
each test here kills a mutation that only the local real-data test, or nothing, caught)."""

import json
from collections import Counter

import pytest

from ggh.schemas import TrainExample
from ggh.targets import audit, make, rules
from tests.targets.test_targets import FakeScorer, pool, row


class AllWords:
    """Entails only when every content word of the hypothesis is in the premise."""

    def entail(self, pairs):
        return [0.9 if rules.content_words(h) <= rules.content_words(p) else 0.1 for p, h in pairs]


class Fixed:
    def __init__(self, score):
        self.score = score

    def entail(self, pairs):
        return [self.score] * len(pairs)


# ---- TGT-10: the digest covers everything the page shows ----


@pytest.mark.parametrize(
    "change",
    [
        {"context": ("Edited sentence one.", "Edited sentence two.")},
        {"question": "An edited question?"},
        {"target": {"label": "yes", "answer": "An edited answer.", "citations": [1]}},
        {"target": {"label": "yes", "answer": "x", "citations": [2]}},
    ],
)
def test_digest_changes_with_every_shown_field(change):
    ex = make.answer_example(row(1), make.AnswerTarget("x", (1,), 1.0, 1.0), "train")
    base = json.loads(ex.model_dump_json())
    if "target" in change:
        base["target"] = change["target"]
    else:
        base.update(change)
    edited = TrainExample.model_validate(base)
    assert audit.digest(edited) != audit.digest(ex)


def test_shown_holds_every_field_the_page_renders():
    ex = make.answer_example(row(1), make.AnswerTarget("x", (1,), 1.0, 1.0), "train")
    assert set(audit.shown(ex)) == {
        "id", "kind", "question", "context", "label", "answer", "citations"
    }  # fmt: skip
    assert audit.shown(ex)["context"] == list(ex.context)


# ---- TGT-06: plan and fill order ----


def test_plan_counts_the_val_split_no_rows_and_is_maximal():
    assert make.plan(no_val_split=25, no_pool=100) == (282, 28)
    assert make.plan(no_val_split=3, no_pool=40) == (110, 11)
    assert make.plan(no_val_split=0, no_pool=40) == (102, 10)


def test_quotas_round_half_up_at_the_real_sizes():
    assert make.Quota.for_size(8436) == make.Quota(no=3003, yes=4505, insufficient=844, offscope=84)
    assert make.Quota.for_size(843) == make.Quota(no=300, yes=451, insufficient=84, offscope=8)
    assert make.Quota.for_size(15) == make.Quota(no=5, yes=8, insufficient=2, offscope=0)


def test_val_is_filled_before_train_and_every_surviving_no_row_is_used():
    val_rows = pool(10, 1, start=0)  # one surviving "no" in the N1.2 val split
    pool_rows = pool(400, 40, start=100)
    sel = make.select(val_rows, pool_rows, FakeScorer(), seed=7, chunk=25)
    used_no = sum(e.kind == "answer" and e.source_label == "no" for e in sel.train + sel.val)
    assert used_no == 41  # plan is maximal: no surviving "no" row is left over
    assert (len(sel.train), len(sel.val)) == make.plan(1, 40)


# ---- TGT-02 / TGT-03 / TGT-05: thresholds are inclusive ----


def test_support_of_exactly_one_half_is_kept():
    r = row(2, long_answer="Alpha bravo charlie delta.", sentences=["Alpha one.", "Bravo one."])
    r["sections"] = ["RESULTS", "RESULTS"]
    (cand,) = make.candidates(r)
    assert cand.options[-1].support == 0.5


def test_entailment_of_exactly_one_half_is_kept():
    (t,) = make.targets_for([row(3)], Fixed(0.5))
    assert isinstance(t, make.AnswerTarget) and t.entail == 0.5
    (t2,) = make.targets_for([row(3)], Fixed(0.499999))
    assert t2 == "no_supported_sentence"


def test_eighty_words_are_kept_and_eighty_one_dropped():
    words = ("Outcome beta4 improved markedly " * 20).split()
    (t80,) = make.targets_for([row(4, long_answer=" ".join(words) + ".")], FakeScorer())
    assert isinstance(t80, make.AnswerTarget) and len(t80.answer.split()) == 80
    (t81,) = make.targets_for([row(4, long_answer=" ".join([*words, "again"]) + ".")], FakeScorer())
    assert t81 == "over_80_words"


def test_reported_support_and_entailment_are_the_weakest_kept_sentence():
    class TwoScores:
        def entail(self, pairs):
            return [0.7 if "secondary" in h.lower() else 0.95 for _, h in pairs]

    la = "Outcome beta5 improved markedly. Secondary measure gamma5 unchanged in treated patients."
    (t,) = make.targets_for([row(5, long_answer=la)], TwoScores())
    assert isinstance(t, make.AnswerTarget) and t.entail == 0.7
    sups = [make.candidates(row(5, long_answer=s))[0].options[0].support for s in la.split(". ")]
    assert t.support == min(sups) < max(sups)


# ---- TGT-01: the recommendation regex starts at a word ----


def test_recommendation_patterns_start_at_a_word_boundary():
    assert not rules.is_recommendation("Treatment was inadvisable in two patients.")
    assert not rules.is_recommendation("Nonclinicians scored the images.")
    assert rules.is_recommendation("Advisory boards were consulted.")  # "advis" from a word start


# ---- TGT-04 / TGT-R6: the cap and option order ----


def test_the_cap_drops_a_sentence_that_would_make_five_citations():
    sents = ["Alpha bravo.", "Charlie delta.", "Echo foxtrot.", "Golf hotel.", "India juliet."]
    la = "Alpha bravo charlie delta echo foxtrot. Golf hotel india juliet."
    r = row(6, long_answer=la, sentences=sents, sections=["RESULTS"] * 5)
    counts = Counter()
    (t,) = make.targets_for(
        [r], AllWords(), counts, version="v2"
    )  # TGT-R6 mechanics: v2 rules (v2.2 ranks by entailment, test_targets_support.py)
    assert isinstance(t, make.AnswerTarget)
    assert t.citations == (1, 2, 3) and counts["sentence:citation_cap"] == 1


def test_citation_options_are_prefixes_of_the_ranking_each_sorted():
    assert rules.citation_options((3, 1, 2)) == ((3,), (1, 3), (1, 2, 3))


# ---- TGT-07: insufficient examples ----


def test_insufficient_drops_cited_non_results_sentences_and_cites_two_question_matches():
    sents = [
        "Background on zeta therapy in older adults.",  # overlaps the question
        "Methods enrolled patients receiving zeta.",  # cited by the answer
        "Dosing of zeta followed older adults weekly.",  # overlaps the question
        "Results showed outcome omega improved.",
    ]
    r = row(
        7,
        long_answer="Patients receiving zeta improved outcome omega.",
        sentences=sents,
        sections=["BACKGROUND", "METHODS", "METHODS", "RESULTS"],
        question="Does zeta help older adults?",
    )
    (t,) = make.targets_for(
        [r], AllWords(), version="v2"
    )  # TGT-R6 mechanics: v2 rules (v2.2 ranks by entailment, test_targets_support.py)
    assert isinstance(t, make.AnswerTarget) and 2 in t.citations
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None
    assert ex.context == (sents[0], sents[2])
    assert ex.target.citations == (1, 2)


def test_both_unknown_section_labels_are_removed_from_insufficient_context():
    assert frozenset({"UNLABELLED", "UNASSIGNED"}) == rules.UNKNOWN_SECTIONS
    r = row(8, sections=["UNASSIGNED", "METHODS", "RESULTS", "RESULTS"])
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    assert make.insufficient_example(r, t, "train", 7) is None


# ---- template choice is salted per use ----


def test_pick_salts_each_choice_separately():
    pmids = [str(1000 + i) for i in range(60)]
    assert any(rules.pick(7, p, "offa", 6) != rules.pick(7, p, "offq", 12) % 6 for p in pmids)
    assert any(rules.pick(7, p, "ins", 4) != rules.pick(7, p, "offq", 12) % 4 for p in pmids)


def test_pool_rows_go_to_val_before_train_in_seeded_order():
    val_rows = pool(10, 1, start=0)
    pool_rows = pool(400, 40, start=100)
    sel = make.select(val_rows, pool_rows, FakeScorer(), seed=7, chunk=25)
    need = make.Quota.for_size(len(sel.val)).no - 1  # val's "no" answers taken from the pool
    first_no = [
        r["pmid"]
        for r in make.seeded(pool_rows, 7)
        if r["label"] == "no" and r["long_answer"] != "Zebra quokka narwhal."
    ][:need]
    val_no = {e.pmid for e in sel.val if e.kind == "answer" and e.source_label == "no"}
    assert need > 0 and set(first_no) <= val_no
