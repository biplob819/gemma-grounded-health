"""TGT-R11 (OWNER D-48, VTG-02): answer citations chosen by single-sentence support. Each
context sentence that TGT-R10 allows is scored alone as the premise of each claim; candidates
report a result (or entail the claim alone) and entail it at least SENTENCE_FLOOR; they are
ranked by that entailment, and the smallest top-k set that passes TGT-R3 and entails the claim
(TGT-R4) is cited. The first two v2 audits failed on citations that landed on background
sentences while the result sentence went uncited."""

from collections import Counter

import pytest

from ggh.targets import make, nli, rules
from tests.targets.test_targets import row


class Table:
    """Entailment by premise: exact premises map to scores, anything else gets `default`."""

    def __init__(self, scores, default=0.0):
        self.scores, self.default, self.pairs = scores, default, []

    def entail(self, pairs):
        self.pairs += list(pairs)
        return [self.scores.get(p, self.default) for p, _ in pairs]


BACKGROUND = "Ara h 6 testing for peanut allergy in pediatric patients is common."
METHOD = "In 181 pediatric patients, peanut allergy diagnosis was founded on oral food challenge."
RESULT = "Ara h 6 was the best predictor of allergy (AUC 0.95)."
OTHER = "Guidelines for Mediterranean patients are lacking."
CLAIM = "Ara h 6 is the best predictor of peanut allergy in pediatric patients."


def ara(i=1):
    return row(i, long_answer=CLAIM, sentences=[BACKGROUND, METHOD, OTHER, RESULT],
               sections=["BACKGROUND", "METHODS", "BACKGROUND", "RESULTS"])  # fmt: skip


def test_the_result_sentence_wins_over_background_with_more_word_overlap():
    r = ara()
    ctx = tuple(r["sentences"])
    # word overlap prefers the background sentence; v2 cites it when NLI lets it through
    assert rules.ranked_citations(CLAIM, ctx)[0] == 1
    sc = Table({BACKGROUND: 0.05, RESULT: 0.97, OTHER: 0.0})
    (t,) = make.targets_for([r], sc)
    assert isinstance(t, make.AnswerTarget) and t.citations == (4,)
    (v2,) = make.targets_for([r], Table({}, default=0.9), version="v2")
    assert isinstance(v2, make.AnswerTarget) and 1 in v2.citations  # what overlap picked


def test_tgt_r10_sentences_are_never_scored_or_cited():
    r = ara()
    sc = Table({RESULT: 0.97}, default=0.99)
    (t,) = make.targets_for([r], sc)
    assert METHOD not in {p for p, _ in sc.pairs}  # excluded before NLI
    assert isinstance(t, make.AnswerTarget) and 2 not in t.citations


def test_a_claim_no_sentence_supports_is_dropped_and_counted():
    r = ara()
    counts = Counter()
    (t,) = make.targets_for([r], Table({RESULT: 0.29, BACKGROUND: 0.2}), counts)
    assert t == "no_supported_sentence" and counts["sentence:no_supporting_sentence"] == 1


def test_a_claim_no_set_entails_is_dropped_as_low_entailment():
    r = ara()
    counts = Counter()
    (t,) = make.targets_for([r], Table({RESULT: 0.4}), counts)  # above the floor, set < 0.5
    assert t == "no_supported_sentence" and counts["sentence:low_entailment"] == 1


def test_the_floor_and_the_result_requirement():
    assert rules.SENTENCE_FLOOR == 0.3 and rules.ENTAIL_MIN == 0.5
    ctx = (BACKGROUND, METHOD, OTHER, RESULT)
    assert rules.support_ranking(CLAIM, ctx, {4: 0.3}, frozenset()) == (4,)  # at the floor
    assert rules.support_ranking(CLAIM, ctx, {4: 0.299999}, frozenset()) == ()
    # a non-result sentence must entail the claim on its own (ENTAIL_MIN), not just the floor
    assert not rules.is_result_like(OTHER) and rules.is_result_like(RESULT)
    assert rules.support_ranking(CLAIM, ctx, {3: 0.45}, frozenset()) == ()
    assert rules.support_ranking(CLAIM, ctx, {3: 0.5}, frozenset()) == (3,)
    assert rules.support_ranking(CLAIM, ctx, {4: 0.9}, frozenset({4})) == ()


def test_ranking_is_by_entailment_then_overlap_then_position_and_capped_at_four():
    ctx = tuple(f"Outcome zeta{i} was higher in group {i}." for i in range(6))
    ent = {1: 0.4, 2: 0.9, 3: 0.4, 4: 0.6, 5: 0.35, 6: 0.31}
    assert rules.support_ranking("Outcome zeta0 rose.", ctx, ent, frozenset()) == (2, 4, 1, 3)


def test_two_partly_supporting_results_are_cited_together_in_entailment_order():
    a = "Statin use was associated with lower troponin release after surgery."
    b = "Statin use was associated with shorter stays in intensive care."
    claim = "Statin use lowers troponin release and shortens intensive care stays."
    r = row(5, long_answer=claim, sentences=["Background on statins.", a, b],
            sections=["BACKGROUND", "RESULTS", "RESULTS"])  # fmt: skip
    sc = Table({a: 0.35, b: 0.45, f"{a} {b}": 0.92})
    (t,) = make.targets_for([r], sc)
    assert isinstance(t, make.AnswerTarget) and t.citations == (2, 3) and t.entail == 0.92
    assert (b, claim) in sc.pairs and (f"{a} {b}", claim) in sc.pairs  # b ranked first


def test_rule_ids_and_rebuilds_from_them():
    assert rules.ANSWER_RULES == (
        "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R12", "TGT-R11", "TGT-R3", "TGT-R4", "TGT-R5", "TGT-R7",
    )  # fmt: skip
    assert rules.rules_version(rules.ANSWER_RULES) == "v2.2"
    assert rules.rules_version(rules.V21_ANSWER_RULES) == "v2.1"
    assert rules.rules_version(rules.V2_INSUFFICIENT_RULES) == "v2"
    assert rules.rules_version(rules.V1_ANSWER_RULES) == "v1"
    r = ara()
    rec = nli.RecordingScorer(Table({BACKGROUND: 0.05, RESULT: 0.97}, default=0.9), 6)
    ex = make.reconstruct(r, rules.ANSWER_RULES, split="train", seed=7, scorer=rec)
    again = make.reconstruct(
        r, rules.ANSWER_RULES, split="train", seed=7, scorer=nli.ReplayScorer(rec.table)
    )
    assert again == ex and ex.target.citations == (4,)
    v2 = make.reconstruct(r, rules.V2_ANSWER_RULES, split="train", seed=7, scorer=rec)
    assert 1 in v2.target.citations and v2.rule_ids == rules.V2_ANSWER_RULES  # overlap picks 1
    with pytest.raises(ValueError, match="unknown rules version"):
        make.targets_for([r], rec, version="v3")
