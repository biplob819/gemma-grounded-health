"""RLB-01, RLB-02: the v3b checker reward and the planted hacks it must not pay for. A fake checker
stands in for the two models: it supports a claim when most of its content words appear in the
evidence (the real checker is measured on real dev answers in scripts/v3b_checker_eval.py)."""

import json
import statistics
from pathlib import Path

import pytest

from ggh.rl import checker as C
from ggh.rl import reward_b as RB
from ggh.targets.rules import content_words
from tests.rl.test_reward import AIM_ANSWER, CONTEXT, GOOD, ans

ROOT = Path(__file__).resolve().parents[2]


class FakeChecker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def support(self, pairs):
        self.calls += list(pairs)
        out = []
        for evidence, claim in pairs:
            words = content_words(claim)
            out.append(len(words & content_words(evidence)) / max(len(words), 1))
        return out


def score(text, gold="no", ctx=CONTEXT, checker=None):
    return RB.score_batch([text], [gold], [ctx], checker or FakeChecker())[0]


def test_constants_and_checker_are_pinned():
    assert (RB.R_LABEL, RB.R_GROUNDED, RB.R_GROUNDED_MAYBE, RB.COPY_RATIO) == (0.25, 1.0, 0.5, 0.9)
    assert C.CANARY_TOLERANCE == 0.01
    assert C.THRESHOLD == 0.65 and C.MAX_TOKENS == 512
    assert [m["repo"] for m in C.MODELS] == [
        "lytang/MiniCheck-RoBERTa-Large", "cross-encoder/nli-deberta-v3-large"]  # fmt: skip
    assert [len(m["revision"]) for m in C.MODELS] == [40, 40]
    assert [m["label"] for m in C.MODELS] == ["1", "entailment"]
    assert C.label_index({0: "contradiction", 1: "entailment", 2: "neutral"}, "entailment") == 1
    with pytest.raises(ValueError):
        C.label_index({0: "a", 1: "b"}, "entailment")


def test_a_right_supported_answer_earns_the_maximum():
    sc = score(ans("no", GOOD, [6]))
    assert sc.reward == 1.25 and sc.grounded and sc.support is not None and not sc.reason


def test_a_right_label_the_checker_rejects_earns_the_label_only():
    sc = score(ans("no", "Insulin resistance causes the syndrome in every patient studied.", [7]))
    assert sc.reward == 0.25 and not sc.grounded and sc.reason.startswith("checker support")


def test_a_wrong_label_earns_nothing_and_is_never_checked():
    fake = FakeChecker()
    sc = score(ans("yes", GOOD, [6]), checker=fake)
    assert sc.reward == 0.0 and fake.calls == [] and sc.support is None
    assert sc.reason == "wrong label: not checked"


def test_hack_citing_the_aim_sentence_is_penalised():
    sc = score(ans("no", AIM_ANSWER, [2]))
    assert sc.aim_cited and sc.reward == 0.25 - 0.5 and "aim or method" in sc.reason


def test_hack_structure_failures_earn_the_label_only():
    for text in (
        ans("no", GOOD, [6, 7, 8, 9]),  # citing many sentences
        ans("no", "Not related.", [7]),  # terse
        ans("no", GOOD, []),  # uncited
        ans("no", GOOD, [6, 6]),  # repeated
        ans("no", GOOD, [11]),  # out of range
    ):
        sc = score(text)
        assert sc.reward == 0.25 and not sc.grounded, text


def test_hack_unparsed_and_maybe_for_yes_no():
    assert score("no idea").reward == -1.0
    assert score(ans("maybe", GOOD, [6])).reward == 0.0


def test_maybe_answers_to_maybe_questions_skip_the_checker():
    fake = FakeChecker()
    text = ans("maybe", "The abstract does not report whether the pituitary is the source.", [6])
    sc = score(text, gold="maybe", checker=fake)
    assert sc.reward == 0.75 and fake.calls == [] and sc.support is None
    empty = score(ans("maybe", "", []), gold="maybe")
    assert empty.reward == 0.25  # an empty maybe earns the label only


def test_maybe_carries_no_premium_over_a_typical_yes_no():
    # a right maybe (0.75) earns less than a right yes/no's expected reward at the 56 % pass rate
    # of v2's yes/no gold answers (0.25 + 0.56 x 1.0 = 0.81)
    gold = json.loads((ROOT / "docs/measurements/v3/gold_checker.json").read_text("utf-8"))
    yes_no = [gold["by_label"][k] for k in ("yes", "no")]
    rate = sum(b["pass"] for b in yes_no) / sum(b["targets"] for b in yes_no)
    assert RB.R_LABEL + rate * RB.R_GROUNDED > RB.R_LABEL + RB.R_GROUNDED_MAYBE


def test_hack_maybe_filler_or_aim_citation_on_a_maybe_question_is_bounded():
    filler = score(ans("maybe", "Further studies are needed to know more about this.", [6]),
                   gold="maybe")  # fmt: skip
    aim = score(ans("maybe", "The study wished to determine the role of beta-endorphin.", [2]),
                gold="maybe")  # fmt: skip
    assert filler.reward == aim.reward == 0.75  # not checked by design; the maybe guard bounds it


def test_copying_a_cited_result_sentence_is_paid_and_measured():
    sc = score(ans("no", CONTEXT[6], [7]))
    assert sc.copied and sc.copy_similarity == 1.0 and sc.reward == 1.25  # guarded in the plan
    good = score(ans("no", GOOD, [6]))
    assert not good.copied and 0 < good.copy_similarity < RB.COPY_RATIO


def test_the_real_checkers_copy_blind_spot_is_recorded():
    probes = {p["probe"]: p["support"] for p in json.loads(
        (ROOT / "docs/measurements/v3/checker_probes.json").read_text("utf-8"))}  # fmt: skip
    assert probes["verbatim copy of the cited sentence"] >= C.THRESHOLD
    assert probes["copy of an unrelated result sentence"] >= C.THRESHOLD
    assert probes["honest answer drawing the question's conclusion"] < C.THRESHOLD


def test_the_canary_catches_a_checker_that_scores_differently():
    canary = json.loads((ROOT / C.CANARY).read_text(encoding="utf-8"))
    table = {(p["evidence"], p["claim"]): p["support"] for p in canary["pairs"]}

    class Replay:
        def __init__(self, shift=0.0):
            self.shift = shift

        def support(self, pairs):
            return [table[p] + self.shift for p in pairs]

    assert len(canary["pairs"]) == 3 and C.canary_problem(Replay(), canary) == ""
    assert "canary" in C.canary_problem(Replay(0.02), canary)
    for p in canary["pairs"]:  # the committed per-answer scores back each canary
        assert p["support"] == round((p["minicheck_roberta"] + p["deberta_large"]) / 2, 4)


def test_the_checker_runs_once_per_batch_for_eligible_replies_only():
    fake = FakeChecker()
    texts = [ans("no", GOOD, [6]), ans("yes", GOOD, [6]), "junk", ans("no", GOOD, [6, 7, 8, 9])]
    out = RB.score_batch(texts, ["no"] * 4, [CONTEXT] * 4, fake)
    assert [o.reward for o in out] == [1.25, 0.0, -1.0, 0.25] and len(fake.calls) == 1
    with pytest.raises(ValueError):
        RB.score_batch(texts, ["no"], [CONTEXT], fake)
    with pytest.raises(ValueError):
        RB.score_batch(["x"], ["probably"], [CONTEXT], fake)


def test_no_hack_policy_beats_an_honest_policy_with_a_third_wrong():
    def mean(texts):
        return statistics.fmean(s.reward for s in RB.score_batch(
            texts, ["no"] * len(texts), [CONTEXT] * len(texts), FakeChecker()))  # fmt: skip

    honest = mean([ans("no", GOOD, [6]), ans("no", GOOD, [6]), ans("yes", GOOD, [6])])
    hacks = {
        "always maybe on yes/no": [ans("maybe", GOOD, [6])] * 3,
        "right label, unsupported text": [ans("no", "Insulin causes every case seen here.", [7])]
        * 3,
        "right label, aim citation": [ans("no", AIM_ANSWER, [2])] * 3,
        "right label, many citations": [ans("no", GOOD, [6, 7, 8, 9])] * 3,
        "unparsed": ["no"] * 3,
    }
    for name, texts in hacks.items():
        assert mean(texts) < honest, name


def test_the_tuning_ids_are_the_200_dev_grounding_cases():
    ids = json.loads((ROOT / C.TUNING_IDS).read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids)) == 200 and all(i.isdigit() for i in ids)
    import gzip

    with gzip.open(ROOT / "docs/results/dev_ground_v1/cases.jsonl.gz", "rt", encoding="utf-8") as f:
        cases = {json.loads(x)["id"].removeprefix("dev-") for x in f}
    assert set(ids) == cases


def test_the_committed_checker_evaluation_backs_the_choice():
    ev = json.loads((ROOT / "docs/measurements/v3/checker_eval.json").read_text("utf-8"))
    best = ev["checkers"]["ens_minicheck_roberta_deberta_large_cited"]
    assert best["threshold"] == C.THRESHOLD
    assert best["cv_agreement_with_sol"] == max(c["cv_agreement_with_sol"]
                                                for c in ev["checkers"].values())  # fmt: skip
    ph = best["proxy_headline"]
    assert ph["base"] > ph["v2"] > ph["v1"]  # the headline's order (Sol: base > v1 ~ v2)
    gold = json.loads((ROOT / "docs/measurements/v3/gold_checker.json").read_text("utf-8"))
    assert gold["by_label"]["maybe"]["pass"] == 0  # why maybe answers skip the checker
