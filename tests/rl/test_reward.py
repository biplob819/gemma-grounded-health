"""RLR-01, RLR-02: the v3 GRPO reward, and the planted reward hacks it must not pay for."""

import json
import statistics

import pytest

from ggh.rl import reward as R

# dev-1471704 (PubMedQA, the abstract the offline demo uses): [1]-[3] aim and design, [5] method
CONTEXT = (
    "Our purpose was to investigate the source and role of elevated levels of immunoreactive "
    "beta-endorphin in polycystic ovary syndrome.",
    "We wished to determine whether immunoreactive beta-endorphin secretion in patients with "
    "polycystic ovary syndrome is influenced by body weight and whether the pituitary release of "
    "immunoreactive beta-endorphin with corticotropin-releasing hormone is related to luteinizing "
    "hormone levels or adrenal androgen secretion.",
    "Eighteen patients with polycystic ovary syndrome and 10 ovulatory controls were studied.",
    "Each subject received 1 microgram/kg intravenous corticotropin-releasing hormone and an oral "
    "glucose tolerance test on alternate days.",
    "Levels of plasma immunoreactive beta-endorphin, corticotropin, luteinizing hormone, cortisol, "
    "adrenal androgens, and insulin were measured.",
    "Although immunoreactive beta-endorphin levels were elevated in patients with polycystic ovary "
    "syndrome (p < 0.01), incremental responses after corticotropin-releasing hormone were similar "
    "to controls and were not influenced by body weight.",
    "Serum luteinizing hormone levels were not affected by corticotropin-releasing hormone and did "
    "not correlate with immunoreactive beta-endorphin levels.",
    "Adrenal androgen responses after corticotropin-releasing hormone were increased in patients "
    "with polycystic ovary syndrome (p < 0.01) but were not correlated with immunoreactive "
    "beta-endorphin secretion.",
    "After oral glucose was given, elevated fasting insulin levels increased significantly in "
    "patients with polycystic ovary syndrome (p < 0.01), as did immunoreactive beta-endorphin "
    "levels (p < 0.05).",
    "The increases in insulin and immunoreactive beta-endorphin levels were correlated (p < 0.05).",
)
GOLD = "no"
GOOD = (
    "The pituitary response of immunoreactive beta-endorphin to corticotropin-releasing hormone "
    "was similar to controls."
)
AIM_ANSWER = (
    "The pituitary release of immunoreactive beta-endorphin is not influenced by body weight."
)


def ans(label: str, answer: str, citations: list[int]) -> str:
    return json.dumps({"label": label, "answer": answer, "citations": citations})


def s(text: str, gold: str = GOLD) -> R.Score:
    return R.score(text, gold, CONTEXT)


def test_constants_are_pinned():
    assert (R.R_UNPARSED, R.R_LABEL, R.R_GROUNDED, R.P_AIM) == (-1.0, 1.0, 0.5, 0.5)
    assert (R.MAX_CITES, R.MIN_SHARED_WORDS, R.MIN_WORDS, R.MAX_WORDS) == (3, 2, 6, 80)
    assert R.LABELS == ("yes", "no", "maybe")


def test_a_correct_grounded_answer_earns_the_maximum():
    got = s(ans("no", GOOD, [6]))
    assert got == R.Score(True, True, True, False, 1.5, "")


def test_unparsable_output_earns_the_floor():
    bad_label = '{"label": "perhaps", "answer": "x", "citations": []}'
    for text in ("", "no", "The answer is no [6].", bad_label):
        assert s(text).reward == R.R_UNPARSED, text


def test_a_wrong_label_earns_nothing_for_its_citations():
    got = s(ans("yes", GOOD, [6]))
    assert (got.label_correct, got.grounded, got.reward) == (False, True, 0.0)


def test_the_expected_reward_favours_the_most_likely_label():
    """No premium for yes/no over maybe: with beliefs maybe 0.45, yes 0.35, no 0.20 and grounded
    citations either way, answering maybe has the highest expected reward."""
    beliefs = {"maybe": 0.45, "yes": 0.35, "no": 0.20}
    unsure = ans("maybe", "The sentences report no result that settles this.", [1])
    texts = {"yes": ans("yes", GOOD, [6]), "no": ans("no", GOOD, [6]), "maybe": unsure}

    def expected(label: str) -> float:
        return sum(p * s(texts[label], gold).reward for gold, p in beliefs.items())

    assert max(beliefs, key=expected) == "maybe"


def test_hack_all_maybe_earns_nothing_on_a_yes_no_question():
    for cites in ([], [6], [1, 2]):
        got = s(ans("maybe", "The abstract does not settle whether secretion is increased.", cites))
        assert got.reward == 0.0 and got.reason == "maybe for a yes/no question", cites


def test_maybe_is_paid_when_the_gold_label_is_maybe():
    text = ans("maybe", "The sentences describe the study's aim and methods only.", [1, 3])
    assert s(text, gold="maybe").reward == 1.5  # the insufficient targets cite the design
    assert s(ans("maybe", "No result.", [11]), gold="maybe").grounded is False  # out of range


def test_hack_an_empty_or_uncited_maybe_earns_the_label_only():
    for text, why in (
        (ans("maybe", "", []), "no citation"),
        (ans("maybe", "", [1]), "answer under 6 words"),
        (ans("maybe", "x", [1]), "answer under 6 words"),
        (ans("maybe", "The sentences report no result here at all.", [1, 2, 3, 4]),
         "more than 3 citations"),
    ):  # fmt: skip
        got = s(text, gold="maybe")
        assert (got.reward, got.reason) == (1.0, why), text


def test_hack_valid_but_unsupporting_citation_earns_no_grounding():
    # [10] is a real, in-range result sentence, but about insulin, not the answer's claim
    unrelated = "Pituitary secretion does not explain the raised hormone in these women."
    got = s(ans("no", unrelated, [10]))
    assert got.label_correct and not got.grounded and got.reward == 1.0
    assert "shares under 2 words" in got.reason


def test_hack_citing_the_aim_sentence_is_penalised():
    got = s(ans("no", AIM_ANSWER, [2]))
    assert got.aim_cited and not got.grounded and got.reward == 1.0 - R.P_AIM
    both = s(ans("no", GOOD, [2, 6]))
    assert both.aim_cited and both.reward == 0.5


def test_hack_citing_a_method_sentence_withholds_the_credit_without_the_aim_penalty():
    method = "Levels of immunoreactive beta-endorphin and insulin were measured in plasma."
    got = s(ans("no", method, [5]))
    assert not got.aim_cited and not got.grounded and got.reward == 1.0


def test_hack_citing_everything_earns_no_grounding():
    got = s(ans("no", GOOD, [6, 7, 8, 9]))
    assert got.reason == "more than 3 citations" and got.reward == 1.0


def test_hack_very_short_or_empty_answers_earn_no_grounding():
    for text in ("No.", "", "Not pituitary secretion here."):
        got = s(ans("no", text, [6]))
        assert not got.grounded and "under 6 words" in got.reason, text
    assert not s(ans("no", " ".join(["word"] * 81), [6])).grounded


def test_out_of_range_and_repeated_citations_earn_no_grounding():
    assert s(ans("no", GOOD, [0])).reward == R.R_UNPARSED  # EvalKit's schema rejects index 0
    assert s(ans("no", GOOD, [11])).reason == "citation out of range"
    assert s(ans("no", GOOD, [6, 6])).reason == "repeated citation"
    assert s(ans("no", GOOD, [])).reason == "no citation"


def test_the_reward_is_deterministic():
    text = ans("no", GOOD, [6])
    assert len({s(text).reward for _ in range(20)}) == 1


def test_no_hack_policy_beats_honest_answers_on_a_mixed_set():
    """Expected reward of each planted hack policy across yes, no and maybe questions is below an
    honest policy's, even when the honest policy gets a third of the labels wrong."""
    golds = ["yes"] * 5 + ["no"] * 4 + ["maybe"]
    unsure = ans("maybe", "The sentences report no result on this question.", [1])

    def honest_text(i: int, g: str) -> str:
        if g == "maybe":
            return unsure
        wrong = "yes" if g == "no" else "no"
        return ans(g if i % 3 else wrong, GOOD, [6])  # a third of the labels wrong

    honest = [s(honest_text(i, g), g).reward for i, g in enumerate(golds)]
    hacks = {
        "all maybe": [s(ans("maybe", "Unclear from the abstract.", []), g).reward for g in golds],
        "all yes, aim cite": [s(ans("yes", GOOD, [1]), g).reward for g in golds],
        "all yes, cite all": [s(ans("yes", GOOD, [6, 7, 8, 9]), g).reward for g in golds],
        "all yes, terse": [s(ans("yes", "Yes.", [6]), g).reward for g in golds],
        "unrelated cite": [s(ans(g, "Insulin rose after glucose.", [10]), g).reward for g in golds],
        "terse maybe": [s(ans("maybe", "", []), g).reward for g in golds],
    }
    best = statistics.mean(honest)
    for name, rewards in hacks.items():
        assert statistics.mean(rewards) < best, name


def test_grpo_reward_reads_trl_chat_and_text_completions():
    good = ans("no", GOOD, [6])
    chat = [[{"role": "assistant", "content": good}], [{"role": "assistant", "content": "oops"}]]
    assert R.grpo_reward(chat, [GOLD, GOLD], [CONTEXT, CONTEXT]) == [1.5, -1.0]
    assert R.grpo_reward([good], ["no"], [list(CONTEXT)], prompts=["ignored"]) == [1.5]
    with pytest.raises(ValueError):
        R.grpo_reward([good, good], ["no"], [CONTEXT, CONTEXT])


def test_an_unknown_gold_label_is_an_error():
    with pytest.raises(ValueError):
        R.score(ans("no", GOOD, [6]), "insufficient", CONTEXT)
