import json
import re
from collections import Counter
from pathlib import Path

import pytest

from ggh.data import build as data
from ggh.schemas import TrainExample
from ggh.targets import make, nli, rules

ROOT = Path(__file__).resolve().parents[2]


class FakeScorer:
    """Entailed unless the hypothesis carries the word NOTENTAILED; records every pair."""

    def __init__(self):
        self.pairs = []

    def entail(self, pairs):
        self.pairs += list(pairs)
        return [0.1 if "NOTENTAILED" in h else 0.93 for _, h in pairs]


def row(i, label="yes", long_answer=None, sections=None, sentences=None, question=None):
    sentences = sentences or [
        f"Background on topic {i} and its wider clinical context.",
        f"Methods enrolled {i} patients receiving treatment alpha{i}.",
        f"Results showed outcome beta{i} improved markedly in treated patients.",
        f"Secondary measure gamma{i} was unchanged.",
    ]
    return {
        "pmid": str(1000 + i),
        "question": question or f"Does treatment alpha{i} improve outcome beta{i}?",
        "sentences": sentences,
        "sections": sections or ["BACKGROUND", "METHODS", "RESULTS", "RESULTS"],
        "long_answer": long_answer
        if long_answer is not None
        else f"Outcome beta{i} improved markedly.",  # overlaps the results sentence only
        "label": label,
    }


# ---- TGT-01: recommendation filter ----

RECOMMEND_CASES = {
    r"should\b": ("Patients should be screened yearly.", "Shoulder pain fell after surgery."),
    "recommend": ("This regimen is recommended for adults.", "Doses were compared in adults."),
    r"must\b": ("Staff must check levels.", "Mustard exposure was recorded."),
    "we suggest": ("We suggest routine testing.", "We tested routinely."),
    r"clinicians?\b": ("Clinicians may use this score.", "Clinical scores were used."),
    "further stud": ("Further studies are needed.", "Further analysis showed no change."),
    "warrant(?!y)": ("This finding warrants caution.", "A warranty claim was filed."),
    r"need to\b": ("Trials need to confirm this.", "The need for a trial is clear."),
    "advis": ("It is advisable to monitor.", "Monitoring was frequent."),
}


def test_every_recommendation_pattern_has_a_matching_and_a_non_matching_sentence():
    assert tuple(RECOMMEND_CASES) == rules.RECOMMEND_PATTERNS  # pinned list, in order
    for pat, (hit, miss) in RECOMMEND_CASES.items():
        assert rules.is_recommendation(hit), pat
        assert not rules.is_recommendation(miss), pat


def test_recommendation_sentences_never_reach_a_target():
    r = row(1, long_answer="Outcome beta1 improved markedly. Clinicians should use alpha1.")
    counts = Counter()
    cands = make.candidates(r, counts)
    assert [c.sentence for c in cands] == ["Outcome beta1 improved markedly."]
    assert counts["sentence:recommendation"] == 1


# ---- TGT-R9: hedge strip ----


def test_hedge_strip_removes_only_a_leading_hedge():
    assert rules.strip_hedge("These findings suggest that drug X lowers BP.") == (
        "Drug X lowers BP."
    )
    assert rules.strip_hedge("Our results indicate a benefit.") == "A benefit."
    assert rules.strip_hedge("Taken together, the data show that risk fell.") == "Risk fell."
    assert rules.strip_hedge("Drug X lowers BP, as our results suggest.") == (
        "Drug X lowers BP, as our results suggest."
    )


# ---- TGT-02: lexical support ----


def test_content_words_stopwords_and_stemming_are_pinned():
    assert rules.content_words("The patients were treated with the drugs") == {
        "patient",
        "treat",
        "drug",
    }
    assert rules.stem("improvements") == "improve" and rules.stem("studies") == "stud"
    assert rules.stem("was") == "was" and rules.stem("bus") == "bus"  # keeps a 3-letter stem
    assert {"the", "with", "were", "not"} <= rules.STOPWORDS
    assert len(rules.STOPWORDS) == 95


def test_support_is_the_share_of_content_words_found_in_the_cited_sentences():
    s = "Mortality fell sharply in treated patients."  # mortality fell sharply treat patient
    assert rules.support(s, ["Mortality fell in patients."]) == pytest.approx(3 / 5)
    assert rules.support(s, ["Nothing alike here."]) == 0.0
    r = row(2, long_answer="Mortality fell sharply among treated elderly smokers yesterday.")
    r["sentences"] = ["Mortality was recorded.", "Other words entirely."]
    counts = Counter()
    assert make.candidates(r, counts) == [] and counts["sentence:low_support"] == 1


# ---- TGT-03: NLI ----


def test_nli_premise_is_exactly_the_cited_sentences_and_low_entailment_drops():
    r = row(3, long_answer="Outcome beta3 improved markedly. Outcome beta3 improved NOTENTAILED.")
    sc = FakeScorer()
    counts = Counter()
    (t,) = make.targets_for([r], sc, counts)
    assert isinstance(t, make.AnswerTarget)
    ctx = tuple(r["sentences"])
    for premise, hyp in sc.pairs:  # every premise is exactly one citation option's sentences
        options = rules.citation_options(rules.ranked_citations(hyp, ctx))
        assert premise in {" ".join(ctx[i - 1] for i in o) for o in options}
    assert t.answer == "Outcome beta3 improved markedly." and t.entail == 0.93
    assert counts["sentence:low_entailment"] == 1


def test_entailment_index_is_read_from_id2label():
    assert nli.entailment_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 1
    assert nli.entailment_index({"0": "ENTAILMENT", "1": "neutral"}) == 0
    with pytest.raises(ValueError):
        nli.entailment_index({0: "yes", 1: "no"})
    assert (nli.MODEL_ID, nli.MODEL_REVISION) == (
        "cross-encoder/nli-deberta-v3-base",
        "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    )


def test_replay_refuses_an_unrecorded_pair_and_recording_rounds():
    class Fixed:
        def entail(self, pairs):
            return [0.1234567] * len(pairs)

    rec = nli.RecordingScorer(Fixed(), 6)
    assert rec.entail([("p", "h"), ("p", "h")]) == [0.123457, 0.123457]
    assert nli.ReplayScorer(rec.table).entail([("p", "h")]) == [0.123457]
    with pytest.raises(KeyError):
        nli.ReplayScorer(rec.table).entail([("p", "other")])


def test_targets_code_never_downloads_in_tests_and_has_no_other_text_source():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src/ggh/targets").glob("*.py"))
    for banned in ("anthropic", "openai", "google.genai", "requests", "urllib", "http.client"):
        assert banned not in src, banned


# ---- TGT-04: citations ----


def test_citations_are_the_top_four_non_zero_overlaps_sorted():
    ctx = (
        "alpha beta gamma delta.",  # 4/4
        "unrelated words only.",  # 0
        "alpha beta.",  # 2/4
        "alpha.",  # 1/4
        "beta.",  # 1/4
        "gamma.",  # 1/4
    )
    ranked = rules.ranked_citations("alpha beta gamma delta", ctx)
    assert ranked == (1, 3, 4, 5)  # best first, ties by position, zero overlap never ranked
    assert rules.citation_options(ranked) == ((1,), (1, 3), (1, 3, 4), (1, 3, 4, 5))
    assert rules.ranked_citations("nothing shared", ctx) == ()
    r = row(4, long_answer="Zebra quokka narwhal.")
    counts = Counter()
    assert make.candidates(r, counts) == [] and counts["sentence:zero_overlap"] == 1


def test_merged_citations_never_exceed_four_and_stay_in_range():
    sents = [f"word{i} shared{i} common." for i in range(8)]
    # each sentence needs three cited sentences to reach 0.5 support: 6 of 9 content words
    la = (
        "Word0 shared0 word1 shared1 word2 shared2 xaa xbb xcc. "
        "Word4 shared4 word5 shared5 word6 shared6 xdd xee xff."
    )
    r = row(5, long_answer=la, sentences=sents, sections=["RESULTS"] * 8)
    counts = Counter()
    (t,) = make.targets_for([r], FakeScorer(), counts)
    assert isinstance(t, make.AnswerTarget)
    assert len(t.citations) <= 4 and list(t.citations) == sorted(set(t.citations))
    assert counts["sentence:citation_cap"] == 1


def test_the_smallest_citation_option_that_entails_is_chosen():
    class NeedsBoth:  # entails only when the premise holds both halves of the claim
        def entail(self, pairs):
            return [0.9 if "alpha9" in p and "beta9" in p else 0.2 for p, _ in pairs]

    sents = [
        "Background on alpha9 dosing.",
        "Treatment alpha9 was given daily.",
        "Outcome beta9 improved.",
        "Unrelated sentence.",
    ]
    r = row(9, long_answer="Treatment alpha9 improved outcome beta9.", sentences=sents)
    (t,) = make.targets_for([r], NeedsBoth())
    assert isinstance(t, make.AnswerTarget)
    assert t.citations == (2, 3)  # not padded to every overlapping sentence
    (t1,) = make.targets_for([r], FakeScorer())
    assert isinstance(t1, make.AnswerTarget) and len(t1.citations) < 3


# ---- TGT-05: drops ----


def test_long_answers_and_rows_without_a_surviving_sentence_are_dropped_by_reason():
    long = " ".join(["Outcome beta6 improved markedly in treated patients."] * 12)
    (t1,) = make.targets_for([row(6, long_answer=long)], FakeScorer())
    (t2,) = make.targets_for([row(7, long_answer="Patients should rest.")], FakeScorer())
    assert (t1, t2) == ("over_80_words", "no_supported_sentence")


# ---- TGT-07: insufficient ----


def test_insufficient_removes_results_and_cited_sentences_and_cites_the_question_match():
    r = row(8)
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None
    assert ex.context == tuple(r["sentences"][:2])  # BACKGROUND, METHODS kept; RESULTS gone
    assert ex.target.label == "maybe" and ex.target.answer in rules.INSUFFICIENT_ANSWERS
    assert ex.target.citations == (2,)  # the methods sentence shares "treatment alpha8"
    assert ex.prompt == make._prompt(r["pmid"], r["question"], ex.context, "maybe")
    assert "[1] Background on topic 8" in ex.prompt and "Results showed" not in ex.prompt
    no_results = row(9, sections=["BACKGROUND", "METHODS", "METHODS", "METHODS"])
    (t9,) = make.targets_for([no_results], FakeScorer())
    assert isinstance(t9, make.AnswerTarget)
    assert make.insufficient_example(no_results, t9, "train", 7) is None  # INS-R1 skip
    unknown = row(10, sections=["UNLABELLED", "METHODS", "RESULTS", "RESULTS"])
    (t10,) = make.targets_for([unknown], FakeScorer())
    assert isinstance(t10, make.AnswerTarget)
    assert make.insufficient_example(unknown, t10, "train", 7) is None  # < 2 left (INS-R2)


def test_four_fixed_insufficient_answers_and_twelve_by_six_offscope_templates():
    assert len(rules.INSUFFICIENT_ANSWERS) == 4
    assert len(set(rules.OFFSCOPE_QUESTIONS)) == 12 and len(set(rules.OFFSCOPE_ANSWERS)) == 6


# ---- TGT-08: off-scope ----


def _grams(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def test_offscope_templates_share_no_five_gram_with_the_evaluation_set():
    evaluation = _grams((ROOT / data.OFFSCOPE_PATH).read_text(encoding="utf-8"))
    for t in rules.OFFSCOPE_QUESTIONS + rules.OFFSCOPE_ANSWERS:
        assert not (_grams(t) & evaluation), t


def test_offscope_example_declines_and_cites_one():
    ex = make.offscope_example(row(11), "train", 7)
    assert ex.question in rules.OFFSCOPE_QUESTIONS and ex.target.answer in rules.OFFSCOPE_ANSWERS
    assert ex.target.label == "maybe" and ex.target.citations == (1,)


# ---- TGT-06: the mix ----


def pool(n_yes, n_no, start=0):
    out = [row(start + i, "yes") for i in range(n_yes)]
    out += [row(start + n_yes + i, "no") for i in range(n_no)]
    out += [row(start + n_yes + n_no, "no", long_answer="Zebra quokka narwhal.")]  # dropped
    return out


def test_quota_shares_and_plan():
    q = make.Quota.for_size(1000)
    assert (q.insufficient, q.offscope, q.no, q.yes) == (100, 10, 356, 534)
    n, v = make.plan(no_val_split=0, no_pool=40)
    assert v == n // 10 and make.Quota.for_size(n).no + make.Quota.for_size(v).no <= 40
    assert make.Quota.for_size(n + 1).no + make.Quota.for_size((n + 1) // 10).no > 40
    assert make.plan(0, 10**6, cap=500) == (500, 50)


def test_mix_holds_and_train_and_val_share_no_pmid():
    val_rows = pool(30, 4, start=0)
    pool_rows = pool(600, 60, start=100)
    sel = make.select(val_rows, pool_rows, FakeScorer(), seed=7, chunk=50)
    for split, rows in (("train", sel.train), ("val", sel.val)):
        q = make.Quota.for_size(len(rows))
        kinds = Counter((e.kind, e.source_label if e.kind == "answer" else None) for e in rows)
        assert kinds[("answer", "no")] == q.no and kinds[("answer", "yes")] == q.yes, split
        assert kinds[("insufficient", None)] == q.insufficient
        assert kinds[("offscope", None)] == q.offscope
        for e in rows:
            TrainExample.model_validate_json(e.model_dump_json())
            assert e.split == split
    assert len(sel.val) == len(sel.train) // 10
    assert not {e.pmid for e in sel.train} & {e.pmid for e in sel.val}
    val_pmids = {r["pmid"] for r in val_rows}
    assert all(e.pmid not in val_pmids for e in sel.train)  # train draws from the pool only
    assert {e.pmid for e in sel.val if e.kind == "answer" and e.source_label == "no"} >= {
        r["pmid"]
        for r in val_rows
        if r["label"] == "no" and r["long_answer"] != "Zebra quokka narwhal."
    }  # val uses the N1.2 val rows first
    assert sel.report["shortfall_vs_cap"] == make.TRAIN_CAP - len(sel.train)
    assert sel.report["row_drops"] == {"no:no_supported_sentence": 2}


def test_selection_is_deterministic_and_seeded():
    a = make.select(pool(30, 4), pool(300, 40, start=100), FakeScorer(), seed=7)
    b = make.select(pool(30, 4), pool(300, 40, start=100), FakeScorer(), seed=7)
    c = make.select(pool(30, 4), pool(300, 40, start=100), FakeScorer(), seed=8)
    assert [e.id for e in a.train] == [e.id for e in b.train]
    assert [e.id for e in a.train] != [e.id for e in c.train]


# ---- TGT-09: provenance ----


def test_every_example_is_reconstructed_from_its_row_and_rule_ids():
    val_rows, pool_rows = pool(30, 4), pool(300, 40, start=100)
    rec = nli.RecordingScorer(FakeScorer(), rules.SCORE_DECIMALS)
    sel = make.select(val_rows, pool_rows, rec, seed=7)
    by_pmid = {r["pmid"]: r for r in val_rows + pool_rows}
    replay = nli.ReplayScorer(rec.table)
    for e in sel.train + sel.val:
        again = make.reconstruct(by_pmid[e.pmid], e.rule_ids, split=e.split, seed=7, scorer=replay)
        assert again == e, e.id
    edited = dict(by_pmid[sel.train[0].pmid], long_answer="Something else entirely.")
    with pytest.raises((ValueError, KeyError)):
        make.reconstruct(edited, sel.train[0].rule_ids, split="train", seed=7, scorer=replay)


def test_rule_ids_are_pinned():
    # TGT-R10: OWNER D-43, tests/targets/test_targets_aim.py
    assert rules.ANSWER_RULES == (
        "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R10", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7",
    )  # fmt: skip
    assert (*rules.ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3") == rules.INSUFFICIENT_RULES
    assert rules.OFFSCOPE_RULES == ("OFF-R3",)
    assert (rules.SUPPORT_MIN, rules.ENTAIL_MIN, rules.MAX_WORDS, rules.MAX_CITES) == (
        0.5, 0.5, 80, 4,
    )  # fmt: skip


# ---- the real build (local artifacts only) ----

TARGETS = ROOT / make.OUT_DIR


@pytest.mark.data
@pytest.mark.skipif(not (TARGETS / "train.jsonl").exists(), reason="targets are built locally")
def test_real_targets_reconstruct_from_their_n12_rows():
    table = {r["key"]: r["entail"] for r in data.read_jsonl(TARGETS / "nli_scores.jsonl")}
    replay = nli.ReplayScorer(table)
    rows = {}
    for name in ("val.jsonl", "train_pool.jsonl"):
        for r in data.read_jsonl(ROOT / data.OUT_DIR / name):
            rows[r["pmid"]] = r
    n = 0
    for split in ("train", "val"):
        for line in (TARGETS / f"{split}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            e = TrainExample.model_validate_json(line)
            again = make.reconstruct(rows[e.pmid], e.rule_ids, split=split, seed=7, scorer=replay)
            assert again == e, e.id
            n += 1
    report = json.loads((ROOT / make.REPORT_COPY).read_text(encoding="utf-8"))
    assert n == report["train"] + report["val"]


def test_yes_rows_are_not_scored_once_their_quotas_are_full():
    """The scan must reach every scarce "no" row, but scoring every "yes" row on the way made
    the first real build run for an hour (the no rows sit across all 210,769 rows)."""
    sc = FakeScorer()
    yes_late = [row(5000 + i, "yes") for i in range(3000)]
    sel = make.select(pool(30, 4), pool(200, 40, start=100) + yes_late, sc, seed=7, chunk=100)
    yes_needed = sum(1 for e in sel.train + sel.val if e.kind != "offscope")
    assert len(sc.pairs) < 3000  # far fewer than the 3,230 yes rows available
    assert len(sc.pairs) <= 45 + 100 * 3 + yes_needed * 3  # no rows + a few chunks of slack
