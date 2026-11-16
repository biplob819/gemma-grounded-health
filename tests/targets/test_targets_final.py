"""N1.3 verifier round 3: hedged conclusions through the pipeline, INS-R1/R3 coverage, the cap
with overlapping options, pinned stemming, a tie-free ranking and the drop report fields."""

import hashlib
import json
from collections import Counter

from ggh.data import build as data
from ggh.targets import make, nli, rules
from tests.targets.test_targets import FakeScorer, row
from tests.targets.test_targets_build import repo  # noqa: F401  (fixture)
from tests.targets.test_targets_edges import AllWords

# ---- TGT-R9 through the pipeline ----


def test_a_hedged_conclusion_is_stripped_before_ranking_support_and_the_answer():
    r = row(40, long_answer="These findings suggest that outcome beta40 improved markedly.")
    sc = FakeScorer()
    (t,) = make.targets_for([r], sc)
    assert isinstance(t, make.AnswerTarget)
    assert t.answer == "Outcome beta40 improved markedly."
    assert t.citations == (3,)  # ranked on the stripped text: the results sentence
    assert {h for _, h in sc.pairs} == {"Outcome beta40 improved markedly."}
    # support is computed on the stripped sentence: "findings suggest" would lower it
    (cand,) = make.candidates(r)
    assert cand.sentence == t.answer and cand.options[0].support == 1.0


def test_every_hedge_opener_is_stripped():
    for opener in ("Taken together, ", "In conclusion, ", "Overall, ", "Thus, ", "Therefore, ",
                   "Hence, ", "In summary, ", ""):  # fmt: skip
        for subject in ("these results", "our findings", "the data", "this study",
                        "the present analysis", "our observations"):  # fmt: skip
            s = f"{opener}{subject} indicate that risk fell."
            assert rules.strip_hedge(s[0].upper() + s[1:]) == "Risk fell.", s
    assert rules.strip_hedge("The results of our study show that risk fell.") == "Risk fell."


# ---- INS-R1 and INS-R3 ----


def test_results_words_inside_longer_labels_mark_results_sections():
    for label in ("METHODS AND RESULTS", "PRINCIPAL FINDINGS", "MAIN OUTCOME MEASURES"):
        assert rules.is_results_section(label), label
    r = row(41, sections=["BACKGROUND", "METHODS", "METHODS AND RESULTS", "RESULTS"])
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None and ex.context == tuple(r["sentences"][:2])


def test_insufficient_cites_at_most_the_two_best_question_matches():
    sents = [
        "Background on zeta in older adults with frailty.",  # 3 question words
        "Prior work on zeta was sparse.",  # 1
        "Methods enrolled older adults given zeta.",  # 3
        "Dosing of zeta was weekly.",  # 1
        "Results showed outcome omega improved.",
    ]
    r = row(
        42, long_answer="Outcome omega improved.", sentences=sents,
        sections=["BACKGROUND", "BACKGROUND", "METHODS", "METHODS", "RESULTS"],
        question="Does zeta help older adults?",
    )  # fmt: skip
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None and len(ex.context) == 4
    assert ex.target.citations == (1, 3)


# ---- TGT-R6 cap ----


def test_the_cap_counts_the_union_and_keeps_later_sentences_that_still_fit():
    sents = ["Alpha bravo.", "Charlie delta.", "Echo foxtrot.", "Golf hotel.", "India juliet."]
    la = (
        "Alpha bravo charlie delta echo foxtrot. "  # needs 1, 2, 3
        "Golf hotel india juliet. "  # needs 4, 5: would make five, dropped
        "Alpha bravo charlie delta."  # needs 1, 2: already cited, kept
    )
    r = row(43, long_answer=la, sentences=sents, sections=["RESULTS"] * 5)
    counts = Counter()
    (t,) = make.targets_for([r], AllWords(), counts)
    assert isinstance(t, make.AnswerTarget)
    assert t.citations == (1, 2, 3) and counts["sentence:citation_cap"] == 1
    assert t.answer.endswith("Alpha bravo charlie delta.")


# ---- TGT-02: stemming and stopwords pinned in full ----


def test_stopwords_suffixes_and_the_stem_boundary_are_pinned():
    # the whole set, by hash: any added, removed or swapped word changes it
    assert hashlib.sha256(" ".join(sorted(rules.STOPWORDS)).encode()).hexdigest() == (
        "d8cb322f3427f652c274709375c56c24cc9556fdd232502c9133945f578cba1d"
    )
    assert "yours" in rules.STOPWORDS and "patient" not in rules.STOPWORDS
    assert len(rules.STOPWORDS) == 95
    assert rules._SUFFIXES == (
        "ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "ly", "s",
    )  # fmt: skip
    assert rules.stem("quickly") == "quick" and rules.stem("only") == "only"  # 4 letters: kept
    assert rules.stem("relations") == "rel"  # "ations" first, stem of 3 allowed
    assert rules.stem("cats") == "cat" and rules.stem("its") == "its"  # stem of 2 refused
    assert rules.stem("dogs") == "dog"  # a 3-letter stem is kept, not 4


def test_overlap_divides_by_the_conclusion_and_ranks_without_ties():
    ctx = ("alpha.", "alpha bravo charlie delta echo foxtrot golf.", "alpha bravo.")
    scores = rules.overlaps("alpha bravo charlie", ctx)
    assert scores == [1 / 3, 1.0, 2 / 3]  # shared words over the conclusion's three
    assert rules.ranked_citations("alpha bravo charlie", ctx) == (2, 3, 1)


# ---- TGT-05: the drop report records the plan and thresholds ----


def test_drop_report_records_quotas_shortfall_thresholds_and_model(repo):  # noqa: F811
    report = make.build(repo, FakeScorer, seed=7, cap=200, chunk=100)
    saved = json.loads((repo / make.REPORT_COPY).read_text(encoding="utf-8"))
    assert saved["quotas"]["train"] == vars(make.Quota.for_size(report["train"]))
    assert saved["quotas"]["val"] == vars(make.Quota.for_size(report["val"]))
    assert saved["shortfall_vs_cap"] == 200 - report["train"] and saved["cap_train"] == 200
    assert saved["thresholds"] == {
        "support_min": 0.5, "entail_min": 0.5, "max_words": 80, "max_cites": 4,
    }  # fmt: skip
    assert saved["nli_model"] == f"{nli.MODEL_ID}@{nli.MODEL_REVISION}" and saved["seed"] == 7
    train = data.read_jsonl(repo / make.OUT_DIR / "train.jsonl")
    sample = data.read_jsonl(repo / make.AUDIT_COPY)
    assert {e["id"] for e in sample} <= {e["id"] for e in train}  # drawn from train only
    other = repo / "other"
    other.mkdir()
    for p in (repo / data.OUT_DIR).iterdir():
        (other / data.OUT_DIR).mkdir(parents=True, exist_ok=True)
        (other / data.OUT_DIR / p.name).write_bytes(p.read_bytes())
    make.build(other, FakeScorer, seed=8, cap=200, chunk=100)
    assert data.read_jsonl(other / make.AUDIT_COPY) != sample  # the seed reaches the build


def test_offscope_context_is_the_rows_full_abstract():
    r = row(44, sentences=["One a.", "Two b.", "Three c.", "Four d.", "Five e."])
    r["sections"] = ["X"] * 5
    assert make.offscope_example(r, "train", 7).context == tuple(r["sentences"])
