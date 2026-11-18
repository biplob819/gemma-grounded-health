"""TGT-R10 (OWNER D-43, VTG-01): an answer never cites an aim, objective or study-description
sentence. v1 let "To investigate whether X..." through lexical ranking and NLI at 0.5, and the
fine-tuned model learned to cite it. Insufficient and off-scope examples keep their rules."""

import hashlib
from collections import Counter

import pytest

from ggh.schemas import TrainExample
from ggh.targets import make, nli, rules
from tests.targets.test_targets import FakeScorer, pool, row

# ---- the classifier, both directions ----

AIM = (
    "To investigate whether statins reduce mortality in dialysis patients.",
    "The aim of this study was to evaluate whether vitamin D lowers blood pressure.",
    "Therefore, the purpose of the present study was to assess bone density after surgery.",
    "Our primary objective was to compare two doses of aspirin.",
    "We hypothesized that exercise would lower HbA1c by 10%.",  # a number, but a hypothesis
    "We sought to determine whether obesity is associated with higher mortality.",
    "In this prospective cohort study, we examined whether serum ferritin predicts relapse.",
    "This study was designed to test the hypothesis that salt increases blood pressure.",
    "The present study investigated the effect of caffeine on sleep in rats.",
    "Since the mechanism is unclear, we investigated the role of IL-6 in sepsis.",  # mid-clause
    "The effect of melatonin on invasion was examined in MCF-7 cells.",  # passive, no result
    "It was hypothesized that smokers would have more complications.",
    "(1) To evaluate the prevalence of atopy in aspirin-sensitive patients.",
    "We conducted this double-blind study to determine whether clonidine has analgesic effects.",
    "This double-blind, placebo-controlled study was performed in order to investigate whether "
    "steroids reduce polyp recurrence.",
    "Here, we have now evaluated whether dietary calcium decreases oxidative stress.",
    "In the present study, the authors tested the hypothesis that facilitation is increased in "
    "schizophrenia.",
    "Our hypothesis was that composites with higher shrinkage cause more strain.",
    "The question addressed in this study is whether bisphosphonates change glucose levels.",
    "This multicenter, double-blinded RCT investigated the efficacy of GnRH antagonists.",
    "The aim of the current study was to determine whether diluting lidocaine from 2.0% to 0.5% "
    "alters symptoms.",  # percentages inside an aim
    "This study aimed to evaluate surgical timing, and phases were compared to find the one "
    "which showed the best survival.",  # "which showed" is part of the aim
    # TGT-R10 v2: forms the first v2 audit found cited (OWNER D-45)
    "The study aims to investigate whether information from multimodal imaging increases "
    "ASTRAL score's accuracy.",
    "Here we determined the expression and function of proteins involved in MAPK signaling in "
    "benign, atypical, and malignant meningiomas.",
    "This article reports the importance of serum protein mediums in the performance of the "
    "manual hexadimethrine bromide test.",
    "We recently have reported the first randomized, controlled study on the effects of "
    "short-term rhGH therapy on the nutritional status of malnourished dialysis patients.",
    "A serotonin transporter promoter polymorphism has been extensively studied in neuroticism "
    "and several psychiatric disorders, with inconclusive results.",
    "Hypertension has been investigated extensively in adults.",  # earlier work, not a finding
    "As the proportion of CD39 lymphocytes is decreased in juvenile AILD, we have explored "
    "whether decreased CD39 expression is present on Th17 cells.",  # background clause first
)
NOT_AIM = (
    "We found that obesity was associated with higher mortality.",
    "Statin use was associated with a 30% reduction in mortality (p < 0.01).",
    "We examined 120 patients and found that ferritin was higher in those who relapsed.",
    "We investigated the relationship between sunlight and prostate cancer, and found a lower "
    "incidence in sunnier regions.",
    "The effect of caffeine on sleep was examined, and sleep latency was significantly longer.",
    "This study showed that caffeine shortens sleep.",
    "To date, no trial has tested this drug in children.",
    "Total mortality fell from 12 to 8 per 1000.",
    "Treatment goals were achieved in most patients.",
    "Patients were studied for a median of 5 years and 12% died.",
    "The objective response rate was 25%.",
    "These findings suggest that the aim of therapy should shift towards prevention.",
    "In this study, we have demonstrated that Notch1 mediates the effect of COX-2 inhibitors.",
    "There was no difference in pain scores between the groups.",
    "Elevated troponin was associated with right ventricular dilatation.",
    "We investigated 50 patients; troponin levels were elevated in 30 of them.",
    "The aim of this study was to evaluate X, and we found that X was reduced in treated rats.",
)


@pytest.mark.parametrize("sentence", AIM)
def test_aim_objective_and_study_description_sentences_are_classified_as_aims(sentence):
    assert rules.is_aim_sentence(sentence)


@pytest.mark.parametrize("sentence", NOT_AIM)
def test_result_and_other_sentences_are_not_aims(sentence):
    assert not rules.is_aim_sentence(sentence)


def test_reports_result_reads_effects_only_outside_a_whether_clause():
    assert not rules.reports_result("We examined whether X was associated with Y.")
    assert rules.reports_result("We examined X; it was associated with Y.")
    assert rules.reports_result("We examined whether X matters and found it does.")
    assert rules.reports_result("Mortality was 12% (p = 0.03).")


# ---- TGT-R10 v2: method and procedure sentences that report no finding ----

METHOD = (  # from the first v2 audit's answer examples, and the owner's ruling
    "Antibodies to CD14, CD19-ECD, HLA-A-B-C and anti-HLA-B27 were used to analyse peripheral "
    "blood mononuclear cells.",
    "Expression levels of ER stress markers (GRP78 and CHOP) and proinflammatory genes were "
    "assessed by qPCR.",
    "CD 34 counts (blood vessels) and VEGF expression was evaluated by means of "
    "immunohistochemistry on specimens from patients undergoing cytoreductive surgery.",
    "Blood group O red blood cells and Rh positive serum were collected at random from healthy "
    "blood donators, then tested with each other by the manual hexadimethrine bromide methods.",
    "Each patient was tested with TRH (400 microg bolus i.v.) on two separate occasions, just "
    "before and immediately after the treatment period.",
    "After tracheal intubation, anesthesia was maintained with an end-tidal isoflurane "
    "concentration of 1.0% for 20 min.",  # a dose, not a finding
    "In 181 pediatric patients, PA diagnosis was founded on medical history, skin prick testing "
    "and oral food challenge.",
    "The effect of MAPK and PI3K pathway inhibition on cell proliferation and apoptosis was "
    "determined using a primary malignant meningioma cell culture.",
    "GH, PRL and TSH responses to TRH before and 1 month after rhGH therapy were evaluated.",
    "We measured serum CML and blood pressure in 3044 adults aged 70-79 years.",
)
NOT_METHOD = (  # results, even when they name a method
    "Atypical and malignant meningiomas showed higher levels of phospho-Akt compared with "
    "benign tumors.",
    "rhGH treatment did not modify baseline serum GH concentrations (6.6 vs 4.1 microg/l).",
    "Immunohistochemical staining demonstrated that TnC expression began in adventitial "
    "myofibroblasts three days after injury.",
    "When tranexamic acid was administered, the need for transfusions was lower.",
    "Expression of phospho-MAPK was decreased in aggressive meningiomas, as assessed by "
    "western blot.",
    "Patients treated with X were followed for 5 years; mortality was 12% (p = 0.03).",
    "No significant difference was found when the groups were compared.",
    "Ara h 6 was the best predictor of PA.",
    "Forty-five patients after TKR receive treatment with tranexamic acid.",
)


@pytest.mark.parametrize("sentence", METHOD)
def test_method_sentences_are_excluded_from_answers(sentence):
    assert rules.is_method_sentence(sentence) and rules.is_excluded_from_answers(sentence)


@pytest.mark.parametrize("sentence", NOT_METHOD)
def test_result_sentences_naming_a_method_stay_citable(sentence):
    assert not rules.is_method_sentence(sentence)
    assert not rules.is_excluded_from_answers(sentence)


def test_the_patterns_are_pinned():
    # the whole pattern set, by hash: changing a pattern changes the targets (§4)
    guards = (rules._FOUND.pattern, rules._STATED.pattern, rules._OPEN_CLAUSE.pattern)
    method = (*rules.METHOD_PATTERNS, rules._FINDING.pattern)
    text = "\n".join(
        (*rules.AIM_PATTERNS, "--", *rules.DESCRIPTION_PATTERNS, "--", *guards, "--", *method)
    )
    assert hashlib.sha256(text.encode()).hexdigest() == AIM_PATTERNS_SHA256
    assert (len(rules.AIM_PATTERNS), len(rules.DESCRIPTION_PATTERNS)) == (10, 5)
    assert len(rules.METHOD_PATTERNS) == 3


AIM_PATTERNS_SHA256 = "75ecfc463739ecad1416e4943d437f64db93440e403e3be986f2b1818aad05bd"

# ---- the pipeline: answers only ----

AIM_FIRST = [
    "To determine whether treatment alpha1 improves outcome beta1 in adults.",  # aim
    "Methods enrolled 80 adults receiving treatment alpha1.",
    "Outcome beta1 improved markedly with treatment alpha1.",  # the result
    "Secondary measure gamma1 was unchanged.",
]


def test_an_answer_never_cites_an_aim_sentence_even_when_it_overlaps_best():
    r = row(1, long_answer="Treatment alpha1 improves outcome beta1 in adults.",
            sentences=AIM_FIRST)  # fmt: skip
    ctx = tuple(r["sentences"])
    assert rules.aim_positions(ctx) == {1}
    assert rules.ranked_citations(r["long_answer"], ctx)[0] == 1  # v1 ranks the aim first
    counts = Counter()
    (t,) = make.targets_for([r], FakeScorer(), counts, version="v2")
    assert isinstance(t, make.AnswerTarget) and 1 not in t.citations and 3 in t.citations
    assert counts["ranking:aim_or_method_removed"] == 1
    (t22,) = make.targets_for([r], FakeScorer())  # v2.2: the aim is never even scored
    assert isinstance(t22, make.AnswerTarget) and 1 not in t22.citations
    (v1,) = make.targets_for([r], FakeScorer(), version="v1")
    assert isinstance(v1, make.AnswerTarget) and v1.citations[0] == 1  # what v1 trained on


def test_a_sentence_supported_only_by_an_aim_is_dropped_and_counted():
    sents = ["To determine whether zeta2 lowers omega2.", "Patients were enrolled.",
             "Nothing else here."]  # fmt: skip
    r = row(2, long_answer="Zeta2 lowers omega2.", sentences=sents, sections=["X"] * 3)
    counts = Counter()
    (t,) = make.targets_for([r], FakeScorer(), counts, version="v2")
    assert t == "no_supported_sentence"
    assert (
        counts["sentence:aim_or_method_only_overlap"] == 1 and counts["sentence:zero_overlap"] == 0
    )
    counts = Counter()  # v2.2: the aim is never scored; what is left shares no word (TGT-R3)
    (t,) = make.targets_for([r], FakeScorer(), counts)
    assert t == "no_supported_sentence" and counts["sentence:low_support"] == 1


def test_insufficient_examples_still_cite_the_question_match_even_an_aim():
    sections = ["OBJECTIVE", "METHODS", "RESULTS", "RESULTS"]
    r = row(3, sentences=["To determine whether treatment alpha3 improves outcome beta3.",
                          "Patients were enrolled at two sites.",
                          "Outcome beta3 improved markedly in treated patients.",
                          "Secondary measure gamma3 was unchanged."],
            sections=sections)  # fmt: skip
    (t,) = make.targets_for([r], FakeScorer())
    assert isinstance(t, make.AnswerTarget) and t.citations == (3,)
    ex = make.insufficient_example(r, t, "train", 7)
    assert ex is not None and ex.context == tuple(r["sentences"][:2])
    assert ex.target.citations == (1,)  # INS-R3 unchanged: the aim matches the question
    assert ex.rule_ids == rules.INSUFFICIENT_RULES and "TGT-R10" in ex.rule_ids


def test_offscope_examples_keep_citing_sentence_one():
    r = row(4, sentences=AIM_FIRST)
    ex = make.offscope_example(r, "train", 7)
    assert ex.target.citations == (1,) and ex.rule_ids == ("OFF-R3",)


# ---- provenance: v2 ids and v1 ids both rebuild ----


def test_rule_ids_carry_tgt_r10_and_v1_ids_stay_known():
    assert "TGT-R10" in rules.ANSWER_RULES and "TGT-R11" in rules.ANSWER_RULES
    assert rules.V2_ANSWER_RULES == (
        "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R10", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7",
    )  # fmt: skip
    assert rules.V1_ANSWER_RULES == (
        "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7",
    )  # fmt: skip
    assert (*rules.V1_ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3") == rules.V1_INSUFFICIENT_RULES


def test_v1_rule_ids_rebuild_the_v1_target_and_v2_ids_the_v2_target():
    r = row(5, long_answer="Treatment alpha1 improves outcome beta1 in adults.",
            sentences=AIM_FIRST)  # fmt: skip
    rec = nli.RecordingScorer(FakeScorer(), rules.SCORE_DECIMALS)
    v2 = make.reconstruct(r, rules.V2_ANSWER_RULES, split="train", seed=7, scorer=rec)
    v1 = make.reconstruct(r, rules.V1_ANSWER_RULES, split="train", seed=7, scorer=rec)
    v22 = make.reconstruct(r, rules.ANSWER_RULES, split="train", seed=7, scorer=rec)
    assert 1 not in v2.target.citations and 1 in v1.target.citations
    assert 1 not in v22.target.citations
    assert v1.rule_ids == rules.V1_ANSWER_RULES and v2.rule_ids == rules.V2_ANSWER_RULES
    assert v22.rule_ids == rules.ANSWER_RULES


def aim_pool(n_yes, n_no, start=0):
    """Rows whose first sentence is an aim that overlaps the conclusion best."""
    out = pool(n_yes, n_no, start)
    for r in out:
        i = int(r["pmid"]) - 1000
        r["sentences"] = [f"To determine whether alpha{i} improves outcome beta{i}.",
                          *r["sentences"][1:]]  # fmt: skip
    return out


def test_selected_answers_cite_no_aim_and_every_example_reconstructs():
    val_rows, pool_rows = aim_pool(30, 4), aim_pool(300, 40, start=100)
    rec = nli.RecordingScorer(FakeScorer(), rules.SCORE_DECIMALS)
    sel = make.select(val_rows, pool_rows, rec, seed=7)
    by_pmid = {r["pmid"]: r for r in val_rows + pool_rows}
    replay = nli.ReplayScorer(rec.table)
    answers = [e for e in sel.train + sel.val if e.kind == "answer"]
    assert answers
    for e in answers:
        assert not any(rules.is_aim_sentence(e.context[i - 1]) for i in e.target.citations)
    for e in sel.train + sel.val:
        again = make.reconstruct(by_pmid[e.pmid], e.rule_ids, split=e.split, seed=7, scorer=replay)
        assert again == e, e.id
        TrainExample.model_validate_json(e.model_dump_json())
    assert sum(sel.report["sentence_drops"].values()) > 0
