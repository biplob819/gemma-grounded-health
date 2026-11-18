"""Pinned target rules (Tech Spec §4.2–4.5, OWNER D-15, D-16, D-34, D-43).

Every constant here is part of the training data's definition: changing one changes the
targets, so each is pinned by a test and changed only through a decision entry (§4).
"""

from __future__ import annotations

import hashlib
import re

from evalkit.text import split_sentences

# ---- rule ids stamped into TrainExample.rule_ids (§4) ----
# v2.2 (OWNER D-48, D-49): TGT-R12 (the tightened aim and method exclusion) and TGT-R11
# (citations by single-sentence entailment, replacing TGT-R6's word-overlap ranking; its cap
# of 4 citations is kept, as MAX_CITES)
ANSWER_RULES = (
    "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R12", "TGT-R11", "TGT-R3", "TGT-R4", "TGT-R5", "TGT-R7",
)  # fmt: skip
INSUFFICIENT_RULES = (*ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3")
# earlier builds' ids; `reconstruct` still rebuilds each exactly (PRV-R1)
V21_ANSWER_RULES = (  # v2.1: TGT-R12 with TGT-R6
    "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R12", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7",
)  # fmt: skip
V21_INSUFFICIENT_RULES = (*V21_ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3")
V2_ANSWER_RULES = (  # the first v2 build: TGT-R10 as first written (rules_r10.py)
    "TGT-R1", "TGT-R2", "TGT-R9", "TGT-R10", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7",
)  # fmt: skip
V2_INSUFFICIENT_RULES = (*V2_ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3")
V1_ANSWER_RULES = tuple(r for r in V2_ANSWER_RULES if r != "TGT-R10")
V1_INSUFFICIENT_RULES = (*V1_ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3")


def rules_version(rule_ids: tuple[str, ...]) -> str:
    """Which build's citation rules a set of answer or insufficient rule ids names."""
    if "TGT-R11" in rule_ids:
        return "v2.2"
    if "TGT-R12" in rule_ids:
        return "v2.1"
    return "v2" if "TGT-R10" in rule_ids else "v1"


OFFSCOPE_RULES = ("OFF-R3",)

# ---- thresholds (TGT-R8: revised at most once, after the hand audit, by a decision) ----
SUPPORT_MIN = 0.5  # TGT-R3
ENTAIL_MIN = 0.5  # TGT-R4
MAX_WORDS = 80  # TGT-R5
MAX_CITES = 4  # TGT-R6, and TGT-R11's cap
SENTENCE_FLOOR = 0.3  # TGT-R11: each cited sentence alone entails the claim at least this much
SCORE_DECIMALS = 6  # NLI scores are rounded before comparison, so a replay decides identically

# ---- TGT-R2: recommendation and implication sentences are dropped ----
RECOMMEND_PATTERNS = (  # regexes, matched case-insensitively from a word start
    r"should\b",  # not "shoulder"
    r"recommend",  # recommended, recommendation
    r"must\b",  # not "mustard"
    r"we suggest",
    r"clinicians?\b",
    r"further stud",  # study, studies
    r"warrant(?!y)",  # warrants, warranted; not "warranty"
    r"need to\b",
    r"advis",  # advise, advisable
)
_RECOMMEND = re.compile(r"\b(?:" + "|".join(RECOMMEND_PATTERNS) + ")", re.I)


def is_recommendation(sentence: str) -> bool:
    return _RECOMMEND.search(sentence) is not None


# ---- TGT-R9: a leading hedge is not a claim (OWNER D-34) ----
_HEDGE = re.compile(
    r"^(?:(?:taken together|in conclusion|overall|thus|therefore|hence|in summary),?\s+)?"
    r"(?:(?:the|these|our|this|present)\s+)*"
    r"(?:results?|findings?|data|study|analysis|observations?)\s+"
    r"(?:of\s+(?:our|this|the\s+present)\s+(?:study|analysis)\s+)?"
    r"(?:suggests?|indicates?|shows?|showed|demonstrates?|demonstrated|confirms?|supports?|"
    r"implies|imply|reveals?)\s+(?:that\s+)?",
    re.I,
)


def strip_hedge(sentence: str) -> str:
    rest = _HEDGE.sub("", sentence, count=1)
    if rest == sentence or not rest:
        return sentence
    return rest[:1].upper() + rest[1:]


# ---- TGT-R12 (TGT-R10 tightened; the first TGT-R10 is frozen in rules_r10.py): an answer
# never cites an aim, objective, study-description or method sentence
# (OWNER D-43; v2 of the rule after the owner's per-sentence audit, D-45). v1 let "To
# investigate whether X..." through lexical ranking and NLI at 0.5, so 43% of its answer
# targets cited one; the first v2 audit then found method sentences ("X was assessed by qPCR")
# cited too. The classifier is deliberately conservative: aim phrasing always counts; study
# descriptions ("We examined X", "X was investigated", "This article reports") and method or
# procedure sentences ("X were used to analyse", "Each patient was tested with") count only
# when the sentence reports no result. Insufficient and off-scope examples are untouched: their
# citations come from INS-R3 and OFF-R3, not from TGT-R6.
_LEAD = (  # sentence start, with an optional opener before the aim phrasing
    r"^\W*(?:\d+\)\s*)?"  # "(1) To evaluate..."
    r"(?:(?:therefore|thus|hence|accordingly|consequently|here(?:in)?|now|so),?\s+)?"
    r"(?:(?:in|for)\s+(?:this|the\s+(?:present|current)|our)\s+(?:[\w-]+,?\s+){0,3}?"
    r"(?:study|analysis|report|paper|work|trial|investigation|review|survey|article),?\s+)?"
    r"(?:here(?:\s*in)?,?\s+)?"
)
_CLAUSE = (  # sentence start, or a clause start ("Since X is unclear, we investigated...")
    r"(?:" + _LEAD + r"|[,;:]\s+(?:(?:therefore|thus|hence|here(?:in)?|so),?\s+)?)"
)
_STUDY = (  # "this study", "the study", "the present prospective cohort study", "this RCT"
    r"(?:this|the(?:\s+(?:present|current))?|our)\s+(?:[\w-]+,?\s+){0,3}?"
    r"(?:study|analysis|report|paper|article|work|trial|rct|investigation|review|project|"
    r"survey|experiment|research)"
)
_ADV = (  # "we have now examined", "we therefore aimed"
    r"(?:(?:have|has|therefore|thus|also|further|then|next|now|here|first|specifically|"
    r"recently|previously|retrospectively|prospectively)\s+){0,2}"
)
_WE = r"(?:we|the\s+authors)\s+"
AIM_PATTERNS = (  # aim phrasing: an aim unless the sentence also says what was found
    _LEAD + r"(?:in\s+order\s+)?to\s+(?:investigate|determine|evaluate|assess|examine|"
    r"explore|compare|study|test|identify|analy[sz]e|describe|estimate|establish|"
    r"characteri[sz]e|clarify|elucidate|measure|quantify|ascertain|verify|validate|define|"
    r"understand|address|confirm|document|report|review|present)\b",
    _CLAUSE + r"(?:(?:the|our|a|one|another)\s+)?(?:(?:primary|main|secondary|first|second|"
    r"overall|specific|principal|major|study|research|present|current|ultimate|key)\s+){0,2}"
    r"(?:aims?|purposes?|objectives?|goals?|intent|intention)\s+(?:of|was|were|is|are)\b",
    r"\b(?:aims?|purposes?|objectives?|goals?)\s+of\s+" + _STUDY + r"\b",
    r"\b(?:it\s+(?:is|was)|we|authors)\s+(?:\w+\s+)?hypothesi[sz]ed\b",
    r"\btest(?:ed|s)?\s+(?:the|our|a|this|these|two)\s+(?:\w+\s+)?hypothes[ie]s\b",
    r"\b(?:our|the|a)\s+(?:\w+\s+)?hypothesis\s+(?:is|was)\s+that\b",
    r"\bquestions?\s+addressed\b",
    _CLAUSE + _WE + _ADV + r"(?:conducted|undertook|performed|did|designed|carried\s+out|"
    r"initiated)\s+(?:this|the\s+present|the\s+current|a|an)\s+(?:[\w-]+,?\s+){0,4}?"
    r"(?:study|analysis|trial|survey|review|investigation|experiment)\s+(?:in\s+order\s+)?to\b",
    _CLAUSE + _WE + _ADV + r"(?:hypothesi[sz]e|aimed|aim|sought|seek|set\s+out|wished|"
    r"wanted|intended)\b",
    _CLAUSE + _STUDY + r"\s+(?:was\s+|is\s+)?(?:aimed|aims|sought|seeks|designed|"
    r"undertaken|initiated|intends|intended|attempts|attempted|set\s+out|"
    r"focus(?:es|ed)\s+on|to\s+\w+|(?:conducted|performed|carried\s+out|done)\s+"
    r"(?:in\s+order\s+)?to)\b",
)
DESCRIPTION_PATTERNS = (  # study description: an aim only when no result is reported
    _CLAUSE + _WE + _ADV + r"(?:investigated?|examined?|evaluated?|assessed?|studied|"
    r"study|explored?|tested?|determined?|reported|described|published)\b",
    _CLAUSE + _STUDY + r"\s+(?:investigated?|investigates|examined?|examines|evaluated?|"
    r"evaluates|assessed?|assesses|explored?|explores|describes|described|presents|"
    r"compared?|compares|reports|reported|determined|determines)\b",
    r"\b(?:has|have)\s+been\s+(?:\w+ly\s+)?(?:studied|investigated|examined|explored|"
    r"evaluated|assessed)\b",  # earlier work: not this study's finding
    r"\b(?:was|were)\s+(?:\w+ly\s+)?(?:studied|investigated|examined|explored)\b",
    r"\b(?:study|trial|analysis|investigation)\s+(?:was\s+)?(?:(?:conducted|performed|"
    r"undertaken|designed|carried\s+out)\s+)?(?:in\s+order\s+)?to\s+(?:investigate|determine|"
    r"evaluate|assess|examine|explore|test|compare|clarify|elucidate|identify)\b",
)
_FOUND = re.compile(  # a result reported in so many words ("we found", "results showed")
    r"\b(?:we|and|but|they|authors|it|was|were|is|are|has|have|had)\s+(?:also\s+|further\s+|"
    r"then\s+|not\s+)?(?:found|showed|shown|revealed|demonstrated|observed)\b|"
    r"\b(?:results?|findings?|data|analys[ie]s)\s+(?:\w+\s+)?(?:showed|revealed|"
    r"demonstrated|indicated|suggested)\b|"
    r"\b(?:showed|revealed|demonstrated|confirmed)\s+(?:that|a|an|no|significant)\b",
    re.I,
)
_STATED = re.compile(  # a result stated as fact; read outside any "whether"/"if" clause
    r"\bp\s*[<>=\u2264\u2265]|\d\s*%|\b(?:odds|hazard|risk)\s+ratios?\b|"
    r"\b(?:is|are|was|were|be|been)\s+(?:not\s+|significantly\s+|strongly\s+|"
    r"independently\s+|positively\s+|negatively\s+|inversely\s+)*(?:associated|correlated|"
    r"related|linked|higher|lower|greater|smaller|increased|decreased|reduced|elevated|"
    r"improved|unchanged|similar|different|comparable|superior|inferior|effective|"
    r"predictive)\b|\bsignificant(?:ly)?\b",
    re.I,
)
_OPEN_CLAUSE = re.compile(r"\b(?:whether|if)\b.*", re.I | re.S)
_AIM = re.compile("|".join(f"(?:{p})" for p in AIM_PATTERNS), re.I)
_DESCRIPTION = re.compile("|".join(f"(?:{p})" for p in DESCRIPTION_PATTERNS), re.I)

_DONE = (  # what a method sentence says was done to, or with, its subject
    r"used|performed|measured|assessed|evaluated|determined|analy[sz]ed|collected|recorded|"
    r"obtained|tested|administered|given|randomi[sz]ed|enrolled|included|recruited|monitored|"
    r"calculated|estimated|quantified|compared|treated|divided|allocated|assigned|maintained|"
    r"scored|reviewed|selected|genotyped|sequenced|stained|injected|implanted|exposed|"
    r"interviewed|surveyed|applied|conducted|carried\s+out|undertaken|subjected|sampled|"
    r"harvested|prepared|incubated|cultured|transfected|imaged|scanned|screened|based|founded|"
    r"defined|categori[sz]ed|classified|matched|followed(?:\s+up)?|retrieved|extracted|"
    r"identified|evaluated|investigated|studied|examined"
)
METHOD_PATTERNS = (  # a method or procedure: excluded only when no result is reported
    r"\b(?:was|were|is|are|been|being)\s+(?:\w+ly\s+)?(?:" + _DONE + r")\b",
    _CLAUSE + _WE + _ADV + r"(?:used|measured|performed|conducted|analy[sz]ed|recruited|"
    r"enrolled|collected|reviewed|compared|calculated|obtained|genotyped|followed|included|"
    r"selected|identified|applied|developed|constructed|designed|administered|randomi[sz]ed|"
    r"retrieved|extracted|searched)\b",
    r"\b(?:using|by\s+means\s+of)\b.*\b(?:was|were)\s+(?:\w+ly\s+)?(?:" + _DONE + r")\b",
)
_METHOD = re.compile("|".join(f"(?:{p})" for p in METHOD_PATTERNS), re.I)
_FINDING = re.compile(  # a finding, for method sentences (a bare dose "1.0%" is not one)
    r"\bp\s*[<>=\u2264\u2265]|\b(?:odds|hazard|risk)\s+ratios?\b|\bci\b|"
    r"\b(?:did|does|do|was|were|is|are|could|can)\s+not\b|\bno\s+(?:\w+\s+)?(?:difference|"
    r"change|effect|association|correlation|relationship)s?\b|"
    r"\b(?:increased|decreased|reduced|improved|declined|rose|fell|lowered|raised|predicted|"
    r"correlated|associated|higher|lower|greater|fewer|smaller|larger|elevated|diminished|"
    r"inhibited|induced|abolished|attenuated|enhanced|prevented|caused|resulted|replicated)\b|"
    r"\b(?:significant(?:ly)?|best|worse|better|more|less|associations?|increases?|"
    r"decreases?|reductions?|elevations?)\b|"
    r"\(\s*\d+(?:\.\d+)?\s*%|\d\s*%\s+(?:of|were|was|had|in)\b",  # "127 cases (73.4%)"
    re.I,
)


def reports_result(sentence: str) -> bool:
    """A result in so many words ("we found"), or one stated as fact outside a "whether" or
    "if" clause ("X was associated with Y", "p < 0.05", "12%")."""
    return bool(_FOUND.search(sentence) or _STATED.search(_OPEN_CLAUSE.sub("", sentence)))


def is_aim_sentence(sentence: str) -> bool:
    """TGT-R10: an aim, objective, hypothesis or study-description sentence."""
    if _AIM.search(sentence):  # aim wording: only a result in so many words overrides it
        return _FOUND.search(sentence) is None
    m = _DESCRIPTION.search(sentence)
    if m:  # a leading background clause ("As X is decreased in Y, we explored...") is not read
        return not reports_result(sentence[m.start() :])
    return False


def is_method_sentence(sentence: str) -> bool:
    """TGT-R10 (v2): a method or procedure sentence that reports no finding."""
    if not _METHOD.search(sentence):
        return False
    return not (_FOUND.search(sentence) or _FINDING.search(_OPEN_CLAUSE.sub("", sentence)))


def is_result_like(sentence: str) -> bool:
    """TGT-R11: the sentence reports a result (the TGT-R10 result and finding detectors)."""
    return reports_result(sentence) or _FINDING.search(_OPEN_CLAUSE.sub("", sentence)) is not None


def is_excluded_from_answers(sentence: str) -> bool:
    """TGT-R10: an answer never cites this sentence."""
    return is_aim_sentence(sentence) or is_method_sentence(sentence)


# ---- TGT-R3 / TGT-R6: content words ----
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "among",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "does",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "nor",
        "not",
        "of",
        "on",
        "onto",
        "or",
        "other",
        "our",
        "ours",
        "over",
        "shall",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "too",
        "under",
        "upon",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yours",
    ]
)
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "ly", "s")
_WORD3 = re.compile(r"\w{3,}")


def stem(word: str) -> str:
    """Light suffix stripping: the first matching suffix, keeping a stem of at least 3."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def content_words(text: str) -> frozenset[str]:
    return frozenset(stem(w) for w in _WORD3.findall(text.lower()) if w not in STOPWORDS)


def support(sentence: str, cited: list[str]) -> float:
    """TGT-R3: share of the sentence's content words found in the cited sentences."""
    words = content_words(sentence)
    if not words:
        return 0.0
    pool: set[str] = set()
    for c in cited:
        pool |= content_words(c)
    return len(words & pool) / len(words)


def overlaps(sentence: str, context: tuple[str, ...]) -> list[float]:
    words = content_words(sentence)
    if not words:
        return [0.0] * len(context)
    return [len(words & content_words(c)) / len(words) for c in context]


def aim_positions(context: tuple[str, ...], first: bool = False) -> frozenset[int]:
    """TGT-R12 (or, with `first`, TGT-R10 as first written): the 1-based positions an answer
    may not cite."""
    if first:
        from ggh.targets import rules_r10

        return frozenset(i + 1 for i, c in enumerate(context) if rules_r10.is_aim_sentence(c))
    return frozenset(i + 1 for i, c in enumerate(context) if is_excluded_from_answers(c))


def ranked_citations(
    sentence: str, context: tuple[str, ...], exclude: frozenset[int] = frozenset()
) -> tuple[int, ...]:
    """The up-to-4 context sentences (1-based) with the highest non-zero overlap, best first,
    ties broken by position, never one in `exclude` (TGT-R10). Empty when nothing overlaps
    (E1: never cite at random)."""
    scores = overlaps(sentence, context)
    ranked = sorted(
        (i for i, s in enumerate(scores) if s > 0 and i + 1 not in exclude),
        key=lambda i: (-scores[i], i),
    )
    return tuple(i + 1 for i in ranked[:MAX_CITES])


def support_ranking(
    sentence: str, context: tuple[str, ...], entail: dict[int, float], exclude: frozenset[int]
) -> tuple[int, ...]:
    """TGT-R11 (OWNER D-48): the up-to-4 context sentences (1-based) that may support
    `sentence`, best first. `entail[i]` is sentence i's entailment of the claim on its own. A
    candidate is not excluded by TGT-R10, entails at least SENTENCE_FLOOR alone, and either
    reports a result or entails at least ENTAIL_MIN alone. Ranked by that entailment; ties by
    word overlap, then position."""
    lex = overlaps(sentence, context)
    cands = [
        i
        for i, e in entail.items()
        if i not in exclude
        and e >= SENTENCE_FLOOR
        and (e >= ENTAIL_MIN or is_result_like(context[i - 1]))
    ]
    return tuple(sorted(cands, key=lambda i: (-entail[i], -lex[i - 1], i))[:MAX_CITES])


def citation_options(ranked: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """TGT-R6 (D-35): the top 1, top 2, ... top 4 of the ranking, each in ascending order. The
    smallest option that passes TGT-R3 and TGT-R4 is cited, so citations stay minimal."""
    return tuple(tuple(sorted(ranked[:k])) for k in range(1, len(ranked) + 1))


def conclusion_sentences(long_answer: str) -> tuple[str, ...]:
    """TGT-R1: EvalKit's own splitter."""
    return tuple(split_sentences(long_answer))


# ---- INS-R1 to INS-R3 (OWNER D-15) ----
_RESULTS_WORDS = ("RESULT", "FINDING", "CONCLUSION", "OUTCOME")
UNKNOWN_SECTIONS = frozenset({"UNLABELLED", "UNASSIGNED"})


def is_results_section(label: str) -> bool:
    return any(w in label.upper() for w in _RESULTS_WORDS)


INSUFFICIENT_ANSWERS = (
    "The sentences describe the study's aim and methods but report no result that settles "
    "this question.",
    "The abstract as given sets out what was studied and how, but none of its sentences "
    "reports an outcome that answers this.",
    "These sentences cover the background and design of the study only; without its results "
    "the question cannot be answered from them.",
    "No sentence here reports a finding on this question; the text describes the methods, "
    "not what the study found.",
)

# ---- OFF-R3 (D-16): written separately from data/offscope_eval.jsonl; a test checks that no
# 5-word sequence of these templates occurs in the evaluation prompts ----
OFFSCOPE_QUESTIONS = (
    "Given my own health history, is this something I personally ought to start?",
    "My doctor mentioned this during a visit last month. Would it be safe for me?",
    "Can you tell me what dose I personally ought to be taking of this?",
    "I am pregnant at the moment. Is it fine for me to go ahead with this?",
    "My elderly father has this condition. What treatment would you pick for him?",
    "Is it okay for me to stop my current medication and try this approach instead?",
    "I was diagnosed recently. Which option here is the right one for me?",
    "How many weeks of this would I personally need before seeing a benefit?",
    "Would you advise me to ask my surgeon for this procedure next week?",
    "My child has symptoms like the ones described. What do you think we ought to do?",
    "I already take several pills each day. Can I safely add this one to them?",
    "Based on this research, should I change the treatment plan my specialist gave me?",
)
OFFSCOPE_ANSWERS = (
    "I can't advise on your personal treatment or dosing. I can only report what this "
    "abstract says about the study it describes; please discuss your own situation with "
    "your doctor.",
    "That is a question about your own care, which this abstract cannot answer. It reports "
    "a study's findings, not advice for an individual; a clinician who knows your history "
    "can help.",
    "I'm not able to recommend a treatment or dose for you. The abstract only describes what "
    "a study found, and your doctor or pharmacist is the right person to ask.",
    "Personal medical decisions are outside what I can help with here. The text summarises "
    "research results, which are not a substitute for advice from your own clinician.",
    "I can't tell you what to do for your own health. The abstract describes a study and its "
    "results; please talk to a healthcare professional about your case.",
    "This asks for individual advice, which the abstract cannot give. It reports on a group "
    "of patients in a study; your doctor can say whether any of it applies to you.",
)


def pick(seed: int, pmid: str, salt: str, n: int) -> int:
    """A seeded, order-independent choice for one row (stable across processes)."""
    digest = hashlib.sha256(f"{seed}:{salt}:{pmid}".encode()).hexdigest()
    return int(digest, 16) % n
