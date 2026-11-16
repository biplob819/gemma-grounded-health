"""Pinned target rules (Tech Spec §4.2–4.5, OWNER D-15, D-16, D-34).

Every constant here is part of the training data's definition: changing one changes the
targets, so each is pinned by a test and changed only through a decision entry (§4).
"""

from __future__ import annotations

import hashlib
import re

from evalkit.text import split_sentences

# ---- rule ids stamped into TrainExample.rule_ids (§4) ----
ANSWER_RULES = ("TGT-R1", "TGT-R2", "TGT-R9", "TGT-R3", "TGT-R6", "TGT-R4", "TGT-R5", "TGT-R7")
INSUFFICIENT_RULES = (*ANSWER_RULES, "INS-R1", "INS-R2", "INS-R3")
OFFSCOPE_RULES = ("OFF-R3",)

# ---- thresholds (TGT-R8: revised at most once, after the hand audit, by a decision) ----
SUPPORT_MIN = 0.5  # TGT-R3
ENTAIL_MIN = 0.5  # TGT-R4
MAX_WORDS = 80  # TGT-R5
MAX_CITES = 4  # TGT-R6
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


def ranked_citations(sentence: str, context: tuple[str, ...]) -> tuple[int, ...]:
    """The up-to-4 context sentences (1-based) with the highest non-zero overlap, best first,
    ties broken by position. Empty when nothing overlaps (E1: never cite at random)."""
    scores = overlaps(sentence, context)
    ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
    return tuple(i + 1 for i in ranked[:MAX_CITES])


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
