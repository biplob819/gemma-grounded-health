"""RLB-02: the v3b GRPO reward (OWNER D-70), shaped like the headline it is meant to move.

v3's reward (`ggh.rl.reward`) ranked v2 above base on dev while GPT-6.1 Sol's headline ranks base
far above v2 (`docs/measurements/v3/reward_alignment.json`), because it paid a full point for the
label alone and its lexical grounding check passes answers that over-claim. v3b pays mostly for
what the headline counts, a right label whose answer is supported by its cited sentences:

- `R_UNPARSED` when the reply does not parse as EvalKit's answer;
- `R_LABEL` (small) for the gold label;
- `R_GROUNDED` more when the label is right, the citations pass the structural checks of
  `structure_problem`, and the checker (`ggh.rl.checker`) finds the answer supported by the cited
  sentences, at or above `checker.THRESHOLD`;
- minus `P_AIM` when a yes/no answer cites an aim sentence (TGT-R12), as in v3.

On the 600 dev answers the checker was chosen on, this reward ranks base (0.56) above v2 (0.52)
and v1 (0.17), the headline's order; with label credit 0.5 base and v2 nearly tie, so the label
credit stays at 0.25. A wrong label earns no grounding credit.

"Maybe" answers to "maybe" questions are paid for their label and structure but are not sent to
the checker: v2's insufficient targets say what the abstract does not report, which a support
checker rejects by design (`docs/measurements/v3/gold_checker.json`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from evalkit.schemas import Answer

from ggh.rl.checker import THRESHOLD, Checker
from ggh.rl.reward import LABELS, MAX_CITES, MAX_WORDS, MIN_WORDS, P_AIM, R_UNPARSED, parse
from ggh.targets.rules import is_aim_sentence, is_excluded_from_answers

R_LABEL = 0.25
R_GROUNDED = 1.0
COPY_RATIO = 0.9  # an answer this close to one cited sentence counts as copied (a probe)


@dataclass(frozen=True)
class ScoreB:
    parsed: bool
    label_correct: bool
    grounded: bool  # structure passes and, for yes/no, the checker supports the answer
    aim_cited: bool
    copied: bool  # the answer is a near-verbatim copy of one cited sentence
    support: float | None  # the checker's mean support, None when not checked
    reward: float
    reason: str  # why the grounding credit was withheld ("" when granted or unparsed)


def structure_problem(answer: Answer, gold: str, context: Sequence[str]) -> str:
    """v3's structural checks without its lexical overlap test (the checker replaces it)."""
    cites = list(answer.citations)
    if len(set(cites)) != len(cites):
        return "repeated citation"
    if any(not 1 <= c <= len(context) for c in cites):
        return "citation out of range"
    if answer.label == "maybe" and gold != "maybe":
        return "maybe for a yes/no question"
    if not cites:
        return "no citation"
    if len(cites) > MAX_CITES:
        return f"more than {MAX_CITES} citations"
    n = len(answer.answer.split())
    if n < MIN_WORDS:
        return f"answer under {MIN_WORDS} words"
    if n > MAX_WORDS:
        return f"answer over {MAX_WORDS} words"
    if answer.label != "maybe":
        for c in cites:
            if is_excluded_from_answers(context[c - 1]):
                return f"cites aim or method sentence [{c}]"
    return ""


def evidence(answer: Answer, context: Sequence[str]) -> str:
    return " ".join(context[c - 1] for c in sorted(set(answer.citations)))


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def copied(answer: Answer, context: Sequence[str]) -> bool:
    a = _norm(answer.answer)
    return any(
        SequenceMatcher(None, a, _norm(context[c - 1])).ratio() >= COPY_RATIO
        for c in set(answer.citations)
        if 1 <= c <= len(context)
    )


def score_batch(
    texts: Sequence[str],
    golds: Sequence[str],
    contexts: Sequence[Sequence[str]],
    checker: Checker,
) -> list[ScoreB]:
    """Every reply scored; the checker is called once, only for the right-label yes/no replies
    whose structure passes (no other reply can earn the grounding credit)."""
    if not len(texts) == len(golds) == len(contexts):
        raise ValueError("one gold label and one context per reply")
    parsed = []
    for text, gold, ctx in zip(texts, golds, contexts, strict=True):
        if gold not in LABELS:
            raise ValueError(f"gold label must be one of {LABELS}, got {gold!r}")
        a = parse(text)
        problem = structure_problem(a, gold, ctx) if a is not None else ""
        parsed.append((a, problem))
    ask = [i for i, (a, p) in enumerate(parsed)
           if a is not None and a.label == golds[i] and not p and a.label != "maybe"]  # fmt: skip
    pairs = []
    for i in ask:
        a = parsed[i][0]
        assert a is not None
        pairs.append((evidence(a, contexts[i]), a.answer))
    got = dict(zip(ask, checker.support(pairs), strict=True))
    out = []
    for i, (a, problem) in enumerate(parsed):
        if a is None:
            out.append(ScoreB(False, False, False, False, False, None, R_UNPARSED, ""))
            continue
        ctx = contexts[i]
        correct = a.label == golds[i]
        in_range = [c for c in a.citations if 1 <= c <= len(ctx)]
        aim = a.label != "maybe" and any(is_aim_sentence(ctx[c - 1]) for c in in_range)
        support = got.get(i)
        if problem:
            reason = problem
        elif a.label != "maybe" and support is None:
            reason = "wrong label: not checked"
        elif support is not None and support < THRESHOLD:
            reason = f"checker support {support:.2f} under {THRESHOLD}"
        else:
            reason = ""
        grounded = not reason
        reward = R_LABEL * correct + R_GROUNDED * (correct and grounded) - P_AIM * aim
        out.append(ScoreB(True, correct, grounded, aim, copied(a, ctx), support, reward, reason))
    return out
