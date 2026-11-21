"""RLR-01: the v3 GRPO reward (OWNER D-66). Programmatic and deterministic: no judge or model
call in the training loop.

One completion is scored against its example's gold label and its numbered abstract sentences:

- it must parse as EvalKit's answer (`parse_answer`, the same parser the benchmark uses);
  otherwise the reward is `R_UNPARSED`;
- `R_LABEL` when the label equals the gold label, plus `R_GROUNDED` when the label is right AND
  the citations are grounded (defined in `grounding_problem`); a wrong label earns nothing for
  its citations, so the expected reward of answering a label is P(label) x (1 + 0.5 x grounded)
  and the best reply is the most likely label: there is no premium for yes/no over "maybe";
- minus `P_AIM` when a yes/no answer cites an aim, objective or study-description sentence
  (`rules.is_aim_sentence`, the tightened aim rule of TGT-R12; the v1 failure). "Maybe"
  answers are exempt: v2's insufficient targets cite the design sentences on purpose.

Grounded is stricter than "every index is in range", because the planted hacks (citing valid but
unrelated sentences, citing many sentences, a few words or none) pass a range check. Citations
must be 1 to `MAX_CITES` distinct indices in range and the answer `MIN_WORDS` to `MAX_WORDS` words
long, for every label. A yes/no answer must also cite no aim or method sentence (TGT-R12's
exclusion) and share at least `MIN_SHARED_WORDS` content words with each cited sentence. A
"maybe" answer is grounded only when the gold label is "maybe". The text check is lexical, not a
support judgement: word salad built from a cited sentence's words passes it (known issue V3.1).
A first draft that also required each cited sentence to report a result was dropped before the
plan was hashed: it withheld the credit from 459 of v2's 2,423 gold answers, whose cited
sentences entail them.

Every constant is pinned by a test and pre-registered with the v3 plan; changing one changes what
v3 was trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evalkit.pipeline.answer import parse_answer
from evalkit.schemas import Answer

from ggh.targets.rules import content_words, is_aim_sentence, is_excluded_from_answers

LABELS = ("yes", "no", "maybe")
R_UNPARSED = -1.0
R_LABEL = 1.0
R_GROUNDED = 0.5
P_AIM = 0.5
MAX_CITES = 3  # the v2 targets: 2,421 of 2,423 answers cite 1 to 3 sentences
MIN_SHARED_WORDS = 2
MIN_WORDS = 6
MAX_WORDS = 80  # the §5.2 length bar


@dataclass(frozen=True)
class Score:
    parsed: bool
    label_correct: bool
    grounded: bool  # the citations pass the grounding check (paid only with the right label)
    aim_cited: bool
    reward: float
    reason: str  # why the grounding credit was withheld ("" when granted or unparsed)


def parse(text: str) -> Answer | None:
    try:
        return parse_answer(text, LABELS)
    except Exception:  # EvalKit's SchemaError, or anything else wrong with the text
        return None


def _words(text: str) -> int:
    return len(text.split())


def grounding_problem(answer: Answer, gold: str, context: Sequence[str]) -> str:
    """Why `answer` is not grounded, or "" when it is."""
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
    n = _words(answer.answer)
    if n < MIN_WORDS:
        return f"answer under {MIN_WORDS} words"
    if n > MAX_WORDS:
        return f"answer over {MAX_WORDS} words"
    if answer.label == "maybe":
        return ""
    words = content_words(answer.answer)
    for c in cites:
        sentence = context[c - 1]
        if is_excluded_from_answers(sentence):
            return f"cites aim or method sentence [{c}]"
        if len(words & content_words(sentence)) < MIN_SHARED_WORDS:
            return f"cites sentence [{c}], which shares under {MIN_SHARED_WORDS} words"
    return ""


def score(text: str, gold: str, context: Sequence[str]) -> Score:
    if gold not in LABELS:
        raise ValueError(f"gold label must be one of {LABELS}, got {gold!r}")
    answer = parse(text)
    if answer is None:
        return Score(False, False, False, False, R_UNPARSED, "")
    correct = answer.label == gold
    problem = grounding_problem(answer, gold, context)
    in_range = [c for c in answer.citations if 1 <= c <= len(context)]
    aim = answer.label != "maybe" and any(is_aim_sentence(context[c - 1]) for c in in_range)
    reward = R_LABEL * correct + R_GROUNDED * (correct and not problem) - P_AIM * aim
    return Score(True, correct, not problem, aim, reward, problem)


def _text(completion: Any) -> str:
    """TRL passes a string, or for chat prompts a list of messages ending with the reply."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def grpo_reward(
    completions: Sequence[Any],
    gold_label: Sequence[str],
    context: Sequence[Sequence[str]],
    **_: Any,
) -> list[float]:
    """TRL `GRPOTrainer` reward function: dataset columns arrive as keyword lists."""
    return [
        score(_text(c), g, ctx).reward
        for c, g, ctx in zip(completions, gold_label, context, strict=True)
    ]
