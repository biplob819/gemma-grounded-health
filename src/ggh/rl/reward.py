"""RLR-01: the v3 GRPO reward (OWNER D-66). Programmatic and deterministic: no judge or model
call in the training loop.

One completion is scored against its example's gold label and its numbered abstract sentences:

- it must parse as EvalKit's answer (`parse_answer`, the same parser the benchmark uses);
  otherwise the reward is `R_UNPARSED`;
- `R_LABEL` when the label equals the gold label;
- `R_GROUNDED` when the citations are grounded, as defined in `grounded` below;
- minus `P_AIM` when a yes/no answer cites an aim, objective or study-description sentence
  (TGT-R12's `is_aim_sentence`: the v1 failure the benchmark's aim-citation rate measures).

The grounding check is stricter than "every index is in range", because the plan's planted hacks
(an answer citing valid but unrelated sentences, citing many sentences, or a few words) would
all pass a range check. A yes/no answer is grounded only when it has 1 to `MAX_CITES` distinct
citations, all in range, none an aim or method sentence (TGT-R12's exclusion), each sharing at
least `MIN_SHARED_WORDS` content words with the answer, and the answer is `MIN_WORDS` to
`MAX_WORDS` words long. (A result-sentence check was tried and dropped: it withheld the credit
from 459 of v2's 2,423 gold answers, whose cited sentences entail them.) A "maybe" answer earns
the grounding credit only when the gold label is "maybe" (the insufficient and off-scope
targets) and its citations are in range: answering "maybe" to a yes/no question earns nothing
beyond parsing.

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
    grounded: bool
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
    if answer.label == "maybe":
        return "" if gold == "maybe" else "maybe for a yes/no question"
    if not cites:
        return "no citation"
    if len(cites) > MAX_CITES:
        return f"more than {MAX_CITES} citations"
    n = _words(answer.answer)
    if n < MIN_WORDS:
        return f"answer under {MIN_WORDS} words"
    if n > MAX_WORDS:
        return f"answer over {MAX_WORDS} words"
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
    reward = R_LABEL * correct + R_GROUNDED * (not problem) - P_AIM * aim
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
