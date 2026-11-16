"""RCP-02: training text and completion-only labels (Tech Spec §5.1, D-2).

The user turn is the example's `prompt` (EvalKit's rendering). The assistant turn is the
target as compact JSON, exactly what EvalKit's `parse_answer` reads. Both go through the
pinned tokenizer's chat template; every token up to and including the generation prompt is
labelled -100, so the loss sees only the answer and its end-of-turn token.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evalkit.schemas import Answer

from ggh.schemas import TrainExample

IGNORE = -100
END_OF_TURN = "<turn|>"


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def convert_tokens_to_ids(self, tokens: str) -> int: ...


def target_text(answer: Answer) -> str:
    """The assistant turn: compact JSON in EvalKit's answer shape."""
    return json.dumps(
        {"label": answer.label, "answer": answer.answer, "citations": list(answer.citations)},
        ensure_ascii=False,
    )


def messages(e: TrainExample, *, with_answer: bool = True) -> list[dict[str, str]]:
    turns = [{"role": "user", "content": e.prompt}]
    if with_answer:
        turns.append({"role": "assistant", "content": target_text(e.target)})
    return turns


def _ids(out: Any) -> list[int]:
    ids = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
    return [int(i) for i in ids]


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_prompt: int
    n_target: int  # labelled tokens: the answer plus the end-of-turn token


class FormatError(ValueError):
    pass


def encode(tok: ChatTokenizer, e: TrainExample) -> Encoded:
    prompt = _ids(tok.apply_chat_template(messages(e, with_answer=False), tokenize=True,
                                          add_generation_prompt=True))  # fmt: skip
    full = _ids(tok.apply_chat_template(messages(e), tokenize=True))
    if full[: len(prompt)] != prompt:
        raise FormatError(f"{e.id}: the prompt is not a token prefix of the full conversation")
    eot = tok.convert_tokens_to_ids(END_OF_TURN)
    try:
        end = full.index(eot, len(prompt)) + 1  # keep the end-of-turn token, drop what follows
    except ValueError:
        raise FormatError(f"{e.id}: no {END_OF_TURN} after the answer") from None
    ids = full[:end]
    labels = [IGNORE] * len(prompt) + ids[len(prompt) :]
    return Encoded(ids, labels, len(prompt), end - len(prompt))


@dataclass(frozen=True)
class Dataset:
    rows: list[Encoded]
    dropped_too_long: int
    max_len: int

    @property
    def tokens(self) -> int:
        return sum(len(r.input_ids) for r in self.rows)


def build(tok: ChatTokenizer, examples: Iterable[TrainExample], max_len: int) -> Dataset:
    """Encode every example; an example longer than `max_len` is dropped and counted, never
    truncated (a truncated example would teach a cut-off answer)."""
    rows, dropped = [], 0
    for e in examples:
        enc = encode(tok, e)
        if len(enc.input_ids) > max_len:
            dropped += 1
            continue
        rows.append(enc)
    return Dataset(rows, dropped, max_len)


def collate(rows: Sequence[Encoded], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch; padding is ignored by the loss and masked out of attention."""
    width = max(len(r.input_ids) for r in rows)
    return {
        "input_ids": [r.input_ids + [pad_id] * (width - len(r.input_ids)) for r in rows],
        "labels": [r.labels + [IGNORE] * (width - len(r.labels)) for r in rows],
        "attention_mask": [[1] * len(r.input_ids) + [0] * (width - len(r.input_ids)) for r in rows],
    }
