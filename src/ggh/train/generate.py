"""TRN-03: greedy dev generation, constrained or not (D-4, D-6, Tech Spec §5.2).

Constrained replies are decoded under EvalKit's answer schema, compiled by xgrammar with the
separators the training targets use (`, ` and `: `, no other whitespace), so "schema valid"
means what it means at benchmark time. Prompts go through the pinned chat template with
thinking off, as in training and serving. Batches are left-padded so every row's generation
starts at the same position.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from evalkit.pipeline.answer import answer_schema

from ggh.train.format import END_OF_TURN

LABELS = ("yes", "no", "maybe")
MAX_NEW_TOKENS = 384
BATCH = 16
SEPARATORS = (", ", ": ")  # json.dumps defaults: what ggh.train.format.target_text writes


def schema_json() -> str:
    return json.dumps(dict(answer_schema(LABELS)))


def compile_grammar(tok: Any, vocab_size: int) -> Any:  # pragma: no cover - needs xgrammar
    import xgrammar as xgr  # pyright: ignore[reportMissingImports]

    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size)
    return xgr.GrammarCompiler(info).compile_json_schema(
        schema_json(), any_whitespace=False, separators=SEPARATORS
    )


def grammar_processor(grammar: Any) -> Any:  # pragma: no cover - needs xgrammar
    """A fresh processor per batch: it holds the per-row matcher state."""
    import xgrammar as xgr  # pyright: ignore[reportMissingImports]

    return xgr.contrib.hf.LogitsProcessor(grammar)


def prompt_ids(tok: Any, prompt: str) -> list[int]:
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    ids = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
    return [int(i) for i in ids]


def left_pad(rows: Sequence[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    width = max(len(r) for r in rows)
    ids = [[pad_id] * (width - len(r)) + r for r in rows]
    mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
    return ids, mask


def stop_ids(tok: Any) -> list[int]:
    ids = [tok.convert_tokens_to_ids(END_OF_TURN)]
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def generate(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    *,
    grammar: Any = None,
    batch_size: int = BATCH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    processor: Any = grammar_processor,
) -> list[str]:
    """Greedy replies, in the order of `prompts`; constrained when `grammar` is given."""
    import torch  # pyright: ignore[reportMissingImports]

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    device = next(model.parameters()).device
    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = [prompt_ids(tok, p) for p in prompts[start : start + batch_size]]
        ids, mask = left_pad(chunk, pad)
        x = torch.tensor(ids, device=device)
        kwargs: dict[str, Any] = {
            "attention_mask": torch.tensor(mask, device=device),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": stop_ids(tok),
            "pad_token_id": pad,
        }
        if grammar is not None:
            kwargs["logits_processor"] = [processor(grammar)]
        with torch.no_grad():
            gen = model.generate(x, **kwargs)
        for row in gen[:, x.shape[1] :]:
            out.append(tok.decode(row, skip_special_tokens=True).strip())
    return out
