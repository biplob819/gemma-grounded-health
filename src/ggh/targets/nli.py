"""TGT-R4: entailment scoring with a pinned open cross-encoder.

The real model is used only by `ggh targets` and needs the optional `nli` dependency group
(`uv sync --group nli`). Tests inject a fake scorer. Every score the build uses is recorded,
so `reconstruct` can replay the exact decisions without the model (PRV-R1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

MODEL_ID = "cross-encoder/nli-deberta-v3-base"
MODEL_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"  # Apache-2.0
MAX_TOKENS = 512
BATCH = 32

Pair = tuple[str, str]  # (premise, hypothesis)


class Scorer(Protocol):
    def entail(self, pairs: Sequence[Pair]) -> list[float]:
        """Entailment probability of each hypothesis given its premise, in order."""
        ...


def pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{premise}\x00{hypothesis}".encode()).hexdigest()


class ReplayScorer:
    """Scores from a recorded table; an unknown pair is an error, never a guess."""

    def __init__(self, table: Mapping[str, float]) -> None:
        self.table = dict(table)

    def entail(self, pairs: Sequence[Pair]) -> list[float]:
        out = []
        for p, h in pairs:
            key = pair_key(p, h)
            if key not in self.table:
                raise KeyError(f"no recorded NLI score for pair {key[:12]}…")
            out.append(self.table[key])
        return out


class RecordingScorer:
    """Wraps a scorer, rounds its scores and remembers each one by pair key."""

    def __init__(self, inner: Scorer, decimals: int) -> None:
        self.inner = inner
        self.decimals = decimals
        self.table: dict[str, float] = {}

    def entail(self, pairs: Sequence[Pair]) -> list[float]:
        todo = [(p, h) for p, h in dict.fromkeys(pairs) if pair_key(p, h) not in self.table]
        if todo:
            for (p, h), s in zip(todo, self.inner.entail(todo), strict=True):
                self.table[pair_key(p, h)] = round(float(s), self.decimals)
        return [self.table[pair_key(p, h)] for p, h in pairs]


def entailment_index(id2label: Mapping[Any, str]) -> int:
    """The output index labelled entailment, read from the model config, never assumed."""
    hits = [int(k) for k, v in id2label.items() if str(v).strip().lower() == "entailment"]
    if len(hits) != 1:
        raise ValueError(f"model id2label {dict(id2label)} has no single 'entailment' label")
    return hits[0]


class HFScorer:  # pragma: no cover - needs the nli group and a model download
    """The pinned cross-encoder on the best local device."""

    def __init__(self, device: str | None = None) -> None:
        import torch  # pyright: ignore[reportMissingImports]  (the optional nli group)
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.model = model.eval().to(device)
        self.ent = entailment_index(model.config.id2label)

    def entail(self, pairs: Sequence[Pair]) -> list[float]:
        # similar lengths batched together and padded to a multiple of 64, so the GPU sees
        # few distinct shapes; results go back in the caller's order
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [0.0] * len(pairs)
        with self._torch.no_grad():
            for i in range(0, len(order), BATCH):
                idx = order[i : i + BATCH]
                enc = self.tok(
                    [pairs[j][0] for j in idx],
                    [pairs[j][1] for j in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    pad_to_multiple_of=64,
                ).to(self.device)
                probs = self.model(**enc).logits.softmax(-1)[:, self.ent].float().cpu().tolist()
                for j, pr in zip(idx, probs, strict=True):
                    out[j] = pr
        return out
