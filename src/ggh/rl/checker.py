"""RLB-01: the v3b grounding checker (OWNER D-70). Two local models at pinned revisions read each
answer against its cited sentences, and their support probabilities are averaged:

- lytang/MiniCheck-RoBERTa-Large (MIT), built to check a sentence against a document; output
  label "1" is "supported";
- cross-encoder/nli-deberta-v3-large (Apache-2.0), an NLI cross-encoder; output label
  "entailment".

The pair and the threshold were chosen on 600 committed dev answers (base, v1 and v2 on the 200
dev grounding cases) against GPT-6.1 Sol's correct-and-grounded verdict
(`scripts/v3b_checker_eval.py`, `docs/measurements/v3/checker_eval.json`). The pair had the best
AUC (0.868) and 5-fold cross-validated agreement with Sol (0.82), against 0.69 for v3's lexical
check, and its folds chose 0.65 four times out of five. Those 200 dev cases are left out of v3b's
checkpoint selection (`TUNING_IDS`).

No API call: the weights download once from the public Hub (no token) and run locally. The
MiniCheck weights are a pickled .bin; transformers 5 loads it with torch's weights-only loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

THRESHOLD = 0.65
MAX_TOKENS = 512
BATCH = 16
TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json"]  # fmt: skip
MODELS = (
    {
        "repo": "lytang/MiniCheck-RoBERTa-Large",
        "revision": "74c8919647e61ed0f71bc177d94f10930f090068",
        "label": "1",
        "files": ["pytorch_model.bin", "vocab.json", "merges.txt", *TOKENIZER_FILES],
    },
    {
        "repo": "cross-encoder/nli-deberta-v3-large",
        "revision": "bab4bc7178836f731dcfd18c06ca9def0a137712",
        "label": "entailment",
        "files": ["model.safetensors", "spm.model", *TOKENIZER_FILES],
    },
)
TUNING_IDS = "docs/measurements/v3/checker_tuning_ids.json"  # the 200 dev cases it was chosen on

Pair = tuple[str, str]  # (evidence: the cited sentences joined, claim: the answer)


class Checker(Protocol):
    def support(self, pairs: Sequence[Pair]) -> list[float]:
        """The mean support probability of each claim given its evidence, in order."""
        ...


def label_index(id2label: dict[Any, str], want: str) -> int:
    """The output index carrying `want`, read from the model config, never assumed."""
    hits = [int(k) for k, v in id2label.items() if str(v).strip().lower() == want]
    if len(hits) != 1:
        raise ValueError(f"model id2label {dict(id2label)} has no single {want!r} label")
    return hits[0]


class HFChecker:  # pragma: no cover - downloads two models; checked on real dev answers
    def __init__(self, device: str = "cpu") -> None:
        import torch  # pyright: ignore[reportMissingImports]
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self.device = device
        self.models = []
        for spec in MODELS:
            path = snapshot_download(spec["repo"], revision=spec["revision"],
                                     allow_patterns=spec["files"])  # fmt: skip
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
            self.models.append((tok, model, label_index(model.config.id2label, spec["label"])))

    def support(self, pairs: Sequence[Pair]) -> list[float]:
        if not pairs:
            return []
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        total = [0.0] * len(pairs)
        with self._torch.no_grad():
            for tok, model, idx in self.models:
                for i in range(0, len(order), BATCH):
                    rows = order[i : i + BATCH]
                    enc = tok([pairs[j][0] for j in rows], [pairs[j][1] for j in rows],
                              truncation=True, max_length=MAX_TOKENS, padding=True,
                              return_tensors="pt").to(self.device)  # fmt: skip
                    probs = self._torch.softmax(model(**enc).logits.float(), dim=-1)[:, idx]
                    for j, p in zip(rows, probs.cpu().tolist(), strict=True):
                        total[j] += p
        return [t / len(self.models) for t in total]
