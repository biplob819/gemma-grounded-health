"""Data schemas: training examples, drops, the data manifest and hand-audit labels (§3, §4)."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from evalkit.schemas import Answer
from pydantic import StringConstraints, field_validator, model_validator

from ggh.schemas._base import (
    SCHEMA_VERSION,
    FrozenMap,
    GitSha,
    Model,
    NonBlank,
    NonNegInt,
    Prob,
    Sha256,
)

Kind = Literal["answer", "insufficient", "offscope"]
RuleId = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}-R[0-9]+$")]
_ID_RE = re.compile(r"pqa-a-(?P<pmid>[0-9]+)(?::(?P<kind>insufficient|offscope))?")


class TrainExample(Model):
    """One rendered training example. `target` is EvalKit's own `Answer` (D-2)."""

    id: NonBlank
    pmid: Annotated[str, StringConstraints(pattern=r"^[0-9]+$")]
    kind: Kind
    split: Literal["train", "val", "dev"]
    question: NonBlank
    context: tuple[NonBlank, ...]
    prompt: NonBlank
    target: Answer
    source_label: Literal["yes", "no"]
    rule_ids: tuple[RuleId, ...]
    support: Prob
    nli_entail: Prob

    @field_validator("context")
    @classmethod
    def _context_not_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("context needs at least one sentence")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> TrainExample:
        m = _ID_RE.fullmatch(self.id)  # fullmatch: a trailing newline is not an id
        if not m:
            raise ValueError(f"id {self.id!r} must be pqa-a-<pmid>[:insufficient|:offscope]")
        if m["pmid"] != self.pmid:
            raise ValueError(f"id {self.id!r} does not carry pmid {self.pmid}")
        id_kind = m["kind"] or "answer"
        if id_kind != self.kind:
            raise ValueError(f"id {self.id!r} implies kind {id_kind!r}, not {self.kind!r}")
        cites = self.target.citations
        if not cites:
            raise ValueError("target needs at least one citation (B1)")
        if list(cites) != sorted(set(cites)):
            raise ValueError("citations must be sorted and unique")
        if cites[-1] > len(self.context):
            raise ValueError(f"citation {cites[-1]} is outside 1..{len(self.context)} (B7)")
        label = self.target.label
        if self.kind in ("insufficient", "offscope") and label != "maybe":
            raise ValueError(f"a {self.kind} example must be labelled maybe, not {label!r}")
        if self.kind == "answer" and label != self.source_label:
            raise ValueError(f"answer label {label!r} differs from source label")
        return self


class DropRecord(Model):
    pmid: Annotated[str, StringConstraints(pattern=r"^[0-9]+$")]
    reason: Literal[
        "too_long",
        "no_supported_sentence",
        "over_80_words",
        "no_citation",
        "overlap_with_pqa_l",
        "duplicate",
    ]


class DataManifest(Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    dataset: NonBlank
    dataset_revision: GitSha
    evalkit_version: NonBlank
    template_hash: Sha256
    splitter_version: NonBlank
    seed: NonNegInt
    counts: FrozenMap[FrozenMap[NonNegInt]]
    drops: FrozenMap[NonNegInt]
    files: FrozenMap[Sha256]
    offscope_eval_hash: Sha256


class AuditLabel(Model):
    example_id: NonBlank
    content_digest: Sha256  # binds the label to the exact text shown (EvalKit L-26)
    verdict: Literal["ok", "weak_citation", "unsupported_answer", "wrong_label"]
    note: str = ""
