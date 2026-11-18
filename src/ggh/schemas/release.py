"""Release schemas: exported artifacts, parity, laptop speed, behaviour rows, the bundle (§3)."""

from __future__ import annotations

from typing import Annotated, Literal

from evalkit.schemas import Interval
from pydantic import StringConstraints, model_validator

from ggh.schemas._base import (
    SCHEMA_VERSION,
    FrozenMap,
    Model,
    NonBlank,
    NonNegFloat,
    NonNegInt,
    PosInt,
    Prob,
    Sha256,
)
from ggh.schemas.train import TrainRun


class ExportArtifact(Model):
    kind: Literal["adapter", "merged_bf16", "gguf"]
    model: Literal["base", "ft"]
    quant: NonBlank | None = None  # "Q4_K_M", "Q8_0"
    path: NonBlank
    sha256: Sha256
    llama_cpp_commit: NonBlank | None = None
    parent_key: Sha256

    @model_validator(mode="after")
    def _gguf_fields(self) -> ExportArtifact:
        gguf = self.kind == "gguf"
        if gguf and (self.quant is None or self.llama_cpp_commit is None):
            raise ValueError("a gguf artifact needs quant and llama_cpp_commit")
        if not gguf and (self.quant is not None or self.llama_cpp_commit is not None):
            raise ValueError(f"a {self.kind} artifact has no quant or llama_cpp_commit")
        return self


class ParityReport(Model):
    n: PosInt
    label_agreement: Prob  # Q4_K_M vs bf16, same prompts, greedy
    citation_exact_match: Prob
    schema_valid: FrozenMap[Prob]
    judged_headline_delta: FrozenMap[Interval] | None = None  # per judge, 200 PQA-L cases


class LaptopBench(Model):
    chip: NonBlank
    ram_gb: PosInt
    os: NonBlank
    power: Literal["ac", "battery"]
    llama_cpp_commit: NonBlank
    quant: NonBlank
    model_file_mb: NonNegFloat
    n_prompts: PosInt
    repetitions: PosInt
    prompt_tps_p50: NonNegFloat
    gen_tps_p50: NonNegFloat
    gen_tps_p95: NonNegFloat
    ttft_ms_p50: NonNegInt
    answer_latency_ms_p50: NonNegInt
    answer_latency_ms_p95: NonNegInt
    peak_rss_mb: NonNegFloat
    llama_bench: FrozenMap[NonNegFloat]  # pp512, tg128


Source = Annotated[str, StringConstraints(pattern=r"^(?:(?:evalkit|ggh):\S+|owner_review)$")]


class BehaviourRow(Model):
    rule: Literal["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    model: NonBlank
    metric: NonBlank
    value: Interval | float | None
    n: NonNegInt
    source: Source
    run: NonBlank | None = None  # the results folder under docs/results/ the row was read from


Version = Literal["v1", "v2"]
Finding = Literal["ok", "model_fault", "pipeline_fault", "unclear"]


class RunRef(Model):
    """One committed EvalKit run folder under docs/results/, as the bundle read it."""

    name: NonBlank  # the folder name: "main", "main_v2", ...
    run_id: NonBlank
    task: NonBlank
    n: PosInt
    models: tuple[NonBlank, ...]
    judges: tuple[NonBlank, ...]
    judge_mode: Literal["dual", "single"]
    calibration_status: Literal["calibrated", "uncalibrated"]
    judging_status: Literal["pending", "complete"]
    started_at: NonBlank
    spent_usd: NonNegFloat
    summary_sha256: Sha256
    records_sha256: Sha256


class ModelResult(Model):
    """One model's numbers in one run, copied from EvalKit's summary (never recomputed)."""

    run: NonBlank
    model: NonBlank
    n: NonNegInt
    accuracy: Interval
    headline: FrozenMap[Interval]  # per judge
    groundedness_mean: FrozenMap[Interval]
    citation_validity: Interval
    citation_precision: FrozenMap[Interval] | None = None  # citation_support runs (OWNER D-44)
    aim_citation_rate: FrozenMap[Interval] | None = None


class Comparison(Model):
    """A paired comparison from EvalKit's summary, oriented as model_a minus model_b."""

    plan: NonBlank  # the pre-registration that named it, or "secondary"
    run: NonBlank
    model_a: NonBlank
    model_b: NonBlank
    metric: NonBlank
    judge: NonBlank | None
    diff: Interval
    p_value: Prob | None
    separated: bool  # EvalKit's rule: paired bootstrap, Holm across the run's pairs
    verdict: Literal["a_better", "b_better", "tie"]

    @model_validator(mode="after")
    def _verdict_follows_the_rule(self) -> Comparison:
        want = "tie" if not self.separated else ("a_better" if self.diff.point > 0 else "b_better")
        if self.verdict != want:
            raise ValueError(f"verdict {self.verdict!r} contradicts the separation rule ({want})")
        return self


class ProbeResult(Model):
    """The paraphrase probe for one model (D-42): accuracy on each wording, paired drop."""

    model: NonBlank
    n: PosInt
    accuracy_original: Prob
    accuracy_paraphrase: Prob
    drop: Interval
    contamination_signal: bool


class AuditSummary(Model):
    name: NonBlank  # "v1 (N1.3)", "v2.2 round 3 (AUD-R4)", ...
    path: NonBlank
    labelled: NonNegInt
    pass_rate: Prob  # share of examples judged ok overall
    passed: bool


class ReleaseEntry(Model):
    """Everything about one fine-tune version the card reports (v1, and v2 once it is run)."""

    version: Version
    model: NonBlank  # the EvalKit model id of its Q4_K_M GGUF
    plan: NonBlank
    plan_sha256: Sha256 | None  # None only while the plan is a draft
    plan_status: Literal["draft", "final"]
    built_after_test_seen: bool  # OWNER D-43: v2 was built after v1's test results were seen
    runs: tuple[NonBlank, ...]  # RunRef names that belong to this version
    train: TrainRun
    adapter_sha256: Sha256 | None = None
    audits: tuple[AuditSummary, ...]
    parity: ParityReport | None = None
    parity_headline: FrozenMap[Interval] | None = None  # bf16 minus Q4_K_M, per judge
    probe: ProbeResult | None = None
    artifacts: tuple[ExportArtifact, ...]


class CitedSentenceVerdict(Model):
    n: PosInt
    supports: bool
    role: NonBlank


class SampleVerdict(Model):
    verdict: NonBlank
    evidence: tuple[PosInt, ...]
    rationale: str
    citation_support: tuple[CitedSentenceVerdict, ...] = ()


class SampleClaim(Model):
    id: NonBlank
    text: NonBlank
    cited: tuple[PosInt, ...]
    verdicts: FrozenMap[SampleVerdict]  # judge -> verdict


class SampleReadRow(Model):
    """One line of `docs/measurements/sample_read_<run>.jsonl` (Tech Spec §7.6, D-18, RES-05).

    `ggh results sample` writes the record view with `finding` null; the reader fills
    `finding` and `note`. A pipeline fault stops publication until it is explained."""

    run: NonBlank
    model: NonBlank
    case_id: NonBlank
    record_sha256: Sha256  # canonical JSON of the record read, so an edited record is caught
    reference_label: NonBlank | None
    label: NonBlank | None
    answer: str
    citations: tuple[PosInt, ...]
    cited_sentences: FrozenMap[str]  # sentence number (as text) -> sentence
    claims: tuple[SampleClaim, ...]
    finding: Finding | None = None
    note: str = ""
    read_by: NonBlank | None = None

    @model_validator(mode="after")
    def _fault_needs_a_note(self) -> SampleReadRow:
        if self.finding in ("pipeline_fault", "model_fault", "unclear") and not self.note.strip():
            raise ValueError(f"a {self.finding} finding needs a note")
        if self.finding is not None and self.read_by is None:
            raise ValueError("a finding needs read_by")
        return self


class SampleReadSummary(Model):
    run: NonBlank
    model: NonBlank
    drawn: NonNegInt
    read: NonNegInt
    findings: FrozenMap[NonNegInt]
    complete: bool  # every drawn record has a finding


class ResultsBundle(Model):
    """`results_bundle.json` (D-20): the single source the model card, README and project page
    read. Built by `ggh results bundle` from committed files only; every number is copied from
    EvalKit's summaries or the committed measurements, never typed."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    evalkit_run_ids: tuple[NonBlank, ...]
    evalkit_version: NonBlank
    judge_mode: Literal["dual", "single"]
    calibration_status: Literal["calibrated", "uncalibrated"]
    gemma_family_check: FrozenMap[Prob] | None = None  # judge -> agreement on Gemma claims
    summary_paths: tuple[NonBlank, ...]
    runs: tuple[RunRef, ...]
    releases: tuple[ReleaseEntry, ...]
    results: tuple[ModelResult, ...]
    primary: tuple[Comparison, ...]  # pre-registered comparisons (§7.4), per judge
    secondary: tuple[Comparison, ...]
    behaviour: tuple[BehaviourRow, ...]
    bench: FrozenMap[LaptopBench]  # variant ("ft-Q4_K_M", ...) -> measurement
    sample_read: tuple[SampleReadSummary, ...]
    spend_usd: NonNegFloat  # the spend ledger's total
    lineage_ok: bool
    lineage_scope: Literal["committed", "full"]  # committed manifests only, or artifacts/ too
    lineage_problems: tuple[str, ...]
    publish_blockers: tuple[str, ...]  # what must happen before the card is published

    @model_validator(mode="after")
    def _consistent(self) -> ResultsBundle:
        if self.calibration_status == "calibrated" and not self.gemma_family_check:
            raise ValueError("calibrated results need the Gemma-family judge check (OWNER D-24)")
        if self.lineage_ok == bool(self.lineage_problems):
            raise ValueError("lineage_ok must be true exactly when lineage_problems is empty")
        names = {r.name for r in self.runs}
        for rel in self.releases:
            missing = set(rel.runs) - names
            if missing:
                raise ValueError(f"{rel.version} names runs the bundle lacks: {sorted(missing)}")
        for c in self.primary:
            if c.plan == "secondary":
                raise ValueError("a primary comparison must name its plan")
        if [r.version for r in self.releases] != sorted({r.version for r in self.releases}):
            raise ValueError("releases must be unique and ordered v1, v2")
        return self
