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


Source = Annotated[
    str, StringConstraints(pattern=r"^(?:(?:evalkit|ggh):\S+|owner_review|model_review)$")
]


class BehaviourRow(Model):
    rule: Literal["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    model: NonBlank
    metric: NonBlank
    value: Interval | float | None
    n: NonNegInt
    source: Source
    run: NonBlank | None = None  # the results folder under docs/results/ the row was read from
    # a `model_review` row (OWNER D-61): who reviewed, as the review file records it, and the
    # committed prompt the reviewer was given
    reviewer: NonBlank | None = None
    review_prompt: NonBlank | None = None


Version = Literal["v1", "v2"]
Finding = Literal["ok", "model_fault", "pipeline_fault", "unclear"]


CalibrationStamp = Literal["calibrated", "below_bar", "not_stampable_self_reference"]


class GemmaFamilyAgreement(Model):
    """One judge on the 40 Gemma-answer claims (OWNER D-24), from gemma_family_report.json."""

    run: NonBlank  # the GGH run the claims were drawn from (main_v2)
    run_id: NonBlank
    judge_prompt_hash: NonBlank
    n_claims: PosInt
    coverage: Prob
    agreement: Prob
    kappa: Interval
    by_model: FrozenMap[Prob]
    source: NonBlank


class JudgeCalibration(Model):
    """One judge's calibration against the reference labels (OWNER D-24, D-59, D-60), read from
    the committed copy of EvalKit's `calibration.json`. `status` follows from the numbers:
    the reference labeller's own model is never stamped (it would be agreeing with itself);
    any other judge is stamped exactly when its held-out kappa's 95% lower bound reaches
    `bar`. A judge under the bar keeps its numbers published, with the miss disclosed."""

    judge: NonBlank
    task: NonBlank  # the EvalKit task the calibration covers ("pubmedqa")
    judge_prompt_hash: NonBlank
    split: Literal["heldout"]
    n_claims: PosInt
    coverage: Prob
    agreement: Prob
    kappa: Interval
    bar: Prob  # the kappa lower bound needed (OWNER D-59: 0.4)
    meets_bar: bool
    reference_labeller: NonBlank  # who made the labels, as reports must say (OWNER D-60)
    reference_model: NonBlank  # the labeller's model id, for the self-reference rule
    status: CalibrationStamp
    wording: NonBlank  # the sentence reports print
    source: NonBlank  # the committed calibration file
    scored_at: NonBlank
    gemma_family: GemmaFamilyAgreement | None = None

    @model_validator(mode="after")
    def _status_follows_the_rule(self) -> JudgeCalibration:
        if self.meets_bar != (self.kappa.lo >= self.bar):
            raise ValueError("meets_bar must say whether the kappa lower bound reaches the bar")
        if self.judge == self.reference_model:
            want = "not_stampable_self_reference"
        else:
            want = "calibrated" if self.meets_bar else "below_bar"
        if self.status != want:
            raise ValueError(f"status {self.status!r} contradicts OWNER D-59/D-60 ({want})")
        return self


class LabellerConsistency(Model):
    """The reference labeller's 20-answer relabel (OWNER D-24, D-60), from labeller.json."""

    labeller: NonBlank
    prompt: NonBlank
    prompt_sha256: Sha256
    claims: PosInt
    answers: PosInt
    identical: NonNegInt
    kappa: Interval
    method: NonBlank
    source: NonBlank


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
    # the claim extractor, from the release's plan or the EvalKit config it names (E3.2: Claude
    # Haiku 5.5; V2.2, OWNER D-57: GPT-6 Luna), so numbers from different set-ups stay labelled
    extractor: NonBlank | None = None
    # `calibration_status` above is EvalKit's stamp in the committed summary, copied as is
    # (OWNER D-59: summaries are never rewritten). The bundle's own stamp is per judge, in
    # ResultsBundle.calibration; a run whose task a `pubmedqa` calibration cannot cover
    # (off-scope, probe) is exempt from it, and this says so.
    calibration_exempt: NonBlank | None = None


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
    """A paired comparison, oriented as model_a minus model_b.

    `method` "evalkit": copied from EvalKit's summary (Holm across the run's model pairs).
    `method` "plan_family": computed with EvalKit's bootstrap over the plan's own Holm family
    (OWNER D-53); `family_size` is the number of tests corrected together."""

    plan: NonBlank  # the pre-registration that named it, or "secondary"
    run: NonBlank
    model_a: NonBlank
    model_b: NonBlank
    metric: NonBlank
    judge: NonBlank | None
    diff: Interval
    p_value: Prob | None
    separated: bool  # after Holm, see `method`
    verdict: Literal["a_better", "b_better", "tie"]
    better: Literal["higher", "lower"] = "higher"  # the outcome's direction
    method: Literal["evalkit", "plan_family"] = "evalkit"
    family_size: PosInt | None = None

    @model_validator(mode="after")
    def _verdict_follows_the_rule(self) -> Comparison:
        a_ahead = self.diff.point > 0 if self.better == "higher" else self.diff.point < 0
        want = "tie" if not self.separated else ("a_better" if a_ahead else "b_better")
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
    parity: ParityReport | None = None  # judged_headline_delta: bf16 minus Q4_K_M, per judge
    probe: ProbeResult | None = None
    artifacts: tuple[ExportArtifact, ...]
    # the judges the release's primary comparisons rely on (v1: Opus and Sol; v2, OWNER D-57:
    # Sol), and "calibrated" only when every one of them is stamped and Gemma-family checked
    calibration_judges: tuple[NonBlank, ...] = ()
    calibration_status: Literal["calibrated", "uncalibrated"] = "uncalibrated"
    # the release's own laptop bench (configs/results.yaml: the release's `bench` file and
    # `bench_variant`), so the card quotes the shipped model's speed, not an earlier one's
    bench: LaptopBench | None = None
    bench_source: NonBlank | None = None  # "<file>#<variant>"


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
    # a pipeline fault stops publication until this names its committed explanation:
    # "<path under docs/>#<section>: <what it is and what it changes>"
    explanation: str = ""

    @model_validator(mode="after")
    def _fault_needs_a_note(self) -> SampleReadRow:
        if self.explanation and self.finding != "pipeline_fault":
            raise ValueError("only a pipeline fault takes an explanation")
        if self.explanation and not self.explanation.startswith("docs/"):
            raise ValueError("an explanation names the committed file that explains the fault")
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
    read_by: tuple[NonBlank, ...] = ()  # each distinct `read_by` of the rows (OWNER D-61)


class ResultsBundle(Model):
    """`results_bundle.json` (D-20): the single source the model card, README and project page
    read. Built by `ggh results bundle` from committed files only; every number is copied from
    EvalKit's summaries or the committed measurements, never typed."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    evalkit_run_ids: tuple[NonBlank, ...]
    evalkit_version: NonBlank
    judge_mode: Literal["dual", "single"]
    # "calibrated" only when every release in the bundle is (each release's own stamp is on
    # its ReleaseEntry; each judge's on `calibration`)
    calibration_status: Literal["calibrated", "uncalibrated"]
    # judge -> agreement on Gemma claims, set only with the bundle's `calibrated` stamp (the
    # per-judge numbers are always in `calibration[].gemma_family`)
    gemma_family_check: FrozenMap[Prob] | None = None
    calibration: tuple[JudgeCalibration, ...] = ()  # one per judge of the pubmedqa runs
    labeller_consistency: LabellerConsistency | None = None
    summary_paths: tuple[NonBlank, ...]
    runs: tuple[RunRef, ...]
    releases: tuple[ReleaseEntry, ...]
    results: tuple[ModelResult, ...]
    # pre-registered comparisons (§7.4) as "<metric>: <a> minus <b> (<judge>)" -> difference, the
    # shape the R4 card reads; `primary_comparisons` holds the same with p, Holm and verdict
    primary: FrozenMap[Interval]
    primary_comparisons: tuple[Comparison, ...]
    secondary: tuple[Comparison, ...]
    behaviour: tuple[BehaviourRow, ...]
    bench: LaptopBench  # the shipped release's own bench (its ReleaseEntry.bench)
    bench_variants: FrozenMap[LaptopBench]  # every variant in the shipped release's bench file
    sample_read: tuple[SampleReadSummary, ...]
    spend_usd: NonNegFloat  # the spend ledger's total for this bundle's releases (no Wave 7 rows)
    lineage_ok: bool
    lineage_scope: Literal["committed", "full"]  # committed manifests only, or artifacts/ too
    lineage_problems: tuple[str, ...]
    publish_blockers: tuple[str, ...]  # what must happen before the card is published

    @model_validator(mode="after")
    def _consistent(self) -> ResultsBundle:
        if self.calibration_status == "calibrated" and not self.gemma_family_check:
            raise ValueError("calibrated results need the Gemma-family judge check (OWNER D-24)")
        judges = {c.judge: c for c in self.calibration}
        if len(judges) != len(self.calibration):
            raise ValueError("calibration must hold one entry per judge")
        for rel in self.releases:
            if rel.calibration_status != "calibrated":
                continue
            if not rel.calibration_judges:
                raise ValueError(f"{rel.version} is calibrated but names no judges")
            for j in rel.calibration_judges:
                c = judges.get(j)
                if c is None or c.status != "calibrated" or c.gemma_family is None:
                    raise ValueError(
                        f"{rel.version} is calibrated but {j} is not stamped and Gemma-family "
                        "checked (OWNER D-24, D-59, D-60)"
                    )
        if self.calibration_status == "calibrated":
            loose = [r.version for r in self.releases if r.calibration_status != "calibrated"]
            if loose:
                raise ValueError(f"the bundle is calibrated but {', '.join(loose)} is not")
        if self.lineage_ok == bool(self.lineage_problems):
            raise ValueError("lineage_ok must be true exactly when lineage_problems is empty")
        names = {r.name for r in self.runs}
        for rel in self.releases:
            missing = set(rel.runs) - names
            if missing:
                raise ValueError(f"{rel.version} names runs the bundle lacks: {sorted(missing)}")
        for c in self.primary_comparisons:
            if c.plan == "secondary":
                raise ValueError("a primary comparison must name its plan")
        if set(self.primary) != {primary_key(c) for c in self.primary_comparisons}:
            raise ValueError("primary must hold exactly the primary comparisons")
        if [r.version for r in self.releases] != sorted({r.version for r in self.releases}):
            raise ValueError("releases must be unique and ordered v1, v2")
        if not self.releases:
            raise ValueError("a bundle needs at least one release")
        return self

    # The flat fields of the first ResultsBundle (Tech Spec §3), read by the R4 card and Hub
    # code, now come from the shipped release: the newest one in the bundle.
    @property
    def shipped(self) -> ReleaseEntry:
        return self.releases[-1]

    @property
    def train(self) -> TrainRun:
        return self.shipped.train

    @property
    def parity(self) -> ParityReport | None:
        return self.shipped.parity

    @property
    def artifacts(self) -> tuple[ExportArtifact, ...]:
        return self.shipped.artifacts

    @property
    def audit_pass_rate(self) -> float:
        """The shipped release's latest audit (the one its training data was cleared on)."""
        return self.shipped.audits[-1].pass_rate


def primary_key(c: Comparison) -> str:
    return f"{c.metric}: {c.model_a} minus {c.model_b}" + (f" ({c.judge})" if c.judge else "")
