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


class ResultsBundle(Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    evalkit_run_ids: tuple[NonBlank, ...]
    evalkit_version: NonBlank
    judge_mode: Literal["dual", "single"]
    calibration_status: Literal["calibrated", "uncalibrated"]
    gemma_family_check: FrozenMap[Prob] | None = None  # judge -> agreement on Gemma claims
    summary_paths: tuple[NonBlank, ...]
    primary: FrozenMap[Interval]  # pre-registered comparisons (§7.4), per judge
    behaviour: tuple[BehaviourRow, ...]
    parity: ParityReport
    bench: LaptopBench
    train: TrainRun
    audit_pass_rate: Prob
    artifacts: tuple[ExportArtifact, ...]
    lineage_ok: bool

    @model_validator(mode="after")
    def _calibration_needs_gemma_check(self) -> ResultsBundle:
        if self.calibration_status == "calibrated" and not self.gemma_family_check:
            raise ValueError("calibrated results need the Gemma-family judge check (OWNER D-24)")
        return self
