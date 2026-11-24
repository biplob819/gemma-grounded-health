"""The inputs every release document is generated from (D-20).

The results bundle is the root. Files the bundle does not carry inline are read from the paths
it names (`summary_paths`) or from committed measurement files whose sha256 the bundle or a
committed lineage manifest records, so a number in the card can always be traced to a hashed
file. Nothing here is typed by hand: when v2 lands, E3.3 writes a new bundle and every document
is regenerated from it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from evalkit.schemas import RunSummary
from evalkit.schemas.summary import ModelSummary

from ggh.config import load_recipe
from ggh.lineage import committed_output, read_manifest, sha256_file
from ggh.schemas import (
    DataManifest,
    RecipeConfig,
    ReleaseEntry,
    ResultsBundle,
    RLRun,
    RunRef,
    TrainRun,
)

BUNDLE = Path("docs") / "results" / "results_bundle.json"
DATA_MANIFEST = Path("docs") / "measurements" / "data_manifest.json"
TARGETS_REPORT = Path("docs") / "measurements" / "targets_drop_report.json"
TARGETS_LINEAGE = Path("docs") / "measurements" / "lineage" / "targets.json"
RECIPE = Path("configs") / "recipe.yaml"
MAIN_TASK = "pubmedqa"  # the PQA-L benchmark task in EvalKit
PROBE_TASK_PREFIX = "ggh_probe_"
OFFSCOPE_TASK = "ggh_offscope"
BASE_MODEL_NAME = "gemma-e2b-base-q4"
FT_MODEL_NAME = "gemma-e2b-grounded-q4"  # v1's name; the default is the shipped release's model


# each release's training inputs: (recipe, targets lineage manifest, targets drop report)
SHIPPED_INPUTS: dict[str, tuple[Path, Path, Path]] = {
    "v1": (RECIPE, TARGETS_LINEAGE, TARGETS_REPORT),
    "v2": (
        Path("configs") / "recipe_v2.yaml",
        Path("docs") / "measurements" / "lineage" / "targets_v2.json",
        TARGETS_REPORT,
    ),
}


# a GRPO release's own recipe (its SFT recipe, data and targets are those of the release whose
# adapter it started from, `sft_release`)
RL_RECIPES: dict[str, Path] = {"v3b": Path("configs") / "recipe_v3b_grpo.yaml"}
# v3b (V3.8): GRPO from v2's adapter on v2's prompts and gold labels, so v2's training inputs
SHIPPED_INPUTS["v3b"] = SHIPPED_INPUTS["v2"]


# OWNER D-62: the post-hoc aim-rule count, read only for the release whose plan it names and
# only while the run summaries it records still match the bundle's
AIM_CHECK = Path("docs") / "measurements" / "aim_rule_check.json"
# CRD-05: what was read about the base model's licence, when and from where
LICENCE_RECORD = Path("docs") / "licence_check.json"


@dataclass(frozen=True)
class DevCheck:
    """A release's pre-registered dev grounding check (V2.1, D-52): the committed EvalKit
    summary, the bars file and the file recording the bars' sha256, and the decision under
    which a missed bar was disclosed rather than gated (OWNER D-55)."""

    summary: Path
    bars: Path
    bars_sha256: Path
    decision: str


DEV_CHECKS: dict[str, DevCheck] = {
    "v2": DevCheck(
        Path("docs") / "results" / "dev_ground_v2_sol" / "summary.json",
        Path("configs") / "v2_dev_bars.yaml",
        Path("docs") / "measurements" / "v2_dev_bars.sha256",
        "OWNER D-55",
    ),
}


class SourceError(ValueError):
    """A release input that is missing, unreadable or does not match its recorded hash."""


@dataclass(frozen=True)
class Paths:
    """Where each input lives, relative to the repo root. v2 may point any of them elsewhere."""

    bundle: Path = BUNDLE
    data_manifest: Path = DATA_MANIFEST
    # None: the shipped release's own inputs (SHIPPED_INPUTS); a path given overrides
    targets_report: Path | None = None
    targets_lineage: Path | None = None
    recipe: Path | None = None
    base_model: str = BASE_MODEL_NAME
    ft_model: str | None = None  # None: the bundle's shipped release model (--ft-model overrides)


@dataclass(frozen=True)
class Sources:
    bundle: ResultsBundle
    summaries: tuple[tuple[str, RunSummary], ...]  # (path, summary), in bundle order
    data: DataManifest
    targets: dict[str, Any]
    recipe: RecipeConfig
    paths: Paths
    raw: dict[str, Any] = field(default_factory=dict[str, Any])  # every input as parsed JSON
    plan: dict[str, Any] = field(default_factory=dict[str, Any])  # the shipped release's plan
    aim_check: dict[str, Any] | None = None  # OWNER D-62, when it covers the shipped release
    licence: dict[str, Any] | None = None  # LICENCE_RECORD
    dev_check: dict[str, Any] | None = None  # {"summary", "bars", "decision", "path", ...}
    rl_recipe: dict[str, Any] | None = None  # a GRPO release's recipe (RL_RECIPES), as parsed

    @property
    def sft(self) -> ReleaseEntry:
        """The release whose supervised training the shipped one rests on (`sft_release`)."""
        return sft_release(self.bundle)

    @property
    def ft_model(self) -> str:
        """The released fine-tune's EvalKit name: `--ft-model` if given, else the shipped
        release's model, so a v2 bundle never reports v1 as the release by default."""
        return self.paths.ft_model or self.bundle.shipped.model

    def summary(self, task: str) -> RunSummary | None:
        for _, s in self.summaries:
            if s.task == task:
                return s
        return None

    def release_summaries(self, release: ReleaseEntry) -> list[RunSummary]:
        """The summaries of one release's runs, in the order the release lists them; its
        primary run comes first."""
        ids = {r.name: r.run_id for r in self.bundle.runs}
        by_id = {s.run_id: s for _, s in self.summaries}
        return [by_id[ids[n]] for n in release.runs if n in ids and ids[n] in by_id]

    def shipped_summaries(self) -> list[RunSummary]:
        """The summaries of the shipped release's runs (the newest release in the bundle)."""
        return self.release_summaries(self.bundle.shipped)

    def release_main(self, release: ReleaseEntry) -> RunSummary:
        s = next((s for s in self.release_summaries(release) if s.task == MAIN_TASK), None)
        if s is None:
            raise SourceError(f"{release.version} names no summary for the {MAIN_TASK!r} task")
        return s

    def main(self) -> RunSummary:
        return self.release_main(self.bundle.shipped)

    def earlier(self) -> list[ReleaseEntry]:
        """Releases before the shipped one. Their numbers stand under their own plans and
        scoring set-up; they are shown beside the shipped release's, labelled, never compared."""
        return list(self.bundle.releases[:-1])

    def ref(self, summary: RunSummary) -> RunRef:
        """The bundle's record of the run a summary came from (judges, extractor, name)."""
        r = next((r for r in self.bundle.runs if r.run_id == summary.run_id), None)
        if r is None:
            raise SourceError(f"run {summary.run_id} is not in the bundle")
        return r

    def citation_run(self, release: ReleaseEntry) -> RunRef | None:
        """The release's first run with citation precision in the bundle (v1: the citation
        rescore, main_citation; v2: main_v2 itself)."""
        has = {r.run for r in self.bundle.results if r.citation_precision}
        refs = {r.name: r for r in self.bundle.runs}
        return next((refs[n] for n in release.runs if n in has and n in refs), None)

    def probes(self) -> list[RunSummary]:
        return [s for s in self.shipped_summaries() if s.task.startswith(PROBE_TASK_PREFIX)]


def sft_release(bundle: ResultsBundle) -> ReleaseEntry:
    """The shipped release when it was trained by SFT (v1, v2); for a GRPO release (v3b), the
    release whose frozen adapter its run started from (`RLRun.init_adapter_sha256`)."""
    shipped = bundle.shipped
    if isinstance(shipped.train, TrainRun):
        return shipped
    start = shipped.train.init_adapter_sha256
    for r in bundle.releases:
        if r.adapter_sha256 == start and isinstance(r.train, TrainRun):
            return r
    raise SourceError(f"no release in the bundle has {shipped.version}'s start {start[:12]}")


def sft_train(bundle: ResultsBundle) -> TrainRun:
    t = sft_release(bundle).train
    assert isinstance(t, TrainRun)
    return t


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc


def read_summary(text: str) -> RunSummary:
    """A run summary as the installed EvalKit reads it. Runs made at a later EvalKit pin (the
    citation rescore, OWNER D-44) carry outcome fields this pin's schema lacks; those are
    dropped here and reach the card through the bundle (`ModelResult`), never from here."""
    data = json.loads(text)
    known = set(ModelSummary.model_fields)
    data["models"] = [{k: v for k, v in m.items() if k in known} for m in data.get("models", [])]
    top = set(RunSummary.model_fields)
    return RunSummary.model_validate({k: v for k, v in data.items() if k in top})


def load_bundle(path: Path) -> ResultsBundle:
    try:
        return ResultsBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SourceError(f"cannot read the results bundle {path}: {exc}") from exc


def load(root: Path, paths: Paths | None = None) -> Sources:
    """Every input, each checked against the hash the bundle or a lineage manifest records."""
    p = paths or Paths()
    bundle_path = root / p.bundle
    bundle = load_bundle(bundle_path)
    version = bundle.shipped.version
    if version not in SHIPPED_INPUTS and None in (p.recipe, p.targets_lineage, p.targets_report):
        raise SourceError(f"no default training inputs for release {version}; pass them")
    recipe_d, lineage_d, report_d = SHIPPED_INPUTS.get(
        version, (RECIPE, TARGETS_LINEAGE, TARGETS_REPORT)
    )
    p = replace(
        p,
        recipe=p.recipe or recipe_d,
        targets_lineage=p.targets_lineage or lineage_d,
        targets_report=p.targets_report or report_d,
    )
    assert p.recipe and p.targets_lineage and p.targets_report
    raw: dict[str, Any] = {"bundle": _read_json(bundle_path)}
    summaries: list[tuple[str, RunSummary]] = []
    for rel in bundle.summary_paths:
        f = root / rel
        try:
            s = read_summary(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SourceError(f"cannot read summary {rel}: {exc}") from exc
        if bundle.evalkit_run_ids and s.run_id not in bundle.evalkit_run_ids:
            raise SourceError(f"summary {rel} is run {s.run_id}, not one the bundle lists")
        summaries.append((rel, s))
        raw[f"summary:{rel}"] = _read_json(f)

    sft = sft_train(bundle)
    data_file = root / p.data_manifest
    if not data_file.is_file() or sha256_file(data_file) != sft.data_manifest_hash:
        raise SourceError(f"{p.data_manifest} is missing or not the manifest the run trained on")
    data = DataManifest.model_validate_json(data_file.read_text(encoding="utf-8"))
    raw["data"] = _read_json(data_file)

    recipe_file = root / p.recipe
    if not recipe_file.is_file() or sha256_file(recipe_file) != sft.config_hash:
        raise SourceError(f"{p.recipe} is missing or not the recipe the run trained with")
    recipe = load_recipe(recipe_file)
    raw["recipe"] = recipe.model_dump(mode="json")

    lineage = read_manifest(root / p.targets_lineage)
    want = lineage.outputs.get(p.targets_report.as_posix())
    targets_file = committed_output(root, p.targets_report.as_posix(), want) if want else None
    if targets_file is None:
        raise SourceError(f"{p.targets_report} does not match the targets lineage manifest")
    targets = _read_json(targets_file)
    raw["targets"] = targets

    shipped = bundle.shipped
    plan = _read_plan(root, shipped.plan, shipped.plan_sha256)
    raw["plan"] = plan
    aim = _read_aim_check(root, bundle)
    if aim is not None:
        raw["aim_check"] = aim
    licence = _read_licence(root, recipe)
    if licence is not None:
        raw["licence"] = licence
    dev = _read_dev_check(root, version)
    if dev is not None:
        raw["dev_check"] = dev
    rl = _read_rl_recipe(root, bundle)
    if rl is not None:
        raw["rl_recipe"] = rl
    return Sources(
        bundle, tuple(summaries), data, targets, recipe, p, raw, plan, aim, licence, dev, rl
    )


def _read_rl_recipe(root: Path, bundle: ResultsBundle) -> dict[str, Any] | None:
    """A GRPO release's recipe, the file its run recorded (`RLRun.config_hash`)."""
    run = bundle.shipped.train
    if not isinstance(run, RLRun):
        return None
    rel = RL_RECIPES.get(bundle.shipped.version)
    if rel is None:
        raise SourceError(f"no GRPO recipe is registered for {bundle.shipped.version}")
    f = root / rel
    if not f.is_file() or sha256_file(f) != run.config_hash:
        raise SourceError(f"{rel} is missing or not the recipe the GRPO run trained with")
    return yaml.safe_load(f.read_text(encoding="utf-8")) or {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_plan(root: Path, rel: str, sha: str | None) -> dict[str, Any]:
    f = root / rel
    if not f.is_file():
        raise SourceError(f"the shipped release's plan {rel} is missing")
    if sha is not None and _sha256(f) != sha:
        raise SourceError(f"{rel} is not the plan the bundle records")
    return yaml.safe_load(f.read_text(encoding="utf-8")) or {}


def _read_aim_check(root: Path, bundle: ResultsBundle) -> dict[str, Any] | None:
    """OWNER D-62's post-hoc count, if it covers the shipped release's plan. Each run it read
    must still be the run in the bundle (summary sha256), or the count is stale and refused."""
    f = root / AIM_CHECK
    if not f.is_file():
        return None
    aim = _read_json(f)
    if aim.get("plan") != bundle.shipped.plan:
        return None
    refs = {r.name: r for r in bundle.runs}
    for run, s in (aim.get("sources") or {}).items():
        r = refs.get(run)
        if r is not None and r.summary_sha256 != s.get("summary_sha256"):
            raise SourceError(f"{AIM_CHECK} read another {run} summary than the bundle's")
    return aim


def _read_licence(root: Path, recipe: RecipeConfig) -> dict[str, Any] | None:
    f = root / LICENCE_RECORD
    if not f.is_file():
        return None
    rec = _read_json(f)
    base = rec.get("base_model") or {}
    if (base.get("id"), base.get("revision")) != (recipe.base_model, recipe.base_revision):
        raise SourceError(f"{LICENCE_RECORD} records another base model than the recipe's")
    return rec


def _read_dev_check(root: Path, version: str) -> dict[str, Any] | None:
    d = DEV_CHECKS.get(version)
    if d is None or not (root / d.summary).is_file():
        return None
    bars_file, sha_file = root / d.bars, root / d.bars_sha256
    if not bars_file.is_file() or not sha_file.is_file():
        raise SourceError(f"{d.bars} or its recorded sha256 is missing")
    if _sha256(bars_file) != sha_file.read_text(encoding="utf-8").split()[0]:
        raise SourceError(f"{d.bars} does not match {d.bars_sha256}")
    bars = yaml.safe_load(bars_file.read_text(encoding="utf-8")) or {}
    summary = _read_json(root / d.summary)
    if summary.get("task") != bars.get("task"):
        raise SourceError(f"{d.summary} is not a run of the {bars.get('task')!r} task")
    return {
        "summary": summary,
        "bars": bars,
        "path": d.summary.as_posix(),
        "bars_path": d.bars.as_posix(),
        "decision": d.decision,
    }
