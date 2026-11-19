"""The inputs every release document is generated from (D-20).

The results bundle is the root. Files the bundle does not carry inline are read from the paths
it names (`summary_paths`) or from committed measurement files whose sha256 the bundle or a
committed lineage manifest records, so a number in the card can always be traced to a hashed
file. Nothing here is typed by hand: when v2 lands, E3.3 writes a new bundle and every document
is regenerated from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.schemas import RunSummary
from evalkit.schemas.summary import ModelSummary

from ggh.config import load_recipe
from ggh.lineage import committed_output, read_manifest, sha256_file
from ggh.schemas import DataManifest, RecipeConfig, ReleaseEntry, ResultsBundle, RunRef

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


class SourceError(ValueError):
    """A release input that is missing, unreadable or does not match its recorded hash."""


@dataclass(frozen=True)
class Paths:
    """Where each input lives, relative to the repo root. v2 may point any of them elsewhere."""

    bundle: Path = BUNDLE
    data_manifest: Path = DATA_MANIFEST
    targets_report: Path = TARGETS_REPORT
    targets_lineage: Path = TARGETS_LINEAGE
    recipe: Path = RECIPE
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

    data_file = root / p.data_manifest
    if not data_file.is_file() or sha256_file(data_file) != bundle.train.data_manifest_hash:
        raise SourceError(f"{p.data_manifest} is missing or not the manifest the run trained on")
    data = DataManifest.model_validate_json(data_file.read_text(encoding="utf-8"))
    raw["data"] = _read_json(data_file)

    recipe_file = root / p.recipe
    if not recipe_file.is_file() or sha256_file(recipe_file) != bundle.train.config_hash:
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
    return Sources(bundle, tuple(summaries), data, targets, recipe, p, raw)
