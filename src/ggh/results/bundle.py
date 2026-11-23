"""`ggh results bundle` (E3.3, RES-06, D-20): `results_bundle.json`, the one file the model card,
the README and the project page read their numbers from.

Inputs are committed files only (`configs/results.yaml` names them): EvalKit run folders under
docs/results/, the plans, the measurements (laptop bench, parity, probe report, train run and
selection, audits, export records, spend ledger) and the committed lineage manifests. A release
(v1, v2) is included when its primary run's summary exists, so v2 appears once V2.2's main run
is committed and nothing changes before then. Every number is copied from EvalKit's summary or
a committed measurement; the only computing done here is the behaviour table's B1, B3, B4 and B6
counts over EvalKit's records. Output is deterministic: the same inputs give the same bytes.

`publish_blockers` lists what must still happen before the card may be published (a missing or
invalid calibration input, owner reviews, sample read, lineage, a draft plan); it never stops
the bundle being written. The judges' calibration stamp is `ggh.results.calibration` (RES-03).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from evalkit.schemas import Interval

from ggh import ledger
from ggh.lineage import LineageError, StageManifest, committed_output, read_manifest
from ggh.results import calibration as C
from ggh.results import family as F
from ggh.results import sample as S
from ggh.results import table as T
from ggh.results.plan import Plan, load_plan
from ggh.results.runs import ResultsError, Run
from ggh.schemas import (
    AuditSummary,
    BehaviourRow,
    Comparison,
    ExportArtifact,
    LaptopBench,
    ModelResult,
    ParityReport,
    ProbeResult,
    ReleaseEntry,
    ResultsBundle,
    RunRef,
    SampleReadSummary,
    TrainRun,
    primary_key,
)

CONFIG = Path("configs") / "results.yaml"
OUT = Path("docs") / "results"
BUNDLE = "results_bundle.json"
TABLE_JSON = "behaviour_table.json"
TABLE_MD = "behaviour_table.md"
CITATION_METRICS = ("citation_precision", "aim_citation_rate")


@dataclass(frozen=True)
class Built:
    bundle: ResultsBundle
    behaviour_md: str


def _json(root: Path, rel: str) -> Any:
    path = root / rel
    if not path.is_file():
        raise ResultsError(f"{rel} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _opt_json(root: Path, rel: str | None) -> Any:
    return _json(root, rel) if rel and (root / rel).is_file() else None


def _iv(d: dict[str, Any]) -> Interval:
    return Interval(point=d["point"], lo=d["lo"], hi=d["hi"])


def _flip(d: dict[str, Any]) -> dict[str, Any]:
    return {"point": -d["point"], "lo": -d["hi"], "hi": -d["lo"]}


def _verdict(separated: bool, point: float, better: str = "higher") -> str:
    a_ahead = point > 0 if better == "higher" else point < 0
    return "tie" if not separated else ("a_better" if a_ahead else "b_better")


def comparison(c: dict[str, Any], *, plan: str, run: str, flip: bool = False) -> Comparison:
    a, b, diff = c["model_a"], c["model_b"], c["diff"]
    if flip:
        a, b, diff = b, a, _flip(diff)
    return Comparison.model_validate(
        {
            "plan": plan,
            "run": run,
            "model_a": a,
            "model_b": b,
            "metric": c["metric"],
            "judge": c["judge"],
            "diff": diff,
            "p_value": c.get("p_value"),
            "separated": c["separated"],
            "verdict": _verdict(c["separated"], diff["point"]),
        }
    )


def comparisons(
    run: Run, plan_rel: str, primary: list[tuple[str, str]], metric: str, judges: list[str]
) -> tuple[list[Comparison], list[Comparison]]:
    """The plan's primary comparisons (each pair, each judge, oriented as the plan writes it)
    and every other comparison in the run's summary as secondary."""
    rows = run.summary.get("comparisons", [])
    prim: list[Comparison] = []
    used: set[int] = set()
    for a, b in primary:
        for j in judges:
            hit = [
                (i, c)
                for i, c in enumerate(rows)
                if c["metric"] == metric
                and c["judge"] == j
                and {c["model_a"], c["model_b"]} == {a, b}
            ]
            if len(hit) != 1:
                raise ResultsError(f"{run.name}: no unique {metric} comparison {a} vs {b} ({j})")
            i, c = hit[0]
            used.add(i)
            prim.append(comparison(c, plan=plan_rel, run=run.name, flip=c["model_a"] != a))
    sec = [
        comparison(c, plan="secondary", run=run.name) for i, c in enumerate(rows) if i not in used
    ]
    return prim, sec


def plan_comparisons(run: Run, plan: Plan) -> tuple[list[Comparison], list[Comparison]]:
    """A plan with `primary_metrics` (V2.2, OWNER D-53) gets its own Holm family computed from
    the records; the E3.2 plan's comparisons are EvalKit's, as published. Either way every
    comparison in EvalKit's summary that is not primary is secondary."""
    d = plan.data
    if "primary_metrics" not in d:
        return comparisons(run, plan.path.as_posix(), plan.primary, d["metric"], list(d["judges"]))
    prim = [
        Comparison.model_validate(
            {
                **t,
                "diff": t["diff"].model_dump(),
                "plan": plan.path.as_posix(),
                "run": run.name,
                "method": "plan_family",
                "verdict": _verdict(t["separated"], t["diff"].point, t["better"]),
            }
        )
        for t in F.family_tests(run, d)
    ]
    reported = {tuple(x) for x in d.get("reported_not_primary", [])}
    sec = [
        comparison(c, plan="secondary", run=run.name, flip=(c["model_b"], c["model_a"]) in reported)
        for c in run.summary.get("comparisons", [])
    ]
    return prim, sec


def model_results(run: Run) -> list[ModelResult]:
    out = []
    for s in run.summary["models"]:
        out.append(
            ModelResult.model_validate(
                {
                    "run": run.name,
                    "model": s["model"],
                    "n": s["n"],
                    "accuracy": s["accuracy"],
                    "headline": s["headline"],
                    "groundedness_mean": s["groundedness_mean"],
                    "citation_validity": s["citation_validity"],
                    **{m: s[m] for m in CITATION_METRICS if s.get(m)},
                }
            )
        )
    return out


def run_ref(run: Run, extractor: str | None = None, exempt: str | None = None) -> RunRef:
    m, s = run.manifest, run.summary
    return RunRef(
        name=run.name,
        run_id=m["run_id"],
        task=m["task"],
        n=m["n"],
        models=tuple(m["models"]),
        judges=tuple(m["judges"]),
        judge_mode=s["judge_mode"],
        calibration_status=s["calibration_status"],
        judging_status=s["judging_status"],
        started_at=m["started_at"],
        spent_usd=round(float(m["spent_usd"]), 6),
        summary_sha256=run.summary_sha256,
        records_sha256=run.records_sha256,
        extractor=extractor,
        calibration_exempt=exempt,
    )


def _config_extractor(root: Path, rel: str | None) -> str | None:
    if not rel or not (root / rel).is_file():
        return None
    return (yaml.safe_load((root / rel).read_text(encoding="utf-8")) or {}).get("extractor")


def release_extractor(root: Path, plan: Plan) -> str | None:
    """The claim extractor every run of a release used: the plan's `extractor` (V2.2, OWNER
    D-57: GPT-6 Luna), else the one in the EvalKit config the plan names (E3.2:
    configs/evalkit.yaml, Claude Haiku 5.5). Every EvalKit config the plan names, its own and
    each run's, must agree with it."""
    d = plan.data
    configs = [d.get("config"), *(r.get("config") for r in d.get("runs") or [])]
    found = {c: e for c in configs if c and (e := _config_extractor(root, c)) is not None}
    want = d.get("extractor") or found.get(d.get("config")) or next(iter(found.values()), None)
    bad = sorted(c for c, e in found.items() if e != want)
    if bad:
        raise ResultsError(f"{plan.path}: extractor {want!r} differs from {', '.join(bad)}")
    return want


def release_bench(root: Path, cfg: dict[str, Any], rel: dict[str, Any]) -> tuple[LaptopBench, str]:
    """The release's own laptop bench: its `bench` file (else the top-level one) and its
    `bench_variant` (else the top-level `bench_shipped`, else ft-Q4_K_M)."""
    rel_file = rel.get("bench", cfg["bench"])
    variant = rel.get("bench_variant", cfg.get("bench_shipped", "ft-Q4_K_M"))
    variants = _json(root, rel_file)["variants"]
    if variant not in variants:
        raise ResultsError(f"{rel['version']}: {rel_file} has no bench variant {variant}")
    return LaptopBench.model_validate(variants[variant]), f"{rel_file}#{variant}"


def audits(root: Path, specs: list[dict[str, Any]]) -> list[AuditSummary]:
    out = []
    for a in specs:
        path = root / a["path"]
        if a["kind"] == "v1_labels":
            from ggh.targets import audit

            r = audit.score(root)
            ok = r["verdicts"].get("ok", 0)
            out.append(
                AuditSummary(
                    name=a["name"],
                    path=a["path"],
                    labelled=r["labelled"],
                    pass_rate=ok / r["labelled"] if r["labelled"] else 0.0,
                    passed=r["passed"],
                )
            )
        elif a["kind"] == "result":
            if not path.is_file():
                raise ResultsError(f"{a['path']} is missing")
            r = json.loads(path.read_text(encoding="utf-8"))
            n = r["labelled"]
            out.append(
                AuditSummary(
                    name=a["name"],
                    path=a["path"],
                    labelled=n,
                    pass_rate=r["overall"].get("ok", 0) / n if n else 0.0,
                    passed=r["passed"],
                )
            )
        else:
            raise ResultsError(f"audit kind {a['kind']!r} is unknown")
    return out


def committed_lineage_problems(root: Path, manifests: list[str]) -> list[str]:
    """The chain from each named committed manifest back to its roots, among the committed
    manifests in the same folder: every parent present once, every input a parent produced
    carrying that parent's hash, and every committed (non-artifacts/) output still matching.
    Outputs under artifacts/ are not on disk in CI; `--full-lineage` checks them."""
    problems: list[str] = []
    if not manifests:
        return problems
    folder = (root / manifests[0]).parent
    index: dict[str, StageManifest] = {}
    for p in sorted(folder.glob("*.json")):
        try:
            m = read_manifest(p)
        except LineageError as exc:
            problems.append(str(exc))
            continue
        index[m.key] = m
    for rel in manifests:
        if not (root / rel).is_file():
            problems.append(f"{rel} is missing")
            continue
        stack = [read_manifest(root / rel)]
        seen: set[str] = set()
        while stack:
            m = stack.pop()
            if m.key in seen:
                continue
            seen.add(m.key)
            for out, digest in m.outputs.items():
                if out.startswith("artifacts/"):
                    continue
                f = root / out
                if not f.is_file():
                    problems.append(f"{m.stage}: committed output {out} is missing")
                elif committed_output(root, out, digest) is None:
                    problems.append(f"{m.stage}: committed output {out} no longer matches")
            for parent in m.parents:
                pm = index.get(parent)
                if pm is None:
                    problems.append(f"{m.stage}: parent {parent[:12]} is not committed")
                    continue
                for inp, digest in m.inputs.items():
                    if inp in pm.outputs and pm.outputs[inp] != digest:
                        problems.append(f"{m.stage}: input {inp} differs from {pm.stage}'s output")
                stack.append(pm)
    return sorted(set(problems))


def full_lineage_problems(root: Path, manifests: list[str]) -> list[str]:
    """`ggh lineage verify` on the artifacts/ manifest matching each committed one."""
    from ggh.lineage import manifest_paths, verify

    problems = []
    # the same search `verify` uses: Kaggle output mirrors and nested checkouts are not lineage
    arts = root / "artifacts"
    found = {read_manifest(p).key: p for p in manifest_paths(root) if arts in p.parents}
    for rel in manifests:
        key = read_manifest(root / rel).key
        if key not in found:
            problems.append(f"{rel}: no artifacts/ manifest with key {key[:12]}")
            continue
        problems += verify(found[key], root)
    return problems


def snapshot_problems(
    run: Run, model: str, gguf: str, artifacts: list[ExportArtifact]
) -> list[str]:
    """The run's answers for the release came from the GGUF the release's export recorded."""
    snaps = {
        r["completion"]["model_snapshot"]
        for r in run.raw_records
        if r["completion"]["model"] == model
    }
    p = []
    if snaps != {gguf}:
        p.append(f"{run.name}: {model} answered from {sorted(snaps)}, not {gguf}")
    if not any(a.path == gguf and a.kind == "gguf" for a in artifacts):
        p.append(f"{gguf} is not in the release's export records")
    return p


# Ledger rows written before the ledger recorded the EvalKit commit (E3.1, E3.2) all ran on the
# D-28 git pin; later rows name their commit.
LEGACY_EVALKIT = "06591f8de9b178078c971940a83019a6530fd119"


def _evalkit_versions(root: Path, run_ids: set[str]) -> str:
    from ggh.evals.config import pinned_commit

    pin = pinned_commit(root)
    rows = [r for r in _ledger_rows(root) if r.get("run_id") in run_ids]
    shas = sorted({str(r.get("evalkit", LEGACY_EVALKIT))[:12] for r in rows} or {pin[:12]})
    return ", ".join(shas)


def _in_scope(cfg: dict[str, Any]) -> Callable[[dict], bool] | None:
    """Ledger rows outside the bundle: nodes whose id starts with a listed prefix (Wave 7)."""
    prefixes = tuple(cfg.get("ledger_exclude_node_prefixes") or ())
    if not prefixes:
        return None
    return lambda row: not str(row.get("node", "")).startswith(prefixes)


def _ledger_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "docs/measurements/spend-ledger.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build(
    root: Path,
    *,
    full_lineage: bool = False,
    config: Path = CONFIG,
    versions: tuple[str, ...] | None = None,
) -> Built:
    """`versions` limits the bundle to those releases (the v1 release fixture is v1 alone);
    by default every release whose primary run is committed is in it."""
    cfg = yaml.safe_load((root / config).read_text(encoding="utf-8"))
    included = [
        r
        for r in cfg["releases"]
        if Run.exists(root, r["primary_run"]) and (versions is None or r["version"] in versions)
    ]
    if not included:
        raise ResultsError("no release has its primary run under docs/results/")
    runs: dict[str, Run] = {}
    for rel in included:
        for name in rel["runs"]:
            if not Run.exists(root, name):
                raise ResultsError(f"{rel['version']}: run {name} has no results yet")
            runs.setdefault(name, Run(root, name))
    releases: list[ReleaseEntry] = []
    primary: list[Comparison] = []
    secondary: list[Comparison] = []
    behaviour: list[BehaviourRow] = []
    lineage_problems: list[str] = []
    blockers: list[str] = []
    extractors: dict[str, str | None] = {}
    review = T.load_b8_review(root / cfg["b8_review"])
    reviewers = T.load_b8_reviewers(root / cfg["b8_review"])
    sample_sums: list[SampleReadSummary] = []
    cal = C.stamp(root, cfg.get("calibration"), runs, cfg.get("gemma_family_check"))
    blockers += cal.problems
    for name in sorted(runs):
        sums, probs = S.summarise(root, runs[name])
        sample_sums += sums
        blockers += probs
    for rel in included:
        plan = load_plan(root, rel["plan"])
        extractor = release_extractor(root, plan)
        extractors.update(dict.fromkeys(rel["runs"], extractor))
        main = runs[rel["primary_run"]]
        p, s = plan_comparisons(main, plan)
        relied = sorted({c.judge for c in p if c.judge})
        primary += p
        secondary += s
        cit = runs[rel["citation_run"]] if rel["citation_run"] != rel["behaviour_run"] else None
        brun = runs[rel["behaviour_run"]]
        behaviour += T.run_rows(brun, citation=cit)
        behaviour += T.b5_rows(sample_sums, brun.models(), brun.name)
        off = runs[rel["offscope_run"]]
        behaviour += T.b8_rows(off, review, reviewers=reviewers)
        artifacts = [ExportArtifact.model_validate(a) for a in _json(root, rel["artifacts"])]
        lineage_problems += committed_lineage_problems(root, rel["lineage"])
        if full_lineage:
            lineage_problems += full_lineage_problems(root, rel["lineage"])
        lineage_problems += snapshot_problems(main, rel["model"], rel["gguf"], artifacts)
        probe = _opt_json(root, rel.get("probe"))
        pm = (probe or {}).get("models", {}).get(rel["model"])
        ph = _opt_json(root, rel.get("parity_headline"))
        parity = _opt_json(root, rel.get("parity"))
        if parity is not None and ph is not None:
            parity["judged_headline_delta"] = ph["bf16_minus_q4_headline"]
        selection = _opt_json(root, rel.get("train_selection")) or {}
        bench, bench_source = release_bench(root, cfg, rel)
        if plan.status != "final":
            blockers.append(f"{rel['version']}: plan {plan.path} is {plan.status}")
        releases.append(
            ReleaseEntry.model_validate(
                {
                    "version": rel["version"],
                    "model": rel["model"],
                    "plan": plan.path.as_posix(),
                    "plan_sha256": plan.sha256 if plan.status == "final" else None,
                    "plan_status": plan.status,
                    "built_after_test_seen": rel["built_after_test_seen"],
                    "runs": rel["runs"],
                    "train": TrainRun.model_validate(_json(root, rel["train_run"])),
                    "adapter_sha256": selection.get("adapter_sha256"),
                    "audits": [a.model_dump() for a in audits(root, rel["audits"])],
                    "parity": ParityReport.model_validate(parity) if parity else None,
                    "probe": ProbeResult.model_validate({"model": rel["model"], **pm})
                    if pm
                    else None,
                    "artifacts": [a.model_dump() for a in artifacts],
                    "calibration_judges": relied,
                    "calibration_status": cal.release_status(relied),
                    "bench": bench.model_dump(),
                    "bench_source": bench_source,
                }
            )
        )
    ordered = [runs[n] for n in sorted(runs)]
    refs = [
        run_ref(r, extractors.get(r.name), cal.exempt_note(r.manifest["task"])) for r in ordered
    ]
    # the bundle is calibrated only when every release is (a release only when every judge
    # its primary comparisons rely on is stamped, OWNER D-59/D-60); the summaries' own stamps
    # stay as EvalKit wrote them. A judge under the bar is disclosed, not a blocker.
    calibrated = all(r.calibration_status == "calibrated" for r in releases)
    gfc = _opt_json(root, cfg.get("gemma_family_check"))
    for r in refs:
        if r.judging_status != "complete":
            blockers.append(f"{r.name}: judging is {r.judging_status}")
    for row in behaviour:
        if row.value is None and row.source in ("owner_review", "model_review"):
            blockers.append(f"{row.run}: {row.rule} {row.metric} for {row.model} is not reviewed")
    shipped = included[-1]
    bench_raw = _json(root, shipped.get("bench", cfg["bench"]))["variants"]
    bundle = ResultsBundle.model_validate(
        {
            "evalkit_run_ids": [r.run_id for r in refs],
            "evalkit_version": _evalkit_versions(root, {r.run_id for r in refs}),
            "judge_mode": "dual" if all(r.judge_mode == "dual" for r in refs) else "single",
            "calibration_status": "calibrated" if calibrated else "uncalibrated",
            "gemma_family_check": gfc if calibrated else None,
            "calibration": [c.model_dump() for c in cal.judges],
            "labeller_consistency": cal.labeller.model_dump() if cal.labeller else None,
            "summary_paths": [r.summary_rel for r in ordered],
            "runs": [r.model_dump() for r in refs],
            "releases": [r.model_dump() for r in releases],
            "results": [m.model_dump() for r in ordered for m in model_results(r)],
            "primary": {primary_key(c): c.diff.model_dump() for c in primary},
            "primary_comparisons": [c.model_dump() for c in primary],
            "secondary": [c.model_dump() for c in secondary],
            "behaviour": [b.model_dump() for b in behaviour],
            "bench": releases[-1].bench,
            "bench_variants": {
                k: LaptopBench.model_validate(v).model_dump() for k, v in sorted(bench_raw.items())
            },
            "sample_read": [s.model_dump() for s in sample_sums],
            "spend_usd": round(ledger.total(root / cfg["ledger"], "spent_usd", _in_scope(cfg)), 6),
            "lineage_ok": not lineage_problems,
            "lineage_scope": "full" if full_lineage else "committed",
            "lineage_problems": sorted(set(lineage_problems)),
            "publish_blockers": sorted(set(blockers))
            + sorted(f"lineage: {p}" for p in set(lineage_problems)),
        }
    )
    return Built(bundle, T.markdown(bundle.behaviour))


def write(root: Path, built: Built, out: Path = OUT) -> list[Path]:
    folder = root / out
    folder.mkdir(parents=True, exist_ok=True)
    b = built.bundle
    files = {
        BUNDLE: b.model_dump_json(indent=2) + "\n",
        TABLE_JSON: json.dumps([r.model_dump(mode="json") for r in b.behaviour], indent=2) + "\n",
        TABLE_MD: built.behaviour_md,
    }
    paths = []
    for name, text in files.items():
        p = folder / name
        p.write_text(text, encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def digest(built: Built) -> str:
    return hashlib.sha256(built.bundle.model_dump_json(indent=2).encode()).hexdigest()
