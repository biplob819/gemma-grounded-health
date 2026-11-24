"""OSS-04: the GitHub release notes, generated from the results bundle (D-20).

`ggh release-notes` writes `docs/release_notes_v<version>.md` for the bundle's shipped release
(`NOTES`: v2 is v1.0.0, v3b is v1.1.0, V3.8); the builder pastes it as the body of the GitHub
release when tagging. v1.0.0's notes stay reproducible from its frozen bundle
(docs/results/releases/v1.0.0/, D-76). Every number comes from the bundle (the shipped release's
table and its pre-registered comparisons, with the OWNER D-62 caveat), every artifact and
sha256 from the Hub release plan (`docs/measurements/release_plan_v2.json`), compared with
what the R4.2 push downloaded back (`release_record_v2.json`), and every known limit from
`docs/known-issues.md`. Links are absolute, since the release page is read outside the repo.
The notes are those of the published v1.0.0 release: a link added to the registry after it (the
P5.1 write-up, `links.AFTER_RELEASE`) renders as it did then, so regenerating never changes the
file the release used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ggh import __version__
from ggh.release import links
from ggh.release.card import (
    outcome_table,
    primary_table,
    q8_recommendation,
    release_caveats,
    scoring,
)
from ggh.release.readme import disclosures
from ggh.release.sources import Sources

OUT = Path("docs") / f"release_notes_v{__version__}.md"
PLAN = Path("docs") / "measurements" / "release_plan_v2.json"
RECORD = Path("docs") / "measurements" / "release_record_v2.json"
KNOWN_ISSUES = Path("docs") / "known-issues.md"
# the release path's nodes; earlier nodes' open rows are build-harness notes, not model limits
RELEASE_NODES = ("V2.1", "V2.2", "E3.3", "R4.1", "R4.2", "R4.3", "R4.4", "R4.5")
LIMIT_STATUS = ("Open", "Disclosed", "Owner-waived")


@dataclass(frozen=True)
class NotesSpec:
    """One tagged release's notes: its GGH version, Hub plan and push record, the nodes whose
    known-issue rows are its limits, and its opening paragraph."""

    version: str
    plan: Path
    record: Path
    nodes: tuple[str, ...]
    intro: str  # str.format fields: release, model, hf_model, hf_gguf, reproduce, tag


INTRO_V1_0_0 = (
    "The first tagged release. It ships the {release} fine-tune "
    "(`{model}`): Gemma 4 E2B with LoRA, answering a research question only from "
    "the abstract it is given, as JSON with a yes, no or maybe label, a short answer and "
    "the numbers of the sentences it relied on. Weights: {hf_model} "
    "(merged safetensors and the LoRA adapter) and {hf_gguf} (Q4_K_M, the "
    "benchmarked file, and Q8_0). Reproduce it from a fresh clone with "
    "{reproduce}."
)
INTRO_V1_1_0 = (
    "The second tagged release (OWNER D-75). It ships the {release} fine-tune (`{model}`): "
    "v2's Gemma 4 E2B LoRA adapter trained further by GRPO reinforcement learning on v2's "
    "training prompts and gold labels, answering a research question only from the abstract "
    "it is given, as JSON with a yes, no or maybe label, a short answer and the numbers of the "
    "sentences it relied on. Weights: {hf_model} (merged safetensors and the LoRA adapter) and "
    "{hf_gguf} (Q4_K_M, the registered and benchmarked file, and Q8_0, recommended where it "
    "fits), Hub tag `{tag}`. Reproduce it from a fresh clone with {reproduce}."
)
MEASURE = Path("docs") / "measurements"
NOTES: dict[str, NotesSpec] = {
    "v2": NotesSpec("1.0.0", PLAN, RECORD, RELEASE_NODES, INTRO_V1_0_0),
    "v3b": NotesSpec(
        "1.1.0",
        MEASURE / "release_plan_v3b.json",
        MEASURE / "release_record_v3b.json",  # written by the push, after the owner's go-ahead
        ("V3.3", "V3.4", "V3.5", "V3.6", "V3.7", "V3.8"),
        INTRO_V1_1_0,
    ),
}


def spec_for(src: Sources) -> NotesSpec:
    v = src.bundle.shipped.version
    if v not in NOTES:
        raise ValueError(f"no release notes are registered for {v}; known: {', '.join(NOTES)}")
    return NOTES[v]


def out_path(src: Sources) -> Path:
    """`docs/release_notes_v<version>.md` for the bundle's shipped release."""
    return Path("docs") / f"release_notes_v{spec_for(src).version}.md"


@dataclass(frozen=True)
class Issue:
    node: str
    finding: str
    status: str


def known_limits(text: str, nodes: tuple[str, ...] = RELEASE_NODES) -> list[Issue]:
    """Rows of the known-issues table from the release path whose status is still open,
    disclosed or owner-waived (accepted and fixed rows are left out)."""
    out: list[Issue] = []
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 5 or cells[0] not in nodes:
            continue
        node, finding, _, _, status = cells
        if status.startswith(LIMIT_STATUS):
            out.append(Issue(node, finding, status))
    return out


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000
    return f"{size} B"  # pragma: no cover


def artifacts(plan: dict[str, Any], record: dict[str, Any] | None) -> str:
    """Every file of both Hub repos with its planned sha256, and whether the Hub holds those
    bytes (the R4.2 download-back) or awaits the R4.5 card re-upload. With no push record yet
    (a release prepared to its dry run, V3.8), every file awaits the push."""
    if record is None:
        return _unpushed(plan)
    published = {d["file"]: d["sha256"] for d in record.get("downloads", []) if d.get("ok")}
    rows = []
    for f in plan["files"]:
        key = f"{f['repo']}/{f['path_in_repo']}"
        hub = "published" if published.get(key) == f["sha256"] else "re-upload pending"
        rows.append(f"| `{key}` | {_human(f['size'])} | `{f['sha256']}` | {hub} |")
    pending = sum(r.endswith("re-upload pending |") for r in rows)
    head = (
        f"Release `{plan['release']}` (`{plan['model']}`), {len(plan['files'])} files, "
        f"{plan['total_bytes']:,} bytes, lineage verified from `{plan['lineage_verified_from']}` "
        f"({len(plan['lineage_chain'])} stages), plan problems: {len(plan['problems'])}. "
        f'"published" means the file on the Hub was downloaded back with this sha256 '
        f"(tag `{record.get('tag', '?')}`)"
    )
    if pending:
        head += (
            f"; {pending} regenerated card(s) are re-uploaded at this release (known issue R4.2)"
        )
    return (
        head + ".\n\n| File | Size | sha256 | Hub |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def _unpushed(plan: dict[str, Any]) -> str:
    rows = [
        f"| `{f['repo']}/{f['path_in_repo']}` | {_human(f['size'])} | `{f['sha256']}` | "
        "push pending |"
        for f in plan["files"]
    ]
    tag = plan.get("tag", f"{plan['release']}.0.0")
    head = (
        f"Release `{plan['release']}` (`{plan['model']}`), {len(plan['files'])} files, "
        f"{plan['total_bytes']:,} bytes, lineage verified from `{plan['lineage_verified_from']}` "
        f"({len(plan['lineage_chain'])} stages), plan problems: {len(plan['problems'])}. "
        f"The Hub push under tag `{tag}` waits for the owner's go-ahead; each file is then "
        "downloaded back and checked against this sha256"
    )
    return (
        head + ".\n\n| File | Size | sha256 | Hub |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def results(src: Sources) -> str:
    main, b = src.main(), src.bundle
    ref = src.ref(main)
    n = max((m.n for m in main.models), default=0)
    caveats = "; ".join(disclosures(src))
    return (
        f"{b.shipped.version} on {n} expert-labelled PubMedQA questions (EvalKit run "
        f"`{main.run_id}`, {scoring(ref)}, calibration of {b.shipped.version}: "
        f"{b.shipped.calibration_status}); every model in this table was scored in that run. "
        "95% intervals in square brackets.\n\n"
        + outcome_table(src, ref, bold=src.ft_model)
        + "\n\n### Pre-registered comparisons\n\n"
        + primary_table(src).replace(
            "So this card does not claim", "So this release does not claim"
        )
        + f"\n\nCaveats: {caveats}. Earlier releases' published numbers stand under their own "
        "plans and are not compared here; they are on the model card and in the bundle."
        + "".join(f"\n\n{x}" for x in q8_recommendation(src))
    )


def render(src: Sources, root: Path) -> str:
    spec = spec_for(src)
    plan = json.loads((root / spec.plan).read_text(encoding="utf-8"))
    rec_file = root / spec.record
    record = json.loads(rec_file.read_text(encoding="utf-8")) if rec_file.is_file() else None
    issues = known_limits((root / KNOWN_ISSUES).read_text(encoding="utf-8"), spec.nodes)
    shipped = src.bundle.shipped
    limit_lines = "\n".join(f"- **{i.node}.** {i.finding} *Status:* {i.status}" for i in issues)
    caveats = release_caveats(src)
    if caveats:  # the model card's own disclosures for this release (V3.7, v3b)
        limit_lines = "\n".join(x for x in (limit_lines, *caveats) if x)
    v = spec.version
    link_lines = "\n".join(
        f"- {links.md_abs(name, as_released=True, version=v)}"
        for name in links.links_at(v)
        if name != "release_notes"
    )
    intro = spec.intro.format(
        release=shipped.version,
        model=shipped.model,
        hf_model=links.md_abs("hf_model", version=v),
        hf_gguf=links.md_abs("hf_gguf", version=v),
        reproduce=links.md_abs("reproduce", version=v),
        tag=plan.get("tag", f"{shipped.version}.0.0"),
    )
    return (
        f"# Gemma Grounded Health v{v}\n\n"
        "<!-- generated by `ggh release-notes` from the results bundle, the Hub release plan "
        "and docs/known-issues.md; do not edit -->\n\n"
        "> **Research only, not medical advice.** The model summarises what a study abstract "
        "says. It is not guidance for anyone's diagnosis or treatment.\n\n"
        f"{intro}\n\n"
        f"## Results\n\n{results(src)}\n\n"
        f"## Artifacts\n\n{artifacts(plan, record)}\n"
        f"## Known limits\n\nFrom {links.md_abs('known_issues', version=v)} (the release "
        "path's open, disclosed and owner-waived rows; the model card's Limits section has the "
        f"rest).\n\n{limit_lines}\n\n"
        f"## Links\n\n{link_lines}\n"
    )
