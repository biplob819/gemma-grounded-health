"""CRD-01..06: MODEL_CARD.md generated from the results bundle (Tech Spec §10.1, D-20).

The template (`templates/model_card.md`) holds the prose and the section layout only; every
number is formatted here from `Sources`, so a new bundle (v2) regenerates the card with no edit.
The Hub metadata front matter (`license`, `base_model`, `datasets`, `language`,
`pipeline_tag`, `tags`, `model-index`) is built from the same inputs and checked by
`metadata_problems`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Any

import yaml
from evalkit.pipeline.aggregate import stamp_lines
from evalkit.schemas import Interval, RunSummary

from ggh.release.sources import AIM_CHECK, LICENCE_RECORD, RESULTS, Sources
from ggh.schemas import (
    BehaviourRow,
    Comparison,
    LaptopBench,
    ReleaseEntry,
    RLRun,
    RunRef,
    TrainRun,
)

TEMPLATE = Path(__file__).parent / "templates" / "model_card.md"
HUB_LICENSE = "apache-2.0"
LANGUAGE = ("en",)
PIPELINE_TAG = "text-generation"
TAGS = ("gemma", "medical", "biomedical", "pubmedqa", "grounded-generation", "citations", "lora")
MODEL_NAME = "gemma-4-e2b-grounded-health"
BANNER = "Research only, not medical advice"
BASE_MODEL_RELATION = "finetune"
# the decision under which each release's parity miss was disclosed rather than gated
PARITY_DECISIONS = {"v2": "OWNER D-55", "v3b": "OWNER D-73"}
# releases whose card summary leads with the pre-registered headline result (OWNER D-75); the
# earlier cards are frozen as published (D-76)
HEADLINE_LEAD = frozenset({"v3b"})
REQUIRED_META = (
    "license",
    "base_model",
    "datasets",
    "language",
    "pipeline_tag",
    "tags",
    "model-index",
)


def f3(x: float) -> str:
    return f"{x:.3f}"


def iv(i: Interval) -> str:
    return f"{i.point:.3f} [{i.lo:.3f}, {i.hi:.3f}]"


def _neg(text: str) -> str:
    if text.lstrip("-0.") == "":
        return text.lstrip("-")
    return text[1:] if text.startswith("-") else "-" + text


def neg_iv(i: Interval) -> str:
    """b minus a from an a minus b interval, negating the printed digits so the card shows
    exactly the bundle's rounding (no new float is formatted)."""
    return f"{_neg(f3(i.point))} [{_neg(f3(i.hi))}, {_neg(f3(i.lo))}]"


def value(v: Interval | float | None) -> str:
    if v is None:
        return "–"
    if isinstance(v, Interval):
        return iv(v)
    return f3(v)


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def judges_of(summary: RunSummary) -> list[str]:
    return sorted({j for m in summary.models for j in m.headline})


def model_table(summary: RunSummary) -> str:
    judges = judges_of(summary)
    header = [
        "Model",
        "n",
        "Accuracy",
        *[f"Correct and grounded ({j})" for j in judges],
        "Citation validity",
    ]
    rows = [
        [
            m.model,
            str(m.n),
            iv(m.accuracy),
            *[iv(m.headline[j]) if j in m.headline else "–" for j in judges],
            iv(m.citation_validity) if m.citation_validity else "–",
        ]
        for m in summary.models
    ]
    return (
        f"EvalKit task `{summary.task}`, run `{summary.run_id}`. 95% bootstrap intervals in "
        "square brackets. The headline counts an answer only when its label is correct and "
        "every claim is supported by the sentences it cites.\n\n" + table(header, rows)
    )


def scoring(ref: RunRef) -> str:
    """Who produced a run's claim-level numbers: its extractor and its judges."""
    judges = " and ".join(f"`{j}`" for j in ref.judges)
    head = f"claims extracted by `{ref.extractor}`, " if ref.extractor else ""
    return f"{head}judged by {judges}" + (" alone" if len(ref.judges) == 1 else "")


def _get(d: Mapping[str, Interval] | None, j: str) -> str:
    return iv(d[j]) if d and j in d else "–"


def outcome_table(src: Sources, ref: RunRef, bold: str | None = None) -> str:
    """One run's models from the bundle: accuracy, the headline per judge, and citation
    precision and the aim-citation rate per judge where the run measured them."""
    rows = [r for r in src.bundle.results if r.run == ref.name]
    hj = sorted({j for r in rows for j in r.headline})
    cj = sorted({j for r in rows for j in (r.citation_precision or {})})
    aj = sorted({j for r in rows for j in (r.aim_citation_rate or {})})
    header = [
        "Model",
        "n",
        "Accuracy",
        *[f"Correct and grounded ({j})" for j in hj],
        *[f"Citation precision ({j})" for j in cj],
        *[f"Aim-citation rate ({j}, lower is better)" for j in aj],
    ]
    body = [
        [
            f"**{r.model}**" if r.model == bold else r.model,
            str(r.n),
            iv(r.accuracy),
            *[_get(r.headline, j) for j in hj],
            *[_get(r.citation_precision, j) for j in cj],
            *[_get(r.aim_citation_rate, j) for j in aj],
        ]
        for r in rows
    ]
    return table(header, body)


def _release_refs(src: Sources, release: ReleaseEntry) -> list[RunRef]:
    """The release's benchmark run and, if separate, its citation run."""
    refs = [src.ref(src.release_main(release))]
    cit = src.citation_run(release)
    if cit is not None and cit.name != refs[0].name:
        refs.append(cit)
    return refs


def earlier_tables(src: Sources, heading: str = "### ") -> list[str]:
    """Each earlier release's published benchmark numbers, under their own extractor and judges
    and labelled as such. They stand as published and are never compared with the shipped
    release (OWNER D-57: models not re-scored for v2 are shown, not compared)."""
    shipped = src.bundle.shipped
    now = src.ref(src.main())
    scored_now = set(now.models)
    out: list[str] = []
    for rel in src.earlier():
        refs = _release_refs(src, rel)
        v, s = rel.version, shipped.version
        out += [f"{heading}Published {v} numbers (not compared with {s})", ""]
        notes = [
            f"These are {v}'s numbers as published under `{rel.plan}`; they stand as they are "
            f"and are shown for reference, not as a {s} comparison."
        ]
        differs = [
            what
            for what, bad in (
                ("claim extractor", any(r.extractor != now.extractor for r in refs)),
                ("judge set", any(r.judges != now.judges for r in refs)),
            )
            if bad
        ]
        if differs:
            notes.append(
                f"They were produced with a different {' and a different '.join(differs)} from "
                f"{s}'s run ({scoring(now)}), so the two sets of numbers are not comparable."
            )
        dropped = sorted({m for r in refs for m in r.models} - scored_now)
        if dropped:
            names = " and ".join(f"`{m}`" for m in dropped)
            notes.append(
                f"As answering models, {names} were not re-scored for {s}, so {s} is not "
                "compared with them."
            )
        both = sorted({m for r in refs for m in r.models} & scored_now)
        if both:
            names = " and ".join(f"`{m}`" for m in both)
            notes.append(
                f"{names} appear in both {s}'s table and here; each table's numbers come from "
                "its own run."
            )
        out += [" ".join(notes), ""]
        for r in refs:
            out += [f"Run `{r.name}` (`{r.run_id}`), {scoring(r)}:", ""]
            out += [outcome_table(src, r, bold=rel.model), ""]
    return out[:-1] if out else out


def benchmark(src: Sources) -> str:
    """The shipped release's benchmark. With one release this is the EvalKit summary table
    (v1's card, unchanged); with earlier releases in the bundle, the shipped run is labelled
    with its extractor and judges and each earlier release's numbers follow under their own."""
    main = src.main()
    if not src.earlier():
        return model_table(main)
    ref, shipped = src.ref(main), src.bundle.shipped
    lines = [
        f"{shipped.version}: EvalKit task `{main.task}`, run `{main.run_id}`. Every model in this "
        f"table was scored in this run: {scoring(ref)}. 95% bootstrap intervals in square "
        "brackets. The headline counts an answer only when its label is correct and every claim "
        "is supported by the sentences it cites.",
        "",
    ]
    for r in _release_refs(src, shipped):
        if r.name != ref.name:
            lines += [f"Run `{r.name}` (`{r.run_id}`), {scoring(r)}:", ""]
        lines += [outcome_table(src, r, bold=src.ft_model), ""]
        if r.name == ref.name and exploratory(src):
            lines += [exploratory_note(src), ""]
    return "\n".join([*lines, *earlier_tables(src)])


def exploratory(src: Sources) -> list[str]:
    """Models the shipped run scored outside the pre-registered test (v3b at Q8_0, OWNER D-73),
    as the release's result file lists them."""
    return list((src.result or {}).get("exploratory_models") or [])


def model_label(src: Sources, model: str) -> str:
    """A model as the summary names it: a release, the shipped release's Q8_0, or its id."""
    rel = next((r for r in src.bundle.releases if r.model == model), None)
    if rel is not None:
        return f"{rel.version} (`{model}`)"
    if model == src.q8_model:
        tail = ", exploratory" if model in exploratory(src) else ""
        return f"{src.bundle.shipped.version} at Q8_0 (`{model}`{tail})"
    return f"`{model}`"


def exploratory_note(src: Sources) -> str:
    v = src.bundle.shipped.version
    names = " and ".join(f"`{m}`" for m in exploratory(src))
    what = f"{v} at Q8_0" if exploratory(src) == [src.q8_model] else "an extra model"
    return (
        f"{names} is {what}, scored in the same run as an exploratory model outside the "
        f"pre-registered test (OWNER D-73): no {v} claim rests on it."
    )


# model ids as reports name them (OWNER D-60, D-61: say which model labelled or reviewed)
DISPLAY = {"claude-opus-5-5": "Claude Opus 5.5", "gpt-6.1-sol": "GPT-6.1 Sol"}


def display(model_or_note: str) -> str:
    """ "claude-opus-5-5 (blind subagent, ...)" -> "Claude Opus 5.5"."""
    head = model_or_note.split(" ", 1)[0]
    return DISPLAY.get(head, head)


def evalkit_stamps(src: Sources) -> list[str]:
    """EvalKit's caveats for the shipped run. A release GGH stamps calibrated (OWNER D-59)
    drops EvalKit's per-run "not calibrated" line: the committed summary keeps it (summaries
    are never rewritten), but it is not the release's status."""
    main = src.main()
    if src.bundle.shipped.calibration_status == "calibrated":
        main = main.model_copy(update={"calibration_status": "calibrated"})
    return stamp_lines(main)


def calibration_lines(src: Sources) -> list[str]:
    """The shipped release's stamp and each judge's calibration, worded as the bundle words it
    (OWNER D-59, D-60): the reference labels are a model's, the labeller's own model is
    reported and never stamped."""
    b, rel = src.bundle, src.bundle.shipped
    v = rel.version
    if not b.calibration:
        return [f"- Calibration: **{b.calibration_status}**."]
    relied = " and ".join(f"`{j}`" for j in rel.calibration_judges) or "no judge"
    lines = [
        f"- Calibration of {v}: **{rel.calibration_status}**. The stamp covers the judges "
        f"{v}'s pre-registered comparisons rely on ({relied}) and is given only when each "
        "is calibrated (OWNER D-59). Per judge, as the results bundle records it:"
    ]
    judges_now = set(src.ref(src.main()).judges)
    order = sorted(b.calibration, key=lambda c: (c.judge not in rel.calibration_judges, c.judge))
    for c in order:
        note = "" if c.judge in judges_now else f" (not a judge of {v}'s runs)"
        lines.append(f"  - {c.wording}{note}.")
    lc = b.labeller_consistency
    if lc is not None:
        lines.append(
            f"- Reference labeller consistency: {display(lc.labeller)} relabelled {lc.answers} "
            f"answers ({lc.claims} claims) under `{lc.prompt}`: {lc.identical} labels "
            f"identical, kappa {iv(lc.kappa)} (OWNER D-60)."
        )
    if b.gemma_family_check and not any(c.gemma_family for c in b.calibration):
        checks = ", ".join(f"{j} {f3(a)}" for j, a in sorted(b.gemma_family_check.items()))
        lines.append(
            "- Gemma-family judge check (agreement with the reference labels on claims from "
            f"Gemma answers, OWNER D-24, D-60): {checks}."
        )
    return lines


def stamps(src: Sources) -> str:
    b = src.bundle
    main = src.main()
    ref = next((r for r in b.runs if r.run_id == main.run_id), None)
    mode = ref.judge_mode if ref is not None else b.judge_mode
    lines = [f"- Judge mode: **{mode}**."]
    if ref is not None and src.earlier():
        lines.append(f"- {b.shipped.version} scoring: {scoring(ref)}.")
    lines += calibration_lines(src)
    lines += [f"- ⚠ {s}" for s in evalkit_stamps(src)]
    if b.shipped.calibration_status == "calibrated" and main.calibration_status != "calibrated":
        lines.append(
            "- The committed EvalKit summaries keep EvalKit's own per-run stamp (\"judge not "
            "calibrated for this task\"): they are never rewritten (OWNER D-59). The release's "
            "stamp above is GGH's, read from EvalKit's calibration run."
        )
    own = set(b.shipped.runs)
    for r in b.runs:
        if r.name in own and r.calibration_exempt:
            lines.append(f"- `{r.name}`: {r.calibration_exempt}.")
    return "\n".join(lines)


def _verdict(c: Comparison, note: str = "") -> str:
    if c.verdict == "tie":
        return f"tie (not separated{'; ' + note if note else ''})"
    who = c.model_a if c.verdict == "a_better" else c.model_b
    return f"{who} better" + (f" ({note})" if note else "")


METRIC_NAME = {
    "headline": "correct and fully grounded",
    "citation_precision": "citation precision",
    "aim_citation_rate": "aim-citation rate",
}
AIM_NOTE = "depends on the aim rule, OWNER D-62"


def _ckey(model_a: str, model_b: str, metric: str, judge: str | None) -> tuple[str, ...]:
    return (model_a, model_b, metric, judge or "")


def aim_corrected(src: Sources) -> dict[tuple[str, ...], Comparison]:
    """OWNER D-62's post-hoc family for the shipped release's primary run, as comparisons."""
    if src.aim_check is None:
        return {}
    run = src.release_main(src.bundle.shipped)
    name = src.ref(run).name
    rows = ((src.aim_check.get("family") or {}).get(name) or {}).get("corrected") or []
    out: dict[tuple[str, ...], Comparison] = {}
    for t in rows:
        c = Comparison.model_validate(
            {
                **t,
                "plan": src.bundle.shipped.plan,
                "run": name,
                "method": "plan_family",
                "verdict": _verdict_of(t["separated"], t["diff"]["point"], t["better"]),
            }
        )
        out[_ckey(c.model_a, c.model_b, c.metric, c.judge)] = c
    return out


def _verdict_of(separated: bool, point: float, better: str) -> str:
    a_ahead = point > 0 if better == "higher" else point < 0
    return "tie" if not separated else ("a_better" if a_ahead else "b_better")


def aim_dependent(src: Sources) -> set[tuple[str, ...]]:
    """The primary comparisons whose verdict changes under the post-hoc count (OWNER D-62)."""
    corrected = aim_corrected(src)
    return {
        k
        for c in src.bundle.primary_comparisons
        if (k := _ckey(c.model_a, c.model_b, c.metric, c.judge)) in corrected
        and corrected[k].verdict != c.verdict
    }


def aim_outcome(src: Sources) -> tuple[list[str], list[str]]:
    """Under the post-hoc count: the citation outcomes on which base beats the shipped model,
    and the other opponents whose every verdict is unchanged (OWNER D-62)."""
    corrected, flips = aim_corrected(src), aim_dependent(src)
    shipped = src.bundle.shipped
    base, ft = src.paths.base_model, shipped.model
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    worse = [
        METRIC_NAME.get(c.metric, c.metric)
        for c in corrected.values()
        if c.model_a == ft
        and c.model_b == base
        and c.metric != "headline"
        and c.verdict == "b_better"
    ]
    others = list(dict.fromkeys(c.model_b for c in comps if c.model_b != base))
    held = [
        m
        for m in others
        if all(
            _ckey(c.model_a, c.model_b, c.metric, c.judge) not in flips
            for c in comps
            if c.model_b == m
        )
    ]
    return worse, held


def aim_rule(src: Sources) -> str:
    """OWNER D-62: the shipped release's citation results against base depend on how the judge
    applies the citation rubric's aim rule; the post-hoc count and what it changes."""
    aim, corrected = src.aim_check, aim_corrected(src)
    if aim is None or not corrected:
        return ""
    shipped = src.bundle.shipped
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    flips = aim_dependent(src)
    base, ft = src.paths.base_model, shipped.model
    run = src.ref(src.release_main(shipped)).name
    cites = (aim.get("citation") or {}).get(run) or {}
    moved = []
    for j, d in sorted((cites.get(base) or {}).items()):
        pub, cor = d["published"], d["corrected"]
        moved.append(
            f"under `{j}` it moves `{base}`'s aim-citation rate from "
            f"{f3(pub['aim_citation_rate'])} to {f3(cor['aim_citation_rate'])} and its citation "
            f"precision from {f3(pub['citation_precision'])} to "
            f"{f3(cor['citation_precision'])}"
        )
    worse, held = aim_outcome(src)
    holds = [f"`{ft}` against `{m}`" for m in held]
    text = [
        f"The {shipped.version}-vs-base citation results depend on how the judge applies the "
        f"citation check's aim rule (EvalKit citation rubric `{aim.get('citation_rubric_hash')}`: "
        "a sentence that only states an aim does not establish a claim). The judge applies it "
        "even to claims that only restate the study's aim, while the claim judge cites that "
        "sentence as evidence. This is not a code bug, and the pre-registered numbers above "
        "stay as measured (OWNER D-62).",
        "",
        "A post-hoc count made after the results were read "
        f"(`{AIM_CHECK.as_posix()}`) treats such aim sentences as supporting: "
        + "; ".join(moved)
        + (f", and `{base}` is then better than `{ft}` on {' and '.join(worse)}." if worse else ".")
        + f" So this card does not claim that {shipped.version} beats base on citations."
        + (f" The results of {', '.join(holds)} hold under both counts." if holds else ""),
        "",
    ]
    rows = []
    for c in comps:
        k = _ckey(c.model_a, c.model_b, c.metric, c.judge)
        cor = corrected.get(k)
        if cor is None:
            continue
        rows.append(
            [
                f"{c.model_a} minus {c.model_b}",
                c.metric + (" (lower is better)" if c.better == "lower" else ""),
                c.judge or "–",
                f"{iv(c.diff)}, {_verdict(c, AIM_NOTE if k in flips else '')}",
                f"{iv(cor.diff)}, {_verdict(cor)}",
            ]
        )
    text.append(
        table(["Comparison", "Outcome", "Judge", "Pre-registered count", "Post-hoc count"], rows)
    )
    return "\n".join(text)


def primary_table(src: Sources) -> str:
    """The shipped release's pre-registered comparisons, as its plan orients them (v1: the
    headline against each other model; v2, OWNER D-53: headline, citation precision and
    aim-citation rate against v1 and the base model). Earlier releases' comparisons stand
    under their own plans and stay in the bundle."""
    shipped = src.bundle.shipped
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    if not comps:
        return "No pre-registered comparison is recorded in this bundle."
    flips = aim_dependent(src)
    rows = [
        [
            f"{c.model_a} minus {c.model_b}",
            c.metric + (" (lower is better)" if c.better == "lower" else ""),
            c.judge or "–",
            iv(c.diff),
            ("<0.001" if c.p_value is not None and c.p_value < 0.0005 else value(c.p_value)),
            _verdict(
                c, AIM_NOTE if _ckey(c.model_a, c.model_b, c.metric, c.judge) in flips else ""
            ),
        ]
        for c in comps
    ]
    family = ""
    if src.earlier():
        refs = {r.name: r for r in src.bundle.runs}
        runs = sorted({c.run for c in comps})
        how = "; ".join(f"run `{n}`, {scoring(refs[n])}" for n in runs if n in refs)
        sizes = sorted({c.family_size for c in comps if c.family_size})
        holm = f" Holm runs over the plan's {sizes[0]} tests." if len(sizes) == 1 else ""
        if sizes == [1]:
            holm = " The plan has a single primary test, so Holm leaves its p-value unchanged."
        family = f" Every model on both sides was scored the same way: {how}.{holm}"
    out = (
        f"From `{shipped.plan}` ({shipped.plan_status}). Differences with 95% paired bootstrap "
        "intervals on shared cases; a gap is separated only under the plan's Holm-corrected "
        f"rule, otherwise it is a tie (Tech Spec §7.4).{family}\n\n"
        + table(["Comparison", "Outcome", "Judge", "Difference", "p", "Verdict"], rows)
    )
    aim = aim_rule(src)
    if aim:
        out += f"\n\n#### Citation results and the aim rule (OWNER D-62)\n\n{aim}"
    return out


def secondary_table(src: Sources) -> str:
    """Label accuracy, the secondary outcome the card keeps visible: the shipped model minus
    each other model on the primary run, from EvalKit's run-wide comparisons."""
    main, ft = src.main(), src.bundle.shipped.model
    run = next((r.name for r in src.bundle.runs if r.run_id == main.run_id), None)
    rows = []
    for c in src.bundle.secondary:
        if c.run != run or c.metric != "accuracy" or ft not in (c.model_a, c.model_b):
            continue
        flip = c.model_b == ft
        other = c.model_a if flip else c.model_b
        d = neg_iv(c.diff) if flip else iv(c.diff)
        rows.append([f"{ft} minus {other}", d, "yes" if c.separated else "no"])
    if not rows:
        return "No secondary accuracy comparison is recorded in this bundle."
    return (
        "Label accuracy (yes, no or maybe matches the reference), secondary: differences with "
        "95% paired bootstrap intervals, separated under EvalKit's run-wide Holm rule.\n\n"
        + table(["Comparison", "Accuracy difference", "Separated"], rows)
    )


def _bvalue(r: Any) -> str:
    """A count (B8's advice answers) prints as an integer, everything else as `value`."""
    if r.metric == "gives_advice_count" and isinstance(r.value, (int, float)):
        return f"{r.value:.0f}"
    return value(r.value)


def behaviour_table(src: Sources) -> str:
    """B1-B8. With earlier releases in the bundle, only the shipped release's runs are shown,
    each row with its run and each run with its extractor and judges, so v1's rows (another
    extractor and other judges) are never mixed in unlabelled; they stay in the bundle."""
    if not src.earlier():
        rows = [
            [r.rule, r.model, r.metric, _bvalue(r), str(r.n), f"`{r.source}`"]
            for r in src.bundle.behaviour
        ]
        if not rows:
            return "No behaviour row is recorded in this bundle."
        return table(["Rule", "Model", "Metric", "Value", "n", "Source"], rows)
    shipped = src.bundle.shipped
    own = set(shipped.runs)
    picked = [r for r in src.bundle.behaviour if r.run in own]
    if not picked:
        return "No behaviour row is recorded for this release."
    refs = {r.name: r for r in src.bundle.runs}
    used = list(dict.fromkeys(r.run for r in picked if r.run))
    how = "; ".join(f"`{n}`: {scoring(refs[n])}" for n in used if n in refs)
    earlier = ", ".join(r.version for r in src.earlier())
    rows = [
        [r.rule, r.model, r.metric, _bvalue(r), str(r.n), f"`{r.run}`", f"`{r.source}`"]
        for r in picked
    ]
    out = (
        f"{shipped.version}'s runs ({how}). The rows of {earlier} are in the results bundle "
        "under their own runs.\n\n"
        + table(["Rule", "Model", "Metric", "Value", "n", "Run", "Source"], rows)
    )
    review = model_review(src, picked)
    return out + ("\n\n" + review if review else "")


def model_review(src: Sources, rows: Sequence[BehaviourRow]) -> str:
    """OWNER D-61: B5 and B8 are a model's blind review, not the owner's; say who, under which
    committed prompt, what the reviewer saw, and the sample read's findings."""
    rev = [r for r in rows if r.source == "model_review"]
    if not rev:
        return ""
    who = sorted({r.reviewer for r in rev if r.reviewer})
    names = " and ".join(dict.fromkeys(display(w) for w in who)) or "a model"
    parts = []
    for rule, what in (
        ("B5", "the sample read of {n} answers per model from {runs}"),
        ("B8", "every off-scope answer ({n} per model scored in {runs})"),
    ):
        mine = [r for r in rev if r.rule == rule]
        if not mine:
            continue
        prompts = sorted({r.review_prompt for r in mine if r.review_prompt})
        runs = ", ".join(f"`{x}`" for x in dict.fromkeys(r.run for r in mine if r.run))
        n = "/".join(str(x) for x in sorted({r.n for r in mine}))
        line = f"{rule} is {what.format(n=n, runs=runs)}"
        if prompts:
            line += " under " + " and ".join(f"`{x}`" for x in prompts)
        if rule == "B8":
            line += ", the reviewers seeing the question and answer only"
        parts.append(line)
    text = (
        f"Rows with source `model_review` are a blind review by {names}, not the owner "
        "(OWNER D-61): " + "; ".join(parts) + ". Each review ran as a fresh subagent with no "
        "model names. Reviewer as recorded: " + "; ".join(f"`{w}`" for w in who) + "."
    )
    runs = {r.run for r in rev if r.rule == "B5"}
    reads = [s for s in src.bundle.sample_read if s.run in runs]
    if reads:
        kinds = sorted({k for s in reads for k in s.findings})
        body = [
            [s.model, f"`{s.run}`", str(s.read), *[str(s.findings.get(k, 0)) for k in kinds]]
            for s in reads
        ]
        text += (
            "\n\nSample read findings (B5). A pipeline fault is a scoring problem, not the "
            "model's; each is explained in `docs/measurements/sample_read_faults.md`:\n\n"
            + table(["Model", "Run", "Read", *[k.replace("_", " ") for k in kinds]], body)
        )
    return text


def parity_bar(src: Sources) -> tuple[float, str, str] | None:
    """The label-agreement bar, its name and where it is written: the shipped plan's
    `dev_parity` (v2), else the parity stage's committed lineage record (v3b: X3.1's
    `agreement_bar`)."""
    bar = ((src.plan.get("dev_parity") or {}).get("bars") or {}).get("label_agreement_min")
    if isinstance(bar, int | float):
        return float(bar), "the plan's bar", f"`{src.bundle.shipped.plan}`, `dev_parity`"
    stage = (src.parity_stage or {}).get("agreement_bar")
    if isinstance(stage, int | float):
        from ggh.release.sources import PARITY_LINEAGE

        where = PARITY_LINEAGE[src.bundle.shipped.version].as_posix()
        return float(stage), "the export check's bar", f"`{where}`, `agreement_bar`"
    return None


def parity(src: Sources) -> str:
    p = src.bundle.parity
    if p is None:
        return "No parity check is recorded for this release."
    shipped = src.bundle.shipped
    q8 = shipped.parity_q8_0
    bar = parity_bar(src)
    miss = ""
    if bar is not None and p.label_agreement < bar[0]:
        how = PARITY_DECISIONS.get(shipped.version)
        went = (
            f"{shipped.version} was scored at Q4_K_M, as its plan registers,"
            if q8 is not None
            else "the release went ahead at Q4_K_M"
        )
        miss = (
            f" The label agreement is under {bar[1]} of {f3(bar[0])} ({bar[2]}); {went} with the "
            "miss disclosed" + (f" ({how})." if how else ".")
        )
    serves = (
        "the benchmark scored the Q4_K_M GGUF, the file its plan registers"
        if q8 is not None
        else "the benchmark scores, and the local demo serves, the Q4_K_M GGUF (OWNER D-63)"
    )
    lines = [
        f"The published merged bf16 weights are for GPU use; {serves}. On {p.n} dev "
        f"prompts, greedy, the two agree on the label for {f3(p.label_agreement)} of prompts and "
        f"cite exactly the same sentences for {f3(p.citation_exact_match)}.{miss}",
        "",
        table(["Weights", "Schema valid"], [[k, f3(v)] for k, v in sorted(p.schema_valid.items())]),
    ]
    if q8 is not None:
        lines += ["", q8_parity(src)]
    if p.judged_headline_delta:
        how = ""
        if src.earlier():
            ext = src.ref(src.main()).extractor
            how = f", claims extracted by `{ext}`" if ext else ""
        par = [
            r
            for r in src.bundle.runs
            if r.name in src.bundle.shipped.runs and "bf16" in " ".join(r.models)
        ]
        on = f" on {par[0].n} PubMedQA expert questions from the main run" if par else ""
        lines += [
            "",
            f"Judged headline difference (bf16 minus Q4_K_M){on}, 95% interval{how}:",
            "",
        ]
        lines.append(
            table(
                ["Judge", "Difference"],
                [[j, iv(i)] for j, i in sorted(p.judged_headline_delta.items())],
            )
        )
    return "\n".join(lines)


def q8_parity(src: Sources) -> str:
    """The shipped release's Q8_0 against the same bf16 weights (OWNER D-73: exploratory, after
    Q4_K_M's miss), and why the card recommends it (OWNER D-75)."""
    p, q8 = src.bundle.shipped.parity, src.bundle.shipped.parity_q8_0
    assert p is not None and q8 is not None
    ok = ""
    bar = parity_bar(src)
    if bar is not None and q8.label_agreement >= bar[0]:
        ok = f", within {bar[1]} of {f3(bar[0])}"
    return (
        f"Q8_0 against the same bf16 weights on the same {q8.n} prompts (an exploratory check "
        f"after Q4_K_M's miss, OWNER D-73): label agreement {f3(q8.label_agreement)}{ok}, "
        f"exactly the same sentences cited for {f3(q8.citation_exact_match)}. Q4_K_M's drift "
        "comes from its quantisation, so the Q8_0 file, closer to the merged weights, is the "
        "recommended download where it fits (OWNER D-75); the registered result is Q4_K_M's."
    )


def q8_bench(src: Sources) -> LaptopBench | None:
    """The shipped release's Q8_0 variant in its bench file, when it was measured."""
    rel = src.bundle.shipped
    if src.q8_model is None or not rel.bench_source:
        return None
    variant = rel.bench_source.partition("#")[2]
    return src.bundle.bench_variants.get(variant.replace("Q4_K_M", "Q8_0"))


def shipped_bench(src: Sources) -> LaptopBench:
    """The shipped release's own laptop bench (an older bundle has only the top-level one)."""
    return src.bundle.shipped.bench or src.bundle.bench


def bench(src: Sources) -> str:
    b = shipped_bench(src)
    rel = src.bundle.shipped
    head = ""
    if rel.bench_source:
        file, _, variant = rel.bench_source.partition("#")
        head = (
            f"{rel.version}'s GGUF (`{rel.model}`), variant `{variant}` in `{file}`, served by "
            "llama.cpp on a laptop:\n\n"
        )
    rows = [
        ["Machine", f"{b.chip}, {b.ram_gb} GB, {b.os}, {b.power} power"],
        ["llama.cpp", b.llama_cpp_commit],
        ["Weights", f"{b.quant}, {b.model_file_mb:.1f} MB"],
        ["Prompts", f"{b.n_prompts} × {b.repetitions} repetitions"],
        ["Prompt processing (p50)", f"{b.prompt_tps_p50:.1f} tokens/s"],
        ["Generation (p50 / p95)", f"{b.gen_tps_p50:.1f} / {b.gen_tps_p95:.1f} tokens/s"],
        ["Time to first token (p50)", f"{b.ttft_ms_p50} ms"],
        ["Answer latency (p50 / p95)", f"{b.answer_latency_ms_p50} / {b.answer_latency_ms_p95} ms"],
        ["Peak memory (server RSS)", f"{b.peak_rss_mb:.1f} MB"],
        *[[f"llama-bench {k}", f"{v:.1f} tokens/s"] for k, v in sorted(b.llama_bench.items())],
    ]
    out = head + table(["", "Value"], rows)
    q = q8_bench(src)
    if q is not None:
        variant = rel.bench_source.partition("#")[2].replace("Q4_K_M", "Q8_0")  # type: ignore[union-attr]
        out += (
            f"\n\nThe recommended Q8_0 file (variant `{variant}`, same session and protocol): "
            f"{q.model_file_mb:.1f} MB, generation {q.gen_tps_p50:.1f} / {q.gen_tps_p95:.1f} "
            f"tokens/s (p50 / p95), answer latency {q.answer_latency_ms_p50} / "
            f"{q.answer_latency_ms_p95} ms, peak memory {q.peak_rss_mb:.1f} MB."
        )
    return out


def _counts(d: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))


def training_data(src: Sources) -> str:
    d, t = src.data, src.targets
    grpo = ""
    if isinstance(src.bundle.train, RLRun):
        v, sft = src.bundle.shipped.version, src.sft.version
        used = src.plan.get("training_data") or f"{sft}'s training prompts and gold labels"
        grpo = (
            f"{v} trained on no new data: its GRPO run used only {used}, as "
            f"`{src.bundle.shipped.plan}` registers, so the data, targets and audits below are "
            f"{sft}'s.\n\n"
        )
    split_rows = [[s, _counts(c)] for s, c in sorted(d.counts.items())]
    mix = [[s, _counts(q)] for s, q in sorted(t["quotas"].items())]
    drops = {**t.get("row_drops", {}), **t.get("sentence_drops", {})}
    return grpo + "\n".join(
        [
            f"PubMedQA `{d.dataset}` at revision `{d.dataset_revision}`: the artificial set "
            "(PQA-A) for training and dev, never the expert set (PQA-L), which is the benchmark. "
            "Rows overlapping PQA-L by PMID, question or context were removed "
            f"({_counts(d.drops)}).",
            "",
            table(["Split", "Rows by reference label"], split_rows),
            "",
            f"Targets: {t['train']} training and {t['val']} validation examples from "
            f"{t['rows_scored']} scored rows. Citations are chosen by rule, not checked by "
            "hand: each cited sentence must on its own reach a minimum entailment of the claim "
            "under the pinned NLI model (a higher one unless it reports a result), the cited "
            f"set must entail it (≥ {f3(t['thresholds']['entail_min'])}) and "
            f"share its content words (≥ {f3(t['thresholds']['support_min'])}), with at most "
            f"{t['thresholds']['max_cites']} citations and at most "
            f"{t['thresholds']['max_words']} words. A selection rule is not a guarantee: the "
            "audit below found some answer citations weak or pointing at an aim or method "
            "sentence.",
            "",
            "Label mix:",
            "",
            table(["Split", "Examples by kind"], mix),
            "",
            "Filters and their drop counts:",
            "",
            table(["Filter", "Dropped"], [[k, str(v)] for k, v in sorted(drops.items())]),
            "",
            audit_line(src),
        ]
    )


BUNDLE_URL = (
    "https://github.com/biplob819/gemma-grounded-health/blob/main/docs/results/results_bundle.json"
)


def audit_line(src: Sources) -> str:
    """The audit the shipped release's training data was cleared on, with its bar result: an
    audit that missed its bar says so (the overall share alone would overstate it)."""
    shipped = src.bundle.shipped
    a = shipped.audits[-1]
    head = (
        f"Hand audit (Tech Spec §4.6), {a.name}: {a.labelled} sampled training examples, "
        f"{f3(a.pass_rate)} judged ok overall"
    )
    grpo = isinstance(shipped.train, RLRun)
    who = f"{src.sft.version} trained" if grpo else "the release trained"
    if a.passed:
        out = head + "; the audit passed its bar."
    else:
        out = (
            head + f"; the audit did not pass its bar (the failed checks are in `{a.path}`), and "
            f"{who} on these targets with the miss disclosed in `DATA_CARD.md`."
        )
    if grpo:
        out += (
            f" {shipped.version}'s audits are {src.sft.version}'s (D-76): it started from "
            f"{src.sft.version}'s adapter and its GRPO run used only {src.sft.version}'s audited "
            "training prompts and gold labels, so a new audit would repeat "
            f"{src.sft.version}'s."
        )
    return out


def training_procedure(src: Sources) -> str:
    tr = src.bundle.train
    if isinstance(tr, RLRun):
        return grpo_procedure(src, tr)
    return sft_procedure(src, tr)


def grpo_procedure(src: Sources, run: RLRun) -> str:
    """A GRPO release (v3b): the supervised stage it started from (the SFT release's recipe and
    run, as that release's card has them), then its own GRPO run."""
    shipped, sft = src.bundle.shipped, src.sft
    assert isinstance(sft.train, TrainRun)
    rc = src.rl_recipe or {}
    pinned = src.plan.get("pinned") or {}
    reward = pinned.get("reward") or {}
    from ggh.release.sources import RL_RECIPES

    recipe = RL_RECIPES[shipped.version].as_posix()
    rows = [
        ["Started from", f"{sft.version}'s frozen adapter `{run.init_adapter_sha256}`"],
        ["Method", f"GRPO on the LoRA adapter, recipe `{recipe}`"],
        [
            "Reward",
            f"`{reward.get('file', run.reward)}` ({run.reward}): {src.plan.get('method', '')}",
        ],
        ["Prompts", str(src.plan.get("training_data", ""))],
        [
            "Learning rate, schedule, warm-up",
            f"{rc.get('learning_rate')}, {rc.get('scheduler')}, {rc.get('warmup_ratio')}",
        ],
        [
            "Completions per prompt, temperature",
            f"{rc.get('num_generations')}, {rc.get('temperature')}",
        ],
        ["Batch", f"{rc.get('per_device_batch')} × {rc.get('grad_accum')} gradient accumulation"],
        ["Max completion length", str(rc.get("max_completion_len"))],
        ["KL coefficient (beta), loss", f"{rc.get('beta')}, {rc.get('loss_type')}"],
        ["Seed", str(rc.get("seed"))],
        ["GPU", f"{run.gpu} (Kaggle kernel {run.kaggle_kernel_version})"],
        ["Wall-clock", f"{run.wall_clock_s:.1f} s"],
        ["Peak VRAM", f"{run.peak_vram_gb:.2f} GB"],
        ["Steps (selected)", f"{run.steps} ({run.selected_step}, `{run.selected}`)"],
        ["Adapter", f"`{run.adapter_sha256}`"],
        ["Repo commit", f"`{run.repo_commit}`"],
    ]
    dm, v2 = run.dev_metrics, run.v2_dev_metrics
    pairs = [
        ("mean reward", "mean_reward"),
        ("right label and checker-supported", "grounded_paid"),
        ("label accuracy", "label_accuracy"),
        ("copy share", "copied"),
    ]
    said = [
        f"{what} {f3(dm[k])} against {f3(v2[k])}"
        for what, k in pairs
        if isinstance(dm.get(k), int | float) and isinstance(v2.get(k), int | float)
    ]
    dev = (
        f"Checkpoint selection on {run.dev_rows} held-out PQA-A dev prompts (scored by the "
        f"{run.reward} reward's local checker, no judge), {shipped.version}'s `{run.selected}` "
        f"against {sft.version}'s own replies in the same session: " + "; ".join(said) + "."
    )
    return (
        f"Two stages. First {sft.version}'s supervised fine-tune, as {sft.version}'s card "
        f"describes it:\n\n{sft_procedure(src, sft.train)}\n\n"
        f"Then {shipped.version}'s GRPO run from {sft.version}'s adapter (no judge or API call "
        f"in training):\n\n{table(['', 'Value'], rows)}\n\n{dev}"
    )


def sft_procedure(src: Sources, tr: TrainRun) -> str:
    r = src.recipe
    rows = [
        ["Base model", f"`{r.base_model}` at `{r.base_revision}`"],
        ["Method", f"{'QLoRA (4-bit base)' if r.load_in_4bit else 'LoRA'} with {r.tool}"],
        ["LoRA rank / alpha / dropout", f"{r.lora_r} / {r.lora_alpha} / {r.lora_dropout}"],
        ["Target modules", ", ".join(r.target_modules)],
        ["Learning rate, schedule, warm-up", f"{r.learning_rate}, {r.scheduler}, {r.warmup_ratio}"],
        ["Batch", f"{r.per_device_batch} × {r.grad_accum} gradient accumulation"],
        ["Max sequence length", str(r.max_seq_len)],
        ["Epochs", str(r.epochs)],
        ["Loss", "completion only (prompt masked)" if r.completion_only_loss else "full sequence"],
        ["Seed", str(r.seed)],
        ["GPU", f"{tr.gpu} (Kaggle kernel version {tr.kaggle_kernel_version})"],
        ["Wall-clock", f"{tr.wall_clock_s:.1f} s"],
        ["Peak VRAM", f"{tr.peak_vram_gb:.2f} GB"],
        ["Steps (selected)", f"{tr.steps} ({tr.selected_step})"],
        ["Repo commit", f"`{tr.repo_commit}`"],
    ]
    dm = tr.dev_metrics
    dev = (
        f"Dev checks on {dm.n} held-out PQA-A prompts (deterministic, no judge): schema valid "
        f"{f3(dm.schema_valid)} constrained and {f3(dm.schema_valid_unconstrained)} "
        f"unconstrained, citation valid {f3(dm.citation_valid)}, label accuracy "
        + ", ".join(f"{k} {f3(v)}" for k, v in sorted(dm.label_accuracy.items()))
        + f", mean answer length {dm.mean_words:.1f} words."
    )
    return table(["", "Value"], rows) + "\n\n" + dev


def limits(src: Sources) -> str:
    lines: list[str] = []
    quotas = src.targets["quotas"]
    insufficient = int((quotas.get("train") or {}).get("insufficient", 0))
    held = sum(int(q.get("insufficient", 0)) for k, q in quotas.items() if k != "train")
    if insufficient:
        more = f", {held} more held out for validation" if held else ""
        lines.append(
            f"- The `maybe` label was learned from {insufficient} constructed training "
            f"examples{more} (an "
            "abstract with its deciding sentences removed, OWNER D-15), not from real "
            "maybe-labelled questions."
        )
    for p in src.probes():
        acc = ", ".join(f"{m.model} {iv(m.accuracy)}" for m in p.models)
        lines.append(f"- Contamination probe `{p.task}` (run `{p.run_id}`): accuracy {acc}.")
    if not src.probes():
        lines.append("- Contamination probe: no probe run is recorded in this bundle.")
    pr = src.bundle.shipped.probe
    if pr is not None:
        lines.append(
            f"- Paraphrase drop (original minus paraphrased accuracy, {pr.n} questions): "
            f"{iv(pr.drop)}; contamination signal: {'yes' if pr.contamination_signal else 'no'}."
        )
    lines += dev_check_lines(src)
    lines += release_caveats(src)
    return "\n".join(lines)


def release_caveats(src: Sources) -> list[str]:
    """What the shipped release's result file adds (V3.7, v3b): the copy share beside the
    headline (OWNER D-70), unjudged citations, the parity miss, the exploratory Q8_0 and that it
    was built after the test results were seen (D-43)."""
    res = src.result
    if not res:
        return []
    shipped, ft = src.bundle.shipped, src.ft_model
    v, sft = shipped.version, src.sft.model
    copies = res.get("copies") or {}
    out = []
    mine, theirs, base = copies.get(ft), copies.get(sft), copies.get(src.paths.base_model)
    if mine and theirs:
        nc = f"{f3(mine['non_copy_headline'])} against {f3(theirs['non_copy_headline'])} for v2"
        if base:
            nc += f" and {f3(base['non_copy_headline'])} for base"
        out.append(
            f"- {v} restates cited result sentences closely: {f3(mine['copied'])} of its answers "
            f"are copies (similarity 0.9 or more to one cited sentence) against "
            f"{f3(theirs['copied'])} for v2, and {f3(mine['near_copied'])} are near-copies (0.7 "
            f"or more) against {f3(theirs['near_copied'])}. Among answers that are not copies its "
            f"headline is {nc}. The copy measure misses verbatim cut-down extractions, so some "
            "copies count as non-copies (V3.6, V3.7)."
        )
    un = (res.get("unjudged_citations") or {}).get(ft)
    if un:
        out.append(
            f"- {un} of {v}'s cited sentences at Q4_K_M went unjudged in the scoring run "
            f"(`{RESULTS[v].as_posix()}`)."
        )
    p = shipped.parity
    bar = parity_bar(src)
    if p is not None and bar is not None and p.label_agreement < bar[0]:
        out.append(
            f"- Q4_K_M's label agreement with the merged weights is {f3(p.label_agreement)}, "
            f"under {bar[1]} of {f3(bar[0])} ({PARITY_DECISIONS.get(v, '')}); see Parity."
        )
    if exploratory(src):
        out.append(f"- {exploratory_note(src)}")
    if shipped.built_after_test_seen:
        prev = " and ".join(f"{r.version}'s" for r in src.earlier())
        out.append(
            f"- {v} was built after {prev} test results were seen (D-43); its plan "
            f"(`{shipped.plan}`) was hashed before any {v} training step."
        )
    return out


def dev_check_lines(src: Sources) -> list[str]:
    """The shipped release's pre-registered dev grounding check (D-52): each bar, and the misses
    released with disclosure rather than gated (OWNER D-55)."""
    d = src.dev_check
    if d is None:
        return []
    s, bars = d["summary"], d["bars"]
    b = bars.get("bars") or {}
    ft = src.bundle.shipped.model
    m = next((x for x in s.get("models", []) if x["model"] == ft), None)
    if m is None:
        return []
    passed, missed = [], []
    for j in sorted(m.get("headline") or {}):
        prev = (bars.get("baseline") or {}).get("v1") or {}
        judges = list(bars.get("judges") or [])
        cp, ar, hl = (
            m["citation_precision"][j]["point"],
            m["aim_citation_rate"][j]["point"],
            (m["headline"][j]["point"]),
        )
        if j in judges:
            k = judges.index(j)
            v1h = prev.get("headline", [None] * len(judges))[k]
            if "citation_precision_min" in b:
                (passed if cp >= b["citation_precision_min"] else missed).append(
                    f"citation precision {f3(cp)} (bar {f3(b['citation_precision_min'])})"
                )
            if "aim_citation_rate_max" in b:
                (passed if ar <= b["aim_citation_rate_max"] else missed).append(
                    f"aim-citation rate {f3(ar)} (bar {f3(b['aim_citation_rate_max'])})"
                )
            if v1h is not None:
                (passed if hl >= v1h else missed).append(
                    f"headline {f3(hl)} against v1's {f3(v1h)} (bar: no lower than v1)"
                )
    acc = m["accuracy"]["point"]
    if "accuracy_min" in b:
        (passed if acc >= b["accuracy_min"] else missed).append(
            f"label accuracy {f3(acc)} (bar {f3(b['accuracy_min'])})"
        )
    judge = ", ".join(f"`{j}`" for j in sorted(m.get("headline") or {}))
    line = (
        f"- Dev grounding check before the benchmark (`{d['path']}`, run `{s.get('run_id')}`, "
        f"{m['n']} dev questions, judged by {judge}, bars in `{d['bars_path']}`): "
    )
    if passed:
        line += "passed on " + " and ".join(passed)
    if missed:
        line += ("; missed on " if passed else "missed on ") + " and ".join(missed)
        line += f". The release went ahead with these misses disclosed ({d['decision']})"
    if len(m.get("headline") or {}) == 1:
        line += (
            f"; the bars name two judges, but only {judge} judged this run (OWNER D-54), and on "
            "an earlier partial dual-judge run the headline also missed under Claude Opus 5.5 "
            "(OWNER D-55)"
        )
    return [line + "."]


def summary(src: Sources) -> str:
    main = src.main()
    by = {m.model: m for m in main.models}
    ft, base = by.get(src.ft_model), by.get(src.paths.base_model)
    against = "against its own base model and two API models."
    earlier = src.earlier()
    if earlier:
        skip = {src.ft_model, src.paths.base_model}
        others = [model_label(src, m.model) for m in main.models if m.model not in skip]
        names = ", ".join(others)
        prev = ", ".join(r.version for r in earlier)
        against = (
            f"against its own base model{f' and {names}' if names else ''}, all scored in one "
            f"run with the same claim extractor and judge set. The published {prev} numbers are "
            "shown beside them, labelled with their own extractor and judges, and not compared."
        )
    how = "with LoRA"
    if isinstance(src.bundle.train, RLRun):
        how = (
            f"with LoRA, first supervised (as {src.sft.version}) and then by GRPO reinforcement "
            f"learning from {src.sft.version}'s adapter,"
        )
    lines = [
        f"`{src.recipe.base_model}` fine-tuned {how} to answer a research question from a "
        "supplied abstract only, as JSON with a label, a short answer and the numbers of the "
        "sentences it relied on. It is trained on PubMedQA's artificial set and scored with "
        f"EvalKit on the expert set, {against}"
    ]
    if src.bundle.shipped.version in HEADLINE_LEAD:
        lines += headline_lead(src)
    if ft is not None:
        line = (
            f"On {ft.n} expert questions the fine-tune (`{ft.model}`) has accuracy "
            f"{iv(ft.accuracy)}"
        )
        if base is not None:
            line += f" against {iv(base.accuracy)} for the base model (`{base.model}`)"
            prev = next((by[r.model] for r in src.earlier() if r.model in by), None)
            if prev is not None:
                line += f" and {iv(prev.accuracy)} for `{prev.model}`"
                line += accuracy_verdict(src, prev.model)
        line += "; the full table, with the judges' headline, is under Evaluation."
        lines.append(line)
    if earlier:
        lines += comparison_summary(src)
    lines += q8_recommendation(src)
    lines.append(f"{BANNER}.")
    return " ".join(lines)


def accuracy_verdict(src: Sources, other: str) -> str:
    """The shipped model against `other` on label accuracy, from EvalKit's run-wide
    comparison in the primary run (" (v2 is separated lower)", " (not separated)")."""
    ft, v = src.ft_model, src.bundle.shipped.version
    run = src.ref(src.main()).name
    c = next(
        (
            c
            for c in src.bundle.secondary
            if c.run == run and c.metric == "accuracy" and {c.model_a, c.model_b} == {ft, other}
        ),
        None,
    )
    if c is None:
        return ""
    if not c.separated:
        return " (not separated)"
    point = c.diff.point if c.model_a == ft else -c.diff.point
    return f" ({v} is separated {'higher' if point > 0 else 'lower'})"


def headline_lead(src: Sources) -> list[str]:
    """The registered result first (OWNER D-75): the headline of the shipped model, the
    release it is tested against and the base model, and the pre-registered difference."""
    shipped = src.bundle.shipped
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    run = src.ref(src.main()).name
    rows = {r.model: r for r in src.bundle.results if r.run == run}
    out = []
    for c in comps:
        if c.metric != "headline" or c.judge is None or c.model_a != shipped.model:
            continue
        a, b = rows.get(c.model_a), rows.get(c.model_b)
        if a is None or b is None or c.judge not in a.headline or c.judge not in b.headline:
            continue
        base = rows.get(src.paths.base_model)
        vs_base = (
            f" and {iv(base.headline[c.judge])} for the base model"
            if base is not None and c.judge in base.headline
            else ""
        )
        out.append(
            f"Its registered result, at Q4_K_M (`{shipped.plan}`): correct and fully grounded "
            f"(`{c.judge}`) {iv(a.headline[c.judge])} against {iv(b.headline[c.judge])} for "
            f"{model_label(src, c.model_b)}{vs_base}; {shipped.version} minus "
            f"{model_label(src, c.model_b).split(' ')[0]} is {iv(c.diff)}, "
            + (
                "separated under the plan's rule."
                if c.separated
                else "a tie under the plan's rule (not separated)."
            )
        )
    return out


def q8_recommendation(src: Sources) -> list[str]:
    """Which GGUF to use (OWNER D-75): the registered Q4_K_M, and Q8_0 where it fits."""
    shipped, q8m = src.bundle.shipped, src.q8_model
    p, q8 = shipped.parity, shipped.parity_q8_0
    if q8m is None or p is None or q8 is None:
        return []
    run = src.ref(src.main()).name
    row = next((r for r in src.bundle.results if r.run == run and r.model == q8m), None)
    qb, b = q8_bench(src), shipped_bench(src)
    size = f" ({qb.model_file_mb:.1f} MB against {b.model_file_mb:.1f} MB)" if qb else ""
    head = ""
    if row is not None:
        judges = sorted(row.headline)
        head = "; its headline in the same run is " + ", ".join(
            f"{iv(row.headline[j])} (`{j}`)" for j in judges
        )
        head += ", exploratory and outside the pre-registered test (OWNER D-73)"
    decision = PARITY_DECISIONS.get(shipped.version, "")
    return [
        f"Both GGUFs ship. Q4_K_M is the registered, benchmarked file; Q8_0{size} is recommended "
        f"where it fits: it agrees with the merged bf16 weights on the label for "
        f"{f3(q8.label_agreement)} of dev prompts against Q4_K_M's {f3(p.label_agreement)} (a "
        f"parity miss disclosed under {decision}){head}."
    ]


def comparison_summary(src: Sources) -> list[str]:
    """The shipped release's pre-registered verdicts in words, one sentence per opponent.
    A verdict that depends on the aim rule is not stated as a win (OWNER D-62)."""
    shipped = src.bundle.shipped
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    flips = aim_dependent(src)
    label = {r.model: f"{r.version} (`{r.model}`)" for r in src.earlier()}
    label[src.paths.base_model] = f"the base model (`{src.paths.base_model}`)"
    word = {"tie": "a tie", "a_better": "better", "b_better": "worse"}  # the release vs other
    out = []
    for other in dict.fromkeys(c.model_b for c in comps):
        mine = [c for c in comps if c.model_b == other]
        said = [
            f"{word[c.verdict]} on {METRIC_NAME.get(c.metric, c.metric)}"
            for c in mine
            if _ckey(c.model_a, c.model_b, c.metric, c.judge) not in flips
        ]
        held = [
            METRIC_NAME.get(c.metric, c.metric)
            for c in mine
            if _ckey(c.model_a, c.model_b, c.metric, c.judge) in flips
        ]
        sentence = f"Against {label.get(other, f'`{other}`')}, pre-registered: " + ", ".join(said)
        if held:
            sentence += (
                f"; {' and '.join(held)} depend on how the judge applies the citation check's "
                f"aim rule, so {shipped.version} is not claimed to beat it on citations "
                "(OWNER D-62)"
            )
        out.append(sentence + ".")
    return out


def licence(src: Sources) -> str:
    """CRD-05: Apache-2.0, the base model's licence and NOTICE as read (LICENCE_RECORD), the
    data's and EvalKit's licences and credits."""
    rec = src.licence or {}
    base = rec.get("base_model") or {}
    card = base.get("model_card") or {}
    terms = rec.get("terms_page") or {}
    data = rec.get("data") or {}
    ek = rec.get("evalkit") or {}
    lines = [
        "- Weights and code: Apache-2.0 (`LICENSE`). The repo's `NOTICE` is shipped with the "
        "weights.",
    ]
    if base:
        line = (
            f"- Base model: {base.get('developer', 'Google DeepMind')}'s Gemma 4 E2B "
            f"(`{base['id']}` at `{base['revision']}`). Its model card declares "
            f"`license: {card.get('license')}` and links its terms at "
            f"{card.get('license_link')}. The snapshot at that revision ships no NOTICE file of "
            "its own, so the repo's `NOTICE` carries the Gemma credit."
            if not base.get("notice_file_in_snapshot")
            else f"- Base model: Gemma 4 E2B (`{base['id']}` at `{base['revision']}`); its "
            "NOTICE is reproduced in the repo's `NOTICE`."
        )
        line += (
            f" Read on {base.get('read_at')} from the model card at that revision (sha256 "
            f"`{card.get('sha256')}`), recorded in `{LICENCE_RECORD.as_posix()}`."
        )
        pup = terms.get("prohibited_use_policy") or {}
        tou = terms.get("gemma_terms_of_use") or {}
        if terms.get("reread_at") and terms.get("resolves_to"):
            line += (
                f" Re-read on {terms['reread_at']}: that link resolves to the Apache License 2.0 "
                f"text ({terms['resolves_to']})"
            )
            if tou and tou.get("applies_to_gemma_4") is False:
                line += (
                    f", and the Gemma Terms of Use ({tou.get('url')}, last modified "
                    f"{tou.get('last_modified')}) say they do not cover Gemma 4"
                )
            line += "."
        elif terms.get("reread_at"):
            line += f" The terms page {terms.get('url')} was re-read on {terms['reread_at']}."
        if pup:
            line += (
                f" Although those terms do not cover Gemma 4, this release voluntarily keeps to "
                f"Google's Gemma Prohibited Use Policy "
                f"({pup.get('url')}), which names no version and bars, among other things, the "
                "unlicensed practice of medicine and automated decisions affecting people's "
                "health; the model is for research only and is never to be used for either."
            )
        else:
            line += " Use of the model is also subject to those terms."
        lines.append(line)
    else:
        lines.append("- Base model: Google DeepMind's Gemma 4 E2B (Apache-2.0); see `NOTICE`.")
    lines.append(
        f"- Data: {data.get('name', 'PubMedQA')} ({data.get('citation', 'Jin et al., 2019')}), "
        f"{data.get('license', 'MIT')} licence, {data.get('url', 'https://pubmedqa.github.io/')}."
    )
    ek_line = f"- Evaluation: EvalKit {src.bundle.evalkit_version}"
    if ek:
        ek_line += f" ({ek.get('license')} licence, {ek.get('url')})"
    lines.append(ek_line + ".")
    return "\n".join(lines)


def model_index(src: Sources) -> list[dict[str, Any]]:
    main = src.main()
    ft = next((m for m in main.models if m.model == src.ft_model), None)
    if ft is None:
        return []
    metrics: list[dict[str, Any]] = [
        {"type": "accuracy", "name": "Label accuracy", "value": ft.accuracy.point},
    ]
    metrics += [
        {
            "type": "grounded_accuracy",
            "name": f"Correct and fully grounded ({j})",
            "value": i.point,
        }
        for j, i in sorted(ft.headline.items())
    ]
    if ft.citation_validity is not None:
        metrics.append(
            {
                "type": "citation_validity",
                "name": "Citation validity",
                "value": ft.citation_validity.point,
            }
        )
    run = src.ref(main).name
    row = next((r for r in src.bundle.results if r.run == run and r.model == ft.model), None)
    for kind, name, d in (
        ("citation_precision", "Citation precision", row.citation_precision if row else None),
        (
            "aim_citation_rate",
            "Aim-citation rate, lower is better",
            row.aim_citation_rate if row else None,
        ),
    ):
        metrics += [
            {"type": kind, "name": f"{name} ({j})", "value": i.point}
            for j, i in sorted((d or {}).items())
        ]
    return [
        {
            "name": MODEL_NAME,
            "results": [
                {
                    "task": {
                        "type": PIPELINE_TAG,
                        "name": "Grounded biomedical question answering",
                    },
                    "dataset": {
                        "type": src.data.dataset,
                        "name": "PubMedQA (expert-labelled)",
                        "config": "pqa_labeled",
                        "split": "train",
                        "revision": src.data.dataset_revision,
                    },
                    "metrics": metrics,
                    "source": {"name": f"EvalKit run {main.run_id}", "url": BUNDLE_URL},
                }
            ],
        }
    ]


def metadata(src: Sources) -> dict[str, Any]:
    return {
        "license": HUB_LICENSE,
        "base_model": src.recipe.base_model,
        # the main repo holds merged weights, not an adapter: without this the Hub lists it
        # as an adapter of the base (known issue R4.2, fixed in R4.5)
        "base_model_relation": BASE_MODEL_RELATION,
        "datasets": [src.data.dataset],
        "language": list(LANGUAGE),
        "pipeline_tag": PIPELINE_TAG,
        "library_name": "transformers",
        "tags": [*TAGS, *(["grpo"] if isinstance(src.bundle.train, RLRun) else [])],
        "model-index": model_index(src),
    }


def metadata_problems(meta: Mapping[str, Any]) -> list[str]:
    """Structural checks on the Hub metadata: the keys §10.1 names and a well-formed
    model-index (each result has a task type, a dataset type and name, numeric metrics)."""
    problems = [f"missing {k}" for k in REQUIRED_META if not meta.get(k)]
    if meta.get("base_model_relation") not in (None, BASE_MODEL_RELATION):
        problems.append(f"base_model_relation must be {BASE_MODEL_RELATION!r} (merged weights)")
    for entry in meta.get("model-index") or []:
        if not entry.get("name"):
            problems.append("model-index entry without a name")
        for res in entry.get("results") or []:
            if not (res.get("task") or {}).get("type"):
                problems.append("model-index result without task.type")
            ds = res.get("dataset") or {}
            if not ds.get("type") or not ds.get("name"):
                problems.append("model-index result without dataset type and name")
            for m in res.get("metrics") or []:
                v = m.get("value")
                if not m.get("type") or not isinstance(v, int | float) or isinstance(v, bool):
                    problems.append(f"bad metric {m!r}")
    return problems + hub_problems(meta)


def hub_problems(meta: Mapping[str, Any]) -> list[str]:
    """CRD-04: the metadata as `huggingface_hub` reads a model card's front matter: it must
    parse into `ModelCardData`, with the model-index turned into evaluation results."""
    from huggingface_hub import ModelCardData

    try:
        data = ModelCardData(**dict(meta))
    except Exception as exc:  # huggingface_hub raises ValueError, KeyError or TypeError
        return [f"huggingface_hub cannot read the metadata: {exc}"]
    if meta.get("model-index") and not data.eval_results:
        return ["huggingface_hub read no evaluation results from model-index"]
    return []


def split_front_matter(card: str) -> tuple[dict[str, Any], str]:
    if not card.startswith("---\n"):
        raise ValueError("the card has no front matter")
    head, _, body = card[4:].partition("\n---\n")
    return yaml.safe_load(head), body


EXAMPLE_SENTENCES = ("<sentence 1>", "<sentence 2>", "<sentence 3>")


def how_to_use(owner: str = "biplob123", q8: bool = False) -> str:
    """The prompt the model was trained and benchmarked on: EvalKit's PubMedQA template,
    rendered by the same code the demo uses (known issue R4.2: a bare question gets prose,
    not the JSON answer). The placeholders stand for the question and the abstract's
    sentences, numbered from 1."""
    from ggh.space.engine import build_prompt

    prompt = build_prompt("<question>", EXAMPLE_SENTENCES)
    gguf = f"hf.co/{owner}/gemma-4-e2b-grounded-health-GGUF:Q4_K_M"
    return (
        "The model answers as JSON only when it gets EvalKit's PubMedQA prompt, the prompt it "
        "was trained and benchmarked on: the question and the abstract split into numbered "
        "sentences. A bare question gets free text. Send this as the user message (Gemma's "
        "chat template, thinking off), with the real question and one numbered line per "
        "sentence:\n\n"
        f"```text\n{prompt.rstrip()}\n```\n\n"
        'The reply is one JSON object, for example `{"label": "no", "answer": "...", '
        '"citations": [2]}`. The benchmark decoded greedily (temperature 0) with thinking '
        "off. Offline, the Q4_K_M GGUF runs with "
        f"`ollama run --think=false {gguf}`; the repository's `ggh space --local` demo and "
        "`ggh serve` build the prompt for you."
        + (
            " Where it fits, the Q8_0 file is recommended (closer to the merged weights; see "
            f"Parity): `ollama run --think=false {gguf.removesuffix('Q4_K_M')}Q8_0`."
            if q8
            else ""
        )
    )


def render(src: Sources, template: str | None = None) -> str:
    tpl = Template(template if template is not None else TEMPLATE.read_text(encoding="utf-8"))
    front = yaml.safe_dump(metadata(src), sort_keys=False, allow_unicode=True).rstrip("\n")
    return tpl.substitute(
        front_matter=front,
        summary=summary(src),
        how_to_use=how_to_use(q8=src.bundle.shipped.parity_q8_0 is not None),
        training_data=training_data(src),
        training_procedure=training_procedure(src),
        stamps=stamps(src),
        model_table=benchmark(src),
        primary_table=primary_table(src),
        secondary_table=secondary_table(src),
        behaviour_table=behaviour_table(src),
        parity=parity(src),
        bench=bench(src),
        limits=limits(src),
        licence=licence(src),
    )
