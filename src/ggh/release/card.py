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

from ggh.release.sources import Sources
from ggh.schemas import Comparison, ReleaseEntry, RunRef

TEMPLATE = Path(__file__).parent / "templates" / "model_card.md"
HUB_LICENSE = "apache-2.0"
LANGUAGE = ("en",)
PIPELINE_TAG = "text-generation"
TAGS = ("gemma", "medical", "biomedical", "pubmedqa", "grounded-generation", "citations", "lora")
MODEL_NAME = "gemma-4-e2b-grounded-health"
BANNER = "Research only, not medical advice"
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
                ("judges", any(r.judges != now.judges for r in refs)),
            )
            if bad
        ]
        if differs:
            notes.append(
                f"They were produced with a different {' and different '.join(differs)} from "
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
    return "\n".join([*lines, *earlier_tables(src)])


def stamps(src: Sources) -> str:
    b = src.bundle
    main = src.main()
    ref = next((r for r in b.runs if r.run_id == main.run_id), None)
    mode = ref.judge_mode if ref is not None else b.judge_mode
    lines = [f"- Judge mode: **{mode}**. Calibration: **{b.calibration_status}**."]
    if ref is not None and src.earlier():
        lines.append(f"- {b.shipped.version} scoring: {scoring(ref)}.")
    lines += [f"- ⚠ {s}" for s in stamp_lines(main)]
    if b.gemma_family_check:
        checks = ", ".join(f"{j} {f3(a)}" for j, a in sorted(b.gemma_family_check.items()))
        lines.append(f"- Gemma-family judge check (agreement on owner-labelled claims): {checks}.")
    return "\n".join(lines)


def _verdict(c: Comparison) -> str:
    if c.verdict == "tie":
        return "tie (not separated)"
    return f"{c.model_a if c.verdict == 'a_better' else c.model_b} better"


def primary_table(src: Sources) -> str:
    """The shipped release's pre-registered comparisons, as its plan orients them (v1: the
    headline against each other model; v2, OWNER D-53: headline, citation precision and
    aim-citation rate against v1 and the base model). Earlier releases' comparisons stand
    under their own plans and stay in the bundle."""
    shipped = src.bundle.shipped
    comps = [c for c in src.bundle.primary_comparisons if c.plan == shipped.plan]
    if not comps:
        return "No pre-registered comparison is recorded in this bundle."
    rows = [
        [
            f"{c.model_a} minus {c.model_b}",
            c.metric + (" (lower is better)" if c.better == "lower" else ""),
            c.judge or "–",
            iv(c.diff),
            value(c.p_value),
            _verdict(c),
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
        family = f" Every model on both sides was scored the same way: {how}.{holm}"
    return (
        f"From `{shipped.plan}` ({shipped.plan_status}). Differences with 95% paired bootstrap "
        "intervals on shared cases; a gap is separated only under the plan's Holm-corrected "
        f"rule, otherwise it is a tie (Tech Spec §7.4).{family}\n\n"
        + table(["Comparison", "Outcome", "Judge", "Difference", "p", "Verdict"], rows)
    )


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


def behaviour_table(src: Sources) -> str:
    """B1-B8. With earlier releases in the bundle, only the shipped release's runs are shown,
    each row with its run and each run with its extractor and judges, so v1's rows (another
    extractor and other judges) are never mixed in unlabelled; they stay in the bundle."""
    if not src.earlier():
        rows = [
            [r.rule, r.model, r.metric, value(r.value), str(r.n), f"`{r.source}`"]
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
        [r.rule, r.model, r.metric, value(r.value), str(r.n), f"`{r.run}`", f"`{r.source}`"]
        for r in picked
    ]
    return (
        f"{shipped.version}'s runs ({how}). The rows of {earlier} are in the results bundle "
        "under their own runs.\n\n"
        + table(["Rule", "Model", "Metric", "Value", "n", "Run", "Source"], rows)
    )


def parity(src: Sources) -> str:
    p = src.bundle.parity
    if p is None:
        return "No parity check is recorded for this release."
    lines = [
        f"The demo runs merged bf16 weights; the benchmark scores the Q4_K_M GGUF. On {p.n} dev "
        f"prompts, greedy, the two agree on the label for {f3(p.label_agreement)} of prompts and "
        f"cite exactly the same sentences for {f3(p.citation_exact_match)}.",
        "",
        table(["Weights", "Schema valid"], [[k, f3(v)] for k, v in sorted(p.schema_valid.items())]),
    ]
    if p.judged_headline_delta:
        how = ""
        if src.earlier():
            ext = src.ref(src.main()).extractor
            how = f", claims extracted by `{ext}`" if ext else ""
        lines += ["", f"Judged headline difference (bf16 minus Q4_K_M), 95% interval{how}:", ""]
        lines.append(
            table(
                ["Judge", "Difference"],
                [[j, iv(i)] for j, i in sorted(p.judged_headline_delta.items())],
            )
        )
    return "\n".join(lines)


def bench(src: Sources) -> str:
    b = src.bundle.bench
    rows = [
        ["Machine", f"{b.chip}, {b.ram_gb} GB, {b.os}, {b.power} power"],
        ["llama.cpp", b.llama_cpp_commit],
        ["Weights", f"{b.quant}, {b.model_file_mb:.1f} MB"],
        ["Prompts", f"{b.n_prompts} × {b.repetitions} repetitions"],
        ["Prompt processing (p50)", f"{b.prompt_tps_p50:.1f} tokens/s"],
        ["Generation (p50 / p95)", f"{b.gen_tps_p50:.1f} / {b.gen_tps_p95:.1f} tokens/s"],
        ["Time to first token (p50)", f"{b.ttft_ms_p50} ms"],
        ["Answer latency (p50 / p95)", f"{b.answer_latency_ms_p50} / {b.answer_latency_ms_p95} ms"],
        ["Peak memory", f"{b.peak_rss_mb:.1f} MB"],
        *[[f"llama-bench {k}", f"{v:.1f} tokens/s"] for k, v in sorted(b.llama_bench.items())],
    ]
    return table(["", "Value"], rows)


def _counts(d: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))


def training_data(src: Sources) -> str:
    d, t = src.data, src.targets
    split_rows = [[s, _counts(c)] for s, c in sorted(d.counts.items())]
    mix = [[s, _counts(q)] for s, q in sorted(t["quotas"].items())]
    drops = {**t.get("row_drops", {}), **t.get("sentence_drops", {})}
    return "\n".join(
        [
            f"PubMedQA `{d.dataset}` at revision `{d.dataset_revision}`: the artificial set "
            "(PQA-A) for training and dev, never the expert set (PQA-L), which is the benchmark. "
            "Rows overlapping PQA-L by PMID, question or context were removed "
            f"({_counts(d.drops)}).",
            "",
            table(["Split", "Rows by reference label"], split_rows),
            "",
            f"Targets: {t['train']} training and {t['val']} validation examples from "
            f"{t['rows_scored']} scored rows. Each answer cites only sentences that support it "
            f"(support ≥ {f3(t['thresholds']['support_min'])}, entailment ≥ "
            f"{f3(t['thresholds']['entail_min'])}, at most {t['thresholds']['max_cites']} "
            f"citations, at most {t['thresholds']['max_words']} words).",
            "",
            "Label mix:",
            "",
            table(["Split", "Examples by kind"], mix),
            "",
            "Filters and their drop counts:",
            "",
            table(["Filter", "Dropped"], [[k, str(v)] for k, v in sorted(drops.items())]),
            "",
            f"Hand audit: {f3(src.bundle.audit_pass_rate)} of the sampled training examples "
            "passed (Tech Spec §4.6).",
        ]
    )


def training_procedure(src: Sources) -> str:
    r, tr = src.recipe, src.bundle.train
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
    insufficient = sum(int(q.get("insufficient", 0)) for q in src.targets["quotas"].values())
    if insufficient:
        lines.append(
            f"- The `maybe` label was learned from {insufficient} constructed examples (an "
            "abstract with its deciding sentences removed, OWNER D-15), not from real "
            "maybe-labelled questions."
        )
    for p in src.probes():
        acc = ", ".join(f"{m.model} {iv(m.accuracy)}" for m in p.models)
        lines.append(f"- Contamination probe `{p.task}` (run `{p.run_id}`): accuracy {acc}.")
    if not src.probes():
        lines.append("- Contamination probe: no probe run is recorded in this bundle.")
    return "\n".join(lines)


def summary(src: Sources) -> str:
    main = src.main()
    by = {m.model: m for m in main.models}
    ft, base = by.get(src.ft_model), by.get(src.paths.base_model)
    against = "against its own base model and two API models."
    earlier = src.earlier()
    if earlier:
        skip = {src.ft_model, src.paths.base_model}
        label = {r.model: f"{r.version} (`{r.model}`)" for r in earlier}
        others = [label.get(m.model, f"`{m.model}`") for m in main.models if m.model not in skip]
        names = ", ".join(others)
        prev = ", ".join(r.version for r in earlier)
        against = (
            f"against its own base model{f' and {names}' if names else ''}, all scored in one "
            f"run with the same claim extractor and judges. The published {prev} numbers are "
            "shown beside them, labelled with their own extractor and judges, and not compared."
        )
    lines = [
        f"`{src.recipe.base_model}` fine-tuned with LoRA to answer a research question from a "
        "supplied abstract only, as JSON with a label, a short answer and the numbers of the "
        "sentences it relied on. It is trained on PubMedQA's artificial set and scored with "
        f"EvalKit on the expert set, {against}"
    ]
    if ft is not None:
        line = (
            f"On {ft.n} expert questions the fine-tune (`{ft.model}`) has accuracy "
            f"{iv(ft.accuracy)}"
        )
        if base is not None:
            line += f" against {iv(base.accuracy)} for the base model (`{base.model}`)"
        line += "; the full table, with the judges' headline, is under Evaluation."
        lines.append(line)
    return " ".join(lines)


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
                    "source": {"name": f"EvalKit run {main.run_id}", "url": "results_bundle.json"},
                }
            ],
        }
    ]


def metadata(src: Sources) -> dict[str, Any]:
    return {
        "license": HUB_LICENSE,
        "base_model": src.recipe.base_model,
        "datasets": [src.data.dataset],
        "language": list(LANGUAGE),
        "pipeline_tag": PIPELINE_TAG,
        "library_name": "transformers",
        "tags": list(TAGS),
        "model-index": model_index(src),
    }


def metadata_problems(meta: Mapping[str, Any]) -> list[str]:
    """Structural checks on the Hub metadata: the keys §10.1 names and a well-formed
    model-index (each result has a task type, a dataset type and name, numeric metrics)."""
    problems = [f"missing {k}" for k in REQUIRED_META if not meta.get(k)]
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
    return problems


def split_front_matter(card: str) -> tuple[dict[str, Any], str]:
    if not card.startswith("---\n"):
        raise ValueError("the card has no front matter")
    head, _, body = card[4:].partition("\n---\n")
    return yaml.safe_load(head), body


def render(src: Sources, template: str | None = None) -> str:
    tpl = Template(template if template is not None else TEMPLATE.read_text(encoding="utf-8"))
    front = yaml.safe_dump(metadata(src), sort_keys=False, allow_unicode=True).rstrip("\n")
    return tpl.substitute(
        front_matter=front,
        summary=summary(src),
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
        evalkit_version=src.bundle.evalkit_version,
    )
