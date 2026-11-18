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


def stamps(src: Sources) -> str:
    b = src.bundle
    lines = [f"- Judge mode: **{b.judge_mode}**. Calibration: **{b.calibration_status}**."]
    lines += [f"- ⚠ {s}" for s in stamp_lines(src.main())]
    if b.gemma_family_check:
        checks = ", ".join(f"{j} {f3(a)}" for j, a in sorted(b.gemma_family_check.items()))
        lines.append(f"- Gemma-family judge check (agreement on owner-labelled claims): {checks}.")
    return "\n".join(lines)


def primary_table(src: Sources) -> str:
    rows = [[name, iv(i)] for name, i in sorted(src.bundle.primary.items())]
    if not rows:
        return "No pre-registered comparison is recorded in this bundle."
    return "Differences with 95% paired bootstrap intervals (Tech Spec §7.4).\n\n" + table(
        ["Comparison", "Difference"], rows
    )


def behaviour_table(src: Sources) -> str:
    rows = [
        [r.rule, r.model, r.metric, value(r.value), str(r.n), f"`{r.source}`"]
        for r in src.bundle.behaviour
    ]
    if not rows:
        return "No behaviour row is recorded in this bundle."
    return table(["Rule", "Model", "Metric", "Value", "n", "Source"], rows)


def parity(src: Sources) -> str:
    p = src.bundle.parity
    lines = [
        f"The demo runs merged bf16 weights; the benchmark scores the Q4_K_M GGUF. On {p.n} dev "
        f"prompts, greedy, the two agree on the label for {f3(p.label_agreement)} of prompts and "
        f"cite exactly the same sentences for {f3(p.citation_exact_match)}.",
        "",
        table(["Weights", "Schema valid"], [[k, f3(v)] for k, v in sorted(p.schema_valid.items())]),
    ]
    if p.judged_headline_delta:
        lines += ["", "Judged headline difference (bf16 minus Q4_K_M), 95% interval:", ""]
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
    ft, base = by.get(src.paths.ft_model), by.get(src.paths.base_model)
    lines = [
        f"`{src.recipe.base_model}` fine-tuned with LoRA to answer a research question from a "
        "supplied abstract only, as JSON with a label, a short answer and the numbers of the "
        "sentences it relied on. It is trained on PubMedQA's artificial set and scored with "
        "EvalKit on the expert set, against its own base model and two API models."
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
    ft = next((m for m in main.models if m.model == src.paths.ft_model), None)
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
        model_table=model_table(src.main()),
        primary_table=primary_table(src),
        behaviour_table=behaviour_table(src),
        parity=parity(src),
        bench=bench(src),
        limits=limits(src),
        evalkit_version=src.bundle.evalkit_version,
    )
