"""PUB-01..04: the write-up, generated from the results bundle and committed evidence (D-20).

`ggh writeup` renders four files under `docs/writeup/`:

- `how-i-built-it.md` (PUB-01) and `project-page.md` (PUB-02) from number-free templates in
  `templates/writeup/`, whose `${name}` placeholders are filled from `facts()`;
- `decisions-summary.md` (PUB-03) from `DECISIONS`, each line's date and owner mark read from
  `docs/decisions.md`, so an id that does not exist fails the build;
- `video-script.md` (PUB-04) from `SEGMENTS`, whose timestamps are summed from their durations.

The prose is hand-written; every number in it is read here from the bundle (`Sources`) or a
committed evidence file (the aim-rule count, the GPU and spend ledgers, the targets reports,
the build graph, the converge gates, the changelog and the release records). `--check` fails
when a committed file is not the generator's output. `prd-one-page.md` is hand-written and its
numbers are checked against the PRD by a test.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from ggh.release import links
from ggh.release.card import DISPLAY, aim_outcome, f3, iv, outcome_table, shipped_bench
from ggh.release.readme import disclosures
from ggh.release.sources import Sources
from ggh.schemas import Comparison, ModelResult, TrainRun

OUT_DIR = Path("docs") / "writeup"
TEMPLATES = Path(__file__).parent / "templates" / "writeup"
DECISIONS_FILE = Path("docs") / "decisions.md"
GRAPH = Path("docs") / "graph" / "graph.md"
CATALOG = Path("docs") / "graph" / "catalog.tsv"
CONVERGE = Path("scripts") / "node_converge.py"
CHANGELOG = Path("changelog.md")
GPU_LEDGER = Path("docs") / "measurements" / "gpu-ledger.jsonl"
TARGETS_V1 = Path("docs") / "measurements" / "targets_drop_report_v1.json"
PARITY_Q8 = Path("docs") / "measurements" / "v2" / "parity_q8_0.json"
RELEASE_V1 = Path("docs") / "measurements" / "release_v1.json"
UPSTREAM = Path("docs") / "measurements" / "evalkit_upstream_r44.json"
RELEASE_RECORD = Path("docs") / "measurements" / "release_record_v2.json"
PRD_SUMMARY = "prd-one-page.md"
# every committed file a write-up number may come from, besides the bundle and its inputs
EVIDENCE = (
    "docs/measurements/aim_rule_check.json",
    TARGETS_V1.as_posix(),
    PARITY_Q8.as_posix(),
    RELEASE_V1.as_posix(),
    UPSTREAM.as_posix(),
    RELEASE_RECORD.as_posix(),
    "docs/measurements/v2/laptop_bench.json",
)
TRAIN_KERNELS = {"v1": "ggh-t22-train v1", "v2": "ggh-v21-train v1"}
MINUS = "\u2212"
NAMES = {**DISPLAY, "gpt-6-luna": "GPT-6 Luna", "claude-haiku-5-5": "Claude Haiku 5.5"}
# the links the note ends with (the project page lists them all)
NOTE_LINKS = (
    "hf_model",
    "hf_gguf",
    "evalkit_report",
    "model_card",
    "case_study",
    "reproduce",
    "known_issues",
    "bundle",
)


class WriteupError(ValueError):
    pass


def display(model: str) -> str:
    return NAMES.get(model, model)


def signed(x: float) -> str:
    """A difference as printed in prose: +0.374, -0.476 (a typographic minus), 0.000."""
    text = f3(x)
    if text.lstrip("-0.") == "":
        return text.lstrip("-")
    return text.replace("-", MINUS) if text.startswith("-") else "+" + text


def signed_iv(c: Comparison) -> str:
    d = c.diff
    lo, hi = (signed(v).lstrip("+") for v in (d.lo, d.hi))
    return f"{signed(d.point)} [{lo}, {hi}]"


def _name(src: Sources, model: str) -> str:
    if model == src.paths.base_model:
        return "base"
    for rel in src.bundle.releases:
        if rel.model == model:
            return rel.version
    return model


def verdict(src: Sources, c: Comparison) -> str:
    if c.verdict == "tie":
        return "a tie"
    who = c.model_a if c.verdict == "a_better" else c.model_b
    return f"separated, {_name(src, who)} better"


def diff(src: Sources, c: Comparison) -> str:
    return f"{signed_iv(c)}, {verdict(src, c)}"


def _result(src: Sources, run: str, model: str) -> ModelResult:
    for r in src.bundle.results:
        if r.run == run and r.model == model:
            return r
    raise WriteupError(f"the bundle has no result for {model} in {run}")


def _comparison(src: Sources, a: str, b: str, metric: str, judge: str | None) -> Comparison:
    for c in (*src.bundle.primary_comparisons, *src.bundle.secondary):
        if (c.model_a, c.model_b, c.metric, c.judge) == (a, b, metric, judge):
            return c
    raise WriteupError(f"the bundle has no {metric} comparison {a} minus {b} ({judge})")


def _review(src: Sources, run: str, model: str, metric: str) -> tuple[int, int]:
    for r in src.bundle.behaviour:
        if r.run == run and r.model == model and r.metric == metric:
            if not isinstance(r.value, int | float):
                break
            return int(r.value), r.n
    raise WriteupError(f"the bundle has no {metric} row for {model} in {run}")


def _json(root: Path, rel: Path | str) -> Any:
    try:
        return json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WriteupError(f"cannot read {rel}: {exc}") from exc


def _text(root: Path, rel: Path | str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError as exc:
        raise WriteupError(f"cannot read {rel}: {exc}") from exc


def harness_counts(root: Path) -> dict[str, int]:
    """Build-graph nodes, catalog rows, converge gates and the verifier rounds the changelog
    records ("green after N verifier rounds")."""
    nodes = re.findall(r"^\| \*\*([A-Z]\d\.\d)\*\* \|", _text(root, GRAPH), re.M)
    rows = [ln for ln in _text(root, CATALOG).splitlines()[1:] if ln.strip()]
    gates = re.findall(r'^\s+\("[a-z-]+", gate_\w+\),$', _text(root, CONVERGE), re.M)
    rounds = [
        int(n) for n in re.findall(r"green after (\d+) verifier rounds?", _text(root, CHANGELOG))
    ]
    if not (nodes and rows and gates and rounds):
        raise WriteupError("cannot count the harness from the graph, catalog, gates or changelog")
    return {
        "nodes": len(nodes),
        "catalog_rows": len(rows),
        "gates": len(gates),
        "verified_nodes": len(rounds),
        "verifier_rounds": sum(rounds),
        "max_rounds": max(rounds),
    }


def _gpu_hours(root: Path, kernel: str) -> float:
    for line in _text(root, GPU_LEDGER).splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kernel") == kernel:
                return float(row["gpu_hours"])
    raise WriteupError(f"{GPU_LEDGER} has no row for {kernel}")


def _count(n: int) -> str:
    return f"{n:,}"


def facts(src: Sources, root: Path) -> dict[str, str]:
    """Every value the templates name, formatted from the bundle or committed evidence."""
    b = src.bundle
    v1, v2 = b.releases[0], b.shipped
    base = src.paths.base_model
    sol, opus = "gpt-6.1-sol", "claude-opus-5-5"
    refs = {r.name: r for r in b.runs}
    main_v1 = src.release_main(v1)
    v1_run = src.ref(main_v1).name
    cite_v1 = src.citation_run(v1)
    v2_run = src.ref(src.main()).name
    if cite_v1 is None:
        raise WriteupError("v1 has no citation run in the bundle")
    r = lambda run, m: _result(src, run, m)  # noqa: E731
    cmp = lambda a, m, metric, j: _comparison(src, a, m, metric, j)  # noqa: E731

    f: dict[str, str] = {}
    f["n"] = _count(r(v2_run, v2.model).n)
    f["base_model"], f["v1_model"], f["v2_model"] = base, v1.model, v2.model
    f["quant"] = shipped_bench(src).quant

    # v1's published run
    f["base_acc"] = f3(r(v1_run, base).accuracy.point)
    f["v1_acc"] = f3(r(v1_run, v1.model).accuracy.point)
    acc = cmp(base, v1.model, "accuracy", None)
    f["v1_acc_verdict"] = verdict(src, acc)
    f["v1_models"] = str(len(main_v1.models))
    f["v1_judges"] = " and ".join(display(j) for j in refs[v1_run].judges)
    f["v1_extractor"] = refs[v1_run].extractor or "?"
    for key, j in (("opus", opus), ("sol", sol)):
        f[f"base_head_{key}"] = f3(r(v1_run, base).headline[j].point)
        f[f"v1_head_{key}"] = f3(r(v1_run, v1.model).headline[j].point)
        f[f"d_v1base_{key}"] = diff(src, cmp(v1.model, base, "headline", j))
    cp, aim = r(cite_v1.name, v1.model), r(cite_v1.name, base)
    assert cp.citation_precision and cp.aim_citation_rate
    assert aim.citation_precision and aim.aim_citation_rate
    f["v1_cp"] = f3(cp.citation_precision[sol].point)
    f["v1_aim"] = f3(cp.aim_citation_rate[sol].point)
    f["base_cp_v1run"] = f3(aim.citation_precision[sol].point)
    f["base_aim_v1run"] = f3(aim.aim_citation_rate[sol].point)
    f["v1_spend"] = f"${refs[v1_run].spent_usd or 0:.2f}"
    f["total_spend"] = f"${b.spend_usd:.2f}"

    # v2's pre-registered run
    for who, m in (("base", base), ("v1", v1.model), ("v2", v2.model)):
        x = r(v2_run, m)
        assert x.citation_precision and x.aim_citation_rate
        f[f"{who}_head_v2run"] = f3(x.headline[sol].point)
        f[f"{who}_cp_v2run"] = f3(x.citation_precision[sol].point)
        f[f"{who}_aim_v2run"] = f3(x.aim_citation_rate[sol].point)
        f[f"{who}_acc_v2run"] = f3(x.accuracy.point)
    f["v2_grounded"] = f3(r(v2_run, v2.model).groundedness_mean[sol].point)
    for key, other in (("v2v1", v1.model), ("v2base", base)):
        for metric in ("headline", "citation_precision", "aim_citation_rate"):
            short = {"headline": "head", "citation_precision": "cp", "aim_citation_rate": "aim"}
            f[f"d_{key}_{short[metric]}"] = diff(src, cmp(v2.model, other, metric, sol))
    fam = cmp(v2.model, v1.model, "headline", sol).family_size
    f["family"] = str(fam or len([c for c in b.primary_comparisons if c.plan == v2.plan]))
    f["v2_spend"] = f"${refs[v2_run].spent_usd or 0:.2f}"
    f["v2_extractor"] = display(refs[v2_run].extractor or "?")
    f["v2_judge"] = " and ".join(display(j) for j in refs[v2_run].judges)

    # OWNER D-62: the post-hoc aim-rule count
    if src.aim_check is None:
        raise WriteupError("the aim-rule count does not cover the shipped release")
    corrected = src.aim_check["citation"][v2_run][base][sol]["corrected"]
    f["base_aim_corrected"] = f3(corrected["aim_citation_rate"])
    f["base_cp_corrected"] = f3(corrected["citation_precision"])
    mine = src.aim_check["citation"][v2_run][v2.model][sol]["corrected"]
    f["v2_aim_corrected"] = f3(mine["aim_citation_rate"])
    f["v2_cp_corrected"] = f3(mine["citation_precision"])
    worse, _ = aim_outcome(src)
    f["aim_worse"] = " and ".join(worse) or "neither citation outcome"

    # calibration (OWNER D-59, D-60)
    cal = {c.judge: c for c in b.calibration}
    f["sol_kappa"] = iv(cal[sol].kappa)
    f["cal_n"] = str(cal[sol].n_claims)
    f["cal_bar"] = f"{cal[sol].bar:.1f}"
    f["sol_status"] = cal[sol].status
    f["opus_kappa"] = f3(cal[opus].kappa.point)
    if b.labeller_consistency is None:
        raise WriteupError("the bundle has no labeller relabel")
    f["relabel_kappa"] = f3(b.labeller_consistency.kappa.point)
    fam_check = cal[sol].gemma_family
    if fam_check is None:
        raise WriteupError(f"{sol} has no Gemma-family check")
    f["family_kappa"] = f3(fam_check.kappa.point)
    f["family_n"] = str(fam_check.n_claims)

    # model reviews (OWNER D-61)
    off_v1 = next(n for n in v1.runs if refs[n].task == "ggh_offscope")
    off_v2 = next(n for n in v2.runs if refs[n].task == "ggh_offscope")
    f["b8_base"], f["b8_n"] = map(str, _review(src, off_v1, base, "gives_advice_count"))
    f["b8_v1"] = str(_review(src, off_v1, v1.model, "gives_advice_count")[0])
    f["b8_v2"] = str(_review(src, off_v2, v2.model, "gives_advice_count")[0])
    f["read"] = str(sum(s.read for s in b.sample_read))
    f["faults"] = str(sum(s.findings.get("pipeline_fault", 0) for s in b.sample_read))

    # data, targets and training
    f["leak_drops"] = str(sum(src.data.drops.values()))
    t1 = _json(root, TARGETS_V1)
    f["v1_train"] = _count(t1["train"])
    f["v1_answers"] = _count(t1["quotas"]["train"]["yes"] + t1["quotas"]["train"]["no"])
    f["v1_insufficient"] = _count(t1["quotas"]["train"]["insufficient"])
    f["v2_train"] = _count(src.targets["train"])
    f["v1_audit_n"] = str(v1.audits[0].labelled)
    f["v1_audit"] = "passed" if v1.audits[0].passed else "did not pass"
    f["v2_audits"] = str(len(v2.audits))
    failed = sum(not a.passed for a in v2.audits)
    f["v2_audit_result"] = (
        "none passed the bar (the last missed by a small margin)"
        if failed == len(v2.audits)
        else f"{failed} missed the bar"
    )
    for key, rel in (("v1", v1), ("v2", v2)):
        f[f"{key}_steps"] = str(rel.train.steps)
        f[f"{key}_vram"] = str(rel.train.peak_vram_gb)
        f[f"{key}_gpu_hours"] = f"{_gpu_hours(root, TRAIN_KERNELS[key]):.2f}"
    f["gpu"] = v1.train.gpu.removeprefix("Tesla ")
    assert isinstance(v1.train, TrainRun)
    dm = v1.train.dev_metrics
    f["v1_dev_n"] = str(dm.n)
    f["v1_schema"] = f3(dm.schema_valid)
    f["v1_cites"] = f3(dm.citation_valid)

    # export, parity and the laptop
    bench = shipped_bench(src)
    f["chip"], f["ram"] = bench.chip, str(bench.ram_gb)
    f["file_mb"] = f"{bench.model_file_mb:.1f}"
    f["gen_tps"] = f"{bench.gen_tps_p50:.1f}"
    f["latency"] = str(bench.answer_latency_ms_p50)
    f["rss"] = f"{bench.peak_rss_mb:.1f}"
    f["tg128"] = f"{bench.llama_bench['tg128']:.0f}" if bench.llama_bench else "?"
    v1_bench = b.bench_variants.get("ft-Q4_K_M")
    if v1_bench is None or v1_bench.llama_bench is None:
        raise WriteupError("the bundle has no v1 Q4_K_M laptop bench")
    f["v1_gen_tps"] = f"{v1_bench.gen_tps_p50:.1f}"
    f["v1_tg128"] = f"{v1_bench.llama_bench['tg128']:.0f}"
    p = v2.parity
    if p is None:
        raise WriteupError("v2 has no parity record")
    f["parity"], f["parity_n"] = f3(p.label_agreement), str(p.n)
    f["q8_parity"] = f3(_json(root, PARITY_Q8)["label_agreement"])
    jd = (p.judged_headline_delta or {}).get(sol)
    f["parity_judged"] = iv(jd) if jd else "not measured"

    # the dev check (D-52) and its misses (OWNER D-55)
    d = src.dev_check
    if d is None:
        raise WriteupError("the shipped release has no dev check")
    m = next(x for x in d["summary"]["models"] if x["model"] == v2.model)
    bars, baseline = d["bars"]["bars"], d["bars"]["baseline"]
    k = list(d["bars"]["judges"]).index(sol)
    f["dev_n"] = str(m["n"])
    f["dev_cp"] = f3(m["citation_precision"][sol]["point"])
    f["dev_aim"] = f3(m["aim_citation_rate"][sol]["point"])
    f["dev_head"] = f3(m["headline"][sol]["point"])
    f["dev_acc"] = f3(m["accuracy"]["point"])
    f["bar_cp"] = f"{bars['citation_precision_min']:.2f}"
    f["bar_aim"] = f"{bars['aim_citation_rate_max']:.2f}"
    f["bar_acc"] = f"{bars['accuracy_min']:.2f}"
    f["dev_v1_head"] = f3(baseline["v1"]["headline"][k])
    f["dev_v1_cp"] = f3(baseline["v1"]["citation_precision"][k])
    f["dev_v1_aim"] = f3(baseline["v1"]["aim_citation_rate"][k])
    f["dev_v1_acc"] = f3(baseline["v1"]["accuracy"])

    # the harness and the release
    f.update({k: str(v) for k, v in harness_counts(root).items()})
    rel1 = _json(root, RELEASE_V1)
    f["tag"], f["release_url"] = rel1["tag"], rel1["release"]
    up = _json(root, UPSTREAM)
    f["evalkit_pr"], f["evalkit_pr_state"] = up["pr"], str(up["pr_state"]).lower()
    f["evalkit_page"] = up["page"]
    rec = _json(root, RELEASE_RECORD)
    f["ollama_command"] = rec["ollama"]["command"]
    f["hf_model"] = links.LINKS["hf_model"][1] or ""
    f["hf_gguf"] = links.LINKS["hf_gguf"][1] or ""
    f["gguf_repo"] = f["hf_gguf"].removeprefix("https://huggingface.co/")
    return f


# --- rendering ----------------------------------------------------------------------------

HEADER = (
    "<!-- generated by `ggh writeup` from the results bundle and committed evidence; "
    "edit {source}, not this file -->"
)


def rel_link(url: str) -> str:
    """A links-registry target as seen from docs/writeup/."""
    if url.startswith(("https://", "http://")):
        return url
    return f"../../{url}"


def links_list(names: tuple[str, ...] | None = None) -> str:
    lines = []
    for name, (label, url) in links.LINKS.items():
        if name == "writeup" or (names is not None and name not in names):
            continue
        lines.append(f"- [{label}]({rel_link(url)})" if url else f"- {label} {links.PENDING}")
    return "\n".join(lines)


def fill(template: str, values: dict[str, str]) -> str:
    try:
        return Template(template).substitute(values)
    except (KeyError, ValueError) as exc:
        raise WriteupError(f"template placeholder problem: {exc}") from exc


def _template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def how_i_built_it(src: Sources, root: Path, values: dict[str, str] | None = None) -> str:
    v = dict(values or facts(src, root))
    v["header"] = HEADER.format(source="src/ggh/release/templates/writeup/how_i_built_it.md")
    v["links"] = links_list(NOTE_LINKS)
    return fill(_template("how_i_built_it.md"), v)


def project_page(src: Sources, root: Path, values: dict[str, str] | None = None) -> str:
    v = dict(values or facts(src, root))
    v["header"] = HEADER.format(source="src/ggh/release/templates/writeup/project_page.md")
    v["table"] = outcome_table(src, src.ref(src.main()), bold=src.ft_model)
    v["caveats"] = "\n".join(f"- {c}." for c in disclosures(src))
    v["links"] = links_list()
    return fill(_template("project_page.md"), v)


# --- PUB-03: decisions that shaped the result ----------------------------------------------

# (decision id as in docs/decisions.md, one plain line; a phrase in double quotes must appear
# word for word in that decision's row)
DECISIONS: tuple[tuple[str, str], ...] = (
    (
        "D-2",
        "Train on EvalKit's own PubMedQA prompt, so the benchmark measures grounding, not "
        "a new format.",
    ),
    (
        "D-3",
        "Answer targets are the author's conclusion, filtered sentence by sentence for "
        "support, with advice removed.",
    ),
    (
        "D-4",
        "Choose checkpoints on the dev split only; the expert set never chooses anything.",
    ),
    (
        "D-5",
        "Score the shipped artifact: the quantised GGUF served by llama-server, for base "
        "and fine-tune alike.",
    ),
    (
        "D-11",
        'Build with the lean Kova harness "exactly as EvalKit was": graph, catalog, '
        "gates, skeptical verifier.",
    ),
    (
        "OWNER D-15",
        'Teach "maybe" with context-ablation examples, since the training set has only yes and no.',
    ),
    (
        "OWNER D-17",
        "Evaluate on every expert-labelled question, not a sample.",
    ),
    (
        "D-20",
        "Every number in the card, README and project page is generated from the results bundle.",
    ),
    (
        "D-41",
        "Run llama-server with the reasoning budget at zero, after the fine-tune opened a "
        "hidden thinking channel.",
    ),
    (
        "OWNER D-43",
        "After v1 cited aim sentences, build a v2, labelled as built after v1's test "
        "results were seen.",
    ),
    (
        "OWNER D-48",
        "Choose citations by per-sentence support, ranked by entailment rather than word overlap.",
    ),
    (
        "OWNER D-50",
        "Train v2 although its third target audit missed the bar by a small margin, and "
        "disclose the miss.",
    ),
    (
        "OWNER D-55",
        "Release v2 with its dev headline, label-accuracy and parity misses disclosed, "
        "not tuned away.",
    ),
    (
        "OWNER D-57",
        'Run the v2 benchmark on OpenAI only after the Anthropic credit ran out ("Use '
        'Open AI API").',
    ),
    (
        "OWNER D-60",
        "Calibration labels come from Claude Opus 5.5 under a committed prompt, and every "
        "report says so.",
    ),
    (
        "OWNER D-61",
        'The off-scope and sample reviews are blind Claude Opus 5.5 reviews ("Model '
        'reviews both").',
    ),
    (
        "OWNER D-62",
        "Keep the pre-registered numbers, disclose the aim rule, and drop any "
        "v2-over-base citation claim.",
    ),
    (
        "OWNER D-63",
        "The demo is the local app against the benchmarked GGUF, not a paid ZeroGPU Space.",
    ),
)


@dataclass(frozen=True)
class DecisionRow:
    id: str
    date: str
    decision: str
    row: str


def decision_rows(text: str) -> dict[str, DecisionRow]:
    out: dict[str, DecisionRow] = {}
    for line in text.splitlines():
        m = re.match(r"^\| ((?:OWNER )?D-\d+) \| (\d{4}-\d{2}-\d{2}) \| (.*?) \|", line)
        if m:
            out[m.group(1)] = DecisionRow(m.group(1), m.group(2), m.group(3), line)
    return out


def quotes(line: str) -> list[str]:
    return re.findall(r'"([^"]+)"', line)


def decisions_summary(root: Path) -> str:
    rows = decision_rows(_text(root, DECISIONS_FILE))
    lines = []
    for did, text in DECISIONS:
        row = rows.get(did)
        if row is None:
            raise WriteupError(f"{did} is not in {DECISIONS_FILE}")
        missing = [q for q in quotes(text) if q not in row.row]
        if missing:
            raise WriteupError(f"{did} does not say {missing[0]!r}")
        lines.append(f"- **{did}** ({row.date}): {text}")
    return (
        "# Decisions that shaped the result\n\n"
        + HEADER.format(source="DECISIONS in src/ggh/release/writeup.py")
        + "\n\n"
        "The project recorded every decision in [docs/decisions.md](../decisions.md) "
        "(`OWNER` marks a ruling the owner made). These are the ones that most shaped what was "
        "built and what the numbers mean, one line each. What the project is for: the "
        f"[one-page PRD summary]({PRD_SUMMARY}) and the full "
        "[PRD](../../Gemma%20Grounded%20Health%20PRD.md).\n\n" + "\n".join(lines) + "\n"
    )


# --- PUB-04: the video script ---------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    seconds: int
    visual: str
    narration: str  # a template over facts()


SEGMENTS: tuple[Segment, ...] = (
    Segment(
        8,
        "Title card over `docs/assets/screenshots/demo_answer.png`; the research-only notice "
        "on screen",
        "Can a small open model answer a health research question only from the abstract in "
        "front of it, and show which sentences it used? Research only, not medical advice.",
    ),
    Segment(
        12,
        "`docs/assets/screenshots/demo_answer.png`: an abstract with its numbered sentences, "
        "base and v2 answer cards, the cited sentence highlighted",
        "The model is Gemma 4 E2B, tuned with LoRA on a free Kaggle ${gpu}. It reads numbered "
        "sentences and answers with a label, a short answer and the sentences it relied on.",
    ),
    Segment(
        12,
        "`docs/assets/architecture.svg`, then `docs/assets/screenshots/evalkit_report_main_v2.png`",
        "Every model is scored with EvalKit on all ${n} expert-labelled PubMedQA questions. The "
        "score that counts: the label is right and every claim is supported by what it cites.",
    ),
    Segment(
        15,
        "`docs/assets/screenshots/v1_vs_v2_citation.png`: the same answer, v1 citing the aim "
        "sentence, v2 citing the result",
        "Version one raised label accuracy from ${base_acc} to ${v1_acc}, but it cited the "
        "study's aim instead of its results, so it fell below its own base on that score.",
    ),
    Segment(
        15,
        "`docs/assets/headline_v2.svg`: base, v1 and v2 with their intervals",
        "Version two was retrained on citations that each support the answer. Citation "
        "precision rose from ${v1_cp_v2run} to ${v2_cp_v2run}. The headline did not move: "
        "${v2_head_v2run} against v1's ${v1_head_v2run}, and base is still better at "
        "${base_head_v2run}.",
    ),
    Segment(
        18,
        "Offline laptop: `docs/assets/demo.gif` (the local demo against the ${quant} GGUF), "
        "then `docs/assets/ollama_offline.gif`: inside a macOS sandbox that blocks the "
        "internet, the published hf.co/${gguf_repo}:${quant} file, set to greedy decoding "
        "by a Modelfile, answers a sample through `ollama run --think=false`",
        "Offline on a laptop, the ${quant} file answers through llama-server or Ollama, with "
        "thinking off. On an ${chip} it generates ${gen_tps} tokens a second. The demo serves "
        "exactly the file that was benchmarked.",
    ),
    Segment(
        10,
        "`docs/assets/screenshots/hf_model_page.png`, then the links: Hugging Face, the "
        "GitHub release, the EvalKit report",
        "One judge scored version two, ${v2_judge}, calibrated against a Claude Opus 5.5 "
        "reference labeller, not a human. The model, the numbers and the recipe are all public.",
    ),
)


def video_script(src: Sources, root: Path, values: dict[str, str] | None = None) -> str:
    v = values or facts(src, root)
    total = sum(s.seconds for s in SEGMENTS)
    rows, t = [], 0
    for s in SEGMENTS:
        start, t = t, t + s.seconds
        stamp = f"{start // 60}:{start % 60:02d}–{t // 60}:{t % 60:02d}"
        rows.append(f"| {stamp} | {fill(s.visual, v)} | {fill(s.narration, v)} |")
    words = sum(len(fill(s.narration, v).split()) for s in SEGMENTS)
    return (
        "# Video script\n\n"
        + HEADER.format(source="SEGMENTS in src/ggh/release/writeup.py")
        + "\n\n"
        f"About {total} seconds, {words} words of narration. Every visual is a recorded "
        "asset in `docs/assets/`. "
        "Every claim is one the results bundle or the model card makes.\n\n"
        "| Time | Visual | Narration |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )


# --- the command ----------------------------------------------------------------------------

FILES: dict[str, Callable[[Sources, Path, dict[str, str]], str]] = {
    "how-i-built-it.md": how_i_built_it,
    "project-page.md": project_page,
    "decisions-summary.md": lambda _s, root, _v: decisions_summary(root),
    "video-script.md": video_script,
}


def render_all(src: Sources, root: Path) -> dict[str, str]:
    values = facts(src, root)
    return {name: fn(src, root, values) for name, fn in FILES.items()}


def stale(src: Sources, root: Path, out_dir: Path = OUT_DIR) -> list[str]:
    """The generated files whose committed text is not the generator's output."""
    out = []
    for name, text in render_all(src, root).items():
        f = root / out_dir / name
        if not f.is_file() or f.read_text(encoding="utf-8") != text:
            out.append((out_dir / name).as_posix())
    return out


def write(src: Sources, root: Path, out_dir: Path = OUT_DIR) -> list[str]:
    (root / out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_all(src, root).items():
        (root / out_dir / name).write_text(text, encoding="utf-8")
        written.append((out_dir / name).as_posix())
    return written
