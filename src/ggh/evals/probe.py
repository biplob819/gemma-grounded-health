"""RUN-06, D-42: the paraphrase probe set, built by EvalKit's documented DAT-06 method.

100 PQA-L questions drawn with EvalKit's stratified sampler at seed 7 (55 yes, 34 no, 11 maybe).
Each is rewritten once by `gpt-6.1-terra` (not in the comparison), keeping meaning, entities and
numbers; Claude Opus 5.5 (the primary judge) checks, given the abstract, that the paraphrase asks
the same question. Only "same" is kept; a rejected question is replaced by the next question of
its label in a seeded order, until 100 are accepted. Two local EvalKit tasks share the abstracts
and differ only in wording. The owner's spot-check (20 kept, all rejected) is done at E3.3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.sampling import stratified_sample

N = 100
SEED = 7
PARAPHRASE_MODEL = "gpt-6.1-terra"
CHECK_MODEL = "claude-opus-5-5"
DATA = Path("data")
PROBE = DATA / "paraphrase_probe.jsonl"
REJECTS = DATA / "paraphrase_probe_rejects.jsonl"
ORIGINAL_DATA = DATA / "probe_original.jsonl"
PARAPHRASE_DATA = DATA / "probe_paraphrase.jsonl"
ORIGINAL_TASK = Path("configs") / "tasks" / "probe_original.yaml"
PARAPHRASE_TASK = Path("configs") / "tasks" / "probe_paraphrase.yaml"

PARAPHRASE_PROMPT = (
    "Rewrite the research question below so that it asks exactly the same thing. Keep its "
    "meaning, every entity, and every number; change the wording and the word order. Reply "
    "with the rewritten question only, on one line.\n\nQuestion: {question}"
)
CHECK_PROMPT = (
    "Here is the abstract of a study, an original research question about it, and a rewritten "
    "version of that question.\n\nAbstract:\n{context}\n\nOriginal: {original}\nRewritten: "
    "{paraphrase}\n\nGiven the abstract, does the rewritten question ask the same question as "
    "the original, so that the same answer (yes, no or maybe) is correct for both? Reply with "
    "one word, SAME or DIFFERENT, then a short reason on the next line."
)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Verdict:
    same: bool
    reason: str


def parse_verdict(text: str) -> Verdict:
    first, _, rest = text.strip().partition("\n")
    word = first.strip().strip("*. ").upper()
    if word not in ("SAME", "DIFFERENT"):
        raise ValueError(f"check reply does not start with SAME or DIFFERENT: {first[:60]!r}")
    return Verdict(word == "SAME", rest.strip())


def queues(
    cases: Sequence[Any], n: int = N, seed: int = SEED
) -> tuple[list[Any], dict[str, list[Any]]]:
    """The first `n` by EvalKit's sampler; per label, the rest in a seeded order (replacements)."""
    first = list(stratified_sample(cases, n, seed).cases)
    chosen = {c.id for c in first}
    order = stratified_sample(cases, len(cases), seed).cases  # every case, seeded per label
    spare: dict[str, list[Any]] = {}
    for c in order:
        if c.id not in chosen:
            spare.setdefault(str(c.reference_label), []).append(c)
    return first, spare


def build(
    cases: Sequence[Any],
    paraphrase: Callable[[str], str],
    check: Callable[[Any, str], Verdict],
    n: int = N,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first, spare = queues(cases, n)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    todo = list(first)
    while todo:
        c = todo.pop(0)
        para = paraphrase(c.question).strip()
        v = check(c, para)
        row = {"pubid": c.id, "reference_label": c.reference_label, "original": c.question,
               "paraphrase": para, "paraphrase_model": PARAPHRASE_MODEL,
               "paraphrase_prompt_hash": prompt_hash(PARAPHRASE_PROMPT),
               "check_verdict": "same" if v.same else "different", "check_model": CHECK_MODEL,
               "check_prompt_hash": prompt_hash(CHECK_PROMPT),
               "check_reason": v.reason}  # fmt: skip
        if v.same and para.lower() != c.question.strip().lower():
            kept.append(row)
        else:
            rejected.append(row)
            label = str(c.reference_label)
            if not spare.get(label):
                raise ValueError(f"no replacement left for label {label!r}")
            todo.append(spare[label].pop(0))
    order = {c.id: i for i, c in enumerate(first)}
    kept.sort(key=lambda r: order.get(r["pubid"], len(order)))
    return kept, rejected


def task_cases(kept: Sequence[dict[str, Any]], cases: Sequence[Any], wording: str
               ) -> list[dict[str, Any]]:  # fmt: skip
    by_id = {c.id: c for c in cases}
    out = []
    for r in kept:
        c = by_id[r["pubid"]]
        q = r["original"] if wording == "original" else r["paraphrase"]
        out.append({"id": f"probe-{c.id}", "question": q, "context": list(c.context),
                    "label": c.reference_label})  # fmt: skip
    return out


def jsonl(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


def write(root: Path, kept: Sequence[dict[str, Any]], rejected: Sequence[dict[str, Any]],
          cases: Sequence[Any]) -> None:  # fmt: skip
    from ggh.evals.tasks import task_yaml

    (root / DATA).mkdir(parents=True, exist_ok=True)
    (root / PROBE).write_text(jsonl(kept), encoding="utf-8")
    (root / REJECTS).write_text(jsonl(rejected), encoding="utf-8")
    for wording, data, task in (("original", ORIGINAL_DATA, ORIGINAL_TASK),
                                ("paraphrase", PARAPHRASE_DATA, PARAPHRASE_TASK)):  # fmt: skip
        (root / data).write_text(jsonl(task_cases(kept, cases, wording)), encoding="utf-8")
        (root / task).parent.mkdir(parents=True, exist_ok=True)
        (root / task).write_text(task_yaml(f"ggh_probe_{wording}", f"../../{data.as_posix()}",
                                           f"probe_{wording}"), encoding="utf-8")  # fmt: skip


@dataclass
class Spend:
    usd: float = 0.0
    calls: int = 0


def live_callers(
    spend: Spend,
) -> tuple[Callable[[str], str], Callable[[Any, str], Verdict]]:  # pragma: no cover - paid APIs
    """The two paid calls, costed with EvalKit's price table."""
    import anthropic  # pyright: ignore[reportMissingImports]
    import openai  # pyright: ignore[reportMissingImports]
    from evalkit.cost import call_cost, load_prices, price_for

    prices = load_prices()
    oa: Any = openai.OpenAI()
    an: Any = anthropic.Anthropic()

    def paraphrase(question: str) -> str:
        msg = [{"role": "user", "content": PARAPHRASE_PROMPT.format(question=question)}]
        r = oa.chat.completions.create(model=PARAPHRASE_MODEL, reasoning_effort="low",
                                       messages=msg)  # fmt: skip
        u = r.usage
        reasoning = (
            getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        )
        spend.usd += call_cost(price_for(prices, PARAPHRASE_MODEL), tokens_in=u.prompt_tokens,
                               tokens_out=max(u.completion_tokens - reasoning, 0),
                               tokens_reasoning=reasoning, batch=False)  # fmt: skip
        spend.calls += 1
        return (r.choices[0].message.content or "").strip().splitlines()[0]

    def check(case: Any, para: str) -> Verdict:
        ctx = "\n".join(f"[{i}] {s}" for i, s in enumerate(case.context, 1))
        body = CHECK_PROMPT.format(context=ctx, original=case.question, paraphrase=para)
        r = an.messages.create(model=CHECK_MODEL, max_tokens=200,
                               messages=[{"role": "user", "content": body}])  # fmt: skip
        spend.usd += call_cost(price_for(prices, CHECK_MODEL), tokens_in=r.usage.input_tokens,
                               tokens_out=r.usage.output_tokens, tokens_reasoning=0,
                               tokens_cached=0, batch=False)  # fmt: skip
        spend.calls += 1
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return parse_verdict(text)

    return paraphrase, check


# ---- the contamination check (EvalKit DATA_CARD: paired bootstrap, 10,000 resamples, seed 7) ----

REPORT = Path("docs") / "measurements" / "probe_report.json"
RESAMPLES = 10_000


def correct_by_case(records: Sequence[dict[str, Any]], model: str) -> dict[str, bool]:
    return {r["case_id"]: bool(r["scores"]["label_correct"]) for r in records
            if r["completion"]["model"] == model}  # fmt: skip


def paired_drop(orig: dict[str, bool], para: dict[str, bool], seed: int = SEED,
                resamples: int = RESAMPLES) -> dict[str, Any]:  # fmt: skip
    """Accuracy on the original wording minus the paraphrased, on shared cases, with a paired
    bootstrap 95% interval. A drop whose interval excludes zero is a contamination signal."""
    import random

    ids = sorted(set(orig) & set(para))
    if len(ids) != len(orig) or len(ids) != len(para) or not ids:
        raise ValueError("the two runs must cover the same cases")
    d = [int(orig[i]) - int(para[i]) for i in ids]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo, hi = boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]
    point = sum(d) / n
    return {"n": n, "accuracy_original": sum(orig[i] for i in ids) / n,
            "accuracy_paraphrase": sum(para[i] for i in ids) / n,
            "drop": {"point": point, "lo": lo, "hi": hi},
            "contamination_signal": lo > 0}  # fmt: skip


REPORT_V2 = Path("docs") / "measurements" / "probe_report_v2.json"
REPORT_V3B = Path("docs") / "measurements" / "probe_report_v3b.json"  # V3.8 (OWNER D-75)
METHOD = ("accuracy on original minus paraphrased wording, paired bootstrap 95% interval, "
          "10,000 resamples, seed 7 (EvalKit DATA_CARD, D-42)")  # fmt: skip
# per release: the two EvalKit runs, the models compared, and where the report goes
RELEASES: dict[str, tuple[str, str, tuple[str, ...], Path]] = {
    "v1": ("probe_original", "probe_paraphrase",
           ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4"), REPORT),
    "v2": ("probe_original_v2", "probe_paraphrase_v2", ("gemma-e2b-grounded-v2-q4",), REPORT_V2),
    "v3b": ("probe_original_v3b", "probe_paraphrase_v3b", ("gemma-e2b-grounded-v3b-q4",),
            REPORT_V3B),
}  # fmt: skip


def report(root: Path, release: str = "v1") -> dict[str, Any]:
    """The probe report for a release from its two committed EvalKit run folders: each model's
    `paired_drop`, with both run ids."""
    from ggh.results.runs import Run

    if release not in RELEASES:
        raise ValueError(f"release must be one of {sorted(RELEASES)}, got {release!r}")
    orig_name, para_name, models, _ = RELEASES[release]
    orig, para = Run(root, orig_name), Run(root, para_name)
    o, p = orig.raw_records, para.raw_records
    return {"method": METHOD,
            "runs": {"original": orig.manifest["run_id"], "paraphrase": para.manifest["run_id"]},
            "models": {m: paired_drop(correct_by_case(o, m), correct_by_case(p, m))
                       for m in models}}  # fmt: skip


def write_report(root: Path, release: str = "v1") -> Path:
    out = root / RELEASES[release][3]
    out.write_text(json.dumps(report(root, release), indent=2) + "\n", encoding="utf-8")
    return out
