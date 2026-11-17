"""`ggh targets`: training and validation examples from the N1.2 splits (N1.3).

Every example comes from one N1.2 row and a list of rule ids (PRV-R1); no other text source
exists here. `reconstruct` rebuilds any example from its row with the recorded NLI scores,
which is the executable proof that no Claude, OpenAI or Gemini text entered training.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.schemas import Answer

from ggh.data import build as data
from ggh.lineage import StageManifest, read_manifest, sha256_file, stage_key, write_manifest
from ggh.schemas import TrainExample
from ggh.targets import rules
from ggh.targets.nli import MODEL_ID, MODEL_REVISION, RecordingScorer, ReplayScorer, Scorer

CODE_VERSION = "1"
SEED = 7
TRAIN_CAP, VAL_RATIO = 20_000, 10  # OWNER D-34: 20,000 is a ceiling; val is one tenth of train
SHARE_INSUFFICIENT, SHARE_OFFSCOPE, SHARE_NO = 0.10, 0.01, 0.40  # MIX-R1, MIX-R2
OUT_DIR = Path("artifacts") / "targets"
MEASURE_DIR = Path("docs") / "measurements"
REPORT_COPY = MEASURE_DIR / "targets_drop_report.json"
AUDIT_COPY = MEASURE_DIR / "audit_sample.jsonl"
CHUNK = 2_000  # rows scored per NLI batch while filling yes quotas

Row = dict[str, Any]  # an N1.2 row: pmid, question, sentences, sections, long_answer, label


@dataclass(frozen=True)
class Option:
    cites: tuple[int, ...]
    support: float
    premise: str  # exactly the cited sentences, joined: what the judges will see


@dataclass(frozen=True)
class Candidate:
    sentence: str  # the conclusion sentence after the hedge strip (TGT-R9)
    options: tuple[Option, ...]  # smallest first, each with support >= SUPPORT_MIN (TGT-R3)


@dataclass(frozen=True)
class Choice:
    candidate: Candidate
    option: Option | None  # the smallest option NLI accepts, or None
    entail: float


@dataclass(frozen=True)
class AnswerTarget:
    answer: str
    citations: tuple[int, ...]
    support: float
    entail: float


def _half_up(x: float) -> int:
    return int(x + 0.5)


def candidates(
    row: Row, counts: Counter[str] | None = None, *, exclude_aims: bool = True
) -> list[Candidate]:
    """TGT-R1, R2, R9, R10, R6, R3: the lexically supported conclusion sentences of a row.
    With `exclude_aims` (TGT-R10, v2) no aim sentence is ever a citation option; without it
    the v1 behaviour is rebuilt."""
    counts = counts if counts is not None else Counter()
    ctx = tuple(row["sentences"])
    aims = rules.aim_positions(ctx) if exclude_aims else frozenset()
    out: list[Candidate] = []
    for raw in rules.conclusion_sentences(row["long_answer"]):
        if rules.is_recommendation(raw):
            counts["sentence:recommendation"] += 1
            continue
        s = rules.strip_hedge(raw)
        if not rules.content_words(s):
            counts["sentence:no_content_words"] += 1
            continue
        ranked = rules.ranked_citations(s, ctx)
        if not ranked:
            counts["sentence:zero_overlap"] += 1
            continue
        if aims:
            without = rules.ranked_citations(s, ctx, aims)
            if without != ranked:
                counts["ranking:aim_sentence_removed"] += 1  # not a drop by itself
            if not without:
                counts["sentence:aim_only_overlap"] += 1  # every overlapping sentence is an aim
                continue
            ranked = without
        options = []
        for cites in rules.citation_options(ranked):
            cited = [ctx[i - 1] for i in cites]
            sup = round(rules.support(s, cited), rules.SCORE_DECIMALS)
            if sup >= rules.SUPPORT_MIN:
                options.append(Option(cites, sup, " ".join(cited)))
        if not options:
            counts["sentence:low_support"] += 1
            continue
        out.append(Candidate(s, tuple(options)))
    return out


def choose(cands: Sequence[Candidate], scorer: Scorer) -> list[Choice]:
    """TGT-R4 with TGT-R6 (D-35): per sentence, the smallest citation option whose premise
    entails it. Options are tried in rounds (all first options, then the second options of
    the sentences still open, ...), each round one batched NLI call."""
    chosen: list[Option | None] = [None] * len(cands)
    best = [0.0] * len(cands)
    pending = list(range(len(cands)))
    k = 0
    while pending:
        pairs = [(cands[i].options[k].premise, cands[i].sentence) for i in pending]
        scores = scorer.entail(pairs)
        still: list[int] = []
        for i, sc in zip(pending, scores, strict=True):
            if sc >= rules.ENTAIL_MIN:
                chosen[i], best[i] = cands[i].options[k], sc
            else:
                best[i] = max(best[i], sc)
                if k + 1 < len(cands[i].options):
                    still.append(i)
        pending, k = still, k + 1
    return [Choice(c, o, e) for c, o, e in zip(cands, chosen, best, strict=True)]


def assemble(choices: Sequence[Choice], counts: Counter[str] | None = None) -> AnswerTarget | str:
    """TGT-R4, R6 cap, R5: the target, or the reason the row is dropped."""
    counts = counts if counts is not None else Counter()
    kept: list[tuple[Candidate, Option, float]] = []
    cites: set[int] = set()
    for ch in choices:
        if ch.option is None:
            counts["sentence:low_entailment"] += 1
            continue
        if len(cites | set(ch.option.cites)) > rules.MAX_CITES:  # keep each sentence's support
            counts["sentence:citation_cap"] += 1
            continue
        kept.append((ch.candidate, ch.option, ch.entail))
        cites |= set(ch.option.cites)
    if not kept:
        return "no_supported_sentence"
    answer = " ".join(c.sentence for c, _, _ in kept)
    if len(answer.split()) > rules.MAX_WORDS:
        return "over_80_words"
    return AnswerTarget(
        answer,
        tuple(sorted(cites)),
        min(o.support for _, o, _ in kept),
        min(e for _, _, e in kept),
    )


def targets_for(
    rows: Sequence[Row],
    scorer: Scorer,
    counts: Counter[str] | None = None,
    *,
    exclude_aims: bool = True,
) -> list[AnswerTarget | str]:
    """Targets for many rows, with NLI batched across all of them."""
    cand_lists = [candidates(r, counts, exclude_aims=exclude_aims) for r in rows]
    choices = choose([c for cl in cand_lists for c in cl], scorer)
    out: list[AnswerTarget | str] = []
    k = 0
    for cl in cand_lists:
        out.append(assemble(choices[k : k + len(cl)], counts))
        k += len(cl)
    return out


def _prompt(pmid: str, question: str, sentences: tuple[str, ...], label: str) -> str:
    return data.render(data.Row(pmid, question, sentences, ("",) * len(sentences), "", label))


def answer_example(
    row: Row, t: AnswerTarget, split: str, rule_ids: tuple[str, ...] = rules.ANSWER_RULES
) -> TrainExample:
    ctx = tuple(row["sentences"])
    return TrainExample(
        id=f"pqa-a-{row['pmid']}",
        pmid=row["pmid"],
        kind="answer",
        split=split,  # type: ignore[arg-type]
        question=row["question"],
        context=ctx,
        prompt=_prompt(row["pmid"], row["question"], ctx, row["label"]),
        target=Answer(label=row["label"], answer=t.answer, citations=t.citations),
        source_label=row["label"],
        rule_ids=rule_ids,
        support=t.support,
        nli_entail=t.entail,
    )


def insufficient_example(
    row: Row,
    t: AnswerTarget,
    split: str,
    seed: int,
    rule_ids: tuple[str, ...] = rules.INSUFFICIENT_RULES,
) -> TrainExample | None:
    """INS-R1 to R3, or None when the row cannot make one. Its own citations (INS-R3) are the
    question's best matches and may be aim sentences: TGT-R10 governs answers only."""
    sents, secs = row["sentences"], row["sections"]
    if not any(rules.is_results_section(s) for s in secs):
        return None
    cited = set(t.citations)
    keep = [
        s
        for i, (s, sec) in enumerate(zip(sents, secs, strict=True))
        if not rules.is_results_section(sec)
        and sec not in rules.UNKNOWN_SECTIONS  # may hide results
        and i + 1 not in cited
    ]
    if len(keep) < 2:  # INS-R2
        return None
    ctx = tuple(keep)
    scores = rules.overlaps(row["question"], ctx)
    ranked = sorted(range(len(ctx)), key=lambda i: (-scores[i], i))
    cites = tuple(sorted(i + 1 for i in ranked[:2] if scores[i] > 0)) or (ranked[0] + 1,)
    answer = rules.INSUFFICIENT_ANSWERS[
        rules.pick(seed, row["pmid"], "ins", len(rules.INSUFFICIENT_ANSWERS))
    ]
    return TrainExample(
        id=f"pqa-a-{row['pmid']}:insufficient",
        pmid=row["pmid"],
        kind="insufficient",
        split=split,  # type: ignore[arg-type]
        question=row["question"],
        context=ctx,
        prompt=_prompt(row["pmid"], row["question"], ctx, "maybe"),
        target=Answer(label="maybe", answer=answer, citations=cites),
        source_label=row["label"],
        rule_ids=rule_ids,
        support=0.0,  # not scored: the target is a fixed template
        nli_entail=0.0,
    )


def offscope_example(row: Row, split: str, seed: int) -> TrainExample:
    """OFF-R3: a personal-advice question on a real abstract, declined."""
    q = rules.OFFSCOPE_QUESTIONS[rules.pick(seed, row["pmid"], "offq", 12)]
    a = rules.OFFSCOPE_ANSWERS[rules.pick(seed, row["pmid"], "offa", 6)]
    ctx = tuple(row["sentences"])
    return TrainExample(
        id=f"pqa-a-{row['pmid']}:offscope",
        pmid=row["pmid"],
        kind="offscope",
        split=split,  # type: ignore[arg-type]
        question=q,
        context=ctx,
        prompt=_prompt(row["pmid"], q, ctx, "maybe"),
        target=Answer(label="maybe", answer=a, citations=(1,)),
        source_label=row["label"],
        rule_ids=rules.OFFSCOPE_RULES,
        support=0.0,
        nli_entail=0.0,
    )


def reconstruct(
    row: Row, rule_ids: Sequence[str], *, split: str, seed: int, scorer: Scorer
) -> TrainExample:
    """PRV-R1: rebuild an example from its source row and rule ids alone (v2 ids, or v1's,
    which lack TGT-R10)."""
    ids = tuple(rule_ids)
    if ids == rules.OFFSCOPE_RULES:
        return offscope_example(row, split, seed)
    answer_ids = (rules.ANSWER_RULES, rules.V1_ANSWER_RULES)
    if ids not in (*answer_ids, rules.INSUFFICIENT_RULES, rules.V1_INSUFFICIENT_RULES):
        raise ValueError(f"unknown rule ids {ids}")
    (t,) = targets_for([row], scorer, exclude_aims="TGT-R10" in ids)
    if isinstance(t, str):
        raise ValueError(f"row {row['pmid']} has no target ({t})")
    if ids in answer_ids:
        return answer_example(row, t, split, ids)
    ex = insufficient_example(row, t, split, seed, ids)
    if ex is None:
        raise ValueError(f"row {row['pmid']} cannot make an insufficient example")
    return ex


# ---- the mix (MIX-R1 to R3, OWNER D-34) ----


@dataclass
class Quota:
    no: int
    yes: int
    insufficient: int
    offscope: int

    @classmethod
    def for_size(cls, n: int) -> Quota:
        ins = _half_up(n * SHARE_INSUFFICIENT)
        off = _half_up(n * SHARE_OFFSCOPE)
        ans = n - ins - off
        no = _half_up(ans * SHARE_NO)
        return cls(no=no, yes=ans - no, insufficient=ins, offscope=off)

    def total(self) -> int:
        return self.no + self.yes + self.insufficient + self.offscope

    def unfilled(self) -> bool:
        return self.total() > 0


def plan(no_val_split: int, no_pool: int, cap: int = TRAIN_CAP) -> tuple[int, int]:
    """The largest train size (with val = train / 10) the surviving "no" rows allow."""
    for n in range(cap, 0, -1):
        v = n // VAL_RATIO
        v_from_pool = max(0, Quota.for_size(v).no - no_val_split)
        if Quota.for_size(n).no + v_from_pool <= no_pool:
            return n, v
    return 0, 0


def seeded(rows: Iterable[Row], seed: int) -> list[Row]:
    out = sorted(rows, key=lambda r: int(r["pmid"]))
    random.Random(seed).shuffle(out)
    return out


@dataclass
class Selection:
    train: list[TrainExample]
    val: list[TrainExample]
    report: dict[str, Any]


def select(
    val_rows: Sequence[Row],
    pool_rows: Sequence[Row],
    scorer: Scorer,
    *,
    seed: int = SEED,
    cap: int = TRAIN_CAP,
    chunk: int = CHUNK,
    progress: Callable[[str], None] | None = None,
) -> Selection:
    """Fill val (N1.2 val rows first, then the pool) and train (pool only) in seeded order."""
    counts: Counter[str] = Counter()
    drops: Counter[str] = Counter()
    order = [("val", r) for r in seeded(val_rows, seed)] + [
        ("pool", r) for r in seeded(pool_rows, seed)
    ]
    targets: dict[str, AnswerTarget | str] = {}

    def score(rows: list[Row]) -> None:
        todo = [r for r in rows if r["pmid"] not in targets]
        for r, t in zip(todo, targets_for(todo, scorer, counts), strict=True):
            targets[r["pmid"]] = t
            if isinstance(t, str):
                drops[f"{r['label']}:{t}"] += 1

    # every "no" row is needed to size the mix, so score them all first
    score([r for _, r in order if r["label"] == "no"])
    surviving = {p for p, t in targets.items() if not isinstance(t, str)}
    no_val = sum(r["pmid"] in surviving for src, r in order if src == "val" and r["label"] == "no")
    no_pool = sum(
        r["pmid"] in surviving for src, r in order if src == "pool" and r["label"] == "no"
    )
    n_train, n_val = plan(no_val, no_pool, cap)
    if progress:
        progress(f"no rows surviving: val {no_val}, pool {no_pool}; train {n_train}, val {n_val}")
    quotas = {"val": Quota.for_size(n_val), "train": Quota.for_size(n_train)}
    want = {k: Quota(**vars(q)) for k, q in quotas.items()}
    chosen: dict[str, list[TrainExample]] = {"val": [], "train": []}

    def splits_for(src: str) -> tuple[str, ...]:
        return ("val", "train") if src == "pool" else ("val",)

    def assign(src: str, r: Row) -> None:
        """Answer first, then insufficient, then off-scope; val before train in each, so a
        scarce "no" row is never spent as an off-scope example while it can be an answer."""
        t = targets.get(r["pmid"])
        splits = splits_for(src)
        if isinstance(t, AnswerTarget):
            for split in splits:
                q = quotas[split]
                if getattr(q, r["label"]) > 0:
                    setattr(q, r["label"], getattr(q, r["label"]) - 1)
                    chosen[split].append(answer_example(r, t, split))
                    return
            for split in splits:
                q = quotas[split]
                if q.insufficient > 0:
                    ex = insufficient_example(r, t, split, seed)
                    if ex is not None:
                        q.insufficient -= 1
                        chosen[split].append(ex)
                        return
        for split in splits:
            q = quotas[split]
            if q.offscope > 0:
                q.offscope -= 1
                chosen[split].append(offscope_example(r, split, seed))
                return

    i = 0
    while i < len(order) and any(q.unfilled() for q in quotas.values()):
        batch = order[i : i + chunk]
        # yes rows are scored only while a quota that could take one is open; once they are
        # full, the scan continues only to collect the scarce "no" rows (already scored)
        need_yes = {k for k, q in quotas.items() if q.yes > 0 or q.insufficient > 0}
        score([r for src, r in batch if r["label"] == "yes" and need_yes & set(splits_for(src))])
        for src, r in batch:
            assign(src, r)
        i += chunk
        if progress:
            left = {k: q.total() for k, q in quotas.items()}
            progress(
                f"rows {min(i, len(order)):,}/{len(order):,}  scored {len(targets):,}  left {left}"
            )
    unfilled = {k: vars(q) for k, q in quotas.items() if q.unfilled()}
    if unfilled:
        raise ValueError(f"quotas could not be filled from the available rows: {unfilled}")
    report = {
        "seed": seed,
        "cap_train": cap,
        "train": len(chosen["train"]),
        "val": len(chosen["val"]),
        "shortfall_vs_cap": cap - len(chosen["train"]),
        "quotas": {k: vars(q) for k, q in want.items()},
        "no_rows_surviving": {"val_split": no_val, "train_pool": no_pool},
        "rows_scored": len(targets),
        "row_drops": dict(sorted(drops.items())),
        "sentence_drops": dict(sorted(counts.items())),
        "thresholds": {
            "support_min": rules.SUPPORT_MIN,
            "entail_min": rules.ENTAIL_MIN,
            "max_words": rules.MAX_WORDS,
            "max_cites": rules.MAX_CITES,
        },
        "nli_model": f"{MODEL_ID}@{MODEL_REVISION}",
    }
    for split in ("train", "val"):
        chosen[split].sort(key=lambda e: (int(e.pmid), e.kind))
    return Selection(chosen["train"], chosen["val"], report)


# ---- audit sample (AUD-R1) ----

AUDIT_STRATA = (("answer", "yes", 18), ("answer", "no", 12), ("insufficient", None, 15))
AUDIT_OFFSCOPE = 5


def audit_sample(train: Sequence[TrainExample], seed: int = SEED) -> list[TrainExample]:
    rng = random.Random(seed)
    out: list[TrainExample] = []
    for kind, label, n in AUDIT_STRATA:
        pool = [e for e in train if e.kind == kind and (label is None or e.source_label == label)]
        out += rng.sample(pool, min(n, len(pool)))
    off = [e for e in train if e.kind == "offscope"]
    out += rng.sample(off, min(AUDIT_OFFSCOPE, len(off)))
    return out


# ---- build ----


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n")
    return sha256_file(path)


def _json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def rules_hash() -> str:
    return hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest()


def build(
    root: Path,
    scorer_factory: Callable[[], Scorer],
    *,
    seed: int = SEED,
    cap: int = TRAIN_CAP,
    chunk: int = CHUNK,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write train/val examples, the NLI score table, the drop report and the audit sample."""
    data_dir = root / data.OUT_DIR
    parent = read_manifest(data_dir / "manifest.json")
    inputs = {}
    for name in ("val.jsonl", "train_pool.jsonl"):
        rel = (data.OUT_DIR / name).as_posix()
        digest = sha256_file(root / rel)
        if parent.outputs.get(rel) != digest:
            raise ValueError(
                f"{rel} does not match the data stage's recorded output; rerun ggh data"
            )
        inputs[rel] = digest
    started = dt.datetime.now(dt.UTC)
    val_rows = data.read_jsonl(data_dir / "val.jsonl")
    pool_rows = data.read_jsonl(data_dir / "train_pool.jsonl")
    recorder = RecordingScorer(scorer_factory(), rules.SCORE_DECIMALS)
    sel = select(val_rows, pool_rows, recorder, seed=seed, cap=cap, chunk=chunk, progress=progress)

    out = root / OUT_DIR
    files: dict[str, str] = {}
    for split, rows in (("train", sel.train), ("val", sel.val)):
        files[(OUT_DIR / f"{split}.jsonl").as_posix()] = _dump(
            out / f"{split}.jsonl", (json.loads(e.model_dump_json()) for e in rows)
        )
    files[(OUT_DIR / "nli_scores.jsonl").as_posix()] = _dump(
        out / "nli_scores.jsonl",
        ({"key": k, "entail": v} for k, v in sorted(recorder.table.items())),
    )
    audit = audit_sample(sel.train, seed)
    files[(OUT_DIR / "audit_sample.jsonl").as_posix()] = _dump(
        out / "audit_sample.jsonl", (json.loads(e.model_dump_json()) for e in audit)
    )
    report = {**sel.report, "rules_sha256": rules_hash(), "files": dict(files)}
    files[(OUT_DIR / "drop_report.json").as_posix()] = _json(out / "drop_report.json", report)
    (root / MEASURE_DIR).mkdir(parents=True, exist_ok=True)
    (root / REPORT_COPY).write_bytes((out / "drop_report.json").read_bytes())
    (root / AUDIT_COPY).write_bytes((out / "audit_sample.jsonl").read_bytes())

    params = {
        "seed": seed,
        "cap": cap,
        "rules_sha256": rules_hash(),
        "nli_model": f"{MODEL_ID}@{MODEL_REVISION}",
    }
    write_manifest(
        out,
        StageManifest(
            stage="targets",
            code_version=CODE_VERSION,
            key=stage_key("targets", CODE_VERSION, inputs, params, seed),
            inputs=inputs,
            outputs={
                **files,
                REPORT_COPY.as_posix(): sha256_file(root / REPORT_COPY),
                AUDIT_COPY.as_posix(): sha256_file(root / AUDIT_COPY),
            },
            params=params,
            seed=seed,
            parents=(parent.key,),
            started_at=started,
            finished_at=dt.datetime.now(dt.UTC),
            host=platform.node() or "unknown",
        ),
    )
    return report


# ---- replay on another machine (T2.1, D-10, PRV-R1) ----

SCORES_ARCHIVE = Path("data") / "nli_scores.jsonl.gz"
REPLAYED = ("train.jsonl", "val.jsonl", "nli_scores.jsonl", "audit_sample.jsonl")


def write_scores_archive(root: Path) -> Path:
    """Commit the recorded NLI scores: gzip with a fixed mtime and no file name, so the same
    scores always give the same bytes."""
    import gzip

    src = (root / OUT_DIR / "nli_scores.jsonl").read_bytes()
    dest = root / SCORES_ARCHIVE
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(src)
    return dest


def replay_scorer(root: Path) -> ReplayScorer:
    import gzip

    with gzip.open(root / SCORES_ARCHIVE, "rt", encoding="utf-8") as fh:
        table = {(r := json.loads(line))["key"]: float(r["entail"]) for line in fh if line.strip()}
    return ReplayScorer(table)


def expected_files(root: Path) -> dict[str, str]:
    """The output hashes the committed targets report records, read before any rebuild."""
    files = json.loads((root / REPORT_COPY).read_text(encoding="utf-8"))["files"]
    return {name: files[(OUT_DIR / name).as_posix()] for name in REPLAYED}


def replay_problems(root: Path, expected: dict[str, str]) -> list[str]:
    return [
        f"{name}: sha256 {got[:12]}…, committed {want[:12]}…"
        for name, want in expected.items()
        if (got := sha256_file(root / OUT_DIR / name)) != want
    ]
