"""`ggh data`: build the train pool, dev splits and off-scope set from pinned PubMedQA (N1.2).

Rules (Tech Spec §4.1): PQA-L is read only to build the leakage index (DAT-R2); PQA-A rows
sharing a PQA-L PMID, normalised question or near-duplicate context are dropped (DAT-R3);
the split is seeded and by PMID (DAT-R4); prompts are rendered by EvalKit's own template and
splitter (DAT-R5, D-2). The committed manifest holds no timestamps, so a rerun reproduces it
byte for byte (DAT-07).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import random
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evalkit
import pyarrow.parquet as pq
from evalkit.schemas import Case
from evalkit.tasks import builtin_task, render_prompt
from evalkit.text import number_sentences, split_sentences

from ggh.data import fetch
from ggh.lineage import StageManifest, sha256_file, stage_key, write_manifest
from ggh.schemas import DataManifest

CODE_VERSION = "1"
EVALKIT_COMMIT = "06591f8de9b178078c971940a83019a6530fd119"  # D-28
SEED = 7
N_VAL, N_DEV_EVAL = 2_000, 500
SHINGLE = 8
JACCARD_DROP = 0.5
OUT_DIR = Path("artifacts") / "data"
MEASURE_DIR = Path("docs") / "measurements"
MANIFEST_PATH = MEASURE_DIR / "data_manifest.json"
OFFSCOPE_PATH = Path("data") / "offscope_eval.jsonl"
N_OFFSCOPE = 20


@dataclass(frozen=True)
class Row:
    pmid: str
    question: str
    sentences: tuple[str, ...]
    sections: tuple[str, ...]  # section label for each sentence (BACKGROUND, RESULTS, …)
    long_answer: str
    label: str

    def as_json(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "question": self.question,
            "sentences": list(self.sentences),
            "sections": list(self.sections),
            "long_answer": self.long_answer,
            "label": self.label,
        }


def sentences_with_sections(
    contexts: list[str], labels: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """EvalKit's numbering, with each sentence tagged by the section label of its paragraph."""
    if len(labels) != len(contexts):
        labels = ["UNLABELLED"] * len(contexts)
    sents: list[str] = []
    secs: list[str] = []
    for para, label in zip(contexts, labels, strict=True):
        parts = split_sentences(para)
        sents.extend(parts)
        secs.extend([label] * len(parts))
    numbered = number_sentences(contexts)
    if tuple(sents) != numbered:  # EvalKit's flattening must agree with ours, sentence for sentence
        raise ValueError("sentence split disagrees with evalkit.text.number_sentences")
    return numbered, tuple(secs)


def read_parquet(path: Path, *, with_answers: bool) -> list[dict[str, Any]]:
    cols = ["pubid", "question", "context", "final_decision"] + (
        ["long_answer"] if with_answers else []
    )
    return pq.read_table(path, columns=cols).to_pylist()


def to_row(raw: dict[str, Any]) -> Row:
    ctx = raw["context"]
    sents, secs = sentences_with_sections(list(ctx["contexts"]), list(ctx.get("labels") or []))
    return Row(
        pmid=str(raw["pubid"]),
        question=raw["question"],
        sentences=sents,
        sections=secs,
        long_answer=raw.get("long_answer") or "",
        label=raw["final_decision"],
    )


_WORD = re.compile(r"\w+")


def norm_question(q: str) -> str:
    return " ".join(_WORD.findall(q.lower()))


def shingles(text: str, n: int = SHINGLE) -> frozenset[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


class LeakageIndex:
    """PQA-L PMIDs, normalised questions and context shingles, for DAT-R3."""

    def __init__(self, pqa_l: Iterable[dict[str, Any]]) -> None:
        self.pmids: set[str] = set()
        self.questions: set[str] = set()
        self.shingle_sets: list[frozenset[tuple[str, ...]]] = []
        self.by_shingle: dict[tuple[str, ...], list[int]] = {}
        for raw in pqa_l:
            self.pmids.add(str(raw["pubid"]))
            self.questions.add(norm_question(raw["question"]))
            s = shingles(" ".join(raw["context"]["contexts"]))
            idx = len(self.shingle_sets)
            self.shingle_sets.append(s)
            for sh in s:
                self.by_shingle.setdefault(sh, []).append(idx)

    max_jaccard: float = 0.0  # highest context similarity seen, as evidence the check ran

    def reason(self, row: Row) -> str | None:
        """Why row leaks into PQA-L, or None."""
        if row.pmid in self.pmids:
            return "pmid"
        if norm_question(row.question) in self.questions:
            return "question"
        s = shingles(" ".join(row.sentences))
        if not s:
            return None
        shared = Counter(i for sh in s for i in self.by_shingle.get(sh, ()))
        best = 0.0
        for i, inter in shared.items():
            union = len(s) + len(self.shingle_sets[i]) - inter
            best = max(best, inter / union if union else 0.0)
        self.max_jaccard = max(self.max_jaccard, best)
        return "context" if best >= JACCARD_DROP else None


def split(
    rows: list[Row], seed: int = SEED, n_val: int = N_VAL, n_dev: int = N_DEV_EVAL
) -> tuple[list[Row], list[Row], list[Row]]:
    """(val, dev_eval, train_pool): seeded, by PMID, disjoint (DAT-R4)."""
    by_pmid = sorted(rows, key=lambda r: int(r.pmid))
    pmids = [r.pmid for r in by_pmid]
    if len(set(pmids)) != len(pmids):
        raise ValueError("duplicate PMIDs in PQA-A")
    order = list(range(len(by_pmid)))
    random.Random(seed).shuffle(order)
    picked = [by_pmid[i] for i in order]
    if len(picked) <= n_val + n_dev:
        raise ValueError(f"{len(picked)} rows cannot fill val {n_val} and dev_eval {n_dev}")
    val = sorted(picked[:n_val], key=lambda r: int(r.pmid))
    dev = sorted(picked[n_val : n_val + n_dev], key=lambda r: int(r.pmid))
    pool = sorted(picked[n_val + n_dev :], key=lambda r: int(r.pmid))
    return val, dev, pool


def render(row: Row) -> str:
    """The prompt EvalKit would send for this row (D-2, DAT-R5)."""
    case = Case(
        id=f"pqa-a-{row.pmid}",
        question=row.question,
        context=row.sentences,
        reference_label=row.label,
        source="pubmedqa/pqa_a",
    )
    return render_prompt(builtin_task("pubmedqa"), case)


def template_hash() -> str:
    return hashlib.sha256(builtin_task("pubmedqa").prompt_template.encode("utf-8")).hexdigest()


_TOKEN = re.compile(r"\w+|[^\w\s]")


def quantiles(values: list[int]) -> dict[str, float]:
    if len(values) < 2:
        v = float(values[0]) if values else 0.0
        return {"p50": v, "p90": v, "p99": v, "max": v}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": q[49], "p90": q[89], "p99": q[98], "max": float(max(values))}


def length_report(all_rows: list[Row]) -> dict[str, Any]:
    words = [len(r.long_answer.split()) for r in all_rows]
    tokens = [len(_TOKEN.findall(render(r))) for r in all_rows]
    return {
        "pqa_a_labels": dict(sorted(Counter(r.label for r in all_rows).items())),
        "long_answer_words": quantiles(words),
        "long_answer_over_80_words": sum(w > 80 for w in words),
        "rendered_tokens_proxy": quantiles(tokens),
        "token_proxy": "regex \\w+|[^\\w\\s] count; re-measured with Gemma's tokenizer at T2.1",
    }


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            # ASCII-escaped, so U+2028/U+2029 inside a text field never look like line breaks
            fh.write(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n")
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON object per "\\n"-terminated line (never str.splitlines, which also breaks on
    U+2028, U+2029 and other characters that can sit inside a JSON string)."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln.strip()]


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def load_offscope(path: Path, dev_pmids: set[str], n: int = N_OFFSCOPE) -> list[dict[str, Any]]:
    """The frozen off-scope evaluation set (DAT-06): 20 dev_eval abstracts with advice questions."""
    rows = read_jsonl(path)
    if len(rows) != n:
        raise ValueError(f"{path} has {len(rows)} rows, expected {n}")
    if len({r.get("id") for r in rows}) != n:
        raise ValueError(f"{path} has duplicate ids")
    for r in rows:
        if r.get("pmid") not in dev_pmids:
            raise ValueError(
                f"off-scope row {r.get('id')} uses pmid {r.get('pmid')}, not a dev_eval abstract"
            )
        if not str(r.get("question", "")).strip():
            raise ValueError(f"off-scope row {r.get('id')} has no question")
    return rows


def build(
    root: Path,
    *,
    seed: int = SEED,
    download: bool = True,
    sources: tuple[Path, Path] | None = None,
    sizes: tuple[int, int, int, int] | None = None,
    refreeze_offscope: bool = False,
) -> DataManifest:
    """Build everything under root. `sources` (PQA-A, PQA-L parquet) and `sizes` (PQA-A rows,
    val, dev_eval, off-scope rows) exist for tests on small synthetic files; the real build
    uses the pinned, verified downloads and the spec's sizes."""
    # the off-scope set is frozen (DAT-06, OFF-R1): check its hash before doing any work
    offscope_hash = sha256_file(root / OFFSCOPE_PATH)
    if not refreeze_offscope:
        committed = root / MANIFEST_PATH
        if not committed.exists():
            raise ValueError(
                f"{MANIFEST_PATH} is missing, so the frozen {OFFSCOPE_PATH} cannot be checked; "
                "restore it, or pass --refreeze-offscope for a first build after the owner "
                "approves the set"
            )
        frozen = DataManifest.model_validate_json(
            committed.read_text(encoding="utf-8")
        ).offscope_eval_hash
        if frozen != offscope_hash:
            raise ValueError(
                f"{OFFSCOPE_PATH} changed since it was frozen "
                f"({frozen[:12]}… → {offscope_hash[:12]}…); it is frozen before training "
                "(OFF-R1). Revert it, or pass --refreeze-offscope after the owner approves."
            )
    if sources is None:
        a_path = fetch.ensure(fetch.PQA_A, download=download)
        l_path = fetch.ensure(fetch.PQA_L, download=download)
    else:
        a_path, l_path = sources
    n_rows, n_val, n_dev, n_off = sizes or (fetch.PQA_A.rows, N_VAL, N_DEV_EVAL, N_OFFSCOPE)
    started = dt.datetime.now(dt.UTC)

    index = LeakageIndex(read_parquet(l_path, with_answers=False))
    all_rows = [to_row(raw) for raw in read_parquet(a_path, with_answers=True)]
    if len(all_rows) != n_rows:
        raise fetch.DataError(f"PQA-A has {len(all_rows)} rows, expected {n_rows}")

    kept: list[Row] = []
    overlap: Counter[str] = Counter({"pmid": 0, "question": 0, "context": 0})
    for r in all_rows:
        why = index.reason(r)
        if why:
            overlap[why] += 1
        else:
            kept.append(r)
    val, dev, pool = split(kept, seed, n_val, n_dev)
    # every off-scope abstract must be a dev_eval abstract under this seed; checked before
    # anything is written, so a seed that breaks the pairing leaves no partial output
    offscope = load_offscope(root / OFFSCOPE_PATH, {r.pmid for r in dev}, n_off)

    out = root / OUT_DIR
    files: dict[str, str] = {}
    files[(OUT_DIR / "train_pool.jsonl").as_posix()] = _dump(
        out / "train_pool.jsonl", (r.as_json() for r in pool)
    )
    for name, rows in (("val", val), ("dev_eval", dev)):
        files[(OUT_DIR / f"{name}.jsonl").as_posix()] = _dump(
            out / f"{name}.jsonl", ({**r.as_json(), "prompt": render(r)} for r in rows)
        )
    files[(OUT_DIR / "overlap_report.json").as_posix()] = _write_json(
        out / "overlap_report.json",
        {
            "pqa_l_rows": len(index.pmids),
            "pqa_a_rows_checked": len(all_rows),
            "dropped_from_pqa_a": dict(sorted(overlap.items())),
            "max_context_jaccard": round(index.max_jaccard, 6),
            "rule": (
                f"pmid, normalised question, or context {SHINGLE}-gram Jaccard >= {JACCARD_DROP}"
            ),
        },
    )
    files[(OUT_DIR / "length_report.json").as_posix()] = _write_json(
        out / "length_report.json", length_report(all_rows)
    )
    for name in ("overlap_report.json", "length_report.json"):  # committed evidence copies
        (root / MEASURE_DIR).mkdir(parents=True, exist_ok=True)
        (root / MEASURE_DIR / f"data_{name}").write_bytes((out / name).read_bytes())
        copy = (MEASURE_DIR / f"data_{name}").as_posix()
        files[copy] = sha256_file(root / copy)

    def label_counts(rows: list[Row]) -> dict[str, int]:
        return dict(sorted(Counter(r.label for r in rows).items()))

    manifest = DataManifest(
        dataset=fetch.DATASET,
        dataset_revision=fetch.REVISION,
        evalkit_version=f"{evalkit.__version__}+{EVALKIT_COMMIT[:7]}",
        template_hash=template_hash(),
        splitter_version=EVALKIT_COMMIT[:7],
        seed=seed,
        counts={
            "train_pool": label_counts(pool),
            "val": label_counts(val),
            "dev_eval": label_counts(dev),
            "offscope_eval": {"maybe": len(offscope)},
        },
        drops={f"overlap_with_pqa_l:{k}": v for k, v in sorted(overlap.items())},
        files=files,
        offscope_eval_hash=offscope_hash,
    )
    _write_json(root / MANIFEST_PATH, json.loads(manifest.model_dump_json()))

    inputs = {
        fetch.uri(fetch.PQA_A): fetch.PQA_A.sha256,
        fetch.uri(fetch.PQA_L): fetch.PQA_L.sha256,
        OFFSCOPE_PATH.as_posix(): offscope_hash,
    }
    params = {
        "seed": seed,
        "n_val": n_val,
        "n_dev_eval": n_dev,
        "shingle": SHINGLE,
        "jaccard_drop": JACCARD_DROP,
        "evalkit": EVALKIT_COMMIT,
    }
    write_manifest(
        out,
        StageManifest(
            stage="data",
            code_version=CODE_VERSION,
            key=stage_key("data", CODE_VERSION, inputs, params, seed),
            inputs=inputs,
            outputs={**files, MANIFEST_PATH.as_posix(): sha256_file(root / MANIFEST_PATH)},
            params=params,
            seed=seed,
            parents=(),
            started_at=started,
            finished_at=dt.datetime.now(dt.UTC),
            host=platform.node() or "unknown",
        ),
    )
    return manifest
