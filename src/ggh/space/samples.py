"""SPC-03: the six precomputed sample questions the demo shows without a passcode (Tech Spec §9,
OWNER D-36).

`build()` picks the samples from a committed EvalKit run over `dev_eval` (the dev grounding run,
both models), two per reference label where it can, seeded, and keeps each model's recorded raw
reply, so the cards are rendered by the same `to_card` as live answers.

v2 (OWNER D-63): v2's dev run scored only v2, over the same 200 `dev_eval` cases as v1's, so
the base replies come from v1's dev run (`--base-run`) and the fine-tune's from v2's (`--run`);
`build` refuses two runs whose cases differ. The defaults below are v2's.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from evalkit.schemas import Case

from ggh.schemas._base import Model, NonBlank

SAMPLES = Path("space") / "samples.json"
RUN = Path("docs") / "results" / "dev_ground_v2_sol"  # v2's dev run (the fine-tune's replies)
BASE_RUN = Path("docs") / "results" / "dev_ground_v1"  # base's replies: same cases as RUN
N_SAMPLES = 6
SEED = 7
BASE_MODEL = "gemma-e2b-base-q4"
FT_MODEL = "gemma-e2b-grounded-v2-q4"


class Sample(Model):
    id: NonBlank
    question: NonBlank
    context: tuple[str, ...]
    reference_label: str | None = None
    base_raw: str
    ft_raw: str
    # one run: "evalkit:<run_id>:<base model>|<ft model>"; two runs:
    # "evalkit:<base run_id>:<base model>|evalkit:<ft run_id>:<ft model>"
    source: NonBlank


def _lines(run_dir: Path, name: str) -> list[str]:
    """`name`.jsonl, or the committed `name`.jsonl.gz when the plain file is not there."""
    plain = run_dir / f"{name}.jsonl"
    if plain.is_file():
        text = plain.read_text(encoding="utf-8")
    else:
        text = gzip.decompress((run_dir / f"{name}.jsonl.gz").read_bytes()).decode("utf-8")
    return [line for line in text.splitlines() if line.strip()]


def read_cases(run_dir: Path) -> dict[str, Case]:
    cases = [Case.model_validate_json(x) for x in _lines(run_dir, "cases")]
    return {c.id: c for c in cases}


def replies(run_dir: Path) -> tuple[str, dict[str, dict[str, str]]]:
    """(run id, case id -> model -> raw reply). Read as plain JSON: only these three fields
    are needed, and a newer EvalKit may add fields its pinned `Record` does not know."""
    run_id, out = "", defaultdict[str, dict[str, str]](dict)
    for line in _lines(run_dir, "records"):
        r = json.loads(line)
        run_id = r["run_id"]
        out[r["case_id"]][r["completion"]["model"]] = r["completion"]["raw_text"]
    return run_id, dict(out)


def build(
    run_dir: Path,
    *,
    base_run: Path | None = None,
    base_model: str = BASE_MODEL,
    ft_model: str = FT_MODEL,
    n: int = N_SAMPLES,
    seed: int = SEED,
) -> list[Sample]:
    """`run_dir` holds the fine-tune's replies (and base's, unless `base_run` is given)."""
    cases = read_cases(run_dir)
    run_id, raw = replies(run_dir)
    src = f"evalkit:{run_id}:{base_model}|{ft_model}"
    if base_run is not None and base_run.resolve() != run_dir.resolve():
        base_cases = read_cases(base_run)
        if {k: (c.question, c.context) for k, c in base_cases.items()} != {
            k: (c.question, c.context) for k, c in cases.items()
        }:
            raise ValueError(f"{base_run} and {run_dir} did not answer the same cases")
        base_id, base_raw = replies(base_run)
        raw = {  # base from the base run only, the fine-tune from this run only
            cid: {base_model: base_raw[cid][base_model], ft_model: models[ft_model]}
            for cid, models in raw.items()
            if ft_model in models and base_model in base_raw.get(cid, {})
        }
        src = f"evalkit:{base_id}:{base_model}|evalkit:{run_id}:{ft_model}"
    both = sorted(cid for cid, m in raw.items() if base_model in m and ft_model in m)
    if len(both) < n:
        raise ValueError(f"{run_dir} has only {len(both)} cases answered by both models")
    by_label: dict[str | None, list[str]] = defaultdict(list)
    for cid in both:
        by_label[cases[cid].reference_label].append(cid)
    rng = random.Random(seed)
    for ids in by_label.values():
        rng.shuffle(ids)
    picked: list[str] = []
    while len(picked) < n:  # round-robin over labels, so a minority label is represented
        for label in sorted(by_label, key=str):
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop())
    return [
        Sample(
            id=cid,
            question=cases[cid].question,
            context=cases[cid].context,
            reference_label=cases[cid].reference_label,
            base_raw=raw[cid][base_model],
            ft_raw=raw[cid][ft_model],
            source=src,
        )
        for cid in sorted(picked)
    ]


def dump(samples: list[Sample]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in samples], indent=2) + "\n"


def load(path: Path) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Sample.model_validate(d) for d in data]
