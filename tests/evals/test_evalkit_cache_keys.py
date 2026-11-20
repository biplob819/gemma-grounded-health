"""D-56: the EvalKit 0.4.0 pin keeps every cache key of the committed runs, so V2.2 can reuse
the answers, claim sets and verdicts already paid for.

The fixture holds keys computed with EvalKit ee59699 (the B5.12 pin of `main_citation` and
`dev_ground_v1`) for records drawn across models: the answer (`generate`), the claim set
(`extract`), and per judge the claim verdict and the B5.12 citation verdict. Each was checked
present in the local EvalKit store those runs wrote. Here the installed EvalKit derives them
again from the same committed records; any change in key derivation, a node's code version or
an input schema shows as a different key.
"""

import gzip
import json
from pathlib import Path

import pytest
from evalkit.config import load_run_config
from evalkit.pipeline.extract import ExtractClaims, extract_inputs
from evalkit.pipeline.generate import Generate, Generated, generate_inputs
from evalkit.pipeline.judge import CITATION_PROMPT_HASH, CITATIONS, CLAIMS, JUDGE_PROMPT_HASH
from evalkit.run import load_task
from evalkit.runtime import RunContext
from evalkit.schemas import Case, Record

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "fixtures" / "evalkit_cache_keys_ee59699.json"
RUNS = {  # run: (task, config, record indices: one per model)
    "main_citation": (
        "configs/tasks/pubmedqa_cit.yaml", "configs/evalkit.yaml", [0, 1001, 2002, 3003]
    ),
    "dev_ground_v1": ("configs/tasks/dev_ground.yaml", "configs/evalkit_dev_ground.yaml", [0, 399]),
}  # fmt: skip


def _jsonl_gz(path: Path) -> list[str]:
    return [x for x in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines() if x]


def keys_for(run: str, task_ref: str, cfg_rel: str, picks: list[int]) -> dict[str, dict]:
    rd = ROOT / "docs/results" / run
    cfg = load_run_config(ROOT / cfg_rel)
    task, _ = load_task(str(ROOT / task_ref))
    cases = {c.id: c for c in map(Case.model_validate_json, _jsonl_gz(rd / "cases.jsonl.gz"))}
    records = [Record.model_validate_json(x) for x in _jsonl_gz(rd / "records.jsonl.gz")]
    ctx = RunContext(run_id="any", seed=cfg.seed)  # the run id is in no key
    out = {}
    for i in picks:
        r = records[i]
        case = cases[r.case_id]
        (gin,) = [g for g in generate_inputs([case], task, cfg) if g.model == r.completion.model]
        gen = Generated(case=case, model=r.completion.model, completion=r.completion)
        ((_, ein),) = extract_inputs([gen], cfg)
        claim = r.claims[0]
        out[f"{r.case_id}/{r.completion.model}"] = {
            "generate": Generate({}, {}).cache_key(gin, ctx),
            "extract": ExtractClaims({}, {}).cache_key(ein, ctx),
            **{
                f"judge:{j}": CLAIMS.cache_key(CLAIMS.make_input(claim, case, j, cfg.judge_effort))
                for j in cfg.judges
            },
            **{
                f"citation:{j}": CITATIONS.cache_key(
                    CITATIONS.make_input(claim, case, j, cfg.judge_effort)
                )
                for j in cfg.judges
            },
        }
    return out


@pytest.mark.contract
@pytest.mark.parametrize("run", sorted(RUNS))
def test_cache_keys_are_the_ones_ee59699_wrote(run):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[run]
    assert keys_for(run, *RUNS[run]) == golden
    assert len(golden) == len(RUNS[run][2]) and all(len(k) == 6 for k in golden.values())


@pytest.mark.contract
def test_judge_prompt_hashes_are_the_ones_the_committed_verdicts_carry():
    assert JUDGE_PROMPT_HASH == "9a3a8246049f6f85"
    assert CITATION_PROMPT_HASH == "be34ebe577bc96ae"
    for run in ("main", "main_citation", "dev_ground_v1"):
        recs = [json.loads(x) for x in _jsonl_gz(ROOT / "docs/results" / run / "records.jsonl.gz")]
        assert {v["judge_prompt_hash"] for r in recs for v in r["verdicts"]} == {JUDGE_PROMPT_HASH}
