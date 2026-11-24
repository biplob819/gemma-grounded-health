"""V3.8 (OWNER D-75, D-76): v3b's release as GGH v1.1.0, the proof tests of RBR-02..04 and
RBR-06 (RBR-01's and RBR-05's are added with their evidence).

v1.0.0's published bundle, behaviour table and card are frozen under docs/results/releases/v1.0.0/
and must stay what the v1 + v2 build writes. Tests that need V3.8's release evidence (the
off-scope and probe runs, the B8 review, the laptop bench) skip, naming the file they wait for,
until it is committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ggh.results import bundle as B
from ggh.results.runs import Run

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs" / "results" / "releases" / "v1.0.0"
PLAN_V2 = ROOT / "docs" / "measurements" / "release_plan_v2.json"
LEDGER = ROOT / "docs" / "measurements" / "spend-ledger.jsonl"
V3B = ROOT / "docs" / "measurements" / "v3b"
ADAPTER = "0537933d988bdbb7db836c8a4651a278d391f95c72ed4f45b6c3b739c39d2014"
Q4, Q8 = "gemma-e2b-grounded-v3b-q4", "gemma-e2b-grounded-v3b-q8"
V2_MODEL, BASE = "gemma-e2b-grounded-v2-q4", "gemma-e2b-base-q4"

# what a v3b bundle needs beyond what is committed now (V3.8 RBR-01, RBR-05)
EVIDENCE = (
    "docs/results/offscope_v3b/summary.json",
    "docs/results/probe_original_v3b/summary.json",
    "docs/results/probe_paraphrase_v3b/summary.json",
    "docs/measurements/probe_report_v3b.json",
    "docs/measurements/v3b/laptop_bench.json",
)
MISSING = [p for p in EVIDENCE if not (ROOT / p).is_file()]
needs_evidence = pytest.mark.skipif(
    bool(MISSING), reason=f"waits for V3.8's release evidence: {', '.join(MISSING)}"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cfg() -> dict:
    return yaml.safe_load((ROOT / B.CONFIG).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_v2() -> B.Built:
    return B.build(ROOT, versions=("v1", "v2"))


@pytest.fixture(scope="module")
def with_v3b() -> B.Built:
    return B.build(ROOT, versions=("v1", "v2", "v3b"))


# RBR-03: the bundle


def test_v1_0_0_published_files_are_frozen_and_reproducible(v1_v2, tmp_path):
    """D-76: the v1 + v2 build still writes v1.0.0's published bundle and behaviour table byte
    for byte, and the frozen card is the one v1.0.0's Hub plan uploaded."""
    paths = B.write(ROOT, v1_v2, tmp_path)
    for p in paths:
        assert p.read_bytes() == (FROZEN / p.name).read_bytes(), p.name
    assert v1_v2.bundle.spend_usd == 118.035054  # Wave 7 left out, as published
    plan = json.loads(PLAN_V2.read_text(encoding="utf-8"))
    hub = {f["path_in_repo"]: f for f in plan["files"] if f["repo"].endswith("-health")}
    assert hub["README.md"]["record"] == "MODEL_CARD.md"
    assert _sha(FROZEN / "MODEL_CARD.md") == hub["README.md"]["sha256"]
    assert _sha(FROZEN / "results_bundle.json") == hub["results_bundle.json"]["sha256"]


def test_bundle_spend_counts_every_paid_run():
    """OWNER D-75: with v3b a release, every ledger row counts (V3.5's v3 scoring too); the
    v1 + v2 build leaves Wave 7 out (D-76)."""
    from ggh import ledger

    cfg = _cfg()
    rows = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    wave7 = [r for r in rows if str(r.get("node", "")).startswith("V3.")]
    assert {"V3.5", "V3.7"} <= {r["node"] for r in wave7}
    assert B._in_scope(cfg, ("v1", "v2", "v3b")) is None
    keep = B._in_scope(cfg, ("v1", "v2"))
    assert keep is not None and not any(keep(r) for r in wave7)
    every = round(sum(float(r["spent_usd"]) for r in rows), 6)
    assert round(ledger.total(LEDGER, "spent_usd"), 6) == every
    assert round(ledger.total(LEDGER, "spent_usd", keep), 6) < every


def test_the_v3b_entry_reads_a_grpo_run_and_v2s_audits():
    """RBR-03: v3b's training record is its GRPO run (RLRun) and its audits are v2's (D-76)."""
    from ggh.schemas import RLRun

    cfg = _cfg()
    rel = {r["version"]: r for r in cfg["releases"]}
    assert [r["version"] for r in cfg["releases"]] == ["v1", "v2", "v3b"]
    v3b, v2 = rel["v3b"], rel["v2"]
    assert v3b["model"] == Q4 and v3b["primary_run"] == "main_v3b"
    assert v3b["audits"] == v2["audits"]
    run = B.train_record(ROOT, v3b["train_run"])
    assert isinstance(run, RLRun) and run.adapter_sha256 == ADAPTER and run.reward == "v3b"
    assert B.train_record(ROOT, v2["train_run"]).__class__.__name__ == "TrainRun"
    # a held release stays out of the default build until its evidence is committed
    if v3b.get("held"):
        assert Run.exists(ROOT, "main_v3b")
        assert [r.version for r in B.build(ROOT).bundle.releases] == ["v1", "v2"]


@needs_evidence
def test_the_bundle_ships_v3b_after_v1_and_v2(with_v3b):
    b = with_v3b.bundle
    assert [r.version for r in b.releases] == ["v1", "v2", "v3b"]
    assert b.shipped.version == "v3b" and b.shipped.model == Q4
    assert b.shipped.adapter_sha256 == ADAPTER
    assert b.shipped.bench_source == "docs/measurements/v3b/laptop_bench.json#ft3b-Q4_K_M"
    assert {"ft3b-Q4_K_M", "ft3b-Q8_0"} <= set(b.bench_variants)
    assert b.publish_blockers == ()


@needs_evidence
def test_v3b_bundle_numbers_come_from_the_runs(with_v3b):
    b = with_v3b.bundle
    result = json.loads((ROOT / "docs/results/main_v3b/result.json").read_text(encoding="utf-8"))
    sol = "gpt-6.1-sol"
    heads = {m.model: m.headline[sol].point for m in b.results if m.run == "main_v3b"}
    assert heads == pytest.approx(result["headline"], abs=5e-4)
    (prim,) = [c for c in b.primary_comparisons if c.run == "main_v3b"]
    p = result["primary"]
    assert (prim.model_a, prim.model_b) == (Q4, V2_MODEL)
    assert (prim.diff.point, prim.diff.lo, prim.diff.hi) == (p["point"], p["lo"], p["hi"])
    assert prim.separated and prim.verdict == "a_better"
    s = b.shipped
    parity = json.loads((V3B / "parity.json").read_text(encoding="utf-8"))
    q8 = json.loads((V3B / "parity_q8_0.json").read_text(encoding="utf-8"))
    assert s.parity is not None and s.parity.label_agreement == parity["label_agreement"]
    assert s.parity_q8_0 is not None
    assert s.parity_q8_0.label_agreement == q8["q8_0_vs_bf16"]["report"]["label_agreement"]
    rl = json.loads((V3B / "rl_run.json").read_text(encoding="utf-8"))
    assert s.train.model_dump() == rl
    run_ids = {r.name: r.run_id for r in b.runs}
    assert run_ids["main_v3b"] == result["run_id"]
