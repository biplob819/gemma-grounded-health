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


# RBR-02: export and lineage records

LINEAGE = ROOT / "docs" / "measurements" / "lineage"
RECORDS_V3B = ROOT / "docs" / "measurements" / "export_artifacts_v3b.json"


def _copy(name: str):
    from ggh.lineage import read_manifest

    return read_manifest(LINEAGE / f"{name}.json")


def test_v3b_records_name_the_frozen_adapter_everywhere():
    """D-20: every record of v3b names adapter 0537933d…, and its run started from v2's."""
    train, export = _copy("train_v3b"), _copy("export_ft3b")
    v2_sel = json.loads((ROOT / "docs/measurements/v2/train_selection.json").read_text("utf-8"))

    def load(name: str) -> dict:
        return json.loads((V3B / name).read_text(encoding="utf-8"))

    rl, sel, merge, gguf = (
        load(n)
        for n in (
            "rl_run.json",
            "train_selection.json",
            "merge_record.json",
            "gguf_ft3b_manifest.json",
        )
    )
    named = {
        "train_v3b": train.params["adapter_sha256"],
        "export_ft3b": export.params["adapter_sha256"],
        "rl_run": rl["adapter_sha256"],
        "train_selection": sel["adapter_sha256"],
        "merge_record": merge["adapter_sha256"],
        "gguf manifest": gguf["source"]["revision"],
    }
    assert named == dict.fromkeys(named, ADAPTER)
    assert train.params["init_adapter_sha256"] == rl["init_adapter_sha256"]
    assert rl["init_adapter_sha256"] == v2_sel["adapter_sha256"]
    assert train.params["kaggle_kernel"] == rl["kaggle_kernel_version"] == "ggh-v3b-grpo v1"
    assert train.params["selected_step"] == sel["selected_step"] == rl["selected_step"]
    # the adapter files the export merged are the ones the GRPO run produced
    adapter_files = {k: v for k, v in train.outputs.items() if "adapter_v3b/" in k}
    assert len(adapter_files) == 2 and adapter_files.items() <= export.inputs.items()
    adapter = ROOT / "artifacts" / "adapter_v3b"
    if adapter.is_dir():  # not in CI: artifacts/ is not committed
        from ggh.train.select import adapter_hash

        assert adapter_hash(adapter) == ADAPTER


def test_v3b_lineage_chain_reaches_v2_and_the_data():
    """parity → export → GRPO training → v2's training → v2's targets → data, every committed
    output still matching (`ggh lineage verify artifacts/parity_v3b/manifest.json` checks the
    artifacts/ outputs too)."""
    from ggh.release.hub import lineage_chain

    copies = [_copy(p.stem) for p in sorted(LINEAGE.glob("*.json"))]
    chain = lineage_chain(_copy("parity_v3b").key, copies)
    keys = [m.key for m in chain]
    want = ["parity_v3b", "export_ft3b", "train_v3b", "train_v2", "targets_v2", "data"]
    assert keys == [_copy(n).key for n in want]
    train, train_v2, targets = _copy("train_v3b"), _copy("train_v2"), _copy("targets_v2")
    assert train.parents == (train_v2.key, targets.key)
    for rel, digest in train.inputs.items():  # what a parent produced, it produced so
        for parent in (train_v2, targets):
            if rel in parent.outputs:
                assert parent.outputs[rel] == digest, rel
    plan = yaml.safe_load((ROOT / "configs/eval_plan_v3b.yaml").read_text(encoding="utf-8"))
    for k in ("reward", "checker", "recipe"):
        pin = plan["pinned"][k]
        assert train.inputs[pin["file"]] == pin["sha256"], k
    problems = B.committed_lineage_problems(
        ROOT,
        ["docs/measurements/lineage/train_v3b.json", "docs/measurements/lineage/parity_v3b.json"],
    )
    assert problems == []
    par = _copy("parity_v3b")
    assert {"docs/measurements/v3b/parity.json", "docs/measurements/v3b/parity_q8_0.json"} <= set(
        par.outputs
    )
    assert "artifacts/gguf/ft3b-Q8_0.gguf" in par.inputs


def test_the_v3b_export_records_match_the_manifests():
    from ggh.schemas import ExportArtifact

    recs = [ExportArtifact.model_validate(r) for r in json.loads(RECORDS_V3B.read_text("utf-8"))]
    by_key = {m.key: m for m in (_copy(n) for n in ("train_v3b", "export_ft3b", "export_base"))}
    gguf = json.loads((V3B / "gguf_ft3b_manifest.json").read_text(encoding="utf-8"))
    for r in recs:
        m = by_key[r.parent_key]
        rel = f"{r.path}/adapter_model.safetensors" if r.kind == "adapter" else r.path
        assert m.outputs[rel] == r.sha256, r.path
        if r.kind == "gguf" and r.model == "ft":
            assert gguf["files"][Path(r.path).name]["sha256"] == r.sha256
    ft = {(r.kind, r.quant) for r in recs if r.model == "ft"}
    assert ft == {
        ("adapter", None),
        ("merged_bf16", None),
        ("gguf", "F16"),
        ("gguf", "Q4_K_M"),
        ("gguf", "Q8_0"),
    }
    v1 = {
        r["path"]: r
        for r in json.loads((ROOT / "docs/measurements/export_artifacts.json").read_text("utf-8"))
    }
    for r in recs:
        if r.model == "base":  # v1's base GGUFs, unchanged
            assert r.model_dump(mode="json") == v1[r.path]
    paths = {r.path for r in recs if r.model == "ft"}
    assert {"artifacts/gguf/ft3b-Q4_K_M.gguf", "artifacts/gguf/ft3b-Q8_0.gguf"} <= paths


def test_ft3b_bench_probe_and_records_are_wired():
    """`ggh bench --model ft3b`, `ggh probe report --release v3b` and `ggh export-records
    --model ft3b` reach v3b's files (RBR-05's and RBR-01's runs are the main session's)."""
    from ggh.evals import probe
    from ggh.serve import bench

    variants, out = bench.RUNS["ft3b"]
    assert variants == (("ft3b", "Q4_K_M"), ("ft3b", "Q8_0"))
    assert out == Path("docs/measurements/v3b/laptop_bench.json")
    assert bench.RUNS["ft"] == (bench.VARIANTS, bench.OUT)
    assert bench.RUNS["ft2"] == (bench.V2_VARIANTS, bench.V2_OUT)
    orig, para, models, report = probe.RELEASES["v3b"]
    assert (orig, para, models) == ("probe_original_v3b", "probe_paraphrase_v3b", (Q4,))
    assert report == Path("docs/measurements/probe_report_v3b.json")
    cfg = {r["version"]: r for r in _cfg()["releases"]}["v3b"]
    assert cfg["probe"] == report.as_posix() and cfg["bench"] == out.as_posix()
    assert cfg["artifacts"] == RECORDS_V3B.relative_to(ROOT).as_posix()
    from ggh.cli import build_parser

    p = build_parser()
    assert p.parse_args(["bench", "--model", "ft3b"]).model == "ft3b"
    assert p.parse_args(["probe", "report", "--release", "v3b"]).release == "v3b"
    assert p.parse_args(["export-records", "--model", "ft3b"]).model == "ft3b"


# RBR-04: the card, README and notes. Until V3.8's evidence is committed they run on a stand-in
# v3b bundle (v2's off-scope run and laptop bench in place of v3b's); every v3b number in it is
# real (main_v3b, the parity records, the GRPO run)


def _bundle_path(tmp: Path) -> Path:
    cfg = _cfg()
    v3b = cfg["releases"][-1]
    v3b.pop("held", None)
    if MISSING:  # the stand-in: only for the files that do not exist yet
        v3b["runs"] = ["main_v3b", "offscope_v2"]
        v3b["offscope_run"] = "offscope_v2"
        v3b["bench"], v3b["bench_variant"] = "docs/measurements/v2/laptop_bench.json", "ft2-Q4_K_M"
    (tmp / "results.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    built = B.build(ROOT, config=tmp / "results.yaml")
    out = tmp / "results_bundle.json"
    out.write_text(built.bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def v3b_bundle(tmp_path_factory) -> Path:
    return _bundle_path(tmp_path_factory.mktemp("v3b"))


@pytest.fixture(scope="module")
def src3(v3b_bundle):
    from ggh.release import sources

    return sources.load(ROOT, sources.Paths(bundle=v3b_bundle))


@pytest.fixture(scope="module")
def card3(src3) -> str:
    from ggh.release import card

    return card.render(src3)


def _section(text: str, head: str, end: str) -> str:
    return text[text.index(head) : text.index(end, text.index(head))]


def test_the_card_leads_with_q4_and_recommends_q8(src3, card3):
    """OWNER D-75: the registered Q4_K_M result first (0.503 against v2's 0.329 and base's
    0.404, v3b minus v2 +0.174 [0.147, 0.202]), then Q8_0 recommended where it fits."""
    summary = _section(card3, "## Summary", "## Intended use")
    lead = summary.index("Its registered result, at Q4_K_M")
    assert "0.503 [" in summary[lead:] and "0.329 [" in summary[lead:]
    assert "0.404 [" in summary[lead:] and "0.174 [0.147, 0.202], separated" in summary
    rec = summary.index("Q4_K_M is the registered, benchmarked file")
    assert lead < rec and "Q8_0" in summary[rec:] and "is recommended where it fits" in summary
    assert "0.990" in summary[rec:] and "0.541 [" in summary[rec:]
    assert "exploratory and outside the pre-registered test (OWNER D-73)" in summary
    assert "(v2 is separated lower)" not in summary  # the accuracy verdict is read, not typed
    assert "(not separated)" in summary  # v3b and v2 tie on label accuracy
    assert "first supervised (as v2) and then by GRPO" in summary
    how = _section(card3, "## How to use", "## Training data")
    q4, q8 = how.index("GGUF:Q4_K_M`"), how.index("GGUF:Q8_0`")
    assert q4 < q8 and "the Q8_0 file is recommended" in how
    proc = _section(card3, "## Training procedure", "## Evaluation")
    assert "Two stages" in proc and "GRPO" in proc and ADAPTER in proc
    assert "`configs/recipe_v3b_grpo.yaml`" in proc and "checkpoint-250" in proc
    data = _section(card3, "## Training data", "## Training procedure")
    assert "v3b's audits are v2's (D-76)" in data and "trained on no new data" in data
    meta = card3.split("\n---\n", 1)[0]
    assert "grpo" in meta and "gemma-e2b-grounded-v3b-q4" not in meta.split("model-index")[0]


def test_every_v3b_card_number_comes_from_the_bundle(src3, card3):
    from tests.release.test_card import DECIMAL, _numbers

    allowed: set[str] = set()
    _numbers(src3.raw, allowed)
    unsigned = {a.lstrip("-") for a in allowed}
    found = DECIMAL.findall(card3)
    assert len(found) > 100
    assert sorted({x for x in found if x.lstrip("-") not in unsigned}) == []
    b = src3.bundle
    assert b.shipped.version == "v3b" and src3.raw["bundle"]["releases"][-1]["version"] == "v3b"
    # the headline figures are the bundle's, not typed: change one and the card changes
    from ggh.release import card

    q8 = b.shipped.parity_q8_0
    assert q8 is not None and f"{q8.label_agreement:.3f}" in card3
    moved = b.model_copy(
        update={
            "releases": (
                *b.releases[:-1],
                b.shipped.model_copy(update={"parity_q8_0": q8.model_copy(update={"n": 499})}),
            )
        }
    )
    import dataclasses

    assert "same 499 prompts" in card.render(dataclasses.replace(src3, bundle=moved))


def test_the_card_discloses_the_parity_miss_and_caveats(src3, card3):
    parity = _section(card3, "### Parity", "### Laptop speed")
    assert "0.924" in parity and "under the export check's bar of 0.950" in parity
    assert "(OWNER D-73)" in parity and "0.990" in parity and "0.908" in parity
    limits = _section(card3, "## Limits", "## Licence and credits")
    for want in (
        "0.191 of its answers are copies",  # V3.7: v3b restates cited result sentences
        "0.107 for v2",
        "0.461 against 0.283 for v2 and 0.306 for base",
        "misses verbatim cut-down extractions",
        "went unjudged",
        "Q4_K_M's label agreement with the merged weights is 0.924",
        "exploratory model outside the pre-registered test",
        "built after v1's and v2's test results were seen (D-43)",
        BANNER_TEXT,
    ):
        assert want in limits or want in card3, want
    bench = _section(card3, "### Benchmark", "### Published v1 numbers")
    assert f"`{Q8}` is v3b at Q8_0" in bench


BANNER_TEXT = "Research only, not medical advice"
LIVE_BUNDLE = ROOT / "docs" / "results" / "results_bundle.json"
LIVE_IS_V3B = (
    json.loads(LIVE_BUNDLE.read_text(encoding="utf-8"))["releases"][-1]["version"] == "v3b"
)
needs_regeneration = pytest.mark.skipif(
    not LIVE_IS_V3B,
    reason="waits for the live bundle, MODEL_CARD.md, README and notes to be regenerated for v3b",
)


@needs_evidence
@needs_regeneration
def test_the_v3b_card_is_what_the_generator_writes():
    from ggh.release import card, notes, readme, sources

    src = sources.load(ROOT)
    assert src.bundle.shipped.version == "v3b"
    assert (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8") == card.render(src)
    assert readme.results_block(src) in (ROOT / "README.md").read_text(encoding="utf-8")
    assert notes.out_path(src) == Path("docs/release_notes_v1.1.0.md")
    assert (ROOT / notes.out_path(src)).read_text(encoding="utf-8") == notes.render(src, ROOT)
    assert B.digest(B.build(ROOT)) == B.digest(
        B.Built(src.bundle, (ROOT / "docs/results/behaviour_table.md").read_text("utf-8"))
    )


def test_v1_0_0_card_and_notes_regenerate_from_the_frozen_bundle():
    """D-76: v1.0.0's card and release notes are still what the generators write from its
    frozen bundle, whatever the live files ship."""
    from ggh.release import card, notes, sources

    src = sources.load(ROOT, sources.Paths(bundle=FROZEN.relative_to(ROOT) / "results_bundle.json"))
    assert src.bundle.shipped.version == "v2"
    assert card.render(src) == (FROZEN / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert notes.out_path(src) == Path("docs/release_notes_v1.0.0.md")
    assert notes.render(src, ROOT) == (ROOT / "docs/release_notes_v1.0.0.md").read_text("utf-8")
    from ggh.release import links

    assert str(links.links_at("1.0.0")["release"][1]).endswith("/releases/tag/v1.0.0")
    assert links.links_at("1.1.0")["release_notes"][1] == "docs/release_notes_v1.1.0.md"


def test_the_v3b_notes_and_readme_follow_the_bundle(src3, tmp_path):
    from ggh.release import notes, readme

    block = readme.results_block(src3)
    assert "the recommended download where it fits" in block and "0.990" in block
    assert "(a parity miss, OWNER D-73)" in block
    spec = notes.NOTES["v3b"]
    assert spec.version == "1.1.0" and "V3.8" in spec.nodes
    plan = {
        "release": "v3b",
        "model": Q4,
        "tag": "v3b.0.0",
        "files": [{"repo": "o/r-GGUF", "path_in_repo": "x-Q8_0.gguf", "size": 5, "sha256": "s"}],
        "total_bytes": 5,
        "lineage_verified_from": "artifacts/parity_v3b/manifest.json",
        "lineage_chain": ["parity:k"],
        "problems": [],
    }
    text = notes.artifacts(plan, None)  # before the push: no download-back record yet
    assert "push pending" in text and "tag `v3b.0.0`" in text and "published" not in text
    root = tmp_path
    for rel in ("docs/known-issues.md",):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    (root / spec.plan).parent.mkdir(parents=True, exist_ok=True)
    (root / spec.plan).write_text(json.dumps(plan), encoding="utf-8")
    out = notes.render(src3, root)
    assert out.startswith("# Gemma Grounded Health v1.1.0\n")
    assert "The second tagged release (OWNER D-75)" in out and "`v3b.0.0`" in out
    assert "releases/tag/v1.1.0" in out and "release_notes_v1.0.0" not in out
    assert "0.191 of its answers are copies" in out  # the card's caveats are the notes' limits


# RBR-06: the Hugging Face dry run. The plan is built in a copy of the committed files only, so
# it never reads artifacts/ (every artifact shows as missing there) and never pushes

HUB_FILES = (
    "docs/measurements/export_artifacts_v3b.json",
    "docs/measurements/v3b/merge_record.json",
    "LICENSE",
    "NOTICE",
)


@pytest.fixture(scope="module")
def hub_root(tmp_path_factory, v3b_bundle) -> Path:
    import shutil

    root = tmp_path_factory.mktemp("hub")
    shutil.copytree(LINEAGE, root / "docs" / "measurements" / "lineage")
    for rel in HUB_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    (root / "results_bundle.json").write_bytes(v3b_bundle.read_bytes())
    return root


@pytest.fixture(scope="module")
def hub_plan(hub_root):
    from ggh.release import hub

    return hub.plan(
        hub_root, Path("results_bundle.json"), "card text\n", owner="biplob123", layout="v3b"
    )


def test_the_v3b_hub_plan_ships_both_ggufs_and_the_card(hub_plan):
    from ggh.release import hub

    p = hub_plan
    assert p.release == "v3b" and p.model == Q4
    gguf = {f.path_in_repo: f for f in p.by_repo("gguf")}
    export = _copy("export_ft3b")
    for q in ("Q4_K_M", "Q8_0"):
        f = gguf[f"gemma-4-e2b-grounded-health-{q}.gguf"]
        assert f.local == f"artifacts/gguf/ft3b-{q}.gguf"
        assert f.sha256 == export.outputs[f.local]
    main = {f.path_in_repo: f for f in p.by_repo("main")}
    assert main["README.md"].content == b"card text\n"
    assert {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= set(main)
    assert {"model.safetensors", "config.json", "results_bundle.json", "lineage.json"} <= set(main)
    card = gguf["README.md"].content
    assert card is not None and b"The Q8_0 file is recommended where it fits" in card
    assert b"result on the main model card is for the Q4_K_M file" in card
    assert hub.release_tag("v3b") == "v3b.0.0" == hub.LAYOUTS["v3b"].tag
    assert hub.release_tag("v2") == "v2.0.0"
    d = hub.plan_json(p)
    assert d["tag"] == "v3b.0.0" and "tag" not in hub.plan_json(hub.Plan("o", release="v2"))


def test_the_v3b_hub_plan_matches_the_records(hub_plan):
    p = hub_plan
    train, export = _copy("train_v3b"), _copy("export_ft3b")
    for f in p.files:
        if f.record == "docs/measurements/lineage/export_ft3b.json":
            assert export.outputs[f.local] == f.sha256, f.path_in_repo
        if f.record == "docs/measurements/lineage/train_v3b.json":
            assert train.outputs[f.local] == f.sha256, f.path_in_repo
    merge = next(f for f in p.files if f.path_in_repo == "ggh_merge.json")
    assert merge.sha256 == _sha(V3B / "merge_record.json")
    lineage = next(f for f in p.files if f.path_in_repo == "lineage.json")
    assert lineage.content is not None
    stages = [f"{m['stage']}:{m['key']}" for m in json.loads(lineage.content)]
    want = ["parity_v3b", "export_ft3b", "train_v3b", "train_v2", "targets_v2", "data"]
    assert stages == p.lineage_keys == [f"{_copy(n).stage}:{_copy(n).key}" for n in want]
    # only what a checkout without artifacts/ lacks: never a record mismatch
    bad = [x for x in p.problems if "missing" not in x and "cannot be verified" not in x]
    assert bad == [] and any("missing" in x for x in p.problems) and not p.ready
    committed = ROOT / "docs" / "measurements" / "release_plan_v3b.json"
    if committed.is_file():  # the main session's real dry run, once written
        d = json.loads(committed.read_text(encoding="utf-8"))
        assert d["release"] == "v3b" and d["tag"] == "v3b.0.0" and d["problems"] == []
        assert {f["path_in_repo"]: f["sha256"] for f in d["files"] if f["record"]} == {
            f.path_in_repo: f.sha256 for f in p.files if f.record and f.content is None
        } | {
            f["path_in_repo"]: f["sha256"]
            for f in d["files"]
            if f["record"] and f["local"] == "(generated)"
        }


def test_the_v3b_dry_run_never_pushes(hub_root, v3b_bundle, monkeypatch, capsys):
    import types

    from ggh import cli, cli_release
    from ggh.release import card, hub

    def refuse(*_a, **_k):
        raise AssertionError("the dry run must never reach the Hub")

    monkeypatch.setattr(hub, "push", refuse)
    monkeypatch.setattr(hub, "HfHubApi", refuse)
    from ggh.schemas import ResultsBundle

    bundle = ResultsBundle.model_validate_json(v3b_bundle.read_text(encoding="utf-8"))
    fake = types.SimpleNamespace(
        bundle=bundle, paths=types.SimpleNamespace(bundle=Path("results_bundle.json"))
    )
    monkeypatch.setattr(cli_release, "_sources", lambda _args: fake)
    monkeypatch.setattr(card, "render", lambda _src: "card text\n")
    base = ["release", "--root", str(hub_root), "--release", "v3b", "--owner", "biplob123"]
    out_json = hub_root / "plan.json"
    assert cli.main([*base, "--card", "missing.md", "--plan-json", "plan.json"]) == cli.EXIT_ERROR
    text = capsys.readouterr().out
    assert "dry run, nothing pushed" in text and "release: tag v3b.0.0" in text
    assert json.loads(out_json.read_text(encoding="utf-8"))["tag"] == "v3b.0.0"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert cli.main([*base, "--push"]) == cli.EXIT_ERROR
    assert "needs HF_TOKEN" in capsys.readouterr().out
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    assert cli.main([*base, "--push"]) == cli.EXIT_ERROR  # the plan has problems: refused
    assert "nothing pushed" in capsys.readouterr().out


def test_the_bundle_cli_builds_named_releases(tmp_path, capsys):
    """`ggh results bundle --versions v1,v2` writes v1.0.0's bundle byte for byte; a held
    release is in a bundle only when named."""
    from ggh.cli import main

    args = ["results", "bundle", "--root", str(ROOT), "--out", str(tmp_path)]
    assert main([*args, "--versions", "v1,v2"]) == 0
    assert "releases: v1, v2;" in capsys.readouterr().out
    got = (tmp_path / "results_bundle.json").read_bytes()
    assert got == (FROZEN / "results_bundle.json").read_bytes()
