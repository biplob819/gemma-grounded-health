import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evalkit.schemas import Case
from evalkit.tasks import builtin_task, render_prompt
from evalkit.text import number_sentences

from ggh.data import build, fetch
from ggh.lineage import sha256_file
from ggh.schemas import DataManifest

ROOT = Path(__file__).resolve().parents[2]
LEAK_MARK = "Zebrafinchquartzlemma"  # a word only the PQA-L fixture contains


def ctx(paras, labels=None):
    return {"contexts": paras, "labels": labels or ["BACKGROUND"] * len(paras), "meshes": []}


def abstract(i):
    return [
        f"Study {i} enrolled patients in cohort number {i} across several clinics.",
        f"Results for cohort {i} showed measurable change. Outcome {i} improved overall.",
    ]


def write_sources(tmp, n=40, leaks=True):
    a_rows = [
        {
            "pubid": 1000 + i,
            "question": f"Does treatment {i} help condition {i}?",
            "context": ctx(abstract(i), ["METHODS", "RESULTS"]),
            "long_answer": f"Treatment {i} helped.",
            "final_decision": "yes" if i % 3 else "no",
        }
        for i in range(n)
    ]
    # planted leaks: same PMID, same question (other casing and punctuation), near-duplicate context
    l_rows = [
        {
            "pubid": 1000,
            "question": f"{LEAK_MARK} unrelated question one?",
            "context": ctx([f"{LEAK_MARK} text one entirely different words here please now."]),
            "final_decision": "yes",
        },
        {
            "pubid": 9001,
            "question": "DOES treatment 1 -- help condition 1",
            "context": ctx([f"{LEAK_MARK} text two with other words that do not overlap at all."]),
            "final_decision": "no",
        },
        {
            "pubid": 9002,
            "question": f"{LEAK_MARK} three?",
            "context": ctx(abstract(2)),
            "final_decision": "maybe",
        },
        {
            "pubid": 9003,
            "question": f"{LEAK_MARK} four?",
            "context": ctx([f"{LEAK_MARK} nothing in common with any training row at all here."]),
            "final_decision": "maybe",
        },
    ]
    # near-duplicates around the threshold: a context has 15 8-gram shingles, so row 5 plus 12
    # new words is 15/27 ≈ 0.56 (dropped), row 7 plus 15 is exactly 15/30 = 0.5 (dropped, the
    # rule is >=) and row 6 plus 18 is 15/33 ≈ 0.45 (kept)
    for pubid, i, k in ((9004, 5, 12), (9006, 7, 15), (9005, 6, 18)):
        extra = " ".join(f"{LEAK_MARK}{j}" for j in range(k))
        l_rows.append(
            {
                "pubid": pubid,
                "question": f"{LEAK_MARK} near {i}?",
                "context": ctx([" ".join(abstract(i)) + " " + extra]),
                "final_decision": "yes",
            }
        )
    if not leaks:  # nothing to drop; the 0.45 near-duplicate stays as the highest similarity
        l_rows = [r for r in l_rows if r["pubid"] in (9003, 9005)]
    a, lab = tmp / "a.parquet", tmp / "l.parquet"
    pq.write_table(pa.Table.from_pylist(a_rows), a)
    pq.write_table(pa.Table.from_pylist(l_rows), lab)
    return a, lab


def run(tmp, seed=7, n=40, n_off=2, offscope=None, leaks=True):
    root = tmp / "repo"
    (root / "data").mkdir(parents=True, exist_ok=True)
    a, lab = write_sources(tmp, n, leaks)
    if offscope is None:
        # pick two dev PMIDs by running the split the build will run
        rows = [build.to_row(r) for r in build.read_parquet(a, with_answers=True)]
        index = build.LeakageIndex(build.read_parquet(lab, with_answers=False))
        kept = [r for r in rows if not index.reason(r)]
        _, dev, _ = build.split(kept, seed, 6, 4)
        offscope = [
            {"id": f"off-{i}", "pmid": r.pmid, "question": "Should I double my dose?"}
            for i, r in enumerate(dev[:n_off])
        ]
    (root / build.OFFSCOPE_PATH).write_text(
        "".join(json.dumps(o) + "\n" for o in offscope), encoding="utf-8"
    )
    first = not (root / build.MANIFEST_PATH).exists()  # a first build freezes the set
    m = build.build(
        root, seed=seed, sources=(a, lab), sizes=(n, 6, 4, n_off), refreeze_offscope=first
    )
    return root, m


@pytest.mark.data
def test_planted_leaks_of_each_kind_are_dropped_and_counted(tmp_path):
    root, m = run(tmp_path)
    # labels and outputs come through intact: per-label counts and the exact files map
    a_rows = build.read_parquet(tmp_path / "a.parquet", with_answers=True)
    dropped = {1000, 1001, 1002, 1005, 1007}
    want = Counter(r["final_decision"] for r in a_rows if r["pubid"] not in dropped)
    got = Counter()
    for k, c in m.counts.items():
        if k != "offscope_eval":
            got.update(c)
    assert got == want
    assert set(m.files) == {
        f"artifacts/data/{n}"
        for n in (
            "train_pool.jsonl",
            "val.jsonl",
            "dev_eval.jsonl",
            "overlap_report.json",
            "length_report.json",
        )
    } | {f"docs/measurements/data_{n}" for n in ("overlap_report.json", "length_report.json")}
    pool = build.read_jsonl(root / build.OUT_DIR / "train_pool.jsonl")
    src = {str(r["pubid"]): r for r in a_rows}
    assert all(r["long_answer"] == src[r["pmid"]]["long_answer"] for r in pool)
    assert dict(m.drops) == {
        "overlap_with_pqa_l:context": 3,
        "overlap_with_pqa_l:pmid": 1,
        "overlap_with_pqa_l:question": 1,
    }
    kept = sum(sum(c.values()) for k, c in m.counts.items() if k != "offscope_eval")
    assert kept == 40 - 5


@pytest.mark.data
@pytest.mark.safety
def test_no_pqa_l_text_reaches_any_output(tmp_path):
    root, _ = run(tmp_path)
    for f in (root / build.OUT_DIR).iterdir():
        if f.name != "overlap_report.json":
            assert LEAK_MARK not in f.read_text(encoding="utf-8"), f.name
    pool = build.read_jsonl(root / build.OUT_DIR / "train_pool.jsonl")
    assert {r["pmid"] for r in pool}.isdisjoint({"1000", "1001", "1002", "1005", "1007"})
    assert all("prompt" not in r for r in pool)  # N1.3 renders what it draws


@pytest.mark.data
def test_split_sizes_are_disjoint_and_seeded(tmp_path):
    root, _ = run(tmp_path)

    def pmids(name):
        return [r["pmid"] for r in build.read_jsonl(root / build.OUT_DIR / f"{name}.jsonl")]

    val, dev, pool = pmids("val"), pmids("dev_eval"), pmids("train_pool")
    assert (len(val), len(dev), len(pool)) == (6, 4, 25)
    assert not (set(val) & set(dev) or set(val) & set(pool) or set(dev) & set(pool))
    rows = [build.to_row(r) for r in build.read_parquet(tmp_path / "a.parquet", with_answers=True)]
    a1 = build.split(rows, 7, 6, 4)
    assert [r.pmid for r in build.split(rows, 7, 6, 4)[0]] == [r.pmid for r in a1[0]]
    assert [r.pmid for r in build.split(rows, 8, 6, 4)[0]] != [r.pmid for r in a1[0]]


@pytest.mark.data
def test_same_seed_reproduces_the_manifest_byte_for_byte(tmp_path):
    root, _ = run(tmp_path / "one")
    first = (root / build.MANIFEST_PATH).read_bytes()
    root2, _ = run(tmp_path / "two")
    assert (root2 / build.MANIFEST_PATH).read_bytes() == first
    DataManifest.model_validate_json(first)


@pytest.mark.contract
def test_render_is_byte_equal_to_evalkit():
    paras = ["Background sentence one. Another one here.", "Results showed things. Final line."]
    row = build.Row(
        "42",
        "Does it work?",
        *build.sentences_with_sections(paras, ["BACKGROUND", "RESULTS"]),
        "",
        "yes",
    )
    case = Case(
        id="pqa-a-42",
        question="Does it work?",
        context=number_sentences(paras),
        reference_label="yes",
        source="pubmedqa/pqa_a",
    )
    assert build.render(row) == render_prompt(builtin_task("pubmedqa"), case)
    assert row.sections == ("BACKGROUND", "BACKGROUND", "RESULTS", "RESULTS")
    bare = build.Row("43", "Is aspirin linked to bleeding", row.sentences, row.sections, "", "no")
    case = case.model_copy(
        update={
            "id": "pqa-a-43",
            "question": "Is aspirin linked to bleeding",
            "reference_label": "no",
        }
    )
    assert build.render(bare) == render_prompt(builtin_task("pubmedqa"), case)


def test_sections_fall_back_when_labels_do_not_line_up():
    sents, secs = build.sentences_with_sections(["One. Two.", "Three."], ["ONLY_ONE"])
    assert sents == number_sentences(["One. Two.", "Three."]) and set(secs) == {"UNLABELLED"}


def test_length_report_matches_a_recomputation():
    rows = [
        build.Row(
            str(i), "Q?", ("A.",), ("X",), " ".join(["w"] * (i * 10)), "yes" if i % 2 else "no"
        )
        for i in range(1, 12)
    ]
    rep = build.length_report(rows)
    assert rep["pqa_a_labels"] == {"no": 5, "yes": 6}
    assert rep["long_answer_over_80_words"] == sum(i * 10 > 80 for i in range(1, 12))
    assert rep["long_answer_words"]["max"] == 110.0


def test_offscope_rules(tmp_path):
    f = tmp_path / "o.jsonl"
    f.write_text(json.dumps({"id": "a", "pmid": "1", "question": "Q?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2"):
        build.load_offscope(f, {"1"}, 2)
    with pytest.raises(ValueError, match="not a dev_eval abstract"):
        build.load_offscope(f, {"2"}, 1)
    f.write_text(
        "".join(json.dumps({"id": "a", "pmid": "1", "question": "Q?"}) + "\n" for _ in range(2)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate ids"):
        build.load_offscope(f, {"1"}, 2)


def test_fetch_refuses_a_corrupt_file_and_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GGH_CACHE", str(tmp_path))
    target = fetch.cache_dir() / "pqa_labeled.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the real file")
    with pytest.raises(fetch.DataError, match="pqa_labeled"):
        fetch.ensure(fetch.PQA_L, download=False)
    assert target.exists()

    def no_network(*a, **k):
        raise AssertionError("a bad cached file must never be re-downloaded over")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", no_network)
    with pytest.raises(fetch.DataError, match="delete it to download again"):
        fetch.ensure(fetch.PQA_L, download=True)
    assert target.read_bytes() == b"not the real file"
    target.unlink()
    with pytest.raises(fetch.DataError, match="downloading is off"):
        fetch.ensure(fetch.PQA_L, download=False)
    assert (
        fetch.uri(fetch.PQA_A)
        == f"hf://qiaojin/PubMedQA@{fetch.REVISION}/pqa_artificial/train-00000-of-00001.parquet"
    )


# ---- the committed data (these read the repo, not fixtures) ----

MANIFEST = ROOT / build.MANIFEST_PATH
OFFSCOPE = ROOT / build.OFFSCOPE_PATH


@pytest.mark.data
def test_committed_manifest_is_valid_and_consistent():
    m = DataManifest.model_validate_json(MANIFEST.read_text(encoding="utf-8"))
    assert (
        m.dataset_revision == fetch.REVISION
        and m.seed == 7
        and m.template_hash == build.template_hash()
    )
    sizes = {k: sum(v.values()) for k, v in m.counts.items()}
    assert (sizes["val"], sizes["dev_eval"], sizes["offscope_eval"]) == (2000, 500, 20)
    assert (
        sizes["train_pool"] + sizes["val"] + sizes["dev_eval"] + sum(m.drops.values())
        == fetch.PQA_A.rows
    )
    length = json.loads(
        (ROOT / build.MEASURE_DIR / "data_length_report.json").read_text(encoding="utf-8")
    )
    assert length["pqa_a_labels"] == {"no": 15125, "yes": 196144}  # measured 2026-11-15
    assert set(m.files) == {
        "artifacts/data/train_pool.jsonl",
        "artifacts/data/val.jsonl",
        "artifacts/data/dev_eval.jsonl",
        "artifacts/data/overlap_report.json",
        "artifacts/data/length_report.json",
        "docs/measurements/data_overlap_report.json",
        "docs/measurements/data_length_report.json",
    }
    for rel in (
        "docs/measurements/data_overlap_report.json",
        "docs/measurements/data_length_report.json",
    ):
        assert sha256_file(ROOT / rel) == m.files[rel], rel  # committed, so checked everywhere
    if (ROOT / build.OUT_DIR / "dev_eval.jsonl").exists():  # artifacts are local only
        for rel, digest in m.files.items():
            assert sha256_file(ROOT / rel) == digest, rel


@pytest.mark.data
@pytest.mark.safety
def test_offscope_eval_set_is_frozen_and_well_formed():
    m = DataManifest.model_validate_json(MANIFEST.read_text(encoding="utf-8"))
    assert sha256_file(OFFSCOPE) == m.offscope_eval_hash  # changing the set fails here (OFF-R1)
    rows = build.read_jsonl(OFFSCOPE)
    assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
    for r in rows:
        assert r["expected"]["label"] == "maybe"
        row = build.Row(r["pmid"], r["question"], tuple(r["sentences"]), (), "", "maybe")
        assert r["prompt"] == build.render(row)
        assert any(
            w in r["question"] for w in ("I ", "my ", "My ", "I'm")
        )  # personal, not research


@pytest.mark.data
@pytest.mark.skipif(
    not os.environ.get("GGH_DATA_REBUILD"),
    reason="full rebuild takes ~3 min; set GGH_DATA_REBUILD=1",
)
def test_full_real_rebuild_reproduces_the_committed_manifest(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copy(OFFSCOPE, tmp_path / build.OFFSCOPE_PATH)
    (tmp_path / build.MEASURE_DIR).mkdir(parents=True)
    shutil.copy(MANIFEST, tmp_path / build.MANIFEST_PATH)  # the build checks the frozen hash
    build.build(tmp_path, download=False)
    assert (tmp_path / build.MANIFEST_PATH).read_bytes() == MANIFEST.read_bytes()


@pytest.mark.data
def test_data_card_claims_match_the_data():
    """EK-L-16: the card states as fact only what the committed manifest shows."""
    card = (ROOT / "DATA_CARD.md").read_text(encoding="utf-8")
    m = DataManifest.model_validate_json(MANIFEST.read_text(encoding="utf-8"))
    assert fetch.REVISION in card and "MIT" in card
    for split, counts in m.counts.items():
        assert f"{sum(counts.values()):,}" in card, split
    pool = m.counts["train_pool"]
    assert f"yes {pool['yes']:,}, no {pool['no']:,}" in card
    assert ("drops **0 rows**" in card) == (sum(m.drops.values()) == 0)
    assert "Research only, not medical advice." in card
    assert "60:40" in card and "is pinned in a test" in card
    targets = json.loads(
        (ROOT / "docs/measurements/targets_drop_report.json").read_text(encoding="utf-8")
    )  # N1.3's counts, stated as facts only once they exist (EK-L-16)
    assert f"draws {targets['train']:,} training and {targets['val']:,} validation" in card
    for split in ("train", "val"):
        for kind in ("yes", "no", "insufficient", "offscope"):
            assert f"{targets['quotas'][split][kind]:,}" in card, (split, kind)
    assert "PubMedQA's own" in card and "2.21%" in card  # the true source of UNLABELLED
    assert "training and evaluation see" not in card  # training prompts do not exist yet


@pytest.mark.data
def test_threshold_fixtures_really_straddle_0_5(tmp_path):
    a, lab = write_sources(tmp_path)
    rows = {r["pubid"]: build.to_row(r) for r in build.read_parquet(a, with_answers=True)}
    lrows = {r["pubid"]: r for r in build.read_parquet(lab, with_answers=False)}

    import re as _re

    def grams(text):  # independent of build.shingles, so a changed constant cannot hide
        w = _re.findall(r"\w+", text.lower())
        return {tuple(w[i : i + 8]) for i in range(len(w) - 7)}

    def jac(pmid_a, pmid_l):
        s = grams(" ".join(rows[pmid_a].sentences))
        t = grams(" ".join(lrows[pmid_l]["context"]["contexts"]))
        return len(s & t) / len(s | t)

    assert (build.SHINGLE, build.JACCARD_DROP) == (8, 0.5)  # Tech Spec DAT-R3
    assert 0.5 < jac(1005, 9004) < 0.6 and jac(1007, 9006) == 0.5 and 0.4 < jac(1006, 9005) < 0.5


@pytest.mark.data
def test_overlap_report_shows_explicit_zeros_and_the_highest_jaccard(tmp_path):
    root, m = run(tmp_path)
    rep = json.loads((root / build.OUT_DIR / "overlap_report.json").read_text(encoding="utf-8"))
    assert set(rep["dropped_from_pqa_a"]) == {"pmid", "question", "context"}
    assert rep["pqa_a_rows_checked"] == 40 and rep["max_context_jaccard"] == 1.0
    assert set(m.drops) == {
        "overlap_with_pqa_l:pmid",
        "overlap_with_pqa_l:question",
        "overlap_with_pqa_l:context",
    }
    # with nothing to drop, every count is still written, as an explicit zero
    root, m = run(tmp_path / "clean", leaks=False)
    rep = json.loads((root / build.OUT_DIR / "overlap_report.json").read_text(encoding="utf-8"))
    assert rep["dropped_from_pqa_a"] == {"pmid": 0, "question": 0, "context": 0}
    assert dict(m.drops) == {f"overlap_with_pqa_l:{k}": 0 for k in ("context", "pmid", "question")}
    assert rep["max_context_jaccard"] == round(15 / 33, 6)  # full precision, not rounded away


@pytest.mark.data
@pytest.mark.safety
def test_a_changed_offscope_set_is_refused_unless_refrozen(tmp_path):
    root, _ = run(tmp_path)
    f = root / build.OFFSCOPE_PATH
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"] = "Should I stop instead?"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    a, lab = tmp_path / "a.parquet", tmp_path / "l.parquet"
    shutil.rmtree(root / build.OUT_DIR)
    for name in ("data_overlap_report.json", "data_length_report.json"):
        (root / build.MEASURE_DIR / name).unlink()
    manifest = (root / build.MANIFEST_PATH).read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        build.build(root, sources=(a, lab), sizes=(40, 6, 4, 2))
    # refused before writing anything: no artifacts, no evidence copies, manifest untouched
    assert not (root / build.OUT_DIR).exists()
    assert sorted(p.name for p in (root / build.MEASURE_DIR).iterdir()) == ["data_manifest.json"]
    assert (root / build.MANIFEST_PATH).read_bytes() == manifest
    (root / build.MANIFEST_PATH).unlink()
    with pytest.raises(ValueError, match="is missing"):  # no manifest is not a free pass
        build.build(root, sources=(a, lab), sizes=(40, 6, 4, 2))
    assert not (root / build.OUT_DIR).exists()
    build.build(root, sources=(a, lab), sizes=(40, 6, 4, 2), refreeze_offscope=True)


@pytest.mark.data
def test_a_seed_that_breaks_the_offscope_set_writes_nothing(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    a, lab = write_sources(tmp_path)
    rows = [build.to_row(r) for r in build.read_parquet(a, with_answers=True)]
    index = build.LeakageIndex(build.read_parquet(lab, with_answers=False))
    kept = [r for r in rows if not index.reason(r)]
    dev7 = {r.pmid for r in build.split(kept, 7, 6, 4)[1]}
    other = next(
        s for s in range(8, 200) if not {r.pmid for r in build.split(kept, s, 6, 4)[1]} >= dev7
    )
    off = [
        {"id": f"off-{i}", "pmid": p, "question": "Should I?"} for i, p in enumerate(sorted(dev7))
    ]
    (root / build.OFFSCOPE_PATH).write_text(
        "".join(json.dumps(o) + "\n" for o in off), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a dev_eval abstract"):
        build.build(root, seed=other, sources=(a, lab), sizes=(40, 6, 4, 4), refreeze_offscope=True)
    assert not (root / build.OUT_DIR).exists()
    # a val abstract is not a dev_eval abstract either, at the build level too
    val7 = sorted(r.pmid for r in build.split(kept, 7, 6, 4)[0])
    off[0]["pmid"] = val7[0]
    (root / build.OFFSCOPE_PATH).write_text(
        "".join(json.dumps(o) + "\n" for o in off), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a dev_eval abstract"):
        build.build(root, seed=7, sources=(a, lab), sizes=(40, 6, 4, 4), refreeze_offscope=True)
    assert not (root / build.OUT_DIR).exists()


def test_length_report_quantiles_match_an_independent_recomputation():
    import re as _re
    import statistics as st

    rows = [
        build.Row(
            str(i),
            f"Q{i}?",
            tuple(f"Sentence {j} of {i}." for j in range(1 + i % 5)),
            ("X",) * (1 + i % 5),
            " ".join(["w"] * (i * 7 % 120)),
            "yes",
        )
        for i in range(1, 60)
    ]
    rep = build.length_report(rows)
    words = [len(r.long_answer.split()) for r in rows]
    q = st.quantiles(words, n=100, method="inclusive")
    assert rep["long_answer_words"] == {
        "p50": q[49],
        "p90": q[89],
        "p99": q[98],
        "max": float(max(words)),
    }
    toks = [len(_re.findall(r"\w+|[^\w\s]", build.render(r))) for r in rows]
    tq = st.quantiles(toks, n=100, method="inclusive")
    assert (
        rep["rendered_tokens_proxy"]["p50"] == tq[49]
        and rep["rendered_tokens_proxy"]["p99"] == tq[98]
    )


def test_a_bad_download_never_takes_the_cache_name(tmp_path, monkeypatch):
    import io

    monkeypatch.setenv("GGH_CACHE", str(tmp_path))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"truncated"))
    with pytest.raises(fetch.DataError, match="download of"):
        fetch.ensure(fetch.PQA_L, download=True)
    assert list(fetch.cache_dir().iterdir()) == []  # no target, no leftover .part


@pytest.mark.data
def test_manifest_is_reproducible_across_processes_with_different_hash_seeds(tmp_path):
    import subprocess
    import sys

    script = (
        f"import sys; from pathlib import Path; sys.path.insert(0, {str(ROOT)!r});"
        "from tests.data.test_build import run;"
        "root, _ = run(Path(sys.argv[1]));"
        "print((root / 'docs/measurements/data_manifest.json').read_text(encoding='utf-8'))"
    )
    outs = []
    for seed, sub in (("1", "a"), ("2", "b")):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / sub)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        outs.append(out.stdout)
    assert outs[0] == outs[1] and '"seed": 7' in outs[0]


@pytest.mark.data
@pytest.mark.skipif(
    not (ROOT / build.OUT_DIR / "dev_eval.jsonl").exists(), reason="artifacts are local"
)
def test_twenty_real_rows_render_exactly_as_evalkit_renders_them():
    rows = build.read_jsonl(ROOT / build.OUT_DIR / "dev_eval.jsonl")
    for r in rows[:20]:
        case = Case(
            id=f"pqa-a-{r['pmid']}",
            question=r["question"],
            context=tuple(r["sentences"]),
            reference_label=r["label"],
            source="pubmedqa/pqa_a",
        )
        assert r["prompt"] == render_prompt(builtin_task("pubmedqa"), case)


OFFSCOPE_SHA256 = (
    "0c110a35fa7e6182daef833ce388ea72867e2e5973b50d025efb5b0f7890073d"  # owner-approved 2026-11-15
)


@pytest.mark.data
@pytest.mark.safety
def test_offscope_file_matches_the_hash_pinned_in_this_test():
    """Pinned here, independently of the manifest the build writes (DAT-06)."""
    assert sha256_file(OFFSCOPE) == OFFSCOPE_SHA256


PQA_A_CACHE = fetch.cache_dir() / "pqa_artificial.parquet"


@pytest.mark.data
@pytest.mark.skipif(not PQA_A_CACHE.exists(), reason="pinned parquet is not cached here")
def test_real_parquet_rows_go_through_the_build_code_exactly_as_evalkit_renders_them():
    """parquet -> build.to_row -> build.render, against a Case built straight from the raw row."""
    batch = next(pq.ParquetFile(PQA_A_CACHE).iter_batches(batch_size=50))
    for raw in batch.to_pylist():
        case = Case(
            id=f"pqa-a-{raw['pubid']}",
            question=raw["question"],
            context=number_sentences(list(raw["context"]["contexts"])),
            reference_label=raw["final_decision"],
            source="pubmedqa/pqa_a",
        )
        assert build.render(build.to_row(raw)) == render_prompt(builtin_task("pubmedqa"), case)


@pytest.mark.data
@pytest.mark.skipif(not PQA_A_CACHE.exists(), reason="pinned parquet is not cached here")
def test_unlabelled_sections_come_from_pubmedqa_itself():
    """DATA_CARD: UNLABELLED is PubMedQA's own label value (2.21% of abstracts); the
    count-mismatch fallback in sentences_with_sections never fires on the pinned data."""
    ctx_col = pq.read_table(PQA_A_CACHE, columns=["context"]).column("context").to_pylist()
    mismatched = sum(len(c["labels"] or []) != len(c["contexts"]) for c in ctx_col)
    unlabelled = sum("UNLABELLED" in (c["labels"] or []) for c in ctx_col)
    assert mismatched == 0 and unlabelled == 4_662
    assert f"{100 * unlabelled / len(ctx_col):.2f}%" == "2.21%"


def test_jsonl_survives_unicode_line_separators(tmp_path):
    f = tmp_path / "x.jsonl"
    rows = [{"a": "one\u2028two"}, {"a": "three\u2029four\x85five"}]
    build._dump(f, rows)
    assert f.read_bytes().isascii() and f.read_bytes().count(b"\n") == 2
    assert build.read_jsonl(f) == rows
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert build.read_jsonl(f) == rows  # raw separators inside strings are read correctly too


@pytest.mark.data
@pytest.mark.safety
def test_ggh_data_cli_passes_seed_and_never_refreezes_by_default(monkeypatch):
    from ggh import cli

    seen = {}

    def fake_build(root, **kw):
        seen.update(kw)
        raise ValueError("stop here")

    monkeypatch.setattr(build, "build", fake_build)
    assert cli.main(["data", "--root", ".", "--seed", "11", "--offline"]) == cli.EXIT_ERROR
    assert seen == {"seed": 11, "download": False, "refreeze_offscope": False}
    cli.main(["data", "--refreeze-offscope"])
    assert seen == {"seed": 7, "download": True, "refreeze_offscope": True}
