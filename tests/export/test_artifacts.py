"""EXP-05: train, export and parity stage manifests chain back to the dataset, and every export
output gets an `ExportArtifact` with its hash and the key of the stage that produced it."""

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

from ggh import lineage
from ggh.export import artifacts as A
from ggh.export import parity
from ggh.export.merge import ADAPTER_DIR, MERGED_DIR, SELECTION
from ggh.lineage import StageManifest, sha256_file, stage_key, write_manifest
from ggh.schemas import ExportArtifact
from ggh.serve import llama
from ggh.train.devrun import DEV_EVAL, TRAIN_RUN
from ggh.train.select import ADAPTER_FILES, adapter_hash

ROOT = Path(__file__).resolve().parents[2]
T0 = dt.datetime(2026, 11, 17, tzinfo=dt.UTC)


def put(root, rel, data=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if data is not None else rel.as_posix().encode())
    return sha256_file(p)


def stage(root, name, inputs, outputs, parents):
    m = StageManifest(stage=name, code_version="1", key=stage_key(name, "1", inputs, {}, 7),
                      inputs=inputs, outputs=outputs, params={}, seed=7, parents=parents,
                      started_at=T0, finished_at=T0, host="test")  # fmt: skip
    write_manifest(root / "artifacts" / name, m)
    return m


@pytest.fixture
def repo(tmp_path):
    r = tmp_path
    pool, tr, va = (Path("artifacts/data/train_pool.jsonl"), Path("artifacts/targets/train.jsonl"),
                    Path("artifacts/targets/val.jsonl"))  # fmt: skip
    hf = {"hf://qiaojin/PubMedQA@abc/x.parquet": "a" * 64}
    data_out = {pool.as_posix(): put(r, pool), DEV_EVAL.as_posix(): put(r, DEV_EVAL)}
    data = stage(r, "data", hf, data_out, ())
    tgt_out = {tr.as_posix(): put(r, tr), va.as_posix(): put(r, va)}
    stage(r, "targets", {pool.as_posix(): data_out[pool.as_posix()]}, tgt_out, (data.key,))
    put(r, Path("configs/recipe.yaml"), (ROOT / "configs/recipe.yaml").read_bytes())
    put(r, A.LEDGER, (ROOT / A.LEDGER).read_bytes())
    (r / TRAIN_RUN).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / TRAIN_RUN, r / TRAIN_RUN)
    for f in ADAPTER_FILES:
        put(r, ADAPTER_DIR / f)
    sel = {"adapter_sha256": adapter_hash(r / ADAPTER_DIR)}
    put(r, SELECTION, json.dumps(sel).encode())
    for f in A.MERGED_FILES:
        put(r, MERGED_DIR / f)
    for model in ("base", "ft"):
        for name in (f"{model}-f16.gguf", *(f"{model}-{q}.gguf" for q in llama.QUANTS)):
            put(r, llama.GGUF_DIR / name)
    for rel in (parity.REPORT, parity.GENERATIONS, parity.SAMPLE):
        put(r, rel)
    base = r / "hf-base"
    put(base, Path(A.BASE_WEIGHTS), b"base weights")
    return r, base


def test_the_chain_verifies_from_parity_back_to_the_dataset(repo):
    root, base = repo
    recs = A.write_all(root, base, now=lambda: T0)
    for name in ("train", "export_ft", "export_base", "parity"):
        assert lineage.verify(root / A.STAGE_DIRS[name] / "manifest.json", root) == [], name
    train = lineage.read_manifest(root / A.STAGE_DIRS["train"] / "manifest.json")
    ft = lineage.read_manifest(root / A.STAGE_DIRS["export_ft"] / "manifest.json")
    par = lineage.read_manifest(root / A.STAGE_DIRS["parity"] / "manifest.json")
    targets = lineage.read_manifest(root / "artifacts/targets/manifest.json")
    data = lineage.read_manifest(root / "artifacts/data/manifest.json")
    assert train.parents == (targets.key,) and ft.parents == (train.key,)
    assert par.parents == (ft.key, data.key) and train.host == "kaggle"
    for name, rel in (("data", A.DATA_MANIFEST), ("targets", A.TARGETS_MANIFEST)):
        copy = lineage.read_manifest(root / A.LINEAGE_COPY / f"{name}.json")
        orig = lineage.read_manifest(root / rel)
        assert copy.host == "local" and copy == orig.model_copy(update={"host": "local"})
    for name, d in A.STAGE_DIRS.items():  # committed copies, byte for byte
        assert (root / A.LINEAGE_COPY / f"{name}.json").read_bytes() == (
            root / d / "manifest.json"
        ).read_bytes()
    assert recs == [ExportArtifact.model_validate(x) for x in
                    json.loads((root / A.RECORDS).read_text(encoding="utf-8"))]  # fmt: skip


def test_every_output_has_a_record_with_its_hash_and_producing_stage(repo):
    root, base = repo
    recs = A.write_all(root, base, now=lambda: T0)
    kinds = [(r.kind, r.model, r.quant) for r in recs]
    ggufs = [("gguf", m, q) for m in ("base", "ft") for q in ("F16", "Q4_K_M", "Q8_0")]
    assert kinds == [("adapter", "ft", None), ("merged_bf16", "ft", None), *ggufs]
    keys = {
        n: lineage.read_manifest(root / d / "manifest.json").key for n, d in A.STAGE_DIRS.items()
    }
    for r in recs:
        hashed = (
            root / r.path / "adapter_model.safetensors" if r.kind == "adapter" else root / r.path
        )
        assert r.sha256 == sha256_file(hashed)
        want = {"adapter": keys["train"], "merged_bf16": keys["export_ft"]}.get(r.kind)
        want = want or keys[f"export_{r.model}"]
        assert r.parent_key == want
        assert (r.llama_cpp_commit == llama.TAG) == (r.kind == "gguf")


def test_a_targets_file_changed_since_its_stage_ran_is_refused(repo):
    root, base = repo
    (root / "artifacts/targets/train.jsonl").write_bytes(b"edited")
    with pytest.raises(ValueError, match="differs from what the targets stage recorded"):
        A.write_all(root, base, now=lambda: T0)


def test_an_edited_gguf_breaks_the_chain(repo):
    root, base = repo
    A.write_all(root, base, now=lambda: T0)
    (root / llama.GGUF_DIR / "ft-Q4_K_M.gguf").write_bytes(b"swapped")
    problems = lineage.verify(root / A.STAGE_DIRS["parity"] / "manifest.json", root)
    assert problems and any("ft-Q4_K_M.gguf" in p for p in problems)


def test_a_missing_output_is_an_error_not_a_skipped_record(repo):
    root, base = repo
    (root / llama.GGUF_DIR / "ft-Q8_0.gguf").unlink()
    with pytest.raises(FileNotFoundError, match=r"ft-Q8_0\.gguf"):
        A.write_all(root, base, now=lambda: T0)


def test_the_train_manifest_refuses_another_recipe_or_adapter(repo):
    root, _ = repo
    (root / ADAPTER_DIR / "adapter_config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="not the frozen adapter"):
        A.train_manifest(root)
    (root / "configs/recipe.yaml").write_bytes(b"seed: 8\n")
    with pytest.raises(ValueError, match="not the recipe the run recorded"):
        A.train_manifest(root)


def test_the_train_manifest_spans_the_kaggle_session_from_the_ledger(repo):
    root, _ = repo
    m = A.train_manifest(root)
    (row,) = [json.loads(x) for x in (ROOT / A.LEDGER).read_text(encoding="utf-8").splitlines()
              if x and json.loads(x).get("kernel") == "ggh-t22-train v1"]  # fmt: skip
    assert m.finished_at == dt.datetime.fromisoformat(row["at"])
    assert (m.finished_at - m.started_at).total_seconds() == pytest.approx(row["gpu_hours"] * 3600)
    assert m.host == "kaggle" and m.params["kaggle_kernel"] == "ggh-t22-train v1"


# v2 (`ft2`, V2.1): `write_v2`, `ggh export-records --model ft2`


@pytest.fixture
def repo_v2(repo):
    from ggh.versions import version

    root, base = repo
    v = version("ft2")
    put(root, v.recipe, (ROOT / v.recipe).read_bytes())
    put(root, v.train_run, (ROOT / v.train_run).read_bytes())  # step 340 on recipe_v2
    for f in ADAPTER_FILES:
        put(root, v.adapter_dir / f, f"v2 {f}".encode())
    frozen = adapter_hash(root / v.adapter_dir)
    put(root, v.selection, json.dumps({"adapter_sha256": frozen, "selected_step": 340}).encode())
    for f in A.MERGED_FILES:
        put(root, v.merged_dir / f, f"v2 {f}".encode())
    files = {rel.name: {"sha256": put(root, rel)} for rel in A._gguf_rels("ft2")}
    gg = {"files": files, "source": {"revision": frozen},
          "llama_cpp": {"tag": llama.TAG, "release_sha256": llama.BIN_SHA256}}  # fmt: skip
    put(root, A.GGUF_RECORD_V2, json.dumps(gg).encode())
    put(root, A.MERGE_RECORD_V2, json.dumps({"adapter_sha256": frozen}).encode())
    for rel in parity.paths("ft2"):
        put(root, rel)
    A.write_all(root, base, now=lambda: T0)  # v1 first: its export_base manifest is v2's base
    return root, base


def test_v2_chain_verifies_beside_v1_and_leaves_v1_alone(repo_v2):
    root, base = repo_v2
    v1 = {p: p.read_bytes() for p in [root / A.RECORDS, *(root / A.LINEAGE_COPY).glob("*.json")]}
    recs = A.write_v2(root, base, now=lambda: T0)
    assert {p: p.read_bytes() for p in v1} == v1
    for name, d in A.V2_STAGE_DIRS.items():
        assert lineage.verify(root / d / "manifest.json", root) == [], name
        assert (root / A.LINEAGE_COPY / f"{name}.json").read_bytes() == (
            root / d / "manifest.json"
        ).read_bytes()
    m = {n: lineage.read_manifest(root / d / "manifest.json") for n, d in A.V2_STAGE_DIRS.items()}
    targets = lineage.read_manifest(root / A.TARGETS_MANIFEST)
    data = lineage.read_manifest(root / A.DATA_MANIFEST)
    base_m = lineage.read_manifest(root / A.STAGE_DIRS["export_base"] / "manifest.json")
    assert m["train_v2"].parents == (targets.key,) and m["train_v2"].host == "kaggle"
    assert m["train_v2"].params["kaggle_kernel"] == "ggh-v21-train v1"
    assert "configs/recipe_v2.yaml" in m["train_v2"].inputs
    assert m["export_ft2"].parents == (m["train_v2"].key,)
    assert m["parity_v2"].parents == (m["export_ft2"].key, data.key)
    assert set(m["parity_v2"].outputs) == {p.as_posix() for p in parity.paths("ft2")}
    copy = lineage.read_manifest(root / A.LINEAGE_COPY / "targets_v2.json")
    assert copy == targets.model_copy(update={"host": "local"})
    assert recs == [ExportArtifact.model_validate(x) for x in
                    json.loads((root / A.RECORDS_V2).read_text(encoding="utf-8"))]  # fmt: skip
    assert [(r.kind, r.model, r.quant) for r in recs] == [
        ("adapter", "ft", None), ("merged_bf16", "ft", None),
        *(("gguf", k, q) for k in ("base", "ft") for q in ("F16", "Q4_K_M", "Q8_0"))]  # fmt: skip
    for r in recs:
        hashed = (
            root / r.path / "adapter_model.safetensors" if r.kind == "adapter" else root / r.path
        )
        assert r.sha256 == sha256_file(hashed)
        if r.model == "base":
            assert r.parent_key == base_m.key and r.path.startswith("artifacts/gguf/base-")
        else:
            assert "ft2" in r.path or "adapter_v2" in r.path
            assert r.parent_key == (m["train_v2"] if r.kind == "adapter" else m["export_ft2"]).key


def test_v2_records_must_match_what_v21_recorded(repo_v2):
    root, base = repo_v2
    gg = json.loads((root / A.GGUF_RECORD_V2).read_text(encoding="utf-8"))
    gg["files"]["ft2-Q4_K_M.gguf"]["sha256"] = "0" * 64
    (root / A.GGUF_RECORD_V2).write_text(json.dumps(gg), encoding="utf-8")
    with pytest.raises(ValueError, match=r"ft2-Q4_K_M\.gguf differs from what"):
        A.write_v2(root, base, now=lambda: T0)
    (root / A.MERGE_RECORD_V2).write_text(json.dumps({"adapter_sha256": "1" * 64}), "utf-8")
    with pytest.raises(ValueError, match="names another adapter"):
        A.write_v2(root, base, now=lambda: T0)
    assert not (root / A.RECORDS_V2).exists()


def test_v2_refuses_base_ggufs_that_differ_from_v1s_export(repo_v2):
    root, base = repo_v2
    (root / llama.GGUF_DIR / "base-Q4_K_M.gguf").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="base GGUFs differ"):
        A.write_v2(root, base, now=lambda: T0)
