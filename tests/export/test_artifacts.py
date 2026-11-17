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
        A.train_manifest(root, now=lambda: T0)
    (root / "configs/recipe.yaml").write_bytes(b"seed: 8\n")
    with pytest.raises(ValueError, match="not the recipe the run recorded"):
        A.train_manifest(root, now=lambda: T0)


def test_the_train_manifest_spans_the_kaggle_session_from_the_ledger(repo):
    root, _ = repo
    m = A.train_manifest(root, now=lambda: T0)
    (row,) = [json.loads(x) for x in (ROOT / A.LEDGER).read_text(encoding="utf-8").splitlines()
              if x and json.loads(x).get("kernel") == "ggh-t22-train v1"]  # fmt: skip
    assert m.finished_at == dt.datetime.fromisoformat(row["at"])
    assert (m.finished_at - m.started_at).total_seconds() == pytest.approx(row["gpu_hours"] * 3600)
    assert m.host == "kaggle" and m.params["kaggle_kernel"] == "ggh-t22-train v1"
