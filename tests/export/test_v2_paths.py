"""V2.1: the v2 fine-tune (`ft2`) goes through the v1 pipeline under its own paths, and v1's
paths stay where X3.1 built them. Needs no artifacts/, torch or network."""

import json
from pathlib import Path

import pytest

from ggh import cli
from ggh.data import build as data
from ggh.export import gguf, merge
from ggh.serve import llama
from ggh.train import devrun
from ggh.train import select as sel
from ggh.versions import VERSIONS, measurements, version

ROOT = Path(__file__).resolve().parents[2]


def test_v1_paths_are_unchanged_and_v2_has_its_own():
    v1, v2 = VERSIONS["ft"], VERSIONS["ft2"]
    assert merge.ADAPTER_DIR == v1.adapter_dir == Path("artifacts/adapter")
    assert merge.MERGED_DIR == v1.merged_dir == Path("artifacts/merged/ft-bf16")
    assert merge.SELECTION == v1.selection == Path("docs/measurements/train_selection.json")
    assert v1.train_run == devrun.TRAIN_RUN and v1.recipe == Path("configs/recipe.yaml")
    assert v2.adapter_dir == Path("artifacts/adapter_v2")
    assert v2.merged_dir == Path("artifacts/merged/ft2-bf16")
    assert v2.train_run == Path("docs/measurements/v2/train_run.json")
    assert v2.selection == Path("docs/measurements/v2/train_selection.json")
    assert v2.recipe == Path("configs/recipe_v2.yaml")
    assert measurements("base") == measurements("ft") == Path("docs/measurements")
    assert measurements("ft2") == Path("docs/measurements/v2")
    with pytest.raises(ValueError, match="must be one of"):
        version("ft3")


def test_ft2_serves_on_8083_from_its_own_ggufs():
    assert llama.PORTS["ft2"] == 8083 and len(set(llama.PORTS.values())) == len(llama.PORTS)
    assert 8082 not in llama.PORTS.values()  # `ggh serve-hf`
    for q in llama.QUANTS:
        assert llama.model_path(Path("/r"), "ft2", q) == Path(f"/r/artifacts/gguf/ft2-{q}.gguf")
    args = llama.server_args(Path("llama-server"), Path("m.gguf"), llama.PORTS["ft2"])
    assert args[args.index("--reasoning-budget") + 1] == "0"  # D-41 holds for v2 too


def test_frozen_hash_reads_each_versions_selection(tmp_path):
    for model, sha in (("ft", "ab" * 32), ("ft2", "cd" * 32)):
        p = tmp_path / version(model).selection
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"adapter_sha256": sha}), encoding="utf-8")
    assert merge.frozen_hash(tmp_path) == "ab" * 32
    assert merge.frozen_hash(tmp_path, "ft2") == "cd" * 32


def test_merge_ft2_reads_the_v2_adapter_and_writes_the_v2_folder(tmp_path, monkeypatch, capsys):
    p = tmp_path / VERSIONS["ft2"].selection
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"adapter_sha256": "cd" * 32}), encoding="utf-8")
    monkeypatch.setattr(gguf, "hf_snapshot", lambda i, r: tmp_path / "base")
    seen = {}

    def fake(base, adapter, out, *, expected_sha256):
        seen.update(adapter=adapter, out=out, sha=expected_sha256)
        return {"lora_layers_merged": 205}

    monkeypatch.setattr(merge, "merge", fake)
    assert cli.main(["merge", "--model", "ft2", "--root", str(tmp_path)]) == 0
    assert seen == {"adapter": tmp_path / "artifacts/adapter_v2",
                    "out": tmp_path / "artifacts/merged/ft2-bf16", "sha": "cd" * 32}  # fmt: skip
    assert "ft2-bf16" in capsys.readouterr().out


def _merged_v2(root, adapter_sha):
    p = root / VERSIONS["ft2"].selection
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"adapter_sha256": "cd" * 32}), encoding="utf-8")
    d = root / VERSIONS["ft2"].merged_dir
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "ggh_merge.json").write_text(json.dumps({"adapter_sha256": adapter_sha}), encoding="utf-8")


def test_export_ft2_converts_the_v2_merge_and_refuses_another_adapter(tmp_path, monkeypatch):
    _merged_v2(tmp_path, "cd" * 32)
    seen = {}

    def fake_export(root, hf, name, *, source_id, source_revision):
        seen.update(hf=hf, name=name, srev=source_revision)
        return {"files": {"ft2-Q4_K_M.gguf": {"bytes": 1, "sha256": "e" * 64}}}

    monkeypatch.setattr(gguf, "export", fake_export)
    assert cli.main(["export", "--model", "ft2", "--root", str(tmp_path)]) == 0
    assert seen == {"hf": tmp_path / "artifacts/merged/ft2-bf16", "name": "ft2",
                    "srev": "cd" * 32}  # fmt: skip
    (tmp_path / VERSIONS["ft2"].merged_dir / "ggh_merge.json").write_text(
        json.dumps({"adapter_sha256": "ab" * 32}), encoding="utf-8"
    )
    assert cli.main(["export", "--model", "ft2", "--root", str(tmp_path)]) == cli.EXIT_ERROR


def test_the_ft2_gguf_manifest_copy_goes_to_the_v2_folder(tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(llama, "source", lambda cache=None: tmp_path / "src")
    monkeypatch.setattr(llama, "tool", lambda name, cache=None: tmp_path / name)

    def run(cmd, env):  # each step writes the file it names
        out = Path(cmd[cmd.index("--outfile") + 1]) if "--outfile" in cmd else Path(cmd[2])
        out.write_bytes(out.name.encode())

    m = gguf.export(tmp_path, hf, "ft2", source_id="b+adapter", source_revision="cd" * 32, run=run)
    copy = tmp_path / "docs/measurements/v2/gguf_ft2_manifest.json"
    assert json.loads(copy.read_text(encoding="utf-8")) == m
    assert not (tmp_path / "docs/measurements/gguf_ft2_manifest.json").exists()
    assert set(m["files"]) == {"ft2-f16.gguf", "ft2-Q4_K_M.gguf", "ft2-Q8_0.gguf"}


# ---- ggh freeze --model ft2 ----


def _repo(tmp_path):
    root = tmp_path / "repo"
    m = root / data.MANIFEST_PATH
    m.parent.mkdir(parents=True)
    m.write_text('{"x": 1}\n', encoding="utf-8")
    (root / "configs").mkdir()
    for name in ("recipe.yaml", "recipe_v2.yaml"):
        (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in sel.ADAPTER_FILES:
        (adapter / name).write_bytes(name.encode())
    metrics = {"n": 2, "schema_valid": 1.0, "schema_valid_unconstrained": 1.0,
               "citation_valid": 1.0, "has_citation": 1.0,
               "label_accuracy": {"yes": 1.0, "no": 1.0}, "mean_words": 5.0,
               "over_80_words": 0.0}  # fmt: skip
    selection = {"selected_step": 340, "adapter_sha256": sel.adapter_hash(adapter),
                 "tried": [{"step": 340, "eval_loss": 0.66, "passed": True, "failed": [],
                            "metrics": metrics}], "n_prompts": 2}  # fmt: skip
    report = {"config_sha256": devrun.sha256_file(root / "configs/recipe_v2.yaml"),
              "gpu": "Tesla T4", "seconds": 4551.0, "peak_vram_gb": 9.8, "steps": 340,
              "eval_losses": [[300, 0.661], [340, 0.6608]]}  # fmt: skip
    gens = tmp_path / "generations.jsonl"
    rows = [{"pmid": str(i), "label": "yes", "step": 340, "constrained": "{}",
             "unconstrained": "{}"} for i in range(30)]  # fmt: skip
    gens.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log = tmp_path / "kernel.log"
    log.write_text('[{"time": 1.0, "data": "$ ggh train\\n"}]', encoding="utf-8")
    files = {}
    for name, obj in (("rep.json", report), ("sel.json", selection)):
        files[name] = tmp_path / name
        files[name].write_text(json.dumps(obj), encoding="utf-8")
    return root, adapter, files, gens, log


def _freeze_args(root, adapter, files, *extra):
    return ["freeze", "--model", "ft2", "--report", str(files["rep.json"]), "--selection",
            str(files["sel.json"]), "--adapter", str(adapter), "--kernel-version", "1",
            "--repo-commit", "a" * 40, "--root", str(root), *extra]  # fmt: skip


def test_freeze_ft2_writes_the_v2_records_and_leaves_v1_alone(tmp_path, capsys):
    import gzip

    root, adapter, files, gens, log = _repo(tmp_path)
    args = _freeze_args(root, adapter, files, "--generations", str(gens), "--log", str(log))
    assert cli.main(args) == 0
    v2 = root / "docs/measurements/v2"
    run = json.loads((v2 / "train_run.json").read_text(encoding="utf-8"))
    assert run["selected_step"] == 340 and run["repo_commit"] == "a" * 40
    assert not (root / devrun.TRAIN_RUN).exists()  # v1's file is never written
    sel_text = (v2 / "train_selection.json").read_text(encoding="utf-8")
    assert json.loads(sel_text) == json.loads(files["sel.json"].read_text(encoding="utf-8"))
    every = gzip.decompress((v2 / "dev_generations.jsonl.gz").read_bytes()).decode()
    assert every == gens.read_text(encoding="utf-8")
    assert (v2 / "dev_generations.jsonl.gz").read_bytes()[4:8] == b"\0\0\0\0"  # no timestamp
    sample = (v2 / "dev_generations_sample.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["pmid"] for x in sample] == ["0", "25"]
    assert gzip.decompress((v2 / "train_full_log.json.gz").read_bytes()) == log.read_bytes()
    assert "docs/measurements/v2/train_run.json" in capsys.readouterr().out


def test_freeze_ft2_refuses_a_run_from_another_recipe_or_adapter(tmp_path, capsys):
    root, adapter, files, _, _ = _repo(tmp_path)
    rep = json.loads(files["rep.json"].read_text(encoding="utf-8"))
    files["rep.json"].write_text(json.dumps({**rep, "config_sha256": "f" * 64}), encoding="utf-8")
    assert cli.main(_freeze_args(root, adapter, files)) == cli.EXIT_ERROR
    assert "recipe_v2.yaml" in capsys.readouterr().out
    files["rep.json"].write_text(json.dumps(rep), encoding="utf-8")
    (adapter / "adapter_config.json").write_bytes(b"changed")
    assert cli.main(_freeze_args(root, adapter, files)) == cli.EXIT_ERROR
    assert "differs from the recorded" in capsys.readouterr().out
    assert not (root / VERSIONS["ft2"].train_run).exists()


def test_freeze_out_overrides_the_folder_and_generations_must_be_the_selected_step(tmp_path):
    root, adapter, files, gens, _ = _repo(tmp_path)
    assert cli.main(_freeze_args(root, adapter, files, "--out", "elsewhere")) == 0
    assert (root / "elsewhere/train_run.json").is_file()
    gens.write_text(gens.read_text(encoding="utf-8").replace('"step": 340', '"step": 300', 1),
                    encoding="utf-8")  # fmt: skip
    assert cli.main(_freeze_args(root, adapter, files, "--generations", str(gens))) == 2


def test_serve_hf_serves_v2_bf16_under_the_judged_parity_config_model_id():
    from ggh.evals.config import load_run_config
    from ggh.serve import hf

    assert hf.MODEL_IDS == {"ft": hf.MODEL_ID, "ft2": "gemma-e2b-grounded-v2-bf16"}
    parity = load_run_config(ROOT / "configs/evalkit_v2_parity.yaml")
    assert list(parity.models) == [hf.MODEL_IDS["ft2"]]
    assert parity.endpoints[hf.MODEL_IDS["ft2"]].endswith(":8084/v1")
    parser = cli.build_parser()
    args = parser.parse_args(["serve-hf", "--model", "ft2", "--port", "8084"])
    assert args.model == "ft2" and args.port == 8084
    assert parser.parse_args(["serve-hf"]).model == "ft"  # v1 unchanged
