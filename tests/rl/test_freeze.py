"""RLX-01..03 (V3.4): v3's main run frozen, the checkpoint the plan selects, its export and
parity. The adapter, merged weights and GGUFs are not in git; their hashes are."""

import copy
import json
from pathlib import Path

import pytest

from ggh.rl.freeze import FreezeError, RLRun, freeze_rl
from ggh.rl.select import select_checkpoint
from ggh.versions import VERSIONS, version

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "docs/measurements/v3/main"
V3 = ROOT / "docs/measurements/v3"
V2_ADAPTER = "e5f97861693346623bf55305c0b911e10fe9a9dcc0985e535db30dad74f4e3db"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _fake_run(tmp_path: Path, name: str = "checkpoint-550"):
    """A repo root holding the v3 recipe, and a checkpoint folder with the two adapter files."""
    (tmp_path / "configs").mkdir()
    recipe = ROOT / "configs/recipe_v3_grpo.yaml"
    (tmp_path / "configs/recipe_v3_grpo.yaml").write_bytes(recipe.read_bytes())
    ck = tmp_path / "dl" / name
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (ck / "adapter_model.safetensors").write_bytes(b"weights")
    (ck / "trainer_state.json").write_text("{}", encoding="utf-8")  # never copied
    return ck


def _freeze(tmp_path, ck, rl=None, report=None):
    return freeze_rl(
        tmp_path,
        rl or _load(MAIN / "rl_dev.json"),
        report or _load(MAIN / "report.json"),
        ck,
        recipe=Path("configs/recipe_v3_grpo.yaml"),
        init_adapter_sha256=V2_ADAPTER,
        adapter_dir=Path("artifacts/adapter_v3"),
        measurements=Path("m"),
        kernel_version="ggh-v3-grpo v1",
        score_kernel_version="ggh-v3-grpo-score v1",
        repo_commit="305bcac86e5c9add61909801dbd3535d88019a1f",
        gpu="Tesla T4",
    )


def test_the_plan_selects_checkpoint_550_from_the_committed_main_run():
    rl = _load(MAIN / "rl_dev.json")
    sel = select_checkpoint(rl)  # v3's plan: 500 rows, gold-"no" slack 0.05, no copy guard
    assert sel.selected == "checkpoint-550" and sel.beats_v2
    assert rl["run_steps"] == 678 and rl["dev_rows"] == 500 and len(rl["adapters"]) == 16
    assert all(r["keeps_guards"] for r in sel.table)
    rec = _load(V3 / "train_selection.json")
    assert rec["selected"] == "checkpoint-550" and rec["table"] == sel.table
    a = rl["adapters"]
    assert (a["v2"]["mean_reward"], a["checkpoint-550"]["mean_reward"]) == (1.318, 1.372)


def test_the_main_run_summaries_recompute_from_the_committed_replies():
    from ggh.rl.grpo import dev_summary

    rl = _load(MAIN / "rl_dev.json")
    dev = ROOT / "artifacts/data/dev_eval.jsonl"
    rows = None
    if dev.is_file():  # built by `ggh data`, not in CI
        rows = [json.loads(x) for x in dev.read_text(encoding="utf-8").splitlines()][:500]
    for name, summary in rl["adapters"].items():
        assert _load(MAIN / f"dev_{name}.json") == summary
        lines = (MAIN / f"dev_{name}.jsonl").read_text(encoding="utf-8").splitlines()
        replies = [json.loads(x) for x in lines]
        assert len(replies) == 500
        if rows is not None:
            assert [r["pmid"] for r in replies] == [r["pmid"] for r in rows]
            assert dev_summary([r["reply"] for r in replies], rows) == summary


def test_the_run_record_matches_the_run_and_the_selection():
    run = RLRun.model_validate(_load(V3 / "rl_run.json"))
    report = _load(MAIN / "report.json")
    assert run.selected == "checkpoint-550" and run.selected_step == 550 and run.steps == 678
    assert run.config_hash == report["config_hash"] == _plan_pin("recipe")
    assert run.init_adapter_sha256 == V2_ADAPTER == report["init_adapter_sha256"]
    assert run.adapter_sha256 == _load(V3 / "train_selection.json")["adapter_sha256"]
    assert run.dev_metrics == _load(MAIN / "rl_dev.json")["adapters"]["checkpoint-550"]
    assert run.repo_commit.startswith("305bcac")  # the commit the train notebook pins


def _plan_pin(key: str) -> str:
    import yaml

    return yaml.safe_load((ROOT / "configs/eval_plan_v3.yaml").read_text("utf-8"))["pinned"][key][
        "sha256"
    ]


def test_freeze_copies_only_the_adapter_files_and_hashes_them(tmp_path):
    from ggh.train.select import adapter_hash

    ck = _fake_run(tmp_path)
    run = _freeze(tmp_path, ck)
    dest = tmp_path / "artifacts/adapter_v3"
    assert sorted(p.name for p in dest.iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors"]  # fmt: skip
    assert run.adapter_sha256 == adapter_hash(dest)
    assert _freeze(tmp_path, ck).adapter_sha256 == run.adapter_sha256  # repeatable
    assert _load(tmp_path / "m/train_selection.json")["adapter_sha256"] == run.adapter_sha256


def test_freeze_refuses_anything_but_the_selected_checkpoint_from_v2(tmp_path):
    with pytest.raises(FreezeError, match="selects checkpoint-550"):
        _freeze(tmp_path, _fake_run(tmp_path, "checkpoint-600"))
    ck = tmp_path / "dl/checkpoint-550"
    ck.mkdir(parents=True)
    report = _load(MAIN / "report.json")
    for key, value, msg in (("init_adapter_sha256", "0" * 64, "v2 adapter"),
                            ("config_hash", "0" * 64, "config hash"),
                            ("steps", 600, "steps")):  # fmt: skip
        bad = dict(report, **{key: value})
        with pytest.raises(FreezeError, match=msg):
            _freeze_at(tmp_path, ck, bad)
    rl = copy.deepcopy(_load(MAIN / "rl_dev.json"))
    for name, c in rl["adapters"].items():
        if name != "v2":
            c["unparsed"] = 0.5  # every candidate breaks a guard
    with pytest.raises(FreezeError, match="nothing to freeze"):
        _freeze(tmp_path, ck, rl=rl)


def _freeze_at(tmp_path, ck, report):
    if not (tmp_path / "configs/recipe_v3_grpo.yaml").is_file():
        (tmp_path / "configs").mkdir(exist_ok=True)
        (tmp_path / "configs/recipe_v3_grpo.yaml").write_bytes(
            (ROOT / "configs/recipe_v3_grpo.yaml").read_bytes()
        )
    return _freeze(tmp_path, ck, report=report)


def test_v3_has_its_own_paths_and_port():
    from ggh.export.parity import paths, q4_name
    from ggh.serve.llama import PORTS

    v = version("ft3")
    others = [VERSIONS[m] for m in ("ft", "ft2")]
    for attr in ("measurements", "adapter_dir", "merged_dir", "recipe"):
        assert getattr(v, attr) not in [getattr(o, attr) for o in others]
    assert PORTS["ft3"] not in {p for m, p in PORTS.items() if m != "ft3"} | {8082, 8084}
    assert paths("ft3")[0] == Path("docs/measurements/v3/parity.json")
    assert q4_name("ft3") == "gemma-e2b-ft3-q4"


def test_the_export_is_built_from_the_frozen_adapter():
    sha = _load(V3 / "train_selection.json")["adapter_sha256"]
    merge = _load(V3 / "merge_record.json")
    assert merge["adapter_sha256"] == sha and merge["lora_layers_merged"] == 205  # as v1, v2
    gguf = _load(V3 / "gguf_ft3_manifest.json")
    assert gguf["source"] == {"id": "google/gemma-4-E2B-it+adapter", "revision": sha}
    sizes = {k: f["bytes"] for k, f in gguf["files"].items()}
    v2 = {
        k.replace("ft2", "ft3"): f["bytes"]
        for k, f in _load(ROOT / "docs/measurements/v2/gguf_ft2_manifest.json")["files"].items()
    }
    assert sizes == v2  # one architecture and tensor list
    local = ROOT / "artifacts/adapter_v3"
    if local.is_dir():  # not in CI
        from ggh.train.select import adapter_hash

        assert adapter_hash(local) == sha


def _parity_rows():
    import gzip

    raw = gzip.decompress((V3 / "parity_generations.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def test_v3_parity_is_recorded_under_x31_bars():
    from ggh.export import parity
    from ggh.schemas.release import ParityReport

    rep = ParityReport.model_validate_json((V3 / "parity.json").read_text(encoding="utf-8"))
    rows = _parity_rows()
    assert rep.n == len(rows) == 500
    assert parity.build_report([r["bf16"] for r in rows], [r["q4_k_m"] for r in rows]) == rep
    assert parity.failures(rep) == ["label agreement 0.936 < 0.95"]  # a miss, as v2's 0.920
    assert [r["pmid"] for r in rows] == [r["pmid"] for r in _parity_dev()]
    decisions = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    assert "| OWNER D-71 |" in decisions and "label agreement 0.936" in decisions  # owner ruled


def _parity_dev():
    lines = (MAIN / "dev_checkpoint-550.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]


def test_v3_parity_sample_is_every_25th_row():
    rows = _parity_rows()
    sample = [json.loads(x) for x in (V3 / "parity_sample.jsonl").read_text(encoding="utf-8")
              .splitlines() if x]  # fmt: skip
    assert sample == rows[::25] and len(sample) == 20
    agree = sum(json.loads(r["bf16"])["label"] == json.loads(r["q4_k_m"])["label"] for r in sample)
    same = sum(json.loads(r["bf16"])["answer"] == json.loads(r["q4_k_m"])["answer"] for r in sample)
    run_state = (ROOT / "specs/V3.4-v3-export/RUN-STATE.md").read_text(encoding="utf-8")
    assert f"labels agree on {agree}/20" in run_state and f"identical on {same}/20" in run_state
