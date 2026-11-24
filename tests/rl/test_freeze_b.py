"""RBX-01..03 (V3.6): v3b's main run frozen, the checkpoint the v3b plan selects, its export and
parity. The adapter, merged weights and GGUFs are not in git; their hashes are."""

import copy
import json
from pathlib import Path

import pytest

from ggh.rl.freeze import FreezeError, RLRun, freeze_rl
from ggh.rl.select import select_checkpoint
from ggh.versions import VERSIONS, version

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "docs/measurements/v3b/main"
V3B = ROOT / "docs/measurements/v3b"
V2_ADAPTER = "e5f97861693346623bf55305c0b911e10fe9a9dcc0985e535db30dad74f4e3db"
RULE = {"rows": 300, "no_slack": 0.10, "copy_slack": 0.10}  # configs/eval_plan_v3b.yaml


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_v3b_plan_selects_checkpoint_250_from_the_committed_main_run():
    rl = _load(MAIN / "rl_dev.json")
    sel = select_checkpoint(rl, **RULE)
    assert sel.selected == "checkpoint-250" and sel.beats_v2
    assert rl["run_steps"] == 300 and rl["dev_rows"] == 300 and len(rl["adapters"]) == 14
    a = rl["adapters"]
    assert (a["v2"]["mean_reward"], a["checkpoint-250"]["mean_reward"]) == (0.67, 0.885)
    tied = [r["name"] for r in sel.table if r["mean_reward"] == 0.885]
    assert tied == ["checkpoint-250", "checkpoint-275", "checkpoint-300", "final"]  # earliest wins
    broken = {r["name"]: [k for k, ok in r["guards"].items() if not ok] for r in sel.table}
    assert {n: g for n, g in broken.items() if g} == {
        "checkpoint-150": ["label_accuracy_at_least_v2_minus_0.02"]}  # fmt: skip
    rec = _load(V3B / "train_selection.json")
    assert rec["selected"] == "checkpoint-250" and rec["table"] == sel.table and rec["rule"] == RULE


def test_the_v3b_summaries_are_the_held_out_dev_rows():
    # every summary was recomputed from these replies with the real checker (all 14 matched,
    # RUN-STATE); CI has neither the checker weights nor dev_eval.jsonl, so it checks files and rows
    from ggh.rl.grpo_b import dev_rows, tuning_ids

    rl = _load(MAIN / "rl_dev.json")
    held = tuning_ids(ROOT)
    dev = ROOT / "artifacts/data/dev_eval.jsonl"
    for name, summary in rl["adapters"].items():
        assert _load(MAIN / f"dev_{name}.json") == summary
        lines = (MAIN / f"dev_{name}.jsonl").read_text(encoding="utf-8").splitlines()
        replies = [json.loads(x) for x in lines]
        assert len(replies) == 300 and not {r["pmid"] for r in replies} & held
        if dev.is_file():  # built by `ggh data`, not in CI
            assert [r["pmid"] for r in replies] == [r["pmid"] for r in dev_rows(ROOT, 300)]


def test_the_v3b_run_record_matches_the_run_and_the_selection():
    import yaml

    run = RLRun.model_validate(_load(V3B / "rl_run.json"))
    report = _load(MAIN / "report.json")
    plan = yaml.safe_load((ROOT / "configs/eval_plan_v3b.yaml").read_text("utf-8"))
    assert run.reward == report["reward"] == "v3b"
    assert run.selected == "checkpoint-250" and run.selected_step == 250 and run.steps == 300
    assert run.config_hash == report["config_hash"] == plan["pinned"]["recipe"]["sha256"]
    assert run.init_adapter_sha256 == V2_ADAPTER == report["init_adapter_sha256"]
    assert run.adapter_sha256 == _load(V3B / "train_selection.json")["adapter_sha256"]
    assert run.dev_rows == 300 and run.dev_metrics == _load(MAIN / "rl_dev.json")["adapters"][
        "checkpoint-250"]  # fmt: skip
    assert run.repo_commit.startswith("a67891b")  # the commit both v3b main notebooks pin
    state = _load(MAIN / "trainer_state_checkpoint-250.json")
    assert state["global_step"] == 250


def test_each_freeze_follows_its_own_plans_rule():
    from ggh.cli_rl import SELECTION_RULES

    assert SELECTION_RULES["ft3b"] == RULE
    assert SELECTION_RULES["ft3"] == {"rows": 500, "no_slack": 0.05, "copy_slack": None}
    text = (ROOT / "configs/eval_plan_v3b.yaml").read_text("utf-8")
    for phrase in ("300\n  held-out dev rows", "v2's minus 0.10", "at most v2's plus 0.10"):
        assert phrase in text, phrase
    with pytest.raises(ValueError, match="500 dev rows"):  # v3's rule cannot read v3b's dev check
        select_checkpoint(_load(MAIN / "rl_dev.json"), **SELECTION_RULES["ft3"])


def _fake_run(tmp_path: Path, name: str):
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    recipe = ROOT / "configs/recipe_v3b_grpo.yaml"
    (tmp_path / "configs/recipe_v3b_grpo.yaml").write_bytes(recipe.read_bytes())
    ck = tmp_path / "dl" / name
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (ck / "adapter_model.safetensors").write_bytes(b"weights")
    return ck


def _freeze(tmp_path, ck, rl):
    return freeze_rl(
        tmp_path, rl, _load(MAIN / "report.json"), ck,
        recipe=Path("configs/recipe_v3b_grpo.yaml"), init_adapter_sha256=V2_ADAPTER,
        adapter_dir=Path("artifacts/adapter_v3b"), measurements=Path("m"),
        kernel_version="ggh-v3b-grpo v1", score_kernel_version="ggh-v3b-grpo-score v1",
        repo_commit="a67891bbb351c45b513f4fac31d8a982d461af21", gpu="Tesla T4", **RULE,
    )  # fmt: skip


def test_the_copy_guard_moves_the_freeze_off_a_copying_checkpoint(tmp_path):
    rl = copy.deepcopy(_load(MAIN / "rl_dev.json"))
    for name in ("checkpoint-250", "checkpoint-275", "checkpoint-300", "final"):
        rl["adapters"][name]["copied"] = 0.19  # v2's 0.0833 + 0.10 is the bound
    assert select_checkpoint(rl, 300, 0.10, None).selected == "checkpoint-250"  # unguarded
    with pytest.raises(FreezeError, match="selects checkpoint-225"):
        _freeze(tmp_path, _fake_run(tmp_path, "checkpoint-250"), rl)
    run = _freeze(tmp_path / "ok", _fake_run(tmp_path / "ok", "checkpoint-225"), rl)
    assert run.selected == "checkpoint-225" and run.dev_rows == 300


def test_v3b_has_its_own_paths_and_port():
    from ggh.export.parity import paths, q4_name
    from ggh.serve.llama import PORTS

    v = version("ft3b")
    others = [VERSIONS[m] for m in ("ft", "ft2", "ft3")]
    for attr in ("measurements", "adapter_dir", "merged_dir", "recipe"):
        assert getattr(v, attr) not in [getattr(o, attr) for o in others]
    assert PORTS["ft3b"] not in {p for m, p in PORTS.items() if m != "ft3b"} | {8082, 8084}
    assert paths("ft3b")[0] == Path("docs/measurements/v3b/parity.json")
    assert q4_name("ft3b") == "gemma-e2b-ft3b-q4"


def test_the_v3b_export_is_built_from_the_frozen_adapter():
    sha = _load(V3B / "train_selection.json")["adapter_sha256"]
    assert sha == _load(V3B / "rl_run.json")["adapter_sha256"]
    merge = _load(V3B / "merge_record.json")
    assert merge["adapter_sha256"] == sha and merge["lora_layers_merged"] == 205  # as v1 to v3
    gguf = _load(V3B / "gguf_ft3b_manifest.json")
    assert gguf["source"] == {"id": "google/gemma-4-E2B-it+adapter", "revision": sha}
    sizes = {k: f["bytes"] for k, f in gguf["files"].items()}
    v3 = {
        k.replace("ft3", "ft3b"): f["bytes"]
        for k, f in _load(ROOT / "docs/measurements/v3/gguf_ft3_manifest.json")["files"].items()
    }
    assert sizes == v3  # one architecture and tensor list
    local = ROOT / "artifacts/adapter_v3b"
    if local.is_dir():  # not in CI
        from ggh.train.select import adapter_hash

        assert adapter_hash(local) == sha


def _parity_rows():
    import gzip

    raw = gzip.decompress((V3B / "parity_generations.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def test_v3b_parity_is_recorded_under_x31_bars():
    import collections

    from ggh.export import parity
    from ggh.schemas.release import ParityReport

    rep = ParityReport.model_validate_json((V3B / "parity.json").read_text(encoding="utf-8"))
    rows = _parity_rows()
    assert rep.n == len(rows) == 500
    assert parity.build_report([r["bf16"] for r in rows], [r["q4_k_m"] for r in rows]) == rep
    assert parity.failures(rep) == ["label agreement 0.924 < 0.95"]  # a miss, as v2 and v3
    v3 = gzip_rows(ROOT / "docs/measurements/v3/parity_generations.jsonl.gz")
    assert [r["pmid"] for r in rows] == [r["pmid"] for r in v3]  # the same 500 dev prompts
    moves = collections.Counter(
        f"{json.loads(r['bf16'])['label']}->{json.loads(r['q4_k_m'])['label']}" for r in rows
    )
    assert {k: n for k, n in moves.items() if k[: k.index("-")] != k[k.index(">") + 1 :]} == {
        "yes->no": 34, "no->yes": 3, "no->maybe": 1}  # fmt: skip


def gzip_rows(path: Path):
    import gzip

    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
            if x]  # fmt: skip


def test_v3b_parity_sample_is_every_25th_row():
    rows = _parity_rows()
    sample = [json.loads(x) for x in (V3B / "parity_sample.jsonl").read_text(encoding="utf-8")
              .splitlines() if x]  # fmt: skip
    assert sample == rows[::25] and len(sample) == 20
    agree = sum(json.loads(r["bf16"])["label"] == json.loads(r["q4_k_m"])["label"] for r in sample)
    same = sum(json.loads(r["bf16"])["answer"] == json.loads(r["q4_k_m"])["answer"] for r in sample)
    run_state = (ROOT / "specs/V3.6-v3b-export/RUN-STATE.md").read_text(encoding="utf-8")
    assert f"labels agree on {agree}/20" in run_state and f"identical on {same}/20" in run_state


def test_v3b_gpu_hours_are_logged_from_the_session_logs():
    import gzip

    rows = [json.loads(x) for x in (ROOT / "docs/measurements/gpu-ledger.jsonl").read_text(
        encoding="utf-8").splitlines() if x]  # fmt: skip
    mine = {r["kernel"]: r for r in rows if r["node"] == "V3.6"}
    assert set(mine) == {"ggh-v3b-grpo v1", "ggh-v3b-grpo-score v1"}
    for kernel, log in (("ggh-v3b-grpo v1", "train_log"), ("ggh-v3b-grpo-score v1", "score_log")):
        events = json.loads(gzip.decompress((MAIN / f"{log}.json.gz").read_bytes()))
        assert mine[kernel]["gpu_hours"] == round(events[-1]["time"] / 3600, 2)
