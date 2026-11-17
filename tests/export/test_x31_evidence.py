"""EXP-01 to EXP-06 on the committed X3.1 evidence: the merge record, the two GGUF manifests,
the fine-tune's serving checks, the 500-prompt parity run, the lineage copies and the sample."""

import gzip
import json
import re
from pathlib import Path

from ggh.export import parity
from ggh.lineage import StageManifest
from ggh.schemas import ExportArtifact, ParityReport
from ggh.serve import llama

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "docs/measurements"
RUN_STATE = (ROOT / "specs/X3.1-export/RUN-STATE.md").read_text(encoding="utf-8")


def load(name):
    return json.loads((M / name).read_text(encoding="utf-8"))


def stage(name) -> StageManifest:
    return StageManifest.model_validate_json(
        (M / "lineage" / f"{name}.json").read_text(encoding="utf-8")
    )  # validation recomputes the key from stage, inputs, params and seed


# EXP-01


def test_the_merge_used_the_frozen_adapter_in_bf16_on_cpu():
    rec = load("merge_record.json")
    assert rec["adapter_sha256"] == load("train_selection.json")["adapter_sha256"]
    assert rec["dtype"] == "bfloat16" and rec["device"] == "cpu"
    assert rec["lora_layers_merged"] == 205
    assert {"tokenizer.json", "chat_template.jinja", "tokenizer_config.json"} <= set(rec["carried"])


def test_only_the_kv_shared_layers_untrained_tensors_were_restored():
    restored = load("merge_record.json")["restored_from_base"]
    pat = re.compile(
        r"model\.language_model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm)\.weight"
    )
    layers = {int(m.group(1)) for m in map(pat.fullmatch, restored) if m}
    assert len(restored) == 60 and all(pat.fullmatch(k) for k in restored)
    assert layers == set(range(15, 35))  # 20 layers x 3 tensors


# EXP-02


def test_base_and_ft_ggufs_share_the_release_steps_and_tool():
    base, ft = load("gguf_base_manifest.json"), load("gguf_ft_manifest.json")
    for k in ("llama_cpp", "steps"):
        assert base[k] == ft[k]
    assert ft["llama_cpp"]["tag"] == llama.TAG and ft["name"] == "ft"
    assert ft["source"] == {"id": "google/gemma-4-E2B-it+adapter",
                            "revision": load("merge_record.json")["adapter_sha256"]}  # fmt: skip
    assert set(ft["files"]) == {"ft-f16.gguf", "ft-Q4_K_M.gguf", "ft-Q8_0.gguf"}
    for q in ("f16", "Q4_K_M", "Q8_0"):  # same tensor list and types: same byte size to within
        b, f = base["files"][f"base-{q}.gguf"]["bytes"], ft["files"][f"ft-{q}.gguf"]["bytes"]
        assert abs(b - f) < 20_000  # metadata (names, licence fields) only


# EXP-03


def test_template_parity_structured_output_and_reruns_hold_for_the_ft_gguf():
    rep = load("serving_checks_ft.json")
    assert rep["passed"] and rep["gguf"] == "ft-Q4_K_M.gguf"
    assert rep["gguf_sha256"] == load("gguf_ft_manifest.json")["files"]["ft-Q4_K_M.gguf"]["sha256"]
    s = rep["summary"]
    assert s["n"] == s["parity_equal"] == s["constrained_valid"] == s["identical_reruns"] == 20
    i = rep["server_args"].index("--reasoning-budget")
    assert rep["server_args"][i + 1] == "0"  # D-41


def test_d41_left_the_base_replies_unchanged():
    rep = load("serving_checks.json")  # the base, rerun under the D-41 arguments
    assert rep["passed"] and "--reasoning-budget" in rep["server_args"]
    assert "byte-identical to E3.1" in RUN_STATE and "20/20" in RUN_STATE


# EXP-04


def _gens():
    raw = gzip.decompress((M / "parity_generations.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def test_parity_on_500_dev_prompts_passes_the_bar_and_recomputes():
    rep = ParityReport.model_validate_json((M / "parity_report.json").read_text(encoding="utf-8"))
    rows = _gens()
    assert rep.n == len(rows) == 500 and parity.passes(rep)
    assert parity.build_report([r["bf16"] for r in rows], [r["q4_k_m"] for r in rows]) == rep
    dev = gzip.decompress((M / "dev_generations.jsonl.gz").read_bytes()).decode("utf-8")
    assert [r["pmid"] for r in rows] == [json.loads(x)["pmid"] for x in dev.splitlines() if x]


def test_no_reply_carries_a_thought_channel_or_the_reasoning_trace():
    for r in _gens():
        assert "<|channel>" not in r["q4_k_m"] and r["q4_k_m"].startswith('{"label": ')
        assert "Analyze the Request" not in r["q4_k_m"]


def test_the_recorded_agreement_numbers_are_in_run_state():
    rows = _gens()
    flips = sum(json.loads(r["bf16"])["label"] != json.loads(r["q4_k_m"])["label"] for r in rows)
    assert f"{flips} of 500" in RUN_STATE and "0.956" in RUN_STATE


# EXP-05


def test_every_export_output_has_a_record_with_the_producing_stage_key():
    recs = [ExportArtifact.model_validate(x) for x in load("export_artifacts.json")]
    train, base, ft = stage("train"), stage("export_base"), stage("export_ft")
    assert len(recs) == 8
    for r in recs:
        want = {"adapter": train, "merged_bf16": ft}.get(r.kind) or {"base": base, "ft": ft}[
            r.model
        ]
        assert r.parent_key == want.key
        if r.kind == "gguf":
            assert want.outputs[r.path] == r.sha256
    ggufs = {r.path.rsplit("/", 1)[1]: r.sha256 for r in recs if r.kind == "gguf"}
    for name in ("gguf_base_manifest.json", "gguf_ft_manifest.json"):
        for f, v in load(name)["files"].items():
            assert ggufs[f] == v["sha256"]


def test_the_chain_runs_parity_to_export_to_train_to_targets_to_data():
    data, targets, train = stage("data"), stage("targets"), stage("train")
    ft, par = stage("export_ft"), stage("parity")
    assert par.parents == (ft.key, data.key) and ft.parents == (train.key,)
    assert train.parents == (targets.key,) and targets.parents == (data.key,) and data.parents == ()
    for child, parent in ((targets, data), (train, targets), (ft, train), (par, ft), (par, data)):
        shared = set(child.inputs) & set(parent.outputs)
        assert shared and all(child.inputs[k] == parent.outputs[k] for k in shared)
    assert all(m.host in ("local", "kaggle") for m in (data, targets, train, ft, par))


# EXP-06


def test_the_sample_is_every_25th_row_and_its_reading_is_noted():
    rows = _gens()
    sample = [json.loads(x) for x in (M / "parity_sample.jsonl").read_text(encoding="utf-8")
              .splitlines() if x]  # fmt: skip
    assert sample == rows[::25] and len(sample) == 20
    agree = sum(json.loads(r["bf16"])["label"] == json.loads(r["q4_k_m"])["label"] for r in sample)
    same = sum(json.loads(r["bf16"])["answer"] == json.loads(r["q4_k_m"])["answer"] for r in sample)
    assert "Read the sample" in RUN_STATE
    assert f"labels agree on {agree}/20" in RUN_STATE and f"identical on {same}/20" in RUN_STATE
