"""EXP-05 on the committed v2 (`ft2`, V2.1) evidence: `ggh export-records --model ft2` wrote
export_artifacts_v2.json and the lineage copies train_v2, export_ft2, parity_v2 and targets_v2
from the files on disk; every hash agrees with what V2.1 recorded when it built v2."""

import json
from pathlib import Path

from ggh.export import artifacts as A
from ggh.export import parity
from ggh.lineage import StageManifest
from ggh.schemas import ExportArtifact, ParityReport
from ggh.serve import llama
from ggh.versions import version

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "docs/measurements"
V2 = version("ft2")


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def stage(name) -> StageManifest:
    return StageManifest.model_validate_json(
        (M / "lineage" / f"{name}.json").read_text(encoding="utf-8")
    )  # validation recomputes the key from stage, inputs, params and seed


def test_every_v2_export_output_has_a_record_with_the_producing_stage_key():
    recs = [ExportArtifact.model_validate(x) for x in load(A.RECORDS_V2)]
    train, base, ft = stage("train_v2"), stage("export_base"), stage("export_ft2")
    assert len(recs) == 8
    for r in recs:
        want = {"adapter": train, "merged_bf16": ft}.get(r.kind) or {"base": base, "ft": ft}[
            r.model
        ]
        assert r.parent_key == want.key
        if r.kind == "gguf":
            assert want.outputs[r.path] == r.sha256 and r.llama_cpp_commit == llama.TAG
    paths = {r.path for r in recs if r.model == "ft"}
    assert paths == {
        "artifacts/adapter_v2",
        "artifacts/merged/ft2-bf16/model.safetensors",
        *(f"artifacts/gguf/ft2-{q}.gguf" for q in ("f16", "Q4_K_M", "Q8_0")),
    }
    ggufs = {r.path.rsplit("/", 1)[1]: r.sha256 for r in recs if r.kind == "gguf"}
    for name in ("gguf_base_manifest.json", "v2/gguf_ft2_manifest.json"):
        for f, v in load(M / name)["files"].items():
            assert ggufs[f] == v["sha256"]


def test_the_v2_adapter_is_the_frozen_one_the_merge_and_ggufs_were_built_from():
    frozen = load(V2.selection)["adapter_sha256"]
    assert load(A.MERGE_RECORD_V2)["adapter_sha256"] == frozen
    assert load(A.GGUF_RECORD_V2)["source"]["revision"] == frozen
    train = stage("train_v2")
    assert train.outputs["docs/measurements/v2/train_run.json"]
    assert train.params["selected_step"] == load(V2.train_run)["selected_step"] == 340
    assert train.params["kaggle_kernel"] == "ggh-v21-train v1" and train.host == "kaggle"
    ledger = [json.loads(x) for x in (M / "gpu-ledger.jsonl").read_text(encoding="utf-8")
              .splitlines() if x]  # fmt: skip
    (row,) = [r for r in ledger if r.get("kernel") == "ggh-v21-train v1" and r["kind"] == "train"]
    assert train.finished_at.isoformat() == row["at"]


def test_the_v2_chain_runs_parity_to_export_to_train_to_targets_to_data():
    data, targets, train = stage("data"), stage("targets_v2"), stage("train_v2")
    ft, par = stage("export_ft2"), stage("parity_v2")
    assert par.parents == (ft.key, data.key) and ft.parents == (train.key,)
    assert train.parents == (targets.key,) and targets.parents == (data.key,)
    for child, parent in ((targets, data), (train, targets), (ft, train), (par, ft), (par, data)):
        shared = set(child.inputs) & set(parent.outputs)
        assert shared and all(child.inputs[k] == parent.outputs[k] for k in shared)
    assert "configs/recipe_v2.yaml" in train.inputs
    assert targets.key != stage("targets").key  # v2's targets are not v1's
    assert all(m.host in ("local", "kaggle") for m in (data, targets, train, ft, par))


def test_the_v2_parity_record_is_the_one_lineage_names_and_misses_the_bar_as_disclosed():
    par = stage("parity_v2")
    assert set(par.outputs) == {p.as_posix() for p in parity.paths("ft2")}
    rep = ParityReport.model_validate(load(parity.paths("ft2")[0]))
    # OWNER D-55: 0.92 label agreement is under the 0.95 bar, disclosed and kept as measured
    assert rep.label_agreement == 0.92 and not parity.passes(rep)
    assert rep.schema_valid == {"bf16": 1.0, "q4_k_m": 1.0}


def test_v2_template_parity_holds_on_the_served_q4_with_no_thought_channel():
    """VEV-01's template clause: the V2.1 smoke on the GGUF the main run served."""
    checks = json.loads((ROOT / "docs/measurements/serving_checks_ft2.json").read_text("utf-8"))
    manifest = json.loads(
        (ROOT / "docs/measurements/v2/gguf_ft2_manifest.json").read_text(encoding="utf-8")
    )
    assert checks["passed"] is True and checks["gguf"] == "ft2-Q4_K_M.gguf"
    assert checks["gguf_sha256"] in json.dumps(manifest)
    assert len(checks["parity"]) == 20 and all(p["equal"] for p in checks["parity"])
    args = checks["server_args"]
    assert args[args.index("--reasoning") + 1] == "off"
    assert args[args.index("--reasoning-budget") + 1] == "0"
