"""V2.1 task 6: the recorded v2 Kaggle run (`ggh-v21-train` v1, recipe_v2), its selection,
the frozen `TrainRun` in docs/measurements/v2, the merge record, the ft2 GGUF manifest and
the GPU ledger. These check the committed evidence (CI-safe); the local adapter, merge and
GGUFs are checked only where they exist."""

import gzip
import json
import math
from pathlib import Path

from ggh.config import load_recipe
from ggh.data.build import MANIFEST_PATH
from ggh.schemas import TrainRun
from ggh.serve import llama
from ggh.train.devrun import sha256_file
from ggh.train.notebook import render
from ggh.train.run import config_hash
from ggh.train.select import BAR, adapter_hash, failed_bars
from ggh.versions import VERSIONS

ROOT = Path(__file__).resolve().parents[2]
V2 = VERSIONS["ft2"]
M = ROOT / V2.measurements
REPORT = json.loads((M / "train_full.json").read_text(encoding="utf-8"))
RUN = TrainRun.model_validate_json((ROOT / V2.train_run).read_text(encoding="utf-8"))
SEL = json.loads((ROOT / V2.selection).read_text(encoding="utf-8"))
LOG = json.loads(gzip.decompress((M / "train_full_log.json.gz").read_bytes()))
LOG_TEXT = "".join(x["data"] for x in LOG)
ROWS = [json.loads(x) for x in gzip.decompress((M / "dev_generations.jsonl.gz").read_bytes())
        .decode("utf-8").splitlines() if x]  # fmt: skip
SAMPLE = [json.loads(x) for x in (M / "dev_generations_sample.jsonl")
          .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
COMMIT = "8ff61b977580700c592ee2c8443d46570e0126f1"
ADAPTER_SHA = "e5f97861693346623bf55305c0b911e10fe9a9dcc0985e535db30dad74f4e3db"
RUN_STATE = (ROOT / "specs/V2.1-retrain/RUN-STATE.md").read_text(encoding="utf-8")


def test_one_session_ran_the_committed_v2_notebook_at_the_recorded_commit():
    assert f"$ git checkout -q {COMMIT}\n" in LOG_TEXT and RUN.repo_commit == COMMIT
    assert RUN.kaggle_kernel_version == "1" and "Traceback" not in LOG_TEXT
    assert (LOG[-1]["time"] - LOG[0]["time"]) / 3600 < 12
    assert REPORT["seconds"] == RUN.wall_clock_s < LOG[-1]["time"]
    nb_text = (ROOT / "notebooks/v2/train_kaggle.ipynb").read_text(encoding="utf-8")
    evalkit = nb_text.split("pip install -q --no-deps '")[1].split("'")[0]  # pre-D-56 pin
    nb = render(COMMIT, "unsloth", None, V2.recipe.as_posix(), evalkit=evalkit)
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ran = [x["data"][2:].rstrip("\n") for x in LOG if x["data"].startswith("$ ")]
    assert ran[-2:] == ["ggh train --config configs/recipe_v2.yaml --out /kaggle/working/run "
                        "--tool unsloth", "ggh dev-eval --run /kaggle/working/run --out "
                        "/kaggle/working/dev --config configs/recipe_v2.yaml"]  # fmt: skip
    assert len(ran) == 12 and all(cmd in code for cmd in ran)


def test_two_epochs_on_the_pinned_base_with_the_v2_recipe_and_v1_data():
    cfg = load_recipe(ROOT / V2.recipe)
    assert REPORT["epochs"] == cfg.epochs == 2 and REPORT["max_steps"] is None
    assert REPORT["steps"] == RUN.steps == 2 * -(-REPORT["train_examples"] // 16) == 340
    assert REPORT["base_loaded"] == [cfg.base_model, cfg.base_revision]
    assert "unsloth/gemma" not in LOG_TEXT and "bnb-4bit" not in LOG_TEXT
    assert REPORT["config_sha256"] == RUN.config_hash == config_hash(ROOT / V2.recipe)
    assert RUN.data_manifest_hash == sha256_file(ROOT / MANIFEST_PATH)
    assert REPORT["non_finite"] == REPORT["non_finite_steps"] == 0
    assert all(math.isfinite(v) for _, v in REPORT["losses"])
    assert [list(p) for p in RUN.val_loss_curve] == REPORT["eval_losses"]


def test_the_lowest_validation_loss_was_tried_first_and_passed_every_bar():
    evals = dict(REPORT["eval_losses"])
    assert RUN.selected_step == min(evals, key=lambda s: (evals[s], s)) == 340
    assert round(evals[340], 4) == 0.6608
    m = RUN.dev_metrics
    assert m.n == 500 and failed_bars(m) == [] and m.over_80_words == 0.0
    assert min(m.schema_valid, m.schema_valid_unconstrained, m.has_citation,
               m.citation_valid) >= BAR  # fmt: skip
    assert SEL["selected_step"] == 340 and SEL["adapter_sha256"] == ADAPTER_SHA
    assert SEL["tried"][0]["metrics"] == m.model_dump(mode="json")
    assert ADAPTER_SHA[:12] in RUN_STATE
    assert f"yes {m.label_accuracy['yes']:.3f}" in RUN_STATE
    assert f"no {m.label_accuracy['no']:.3f}" in RUN_STATE


def test_the_500_generations_reproduce_the_labels_and_the_sample_is_every_25th():
    assert len(ROWS) == 500 and {r["step"] for r in ROWS} == {340}
    for lab in ("yes", "no"):
        mine = [r for r in ROWS if r["label"] == lab]
        hit = sum(json.loads(r["constrained"])["label"] == lab for r in mine) / len(mine)
        assert hit == RUN.dev_metrics.label_accuracy[lab]
    assert all(next(iter(json.loads(r["constrained"]))) == "label" for r in ROWS)
    assert ROWS[::25] == SAMPLE and len(SAMPLE) == 20


def test_trainrun_round_trips_and_v1s_record_is_untouched():
    assert RUN.model_dump_json(indent=2) + "\n" == (ROOT / V2.train_run).read_text("utf-8")
    v1 = TrainRun.model_validate_json((ROOT / VERSIONS["ft"].train_run).read_text("utf-8"))
    assert v1.selected_step == 527 and v1.repo_commit != RUN.repo_commit


def test_merge_record_and_ft2_gguf_manifest_name_the_frozen_adapter():
    rec = json.loads((M / "merge_record.json").read_text(encoding="utf-8"))
    assert rec["adapter_sha256"] == ADAPTER_SHA and rec["lora_layers_merged"] == 205
    v1 = json.loads((ROOT / "docs/measurements/merge_record.json").read_text(encoding="utf-8"))
    assert rec["restored_from_base"] == v1["restored_from_base"] and len(v1["carried"]) == 5
    man = json.loads((M / "gguf_ft2_manifest.json").read_text(encoding="utf-8"))
    assert man["name"] == "ft2" and man["source"]["revision"] == ADAPTER_SHA
    assert man["llama_cpp"]["tag"] == llama.TAG
    assert set(man["files"]) == {"ft2-f16.gguf", *(f"ft2-{q}.gguf" for q in llama.QUANTS)}
    base = json.loads((ROOT / "docs/measurements/gguf_ft_manifest.json").read_text("utf-8"))
    assert man["steps"] == base["steps"]
    local = ROOT / llama.GGUF_DIR / "ft2.manifest.json"  # gitignored; where it was built
    if local.exists():
        assert json.loads(local.read_text(encoding="utf-8")) == man


def test_the_local_adapter_and_merge_are_the_frozen_ones():
    adapter = ROOT / V2.adapter_dir  # gitignored; present where the run was downloaded
    if adapter.exists():
        assert adapter_hash(adapter) == ADAPTER_SHA
    merged = ROOT / V2.merged_dir / "ggh_merge.json"
    if merged.exists():
        assert merged.read_bytes() == (M / "merge_record.json").read_bytes()


def test_the_ledger_holds_the_v2_session_no_less_than_its_log_span():
    rows = [json.loads(x) for x in (ROOT / "docs/measurements/gpu-ledger.jsonl")
            .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
    (row,) = [r for r in rows if r.get("kernel") == "ggh-v21-train v1"]
    assert row["node"] == "V2.1" and row["kind"] == "train"
    assert row["gpu_hours"] >= math.ceil(LOG[-1]["time"] / 36) / 100
    assert f"{row['gpu_hours']:.2f} h" in RUN_STATE
