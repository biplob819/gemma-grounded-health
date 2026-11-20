import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from ggh import schemas
from ggh.config import ConfigError, load_recipe
from ggh.schemas import UNPINNED

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "schemas"
FIXTURE_SCHEMAS = sorted(p.stem for p in FIX.glob("*.json"))


def load(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.contract
def test_one_fixture_per_schema_except_the_yaml_recipe():
    expected = {
        "TrainExample",
        "DropRecord",
        "DataManifest",
        "AuditLabel",
        "DevMetrics",
        "TrainRun",
        "ExportArtifact",
        "ParityReport",
        "LaptopBench",
        "BehaviourRow",
        "ResultsBundle",
        "SampleReadRow",
    }
    assert set(FIXTURE_SCHEMAS) == expected


@pytest.mark.contract
@pytest.mark.parametrize("name", FIXTURE_SCHEMAS)
def test_fixture_validates_and_round_trips_byte_identically(name):
    cls = getattr(schemas, name)
    obj = cls.model_validate(load(name))
    dumped = obj.model_dump_json()
    again = cls.model_validate_json(dumped)
    assert again == obj
    assert again.model_dump_json() == dumped


def test_fixtures_match_their_builder():
    """Canonical bytes, not ==, so True and 1 or 1 and 1.0 cannot stand in for each other."""
    from tests.schemas import build_fixtures

    for name, data in build_fixtures.FIXTURES.items():
        if data is not None:
            expected = json.dumps(data, indent=2, sort_keys=True) + "\n"
            assert (FIX / f"{name}.json").read_text(encoding="utf-8") == expected, name


def example(**changes) -> dict:
    d = load("TrainExample")
    d.update(changes)
    return d


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"target": {"label": "yes", "answer": "a", "citations": [3]}}, "outside 1..2"),
        ({"target": {"label": "yes", "answer": "a", "citations": [2, 1]}}, "sorted and unique"),
        ({"target": {"label": "yes", "answer": "a", "citations": [1, 1]}}, "sorted and unique"),
        ({"target": {"label": "yes", "answer": "a", "citations": []}}, "at least one citation"),
        ({"id": "pqa-a-999"}, "does not carry pmid"),
        ({"id": "pqa-a-21645374:insufficient"}, "implies kind 'insufficient'"),
        ({"id": "bad-21645374"}, "must be pqa-a-<pmid>"),
        ({"kind": "insufficient", "id": "pqa-a-21645374:insufficient"}, "must be labelled maybe"),
        ({"kind": "offscope", "id": "pqa-a-21645374:offscope"}, "must be labelled maybe"),
        ({"source_label": "no"}, "differs from source label"),
        ({"support": 1.5}, "less than or equal to 1"),
        ({"nli_entail": -0.1}, "greater than or equal to 0"),
        ({"support": math.nan}, "finite"),
        ({"context": []}, "at least one sentence"),
        ({"rule_ids": ["tgt-r1"]}, "pattern"),
        ({"id": "pqa-a-21645374\n"}, "must be pqa-a-<pmid>"),
        ({"pmid": "\u0662\u0661", "id": "pqa-a-\u0662\u0661"}, "pattern"),
        ({"support": True}, "valid number"),
        ({"nli_entail": "0.5"}, "valid number"),
    ],
)
def test_train_example_rejects(changes, message):
    with pytest.raises(ValidationError, match=message):
        schemas.TrainExample.model_validate(example(**changes))


def test_insufficient_and_offscope_examples_are_valid_with_maybe():
    for kind in ("insufficient", "offscope"):
        d = example(
            kind=kind,
            id=f"pqa-a-21645374:{kind}",
            target={"label": "maybe", "answer": "Not settled.", "citations": [1]},
        )
        assert schemas.TrainExample.model_validate(d).kind == kind


@pytest.mark.parametrize(
    "name, field, bad",
    [
        ("DataManifest", "template_hash", "abc"),
        ("DataManifest", "dataset_revision", "Z" * 40),
        ("DataManifest", "offscope_eval_hash", "A" * 64),  # upper-case hex is not our format
        ("AuditLabel", "verdict", "fine"),
        ("AuditLabel", "content_digest", "x"),
        ("DropRecord", "reason", "because"),
    ],
)
def test_data_schema_field_rules(name, field, bad):
    d = load(name)
    d[field] = bad
    with pytest.raises(ValidationError):
        getattr(schemas, name).model_validate(d)


def test_data_manifest_file_hashes_must_be_sha256():
    d = load("DataManifest")
    d["files"] = {"artifacts/data/train.jsonl": "not-a-hash"}
    with pytest.raises(ValidationError):
        schemas.DataManifest.model_validate(d)


def test_live_recipe_loads_and_is_pinned():
    r = load_recipe(ROOT / "configs" / "recipe.yaml")
    assert r.base_model == "google/gemma-4-E2B-it"
    assert r.is_pinned and len(r.base_revision) == 40
    assert (r.lora_r, r.lora_alpha, r.seed, r.max_seq_len) == (16, 32, 7, 1024)
    assert r.compute_dtype == "float16" and r.completion_only_loss


@pytest.mark.parametrize(
    "line, field",
    [
        ("learning_rate: 1.5", "learning_rate"),
        ("max_seq_len: 1000", "max_seq_len"),
        ("lora_dropout: 1.0", "lora_dropout"),
        ("epochs: 0", "epochs"),
        ("lora_r: 0", "lora_r"),
        ("base_revision: main", "base_revision"),
        ("target_modules: []", "target_modules"),
        ("surprise: true", "surprise"),
        ("lora_r: '16'", "lora_r"),
    ],
)
def test_recipe_errors_name_the_field(tmp_path, line, field):
    text = (ROOT / "configs" / "recipe.yaml").read_text(encoding="utf-8")
    key = line.split(":")[0]
    kept = [ln for ln in text.splitlines() if not ln.startswith(key + ":")]
    bad = tmp_path / "recipe.yaml"
    bad.write_text("\n".join([*kept, line]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_recipe(bad)


def test_recipe_loader_reports_unreadable_and_non_mapping_files(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_recipe(tmp_path / "missing.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_recipe(tmp_path / "list.yaml")
    (tmp_path / "bad.yaml").write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_recipe(tmp_path / "bad.yaml")


def test_unpinned_revision_is_allowed_but_reported():
    r = load_recipe(ROOT / "configs" / "recipe.yaml").model_copy(update={"base_revision": UNPINNED})
    assert not r.is_pinned


@pytest.mark.parametrize(
    "field, bad",
    [("schema_valid", 1.2), ("over_80_words", -0.1), ("n", 0), ("mean_words", math.inf)],
)
def test_dev_metrics_ranges(field, bad):
    d = load("DevMetrics")
    d[field] = bad
    with pytest.raises(ValidationError):
        schemas.DevMetrics.model_validate(d)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"selected_step": 300}, "not on val_loss_curve"),
        ({"val_loss_curve": [[500, 1.0], [250, 1.2]]}, "strictly increasing"),
        ({"val_loss_curve": []}, "at least one point"),
        ({"steps": 900}, "past the last training step"),
    ],
)
def test_train_run_curve_rules(changes, message):
    d = load("TrainRun")
    d.update(changes)
    with pytest.raises(ValidationError, match=message):
        schemas.TrainRun.model_validate(d)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"quant": None}, "needs quant and llama_cpp_commit"),
        ({"llama_cpp_commit": None}, "needs quant and llama_cpp_commit"),
        ({"kind": "merged_bf16"}, "has no quant"),
        ({"kind": "adapter", "quant": None}, "has no quant"),
    ],
)
def test_export_artifact_gguf_fields(changes, message):
    d = load("ExportArtifact")
    d.update(changes)
    with pytest.raises(ValidationError, match=message):
        schemas.ExportArtifact.model_validate(d)


def test_non_gguf_artifact_without_quant_is_valid():
    d = load("ExportArtifact")
    d.update(kind="merged_bf16", quant=None, llama_cpp_commit=None)
    assert schemas.ExportArtifact.model_validate(d).quant is None


@pytest.mark.safety
def test_calibrated_bundle_requires_the_gemma_family_check():
    d = load("ResultsBundle")
    d["gemma_family_check"] = None
    with pytest.raises(ValidationError, match="OWNER D-24"):
        schemas.ResultsBundle.model_validate(d)
    d["calibration_status"] = "uncalibrated"
    assert schemas.ResultsBundle.model_validate(d).gemma_family_check is None


def test_behaviour_row_source_format():
    d = load("BehaviourRow")
    for good in ("evalkit:citation_valid", "ggh:b8", "owner_review"):
        d["source"] = good
        schemas.BehaviourRow.model_validate(d)
    d["source"] = "somebody"
    with pytest.raises(ValidationError):
        schemas.BehaviourRow.model_validate(d)


def test_objects_are_deeply_immutable():
    ex = schemas.TrainExample.model_validate(load("TrainExample"))
    with pytest.raises(ValidationError):
        ex.support = 0.1  # type: ignore[misc]
    assert isinstance(ex.context, tuple)
    m = schemas.DataManifest.model_validate(load("DataManifest"))
    with pytest.raises(TypeError):
        m.files["x"] = "y"  # type: ignore[index]
    with pytest.raises(TypeError):
        m.counts["train"]["yes"] = 0  # type: ignore[index]
    assert hash(m.files)


def test_bools_are_not_counts():
    d = load("DataManifest")
    d["seed"] = True
    with pytest.raises(ValidationError):
        schemas.DataManifest.model_validate(d)


def test_calibrated_bundle_rejects_an_empty_gemma_check():
    d = load("ResultsBundle")
    d["gemma_family_check"] = {}
    with pytest.raises(ValidationError, match="OWNER D-24"):
        schemas.ResultsBundle.model_validate(d)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("citation_valid", 1.5),
        ("label_accuracy", {"yes": 1.2}),
        ("has_citation", True),
        ("schema_valid", "0.9"),
    ],
)
def test_dev_metrics_bounds_and_strictness(field, bad):
    d = load("DevMetrics")
    d[field] = bad
    with pytest.raises(ValidationError):
        schemas.DevMetrics.model_validate(d)


def test_val_loss_curve_steps_are_integers():
    d = load("TrainRun")
    d["val_loss_curve"] = [[250.5, 1.2], [1000, 0.9]]
    with pytest.raises(ValidationError):
        schemas.TrainRun.model_validate(d)


@pytest.mark.parametrize(
    "line, field",
    [
        ("warmup_ratio: 1.0", "warmup_ratio"),
        ("lora_alpha: 0", "lora_alpha"),
        ("load_in_4bit: 1", "load_in_4bit"),
        ("epochs: true", "epochs"),
        ("learning_rate: '0.1'", "learning_rate"),
    ],
)
def test_recipe_strict_types_and_ranges(tmp_path, line, field):
    text = (ROOT / "configs" / "recipe.yaml").read_text(encoding="utf-8")
    key = line.split(":")[0]
    kept = [ln for ln in text.splitlines() if not ln.startswith(key + ":")]
    bad = tmp_path / "recipe.yaml"
    bad.write_text("\n".join([*kept, line]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_recipe(bad)


def test_recipe_rejects_duplicate_yaml_keys(tmp_path):
    text = (ROOT / "configs" / "recipe.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "recipe.yaml"
    bad.write_text("lora_r: 0\n" + text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'lora_r'"):
        load_recipe(bad)


def test_recipe_with_an_unhashable_yaml_key_is_a_config_error(tmp_path):
    bad = tmp_path / "recipe.yaml"
    bad.write_text("? [a]\n: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_recipe(bad)


@pytest.mark.parametrize("bad", [True, "3.5"])
def test_non_negative_floats_are_strict(bad):
    d = load("DevMetrics")
    d["mean_words"] = bad
    with pytest.raises(ValidationError):
        schemas.DevMetrics.model_validate(d)


def test_recipe_loader_turns_decode_and_nesting_errors_into_config_errors(tmp_path):
    (tmp_path / "bin.yaml").write_bytes(b"lora_r: \xff\n")
    with pytest.raises(ConfigError):
        load_recipe(tmp_path / "bin.yaml")
    (tmp_path / "deep.yaml").write_text("a: " + "[" * 5000 + "]" * 5000 + "\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_recipe(tmp_path / "deep.yaml")


def test_recipe_seed_is_bounded(tmp_path):
    text = (ROOT / "configs" / "recipe.yaml").read_text(encoding="utf-8")
    kept = [ln for ln in text.splitlines() if not ln.startswith("seed:")]
    bad = tmp_path / "recipe.yaml"
    bad.write_text("\n".join([*kept, "seed: 99999999999999999999999"]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="seed"):
        load_recipe(bad)
