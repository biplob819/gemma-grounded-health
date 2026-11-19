import json
from pathlib import Path

import pytest
import yaml
from evalkit.config import load_run_config

from ggh.evals import config
from ggh.evals.config import EVALKIT_COMMIT

ROOT = Path(__file__).resolve().parents[2]


def copy_configs(tmp_path, edit_main=None, edit_dev=None):
    for rel, edit in ((config.MAIN, edit_main), (config.DEV, edit_dev)):
        data = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        if edit:
            edit(data)
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def test_spec_covers_every_run_config_field():
    from evalkit.schemas import RunConfig

    assert set(config.SPEC) == set(RunConfig.model_fields)  # nothing unchecked or dropped


def test_committed_configs_match_section_7_1_and_each_other():
    assert config.config_problems(ROOT) == []
    main = load_run_config(ROOT / config.MAIN)
    assert dict(main.endpoints) == {
        "gemma-e2b-base-q4": "http://127.0.0.1:8080/v1",
        "gemma-e2b-grounded-q4": "http://127.0.0.1:8081/v1",
    }
    assert main.seed == 7 and main.structured_output


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 8), ("structured_output", False), ("max_output_tokens", 1024),
     ("judges", ["claude-opus-5-5"]), ("max_cost_usd", 100)],
)  # fmt: skip
def test_any_drift_from_section_7_1_is_reported(tmp_path, field, value):
    root = copy_configs(tmp_path, edit_main=lambda d: d.__setitem__(field, value))
    assert any(field in p for p in config.config_problems(root))


def test_the_dev_config_may_differ_only_in_models_endpoints_and_cap(tmp_path):
    root = copy_configs(tmp_path, edit_dev=lambda d: d.__setitem__("max_output_tokens", 1024))
    assert config.config_problems(root) == [
        f"{config.DEV}: max_output_tokens differs from {config.MAIN}"
    ]
    root = copy_configs(
        tmp_path, edit_dev=lambda d: d["endpoints"].__setitem__("gemma-e2b-base-q4", "http://x/v1")
    )
    assert any("endpoint" in p for p in config.config_problems(root))


def test_evalkit_is_pinned_to_the_d48_release_and_installed_from_it():
    assert config.pinned_version(ROOT) == "0.4.0"
    assert config.pinned_commit(ROOT) == EVALKIT_COMMIT
    assert EVALKIT_COMMIT == "f9cbc8e5396c4fc638021cb426cc34e2d6853d78"  # tag v0.4.0
    assert config.pin_problems(ROOT) == []


MINE = {
    "evalkit": EVALKIT_COMMIT,
    "prompt_hash": "p",
    "rubric_hash": "r",
    "extractor_version": "e",
    "seed": 7,
    "structured_output": True,
    "max_output_tokens": 2048,
    "judges": ["claude-opus-5-5", "gpt-6.1-sol"],
    "extractor": "claude-haiku-5-5",
    "judge_effort": "medium",
    "extractor_effort": "medium",
    "effort": {"claude-haiku-5-5": "medium", "gpt-6-luna": "medium"},
}


@pytest.fixture
def store(tmp_path):
    s = tmp_path / "store"
    s.mkdir()
    (s / "entry").write_text("x", encoding="utf-8")
    return s


def test_reuse_is_claimable_only_when_every_condition_matches(store, tmp_path):
    assert config.reuse_problems(MINE, dict(MINE), store) == []
    for field, other in [
        ("evalkit", "unknown"), ("prompt_hash", "q"), ("rubric_hash", "s"),
        ("extractor_version", "f"), ("seed", 8), ("structured_output", False),
        ("max_output_tokens", 1024), ("judges", ["claude-opus-5-5"]),
        ("extractor", "gpt-6-luna"), ("judge_effort", "high"), ("extractor_effort", "low"),
    ]:  # fmt: skip
        problems = config.reuse_problems(MINE, {**MINE, field: other}, store)
        assert len(problems) == 1 and problems[0].startswith(field), field
    assert set(config.COMPARED) | {"effort"} == set(MINE)  # every recorded field is compared
    effort = {**MINE, "effort": {"claude-haiku-5-5": "high"}}
    assert config.reuse_problems(MINE, effort, store) == [
        "effort for claude-haiku-5-5: ours 'medium', reference 'high'"
    ]
    assert config.reuse_problems(MINE, {**MINE, "effort": {"x": "medium"}}, store) == [
        "no generator model is shared with the reference run"
    ]


def test_reuse_needs_the_store_itself(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for s in (None, tmp_path / "missing", empty):
        (p,) = config.reuse_problems(MINE, dict(MINE), s)
        assert p.startswith("no EvalKit store")


def write_run(tmp_path, records, cfg_text, seed=7):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    (run / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    cfg = tmp_path / "ref.yaml"
    cfg.write_text(cfg_text, encoding="utf-8")
    return run, cfg


def rec(model, sha="abc", ph="p", rubric="r", ext="e", effort="medium", structured=True,
        judges=("claude-opus-5-5", "gpt-6.1-sol")):  # fmt: skip
    return {
        "git_sha": sha, "prompt_hash": ph,
        "completion": {
            "model": model, "params": {"effort": effort, "structured_output": structured}
        },
        "claims": [{"extractor_version": ext}],
        "verdicts": [{"judge_prompt_hash": rubric, "judge": j} for j in judges],
    }  # fmt: skip


def test_reference_reads_a_run_folder_and_flags_mixed_values(tmp_path):
    cfg_text = (ROOT / config.MAIN).read_text(encoding="utf-8")
    run, cfg = write_run(tmp_path, [rec("claude-haiku-5-5"), rec("gpt-6-luna")], cfg_text, seed=8)
    ref = config.reference(run, cfg)
    assert (
        ref["evalkit"] == "abc" and ref["prompt_hash"] == "p" and ref["max_output_tokens"] == 2048
    )
    assert ref["seed"] == 8  # read from run.json, not assumed
    assert (
        ref["judges"] == ["claude-opus-5-5", "gpt-6.1-sol"]
        and ref["extractor"] == "claude-haiku-5-5"
    )
    assert ref["effort"] == {"claude-haiku-5-5": "medium", "gpt-6-luna": "medium"}
    mixed = tmp_path / "m"
    mixed.mkdir()
    run2, cfg2 = write_run(mixed, [rec("a", sha="x"), rec("b", sha="y")], cfg_text)
    assert config.reference(run2, cfg2)["evalkit"] == "<2 values of git_sha>"
    loose = tmp_path / "l"
    loose.mkdir()
    run3, cfg3 = write_run(loose, [rec("a", structured=False)], cfg_text)
    assert config.reference(run3, cfg3)["structured_output"] is False  # from the records


def test_ours_reads_the_installed_evalkit():
    from evalkit.identity import EXTRACTOR_VERSION, JUDGE_PROMPT_HASH

    mine = config.ours(load_run_config(ROOT / config.MAIN), EVALKIT_COMMIT)
    assert JUDGE_PROMPT_HASH == "9a3a8246049f6f85"  # as every committed v1 verdict carries
    assert (
        mine["rubric_hash"] == JUDGE_PROMPT_HASH and mine["extractor_version"] == EXTRACTOR_VERSION
    )
    assert set(mine["effort"]) == {"claude-haiku-5-5", "gpt-6-luna"}  # endpoints excluded
    assert len(mine["prompt_hash"]) == 16


def test_reference_reads_every_field_from_its_own_source(tmp_path):
    """Distinct values everywhere, so a field read from the wrong place shows."""
    cfg_text = (
        (ROOT / config.MAIN).read_text(encoding="utf-8")
        .replace("max_output_tokens: 2048", "max_output_tokens: 1536")
        .replace("judge_effort: medium", "judge_effort: high")
        .replace("extractor_effort: medium", "extractor_effort: low")
        .replace("extractor: claude-haiku-5-5", "extractor: gpt-6-luna")
    )  # fmt: skip
    records = [rec("claude-haiku-5-5", ph="PH", rubric="RUB", ext="EXT", judges=("judge-x",))]
    run, cfg = write_run(tmp_path, records, cfg_text)
    ref = config.reference(run, cfg)
    assert ref["prompt_hash"] == "PH" and ref["rubric_hash"] == "RUB"
    assert ref["extractor_version"] == "EXT"
    assert ref["judges"] == ["judge-x"]  # from the verdicts, not the config
    assert ref["max_output_tokens"] == 1536 and ref["extractor"] == "gpt-6-luna"
    assert ref["judge_effort"] == "high" and ref["extractor_effort"] == "low"


def test_ours_reads_every_field_from_the_config(tmp_path):
    text = (
        (ROOT / config.MAIN).read_text(encoding="utf-8")
        .replace("max_output_tokens: 2048", "max_output_tokens: 1536")
        .replace("judge_effort: medium", "judge_effort: high")
    )  # fmt: skip
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    mine = config.ours(load_run_config(p), "sha")
    assert mine["evalkit"] == "sha" and mine["max_output_tokens"] == 1536
    assert mine["judges"] == ["claude-opus-5-5", "gpt-6.1-sol"] and mine["judge_effort"] == "high"


@pytest.mark.parametrize("field", ["judges", "seed", "structured_output", "extractor"])
def test_the_dev_config_may_not_differ_in(tmp_path, field):
    values = {"judges": ["claude-opus-5-5"], "seed": 8, "structured_output": False,
              "extractor": "gpt-6-luna"}  # fmt: skip
    root = copy_configs(tmp_path, edit_dev=lambda d: d.__setitem__(field, values[field]))
    assert any(field in p for p in config.config_problems(root))


def test_a_pyproject_pin_other_than_d48_is_a_problem(tmp_path):
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").replace("==0.4.0", "==0.3.0")
    (tmp_path / "pyproject.toml").write_text(text, encoding="utf-8")
    problems = config.pin_problems(tmp_path)
    assert any("D-56 says 0.4.0" in p for p in problems)
    with pytest.raises(ValueError, match="no known commit"):
        config.pinned_commit(tmp_path)
    git = (
        (ROOT / "pyproject.toml")
        .read_text(encoding="utf-8")
        .replace(
            "evalkit-harness[openai,anthropic]==0.4.0",
            "evalkit-harness[openai,anthropic] @ git+https://github.com/biplob819/evalkit@"
            + EVALKIT_COMMIT,
        )
    )
    (tmp_path / "pyproject.toml").write_text(git, encoding="utf-8")
    with pytest.raises(ValueError, match="one release with =="):
        config.pinned_version(tmp_path)


def test_an_installed_evalkit_other_than_the_pin_is_a_problem(monkeypatch):
    monkeypatch.setattr(config, "installed_version", lambda: "0.3.0")
    assert config.pin_problems(ROOT) == ["installed EvalKit is 0.3.0, pyproject pins 0.4.0"]
    monkeypatch.setattr(config, "installed_version", lambda: "0.4.0")
    ee59699 = "ee5969978c9213acf4838ca0e673fdfaea34a876"
    monkeypatch.setattr(config, "installed_commit", lambda: ee59699)
    (p,) = config.pin_problems(ROOT)
    assert "git build of ee5969978c92" in p
    monkeypatch.setattr(config, "installed_commit", lambda: EVALKIT_COMMIT)
    assert config.pin_problems(ROOT) == []  # a git build of the tag itself is the release
