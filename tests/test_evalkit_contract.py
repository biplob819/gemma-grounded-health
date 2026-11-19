"""SCH-05: the EvalKit symbols this repo relies on, checked against a list written here from the
Tech Spec (D-2, §3, §7), never derived from the code under test (EvalKit L-10)."""

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINNED = "0.4.0"  # D-56: the PyPI release, tag v0.4.0
TAG_COMMIT = "f9cbc8e5396c4fc638021cb426cc34e2d6853d78"

REQUIRED = {
    "evalkit.schemas": ["Answer", "Case", "Interval", "RunSummary"],
    "evalkit.text": ["number_sentences"],
    "evalkit.tasks": ["builtin_task", "render_prompt"],
    # 0.4.0 homes of names this repo imports (EvalKit docs/downstream.md, 0.3 -> 0.4)
    "evalkit.identity": ["EXTRACTOR_VERSION", "JUDGE_PROMPT_HASH"],
    "evalkit.errors": ["SchemaError"],
    "evalkit.adapters": ["CompletionRequest", "RawCompletion"],
}


@pytest.mark.contract
@pytest.mark.parametrize("module, names", sorted(REQUIRED.items()))
def test_required_evalkit_symbols_import(module, names):
    mod = importlib.import_module(module)
    for name in names:
        assert hasattr(mod, name), f"{module}.{name} is gone from the pinned EvalKit"


@pytest.mark.contract
def test_pubmedqa_task_renders_with_the_pinned_evalkit():
    from evalkit.schemas import Case
    from evalkit.tasks import builtin_task, render_prompt
    from evalkit.text import number_sentences

    task = builtin_task("pubmedqa")
    case = Case(
        id="pqa-1",
        question="Q?",
        context=number_sentences(["One. Two."]),
        reference_label="yes",
        source="pubmedqa/pqa_l",
    )
    prompt = render_prompt(task, case)
    assert "[1] One." in prompt and "[2] Two." in prompt and "Question: Q?" in prompt


@pytest.mark.contract
def test_pyproject_pins_the_release_recorded_in_d56():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    pin = [d for d in deps if d.startswith("evalkit-harness")]
    # the SDK extras serve its endpoint route and its judges (E3.1)
    assert pin == [f"evalkit-harness[openai,anthropic]=={PINNED}"]
    decisions = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    d56 = next(ln for ln in decisions.splitlines() if ln.startswith("| D-56 |"))
    assert f"`evalkit-harness=={PINNED}`" in d56 and f"`{TAG_COMMIT[:7]}`" in d56


@pytest.mark.contract
def test_installed_evalkit_is_the_pinned_release():
    import json
    from importlib.metadata import distribution

    import evalkit

    dist = distribution("evalkit-harness")
    assert dist.version == evalkit.__version__ == PINNED
    direct = dist.read_text("direct_url.json")  # absent for a PyPI install
    if direct and "vcs_info" in json.loads(direct):
        assert json.loads(direct)["vcs_info"]["commit_id"] == TAG_COMMIT


@pytest.mark.contract
def test_the_0_4_record_and_summary_fields_this_repo_reads_exist():
    from evalkit.schemas import ModelSummary, Record, RunConfig

    assert {"judge_errors", "citation_support"} <= set(Record.model_fields)
    assert {"missing_verdicts", "efficiency", "citation_precision"} <= set(
        ModelSummary.model_fields
    )
    assert "systems" in RunConfig.model_fields
