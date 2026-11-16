"""SCH-05: the EvalKit symbols this repo relies on, checked against a list written here from the
Tech Spec (D-2, §3, §7), never derived from the code under test (EvalKit L-10)."""

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINNED = "06591f8de9b178078c971940a83019a6530fd119"

REQUIRED = {
    "evalkit.schemas": ["Answer", "Case", "Interval", "RunSummary"],
    "evalkit.text": ["number_sentences"],
    "evalkit.tasks": ["builtin_task", "render_prompt"],
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
def test_pyproject_pins_the_commit_recorded_in_d28():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    pin = [d for d in deps if d.startswith("evalkit-harness")]
    # the SDK extras serve its endpoint route and its judges (E3.1)
    assert pin == [
        f"evalkit-harness[openai,anthropic] @ git+https://github.com/biplob819/evalkit@{PINNED}"
    ]
    decisions = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    d28 = next(ln for ln in decisions.splitlines() if ln.startswith("| D-28 |"))
    assert PINNED in d28 or f"`{PINNED[:7]}`" in d28


@pytest.mark.contract
def test_installed_evalkit_is_the_pinned_commit():
    import json
    from importlib.metadata import distribution

    direct = distribution("evalkit-harness").read_text("direct_url.json")
    assert direct, "evalkit-harness was not installed from the git pin"
    assert json.loads(direct)["vcs_info"]["commit_id"] == PINNED
