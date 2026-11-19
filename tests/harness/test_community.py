"""HAR-09: open-source community files (OWNER D-27)."""

import re

import pytest

import ggh
from harness_lib import ROOT

REPO = "https://github.com/biplob819/gemma-grounded-health"
EVALKIT = "https://github.com/biplob819/evalkit"  # the README credits the evaluation tool
FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/model_behaviour.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_community_files_exist_and_readme_links_them():
    for rel in FILES:
        assert (ROOT / rel).is_file(), rel
    readme = _read("README.md")
    for rel in (
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CITATION.cff",
        ".github/ISSUE_TEMPLATE",
        ".github/PULL_REQUEST_TEMPLATE.md",
    ):
        assert f"]({rel})" in readme, rel


@pytest.mark.contract
def test_links_point_at_this_repo_not_evalkit():
    for rel in FILES:
        allowed = {REPO, EVALKIT} if rel == "README.md" else {REPO}
        for url in re.findall(r"https://github\.com/[\w.-]+/[\w.-]+", _read(rel)):
            assert url in allowed, (rel, url)
    config = _read(".github/ISSUE_TEMPLATE/config.yml")
    assert "blank_issues_enabled: false" in config
    assert f"{REPO}/security/advisories/new" in config
    assert f"{REPO}/security/advisories/new" in _read("SECURITY.md")


def test_contributing_states_project_rules_and_ci_commands():
    text = _read("CONTRIBUTING.md")
    assert "No Claude or OpenAI output may enter training data" in text
    assert "Never tune on the expert set" in text
    ci = _read(".github/workflows/ci.yml")
    ci_cmds = re.findall(r"^\s+- run: uv run (.+)$|^\s+- run: (uv build)$", ci, re.M)
    ci_cmds = [a or b for a, b in ci_cmds]
    assert len(ci_cmds) == 8
    for cmd in ci_cmds:
        assert cmd in text, cmd  # CONTRIBUTING lists every check CI runs
    assert "secret_scan.py" in _read(".github/PULL_REQUEST_TEMPLATE.md")


@pytest.mark.contract
def test_citation_metadata_matches_the_package():
    cff = _read("CITATION.cff")
    assert re.search(r"^cff-version: 1\.2\.0$", cff, re.M)
    assert re.search(r"^license: Apache-2\.0$", cff, re.M)
    assert re.search(rf'^repository-code: "{re.escape(REPO)}"$', cff, re.M)
    m = re.search(r"^version: (\S+)$", cff, re.M)
    assert m and m[1] == ggh.__version__
    assert "PubMedQA" in cff


@pytest.mark.safety
def test_templates_warn_against_pasting_secrets():
    assert "Never paste API keys" in _read(".github/ISSUE_TEMPLATE/bug_report.yml")
    assert "No tokens" in _read(".github/PULL_REQUEST_TEMPLATE.md")


def test_docs_name_future_work_by_node():
    """EK-L-16 / L-4: features that do not exist yet are tied to the node that builds them."""
    security = _read("SECURITY.md")
    assert "build node T2.1" in security and "build node R4.3" in security
    assert "build node R4.5" in security
    contributing = _read("CONTRIBUTING.md")
    assert "build node T2.1" in contributing and "(E3.3)" in contributing
    readme = _read("README.md")
    assert "> **Status.**" in readme and "(build node V2.2)" in readme
    assert "(nodes R4.2 and R4.3)" in readme
    assert "Today nothing in the repo reads a key" in security
    assert "once released" in _read("CITATION.cff") and "build node T2.1" in _read("CITATION.cff")
    assert "will be frozen at node N1.2" in security
    assert "Once released (build node R4.2)" in security
    makefile = _read("Makefile")
    for step in ("verify_graph_eligibility.py", "uv build", "secret_scan.py"):
        assert step in makefile, step
