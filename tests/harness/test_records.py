import json
import re
from pathlib import Path

import pytest

import status
from harness_lib import ROOT, load_catalog, load_graph

WORKFLOWS = ROOT / ".github" / "workflows"
SPEC = ROOT / "Gemma Grounded Health Tech Spec.md"


def _spec_section(start: str, end: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    return text[text.index(start) : text.index(end)]


def _spec_nodes() -> dict[str, tuple[str, list[str]]]:
    """Independent reading of Tech Spec §13.2 (EK-L-10): id -> (title, depends_on)."""
    out = {}
    for line in _spec_section("### 13.2", "### 13.3").splitlines():
        m = re.match(r"^\| \*\*(\w\d\.\d)\*\* \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            deps = [] if m[3].strip() == "—" else [d.strip() for d in m[3].split(",")]
            out[m[1]] = (m[2].strip(), deps)
    return out


def _spec_caps() -> list[tuple[str, str]]:
    """Independent reading of Tech Spec §13.4: (cap_id, node) in order."""
    caps, node = [], None
    for line in _spec_section("### 13.4", "### 13.5").splitlines():
        m = re.match(r"^\*\*(\w\d\.\d) ", line)
        if m:
            node = m[1]
        m = re.match(r"^\| ([A-Z]{3}-\d\d) \|", line)
        if m:
            caps.append((m[1], node))
    return caps


@pytest.mark.contract
def test_graph_matches_tech_spec():
    nodes = load_graph()
    spec = _spec_nodes()
    assert len(nodes) == len(spec) == 20  # V2.1, V2.2 by OWNER D-43
    assert {n: (v.title, v.depends_on) for n, v in nodes.items()} == spec


@pytest.mark.contract
def test_catalog_matches_tech_spec():
    caps = load_catalog()
    assert len(caps) == 128  # SPC-08, OSS-06 (OWNER D-36, D-37); VTG, VEV (OWNER D-43)
    assert [(c.cap_id, c.node) for c in caps] == _spec_caps()
    nodes = load_graph()
    assert {c.node for c in caps} == set(nodes)


def test_decisions_d1_onwards_with_owner_rulings():
    text = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    ids = {int(n) for n in re.findall(r"^\| (?:OWNER )?D-(\d+) \|", text, re.M)}
    # D-1..33 seeded; 34-35 N1.3; 36-39 owner; 40 T2.1; 41 X3.1; 42 E3.2; 43-50 owner;
    # 51-52 V2.1; 53-55 owner; 56 the EvalKit 0.4.0 pin; 57-59 owner (OpenAI, O6.1 cut, calibration)
    assert ids == set(range(1, 60))
    owner = {int(n) for n in re.findall(r"^\| OWNER D-(\d+) \|", text, re.M)}
    assert owner == {
        15,
        17,
        23,
        24,
        25,
        26,
        27,
        33,
        34,
        36,
        37,
        38,
        39,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        53,
        54,
        55,
        57,
        58,
        59,
    }


def test_lessons_carry_exactly_the_appendix_b_lessons():
    text = (ROOT / "docs" / "lessons.md").read_text(encoding="utf-8")
    carried = re.findall(r"^\| EK-L-(\d+) \|", text, re.M)
    appendix = _spec_section("## Appendix B", "## Sources")
    expected = re.findall(r"^\| L-(\d+) ", appendix, re.M)
    assert carried == expected and len(expected) == 16
    assert {"10", "12"} <= set(carried)  # cited by the W0.1 criteria
    assert re.search(r"^\| L-1 \|", text, re.M)  # this project's own lessons


def test_decisions_equal_tech_spec_section_1():
    spec_rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in _spec_section("## 1. Design decisions", "## 2. Artifact").splitlines()
        if re.match(r"^\| (OWNER )?D-\d+ \|", line)
    ]
    text = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    log_rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if re.match(r"^\| (OWNER )?D-\d+ \|", line)
    ]
    assert len(spec_rows) == 59
    assert [r[:1] + r[2:] for r in log_rows] == spec_rows  # drop the date column
    assert "DeepMind" not in text


def test_makefile_offers_harness_targets():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("check:", "status:", "converge:", "selftest:"):
        assert re.search(rf"^{target}", text, re.M), target


def test_changelog_has_w01_entry():
    assert "[W0.1]" in (ROOT / "changelog.md").read_text(encoding="utf-8")


def test_status_prints_graph_node_and_ledgers(capsys):
    assert status.main([]) == 0
    out = capsys.readouterr().out
    assert "W0.1" in out and "O6.1" in out and "capabilities green" in out
    assert "API spend: $" in out and "Kaggle GPU:" in out
    assert status.main(["W0.1"]) == 0
    assert "HAR-08" in capsys.readouterr().out
    assert status.main(["Z9.9"]) == 1


def test_status_ledger_totals_sum_entries(tmp_path):
    (tmp_path / "spend-ledger.jsonl").write_text(
        json.dumps({"spent_usd": 0.1}) + "\n" + json.dumps({"spent_usd": 0.2}) + "\n",
        encoding="utf-8",
    )
    line = status.ledger_totals(tmp_path)
    assert "API spend: $0.30 (2 entries)" in line
    assert "Kaggle GPU: 0.00 h (0 entries)" in line


def _yaml_code(path: Path) -> str:
    """Workflow text with comment-only lines and trailing comments removed."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split(" #", 1)[0].rstrip()
        if code.strip() and not code.lstrip().startswith("#"):
            lines.append(code)
    return "\n".join(lines)


@pytest.mark.contract
def test_ci_workflow_covers_gates_and_matrix():
    ci = _yaml_code(WORKFLOWS / "ci.yml")
    assert re.search(r"^on:\n  push:\n    branches: \[main\]\n  pull_request:", ci, re.M)
    for step in (
        "ruff check",
        "ruff format --check",
        "pyright",
        "pytest",
        "uv build",
        "secret_scan.py",
        "verify_graph_eligibility.py",
        "node_converge.py --self-test",
    ):
        assert step in ci, step
    assert "uv sync --locked --all-extras" in ci
    assert 'python: ["3.11", "3.12"]' in ci
    assert "os: [ubuntu-latest, macos-latest]" in ci


@pytest.mark.safety
def test_workflows_grant_least_privilege():
    ci = _yaml_code(WORKFLOWS / "ci.yml")
    assert re.search(r"^permissions:\n  contents: read$", ci, re.M)
    assert "secrets." not in ci
    assert sorted(p.name for p in WORKFLOWS.iterdir()) == ["ci.yml"]
    assert "persist-credentials: false" in ci


def test_catalog_round_trip_is_byte_identical(tmp_path):
    from harness_lib import CATALOG_TSV, write_catalog

    out = tmp_path / "catalog.tsv"
    write_catalog(load_catalog(), out)
    assert out.read_bytes() == CATALOG_TSV.read_bytes()


def test_status_output_is_deterministic(capsys):
    status.main([])
    first = capsys.readouterr().out
    status.main([])
    assert capsys.readouterr().out == first


def test_gitattributes_force_lf():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text


def test_pre_commit_runs_ruff_and_secret_scan():
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "id: ruff" in text and "scripts/secret_scan.py" in text


@pytest.mark.safety
def test_credentials_and_artifacts_are_ignored():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".env",
        ".env.*",
        "kaggle*.json",
        "artifacts/",
        "*.gguf",
        "*.safetensors",
        ".evalkit/",
    ):
        assert pattern in text.splitlines(), pattern


def test_docs_results_is_not_ignored():
    import subprocess

    for rel in ("docs/results/results_bundle.json", "src/ggh/results/x.py"):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
        assert r.returncode == 1, f"{rel} is ignored"
    r = subprocess.run(["git", "check-ignore", "-q", "results/run.json"], cwd=ROOT, check=False)
    assert r.returncode == 0


@pytest.mark.safety
def test_public_prd_has_no_application_framing():
    """OWNER D-26: job-search content stays out of the public repo."""
    prd = (ROOT / "Gemma Grounded Health PRD.md").read_text(encoding="utf-8")
    for phrase in ("hiring manager", "application", "careers", "Interviewers", "November"):
        assert phrase not in prd, phrase
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "Gemma Grounded Health PRD (application).md" in ignored


@pytest.mark.parametrize(
    "line, bad",
    [
        ('{"spent_usd": NaN}', "nan"),
        ('{"spent_usd": -5}', "-5"),
        ('{"spent_usd": "7"}', "7.00"),
        ('{"spent_usd": true}', "1.00"),
        ('{"kind": "x"}', None),
        ("not json", None),
        ('{"spent_usd": ' + "9" * 5000 + "}", None),
        ('{"spent_usd": 1e308}\n{"spent_usd": 1e308}', None),
    ],
)
def test_status_reports_a_bad_ledger_instead_of_a_total(tmp_path, monkeypatch, capsys, line, bad):
    (tmp_path / "spend-ledger.jsonl").write_text(line + "\n", encoding="utf-8")
    real = status.ledger_totals
    monkeypatch.setattr(status, "ledger_totals", lambda: real(tmp_path))
    assert status.main([]) == 1
    out = capsys.readouterr().out
    assert "Ledgers — ERROR" in out
    if bad:
        assert f"API spend: ${bad}" not in out


def test_status_reports_a_measurements_path_that_is_a_file(tmp_path, monkeypatch, capsys):
    afile = tmp_path / "measurements"
    afile.write_text("x", encoding="utf-8")
    real = status.ledger_totals
    monkeypatch.setattr(status, "ledger_totals", lambda: real(afile))
    assert status.main([]) == 1
    assert "Ledgers — ERROR" in capsys.readouterr().out


def test_status_reports_a_looping_measurements_dir(tmp_path, monkeypatch, capsys):
    loop = tmp_path / "measurements"
    loop.symlink_to(loop)
    real = status.ledger_totals
    monkeypatch.setattr(status, "ledger_totals", lambda: real(loop))
    assert status.main([]) == 1
    assert "Ledgers — ERROR" in capsys.readouterr().out
