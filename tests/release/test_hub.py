"""HUB-01..05: the release plan lists exactly what goes to the two Hub repos, checks lineage and
every sha256, and nothing is pushed without --push, a token and a clean plan. All offline: the
Hub is a fake."""

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

from ggh import cli
from ggh.lineage import StageManifest, sha256_file, stage_key, write_manifest
from ggh.release import hub
from tests.release import build_v1_bundle as B

ROOT = Path(__file__).resolve().parents[2]
T0 = dt.datetime(2026, 11, 18, tzinfo=dt.UTC)
MERGED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
ADAPTER = ("adapter_config.json", "adapter_model.safetensors")
GGUFS = ("ft-Q4_K_M.gguf", "ft-Q8_0.gguf")


def put(root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(f"bytes of {rel}".encode())
    return sha256_file(p)


def manifest(
    root: Path, stage: str, outputs: dict[str, str], parents: tuple[str, ...], where: Path
) -> StageManifest:
    m = StageManifest(
        stage=stage,
        code_version="1",
        key=stage_key(stage, "1", {}, {}, 7),
        inputs={},
        outputs=outputs,
        params={},
        seed=7,
        parents=parents,
        started_at=T0,
        finished_at=T0,
        host="test",
    )
    write_manifest(root / where, m)
    copy = root / hub.LINEAGE_COPIES / f"{stage}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    return m


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    train_out = {f"artifacts/adapter/{f}": put(tmp_path, f"artifacts/adapter/{f}") for f in ADAPTER}
    train = manifest(tmp_path, "train", train_out, (), hub.TRAIN_MANIFEST.parent)
    exp_out = {
        f"artifacts/merged/ft-bf16/{f}": put(tmp_path, f"artifacts/merged/ft-bf16/{f}")
        for f in MERGED
    }
    exp_out |= {f"artifacts/gguf/{g}": put(tmp_path, f"artifacts/gguf/{g}") for g in GGUFS}
    export = manifest(tmp_path, "export", exp_out, (train.key,), hub.EXPORT_MANIFEST.parent)
    for name in hub.LEGAL:
        shutil.copy(ROOT / name, tmp_path / name)
    data = json.loads(B.BUNDLE.read_text(encoding="utf-8"))
    art = lambda kind, path, sha, key, quant=None: {  # noqa: E731
        "kind": kind,
        "model": "ft",
        "quant": quant,
        "path": path,
        "sha256": sha,
        "llama_cpp_commit": "b1" if quant else None,
        "parent_key": key,
    }
    data["artifacts"] = [
        art(
            "adapter",
            "artifacts/adapter",
            train_out["artifacts/adapter/adapter_model.safetensors"],
            train.key,
        ),
        art(
            "merged_bf16",
            "artifacts/merged/ft-bf16/model.safetensors",
            exp_out["artifacts/merged/ft-bf16/model.safetensors"],
            export.key,
        ),
        art(
            "gguf",
            "artifacts/gguf/ft-Q4_K_M.gguf",
            exp_out["artifacts/gguf/ft-Q4_K_M.gguf"],
            export.key,
            "Q4_K_M",
        ),
        art(
            "gguf",
            "artifacts/gguf/ft-Q8_0.gguf",
            exp_out["artifacts/gguf/ft-Q8_0.gguf"],
            export.key,
            "Q8_0",
        ),
    ]
    (tmp_path / "bundle.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def make_plan(root: Path, owner: str = "someone") -> hub.Plan:
    return hub.plan(root, Path("bundle.json"), "# card\n", owner=owner)


def test_plan_lists_exactly_the_d13_files(repo):  # HUB-02, HUB-03
    p = make_plan(repo)
    assert p.problems == []
    assert sorted(f.path_in_repo for f in p.by_repo("main")) == sorted(
        [
            *MERGED,
            *(f"adapter/{a}" for a in ADAPTER),
            "README.md",
            "results_bundle.json",
            "lineage.json",
            *hub.LEGAL,
        ]
    )
    assert sorted(f.path_in_repo for f in p.by_repo("gguf")) == sorted(
        [f"{hub.MAIN_REPO}-Q4_K_M.gguf", f"{hub.MAIN_REPO}-Q8_0.gguf", "README.md", *hub.LEGAL]
    )
    assert p.repo_id("main") == "someone/gemma-4-e2b-grounded-health"
    assert p.repo_id("gguf") == "someone/gemma-4-e2b-grounded-health-GGUF"
    assert p.ready


def test_lineage_json_walks_back_to_the_root(repo):
    p = make_plan(repo)
    content = next(f for f in p.files if f.path_in_repo == "lineage.json").content
    assert content is not None
    chain = json.loads(content)
    assert [m["stage"] for m in chain] == ["export", "train"]


def test_a_changed_file_blocks_the_release(repo):  # HUB-01, NFR5
    (repo / "artifacts/gguf/ft-Q8_0.gguf").write_bytes(b"tampered")
    p = make_plan(repo)
    assert any("hash mismatch" in x for x in p.problems)
    assert any(x.startswith("lineage:") for x in p.problems)
    assert not p.ready


def test_a_missing_file_and_a_placeholder_owner_are_not_ready(repo):
    (repo / "artifacts/adapter/adapter_config.json").unlink()
    p = make_plan(repo, owner=hub.OWNER_PLACEHOLDER)
    assert any("missing" in x for x in p.problems)
    assert not make_plan(repo, owner=hub.OWNER_PLACEHOLDER).ready


def test_bundle_and_manifest_must_agree(repo):
    data = json.loads((repo / "bundle.json").read_text(encoding="utf-8"))
    data["artifacts"][2]["sha256"] = "0" * 64
    (repo / "bundle.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("disagree" in x for x in make_plan(repo).problems)


class FakeHub:
    def __init__(self, corrupt: str | None = None) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.calls: list[str] = []
        self.corrupt = corrupt
        self.tmp: Path | None = None

    def create_repo(self, repo_id):
        self.calls.append(f"create {repo_id}")

    def upload(self, repo_id, path_in_repo, local):
        self.files[(repo_id, path_in_repo)] = Path(local).read_bytes()

    def tag(self, repo_id, tag):
        self.calls.append(f"tag {repo_id} {tag}")

    def download(self, repo_id, path_in_repo, revision):
        assert self.tmp is not None
        out = self.tmp / repo_id / path_in_repo
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[(repo_id, path_in_repo)]
        out.write_bytes(b"x" + data if path_in_repo == self.corrupt else data)
        return out


def test_push_uploads_tags_and_checks_every_download(repo, tmp_path):  # HUB-04, HUB-05
    api = FakeHub()
    api.tmp = tmp_path / "dl"
    asked: list[str] = []
    problems = hub.push(
        make_plan(repo),
        repo,
        api,
        tag="v1.0.0",
        staging=tmp_path / "stage",
        ollama=lambda ref: asked.append(ref) or True,
    )
    assert problems == []
    assert "tag someone/gemma-4-e2b-grounded-health v1.0.0" in api.calls
    assert asked == ["hf.co/someone/gemma-4-e2b-grounded-health-GGUF:Q4_K_M"]
    assert api.files[("someone/gemma-4-e2b-grounded-health", "README.md")] == b"# card\n"


def test_push_reports_a_bad_download(repo, tmp_path):
    api = FakeHub(corrupt="config.json")
    api.tmp = tmp_path / "dl"
    problems = hub.push(make_plan(repo), repo, api, tag="v1", staging=tmp_path / "s")
    assert problems == [
        "someone/gemma-4-e2b-grounded-health/config.json: downloaded sha256 differs"
    ]


def test_push_refuses_a_plan_that_is_not_ready(repo, tmp_path):
    api = FakeHub()
    with pytest.raises(hub.ReleaseError):
        hub.push(
            make_plan(repo, owner=hub.OWNER_PLACEHOLDER), repo, api, tag="v1", staging=tmp_path
        )
    assert api.calls == [] and api.files == {}


def test_token_is_read_by_name_only():
    assert hub.token_from_env({}) is None
    assert hub.token_from_env({"HF_TOKEN": "  "}) is None
    assert hub.token_from_env({"HF_TOKEN": "abc"}) == "abc"


def _cli(*extra: str) -> list[str]:
    rel = B.BUNDLE.relative_to(ROOT)
    return ["release", "--root", str(ROOT), "--bundle", str(rel), *extra]


def test_cli_dry_run_never_pushes(monkeypatch, capsys):
    def boom(*_, **__):
        raise AssertionError("push called")

    monkeypatch.setattr(hub, "push", boom)
    monkeypatch.setenv("HF_TOKEN", "secret-value-123")
    code = cli.main(_cli("--dry-run"))
    out = capsys.readouterr().out
    assert "dry run, nothing pushed" in out
    assert "secret-value-123" not in out
    assert code in (0, 2)  # 2 here: no artifacts/ in CI, so the plan lists missing files


def test_cli_push_without_a_token_refuses(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub, "push", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert cli.main(_cli("--push", "--owner", "someone")) == 2
    assert "needs HF_TOKEN" in capsys.readouterr().out


def test_cli_push_with_a_token_but_problems_refuses(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "secret-value-123")
    monkeypatch.setattr(hub, "HfHubApi", lambda *_: (_ for _ in ()).throw(AssertionError()))
    code = cli.main(_cli("--push", "--owner", "someone", "--export-manifest", "nope/manifest.json"))
    out = capsys.readouterr().out
    assert code == 2 and "nothing pushed" in out and "secret-value-123" not in out


def test_dry_run_on_the_committed_v1_records_lists_every_file():
    """Without artifacts/ (CI) the plan lists from the committed lineage copies; the v1 bundle's
    artifact hashes agree with them, so only missing local files keep it from being ready."""
    p = hub.plan(ROOT, B.BUNDLE.relative_to(ROOT), "# card\n", owner="someone")
    assert len(p.by_repo("main")) == len(MERGED) + len(ADAPTER) + 3 + len(hub.LEGAL)
    assert len(p.by_repo("gguf")) == len(GGUFS) + 1 + len(hub.LEGAL)
    assert not any("disagree" in x or "not a train output" in x for x in p.problems)
    if not (ROOT / hub.EXPORT_MANIFEST).is_file():
        assert not p.ready
