"""HUB-01..05 for the shipped release (v2): the plan follows parity_v2 -> export_ft2 -> train_v2
-> targets -> data, lists the v2 merged weights, adapter and ft2 GGUFs with the hashes their
records hold, and nothing is pushed without --push, a token and a clean plan. All offline: the
Hub and Ollama are fakes."""

import dataclasses
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ggh import cli
from ggh.lineage import StageManifest, read_manifest, sha256_file, stage_key, write_manifest
from ggh.release import hub
from tests.release.test_hub import FakeHub

ROOT = Path(__file__).resolve().parents[2]
T0 = dt.datetime(2026, 11, 20, tzinfo=dt.UTC)
V2 = hub.LAYOUTS["v2"]
assert V2.records is not None
RECORDS = V2.records
MERGED_DIR = "artifacts/merged/ft2-bf16"
MERGED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
EXTRA = ("generation_config.json", "processor_config.json", "ggh_merge.json")
ADAPTER = ("adapter_config.json", "adapter_model.safetensors")
MERGE_RECORD = Path("docs/measurements/v2/merge_record.json")
# v1.0.0's bundle (v2 shipped), frozen as published (D-76)
REAL_BUNDLE = ROOT / "docs" / "results" / "releases" / "v1.0.0" / "results_bundle.json"


def put(root: Path, rel: str, data: bytes | None = None) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if data is not None else f"bytes of {rel}".encode())
    return sha256_file(p)


def stage(
    root: Path, name: str, copy: str, outputs: dict[str, str], parents: tuple[str, ...], where: Path
) -> StageManifest:
    m = StageManifest(
        stage=name,
        code_version="1",
        key=stage_key(name, "1", {}, {"copy": copy}, 7),
        inputs={},
        outputs=outputs,
        params={"copy": copy},
        seed=7,
        parents=parents,
        started_at=T0,
        finished_at=T0,
        host="test",
    )
    write_manifest(root / where, m)
    out = root / hub.LINEAGE_COPIES / f"{copy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    return m


def outs(root: Path, rels: list[str]) -> dict[str, str]:
    return {r: put(root, r) for r in rels}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    data = stage(tmp_path, "data", "data", outs(tmp_path, ["artifacts/data/train_pool.jsonl"]), (),
                 Path("artifacts/data"))  # fmt: skip
    targets = stage(tmp_path, "targets", "targets_v2",
                    outs(tmp_path, ["artifacts/targets/train.jsonl"]), (data.key,),
                    Path("artifacts/targets"))  # fmt: skip
    train_out = outs(tmp_path, [f"artifacts/adapter_v2/{f}" for f in ADAPTER])
    train = stage(tmp_path, "train", "train_v2", train_out, (targets.key,),
                  V2.train_manifest.parent)  # fmt: skip
    exp_out = outs(tmp_path, [f"{MERGED_DIR}/{f}" for f in MERGED])
    exp_out |= outs(tmp_path, [f"artifacts/gguf/ft2-{q}.gguf" for q in ("f16", "Q4_K_M", "Q8_0")])
    export = stage(tmp_path, "export", "export_ft2", exp_out, (train.key,),
                   V2.export_manifest.parent)  # fmt: skip
    stage(tmp_path, "parity", "parity_v2", outs(tmp_path, ["docs/measurements/v2/parity.json"]),
          (export.key, data.key), V2.verify_from.parent)  # fmt: skip
    put(tmp_path, f"{MERGED_DIR}/generation_config.json")
    put(tmp_path, f"{MERGED_DIR}/processor_config.json")
    put(tmp_path, MERGE_RECORD.as_posix(), b'{"merge": 1}\n')
    put(tmp_path, f"{MERGED_DIR}/ggh_merge.json", b'{"merge": 1}\n')
    for name in hub.LEGAL:
        shutil.copy(ROOT / name, tmp_path / name)

    def art(kind, model, path, sha, key, quant=None):
        return {"kind": kind, "model": model, "quant": quant, "path": path, "sha256": sha,
                "llama_cpp_commit": "b12421" if quant else None, "parent_key": key}  # fmt: skip

    records = [
        art("adapter", "ft", "artifacts/adapter_v2",
            train_out["artifacts/adapter_v2/adapter_model.safetensors"], train.key),
        art("merged_bf16", "ft", f"{MERGED_DIR}/model.safetensors",
            exp_out[f"{MERGED_DIR}/model.safetensors"], export.key),
        art("gguf", "base", "artifacts/gguf/base-Q4_K_M.gguf", "1" * 64, "2" * 64, "Q4_K_M"),
    ] + [
        art("gguf", "ft", f"artifacts/gguf/ft2-{q}.gguf", exp_out[f"artifacts/gguf/ft2-{q}.gguf"],
            export.key, q.upper())
        for q in ("f16", "Q4_K_M", "Q8_0")
    ]  # fmt: skip
    (tmp_path / RECORDS).write_text(json.dumps(records, indent=2), encoding="utf-8")
    bundle = json.loads(REAL_BUNDLE.read_text(encoding="utf-8"))
    assert bundle["releases"][-1]["version"] == "v2"
    bundle["releases"][-1]["artifacts"] = records
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    return tmp_path


def make_plan(root: Path, owner: str = "someone", **kw) -> hub.Plan:
    return hub.plan(root, Path("bundle.json"), "# card\n", owner=owner, **kw)


def test_the_plan_follows_the_shipped_release_by_default(repo):  # HUB-02, HUB-03
    p = make_plan(repo)
    assert p.problems == []
    assert p.ready
    assert (p.release, p.model) == ("v2", "gemma-e2b-grounded-v2-q4")
    assert sorted(f.path_in_repo for f in p.by_repo("main")) == sorted(
        [
            *MERGED,
            *EXTRA,
            *(f"adapter/{a}" for a in ADAPTER),
            "README.md",
            "results_bundle.json",
            "lineage.json",
            *hub.LEGAL,
        ]
    )
    ggufs = [f for f in p.by_repo("gguf") if f.path_in_repo.endswith(".gguf")]
    assert {f.path_in_repo: f.local for f in ggufs} == {
        "gemma-4-e2b-grounded-health-Q4_K_M.gguf": "artifacts/gguf/ft2-Q4_K_M.gguf",
        "gemma-4-e2b-grounded-health-Q8_0.gguf": "artifacts/gguf/ft2-Q8_0.gguf",
    }
    assert sorted(f.path_in_repo for f in p.by_repo("gguf")) == sorted(
        [f.path_in_repo for f in ggufs] + ["README.md", *hub.LEGAL]
    )
    assert not any("ft-" in f.local or "/adapter/" in f.local for f in p.files)  # nothing of v1


def test_every_hash_is_the_committed_record(repo):  # HUB-02
    p = make_plan(repo)
    export = read_manifest(repo / hub.LINEAGE_COPIES / "export_ft2.json")
    train = read_manifest(repo / hub.LINEAGE_COPIES / "train_v2.json")
    records = {
        r["path"]: r["sha256"] for r in json.loads((repo / RECORDS).read_text(encoding="utf-8"))
    }
    for f in p.files:
        if f.local.startswith(MERGED_DIR) and f.path_in_repo in MERGED:
            assert f.sha256 == export.outputs[f.local] == sha256_file(repo / f.local)
        if f.local.startswith("artifacts/adapter_v2/"):
            assert f.sha256 == train.outputs[f.local]
        if f.local in records:
            assert f.sha256 == records[f.local]
        if f.path_in_repo == "ggh_merge.json":
            assert f.sha256 == sha256_file(repo / MERGE_RECORD)
        if f.status != "generated":
            assert f.size == (repo / f.local).stat().st_size
    assert sum(f.local in records for f in p.files) == 3  # merged weights and two GGUFs
    assert {f.status for f in p.files} <= {"ok", "generated", "unrecorded"}
    unrecorded = sorted(f.path_in_repo for f in p.files if f.status == "unrecorded")
    assert unrecorded == ["generation_config.json", "processor_config.json"]


def test_lineage_is_verified_over_the_whole_v2_chain(repo):  # HUB-01
    p = make_plan(repo)
    assert p.lineage_verified == V2.verify_from.as_posix()
    content = next(f for f in p.files if f.path_in_repo == "lineage.json").content
    assert content is not None
    chain = json.loads(content)
    assert [m["stage"] for m in chain] == ["parity", "export", "train", "targets", "data"]


def test_a_changed_targets_file_blocks_the_release(repo):  # HUB-01: four stages up the chain
    (repo / "artifacts/targets/train.jsonl").write_bytes(b"v1's bytes")
    p = make_plan(repo)
    assert any(x.startswith("lineage: targets:") for x in p.problems)
    assert not p.ready


def test_a_changed_gguf_blocks_the_release(repo):
    (repo / "artifacts/gguf/ft2-Q8_0.gguf").write_bytes(b"tampered")
    p = make_plan(repo)
    assert "artifacts/gguf/ft2-Q8_0.gguf: hash mismatch" in p.problems
    assert not p.ready


def test_the_bundle_must_carry_the_v2_records(repo):
    recs = json.loads((repo / RECORDS).read_text(encoding="utf-8"))
    recs[-1]["sha256"] = "0" * 64
    (repo / RECORDS).write_text(json.dumps(recs), encoding="utf-8")
    assert any("export_artifacts_v2.json" in x for x in make_plan(repo).problems)


def test_a_live_manifest_other_than_its_committed_copy_is_refused(repo):
    copy = repo / hub.LINEAGE_COPIES / "train_v2.json"
    m = read_manifest(copy)
    other = m.model_copy(
        update={"params": {"copy": "x"}, "key": stage_key("train", "1", {}, {"copy": "x"}, 7)}
    )
    copy.write_text(other.model_dump_json(indent=2), encoding="utf-8")
    assert any("is not the committed lineage copy" in x for x in make_plan(repo).problems)


def test_a_card_other_than_the_committed_one_is_refused(repo):
    (repo / "MODEL_CARD.md").write_text("# an older card\n", encoding="utf-8")
    p = make_plan(repo, committed_card=Path("MODEL_CARD.md"))
    assert "MODEL_CARD.md is not the card generated from the bundle" in p.problems
    (repo / "MODEL_CARD.md").write_text("# card\n", encoding="utf-8")
    assert make_plan(repo, committed_card=Path("MODEL_CARD.md")).problems == []


def test_v1_stays_available(repo):
    p = make_plan(repo, layout="v1")  # this repo has only v2's stages
    assert p.release == "v1" and p.model == "gemma-e2b-grounded-q4"
    assert any("artifacts/export/ft/manifest.json" in x for x in p.problems) and not p.ready
    with pytest.raises(ValueError, match="no release layout"):
        make_plan(repo, layout="v9")


def test_gguf_card_links_the_main_repo_with_the_d63_note(repo):
    p = make_plan(repo)
    card = next(f for f in p.by_repo("gguf") if f.path_in_repo == "README.md").content
    assert card is not None
    text = card.decode()
    assert "Research only, not medical advice." in text
    assert "https://huggingface.co/someone/gemma-4-e2b-grounded-health" in text
    assert "OWNER D-63" in text and "Q4_K_M" in text
    assert "ollama run --think=false hf.co/someone/gemma-4-e2b-grounded-health-GGUF:Q4_K_M" in text
    for f in p.by_repo("gguf"):
        if f.path_in_repo.endswith(".gguf"):
            assert f.sha256 in text


def test_plan_json_lists_every_file_with_size_and_hash(repo):
    p = make_plan(repo)
    d = hub.plan_json(p)
    assert d["release"] == "v2" and d["problems"] == [] and d["ready"]
    assert len(d["files"]) == len(p.files)
    assert all(f["size"] and len(f["sha256"]) == 64 for f in d["files"])
    assert d["total_bytes"] == sum(f["size"] for f in d["files"])
    assert d["lineage_chain"][0].startswith("parity:")
    assert json.loads(json.dumps(d)) == d


def test_push_checks_every_download_and_records_the_answer(repo, tmp_path):  # HUB-04, HUB-05
    api = FakeHub()
    api.tmp = tmp_path / "dl"
    record: dict = {}
    problems = hub.push(
        make_plan(repo),
        repo,
        api,
        tag="v2.0.0",
        staging=tmp_path / "stage",
        ollama=lambda ref: '{"label": "no"}',
        record=record,
    )
    assert problems == []
    assert len(record["downloads"]) == len(make_plan(repo).files)
    assert all(x["ok"] for x in record["downloads"])
    assert record["ollama_answer"] == '{"label": "no"}'
    assert record["ollama_ref"] == "hf.co/someone/gemma-4-e2b-grounded-health-GGUF:Q4_K_M"
    assert "tag someone/gemma-4-e2b-grounded-health-GGUF v2.0.0" in api.calls


def test_push_refuses_a_bad_download_of_a_large_file(repo, tmp_path):  # HUB-04
    api = FakeHub(corrupt="gemma-4-e2b-grounded-health-Q4_K_M.gguf")
    api.tmp = tmp_path / "dl"
    record: dict = {}
    problems = hub.push(
        make_plan(repo), repo, api, tag="v2.0.0", staging=tmp_path / "s", record=record
    )
    assert problems == [
        "someone/gemma-4-e2b-grounded-health-GGUF/gemma-4-e2b-grounded-health-Q4_K_M.gguf: "
        "downloaded sha256 differs"
    ]
    assert [x["ok"] for x in record["downloads"]].count(False) == 1


def test_push_reports_a_failed_ollama_check(repo, tmp_path):  # HUB-05
    api = FakeHub()
    api.tmp = tmp_path / "dl"
    problems = hub.push(
        make_plan(repo), repo, api, tag="v2", staging=tmp_path / "s", ollama=lambda ref: None
    )
    assert problems == ["ollama did not answer the sample prompt"]


def test_push_refuses_a_plan_with_problems(repo, tmp_path):
    (repo / "artifacts/adapter_v2/adapter_config.json").unlink()
    api = FakeHub()
    with pytest.raises(hub.ReleaseError):
        hub.push(make_plan(repo), repo, api, tag="v2", staging=tmp_path / "s")
    assert api.calls == [] and api.files == {}


# CLI on the real repo: the default bundle is v1 + v2, so the plan is v2's


def test_cli_defaults_to_v2_and_never_pushes_on_a_dry_run(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(hub, "push", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setenv("HF_TOKEN", "secret-value-123")
    out_json = tmp_path / "plan.json"
    code = cli.main(["release", "--root", str(ROOT), "--owner", "x", "--plan-json", str(out_json)])
    out = capsys.readouterr().out
    assert "release v2 (gemma-e2b-grounded-v2-q4)" in out
    assert "dry run, nothing pushed" in out and "secret-value-123" not in out
    assert "artifacts/gguf/ft2-Q4_K_M.gguf" in out and "artifacts/gguf/ft-Q4_K_M.gguf" not in out
    assert json.loads(out_json.read_text(encoding="utf-8"))["release"] == "v2"
    assert code in (0, 2)  # 2 without artifacts/ (CI): the plan lists missing files


def test_cli_push_without_a_token_refuses(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub, "HfHubApi", lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert cli.main(["release", "--root", str(ROOT), "--push", "--owner", "x"]) == 2
    assert "needs HF_TOKEN" in capsys.readouterr().out


def test_cli_release_v1_flag_plans_v1(monkeypatch, capsys):
    code = cli.main(["release", "--root", str(ROOT), "--release", "v1", "--owner", "x"])
    out = capsys.readouterr().out
    assert "release v1 (gemma-e2b-grounded-q4)" in out
    assert "artifacts/gguf/ft-Q4_K_M.gguf" in out and "ft2-" not in out
    assert code == 2  # v1's targets lineage no longer matches artifacts/targets (v2's bytes)


def test_committed_plan_matches_the_records():
    """docs/measurements/release_plan_v2.json: every recorded file's sha256 is its record's."""
    d = json.loads((ROOT / "docs/measurements/release_plan_v2.json").read_text(encoding="utf-8"))
    assert d["release"] == "v2" and d["problems"] == [] and d["ready"]
    export = read_manifest(ROOT / hub.LINEAGE_COPIES / "export_ft2.json").outputs
    train = read_manifest(ROOT / hub.LINEAGE_COPIES / "train_v2.json").outputs
    records = {
        r["path"]: r["sha256"] for r in json.loads((ROOT / RECORDS).read_text(encoding="utf-8"))
    }
    produced = {**export, **train}
    seen = 0
    for f in d["files"]:
        if f["local"] in produced:
            assert f["sha256"] == produced[f["local"]]
            seen += 1
        if f["local"] in records:
            assert f["sha256"] == records[f["local"]]
        if f["local"] in ("LICENSE", "NOTICE"):
            assert f["sha256"] == sha256_file(ROOT / f["local"])
    assert seen == 9  # five merged files, two adapter files, two GGUFs
    card = next(f for f in d["files"] if f["path_in_repo"] == "README.md" and f["record"])
    # v1.0.0's card, frozen as published (D-76)
    assert card["sha256"] == sha256_file(ROOT / "docs/results/releases/v1.0.0/MODEL_CARD.md")
    assert {f["repo"] for f in d["files"]} == {
        "biplob123/gemma-4-e2b-grounded-health",
        "biplob123/gemma-4-e2b-grounded-health-GGUF",
    }


# HUB-05: the Ollama check with fake processes


class FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


def test_ollama_check_starts_a_server_and_parses_the_answer():
    calls: list[list[str]] = []
    state = {"up": False}
    proc = FakeProc()
    reply = '{"label": "no", "answer": "No.", "citations": [3]}'

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "list":
            return subprocess.CompletedProcess(cmd, 0 if state["up"] else 1, "", "")
        assert "beta-endorphin" in kw["input"]
        return subprocess.CompletedProcess(cmd, 0, reply + "\n", "")

    def popen(cmd, **kw):
        calls.append(cmd)
        state["up"] = True
        return proc

    r = hub.ollama_check("hf.co/o/r-GGUF:Q4_K_M", ROOT, binary="ollama", run=run, popen=popen,
                         sleep=lambda s: None)  # fmt: skip
    assert r.ok and r.started_server and proc.terminated
    assert r.answer == reply and r.sample_id
    assert ["ollama", "serve"] in calls
    assert ["ollama", "run", "--nowordwrap", "--think=false", "hf.co/o/r-GGUF:Q4_K_M"] in calls


def test_ollama_check_uses_a_running_server_and_fails_gracefully():
    def run(cmd, **kw):
        if cmd[1] == "list":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "pull model manifest: file does not exist")

    def popen(*a, **k):
        raise AssertionError("a running server is reused")

    r = hub.ollama_check("hf.co/o/r:Q4_K_M", ROOT, binary="ollama", run=run, popen=popen)
    assert not r.ok and not r.started_server and "does not exist" in (r.error or "")


def test_ollama_check_without_ollama_or_a_server(monkeypatch):
    monkeypatch.setattr(hub, "ollama_binary", lambda: None)
    r = hub.ollama_check(
        "x", ROOT, run=lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", "")
    )
    assert not r.ok and r.error == "ollama is not installed"
    proc = FakeProc()
    r = hub.ollama_check(
        "x",
        ROOT,
        binary="ollama",
        run=lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", ""),
        popen=lambda *a, **k: proc,
        sleep=lambda s: None,
        wait_s=3,
    )
    assert not r.ok and "did not answer" in (r.error or "") and proc.terminated
    assert dataclasses.asdict(r)["ref"] == "x"


def test_ollama_check_fails_when_the_answer_does_not_parse(tmp_path):
    """An answer that does not parse to a label with citations is a failed check, not ok
    (the first live run printed a thought block and was wrongly reported ok)."""
    import subprocess as _sp
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]

    def run(cmd, **kw):
        out = "<|channel>thought\nsome reasoning" if cmd[1] == "run" else ""
        return _sp.CompletedProcess(cmd, 0, stdout=out, stderr="")

    res = hub.ollama_check("hf.co/o/r-GGUF:Q4_K_M", root, binary="ollama", run=run)
    assert res.ok is False and res.error


def test_ollama_check_fails_on_out_of_range_citations():
    import subprocess as _sp
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    reply = '{"label": "no", "answer": "x", "citations": [99]}'

    def run(cmd, **kw):
        return _sp.CompletedProcess(cmd, 0, stdout=reply if cmd[1] == "run" else "", stderr="")

    res = hub.ollama_check("hf.co/o/r-GGUF:Q4_K_M", root, binary="ollama", run=run)
    assert res.ok is False
