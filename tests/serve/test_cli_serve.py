import json
import shutil
from pathlib import Path

from ggh import cli
from ggh.serve import llama

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "evals" / "fixtures"


def test_e31_commands_are_registered():
    names = set(cli.build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    assert {
        "llama",
        "export",
        "serve",
        "serve-check",
        "tasks",
        "evalkit-check",
        "behaviour",
    } <= names


def test_serve_reports_a_busy_port_and_exits_2(monkeypatch, capsys):
    def busy(*a, **k):
        raise llama.LlamaError("port 8080 is already in use; stop that server first")

    monkeypatch.setattr(llama, "start", busy)
    assert cli.main(["serve", "--model", "base"]) == cli.EXIT_ERROR
    assert "already in use" in capsys.readouterr().out


def test_x31_commands_are_registered():
    names = set(cli.build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    assert {"merge", "parity", "export-records"} <= names


def _merged(root, adapter_sha):
    from ggh.export import merge

    (root / merge.SELECTION).parent.mkdir(parents=True, exist_ok=True)
    (root / merge.SELECTION).write_text(json.dumps({"adapter_sha256": "ab" * 32}), encoding="utf-8")
    d = root / merge.MERGED_DIR
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "ggh_merge.json").write_text(json.dumps({"adapter_sha256": adapter_sha}), encoding="utf-8")


def test_export_ft_converts_the_merged_folder_named_by_the_frozen_adapter(
    tmp_path, monkeypatch, capsys
):
    from ggh.export import gguf, merge

    _merged(tmp_path, "ab" * 32)
    seen = {}

    def fake_export(root, hf, name, *, source_id, source_revision):
        seen.update(hf=hf, name=name, sid=source_id, srev=source_revision)
        return {"files": {"ft-Q4_K_M.gguf": {"bytes": 1, "sha256": "c" * 64}}}

    monkeypatch.setattr(gguf, "export", fake_export)
    assert cli.main(["export", "--model", "ft", "--root", str(tmp_path)]) == 0
    assert seen == {"hf": tmp_path / merge.MERGED_DIR, "name": "ft",
                    "sid": "google/gemma-4-E2B-it+adapter", "srev": "ab" * 32}  # fmt: skip


def test_export_ft_refuses_a_folder_merged_from_another_adapter(tmp_path, monkeypatch, capsys):
    from ggh.export import gguf

    _merged(tmp_path, "cd" * 32)
    monkeypatch.setattr(gguf, "export", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert cli.main(["export", "--model", "ft", "--root", str(tmp_path)]) == cli.EXIT_ERROR
    assert "another adapter" in capsys.readouterr().out


def test_merge_uses_the_frozen_hash_and_reports_a_refusal(tmp_path, monkeypatch, capsys):
    from ggh.export import gguf, merge

    (tmp_path / merge.SELECTION).parent.mkdir(parents=True)
    (tmp_path / merge.SELECTION).write_text(
        json.dumps({"adapter_sha256": "ab" * 32}), encoding="utf-8"
    )
    monkeypatch.setattr(gguf, "hf_snapshot", lambda i, r: tmp_path / "base")
    seen = {}

    def fake(base, adapter, out, *, expected_sha256):
        seen.update(base=base, adapter=adapter, out=out, sha=expected_sha256)
        return {"lora_layers_merged": 205}

    monkeypatch.setattr(merge, "merge", fake)
    assert cli.main(["merge", "--root", str(tmp_path)]) == 0
    assert seen == {"base": tmp_path / "base", "adapter": tmp_path / merge.ADAPTER_DIR,
                    "out": tmp_path / merge.MERGED_DIR, "sha": "ab" * 32}  # fmt: skip
    assert "205 LoRA layers" in capsys.readouterr().out

    def refuse(*a, **k):
        raise merge.MergeError("the base is quantized; merge only into the bf16 base on CPU")

    monkeypatch.setattr(merge, "merge", refuse)
    assert cli.main(["merge", "--root", str(tmp_path)]) == cli.EXIT_ERROR
    assert "quantized" in capsys.readouterr().out


def test_parity_exits_by_the_bar(tmp_path, monkeypatch, capsys):
    from ggh.export import parity
    from ggh.schemas import ParityReport
    from ggh.train.devrun import DEV_EVAL

    (tmp_path / DEV_EVAL).parent.mkdir(parents=True)
    (tmp_path / DEV_EVAL).write_text(json.dumps({"pmid": 1, "label": "yes", "prompt": "p"}) + "\n",
                                     encoding="utf-8")  # fmt: skip
    for agree, code in ((0.96, 0), (0.94, cli.EXIT_ERROR)):
        rep = ParityReport(n=500, label_agreement=agree, citation_exact_match=0.5,
                           schema_valid={"bf16": 1.0, "q4_k_m": 1.0})  # fmt: skip
        monkeypatch.setattr(parity, "run", lambda *a, rep=rep, **k: rep)
        assert cli.main(["parity", "--root", str(tmp_path)]) == code
        assert ("PASS" in capsys.readouterr().out) == (code == 0)


def test_evalkit_check_passes_on_the_repo_and_reports_reuse(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
    shutil.copy(FIX / "record.json", run / "records.jsonl")
    code = cli.main(["evalkit-check", "--root", str(ROOT), "--reference", str(run),
                     "--reference-config", str(ROOT / "configs" / "evalkit.yaml")])  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["config_problems"] == []
    assert out["reuse_claimable"] is False  # the fixture shares no frontier generator


def test_behaviour_reads_an_evalkit_run_folder(tmp_path, capsys):
    shutil.copy(FIX / "record.json", tmp_path / "records.jsonl")
    shutil.copy(FIX / "case.json", tmp_path / "cases.jsonl")
    assert cli.main(["behaviour", str(tmp_path)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["gemma-e2b-base-q4"]["B1"]["n"] == 1


def test_tasks_writes_into_the_given_root(tmp_path, monkeypatch, capsys):
    from ggh.evals import tasks

    monkeypatch.setattr(tasks, "files", lambda root: {Path("configs/tasks/x.yaml"): "a: 1\n"})
    assert cli.main(["tasks", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "configs/tasks/x.yaml").read_text(encoding="utf-8") == "a: 1\n"


def test_serve_check_exits_2_when_a_check_fails_and_writes_the_report(tmp_path, monkeypatch):
    from ggh.serve import checks

    monkeypatch.setattr(checks, "evalkit_complete", lambda base: None)
    monkeypatch.setattr(checks, "hf_tokenizer", lambda: None)
    for ok in (False, True):
        monkeypatch.setattr(
            checks, "run", lambda *a, ok=ok, **k: {"passed": ok, "summary": {"n": 20}}
        )
        (tmp_path / "docs" / "measurements").mkdir(parents=True, exist_ok=True)
        code = cli.main(["serve-check", "--root", str(tmp_path)])
        assert code == (0 if ok else cli.EXIT_ERROR)
        assert json.loads((tmp_path / checks.OUT).read_text(encoding="utf-8"))["passed"] is ok
    assert cli.main(["serve-check", "--root", str(tmp_path), "--model", "ft"]) == 0
    assert (tmp_path / checks.out_path("ft")).name == "serving_checks_ft.json"
    assert (tmp_path / checks.out_path("ft")).is_file()  # the base's E3.1 file is untouched
    assert checks.out_path("base") == checks.OUT


def test_evalkit_check_exits_2_on_config_problems(tmp_path, capsys):
    root = tmp_path
    for rel in ("configs/evalkit.yaml", "configs/evalkit_dev.yaml", "pyproject.toml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    p = root / "configs/evalkit.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("seed: 7", "seed: 8"), encoding="utf-8")
    assert cli.main(["evalkit-check", "--root", str(root)]) == cli.EXIT_ERROR
    assert "seed" in capsys.readouterr().out


def test_evalkit_check_needs_the_reference_config(capsys):
    code = cli.main(["evalkit-check", "--root", str(ROOT), "--reference", "x"])
    assert code == cli.EXIT_ERROR and "--reference-config" in capsys.readouterr().out


def test_evalkit_check_exits_2_on_a_pin_problem(monkeypatch, capsys):
    from ggh.evals import config

    monkeypatch.setattr(config, "installed_commit", lambda: "0" * 40)
    assert cli.main(["evalkit-check", "--root", str(ROOT)]) == cli.EXIT_ERROR
    assert "installed EvalKit" in capsys.readouterr().out


def test_evalkit_check_compares_the_pinned_commit(tmp_path, monkeypatch, capsys):
    from ggh.evals import config

    seen = {}
    real = config.ours

    def spy(cfg, sha):
        seen["sha"] = sha
        return real(cfg, sha)

    monkeypatch.setattr(config, "ours", spy)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
    shutil.copy(FIX / "record.json", run / "records.jsonl")
    cli.main(["evalkit-check", "--root", str(ROOT), "--reference", str(run),
              "--reference-config", str(ROOT / "configs" / "evalkit.yaml")])  # fmt: skip
    assert seen["sha"] == config.pinned_commit(ROOT)
    out = json.loads(capsys.readouterr().out)
    # the fixture run stamped the pinned commit, so the version is not a blocker; passing any
    # other string (e.g. "unknown") would make it one
    assert json.loads((FIX / "record.json").read_text(encoding="utf-8"))["git_sha"] == seen["sha"]
    assert not any(p.startswith("evalkit:") for p in out["reuse_problems"])


def test_serve_throughput_writes_the_throughput_file_for_the_chosen_model(tmp_path, monkeypatch):
    from ggh.serve import checks

    seen = {}
    monkeypatch.setattr(checks, "evalkit_async_client", lambda base: seen.setdefault("base", base))

    def fake(root, model, quant, new_client, host=""):
        seen.update(model=model, quant=quant)
        return {"runs": [], "seconds_per_answer": 1.0}

    monkeypatch.setattr(checks, "throughput", fake)
    (tmp_path / "docs" / "measurements").mkdir(parents=True)
    assert cli.main(["serve-throughput", "--root", str(tmp_path), "--model", "base"]) == 0
    assert seen == {"base": "http://127.0.0.1:8080/v1", "model": "base", "quant": "Q4_K_M"}
    assert json.loads((tmp_path / checks.THROUGHPUT).read_text(encoding="utf-8"))["runs"] == []
    assert not (tmp_path / checks.OUT).exists()


def test_serve_check_talks_to_the_chosen_models_port(tmp_path, monkeypatch):
    from ggh.serve import checks

    seen = []
    monkeypatch.setattr(checks, "evalkit_complete", lambda base: seen.append(base))
    monkeypatch.setattr(checks, "hf_tokenizer", lambda: None)
    monkeypatch.setattr(checks, "run", lambda *a, **k: {"passed": True, "summary": {}})
    (tmp_path / "docs" / "measurements").mkdir(parents=True)
    cli.main(["serve-check", "--root", str(tmp_path), "--model", "base"])
    cli.main(["serve-check", "--root", str(tmp_path), "--model", "ft"])
    assert seen == ["http://127.0.0.1:8080/v1", "http://127.0.0.1:8081/v1"]


def test_serve_passes_a_log_path_to_the_server(tmp_path, monkeypatch, capsys):
    seen = {}

    class Done:
        def wait(self):
            return 0

    def start(root, model, quant, port, log=None):
        seen["log"] = log
        return Done()

    monkeypatch.setattr(llama, "start", start)
    assert cli.main(["serve", "--root", str(tmp_path)]) == 0
    assert seen["log"] == tmp_path / llama.GGUF_DIR / "serve-base.log"
    assert "server log:" in capsys.readouterr().out


def test_behaviour_exits_2_on_a_folder_without_records(tmp_path, capsys):
    assert cli.main(["behaviour", str(tmp_path)]) == cli.EXIT_ERROR
    assert "behaviour: FAIL" in capsys.readouterr().out
