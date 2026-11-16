import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ggh import cli
from ggh.data import build as data
from ggh.schemas import TrainRun
from ggh.train import devrun
from ggh.train import select as sel

ROOT = Path(__file__).resolve().parents[2]
GOOD = '{"label": "yes", "answer": "It worked [1].", "citations": [1]}'


def make_root(tmp_path, n=6):
    root = tmp_path / "repo"
    rows = [{"pmid": str(1000 + i), "label": "yes", "prompt": f"P{i}",
             "sentences": ["a", "b"]} for i in range(n)]  # fmt: skip
    p = root / devrun.DEV_EVAL
    p.parent.mkdir(parents=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    m = root / data.MANIFEST_PATH
    m.parent.mkdir(parents=True)
    m.write_text('{"x": 1}\n', encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs/recipe.yaml").write_bytes((ROOT / "configs/recipe.yaml").read_bytes())
    return root


def make_run(tmp_path, steps=((250, 0.8), (527, 0.7))):
    run = tmp_path / "run"
    for step, loss in steps:
        d = run / f"checkpoint-{step}"
        d.mkdir(parents=True)
        hist = [{"step": step, "eval_loss": loss}]
        (d / "trainer_state.json").write_text(json.dumps({"log_history": hist}), encoding="utf-8")
        (d / "adapter_config.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        (d / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
        (d / "optimizer.pt").write_bytes(b"not copied")
    return run


class Model:
    def __init__(self, path):
        self.path = path

    def get_output_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(262144, 8)))


def fakes(bad_steps=()) -> tuple[list, dict[str, Any]]:
    calls = []

    def load(cfg, path):
        calls.append(("load", path.name))
        return Model(path), "tok"

    def gen(model, tok, prompts, grammar=None):
        calls.append(("gen", model.path.name, grammar))
        if grammar is not None and model.path.name in bad_steps:
            return ["not json"] * len(prompts)
        return [GOOD] * len(prompts)

    def grammar_for(tok, vocab):
        assert vocab == 262144
        return "G"

    return calls, dict(load=load, gen=gen, grammar_for=grammar_for, free=lambda m: None)


def test_dev_eval_tries_the_best_loss_first_and_copies_only_the_adapter(tmp_path):
    root, run, out = make_root(tmp_path), make_run(tmp_path), tmp_path / "dev"
    calls, kw = fakes()
    rec = devrun.dev_eval(root, run, out, root / "configs/recipe.yaml", **kw)
    assert rec["selected_step"] == 527 and rec["n_prompts"] == 6
    assert calls[0] == ("load", "checkpoint-527")
    assert ("gen", "checkpoint-527", "G") in calls and ("gen", "checkpoint-527", None) in calls
    assert not any(c[1] == "checkpoint-250" for c in calls)  # never needed
    assert sorted(p.name for p in (out / "adapter").iterdir()) == list(sel.ADAPTER_FILES)
    assert rec["adapter_sha256"] == sel.adapter_hash(out / "adapter")
    gens = [json.loads(x) for x in (out / "generations.jsonl").read_text().splitlines()]
    assert len(gens) == 6 and gens[0]["step"] == 527 and gens[0]["constrained"] == GOOD
    assert json.loads((out / "selection.json").read_text()) == rec


def test_dev_eval_falls_back_to_the_next_checkpoint_and_records_why(tmp_path):
    root, run, out = make_root(tmp_path), make_run(tmp_path), tmp_path / "dev"
    _, kw = fakes(bad_steps=("checkpoint-527",))
    rec = devrun.dev_eval(root, run, out, root / "configs/recipe.yaml", **kw)
    assert rec["selected_step"] == 250
    assert rec["tried"][0]["step"] == 527 and rec["tried"][0]["failed"][0].startswith("schema")


def test_dev_eval_with_no_passing_checkpoint_selects_nothing(tmp_path):
    root, run, out = make_root(tmp_path), make_run(tmp_path), tmp_path / "dev"
    _, kw = fakes(bad_steps=("checkpoint-527", "checkpoint-250"))
    rec = devrun.dev_eval(root, run, out, root / "configs/recipe.yaml", **kw)
    assert rec["selected_step"] is None and rec["adapter_sha256"] is None
    assert not (out / "adapter").exists()


REPORT = {"config_sha256": "c" * 64, "gpu": "Tesla T4", "seconds": 7000.0, "peak_vram_gb": 9.8,
          "steps": 527, "eval_losses": [[250, 0.8], [500, 0.72], [527, 0.7]]}  # fmt: skip


def selected(tmp_path):
    root, run, out = make_root(tmp_path), make_run(tmp_path), tmp_path / "dev"
    _, kw = fakes()
    rec = devrun.dev_eval(root, run, out, root / "configs/recipe.yaml", **kw)
    return root, rec, out / "adapter"


def test_freeze_writes_a_valid_trainrun_after_checking_the_adapter(tmp_path):
    root, rec, adapter = selected(tmp_path)
    run = devrun.freeze(root, REPORT, rec, adapter, kernel_version="3", repo_commit="a" * 40)
    saved = TrainRun.model_validate_json((root / devrun.TRAIN_RUN).read_text(encoding="utf-8"))
    assert saved == run and run.selected_step == 527 and run.kaggle_kernel_version == "3"
    assert run.data_manifest_hash == devrun.sha256_file(root / data.MANIFEST_PATH)
    assert run.dev_metrics.schema_valid == 1.0 and run.val_loss_curve[-1] == (527, 0.7)


def test_freeze_refuses_a_changed_adapter_or_no_selection(tmp_path):
    root, rec, adapter = selected(tmp_path)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(devrun.FreezeError, match="differs from the recorded"):
        devrun.freeze(root, REPORT, rec, adapter, kernel_version="3", repo_commit="a" * 40)
    with pytest.raises(devrun.FreezeError, match="nothing to freeze"):
        devrun.freeze(root, REPORT, {**rec, "selected_step": None}, adapter,
                      kernel_version="3", repo_commit="a" * 40)  # fmt: skip
    assert not (root / devrun.TRAIN_RUN).exists()


def test_cli_dev_eval_and_freeze_exit_codes(tmp_path, monkeypatch, capsys):
    root, srec, adapter = selected(tmp_path)  # the real dev_eval, before it is faked below
    tried = [{"step": 527, "eval_loss": 0.7, "passed": False, "failed": ["x"]}]
    rec = {"selected_step": None, "adapter_sha256": None, "tried": tried}
    monkeypatch.setattr(devrun, "dev_eval", lambda *a, **k: rec)
    assert cli.main(["dev-eval", "--run", "r", "--out", "o"]) == cli.EXIT_ERROR
    monkeypatch.setattr(devrun, "dev_eval", lambda *a, **k: {**rec, "selected_step": 527,
                                                             "adapter_sha256": "ab"})  # fmt: skip
    assert cli.main(["dev-eval", "--run", "r", "--out", "o"]) == 0
    (tmp_path / "rep.json").write_text(json.dumps(REPORT), encoding="utf-8")
    (tmp_path / "sel.json").write_text(json.dumps(srec), encoding="utf-8")
    args = ["freeze", "--report", str(tmp_path / "rep.json"), "--selection",
            str(tmp_path / "sel.json"), "--adapter", str(adapter), "--kernel-version", "3",
            "--repo-commit", "a" * 40, "--root", str(root)]  # fmt: skip
    assert cli.main(args) == 0
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert cli.main(args) == cli.EXIT_ERROR
