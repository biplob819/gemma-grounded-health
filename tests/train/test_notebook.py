import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ggh.train import kaggle, notebook

ROOT = Path(__file__).resolve().parents[2]
SHA = "eefb8a9ad8f67be68c7491bc29d0d82cb7dc7342"


def code(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_runs_the_pinned_steps_in_order_and_stops_on_failure():
    src = code(notebook.render(SHA, "unsloth", 300))
    order = [
        f"git checkout -q {SHA}",
        "requirements-kaggle-unsloth.txt",
        "ggh doctor --kaggle",
        'sh("ggh data")',
        "git diff --exit-code docs/measurements/data_manifest.json",
        "ggh targets --replay",
        "ggh train --config configs/recipe.yaml --out /kaggle/working/run --tool unsloth"
        " --max-steps 300",
    ]
    pos = [src.index(s) for s in order]
    assert pos == sorted(pos)
    assert "check=True" in src and "!" not in src.replace("!=", "")  # no bare shell lines
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in src
    full = code(notebook.render(SHA, "trl", None))
    assert "--max-steps" not in full and "requirements-kaggle-trl.txt" in full


def test_notebook_holds_no_secrets_or_outputs():
    nb = notebook.render(SHA, "trl", 300)
    assert all(c.get("outputs", []) == [] for c in nb["cells"])
    text = notebook.dumps(nb).lower().replace("tokenizers_parallelism", "")
    for word in ("token", "secret", "api_key", "hf_", "kaggle_key", "password", "getpass"):
        assert word not in text, word


def test_render_refuses_a_short_commit_or_an_unknown_tool():
    with pytest.raises(ValueError, match="40-character"):
        notebook.render("eefb8a9", "trl", 300)
    with pytest.raises(ValueError, match="unsloth or trl"):
        notebook.render(SHA, "axolotl", 300)


def test_committed_notebook_equals_its_template():
    path = ROOT / "notebooks" / "train_kaggle.ipynb"
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    src = code(nb)
    commit = src.split("git checkout -q ")[1][:40]
    tool = src.split("--tool ")[1].split()[0].strip('")')
    steps = int(src.split("--max-steps ")[1].split('"')[0]) if "--max-steps" in src else None
    assert text == notebook.dumps(notebook.render(commit, tool, steps))


def test_requirements_are_pinned_exactly():
    for tool in ("trl", "unsloth"):
        lines = [
            ln for ln in (ROOT / f"notebooks/requirements-kaggle-{tool}.txt")
            .read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")
        ]  # fmt: skip
        assert lines and all("==" in ln for ln in lines), tool
        assert any(ln.startswith("transformers==") for ln in lines)


def test_kernel_metadata_turns_on_a_t4_and_the_internet_and_stays_private():
    m = notebook.kernel_metadata("ggh-x", "train_kaggle.ipynb")
    assert m["enable_gpu"] is True and m["enable_internet"] is True and m["is_private"] is True
    assert m["machine_shape"] == "NvidiaTeslaT4" and m["id"] == "biplob819/ggh-x"


def disk(free_gb):
    return lambda p: SimpleNamespace(free=free_gb * 1e9)


def test_kaggle_doctor_needs_a_gpu_vram_disk_and_internet(tmp_path):
    ok, lines = kaggle.kaggle_report(tmp_path, gpu=lambda: ("Tesla T4", 15.6),
                                     disk=disk(60), online=lambda: True)  # fmt: skip
    assert ok and any("Tesla T4" in ln for ln in lines)
    for kw, why in (
        ({"gpu": lambda: None}, "no CUDA GPU"),
        ({"gpu": lambda: ("Tesla T4", 8.0)}, "gpu Tesla T4"),
        ({"disk": disk(5)}, "disk 5 GB"),
        ({"online": lambda: False}, "internet off"),
    ):
        args = {"gpu": lambda: ("Tesla T4", 15.6), "disk": disk(60), "online": lambda: True, **kw}
        ok, lines = kaggle.kaggle_report(tmp_path, **args)
        assert not ok and any(why in ln and "FAIL" in ln for ln in lines), why
    ok, lines = kaggle.kaggle_report(tmp_path, gpu=lambda: ("Tesla P100", 16.0),
                                     disk=disk(60), online=lambda: True)  # fmt: skip
    assert ok and any("not a T4" in ln for ln in lines)


def test_the_data_and_train_cells_are_exactly_these_commands():
    """Exact text, so a softened step (`|| true`, a dropped check) cannot slip through."""
    cells = ["".join(c["source"]) for c in notebook.render(SHA, "unsloth", 300)["cells"]]
    assert cells[3] == (
        "# rebuild N1.2 data and replay N1.3 targets; every committed hash is checked\n"
        'sh("ggh data")\n'
        'sh("git diff --exit-code docs/measurements/data_manifest.json")\n'
        'sh("ggh targets --replay")'
    )
    assert cells[4] == (
        'sh("ggh train --config configs/recipe.yaml --out /kaggle/working/run'
        ' --tool unsloth --max-steps 300")'
    )
    assert cells[1].count("check=True") == 1 and "|| true" not in cells[1] + cells[3] + cells[4]


def test_doctor_kaggle_failures_make_the_cli_exit_2(monkeypatch, capsys):
    from ggh import cli

    for ok, code in ((False, cli.EXIT_ERROR), (True, 0)):
        monkeypatch.setattr(kaggle, "kaggle_report", lambda work, ok=ok: (ok, ["[x] gpu"]))
        assert cli.main(["doctor", "--kaggle"]) == code
    monkeypatch.setattr(kaggle, "kaggle_report", lambda work: (False, ["[FAIL] no gpu"]))
    assert cli.main(["doctor"]) == 0  # without --kaggle the GPU check does not run


def test_the_full_run_adds_dev_eval_after_training_and_the_pilot_does_not():
    full = ["".join(c["source"]) for c in notebook.render(SHA, "unsloth", None)["cells"]]
    assert full[4] == (
        'sh("ggh train --config configs/recipe.yaml --out /kaggle/working/run --tool unsloth")'
    )
    assert full[5] == (
        "# §5.2: dev generations per checkpoint, in order of validation loss\n"
        'sh("ggh dev-eval --run /kaggle/working/run --out /kaggle/working/dev")'
    )
    pilot = code(notebook.render(SHA, "unsloth", 300))
    assert "dev-eval" not in pilot
