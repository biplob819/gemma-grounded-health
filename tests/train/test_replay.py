import gzip
import shutil

import pytest

from ggh import cli
from ggh.targets import make
from tests.targets.test_targets import FakeScorer
from tests.targets.test_targets_build import repo  # noqa: F401  (fixture)


def built(repo):  # noqa: F811
    make.build(repo, FakeScorer, seed=7, cap=200, chunk=100)
    make.write_scores_archive(repo)
    return repo


def copy_inputs(src, dst):
    for rel in ("artifacts/data", "data", "docs/measurements"):
        shutil.copytree(src / rel, dst / rel)


def test_the_score_archive_is_byte_stable(repo):  # noqa: F811
    built(repo)
    first = (repo / make.SCORES_ARCHIVE).read_bytes()
    make.write_scores_archive(repo)
    assert (repo / make.SCORES_ARCHIVE).read_bytes() == first  # fixed mtime, no file name
    assert first[4:8] == b"\x00\x00\x00\x00"  # gzip header mtime is zero
    raw = gzip.decompress(first)
    assert raw == (repo / make.OUT_DIR / "nli_scores.jsonl").read_bytes()


def test_replay_rebuilds_identical_targets_without_the_model(repo, tmp_path):  # noqa: F811
    built(repo)
    other = tmp_path / "kaggle"
    copy_inputs(repo, other)
    expected = make.expected_files(other)
    make.build(other, lambda: make.replay_scorer(other), seed=7, cap=200, chunk=100)
    assert make.replay_problems(other, expected) == []
    for name in make.REPLAYED:
        assert (other / make.OUT_DIR / name).read_bytes() == (
            repo / make.OUT_DIR / name
        ).read_bytes()


def test_replay_reports_a_hash_mismatch(repo, tmp_path):  # noqa: F811
    built(repo)
    other = tmp_path / "k"
    copy_inputs(repo, other)
    expected = make.expected_files(other)
    make.build(other, lambda: make.replay_scorer(other), seed=7, cap=200, chunk=100)
    (other / make.OUT_DIR / "train.jsonl").write_text("changed\n", encoding="utf-8")
    (problem,) = make.replay_problems(other, expected)
    assert problem.startswith("train.jsonl: sha256")


def test_replay_with_missing_scores_fails_rather_than_guessing(repo, tmp_path):  # noqa: F811
    built(repo)
    other = tmp_path / "k"
    copy_inputs(repo, other)
    with gzip.open(other / make.SCORES_ARCHIVE, "wt", encoding="utf-8") as fh:
        fh.write("")
    with pytest.raises(KeyError, match="no recorded NLI score"):
        make.build(other, lambda: make.replay_scorer(other), seed=7, cap=200, chunk=100)


def test_cli_replay_passes_and_fails_on_the_hashes(repo, tmp_path, monkeypatch, capsys):  # noqa: F811
    built(repo)
    other = tmp_path / "k"
    copy_inputs(repo, other)
    real = make.build
    monkeypatch.setattr(
        make, "build", lambda root, f, **k: real(root, f, seed=7, cap=200, chunk=100)
    )
    assert cli.main(["targets", "--root", str(other), "--replay"]) == 0
    report = other / make.REPORT_COPY
    text = report.read_text(encoding="utf-8")
    report.write_text(text.replace(make.expected_files(other)["val.jsonl"], "0" * 64),
                      encoding="utf-8")  # fmt: skip
    assert cli.main(["targets", "--root", str(other), "--replay"]) == cli.EXIT_ERROR
    assert "differ from the committed" in capsys.readouterr().out


def test_cli_train_reports_and_fails_on_non_finite_losses(monkeypatch, capsys):
    from ggh.train import run as R

    base = {"backend": "trl", "steps": 3, "tokens_per_s": 10.0, "peak_vram_gb": 0.0,
            "seconds": 1.0}  # fmt: skip
    for bad, code in ((0, 0), (2, cli.EXIT_ERROR)):
        monkeypatch.setattr(R, "run", lambda *a, bad=bad, **k: {**base, "non_finite": bad})
        assert cli.main(["train", "--out", "/tmp/x", "--max-steps", "3"]) == code
    assert "not finite" in capsys.readouterr().out

    seen = {}

    def spy(config, root, out, **k):
        seen.update(k)
        return {**base, "non_finite": 0}

    monkeypatch.setattr(R, "run", spy)
    cli.main(["train", "--out", "/tmp/x", "--max-steps", "300", "--tool", "trl"])
    assert seen == {"max_steps": 300, "tool": "trl"}
