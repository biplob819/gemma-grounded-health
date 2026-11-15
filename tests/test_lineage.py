import datetime as dt
import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ggh import cli
from ggh.lineage import (
    MANIFEST,
    LineageError,
    StageManifest,
    read_manifest,
    sha256_file,
    stage_key,
    verify,
    write_manifest,
)

T0 = dt.datetime(2026, 11, 15, 12, 0, tzinfo=dt.UTC)
H1, H2 = "1" * 64, "2" * 64


def manifest(
    stage, outputs, parents=(), inputs=None, params=None, seed=7, code="1"
) -> StageManifest:
    inputs = inputs or {}
    params = params or {}
    return StageManifest(
        stage=stage,
        code_version=code,
        key=stage_key(stage, code, inputs, params, seed),
        inputs=inputs,
        outputs=outputs,
        params=params,
        seed=seed,
        parents=tuple(parents),
        started_at=T0,
        finished_at=T0 + dt.timedelta(minutes=1),
        host="test",
    )


def stage(root: Path, name: str, content: str, parents=()) -> tuple[StageManifest, Path]:
    out = root / "artifacts" / name / "out.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    m = manifest(name, {f"artifacts/{name}/out.txt": sha256_file(out)}, parents, params={"n": name})
    return m, write_manifest(out.parent, m)


json_scalars = st.one_of(st.integers(), st.text(max_size=5), st.booleans(), st.none())


@given(st.dictionaries(st.text(min_size=1, max_size=5), json_scalars, max_size=6))
def test_stage_key_is_stable_under_key_order(params):
    reordered = dict(reversed(list(params.items())))
    inputs = {"b.txt": H2, "a.txt": H1}
    assert stage_key("s", "1", inputs, params, 7) == stage_key(
        "s", "1", dict(reversed(list(inputs.items()))), reordered, 7
    )


def test_stage_key_changes_with_every_part():
    base = stage_key("data", "1", {"a": H1}, {"n": 1}, 7)
    variants = [
        stage_key("targets", "1", {"a": H1}, {"n": 1}, 7),
        stage_key("data", "2", {"a": H1}, {"n": 1}, 7),
        stage_key("data", "1", {"a": H2}, {"n": 1}, 7),
        stage_key("data", "1", {"a": H1}, {"n": 2}, 7),
        stage_key("data", "1", {"a": H1}, {"n": 1}, 8),
    ]
    assert base not in variants and len(set(variants)) == 5
    # inputs are (path, hash) pairs (D-32): swapping two inputs' contents changes the key
    swap_a = stage_key("data", "1", {"train": H1, "val": H2}, {}, 7)
    swap_b = stage_key("data", "1", {"train": H2, "val": H1}, {}, 7)
    assert swap_a != swap_b


def test_manifest_rejects_a_wrong_key_bad_times_and_self_parent():
    m = manifest("data", {"x": H1})
    with pytest.raises(ValidationError, match="key does not match"):
        StageManifest.model_validate({**m.model_dump(), "seed": 8})
    with pytest.raises(ValidationError, match="before started_at"):
        StageManifest.model_validate({**m.model_dump(), "finished_at": T0 - dt.timedelta(1)})
    with pytest.raises(ValidationError, match="own parent"):
        StageManifest.model_validate({**m.model_dump(), "parents": [m.key]})
    with pytest.raises(ValueError):  # refused while computing the key, before validation
        manifest("data", {"x": H1}, params={"bad": float("nan")})


def test_write_is_atomic_when_the_rename_fails(tmp_path, monkeypatch):
    m = manifest("data", {"x": H1})

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_manifest(tmp_path, m)
    assert list(tmp_path.iterdir()) == []  # no manifest and no leftover temp file


def test_write_then_read_round_trips(tmp_path):
    m = manifest("data", {"x": H1}, params={"nested": {"k": [1, 2]}})
    path = write_manifest(tmp_path, m)
    assert path.name == MANIFEST and read_manifest(path) == m


def test_read_of_a_corrupt_manifest_is_a_lineage_error(tmp_path):
    (tmp_path / MANIFEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(LineageError):
        read_manifest(tmp_path / MANIFEST)


def test_verify_passes_an_intact_chain(tmp_path):
    a, _ = stage(tmp_path, "data", "rows")
    b, _ = stage(tmp_path, "targets", "targets", parents=[a.key])
    _, cpath = stage(tmp_path, "train", "adapter", parents=[b.key])
    assert verify(cpath, tmp_path) == []


def test_verify_fails_on_a_missing_parent(tmp_path):
    _, path = stage(tmp_path, "targets", "t", parents=[H1])
    assert any("has no manifest" in p for p in verify(path, tmp_path))


def test_verify_fails_when_a_parent_output_changed(tmp_path):
    a, _ = stage(tmp_path, "data", "rows")
    _, bpath = stage(tmp_path, "targets", "t", parents=[a.key])
    (tmp_path / "artifacts" / "data" / "out.txt").write_text("edited", encoding="utf-8")
    assert any("data: output" in p and "no longer matches" in p for p in verify(bpath, tmp_path))


def test_verify_fails_when_an_output_is_missing(tmp_path):
    _, path = stage(tmp_path, "data", "rows")
    (tmp_path / "artifacts" / "data" / "out.txt").unlink()
    assert any("is missing" in p for p in verify(path, tmp_path))


def test_verify_fails_on_a_cycle(tmp_path):
    a = manifest("a", {}, params={"n": "a"})
    b = manifest("b", {}, parents=[a.key], params={"n": "b"})
    a_cyclic = manifest("a", {}, parents=[b.key], params={"n": "a"})  # same key as a
    write_manifest(tmp_path / "artifacts" / "a", a_cyclic)
    bpath = write_manifest(tmp_path / "artifacts" / "b", b)
    assert any("cycle" in p for p in verify(bpath, tmp_path))


def test_lineage_cli_exit_codes(tmp_path, capsys):
    _, apath = stage(tmp_path, "data", "rows")
    assert cli.main(["lineage", "verify", str(apath), "--root", str(tmp_path)]) == 0
    assert "lineage: OK" in capsys.readouterr().out
    _, bad = stage(tmp_path, "targets", "t", parents=[H2])
    assert cli.main(["lineage", "verify", str(bad), "--root", str(tmp_path)]) == cli.EXIT_ERROR
    assert "lineage: FAIL" in capsys.readouterr().out


def test_verify_judges_only_the_chain(tmp_path):
    a, _ = stage(tmp_path, "data", "rows")
    _, bpath = stage(tmp_path, "targets", "t", parents=[a.key])
    web = tmp_path / "demo" / "web"
    web.mkdir(parents=True)
    (web / MANIFEST).write_text('{"name": "GGH demo", "start_url": "/"}', encoding="utf-8")
    assert verify(bpath, tmp_path) == []  # an unrelated manifest.json is not lineage


def test_verify_flags_a_duplicate_key_only_on_the_chain(tmp_path):
    import shutil

    a, _ = stage(tmp_path, "data", "rows")
    _, bpath = stage(tmp_path, "targets", "t", parents=[a.key])
    shutil.copytree(tmp_path / "artifacts" / "data", tmp_path / "backup" / "data")
    assert any("several manifests" in p for p in verify(bpath, tmp_path))
    _, opath = stage(tmp_path, "other", "x")
    assert verify(opath, tmp_path) == []  # the duplicate is not on this chain


def test_verify_catches_a_parent_edited_and_rehashed_after_the_child_ran(tmp_path):
    a, apath = stage(tmp_path, "data", "rows")
    out = "artifacts/data/out.txt"
    child = manifest("targets", {}, parents=[a.key], inputs={out: a.outputs[out]})
    cpath = write_manifest(tmp_path / "artifacts" / "targets", child)
    assert verify(cpath, tmp_path) == []
    (tmp_path / out).write_text("edited", encoding="utf-8")  # edit the parent output...
    rehashed = manifest("data", {out: sha256_file(tmp_path / out)}, params={"n": "data"})
    write_manifest(apath.parent, rehashed)  # ...and rewrite its manifest to match
    child2 = manifest("targets", {}, parents=[rehashed.key], inputs={out: a.outputs[out]})
    c2 = write_manifest(tmp_path / "artifacts" / "targets", child2)
    assert any("does not match what data recorded" in p for p in verify(c2, tmp_path))


@pytest.mark.parametrize(
    "rel",
    [
        "/etc/hosts",
        "../outside.txt",
        "./artifacts/x",
        "artifacts//x",
        "artifacts/./x",
        "artifacts\\x",
        "artifacts/x/",
        "",
        "a/../b",
    ],
)
def test_manifest_paths_have_one_canonical_spelling(rel):
    with pytest.raises(ValidationError, match="canonical relative path"):
        manifest("s", {rel: H1})
    with pytest.raises(ValidationError, match="canonical relative path"):
        manifest("s", {}, inputs={rel: H1})


def test_inputs_may_name_outside_sources_by_uri():
    m = manifest("data", {}, inputs={"hf://qiaojin/PubMedQA@9001f28/pqa_artificial.parquet": H1})
    assert next(iter(m.inputs)).startswith("hf://")
    with pytest.raises(ValidationError):
        manifest("data", {"hf://x/y": H1})  # outputs are always repo paths


@pytest.mark.parametrize("link", ["file", "dir"])
def test_verify_refuses_any_symlinked_component_inside_the_root(tmp_path, link):
    run = tmp_path / "artifacts" / "run3"
    run.mkdir(parents=True)
    (run / "out.txt").write_text("x", encoding="utf-8")
    if link == "dir":
        (tmp_path / "artifacts" / "latest").symlink_to(run, target_is_directory=True)
        rel = "artifacts/latest/out.txt"
    else:
        (tmp_path / "artifacts" / "link.txt").symlink_to(run / "out.txt")
        rel = "artifacts/link.txt"
    path = write_manifest(
        tmp_path / "artifacts" / "s", manifest("s", {rel: sha256_file(run / "out.txt")})
    )
    assert any("through a symlink" in p for p in verify(path, tmp_path))


def test_source_inputs_no_parent_produced_must_still_match(tmp_path):
    src = tmp_path / "data" / "offscope_eval.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("v1", encoding="utf-8")
    rel = "data/offscope_eval.jsonl"
    path = write_manifest(
        tmp_path / "artifacts" / "s", manifest("s", {}, inputs={rel: sha256_file(src)})
    )
    assert verify(path, tmp_path) == []
    src.write_text("v2", encoding="utf-8")
    assert any("changed since s ran" in p for p in verify(path, tmp_path))
    src.unlink()
    assert any("missing" in p for p in verify(path, tmp_path))


def test_duplicate_parent_in_a_dot_folder_is_still_reported(tmp_path):
    import shutil

    a, _ = stage(tmp_path, "data", "rows")
    _, bpath = stage(tmp_path, "targets", "t", parents=[a.key])
    shutil.copytree(tmp_path / "artifacts" / "data", tmp_path / ".backup" / "data")
    assert any("several manifests" in p for p in verify(bpath, tmp_path))
    shutil.rmtree(tmp_path / ".backup")
    shutil.copytree(tmp_path / "artifacts" / "data", tmp_path / ".venv" / "data")
    assert verify(bpath, tmp_path) == []  # virtualenvs and .git are never lineage


@pytest.mark.parametrize(
    "uri", ["file://artifacts/data/out.txt", "s3://bucket/x", "repo://artifacts/x", "https://x/y"]
)
def test_only_hf_and_cache_uris_name_inputs(uri):
    with pytest.raises(ValidationError, match="canonical relative path"):
        manifest("s", {}, inputs={uri: H1})


def test_a_symlink_in_a_source_input_path_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.jsonl").write_text("v", encoding="utf-8")
    (tmp_path / "data").symlink_to(real, target_is_directory=True)
    path = write_manifest(
        tmp_path / "artifacts" / "s",
        manifest("s", {}, inputs={"data/f.jsonl": sha256_file(real / "f.jsonl")}),
    )
    assert any(
        "source input data/f.jsonl is missing or a symlink" in p for p in verify(path, tmp_path)
    )


def test_nul_in_a_manifest_path_is_refused():
    with pytest.raises(ValidationError, match="canonical relative path"):
        manifest("s", {"artifacts/a\x00b": H1})


def test_unreadable_source_input_is_reported_not_raised(tmp_path):
    src = tmp_path / "data" / "s.jsonl"
    src.parent.mkdir()
    src.write_text("v", encoding="utf-8")
    path = write_manifest(
        tmp_path / "artifacts" / "s", manifest("s", {}, inputs={"data/s.jsonl": sha256_file(src)})
    )
    src.chmod(0)
    try:
        if not os.access(src, os.R_OK):
            assert any("cannot read source input" in p for p in verify(path, tmp_path))
    finally:
        src.chmod(0o644)
