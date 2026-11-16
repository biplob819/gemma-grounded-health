import datetime as dt
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from ggh.export import gguf
from ggh.serve import llama

ROOT = Path(__file__).resolve().parents[2]


def install(cache: Path, pin: llama.Pinned, files: dict[str, bytes], monkeypatch, attr: str):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{pin.folder}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    fake = llama.Pinned(pin.name, pin.url, hashlib.sha256(blob).hexdigest(), pin.folder)
    monkeypatch.setattr(llama, attr, fake)

    def download(url: str, dest: Path) -> None:
        dest.write_bytes(blob)

    llama.extract(fake, cache, download)
    return fake


@pytest.fixture
def tools(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    install(cache, llama.RELEASE, {"llama-quantize": b"q"}, monkeypatch, "RELEASE")
    install(cache, llama.SOURCE, {"convert_hf_to_gguf.py": b"c", "gguf-py/x": b"g"},
            monkeypatch, "SOURCE")  # fmt: skip
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path, cache, hf


def test_commands_convert_to_f16_then_quantize_each_quant():
    steps = gguf.commands(Path("/hf"), "base", Path("/o"), Path("/src"), Path("/q"), "py")
    assert steps[0] == (
        ["py", "/src/convert_hf_to_gguf.py", "/hf", "--outtype", "f16", "--outfile",
         "/o/base-f16.gguf"],
        {"PYTHONPATH": "/src/gguf-py"},
    )  # fmt: skip
    assert [s[0] for s in steps[1:]] == [
        ["/q", "/o/base-f16.gguf", "/o/base-Q4_K_M.gguf", "Q4_K_M"],
        ["/q", "/o/base-f16.gguf", "/o/base-Q8_0.gguf", "Q8_0"],
    ]


def test_export_runs_verified_tools_and_writes_a_hash_manifest(tools):
    root, cache, hf = tools
    ran, envs = [], []

    def run(cmd, env):
        ran.append(cmd)
        envs.append(env)
        out = Path(cmd[-1]) if cmd[-1].endswith(".gguf") else Path(cmd[-2])
        out.write_bytes(f"gguf {out.name}".encode())

    m = gguf.export(root, hf, "base", source_id="g/m", source_revision="abc", run=run,
                    cache=cache, python="py",
                    now=lambda: dt.datetime(2026, 11, 16, tzinfo=dt.UTC))  # fmt: skip
    assert ran[0][1] == str(cache / llama.SOURCE.folder / "convert_hf_to_gguf.py")
    assert ran[1][0] == str(cache / llama.RELEASE.folder / "llama-quantize")
    assert envs[0] == {"PYTHONPATH": str(cache / llama.SOURCE.folder / "gguf-py")}
    assert set(m["files"]) == {"base-f16.gguf", "base-Q4_K_M.gguf", "base-Q8_0.gguf"}
    for name, f in m["files"].items():
        data = (root / llama.GGUF_DIR / name).read_bytes()
        assert f == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert m["llama_cpp"]["tag"] == llama.TAG and m["source"] == {"id": "g/m", "revision": "abc"}
    on_disk = (root / llama.GGUF_DIR / "base.manifest.json").read_text(encoding="utf-8")
    assert on_disk == (root / gguf.MANIFEST_COPY / "gguf_base_manifest.json").read_text(
        encoding="utf-8"
    )
    assert json.loads(on_disk) == m


def test_export_refuses_a_missing_output_or_a_non_hf_folder(tools):
    root, cache, hf = tools
    with pytest.raises(llama.LlamaError, match="did not write"):
        gguf.export(root, hf, "base", source_id="g", source_revision="r",
                    run=lambda c, e: None, cache=cache)  # fmt: skip
    with pytest.raises(llama.LlamaError, match=r"config\.json"):
        gguf.export(root, root, "base", source_id="g", source_revision="r", cache=cache)


def test_export_refuses_unverified_tools(tools, monkeypatch):
    root, cache, hf = tools
    (cache / llama.RELEASE.name).write_bytes(b"swapped")
    with pytest.raises(llama.LlamaError, match="does not match the pin"):
        gguf.export(root, hf, "base", source_id="g", source_revision="r",
                    run=lambda c, e: None, cache=cache)  # fmt: skip


def test_committed_base_manifest_records_the_pinned_inputs():
    m = json.loads((ROOT / gguf.MANIFEST_COPY / "gguf_base_manifest.json").read_text("utf-8"))
    assert m["source"] == {"id": gguf.BASE_ID, "revision": gguf.BASE_REVISION}
    assert m["llama_cpp"] == {
        "tag": llama.TAG, "release_sha256": llama.BIN_SHA256, "source_sha256": llama.SRC_SHA256,
    }  # fmt: skip
    assert set(m["files"]) == {"base-f16.gguf", "base-Q4_K_M.gguf", "base-Q8_0.gguf"}
    q4 = ROOT / llama.GGUF_DIR / "base-Q4_K_M.gguf"
    if q4.exists():  # local artifacts: the file on disk is the one recorded
        assert m["files"]["base-Q4_K_M.gguf"]["bytes"] == q4.stat().st_size


def test_export_refuses_a_tampered_source_tarball(tools):
    root, cache, hf = tools
    (cache / llama.SOURCE.name).write_bytes(b"swapped")
    with pytest.raises(llama.LlamaError, match="does not match the pin"):
        gguf.export(root, hf, "base", source_id="g", source_revision="r",
                    run=lambda c, e: None, cache=cache)  # fmt: skip


def test_manifest_records_each_pin_in_its_own_field(tools):
    root, cache, hf = tools

    def run(cmd, env):
        out = Path(cmd[-1]) if cmd[-1].endswith(".gguf") else Path(cmd[-2])
        out.write_bytes(b"g")

    m = gguf.export(root, hf, "base", source_id="g", source_revision="r", run=run, cache=cache)
    assert m["llama_cpp"]["release_sha256"] == llama.BIN_SHA256
    assert m["llama_cpp"]["source_sha256"] == llama.SRC_SHA256 != llama.BIN_SHA256
