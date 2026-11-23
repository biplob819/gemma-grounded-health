import hashlib
import io
import socket
import subprocess
import tarfile
from pathlib import Path

import pytest

from ggh.serve import llama

ROOT = Path(__file__).resolve().parents[2]


def tarball(folder: str, files: dict[str, bytes], extra: tarfile.TarInfo | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{folder}/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
        if extra is not None:
            tf.addfile(extra, io.BytesIO(b""))
    return buf.getvalue()


def pin_for(blob: bytes, folder: str = "llama-test") -> llama.Pinned:
    sha = hashlib.sha256(blob).hexdigest()
    return llama.Pinned("t.tar.gz", "https://example.invalid/t", sha, folder)


def serving(blob: bytes):
    def download(url, dest):
        dest.write_bytes(blob)

    return download


def no_download(url, dest):
    raise AssertionError("must not download")


# ---- EVK-01: fetch and verify ----


def test_pins_match_the_spec():
    spec = (ROOT / "specs" / "E3.1-serving" / "spec.md").read_text(encoding="utf-8")
    assert llama.TAG == "b12421" and llama.BIN_SHA256 in spec
    assert llama.SRC_SHA256 in (ROOT / "specs" / "E3.1-serving" / "RUN-STATE.md").read_text(
        encoding="utf-8"
    )
    assert llama.BIN_URL.endswith(f"/{llama.TAG}/{llama.BIN_NAME}")


def test_fetch_verifies_before_naming_and_refuses_a_bad_download(tmp_path):
    blob = tarball("llama-test", {"llama-server": b"#!bin"})
    pin = pin_for(blob)
    path = llama.fetch(pin, tmp_path, serving(blob))
    assert path == tmp_path / "t.tar.gz" and path.read_bytes() == blob
    assert llama.fetch(pin, tmp_path, no_download) == path  # cached, verified, not refetched
    bad = llama.Pinned("u.tar.gz", pin.url, "0" * 64, pin.folder)
    with pytest.raises(llama.LlamaError, match=r"sha256"):
        llama.fetch(bad, tmp_path, serving(blob))
    assert not (tmp_path / "u.tar.gz").exists() and not (tmp_path / "u.tar.gz.part").exists()


def test_a_cached_tarball_that_changed_is_refused_not_redownloaded(tmp_path):
    blob = tarball("llama-test", {"llama-server": b"#!bin"})
    pin = pin_for(blob)
    llama.fetch(pin, tmp_path, serving(blob))
    (tmp_path / "t.tar.gz").write_bytes(blob + b"tampered")
    with pytest.raises(llama.LlamaError, match="cached"):
        llama.fetch(pin, tmp_path, no_download)


def test_tool_is_handed_out_only_from_a_verified_extraction(tmp_path):
    blob = tarball("llama-test", {"llama-server": b"#!bin", "llama-quantize": b"#!q"})
    pin = pin_for(blob)
    out = llama.extract(pin, tmp_path, serving(blob))
    assert (out / "llama-server").read_bytes() == b"#!bin"
    assert llama.verified(pin, tmp_path) == out
    (out / llama.STAMP).write_text("f" * 64, encoding="utf-8")
    with pytest.raises(llama.LlamaError, match="not extracted from the pinned"):
        llama.verified(pin, tmp_path)
    llama.extract(pin, tmp_path, no_download)  # a stale stamp re-extracts from the tarball
    assert llama.verified(pin, tmp_path) == out
    (tmp_path / "t.tar.gz").write_bytes(b"swapped")
    with pytest.raises(llama.LlamaError, match="does not match the pin"):
        llama.verified(pin, tmp_path)


def test_release_tool_lookup_uses_the_pinned_release(tmp_path, monkeypatch):
    blob = tarball(llama.RELEASE.folder, {"llama-server": b"#!bin"})
    monkeypatch.setattr(llama, "RELEASE", pin_for(blob, llama.RELEASE.folder))
    llama.extract(llama.RELEASE, tmp_path, serving(blob))
    assert llama.tool("llama-server", tmp_path).name == "llama-server"
    with pytest.raises(llama.LlamaError, match=r"not in llama\.cpp"):
        llama.tool("llama-missing", tmp_path)


def test_extraction_refuses_paths_outside_the_folder(tmp_path):
    evil = tarfile.TarInfo("../escape")
    blob = tarball("llama-test", {"llama-server": b"x"}, extra=evil)
    with pytest.raises(tarfile.TarError):
        llama.extract(pin_for(blob), tmp_path, serving(blob))
    assert not (tmp_path.parent / "escape").exists()


def test_extraction_needs_the_named_top_folder(tmp_path):
    blob = tarball("other", {"llama-server": b"x"})
    with pytest.raises(llama.LlamaError, match="no top-level folder"):
        llama.extract(pin_for(blob), tmp_path, serving(blob))


# ---- EVK-02: serve ----


def test_server_args_pin_metal_one_slot_seed_threads_and_greedy():
    args = llama.server_args(Path("/x/llama-server"), Path("/m.gguf"), 8080)
    flags = dict(zip(args[1::2], args[2::2], strict=False))
    assert args[0] == "/x/llama-server"
    for flag, value in {
        "--model": "/m.gguf",
        "--host": "127.0.0.1",
        "--port": "8080",
        "--n-gpu-layers": "999",
        "--parallel": "1",
        "--seed": "7",
        "--threads": "8",
        "--temp": "0",
        "--top-k": "1",
        "--reasoning": "off",
        "--reasoning-budget": "0",  # D-41
        "--ctx-size": "8192",
    }.items():
        assert args[args.index(flag) + 1] == value, flag
    assert "--jinja" in args and "--no-webui" in args and flags


def test_model_paths_and_ports():
    assert llama.PORTS == {"base": 8080, "ft": 8081, "ft2": 8083, "ft3": 8085}  # v2 V2.1, v3 V3.4
    assert llama.model_path(Path("/r"), "ft", "Q8_0") == Path("/r/artifacts/gguf/ft-Q8_0.gguf")
    with pytest.raises(llama.LlamaError):
        llama.model_path(Path("/r"), "large", "Q4_K_M")
    with pytest.raises(llama.LlamaError):
        llama.model_path(Path("/r"), "base", "Q2_K")


class FakeProc:
    def __init__(self, code=None):
        self.code = code
        self.returncode = code
        self.terminated = False
        self.pid = 4242

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_wait_healthy_returns_on_ok_and_fails_on_exit_or_timeout():
    calls = []
    llama.wait_healthy(FakeProc(), 8080, check=lambda u: calls.append(u) or len(calls) > 2,
                       sleep=lambda s: None)  # fmt: skip
    assert calls[-1] == "http://127.0.0.1:8080/health" and len(calls) == 3
    with pytest.raises(llama.LlamaError, match="exited with code 1"):
        llama.wait_healthy(FakeProc(1), 8080, check=lambda u: False, sleep=lambda s: None)
    p = FakeProc()
    with pytest.raises(llama.LlamaError, match="not ok after"):
        llama.wait_healthy(p, 8080, timeout_s=0, check=lambda u: False)
    assert p.terminated


@pytest.fixture
def served(tmp_path, monkeypatch):
    blob = tarball(llama.RELEASE.folder, {"llama-server": b"#!bin"})
    monkeypatch.setattr(llama, "RELEASE", pin_for(blob, llama.RELEASE.folder))
    cache = tmp_path / "cache"
    llama.extract(llama.RELEASE, cache, serving(blob))
    gguf = llama.model_path(tmp_path, "base", "Q4_K_M")
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    return tmp_path, cache


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_launches_the_verified_server_and_waits_for_health(served):
    root, cache = served
    seen = {}

    def popen(args, **kw):
        seen["args"] = args
        return FakeProc()

    port = free_port()
    checked = []
    llama.start(root, "base", "Q4_K_M", port, popen=popen, cache=cache,
                check=lambda u: checked.append(u) or True)  # fmt: skip
    assert checked == [f"http://127.0.0.1:{port}/health"]  # returns only after /health is ok
    assert seen["args"][0] == str(cache / llama.RELEASE.folder / "llama-server")
    assert seen["args"][seen["args"].index("--port") + 1] == str(port)


def test_a_busy_port_fails_clearly_and_starts_nothing(served):
    root, cache = served
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]

        def popen(*a, **k):
            raise AssertionError("must not start")

        with pytest.raises(llama.LlamaError, match="already in use"):
            llama.start(root, "base", "Q4_K_M", port, popen=popen, cache=cache)


def test_a_missing_model_file_is_an_error(served):
    root, cache = served
    llama.model_path(root, "base", "Q4_K_M").unlink()
    with pytest.raises(llama.LlamaError, match="ggh export"):
        llama.start(root, "base", "Q4_K_M", free_port(), cache=cache)


def test_default_ports_come_from_the_model(served, monkeypatch):
    root, cache = served
    monkeypatch.setattr(llama, "port_busy", lambda p, host=llama.HOST: p == 8080)
    with pytest.raises(llama.LlamaError, match="port 8080"):
        llama.start(root, "base", "Q4_K_M", cache=cache, popen=subprocess.Popen)


def test_start_writes_the_server_log(served):
    root, cache = served
    seen = {}

    def popen(args, **kw):
        seen.update(kw)
        return FakeProc()

    log = root / "serve.log"
    llama.start(root, "base", "Q4_K_M", free_port(), popen=popen, check=lambda u: True,
                cache=cache, log=log)  # fmt: skip
    assert seen["stdout"].name == str(log) and seen["stderr"] == subprocess.STDOUT


def test_a_dead_server_is_reported_even_if_something_else_answers_health():
    with pytest.raises(llama.LlamaError, match="exited"):
        llama.wait_healthy(FakeProc(1), 8080, check=lambda u: True, sleep=lambda s: None)
