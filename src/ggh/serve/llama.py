"""EVK-01, EVK-02: the pinned llama.cpp release, verified before anything in it runs, and
`ggh serve` on top of it (Tech Spec §6, D-7, E4, E6).

Both tarballs are fetched into the cache as `<name>.part`, hashed, and only then given their
final name; a cached file whose hash no longer matches is refused, never re-downloaded over.
`tool()` re-hashes the tarball and checks the extraction stamp every time it hands out a path,
so an unverified binary is never run.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TAG = "b12421"
BIN_NAME = f"llama-{TAG}-bin-macos-arm64.tar.gz"
BIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{BIN_NAME}"
BIN_SHA256 = "bdeff25d862f1c9b9526ce2c373492e67656659b2ac386308c28b4808f4d074d"  # GitHub digest
SRC_NAME = f"llama.cpp-{TAG}.tar.gz"
SRC_URL = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{TAG}.tar.gz"
SRC_SHA256 = "5ff59287cab5dd3d9c90444f7d3692db5b51e5e4a4c6d3eb086cfa3b834ff9de"  # first fetch
STAMP = ".verified-sha256"

HOST = "127.0.0.1"
PORTS = {"base": 8080, "ft": 8081}
QUANTS = ("Q4_K_M", "Q8_0")
SEED = 7
THREADS = 8
CTX = 8192
GGUF_DIR = Path("artifacts") / "gguf"


class Proc(Protocol):
    """The part of `subprocess.Popen` that serving uses."""

    returncode: int | None
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class LlamaError(RuntimeError):
    """A fetch, verification or serving step that must stop the run."""


@dataclass(frozen=True)
class Pinned:
    name: str
    url: str
    sha256: str
    folder: str  # the top-level folder inside the tarball


RELEASE = Pinned(BIN_NAME, BIN_URL, BIN_SHA256, f"llama-{TAG}")
SOURCE = Pinned(SRC_NAME, SRC_URL, SRC_SHA256, f"llama.cpp-{TAG}")


def cache_root() -> Path:
    return Path(os.environ.get("GGH_CACHE", Path.home() / ".cache" / "ggh")) / "llama" / TAG


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:  # pragma: no cover - network
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def fetch(pin: Pinned, root: Path, download: Callable[[str, Path], None] = _download) -> Path:
    """The verified tarball in `root`; downloads only when it is absent."""
    root.mkdir(parents=True, exist_ok=True)
    dest = root / pin.name
    if dest.exists():
        got = sha256_file(dest)
        if got != pin.sha256:
            raise LlamaError(
                f"cached {dest} has sha256 {got[:12]}…, pinned {pin.sha256[:12]}…; "
                "delete it by hand after checking why"
            )
        return dest
    part = root / (pin.name + ".part")
    download(pin.url, part)
    got = sha256_file(part)
    if got != pin.sha256:
        part.unlink()
        raise LlamaError(f"{pin.url} has sha256 {got[:12]}…, pinned {pin.sha256[:12]}…")
    part.replace(dest)
    return dest


def extract(pin: Pinned, root: Path, download: Callable[[str, Path], None] = _download) -> Path:
    """The extracted folder, stamped with the tarball hash it came from."""
    tarball = fetch(pin, root, download)
    out = root / pin.folder
    stamp = out / STAMP
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == pin.sha256:
        return out
    if out.exists():
        shutil.rmtree(out)
    tmp = root / (pin.folder + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with tarfile.open(tarball) as tf:
        tf.extractall(tmp, filter="data")  # no absolute paths, links out or device files
    inner = tmp / pin.folder
    if not inner.is_dir():
        shutil.rmtree(tmp)
        raise LlamaError(f"{tarball.name} has no top-level folder {pin.folder}")
    inner.replace(out)
    shutil.rmtree(tmp)
    stamp.write_text(pin.sha256 + "\n", encoding="utf-8")
    return out


def verified(pin: Pinned, root: Path) -> Path:
    """The extracted folder, only if its tarball still hashes to the pin and its stamp agrees."""
    tarball = root / pin.name
    out = root / pin.folder
    if not tarball.exists() or sha256_file(tarball) != pin.sha256:
        raise LlamaError(f"{tarball} is missing or does not match the pin; run `ggh llama fetch`")
    stamp = out / STAMP
    if not stamp.exists() or stamp.read_text(encoding="utf-8").strip() != pin.sha256:
        raise LlamaError(f"{out} was not extracted from the pinned tarball; run `ggh llama fetch`")
    return out


def tool(name: str, root: Path | None = None) -> Path:
    """A release binary (llama-server, llama-quantize, ...) after verification."""
    path = verified(RELEASE, root or cache_root()) / name
    if not path.is_file():
        raise LlamaError(f"{name} is not in llama.cpp {TAG}")
    return path


def source(root: Path | None = None) -> Path:
    """The pinned source tree (convert_hf_to_gguf.py and gguf-py) after verification."""
    return verified(SOURCE, root or cache_root())


# ---- serving -------------------------------------------------------------------------------


def model_path(root: Path, model: str, quant: str) -> Path:
    if model not in PORTS:
        raise LlamaError(f"model must be one of {sorted(PORTS)}, got {model!r}")
    if quant not in QUANTS:
        raise LlamaError(f"quant must be one of {QUANTS}, got {quant!r}")
    return root / GGUF_DIR / f"{model}-{quant}.gguf"


def server_args(server: Path, gguf: Path, port: int) -> list[str]:
    """All layers on Metal, one slot, fixed seed and threads, greedy decoding (§6, E4).

    EvalKit sends no sampling parameters (its D-22), so the server's defaults decide them:
    temperature 0 and top-k 1 make every request greedy. Thinking is off: the training
    targets have none, and the template must render as `apply_chat_template` does (E9).
    """
    return [
        str(server),
        "--model", str(gguf),
        "--host", HOST,
        "--port", str(port),
        "--n-gpu-layers", "999",
        "--parallel", "1",
        "--seed", str(SEED),
        "--threads", str(THREADS),
        "--ctx-size", str(CTX),
        "--temp", "0",
        "--top-k", "1",
        "--jinja",
        "--reasoning", "off",
        "--reasoning-budget", "0",  # D-41: the chat grammar still admits a thought channel
        "--no-webui",
    ]  # fmt: skip


def port_busy(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _health(url: str) -> bool:  # pragma: no cover - network
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


def wait_healthy(
    proc: Proc,
    port: int,
    timeout_s: float = 300,
    check: Callable[[str], bool] = _health,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    url = f"http://{HOST}:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise LlamaError(f"llama-server exited with code {proc.returncode} before {url} was ok")
        if check(url):
            return
        sleep(1)
    proc.terminate()
    raise LlamaError(f"{url} not ok after {timeout_s:.0f}s")


def start(
    root: Path,
    model: str,
    quant: str,
    port: int | None = None,
    *,
    log: Path | None = None,
    popen: Callable[..., Proc] = subprocess.Popen,
    check: Callable[[str], bool] = _health,
    cache: Path | None = None,
) -> Proc:
    """Start llama-server and return once /health is ok. A busy port is an error, not a
    silent reuse of whatever is listening there."""
    port = PORTS[model] if port is None and model in PORTS else port
    gguf = model_path(root, model, quant)
    assert port is not None
    if port_busy(port):
        raise LlamaError(f"port {port} is already in use; stop that server first")
    if not gguf.is_file():
        raise LlamaError(f"{gguf} is missing; build it with `ggh export`")
    args = server_args(tool("llama-server", cache), gguf, port)
    out = log.open("ab") if log else subprocess.DEVNULL
    proc = popen(args, stdout=out, stderr=subprocess.STDOUT)
    wait_healthy(proc, port, check=check)
    return proc
