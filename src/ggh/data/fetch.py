"""Pinned PubMedQA downloads (DAT-01, DAT-R1).

Files are fetched once into a cache folder and verified against pinned sha256 values before
every use. A file that fails the check is refused and left in place for inspection; it is
never silently re-downloaded over.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ggh.lineage import sha256_file

DATASET = "qiaojin/PubMedQA"
REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"


@dataclass(frozen=True)
class PinnedFile:
    config: str
    path: str  # inside the dataset repo
    sha256: str
    rows: int


PQA_A = PinnedFile(
    "pqa_artificial",
    "pqa_artificial/train-00000-of-00001.parquet",
    "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05",
    211_269,
)
PQA_L = PinnedFile(
    "pqa_labeled",
    "pqa_labeled/train-00000-of-00001.parquet",
    "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c",
    1_000,
)


class DataError(RuntimeError):
    """A dataset file that is missing, corrupt or unexpected."""


def cache_dir() -> Path:
    root = os.environ.get("GGH_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "ggh"
    return base / f"pubmedqa-{REVISION[:7]}"


def uri(pinned: PinnedFile) -> str:
    """The lineage name of a pinned source outside the repo (D-32: hf:// URI)."""
    return f"hf://{DATASET}@{REVISION}/{pinned.path}"


def _url(pinned: PinnedFile) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{pinned.path}"


def ensure(pinned: PinnedFile, *, download: bool = True) -> Path:
    """Path to the verified local copy, downloading it first if it is absent."""
    target = cache_dir() / f"{pinned.config}.parquet"
    if not target.exists():
        if not download:
            raise DataError(f"{target} is missing and downloading is off")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
        os.close(fd)
        try:
            with urllib.request.urlopen(_url(pinned), timeout=120) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            got = sha256_file(Path(tmp))
            if got != pinned.sha256:  # verify before it ever takes the cache name
                raise DataError(
                    f"download of {pinned.path} has sha256 {got[:12]}…, "
                    f"expected {pinned.sha256[:12]}…"
                )
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    digest = sha256_file(target)
    if digest != pinned.sha256:
        raise DataError(
            f"{target} has sha256 {digest[:12]}…, expected {pinned.sha256[:12]}… for "
            f"{pinned.path} at {REVISION[:7]}; delete it to download again"
        )
    return target
