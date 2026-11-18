"""HUB-01..05: the Hugging Face release plan (Tech Spec §10.2, D-13), dry run by default.

`plan()` lists every file that would go to the two Hub repos, with its expected sha256 taken
from the lineage manifests (never recomputed into truth), checks each local file against it,
checks the bundle's artifact records agree with the manifests, and runs `ggh lineage verify` on
the export stage. `push()` is the only function that talks to the Hub, through an injected
`HubApi`; the CLI calls it only with `--push`, a token from the environment and a plan with no
problems. It uploads, tags, downloads every file back and checks its sha256 (NFR5), then runs
the Ollama check on one prompt (US7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from ggh.lineage import StageManifest, read_manifest, sha256_file, verify
from ggh.schemas import ResultsBundle

EXPORT_MANIFEST = Path("artifacts") / "export" / "ft" / "manifest.json"
TRAIN_MANIFEST = Path("artifacts") / "train" / "manifest.json"
LINEAGE_COPIES = Path("docs") / "measurements" / "lineage"
OWNER_PLACEHOLDER = "<owner>"
MAIN_REPO = "gemma-4-e2b-grounded-health"
GGUF_REPO = f"{MAIN_REPO}-GGUF"
GGUF_QUANTS = ("Q4_K_M", "Q8_0")
OLLAMA_QUANT = "Q4_K_M"
TOKEN_ENV = "HF_TOKEN"
LEGAL = ("LICENSE", "NOTICE")

Status = Literal["ok", "missing", "hash mismatch", "generated"]


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedFile:
    repo: str  # "main" or "gguf"
    path_in_repo: str
    local: str  # repo-relative source path, or "(generated)"
    sha256: str  # what the Hub copy must hash to
    status: Status
    content: bytes | None = None  # generated files only


@dataclass
class Plan:
    owner: str
    files: list[PlannedFile] = field(default_factory=list[PlannedFile])
    problems: list[str] = field(default_factory=list[str])

    def repo_id(self, repo: str) -> str:
        return f"{self.owner}/{MAIN_REPO if repo == 'main' else GGUF_REPO}"

    def by_repo(self, repo: str) -> list[PlannedFile]:
        return [f for f in self.files if f.repo == repo]

    @property
    def ready(self) -> bool:
        return not self.problems and self.owner != OWNER_PLACEHOLDER


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local(root: Path, repo: str, dest: str, rel: str, want: str) -> PlannedFile:
    f = root / rel
    if not f.is_file():
        return PlannedFile(repo, dest, rel, want, "missing")
    status: Status = "ok" if sha256_file(f) == want else "hash mismatch"
    return PlannedFile(repo, dest, rel, want, status)


def _generated(repo: str, dest: str, content: bytes) -> PlannedFile:
    return PlannedFile(repo, dest, "(generated)", _sha(content), "generated", content)


def lineage_chain(start_key: str, copies: Sequence[StageManifest]) -> list[StageManifest]:
    """The manifests from start_key back to the dataset, child first (lineage.json)."""
    index = {m.key: m for m in copies}
    out: list[StageManifest] = []
    stack, seen = [start_key], set[str]()
    while stack:
        key = stack.pop(0)
        if key in seen or key not in index:
            continue
        seen.add(key)
        out.append(index[key])
        stack.extend(index[key].parents)
    return out


def gguf_card(owner: str, files: Sequence[PlannedFile], bundle: ResultsBundle) -> str:
    rows = "\n".join(f"| `{f.path_in_repo}` | `{f.sha256}` |" for f in files)
    main = f"{owner}/{MAIN_REPO}"
    return (
        "---\nlicense: apache-2.0\nbase_model: " + main + "\ntags: [gguf, gemma, pubmedqa]\n---\n\n"
        f"# {GGUF_REPO}\n\n"
        "> **Research only, not medical advice.**\n\n"
        f"GGUF builds of [{main}](https://huggingface.co/{main}) for llama.cpp, Ollama and "
        "LM Studio. The benchmark numbers on the main model card are for the "
        f"{OLLAMA_QUANT} file; read that card for training data, evaluation and limits.\n\n"
        f"```bash\nollama run hf.co/{owner}/{GGUF_REPO}:{OLLAMA_QUANT}\n```\n\n"
        "| File | sha256 |\n|---|---|\n" + rows + "\n\n"
        f"Built with llama.cpp `{_gguf_commit(bundle)}`. Licence: Apache-2.0 (see `NOTICE`).\n"
    )


def _gguf_commit(bundle: ResultsBundle) -> str:
    for a in bundle.artifacts:
        if a.kind == "gguf" and a.model == "ft" and a.llama_cpp_commit:
            return a.llama_cpp_commit
    return "unknown"


def _manifest_or_copy(root: Path, live: Path, copy: Path, p: Plan) -> StageManifest:
    """The live stage manifest; on a machine without artifacts/, the committed copy, so the
    dry run can still list every file (each then shows as missing and the plan is not ready)."""
    if (root / live).is_file() or not (root / copy).is_file():
        return read_manifest(root / live)
    p.problems.append(f"{live} is missing: listing from the committed copy {copy}")
    return read_manifest(root / copy)


def plan(
    root: Path,
    bundle_path: Path,
    card_text: str,
    *,
    owner: str = OWNER_PLACEHOLDER,
    export_manifest: Path = EXPORT_MANIFEST,
    train_manifest: Path = TRAIN_MANIFEST,
    lineage_copies: Path = LINEAGE_COPIES,
) -> Plan:
    """What `make release` would push, and every reason it is not ready to."""
    p = Plan(owner)
    bundle = ResultsBundle.model_validate_json((root / bundle_path).read_text(encoding="utf-8"))
    if not bundle.lineage_ok:
        p.problems.append("the bundle says lineage_ok is false")
    try:
        export = _manifest_or_copy(root, export_manifest, lineage_copies / "export_ft.json", p)
        train = _manifest_or_copy(root, train_manifest, lineage_copies / "train.json", p)
    except ValueError as exc:
        p.problems.append(str(exc))
        return p
    if (root / export_manifest).is_file():
        p.problems += [f"lineage: {x}" for x in verify(root / export_manifest, root)]

    ft = {a.kind: a for a in bundle.artifacts if a.model == "ft" and a.kind != "gguf"}
    gguf = {a.quant: a for a in bundle.artifacts if a.model == "ft" and a.kind == "gguf"}
    merged, adapter = ft.get("merged_bf16"), ft.get("adapter")
    if merged is None or adapter is None:
        p.problems.append("the bundle lists no fine-tuned merged_bf16 or adapter artifact")
        return p
    merged_dir = Path(merged.path).parent.as_posix() + "/"
    adapter_dir = adapter.path.rstrip("/") + "/"

    produced = {**train.outputs, **export.outputs}
    for a in bundle.artifacts:  # the bundle's records must be the manifests' hashes
        rel = a.path if a.path in produced else None
        if a.kind == "adapter":
            rel = next(
                (r for r in train.outputs if r.startswith(adapter_dir) and produced[r] == a.sha256),
                None,
            )
            if rel is None:
                p.problems.append(f"adapter sha256 {a.sha256[:12]} is not a train output")
            continue
        if a.model == "ft" and rel is not None and produced[rel] != a.sha256:
            p.problems.append(f"{a.path}: the bundle and the export manifest disagree on sha256")

    for rel, digest in sorted(export.outputs.items()):
        if rel.startswith(merged_dir):
            p.files.append(_local(root, "main", rel[len(merged_dir) :], rel, digest))
    for rel, digest in sorted(train.outputs.items()):
        if rel.startswith(adapter_dir):
            p.files.append(_local(root, "main", "adapter/" + rel[len(adapter_dir) :], rel, digest))
    p.files.append(_generated("main", "README.md", card_text.encode("utf-8")))
    p.files.append(_generated("main", "results_bundle.json", (root / bundle_path).read_bytes()))
    copies = [read_manifest(f) for f in sorted((root / lineage_copies).glob("*.json"))]
    chain = [m.model_dump(mode="json") for m in lineage_chain(export.key, copies)]
    if not chain:
        p.problems.append(f"no committed lineage copy for export key {export.key[:12]}")
    lineage_bytes = (json.dumps(chain, indent=2, sort_keys=True) + "\n").encode("utf-8")
    p.files.append(_generated("main", "lineage.json", lineage_bytes))

    gguf_files: list[PlannedFile] = []
    for q in GGUF_QUANTS:
        a = gguf.get(q)
        if a is None:
            p.problems.append(f"the bundle lists no fine-tuned {q} GGUF")
            continue
        want = export.outputs.get(a.path)
        if want is None:
            p.problems.append(f"{a.path} is not an output of the export manifest")
            continue
        gguf_files.append(_local(root, "gguf", f"{MAIN_REPO}-{q}.gguf", a.path, want))
    p.files += gguf_files
    gguf_text = gguf_card(owner, gguf_files, bundle)
    p.files.append(_generated("gguf", "README.md", gguf_text.encode("utf-8")))
    for name in LEGAL:
        for repo in ("main", "gguf"):
            f = root / name
            if f.is_file():
                p.files.append(_local(root, repo, name, name, sha256_file(f)))
            else:
                p.problems.append(f"{name} is missing")
    p.problems += [
        f"{f.local}: {f.status}" for f in p.files if f.status in ("missing", "hash mismatch")
    ]
    return p


def describe(p: Plan) -> list[str]:
    lines: list[str] = []
    for repo in ("main", "gguf"):
        lines.append(f"{p.repo_id(repo)}:")
        for f in p.by_repo(repo):
            lines.append(f"  {f.path_in_repo:<44} {f.sha256[:12]}  {f.status:<13} {f.local}")
    lines.append(f"problems: {len(p.problems)}")
    lines += [f"  - {x}" for x in p.problems]
    return lines


class HubApi(Protocol):
    def create_repo(self, repo_id: str) -> None: ...
    def upload(self, repo_id: str, path_in_repo: str, local: Path) -> None: ...
    def tag(self, repo_id: str, tag: str) -> None: ...
    def download(self, repo_id: str, path_in_repo: str, revision: str) -> Path: ...


def push(
    p: Plan,
    root: Path,
    api: HubApi,
    *,
    tag: str,
    staging: Path,
    ollama: Callable[[str], bool] | None = None,
) -> list[str]:
    """Upload, tag, download back and check every sha256; then the Ollama check. Refuses a
    plan that is not ready. Returns the problems found after the push (empty: all good)."""
    if not p.ready:
        raise ReleaseError("the release plan is not ready; run the dry run and fix its problems")
    staging.mkdir(parents=True, exist_ok=True)
    for repo in ("main", "gguf"):
        rid = p.repo_id(repo)
        api.create_repo(rid)
        for f in p.by_repo(repo):
            local = root / f.local
            if f.content is not None:
                local = staging / repo / f.path_in_repo
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(f.content)
            api.upload(rid, f.path_in_repo, local)
        api.tag(rid, tag)
    problems: list[str] = []
    for f in p.files:
        back = api.download(p.repo_id(f.repo), f.path_in_repo, tag)
        if sha256_file(back) != f.sha256:
            problems.append(f"{p.repo_id(f.repo)}/{f.path_in_repo}: downloaded sha256 differs")
    if ollama is not None and not ollama(f"hf.co/{p.repo_id('gguf')}:{OLLAMA_QUANT}"):
        problems.append("ollama did not answer the sample prompt")
    return problems


def token_from_env(env: Mapping[str, str]) -> str | None:
    """The Hub token by its variable name only; the value is never printed or logged."""
    tok = env.get(TOKEN_ENV, "").strip()
    return tok or None


class HfHubApi:  # pragma: no cover - network; exercised only by `ggh release --push`
    def __init__(self, token: str) -> None:
        from huggingface_hub import HfApi  # pyright: ignore[reportMissingImports]

        self._api = HfApi(token=token)

    def create_repo(self, repo_id: str) -> None:
        self._api.create_repo(repo_id, repo_type="model", exist_ok=True)

    def upload(self, repo_id: str, path_in_repo: str, local: Path) -> None:
        self._api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )

    def tag(self, repo_id: str, tag: str) -> None:
        self._api.create_tag(repo_id, tag=tag, repo_type="model", exist_ok=False)

    def download(self, repo_id: str, path_in_repo: str, revision: str) -> Path:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

        return Path(
            hf_hub_download(
                repo_id, path_in_repo, revision=revision, token=self._api.token, force_download=True
            )
        )
