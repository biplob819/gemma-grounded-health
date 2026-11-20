"""HUB-01..05: the Hugging Face release plan (Tech Spec §10.2, D-13), dry run by default.

`plan()` lists every file that would go to the two Hub repos, with its expected sha256 taken
from the lineage manifests (never recomputed into truth), checks each local file against it,
checks the bundle's artifact records agree with the manifests (and, for v2, with
`export_artifacts_v2.json`), and runs `ggh lineage verify` on the release's chain: v1 from its
export stage, v2 from `parity_v2` back through `export_ft2`, `train_v2` and `targets_v2` to the
data. The release defaults to the bundle's shipped release (v2, OWNER D-63's benchmarked GGUF);
`layout="v1"` keeps v1's plan. `push()` is the only function that talks to the Hub, through an
injected `HubApi`; the CLI calls it only with `--push`, a token from the environment and a plan
with no problems. It uploads, tags, downloads every file back and checks its sha256 (NFR5),
then runs the Ollama check on one prompt (US7).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ggh.lineage import StageManifest, read_manifest, sha256_file, verify
from ggh.schemas import ExportArtifact, ResultsBundle

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
PLAN_SCHEMA = 1

Status = Literal["ok", "missing", "hash mismatch", "generated", "unrecorded"]


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    """Where one release's lineage and files live. `verify_from` is the chain head that
    `ggh lineage verify` walks before any push; `chain_from` the committed copy lineage.json
    starts at. `extra_merged` are files in the merged folder that no stage records: each is
    checked against the committed file named (a record), or, with None, listed by its local
    hash as "unrecorded"."""

    name: str
    export_manifest: Path
    train_manifest: Path
    export_copy: str
    train_copy: str
    verify_from: Path
    chain_from: str  # committed copy (stem) lineage.json starts from
    records: Path | None = None  # the export stage's committed artifact records
    extra_merged: tuple[tuple[str, Path | None], ...] = ()


LAYOUTS: dict[str, Layout] = {
    "v1": Layout(
        name="v1",
        export_manifest=EXPORT_MANIFEST,
        train_manifest=TRAIN_MANIFEST,
        export_copy="export_ft",
        train_copy="train",
        verify_from=EXPORT_MANIFEST,
        chain_from="export_ft",
    ),
    "v2": Layout(
        name="v2",
        export_manifest=Path("artifacts") / "export" / "ft2" / "manifest.json",
        train_manifest=Path("artifacts") / "train_v2" / "manifest.json",
        export_copy="export_ft2",
        train_copy="train_v2",
        verify_from=Path("artifacts") / "parity_v2" / "manifest.json",
        chain_from="parity_v2",
        records=Path("docs") / "measurements" / "export_artifacts_v2.json",
        extra_merged=(
            ("generation_config.json", None),  # carried from the base model by the merge
            ("processor_config.json", None),
            ("ggh_merge.json", Path("docs") / "measurements" / "v2" / "merge_record.json"),
        ),
    ),
}


@dataclass(frozen=True)
class PlannedFile:
    repo: str  # "main" or "gguf"
    path_in_repo: str
    local: str  # repo-relative source path, or "(generated)"
    sha256: str  # what the Hub copy must hash to
    status: Status
    content: bytes | None = None  # generated files only
    size: int | None = None  # bytes; None when the local file is missing
    record: str | None = None  # the committed file the sha256 comes from


@dataclass
class Plan:
    owner: str
    files: list[PlannedFile] = field(default_factory=list[PlannedFile])
    problems: list[str] = field(default_factory=list[str])
    release: str = "v1"
    model: str = ""
    lineage_verified: str | None = None  # the manifest verify walked, when it was run
    lineage_keys: list[str] = field(default_factory=list[str])  # lineage.json, child first

    def repo_id(self, repo: str) -> str:
        return f"{self.owner}/{MAIN_REPO if repo == 'main' else GGUF_REPO}"

    def by_repo(self, repo: str) -> list[PlannedFile]:
        return [f for f in self.files if f.repo == repo]

    @property
    def ready(self) -> bool:
        return not self.problems and self.owner != OWNER_PLACEHOLDER


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local(
    root: Path, repo: str, dest: str, rel: str, want: str, record: str | None = None
) -> PlannedFile:
    f = root / rel
    if not f.is_file():
        return PlannedFile(repo, dest, rel, want, "missing", record=record)
    status: Status = "ok" if sha256_file(f) == want else "hash mismatch"
    return PlannedFile(repo, dest, rel, want, status, size=f.stat().st_size, record=record)


def _generated(repo: str, dest: str, content: bytes, record: str | None = None) -> PlannedFile:
    return PlannedFile(
        repo, dest, "(generated)", _sha(content), "generated", content, len(content), record
    )


def lineage_chain(start_key: str, copies: Sequence[StageManifest]) -> list[StageManifest]:
    """The manifests from start_key back to the dataset, child first (lineage.json): depth
    first along each first parent, so v2 reads parity, export, train, targets, data."""
    index = {m.key: m for m in copies}
    out: list[StageManifest] = []
    stack, seen = [start_key], set[str]()
    while stack:
        key = stack.pop()
        if key in seen or key not in index:
            continue
        seen.add(key)
        out.append(index[key])
        stack.extend(reversed(index[key].parents))
    return out


def gguf_card(
    owner: str,
    files: Sequence[PlannedFile],
    bundle: ResultsBundle,
    artifacts: Sequence[ExportArtifact] | None = None,
    model: str | None = None,
) -> str:
    rows = "\n".join(f"| `{f.path_in_repo}` | {_human(f.size)} | `{f.sha256}` |" for f in files)
    main = f"{owner}/{MAIN_REPO}"
    arts = bundle.artifacts if artifacts is None else artifacts
    name = f" (`{model}`)" if model else ""
    return (
        "---\nlicense: apache-2.0\nbase_model: " + main + "\n"
        "tags: [gguf, gemma, pubmedqa, research-only]\n---\n\n"
        f"# {GGUF_REPO}\n\n"
        "> **Research only, not medical advice.** Not for clinical use or personal health "
        "decisions.\n\n"
        f"GGUF builds of [{main}](https://huggingface.co/{main}){name} for llama.cpp, Ollama "
        "and LM Studio. The benchmark numbers on the main model card are for the "
        f"{OLLAMA_QUANT} file; read that card for training data, evaluation and limits.\n\n"
        f"The local demo serves this {OLLAMA_QUANT} file through llama-server, the same bytes "
        "that were benchmarked (OWNER D-63), so the demo and the numbers match.\n\n"
        f"```bash\nollama run --think=false hf.co/{owner}/{GGUF_REPO}:{OLLAMA_QUANT}\n```\n\n"
        "Run it with thinking off, as the benchmark did (llama-server `--reasoning off`): "
        "with thinking on, Gemma 4 writes a reasoning block before the JSON answer. The "
        "benchmark decoded greedily (temperature 0); Ollama's default sampling can give a "
        "different answer to the same question.\n\n"
        "| File | Size | sha256 |\n|---|---|---|\n" + rows + "\n\n"
        f"Built with llama.cpp `{_gguf_commit(arts)}`. Licence: Apache-2.0 (see `NOTICE`).\n"
    )


def _human(size: int | None) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000
    return f"{size} B"  # pragma: no cover


def _gguf_commit(artifacts: Sequence[ExportArtifact]) -> str:
    for a in artifacts:
        if a.kind == "gguf" and a.model == "ft" and a.llama_cpp_commit:
            return a.llama_cpp_commit
    return "unknown"


def _manifest_or_copy(root: Path, live: Path, copy: Path, p: Plan) -> StageManifest:
    """The live stage manifest; on a machine without artifacts/, the committed copy, so the
    dry run can still list every file (each then shows as missing and the plan is not ready).
    A live manifest that is not its committed copy is a problem: lineage.json would describe
    other bytes than the ones pushed."""
    if (root / live).is_file() or not (root / copy).is_file():
        m = read_manifest(root / live)
        if (root / copy).is_file() and read_manifest(root / copy).key != m.key:
            p.problems.append(f"{live} is not the committed lineage copy {copy}")
        return m
    p.problems.append(f"{live} is missing: listing from the committed copy {copy}")
    return read_manifest(root / copy)


def _release_entry_artifacts(bundle: ResultsBundle, version: str) -> tuple[str, Any]:
    for r in bundle.releases:
        if r.version == version:
            return r.model, r.artifacts
    raise ValueError(f"the bundle has no release {version!r}")


def default_layout(bundle_path: Path) -> str:
    """The shipped release's layout (the newest release in the bundle)."""
    bundle = ResultsBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    return bundle.shipped.version


def plan(
    root: Path,
    bundle_path: Path,
    card_text: str,
    *,
    owner: str = OWNER_PLACEHOLDER,
    layout: str | None = None,
    export_manifest: Path | None = None,
    train_manifest: Path | None = None,
    lineage_copies: Path = LINEAGE_COPIES,
    committed_card: Path | None = None,
) -> Plan:
    """What `make release` would push, and every reason it is not ready to. `layout` is the
    release (v1, v2); None means the bundle's shipped release."""
    bundle = ResultsBundle.model_validate_json((root / bundle_path).read_text(encoding="utf-8"))
    version = layout or bundle.shipped.version
    if version not in LAYOUTS:
        raise ValueError(f"no release layout {version!r}; known: {', '.join(LAYOUTS)}")
    lay = LAYOUTS[version]
    export_manifest = export_manifest or lay.export_manifest
    train_manifest = train_manifest or lay.train_manifest
    p = Plan(owner, release=version)
    model, artifacts = _release_entry_artifacts(bundle, version)
    p.model = model
    if not bundle.lineage_ok:
        p.problems.append("the bundle says lineage_ok is false")
    if committed_card is not None:
        f = root / committed_card
        if not f.is_file() or f.read_text(encoding="utf-8") != card_text:
            p.problems.append(f"{committed_card} is not the card generated from the bundle")
    try:
        export = _manifest_or_copy(
            root, export_manifest, lineage_copies / f"{lay.export_copy}.json", p
        )
        train = _manifest_or_copy(
            root, train_manifest, lineage_copies / f"{lay.train_copy}.json", p
        )
    except ValueError as exc:
        p.problems.append(str(exc))
        return p
    verify_from = lay.verify_from if export_manifest == lay.export_manifest else export_manifest
    if (root / verify_from).is_file():  # HUB-01: the whole chain, before any push
        p.lineage_verified = verify_from.as_posix()
        p.problems += [f"lineage: {x}" for x in verify(root / verify_from, root)]
    elif (root / export_manifest).is_file():
        p.problems.append(f"{verify_from} is missing: the chain cannot be verified")
    if export.parents and train.key not in export.parents:
        p.problems.append(f"{export_manifest} is not built from {train_manifest}")

    if lay.records is not None:  # the bundle must carry the export stage's committed records
        recs = json.loads((root / lay.records).read_text(encoding="utf-8"))
        theirs = sorted(json.dumps(r, sort_keys=True) for r in recs)
        ours = sorted(json.dumps(a.model_dump(mode="json"), sort_keys=True) for a in artifacts)
        if theirs != ours:
            p.problems.append(f"the bundle's {version} artifacts are not {lay.records}")

    ft = {a.kind: a for a in artifacts if a.model == "ft" and a.kind != "gguf"}
    gguf = {a.quant: a for a in artifacts if a.model == "ft" and a.kind == "gguf"}
    merged, adapter = ft.get("merged_bf16"), ft.get("adapter")
    if merged is None or adapter is None:
        p.problems.append("the bundle lists no fine-tuned merged_bf16 or adapter artifact")
        return p
    merged_dir = Path(merged.path).parent.as_posix() + "/"
    adapter_dir = adapter.path.rstrip("/") + "/"

    produced = {**train.outputs, **export.outputs}
    for a in artifacts:  # the bundle's records must be the manifests' hashes
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

    export_rec = f"{lineage_copies.as_posix()}/{lay.export_copy}.json"
    train_rec = f"{lineage_copies.as_posix()}/{lay.train_copy}.json"
    for rel, digest in sorted(export.outputs.items()):
        if rel.startswith(merged_dir):
            dest = rel[len(merged_dir) :]
            p.files.append(_local(root, "main", dest, rel, digest, export_rec))
    for name, record in lay.extra_merged:
        rel = merged_dir + name
        if record is not None:
            if not (root / record).is_file():
                p.problems.append(f"{record} is missing")
                continue
            p.files.append(_local(root, "main", name, rel, sha256_file(root / record), str(record)))
        elif (root / rel).is_file():
            f = root / rel
            size = f.stat().st_size
            p.files.append(PlannedFile("main", name, rel, sha256_file(f), "unrecorded", size=size))
        else:
            p.problems.append(f"{rel}: missing")
    for rel, digest in sorted(train.outputs.items()):
        if rel.startswith(adapter_dir):
            dest = "adapter/" + rel[len(adapter_dir) :]
            p.files.append(_local(root, "main", dest, rel, digest, train_rec))
    card_rec = committed_card.as_posix() if committed_card is not None else None
    p.files.append(_generated("main", "README.md", card_text.encode("utf-8"), card_rec))
    p.files.append(
        _generated(
            "main", "results_bundle.json", (root / bundle_path).read_bytes(), str(bundle_path)
        )
    )
    copies = [read_manifest(f) for f in sorted((root / lineage_copies).glob("*.json"))]
    head = root / lineage_copies / f"{lay.chain_from}.json"
    default = export_manifest == lay.export_manifest
    start = read_manifest(head).key if default and head.is_file() else export.key
    chain = lineage_chain(start, copies)
    p.lineage_keys = [f"{m.stage}:{m.key}" for m in chain]
    if not chain:
        p.problems.append(f"no committed lineage copy for key {start[:12]}")
    elif export.key not in {m.key for m in chain}:
        p.problems.append(f"lineage.json does not reach the export stage {export.key[:12]}")
    dump = [m.model_dump(mode="json") for m in chain]
    lineage_bytes = (json.dumps(dump, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
        # Ollama resolves hf.co/<owner>/<repo>:<quant> by the quant tag in the file name
        gguf_files.append(_local(root, "gguf", f"{MAIN_REPO}-{q}.gguf", a.path, want, export_rec))
    p.files += gguf_files
    gguf_text = gguf_card(owner, gguf_files, bundle, artifacts, model)
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
    lines = [f"release {p.release} ({p.model})"]
    if p.lineage_verified:
        lines.append(f"lineage verified from {p.lineage_verified}: {len(p.lineage_keys)} stages")
    for repo in ("main", "gguf"):
        lines.append(f"{p.repo_id(repo)}:")
        for f in p.by_repo(repo):
            size = "?" if f.size is None else f"{f.size:,}"
            lines.append(f"  {f.path_in_repo:<42} {size:>16}  {f.sha256}  {f.status:<10} {f.local}")
    total = sum(f.size or 0 for f in p.files)
    lines.append(f"files: {len(p.files)}, {total:,} bytes")
    lines.append(f"problems: {len(p.problems)}")
    lines += [f"  - {x}" for x in p.problems]
    return lines


def plan_json(p: Plan) -> dict[str, Any]:
    """The plan as committed evidence: every file with size, sha256 and the record it is
    checked against. No time stamps, so the same inputs give the same bytes."""
    return {
        "schema": PLAN_SCHEMA,
        "release": p.release,
        "model": p.model,
        "owner": p.owner,
        "repos": {repo: p.repo_id(repo) for repo in ("main", "gguf")},
        "lineage_verified_from": p.lineage_verified,
        "lineage_chain": p.lineage_keys,
        "ollama_ref": f"hf.co/{p.repo_id('gguf')}:{OLLAMA_QUANT}",
        "files": [
            {
                "repo": p.repo_id(f.repo),
                "path_in_repo": f.path_in_repo,
                "local": f.local,
                "size": f.size,
                "sha256": f.sha256,
                "status": f.status,
                "record": f.record,
            }
            for f in p.files
        ],
        "total_bytes": sum(f.size or 0 for f in p.files),
        "problems": list(p.problems),
        "ready": p.ready,
    }


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
    ollama: Callable[[str], bool | str | None] | None = None,
    record: dict[str, Any] | None = None,
) -> list[str]:
    """Upload, tag, download back and check every sha256; then the Ollama check. Refuses a
    plan that is not ready. Returns the problems found after the push (empty: all good).
    `record`, when given, receives what was checked and the Ollama answer."""
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
    checked: list[dict[str, Any]] = []
    for f in p.files:  # HUB-04: every file, the large ones included (streamed sha256)
        where = f"{p.repo_id(f.repo)}/{f.path_in_repo}"
        try:
            back = api.download(p.repo_id(f.repo), f.path_in_repo, tag)
            got = sha256_file(back)
        except (OSError, RuntimeError) as exc:
            problems.append(f"{where}: download failed: {exc}")
            checked.append({"file": where, "sha256": None, "ok": False})
            continue
        ok = got == f.sha256
        if not ok:
            problems.append(f"{where}: downloaded sha256 differs")
        checked.append({"file": where, "sha256": got, "ok": ok})
    ref = f"hf.co/{p.repo_id('gguf')}:{OLLAMA_QUANT}"
    answer: bool | str | None = None
    if ollama is not None:
        answer = ollama(ref)
        if not answer:
            problems.append("ollama did not answer the sample prompt")
    if record is not None:
        record.update(
            {
                "release": p.release,
                "tag": tag,
                "repos": {repo: p.repo_id(repo) for repo in ("main", "gguf")},
                "downloads": checked,
                "ollama_ref": ref,
                "ollama_answer": answer if isinstance(answer, str) else None,
                "problems": problems,
            }
        )
    return problems


def token_from_env(env: Mapping[str, str]) -> str | None:
    """The Hub token by its variable name only; the value is never printed or logged."""
    tok = env.get(TOKEN_ENV, "").strip()
    return tok or None


# HUB-05: `ollama run hf.co/<owner>/<repo>-GGUF:Q4_K_M` on one demo sample
OLLAMA_BIN_FALLBACK = Path("/opt/homebrew/bin/ollama")
OLLAMA_SAMPLES = Path("space") / "samples.json"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass
class OllamaResult:
    ref: str
    ok: bool
    answer: str = ""
    label: str | None = None  # the parsed answer, when the reply fits the schema
    citations: tuple[int, ...] = ()
    error: str | None = None
    started_server: bool = False
    sample_id: str = ""


def sample_prompt(root: Path, samples: Path = OLLAMA_SAMPLES) -> tuple[str, str, tuple[str, ...]]:
    """(sample id, prompt, sentences) for the first demo sample, rendered with EvalKit's
    PubMedQA template exactly as in training and the benchmark."""
    from ggh.space.engine import build_prompt

    first = json.loads((root / samples).read_text(encoding="utf-8"))[0]
    sentences = tuple(first["context"])
    return first["id"], build_prompt(first["question"], sentences), sentences


def ollama_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    return str(OLLAMA_BIN_FALLBACK) if OLLAMA_BIN_FALLBACK.is_file() else None


def ollama_check(
    ref: str,
    root: Path,
    *,
    binary: str | None = None,
    run: Runner = subprocess.run,
    popen: Callable[..., Any] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    wait_s: float = 30.0,
    timeout_s: float = 1800.0,
) -> OllamaResult:
    """Run one sample prompt through `ollama run <ref>`. Uses a running `ollama serve`, or
    starts one (and stops it afterwards) when none answers; a missing binary is a failed check,
    never an exception. The pull of a fresh hf.co ref can take a while, hence the long timeout."""
    sample_id, prompt, sentences = sample_prompt(root)
    res = OllamaResult(ref, ok=False, sample_id=sample_id)
    exe = binary or ollama_binary()
    if exe is None:
        res.error = "ollama is not installed"
        return res

    def alive() -> bool:
        try:
            return run([exe, "list"], capture_output=True, text=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    server = None
    if not alive():
        try:
            server = popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            res.error = f"cannot start ollama serve: {exc}"
            return res
        res.started_server = True
        waited = 0.0
        while not alive():
            if waited >= wait_s:
                res.error = f"ollama serve did not answer within {wait_s:.0f}s"
                server.terminate()
                return res
            sleep(1.0)
            waited += 1.0
    try:
        done = run(
            # thinking off, as `ggh serve` runs llama-server (--reasoning off): the benchmark
            # never let the model think first, so neither does the check
            [exe, "run", "--nowordwrap", "--think=false", ref],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        res.error = f"ollama run failed: {exc}"
        return res
    finally:
        if server is not None:
            server.terminate()
    res.answer = done.stdout.strip()
    if done.returncode != 0 or not res.answer:
        res.error = (done.stderr or "").strip()[-500:] or f"exit code {done.returncode}"
        return res
    from ggh.space.engine import to_card

    card = to_card(ref, res.answer, sentences)
    res.label, res.citations = card.label, card.citations
    # HUB-05: passes only on a parsed answer with a label and in-range citations
    if card.label is None or not card.citations:
        res.error = "the answer did not parse to a label with citations"
        return res
    res.ok = True
    return res


class HfHubApi:  # pragma: no cover - network; exercised only by `ggh release --push`
    def __init__(self, token: str, download_dir: Path | None = None) -> None:
        from huggingface_hub import HfApi  # pyright: ignore[reportMissingImports]

        self._api = HfApi(token=token)
        self._download_dir = download_dir

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

        local_dir = None if self._download_dir is None else str(self._download_dir / repo_id)
        return Path(
            hf_hub_download(
                repo_id,
                path_in_repo,
                revision=revision,
                token=self._api.token,
                force_download=True,
                local_dir=local_dir,
            )
        )
