"""The 50-example hand audit (AUD-R1 to R3): a local labelling page and its score.

`ggh audit serve` shows `docs/measurements/audit_sample.jsonl` one example at a time on
127.0.0.1 (keys O/W/U/L) and saves every label straight away to
`docs/measurements/audit_labels.jsonl`. Each label carries a sha256 of the exact text shown,
so a later change to an example voids its label (EvalKit L-26). `ggh audit score` applies the
pass bar: at least 45 of 50 `ok`, and no `unsupported_answer` among `answer` examples.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ggh.data.build import read_jsonl
from ggh.schemas import AuditLabel, TrainExample

SAMPLE = Path("docs") / "measurements" / "audit_sample.jsonl"
LABELS = Path("docs") / "measurements" / "audit_labels.jsonl"
HOST = "127.0.0.1"
PASS_OK = 45
N_SAMPLE = 50


def shown(e: TrainExample) -> dict[str, Any]:
    """Exactly what the page shows for an example."""
    return {
        "id": e.id,
        "kind": e.kind,
        "question": e.question,
        "context": list(e.context),
        "label": e.target.label,
        "answer": e.target.answer,
        "citations": list(e.target.citations),
    }


def digest(e: TrainExample) -> str:
    blob = json.dumps(shown(e), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def load_sample(root: Path) -> list[TrainExample]:
    return [TrainExample.model_validate(r) for r in read_jsonl(root / SAMPLE)]


def load_labels(root: Path) -> dict[str, AuditLabel]:
    path = root / LABELS
    if not path.exists():
        return {}
    return {(lab := AuditLabel.model_validate(r)).example_id: lab for r in read_jsonl(path)}


def save_label(root: Path, label: AuditLabel) -> None:
    sample = {e.id: e for e in load_sample(root)}
    e = sample.get(label.example_id)
    if e is None:
        raise ValueError(f"{label.example_id} is not in the audit sample")
    if label.content_digest != digest(e):
        raise ValueError(f"{label.example_id} changed since it was shown; reload the page")
    labels = load_labels(root)
    labels[label.example_id] = label
    path = root / LABELS
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        for ex_id in sample:  # sample order, so the file diff reads naturally
            if ex_id in labels:
                fh.write(labels[ex_id].model_dump_json() + "\n")
    os.replace(tmp, path)


def score(root: Path) -> dict[str, Any]:
    sample = load_sample(root)
    labels = load_labels(root)
    counts: dict[str, int] = {}
    stale: list[str] = []
    missing: list[str] = []
    unsupported_answers: list[str] = []
    for e in sample:
        lab = labels.get(e.id)
        if lab is None:
            missing.append(e.id)
            continue
        if lab.content_digest != digest(e):
            stale.append(e.id)
            continue
        counts[lab.verdict] = counts.get(lab.verdict, 0) + 1
        if e.kind == "answer" and lab.verdict == "unsupported_answer":
            unsupported_answers.append(e.id)
    ok = counts.get("ok", 0)
    complete = len(sample) == N_SAMPLE and not missing and not stale
    return {
        "sample": len(sample),
        "labelled": sum(counts.values()),
        "verdicts": dict(sorted(counts.items())),
        "missing": missing,
        "stale": stale,
        "unsupported_answers": unsupported_answers,
        "passed": complete and ok >= PASS_OK and not unsupported_answers,
    }


PAGE = (Path(__file__).parent / "audit_page.html").read_text(encoding="utf-8")


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE, "text/html")
            elif self.path == "/api/items":
                labels = load_labels(root)
                items = []
                for e in load_sample(root):
                    lab = labels.get(e.id)
                    fresh = lab is not None and lab.content_digest == digest(e)
                    items.append(
                        {
                            **shown(e),
                            "digest": digest(e),
                            "verdict": lab.verdict if fresh and lab else None,
                            "note": lab.note if fresh and lab else "",
                        }
                    )
                self._send(200, json.dumps(items))
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self._send(404, "not found", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                save_label(root, AuditLabel.model_validate_json(self.rfile.read(n)))
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                self._send(400, str(exc).splitlines()[0], "text/plain")
                return
            self._send(200, '{"ok": true}')

    return Handler


def server(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only: the page is for the owner's own browser."""
    if not (root / SAMPLE).exists():
        raise ValueError(f"{SAMPLE} is missing; run ggh targets first")
    return ThreadingHTTPServer((HOST, port), make_handler(root))
