"""The v2 per-sentence hand audit (AUD-R4, OWNER D-45, VTG-07).

v1's audit gave one verdict per example and passed 50 of 50 while about half the answers
cited an aim sentence beside a result sentence. Here every cited sentence of an answer or
insufficient example gets its own verdict (supports, aim, method, weak), plus one overall
verdict per example (ok, wrong label, or, for an insufficient example, an answer that claims a
finding). The sample is drawn exactly as AUD-R1 drew v1's (`make.audit_sample`, seed 7) from the
v2 `train` split. v1's sample, labels and tests are untouched: they live in other files.

`ggh audit sample --v2` draws the sample, `ggh audit serve --v2` shows it on 127.0.0.1 and saves
every click, `ggh audit score --v2` applies the bar and writes `audit_v2_result.json`.
"""

from __future__ import annotations

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ggh.data.build import read_jsonl
from ggh.lineage import sha256_file
from ggh.schemas import Model, TrainExample
from ggh.schemas._base import NonBlank, Sha256
from ggh.targets import make
from ggh.targets.audit import HOST, N_SAMPLE, digest, shown

MEASURE = Path("docs") / "measurements"
SAMPLE = MEASURE / "audit_v2_sample.jsonl"
LABELS = MEASURE / "audit_v2_labels.jsonl"
RESULT = MEASURE / "audit_v2_result.json"
TRAIN = make.OUT_DIR / "train.jsonl"
DEFAULT_PORT = 8766  # not 8765: a v1 audit server may still be running there

# ---- the bar (AUD-R4; the owner may change it by a decision) ----
PASS_OK = 45  # overall verdicts `ok`, of 50
MAX_WEAK_ANSWERS = 3  # answer examples allowed exactly one `weak` sentence (all others `supports`)

SentenceVerdict = Literal["supports", "aim", "method", "weak"]
Overall = Literal["ok", "wrong_label", "claims_finding"]
JUDGED_KINDS = ("answer", "insufficient")  # off-scope refusals cite [1] by rule; overall only


class SentenceLabel(Model):
    """One example's labels, bound to the digest of exactly what the page showed (AUD-R3).
    A partial label (some sentences, no overall yet) is saved as the owner clicks."""

    example_id: NonBlank
    content_digest: Sha256
    sentences: dict[str, SentenceVerdict] = Field(default_factory=dict)  # "3" -> verdict
    overall: Overall | None = None
    note: str = ""


def needs_sentences(e: TrainExample) -> bool:
    return e.kind in JUDGED_KINDS


def draw(root: Path, seed: int = make.SEED) -> list[TrainExample]:
    """AUD-R1's draw on the v2 train split, refused unless that split is the one the committed
    targets report records."""
    report = json.loads((root / make.REPORT_COPY).read_text(encoding="utf-8"))
    want = report["files"].get(TRAIN.as_posix())
    if want is None or sha256_file(root / TRAIN) != want:
        raise ValueError(f"{TRAIN} is not the split {make.REPORT_COPY} records; rerun ggh targets")
    train = [TrainExample.model_validate(r) for r in read_jsonl(root / TRAIN)]
    if not any("TGT-R10" in e.rule_ids for e in train):
        raise ValueError("the targets are not v2 (no example carries TGT-R10)")
    sample = make.audit_sample(train, seed)
    path = root / SAMPLE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for e in sample:
            fh.write(json.dumps(json.loads(e.model_dump_json()), sort_keys=True) + "\n")
    return sample


def load_sample(root: Path) -> list[TrainExample]:
    return [TrainExample.model_validate(r) for r in read_jsonl(root / SAMPLE)]


def load_labels(root: Path) -> dict[str, SentenceLabel]:
    path = root / LABELS
    if not path.exists():
        return {}
    return {(lab := SentenceLabel.model_validate(r)).example_id: lab for r in read_jsonl(path)}


def _check(e: TrainExample, label: SentenceLabel) -> None:
    cited = {str(i) for i in e.target.citations}
    extra = set(label.sentences) - (cited if needs_sentences(e) else set())
    if extra:
        raise ValueError(f"{e.id}: sentences {sorted(extra)} are not judged for this example")
    if label.overall == "claims_finding" and e.kind != "insufficient":
        raise ValueError(f"{e.id}: claims_finding applies to insufficient examples only")


def save_label(root: Path, label: SentenceLabel) -> None:
    sample = {e.id: e for e in load_sample(root)}
    e = sample.get(label.example_id)
    if e is None:
        raise ValueError(f"{label.example_id} is not in the v2 audit sample")
    if label.content_digest != digest(e):
        raise ValueError(f"{label.example_id} changed since it was shown; reload the page")
    _check(e, label)
    labels = load_labels(root)
    labels[label.example_id] = label
    path = root / LABELS
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        for ex_id in sample:  # sample order
            if ex_id in labels:
                fh.write(labels[ex_id].model_dump_json() + "\n")
    os.replace(tmp, path)


def complete(e: TrainExample, lab: SentenceLabel) -> bool:
    if lab.overall is None:
        return False
    return not needs_sentences(e) or set(lab.sentences) == {str(i) for i in e.target.citations}


def score(root: Path) -> dict[str, Any]:
    """AUD-R4's bar. `passed` needs a full sample with a fresh, complete label for each."""
    sample = load_sample(root)
    labels = load_labels(root)
    missing, stale = [], []
    overall: dict[str, int] = {}
    sentences: dict[str, dict[str, int]] = {k: {} for k in JUDGED_KINDS}
    answer_fail: list[str] = []  # an aim or method sentence, or more than one weak
    answer_weak: list[str] = []  # exactly one weak sentence
    finding_claims: list[str] = []
    for e in sample:
        lab = labels.get(e.id)
        if lab is not None and lab.content_digest != digest(e):
            stale.append(e.id)
            continue
        if lab is None or not complete(e, lab):
            missing.append(e.id)
            continue
        assert lab.overall is not None
        overall[lab.overall] = overall.get(lab.overall, 0) + 1
        if lab.overall == "claims_finding":
            finding_claims.append(e.id)
        if not needs_sentences(e):
            continue
        tally = sentences[e.kind]
        for v in lab.sentences.values():
            tally[v] = tally.get(v, 0) + 1
        if e.kind == "answer":
            vs = list(lab.sentences.values())
            if any(v in ("aim", "method") for v in vs) or vs.count("weak") > 1:
                answer_fail.append(e.id)
            elif vs.count("weak") == 1:
                answer_weak.append(e.id)
    ok = overall.get("ok", 0)
    full = len(sample) == N_SAMPLE and not missing and not stale
    bars = {
        "every_label_present_and_fresh": full,
        f"overall_ok_at_least_{PASS_OK}": ok >= PASS_OK,
        "no_answer_cites_aim_or_method_or_two_weak": not answer_fail,
        f"at_most_{MAX_WEAK_ANSWERS}_answers_with_one_weak": len(answer_weak) <= MAX_WEAK_ANSWERS,
        "no_insufficient_answer_claims_a_finding": not finding_claims,
    }
    return {
        "audit": "v2 per-sentence (AUD-R4, OWNER D-45)",
        "sample": len(sample),
        "sample_sha256": sha256_file(root / SAMPLE) if (root / SAMPLE).exists() else None,
        "labelled": sum(overall.values()),
        "overall": dict(sorted(overall.items())),
        "sentences": {k: dict(sorted(v.items())) for k, v in sentences.items()},
        "answers_failing": answer_fail,
        "answers_with_one_weak": answer_weak,
        "insufficient_claiming_a_finding": finding_claims,
        "missing": missing,
        "stale": stale,
        "bars": bars,
        "passed": all(bars.values()),
    }


def write_result(root: Path) -> dict[str, Any]:
    """Score and write `audit_v2_result.json`; refused while a label is missing or stale."""
    r = score(root)
    if r["missing"] or r["stale"] or r["sample"] != N_SAMPLE:
        raise ValueError(
            f"{len(r['missing'])} labels missing or incomplete, {len(r['stale'])} bound to "
            f"changed text, sample of {r['sample']}; nothing written"
        )
    (root / RESULT).write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return r


PAGE = (Path(__file__).parent / "audit_v2_page.html").read_text(encoding="utf-8")


def items(root: Path) -> list[dict[str, Any]]:
    labels = load_labels(root)
    out = []
    for e in load_sample(root):
        lab = labels.get(e.id)
        fresh = lab if lab is not None and lab.content_digest == digest(e) else None
        out.append(
            {
                **shown(e),
                "digest": digest(e),
                "judge_sentences": needs_sentences(e),
                "sentences": dict(fresh.sentences) if fresh else {},
                "overall": fresh.overall if fresh else None,
                "note": fresh.note if fresh else "",
            }
        )
    return out


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
                self._send(200, json.dumps(items(root)))
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self._send(404, "not found", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                save_label(root, SentenceLabel.model_validate_json(self.rfile.read(n)))
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                self._send(400, str(exc).splitlines()[0], "text/plain")
                return
            self._send(200, '{"ok": true}')

    return Handler


def server(root: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only. A busy port is an error, never a takeover."""
    if not (root / SAMPLE).exists():
        raise ValueError(f"{SAMPLE} is missing; run ggh audit sample --v2 first")
    return ThreadingHTTPServer((HOST, port), make_handler(root))
