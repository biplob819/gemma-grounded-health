#!/usr/bin/env python3
"""Fail if anything that looks like an API key is in the repo.

Scans tracked and untracked (not ignored) files. Lines containing the marker
`secret-scan: allow` are skipped (for documented fake examples only).

Known limit: a key split across string literals ("sk-ant-" "abc...") is not detected.
Provider-side secret scanning on GitHub is the backstop for that case.

Usage:
    python scripts/secret_scan.py [paths...]
    python scripts/secret_scan.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from harness_lib import ROOT

# A token starts after a non-alphanumeric, or right after a JSON escape (\n \t \r \b \f \uXXXX)
# whose letter would otherwise look like part of the token.
_B = r"(?:(?<![A-Za-z0-9])|(?<=\\[nrtbf])|(?<=\\u[0-9a-fA-F]{4}))"
PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    # tokens may be glued to other text (x_sk-…, AKIA…_x): boundaries are "no letter or digit"
    "openai key": re.compile(rf"{_B}sk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{{32,}}"),
    "aws access key": re.compile(rf"{_B}(?:AKIA|ASIA)[0-9A-Z]{{16}}(?![A-Za-z0-9])"),
    "aws secret key": re.compile(
        r"(?i)aws_?secret_?(?:access_?)?key\W{0,4}[=:]\W{0,4}[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+])"
    ),
    "github token": re.compile(rf"{_B}gh[pousr]_[A-Za-z0-9]{{36,}}(?![A-Za-z0-9])"),
    "github fine-grained token": re.compile(rf"{_B}github_pat_[A-Za-z0-9_]{{50,}}"),
    "pypi token": re.compile(rf"{_B}pypi-[A-Za-z0-9_\-]{{50,}}"),
    "hugging face token": re.compile(rf"{_B}hf_[A-Za-z0-9]{{30,}}"),
    # KAGGLE_KEY followed within 60 characters on the same line by a 32-hex value; the gap may
    # hold anything but another 32-hex run (quotes, brackets, `value=`, `: str =`, a comment)
    "kaggle key": re.compile(
        r"(?i)kaggle_key(?:(?![0-9a-f]{32})[^\n]){0,60}?(?<![0-9a-f])(?P<v>[0-9a-f]{32})(?![0-9a-f])"
    ),
    "kaggle api token": re.compile(rf"{_B}KGAT_[A-Za-z0-9]{{20,}}"),
    # the key body that follows the header on the same line (JSON strings keep it there, after
    # an escaped \n) identifies one key from another; PEM files carry it on later lines instead
    "private key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----"
        r"(?:\\[nr]|\s)*(?P<v>[A-Za-z0-9+/=]{0,64})"
    ),
}
# Whole-file patterns (may span lines). kaggle.json is exactly {"username": …, "key": <32 hex>},
# compact, pretty-printed, or JSON-escaped inside a notebook cell (\"). Known false positive: any
# other object holding exactly a "username" and a 32-hex "key" looks identical and is flagged;
# mark such a fixture with the allow marker.
_Q = r"""(?:\\\\)*\\?["']"""  # a quote, optionally escaped once or more (\" or \\\" in raw JSON)
# Between kaggle.json tokens: whitespace, an escaped \n/\t/\r inside a JSON string, or the
# `",\n  "` that separates notebook source lines in raw JSON (pretty-printed AND escaped forms).
_S = r"""(?:\s|(?:\\\\)*\\[nrt]|",\s*")*"""
FILE_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "kaggle.json credentials": (
        re.compile(r"\{"),
        re.compile(
            rf"\{{{_S}{_Q}username{_Q}{_S}:{_S}{_Q}[^\"'\n\\]*{_Q}{_S},{_S}{_Q}key{_Q}{_S}:{_S}{_Q}(?P<v>[0-9a-fA-F]{{32}}){_Q}{_S}\}}"
            rf"|\{{{_S}{_Q}key{_Q}{_S}:{_S}{_Q}(?P<v2>[0-9a-fA-F]{{32}}){_Q}{_S},{_S}{_Q}username{_Q}{_S}:{_S}{_Q}[^\"'\n\\]*{_Q}{_S}\}}"
        ),
    ),
    # KAGGLE_KEY and its value split across lines (parenthesised, backslash-continued, setdefault):
    # within 80 characters and at most MAX_MULTILINE_BREAKS line breaks of the name
    "kaggle key (multi-line)": (
        re.compile(r"(?i)kaggle_key"),
        re.compile(
            r"(?is)kaggle_key(?:(?![0-9a-f]{32}).){1,80}?(?<![0-9a-f])(?P<v>[0-9a-f]{32})(?![0-9a-f])"
        ),
    ),
}
ALLOW = "secret-scan: allow"
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".whl",
    ".gz",
    ".zip",
    ".ico",
    ".gguf",
    ".safetensors",
    ".bin",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".arrow",
    ".npy",
    ".npz",
    ".h5",
    ".pkl",
    ".tar",
}


def candidate_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return [root / p for p in out.stdout.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        # build outputs and environments are skipped only at the top level (src/env/ is code);
        # caches and VCS folders anywhere; any virtualenv by its pyvenv.cfg
        top_skip = {"dist", "build", "artifacts", "venv", "env"}
        skip = {
            ".tox",
            ".nox",
            ".mypy_cache",
            ".ipynb_checkpoints",
            ".uv-cache",
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".hypothesis",
            ".ruff_cache",
            ".harness",
            ".evalkit",
        }
        venvs = [cfg.parent for cfg in root.rglob("pyvenv.cfg")]  # any virtualenv, any name
        return [
            p
            for p in root.rglob("*")
            if p.is_file()
            and not (skip & set(p.relative_to(root).parts))
            and p.relative_to(root).parts[0] not in top_skip
            and not any(part.startswith(".venv") for part in p.relative_to(root).parts)
            and not any(p.is_relative_to(v) for v in venvs)
        ]


TEXT_IMAGE_TYPES = ("image/svg+xml",)  # image mime types that are really text


def _notebook_parts(text: str) -> list[tuple[str, str]] | None:
    """A notebook as decoded text: (label suffix, text) parts. None when the file is not a
    parseable notebook, so the caller falls back to a raw scan.

    Per cell: the joined source, each output's joined text and traceback lines, and each
    textual mime value, as the user sees them; then everything else in the cell (metadata,
    JSON-valued outputs, evalue, attachments, …) rendered as indented JSON, which keeps every
    key on the same line as its value, as in the file itself. The rest of the document
    (notebook metadata, widget state) is rendered the same way under `meta`."""

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate keys")  # a decoded dict would hide one: scan raw
        return dict(pairs)

    try:
        nb = json.loads(text.lstrip("\ufeff"), object_pairs_hook=no_duplicates)
    except (ValueError, RecursionError):
        return None
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return None

    def lines_of(value: object, sep: str = "") -> str | None:
        """Joined text for a str or a list of only str; None for anything else."""
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(s, str) for s in value):
            return sep.join(value)
        return None

    class _Unrenderable(Exception):
        pass

    def as_json(value: object) -> str:
        try:
            return json.dumps(value, indent=1, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise _Unrenderable from exc  # never drop a part: the caller falls back to raw

    try:
        return _render_cells(nb, lines_of, as_json)
    except _Unrenderable:
        return None  # e.g. nesting too deep to render: scan the raw file instead


def _render_cells(nb: dict, lines_of, as_json) -> list[tuple[str, str]]:
    out: list[tuple[str, str | None]] = []
    for n, cell in enumerate(nb["cells"], start=1):
        where = f"cell{n}"
        if not isinstance(cell, dict):
            out.append((where, as_json(cell)))
            continue
        rest = dict(cell)
        src = lines_of(rest.get("source"))
        if src is not None:
            out.append((where, src))
            rest.pop("source")
        shown: list[str] = []  # what the cell printed, to pair with its source below
        outputs = rest.get("outputs")
        if isinstance(outputs, list):
            kept = []
            for output in outputs:
                if not isinstance(output, dict):
                    kept.append(output)
                    continue
                o = dict(output)
                for key, sep in (("text", ""), ("traceback", "\n")):
                    joined = lines_of(o.get(key), sep)
                    if joined is not None:
                        out.append((where, joined))
                        shown.append(joined)
                        o.pop(key)
                if isinstance(o.get("evalue"), str):  # an error's message is printed too
                    shown.append(o["evalue"])
                data = o.get("data")
                if isinstance(data, dict):
                    d = {}
                    for mime, value in data.items():
                        low = str(mime).lower()
                        is_image = low.startswith("image/") and low not in TEXT_IMAGE_TYPES
                        if is_image:
                            continue  # base64 pixels
                        is_json = low == "application/json" or low.endswith("+json")
                        joined = None if is_json else lines_of(value)
                        if joined is not None:
                            out.append((where, joined))
                            shown.append(joined)
                        else:  # JSON data (even a list of strings is data, not lines of text)
                            d[mime] = value
                    o["data"] = d
                kept.append(o)
            rest["outputs"] = kept
        if src is not None and shown:  # source then what it printed, as a reader sees them
            pieces = [src, *shown]
            out.append((where, "".join(s if s.endswith("\n") else s + "\n" for s in pieces)))
        out.append((where, as_json(rest)))
    out.append(("meta", as_json({k: v for k, v in nb.items() if k != "cells"})))
    return [(w, s) for w, s in out if s]


def _secret_core(m: re.Match[str], following: str = "") -> str:
    """Identity of the secret a match found. Each pattern names its secret in a `v` group
    (Kaggle forms: the 32-hex value, case-insensitive; private keys: the key body). A private
    key whose body is not on the header line takes its body from the text that follows. Any
    other token is its own identity."""
    groups = m.groupdict()
    value = next((v for k, v in groups.items() if k.startswith("v") and v), None)
    if m.re.pattern.startswith("-----BEGIN"):
        # the whole body up to -----END, skipping armor headers (Proc-Type, DEK-Info, Version…)
        rest = (m.group(0)[m.start("v") - m.start() :] if value else "") + "\n" + following
        rest = rest.replace("\\n", "\n").split("-----END", 1)[0]
        body = "".join(
            re.sub(r"[^A-Za-z0-9+/=]", "", ln) for ln in rest.split("\n") if ":" not in ln
        )
        return m.group(0).split("-----")[1] + ":" + body
    if value is not None:
        return value.lower()
    return m.group(0)


MAX_MULTILINE_BREAKS = 3  # a split KAGGLE_KEY value must sit within 3 line breaks of the name


def scan(files: list[Path]) -> list[str]:
    hits: list[str] = []
    for f in files:
        if f.suffix.lower() in SKIP_SUFFIXES or not f.is_file():
            continue
        try:
            # never skip a file for its encoding: undecodable bytes become U+FFFD
            text = f.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        parts = _notebook_parts(text) if f.suffix.lower() == ".ipynb" else None
        if parts is None:
            hits += _scan_text(str(f), text)
        else:  # a parseable notebook is scanned as the text it holds, never as raw JSON lines
            seen: set[tuple[str, str]] = set()
            for where, part in parts:
                for h, core in _scan_matches(f"{f}:{where}", part):
                    # parts overlap (a source and its pairing), so the same secret can be found
                    # twice; dedup on what matched, never on the label, so distinct secrets that
                    # happen to share a line number are each reported
                    key = (where, core)  # one report per secret per cell
                    if key not in seen:
                        seen.add(key)
                        hits.append(h)
    return hits


def _scan_text(label: str, text: str) -> list[str]:
    """Hit strings for one text; see `_scan_matches`."""
    return [hit for hit, _ in _scan_matches(label, text)]


def _scan_matches(label: str, text: str) -> list[tuple[str, str]]:
    """(hit, secret identity) pairs: line patterns on every line (every match, so two tokens on
    one line are two hits), then whole-text patterns not already reported on a line.

    Lines are counted on "\n" only, everywhere, so a form feed or U+2028 cannot shift the
    allow-marker window. Whole-text patterns are tried from every occurrence of their anchor,
    so one match cannot swallow a later secret, and the allow marker suppresses a match only
    when it sits on one of that match's own lines."""
    found: list[tuple[str, str]] = []
    lines = text.split("\n")
    for lineno, line in enumerate(lines, start=1):
        if ALLOW in line:
            continue
        for name, pat in PATTERNS.items():
            for m in pat.finditer(line):
                # only a private key needs the lines after it (its body); skip the join otherwise
                following = "\n".join(lines[lineno : lineno + 200]) if name == "private key" else ""
                found.append((f"{label}:{lineno}: possible {name}", _secret_core(m, following)))
    reported = {core for _, core in found}  # by secret, never by line or label
    for name, (anchor, pat) in FILE_PATTERNS.items():
        for a in anchor.finditer(text):
            m = pat.match(text, a.start())
            if not m:
                continue
            first = text.count("\n", 0, m.start()) + 1
            last = text.count("\n", 0, m.end()) + 1
            if name == "kaggle key (multi-line)" and not (
                1 <= last - first <= MAX_MULTILINE_BREAKS
            ):
                continue  # same-line values belong to the 60-character line pattern
            core = _secret_core(m)
            if core in reported:
                continue  # this exact secret was already reported in this text
            if not any(ALLOW in ln for ln in lines[first - 1 : last]):
                found.append((f"{label}:{first}: possible {name}", core))
                reported.add(core)
    return found


def self_test() -> int:
    fake_ant = "sk-ant-" + "a1B2" * 10
    fake_oai = "sk-proj-" + "Z9y8" * 10
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "leak.py"
        bad.write_text(f'KEY = "{fake_ant}"\nOTHER = "{fake_oai}"\n', encoding="utf-8")
        allowed = Path(tmp) / "doc.md"
        allowed.write_text(f"example: {fake_ant}  # {ALLOW}\n", encoding="utf-8")
        clean = Path(tmp) / "clean.py"
        clean.write_text('KEY = os.environ["ANTHROPIC_API_KEY"]\n', encoding="utf-8")
        checks = [
            ("detects leaked keys", len(scan([bad])) == 2),
            ("honours allow marker", scan([allowed]) == []),
            ("clean file passes", scan([clean]) == []),
        ]
    for name, ok in checks:
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'WRONG'}] {name}")
    print("self-test", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    files = args.paths or candidate_files(ROOT)
    hits = scan(files)
    if hits:
        print("secret scan: FAIL")
        for h in hits:
            print(f"  - {h}")
        return 1
    print(f"secret scan: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
