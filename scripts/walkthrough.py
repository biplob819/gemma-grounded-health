#!/usr/bin/env python3
"""Build the portfolio walkthrough GIF from committed outputs only.

Every screen is rendered from files in this repo (results, measurements, diagrams, the CLI
and the local audit page), captured at 1280x800 with headless Chrome and encoded with ffmpeg.
Numbers are read from the committed JSON, never typed in, so a re-run after v2 is published
updates the screens. No API keys, no network, no spend.

Usage:
    uv run python scripts/walkthrough.py                 # capture and write the GIF
    uv run python scripts/walkthrough.py --html-only DIR # write the HTML pages only
    uv run python scripts/walkthrough.py --summary docs/results/main/summary.json  # v1's run

Needs Google Chrome or Chromium (or GGH_CHROME=/path/to/chrome) and ffmpeg on PATH.
The pure parts (screen list, rendering, scrubbing) are tested in tests/test_walkthrough.py;
capture and encoding are not run in CI.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 1280, 800
NEUTRAL_REPO = "~/gemma-grounded-health"
NEUTRAL_HOME = "~"
NEUTRAL_USER = "user"

FFMPEG_FILTER = (
    "fps=10,scale=960:-1:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=5"
)
CREDENTIAL_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "KAGGLE", "HF_")

PRIMARY_JUDGE = "claude-opus-5-5"
PARITY_BAR = 0.95  # dev label agreement, Q4_K_M against bf16 (Tech Spec §6)
DISPLAY = {
    "gemma-e2b-base-q4": "Gemma 4 E2B base",
    "gemma-e2b-grounded-q4": "Gemma fine-tuned v1",
    "gemma-e2b-grounded-v2-q4": "Gemma fine-tuned v2",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gpt-6-luna": "GPT-6 Luna",
    "claude-opus-5-5": "Opus 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol",
}


def name(model_id: str) -> str:
    return DISPLAY.get(model_id, model_id)


# --------------------------------------------------------------------------- inputs


@dataclass
class Paths:
    """Every committed input, relative to the repo root."""

    # the shipped release (v2, R4.5); --summary points at another run
    summary: str = "docs/results/main_v2/summary.json"
    run: str = "docs/results/main_v2/run.json"
    bundle: str = "docs/results/results_bundle.json"
    # the v1 failure story stays on v1's run
    failure_summary: str = "docs/results/main/summary.json"
    citation_summary: str = "docs/results/main_citation/summary.json"
    laptop_bench: str = "docs/measurements/v2/laptop_bench.json"
    parity_headline: str = "docs/measurements/parity_headline_v2.json"
    parity_report: str = "docs/measurements/v2/parity.json"
    probe_report: str = "docs/measurements/probe_report_v2.json"
    audit_sample: str = "docs/measurements/audit_sample.jsonl"
    audit_labels: str = "docs/measurements/audit_labels.jsonl"
    dataflow: str = "docs/assets/dataflow.svg"
    hf_page: str = "docs/assets/screenshots/hf_model_page.png"
    demo: str = "docs/assets/screenshots/demo_answer.png"
    portfolio: str = "portfolio.yml"


def load_json(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def iter_jsonl_gz(root: Path, rel: str):
    with gzip.open(root / rel, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- scrubbing

DATA_URI = re.compile(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOME_RE = re.compile(r"(?:/Users|/home)/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+")


def scrub(text: str, root: Path, home: str | None = None, user: str | None = None) -> str:
    """Replace the repo path, home folder and local username with neutral ones."""
    home = str(Path.home()) if home is None else home
    user = _user() if user is None else user
    out = text
    for r in {str(root), str(root.resolve())}:
        out = out.replace(r, NEUTRAL_REPO)
    if home and home not in ("/", "~"):
        out = out.replace(home, NEUTRAL_HOME)
    out = HOME_RE.sub(NEUTRAL_HOME, out)
    if user and len(user) >= 3:
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", NEUTRAL_USER, out)
    return out


def _user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError):  # no passwd entry
        return ""


def find_leaks(text: str, home: str | None = None, user: str | None = None) -> list[str]:
    """What must never appear in a frame: paths, the username, emails, key-like strings."""
    home = str(Path.home()) if home is None else home
    user = _user() if user is None else user
    leaks: list[str] = []
    if home and home not in ("/", "~") and home in text:
        leaks.append("home path")
    if HOME_RE.search(text):
        leaks.append("user folder path")
    if user and len(user) >= 3 and re.search(rf"(?<![A-Za-z0-9]){re.escape(user)}", text):
        leaks.append("local username")
    if EMAIL_RE.search(text):
        leaks.append("email address")
    try:
        import secret_scan

        leaks += secret_scan._scan_text("frame", text)
    except ImportError:  # run outside scripts/ on the path
        pass
    return leaks


# --------------------------------------------------------------------------- page chrome

CSS = """
:root{--bg:#f7f7f5;--fg:#1c1917;--mute:#6b665f;--line:#e4e1dc;--card:#fff;--acc:#0f766e;
--warn:#b45309;--bad:#b91c1c;--good:#15803d;--hi:#fef3c7;--term:#14161a}
*{box-sizing:border-box}
html,body{margin:0;width:1280px;height:800px;overflow:hidden}
body{background:var(--bg);color:var(--fg);font:16px/1.45 -apple-system,"Helvetica Neue",
Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:baseline;gap:16px;padding:26px 44px 10px}
header .proj{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--acc);
font-weight:700}
header h1{font-size:28px;margin:0;font-weight:700;letter-spacing:-.01em}
header .num{margin-left:auto;color:var(--mute);font-size:14px;font-variant-numeric:tabular-nums}
.sub{padding:0 44px 14px;color:var(--mute);font-size:16px}
main{flex:1;padding:0 44px;min-height:0}
footer{display:flex;justify-content:space-between;padding:10px 44px 18px;color:var(--mute);
font-size:13px}
footer code{font-size:12px}
code,.mono{font-family:"SF Mono",Menlo,Consolas,monospace}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);
text-align:left;font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:0}
td.n{text-align:right} th.n{text-align:right} td.nw{white-space:nowrap}
.ci{color:var(--mute);font-size:13px}
.big{font-size:44px;font-weight:700;letter-spacing:-.02em;line-height:1.05}
.lab{color:var(--mute);font-size:14px}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:700}
.sep{background:#dcfce7;color:var(--good)} .tie{background:#f1f0ee;color:var(--mute)}
.grid{display:grid;gap:18px}
b.acc{color:var(--acc)}
.hl{background:var(--hi);padding:1px 4px;border-radius:4px}
"""


def page(
    title: str,
    body: str,
    *,
    num: int,
    total: int,
    source: str,
    sub: str = "",
    extra_css: str = "",
) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}{extra_css}</style></head><body>"
        '<header><span class="proj">Gemma Grounded Health</span>'
        f"<h1>{html.escape(title)}</h1>"
        f'<span class="num">{num} / {total}</span></header>{sub_html}'
        f"<main>{body}</main>"
        f"<footer><span>source: <code>{html.escape(source)}</code></span>"
        "<span>Research only, not medical advice</span></footer></body></html>"
    )


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def ci(m: dict[str, Any] | None) -> str:
    if not m:
        return '<span class="ci">n/a</span>'
    return f'{f3(m["point"])} <span class="ci">[{f3(m["lo"])}, {f3(m["hi"])}]</span>'


def interval_svg(m: dict[str, Any], lo: float = 0.0, hi: float = 1.0, w: int = 150) -> str:
    def x(v: float) -> float:
        return 6 + (w - 12) * (v - lo) / (hi - lo)

    return (
        f'<svg width="{w}" height="18" viewBox="0 0 {w} 18">'
        f'<line x1="6" x2="{w - 6}" y1="9" y2="9" stroke="#e4e1dc" stroke-width="2"/>'
        f'<line x1="{x(m["lo"]):.1f}" x2="{x(m["hi"]):.1f}" y1="9" y2="9" stroke="#0f766e" '
        'stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="{x(m["point"]):.1f}" cy="9" r="5" fill="#0f766e"/></svg>'
    )


# --------------------------------------------------------------------------- model picks


def base_and_tuned(models: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The base Gemma and the newest fine-tuned Gemma in a summary."""
    gemma = [m for m in models if m["model"].startswith("gemma")]
    base = next(m for m in gemma if "base" in m["model"])
    tuned = [m for m in gemma if m is not base]
    if not tuned:
        raise ValueError("summary has no fine-tuned Gemma model")
    return base, tuned[-1]


def judges_of(summary: dict[str, Any]) -> list[str]:
    for m in summary["models"]:
        if isinstance(m.get("headline"), dict):
            return sorted(m["headline"], key=lambda j: (j != PRIMARY_JUDGE, j))
    return []


# --------------------------------------------------------------------------- screens


@dataclass
class Ctx:
    root: Path
    paths: Paths
    num: int = 0
    total: int = 0
    audit_url: str = "about:blank"
    cache: dict[str, Any] = field(default_factory=dict)

    def json(self, rel: str) -> Any:
        if rel not in self.cache:
            self.cache[rel] = load_json(self.root, rel)
        return self.cache[rel]


def gemma_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Base first, then each fine-tuned Gemma in the summary's order (v1, v2)."""
    base, _ = base_and_tuned(models)
    return [base] + [m for m in models if m["model"].startswith("gemma") and m is not base]


def _metric(m: dict[str, Any], key: str, j: str) -> dict[str, Any] | None:
    v = m.get(key)
    return v.get(j) if isinstance(v, dict) and isinstance(v.get(j), dict) else None


def title_metrics(s: dict[str, Any], j: str) -> list[tuple[str, str, bool]]:
    """(label, summary key, lower is better) for each tile the run measured."""
    out = [("Label accuracy", "accuracy", False),
           (f"Correct and fully grounded ({name(j)})", "headline", False)]  # fmt: skip
    if any(_metric(m, "citation_precision", j) for m in s["models"]):
        out += [
            (f"Cited sentences that support a claim ({name(j)})", "citation_precision", False),
            (f"Answers citing an unsupporting aim sentence ({name(j)})", "aim_citation_rate", True),
        ]
    else:
        out.append((f"Claims supported by cited sentences ({name(j)})", "groundedness_mean", False))
    return out


def screen_title(c: Ctx) -> str:
    p = c.paths
    port = yaml.safe_load((c.root / p.portfolio).read_text(encoding="utf-8"))
    s, run = c.json(p.summary), c.json(p.run)
    models = gemma_models(s["models"])
    j = judges_of(s)[0]

    def point(m: dict[str, Any], key: str) -> float | None:
        v = m["accuracy"] if key == "accuracy" else _metric(m, key, j)
        return None if v is None else v["point"]

    def tile(label: str, key: str, lower: bool) -> str:
        vals = " <span style='color:var(--mute)'>&middot;</span> ".join(
            f'<b class="acc">{f3(point(m, key))}</b>' if m is models[-1] else f3(point(m, key))
            for m in models
        )
        names = " &middot; ".join(html.escape(name(m["model"])) for m in models)
        hint = " (lower is better)" if lower else ""
        return (
            f'<div class="card"><div class="lab">{html.escape(label + hint)}</div>'
            f'<div class="big" style="font-size:34px">{vals}</div>'
            f'<div class="lab">{names}</div></div>'
        )

    chips = (
        '<div style="margin-top:22px;display:flex;flex-wrap:wrap;gap:8px">'
        + "".join(
            f'<span class="badge tie" style="font-size:14px;padding:5px 12px">{html.escape(t)}'
            "</span>"
            for t in port.get("stack", [])
        )
        + "</div>"
    )
    tiles = "".join(tile(*t) for t in title_metrics(s, j))
    cols = 2 if len(title_metrics(s, j)) == 4 else 3
    body = (
        f'<p style="font-size:22px;line-height:1.4;max-width:1080px;margin:4px 0 20px">'
        f"{html.escape(port['tagline'])}</p>"
        f'<div class="grid" style="grid-template-columns:repeat({cols},1fr)">{tiles}</div>'
        f'<p class="lab" style="margin-top:18px;font-size:16px">Main run '
        f"<code>{html.escape(run['run_id'])}</code>: {run['n']:,} expert-labelled PubMedQA "
        f"questions, {len(run['models'])} models, judged by "
        f"{', '.join(name(x) for x in run['judges'])}, ${run['spent_usd']:.2f} spent. "
        "Evaluated with EvalKit; the verdicts are on the results screen.</p>" + chips
    )
    return page(
        "A small model that cites its evidence",
        body,
        num=c.num,
        total=c.total,
        source=f"{p.portfolio}, {p.summary}, {p.run}",
    )


def _svg_screen(c: Ctx, rel: str, title: str, sub: str) -> str:
    svg = (c.root / rel).read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<svg\b", '<svg style="width:100%;height:100%"', svg, count=1)
    body = f'<div class="card" style="height:100%;padding:8px">{svg}</div>'
    css = "main{padding-bottom:4px}"
    return page(title, body, num=c.num, total=c.total, source=rel, sub=sub, extra_css=css)


def _image_screen(c: Ctx, rel: str, title: str, sub: str) -> str:
    """A committed screenshot, inlined (no file path in the page), top of the image shown."""
    data = base64.b64encode((c.root / rel).read_bytes()).decode("ascii")
    mime = "image/png" if rel.endswith(".png") else "image/gif"
    body = (
        '<div class="card" style="height:100%;padding:0;overflow:hidden">'
        f'<img alt="" src="data:{mime};base64,{data}" style="width:100%;height:100%;'
        'object-fit:cover;object-position:top"></div>'
    )
    css = "main{padding-bottom:4px}"
    return page(title, body, num=c.num, total=c.total, source=rel, sub=sub, extra_css=css)


def screen_hf(c: Ctx) -> str:
    return _image_screen(
        c,
        c.paths.hf_page,
        "Released on Hugging Face",
        "Merged weights, the LoRA adapter and a card generated from the results bundle; the "
        "GGUF repo holds the benchmarked Q4_K_M file.",
    )


def screen_demo(c: Ctx) -> str:
    return _image_screen(
        c,
        c.paths.demo,
        "The local demo: base against v2, from the abstract only",
        "Served by llama-server on a laptop; sample answers need no passcode, live answers do.",
    )


def screen_dataflow(c: Ctx) -> str:
    return _svg_screen(
        c,
        c.paths.dataflow,
        "Data flow",
        "PubMedQA to training targets, Kaggle training, GGUF export, laptop serving and "
        "EvalKit's dual-judge evaluation.",
    )


METRIC_LABEL = {
    "accuracy": "Label accuracy",
    "headline": "Correct and fully grounded",
    "citation_precision": "Cited sentences that support a claim",
    "aim_citation_rate": "Aim-citation rate (lower is better)",
}


def preregistered(c: Ctx, run_id: str) -> list[dict[str, Any]]:
    """The bundle's pre-registered comparisons on this run, each with `aim_dependent` set when
    its verdict changes under the post-hoc aim-rule count (OWNER D-62)."""
    from ggh.release import card, sources

    bundle = c.json(c.paths.bundle)
    names = {r["name"] for r in bundle.get("runs", []) if r.get("run_id") == run_id}
    comps = [x for x in bundle.get("primary_comparisons", []) if x.get("run") in names]
    if not comps:
        return []
    src = sources.load(c.root, sources.Paths(bundle=Path(c.paths.bundle)))
    flips = card.aim_dependent(src)
    return [
        {**x, "aim_dependent": (x["model_a"], x["model_b"], x["metric"], x["judge"] or "") in flips}
        for x in comps
    ]


def _verdict_badge(cmp: dict[str, Any]) -> str:
    if cmp.get("verdict") == "tie":
        badge = '<span class="badge tie">tie</span>'
    else:
        a_better = cmp.get("verdict") == "a_better"
        who = name(cmp["model_a"] if a_better else cmp["model_b"])
        badge = f'<span class="badge sep">{html.escape(who)} better</span>'
    if cmp.get("aim_dependent"):
        badge += ' <span class="ci">depends on the aim rule (OWNER D-62)</span>'
    return badge


def screen_results(c: Ctx) -> str:
    p = c.paths
    s = c.json(p.summary)
    judges = judges_of(s)
    j0 = judges[0]
    base, ft = base_and_tuned(s["models"])
    cit = any(_metric(m, "citation_precision", j0) for m in s["models"])
    head = (
        "<tr><th>Model</th><th>Label accuracy</th>"
        + "".join(f"<th>Correct and fully grounded, {name(j)}</th>" for j in judges)
        + f"<th>{name(j0)} interval</th>"
        + (
            f"<th>Citation precision ({name(j0)})</th><th>Aim-citation rate ({name(j0)})</th>"
            if cit
            else f"<th class='n'>Claims supported ({name(j0)})</th>"
        )
        + "</tr>"
    )
    rows = []
    for m in s["models"]:
        nm = html.escape(name(m["model"]))
        extra = (
            f"<td class='nw'>{ci(_metric(m, 'citation_precision', j0))}</td>"
            f"<td class='nw'>{ci(_metric(m, 'aim_citation_rate', j0))}</td>"
            if cit
            else f"<td class='n'>{f3(m['groundedness_mean'][j0]['point'])}</td>"
        )
        rows.append(
            "<tr>"
            + (f"<td class='nw'><b>{nm}</b></td>" if m is ft else f"<td class='nw'>{nm}</td>")
            + f"<td class='nw'>{ci(m['accuracy'])}</td>"
            + "".join(f"<td class='nw'>{ci(m['headline'].get(j))}</td>" for j in judges)
            + f"<td>{interval_svg(m['headline'][j0], 0.2, 0.5, w=120)}</td>"
            + extra
            + "</tr>"
        )
    comps = []
    pre = preregistered(c, s["run_id"])
    if pre:
        caption = "difference [95% paired bootstrap, Holm]"
        for cmp in pre:
            d = cmp["diff"]
            comps.append(
                f"<tr><td class='nw'>{html.escape(name(cmp['model_a']))} minus "
                f"{html.escape(name(cmp['model_b']))}</td>"
                f"<td>{METRIC_LABEL.get(cmp['metric'], cmp['metric'])}</td>"
                f"<td class='n nw'>{d['point']:+.3f} <span class='ci'>[{d['lo']:+.3f}, "
                f"{d['hi']:+.3f}]</span></td><td>{_verdict_badge(cmp)}</td></tr>"
            )
    else:  # a run the bundle does not carry: EvalKit's run-wide base-vs-fine-tune tests
        caption = "fine-tuned minus base [95% paired bootstrap, Holm]"
        for cmp in s["comparisons"]:
            if {cmp["model_a"], cmp["model_b"]} != {base["model"], ft["model"]}:
                continue
            sign = -1 if cmp["model_a"] == base["model"] else 1  # fine-tuned minus base
            d = cmp["diff"]
            lo, hi = sorted((sign * d["lo"], sign * d["hi"]))
            metric = METRIC_LABEL["accuracy" if cmp["metric"] == "accuracy" else "headline"]
            judge = f" ({name(cmp['judge'])})" if cmp["judge"] else ""
            badge = (
                '<span class="badge sep">separated</span>'
                if cmp["separated"]
                else '<span class="badge tie">tie</span>'
            )
            comps.append(
                f"<tr><td>{html.escape(name(ft['model']))} minus base</td>"
                f"<td>{metric}{html.escape(judge)}</td><td class='n'>{sign * d['point']:+.3f} "
                f"<span class='ci'>[{lo:+.3f}, {hi:+.3f}]</span></td><td>{badge}</td></tr>"
            )
    css = "td{padding:7px 8px;font-size:14.5px} th{padding:6px 8px}"
    body = (
        f'<div class="card" style="padding:10px 14px"><table>{head}{"".join(rows)}</table></div>'
        '<div class="card" style="margin-top:12px;padding:10px 14px"><table>'
        f"<tr><th>Comparison</th><th>Outcome</th><th class='n'>{caption}</th><th>verdict</th>"
        f"</tr>{''.join(comps)}</table></div>"
    )
    return page(
        f"Results: {len(s['models'])} models, one prompt, one run",
        body,
        num=c.num,
        total=c.total,
        source=f"{p.summary} (run {s['run_id']}), {p.bundle}",
        sub=f"All {base['n']:,} expert-labelled PubMedQA questions. 95% bootstrap intervals; "
        "a gap counts only when its corrected interval excludes zero.",
        extra_css=css,
    )


AIM_RATIONALE = re.compile(
    r"research question|question being (?:studied|asked)|\b(?:aim|objective|purpose)\b"
    r"|reports? no (?:result|finding)",
    re.IGNORECASE,
)


def failure_stats(c: Ctx) -> dict[str, Any]:
    """Unsupported claims (primary judge) and how many cite sentence 1, per model."""
    p = c.paths
    run_dir = Path(p.failure_summary).parent.as_posix()
    stats: dict[str, dict[str, int]] = {}
    example: dict[str, Any] | None = None
    _, ft = base_and_tuned(c.json(p.failure_summary)["models"])
    for r in iter_jsonl_gz(c.root, f"{run_dir}/records.jsonl.gz"):
        model = r["completion"]["model"]
        st = stats.setdefault(model, {"unsupported": 0, "cite_s1": 0, "only_s1": 0})
        claims = {cl["id"]: cl for cl in r["claims"]}
        for v in r["verdicts"]:
            if v["judge"] != PRIMARY_JUDGE or v["verdict"] == "supported":
                continue
            cited = claims[v["claim_id"]]["cited"]
            st["unsupported"] += 1
            st["cite_s1"] += 1 in cited
            st["only_s1"] += cited == [1]
            if (
                example is None
                and model == ft["model"]
                and cited == [1]
                and AIM_RATIONALE.search(v["rationale"])
            ):
                example = {
                    "case_id": r["case_id"],
                    "claim": claims[v["claim_id"]]["text"],
                    "rationale": v["rationale"],
                }
    if example is not None:
        for case in iter_jsonl_gz(c.root, f"{run_dir}/cases.jsonl.gz"):
            if case["id"] == example["case_id"]:
                example["question"] = case["question"]
                example["sentence1"] = case["context"][0]
                break
    return {"stats": stats, "example": example}


def screen_failure(c: Ctx) -> str:
    p = c.paths
    s = c.json(p.failure_summary)
    base, ft = base_and_tuned(s["models"])
    fs = failure_stats(c)
    st, ex = fs["stats"], fs["example"]
    cit = {m["model"]: m for m in c.json(p.citation_summary)["models"]}
    j = PRIMARY_JUDGE

    def stat(m: dict[str, Any]) -> str:
        x = st.get(m["model"], {"unsupported": 0, "cite_s1": 0})
        rate = cit.get(m["model"], {}).get("aim_citation_rate", {}).get(j)
        prec = cit.get(m["model"], {}).get("citation_precision", {}).get(j)
        return (
            f'<div class="card"><div class="lab">{html.escape(name(m["model"]))}</div>'
            f'<div class="big">{x["cite_s1"]} <span style="font-size:22px;color:var(--mute)">'
            f"of {x['unsupported']}</span></div>"
            '<div class="lab">unsupported claims that cite sentence 1</div>'
            f'<div style="margin-top:10px">Citations pointing at the aim: <b>'
            f"{pct(rate['point']) if rate else 'n/a'}</b><br>Cited sentences that support "
            f"the claim: <b>{pct(prec['point']) if prec else 'n/a'}</b></div></div>"
        )

    ex_html = ""
    if ex:
        ex_html = (
            '<div class="card" style="font-size:15px">'
            '<div class="lab">One fine-tuned answer, as judged</div>'
            f"<p style='margin:8px 0'><b>Question:</b> {html.escape(ex['question'])}</p>"
            f"<p style='margin:8px 0'><b>Cited [1]:</b> <span class='hl'>"
            f"{html.escape(ex['sentence1'])}</span></p>"
            f"<p style='margin:8px 0'><b>Claim:</b> {html.escape(ex['claim'])}</p>"
            f"<p style='margin:8px 0;color:var(--bad)'><b>{name(j)}:</b> unsupported. "
            f"&ldquo;{html.escape(ex['rationale'])}&rdquo;</p></div>"
        )
    body = (
        '<div class="grid" style="grid-template-columns:1fr 1fr 1.5fr;align-items:start">'
        f"{stat(ft)}{stat(base)}{ex_html}</div>"
        '<p style="margin-top:20px;font-size:17px;max-width:1150px">Sentence 1 is usually the '
        "study's aim. v1's training citations were picked by word overlap with the conclusion, "
        "so the question ranked high. v2 trains on result sentences only and is chosen on a dev "
        "grounding check, never the test set.</p>"
    )
    return page(
        "What failed: citing the question, not the result",
        body,
        num=c.num,
        total=c.total,
        source=f"{Path(p.failure_summary).parent.as_posix()}/records.jsonl.gz, "
        f"{p.citation_summary}",
        sub=f"Label accuracy rose from {f3(base['accuracy']['point'])} to "
        f"{f3(ft['accuracy']['point'])}, but claims supported fell from "
        f"{f3(base['groundedness_mean'][j]['point'])} to "
        f"{f3(ft['groundedness_mean'][j]['point'])} ({name(j)} judge).",
    )


def screen_laptop(c: Ctx) -> str:
    p = c.paths
    b = c.json(p.laptop_bench)
    variants: dict[str, dict[str, Any]] = b["variants"]
    details: dict[str, dict[str, Any]] = b.get("details", {})
    top = max(v["gen_tps_p50"] for v in variants.values())
    rows = []
    for k, v in variants.items():
        w = 220 * v["gen_tps_p50"] / top
        rows.append(
            f"<tr><td><code>{html.escape(k)}</code></td>"
            f"<td class='n'>{v['model_file_mb'] / 1024:.1f} GB</td>"
            f"<td><svg width='230' height='16'><rect width='{w:.0f}' height='16' rx='3' "
            f"fill='#0f766e'/></svg> {v['gen_tps_p50']:.0f}</td>"
            f"<td class='n'>{v['ttft_ms_p50']} ms</td>"
            f"<td class='n'>{details.get(k, {}).get('answer_tokens_p50', 'n/a')}</td>"
            f"<td class='n'>{v['answer_latency_ms_p50']} / {v['answer_latency_ms_p95']} ms</td>"
            f"<td class='n'>{v['peak_rss_mb'] / 1024:.1f} GB</td></tr>"
        )
    first = next(iter(variants.values()))
    hw = (
        f"{first['chip']}, {first['ram_gb']} GB, {first['os']}, "
        f"llama.cpp {first['llama_cpp_commit']}, on {first['power'].upper()} power. "
        f"{first['n_prompts']} dev prompts x "
        f"{first['repetitions']} repetitions, measured {b['measured_at'][:10]}."
    )
    body = (
        '<div class="card"><table><tr><th>Variant</th><th class="n">File</th>'
        "<th>Generation, tokens/s (p50)</th><th class='n'>First token (p50)</th>"
        "<th class='n'>Answer tokens (p50)</th><th class='n'>Answer latency p50 / p95</th>"
        "<th class='n'>Peak memory</th></tr>"
        f"{''.join(rows)}</table></div>"
        f'<p class="lab" style="font-size:16px;margin-top:18px">{html.escape(hw)}</p>'
        '<p style="font-size:17px">Runs offline through <code>ggh serve</code> (llama.cpp, '
        "JSON-schema constrained, greedy). Prompts are processed fresh each time.</p>"
    )
    return page(
        "Laptop benchmark: offline, on one machine",
        body,
        num=c.num,
        total=c.total,
        source=p.laptop_bench,
    )


def screen_checks(c: Ctx) -> str:
    p = c.paths
    ph, pr, probe = c.json(p.parity_headline), c.json(p.parity_report), c.json(p.probe_report)
    judges = [k.removeprefix("headline_") for k in ph["bf16"] if k.startswith("headline_")]
    prow = "".join(
        f"<tr><td>Correct and fully grounded ({name(j)})</td><td class='n'>"
        f"{f3(ph['bf16']['headline_' + j])}</td><td class='n'>{f3(ph['q4_k_m']['headline_' + j])}"
        f"</td><td class='n'>{ph['bf16_minus_q4_headline'][j]['point']:+.3f} <span class='ci'>"
        f"[{ph['bf16_minus_q4_headline'][j]['lo']:+.3f}, "
        f"{ph['bf16_minus_q4_headline'][j]['hi']:+.3f}]</span></td></tr>"
        for j in judges
    )
    acc = (
        f"<tr><td>Label accuracy</td><td class='n'>{f3(ph['bf16']['accuracy'])}</td>"
        f"<td class='n'>{f3(ph['q4_k_m']['accuracy'])}</td><td></td></tr>"
    )
    mrows = "".join(
        f"<tr><td>{html.escape(name(m))}</td><td class='n'>{f3(v['accuracy_original'])}</td>"
        f"<td class='n'>{f3(v['accuracy_paraphrase'])}</td><td class='n'>"
        f"{v['drop']['point']:+.2f} <span class='ci'>[{v['drop']['lo']:+.2f}, "
        f"{v['drop']['hi']:+.2f}]</span></td><td>"
        + (
            '<span class="badge" style="background:#fee2e2;color:var(--bad)">signal</span>'
            if v["contamination_signal"]
            else '<span class="badge tie">no signal</span>'
        )
        + "</td></tr>"
        for m, v in probe["models"].items()
    )
    body = (
        '<div class="grid" style="grid-template-columns:1fr">'
        f'<div class="card"><div class="lab">Laptop file (Q4_K_M) against full precision (bf16), '
        f"{ph['n']} test questions. Dev label agreement on {pr['n']}: "
        f"<b>{f3(pr['label_agreement'])}</b> (bar {f3(PARITY_BAR)}"
        + ("" if pr["label_agreement"] >= PARITY_BAR else "; missed, disclosed, OWNER D-55")
        + ")</div><table>"
        "<tr><th></th><th class='n'>bf16</th><th class='n'>Q4_K_M</th>"
        "<th class='n'>bf16 minus Q4 [95%]</th></tr>"
        f"{acc}{prow}</table></div>"
        f'<div class="card"><div class="lab">Contamination probe: the same '
        f"{next(iter(probe['models'].values()))['n']} questions reworded</div><table>"
        "<tr><th>Model</th><th class='n'>Original</th><th class='n'>Paraphrased</th>"
        "<th class='n'>Drop [95%]</th><th>Memorisation</th></tr>"
        f"{mrows}</table></div></div>"
    )
    return page(
        "Checks: quantisation parity and a paraphrase probe",
        body,
        num=c.num,
        total=c.total,
        source=f"{p.parity_headline}, {p.parity_report}, {p.probe_report}",
    )


def cli_env() -> dict[str, str]:
    """A clean environment: no credentials, neutral home, fixed width."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "VIRTUAL_ENV")
    }
    env = {k: v for k, v in env.items() if not any(h in k for h in CREDENTIAL_HINTS)}
    env["COLUMNS"] = "104"
    env["HOME"] = tempfile.gettempdir()
    return env


def run_ggh(root: Path, args: list[str]) -> str:
    code = "import sys; sys.argv[0] = 'ggh'; from ggh.cli import main; sys.exit(main())"
    out = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=root,
        env=cli_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    return out.stdout + out.stderr


TERM_CSS = """
.term{background:var(--term);color:#e6e6e6;border-radius:12px;height:100%;overflow:hidden;
box-shadow:0 8px 30px rgba(0,0,0,.18)}
.term .bar{height:34px;background:#24272d;border-radius:12px 12px 0 0;display:flex;
align-items:center;gap:8px;padding:0 14px}
.term .bar i{width:12px;height:12px;border-radius:50%;display:inline-block}
.term pre{margin:0;padding:14px 20px;font:13.5px/1.5 "SF Mono",Menlo,Consolas,monospace;
white-space:pre-wrap}
.term .p{color:#5eead4} .term .ok{color:#86efac} .term .info{color:#93c5fd}
main{padding-bottom:4px}
"""


def terminal(blocks: list[tuple[str, str]], root: Path) -> str:
    parts = []
    for cmd, out in blocks:
        text = html.escape(scrub(out, root).rstrip())
        text = text.replace("[ok]", '<span class="ok">[ok]</span>')
        text = text.replace("[info]", '<span class="info">[info]</span>')
        parts.append(f'<span class="p">{NEUTRAL_REPO} $</span> {html.escape(cmd)}\n{text}\n')
    dots = "".join(f'<i style="background:{c}"></i>' for c in ("#ff5f57", "#febc2e", "#28c840"))
    joined = "\n".join(parts)
    return f'<div class="term"><div class="bar">{dots}</div><pre>{joined}</pre></div>'


def trim_help(text: str) -> str:
    """Drop argparse's usage block and the wrapped {a,b,c} choice list; keep the table."""
    keep: list[str] = []
    in_usage = in_choices = False
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("usage:"):
            in_usage = True
        if in_usage:
            in_usage = bool(stripped)
            continue
        if stripped.startswith("{"):
            in_choices = True
        if in_choices:
            in_choices = not stripped.endswith("}")
            continue
        keep.append(line)
    return "\n".join(keep).strip("\n")


def screen_cli_help(c: Ctx) -> str:
    body = terminal([("uv run ggh --help", trim_help(run_ggh(c.root, ["--help"])))], c.root)
    return page(
        "One CLI for the whole pipeline",
        body,
        num=c.num,
        total=c.total,
        source="uv run ggh --help (usage line trimmed)",
        extra_css=TERM_CSS,
    )


def screen_cli_doctor(c: Ctx) -> str:
    body = terminal(
        [
            ("uv run ggh doctor", run_ggh(c.root, ["doctor"])),
            ("uv run ggh audit score", run_ggh(c.root, ["audit", "score"])),
        ],
        c.root,
    )
    return page(
        "No keys needed to check, test or audit",
        body,
        num=c.num,
        total=c.total,
        source=f"uv run ggh doctor; uv run ggh audit score ({c.paths.audit_labels})",
        sub="Credentials are only read for paid evaluation and releases; this run has none.",
        extra_css=TERM_CSS,
    )


AUDIT_CSS = """
.frame{height:100%;border:1px solid var(--line);border-radius:12px;overflow:hidden;
background:#fafaf9;position:relative}
.frame iframe{border:0;width:1490px;height:1000px;transform:scale(.8);transform-origin:0 0;
position:absolute;left:0;top:0}
main{padding-bottom:4px}
"""


def screen_audit(c: Ctx) -> str:
    """The real `ggh audit serve` page, served on 127.0.0.1 and framed at 80%."""
    from ggh.targets import audit

    sc = audit.score(c.root)  # reads the committed labels; writes nothing
    ok = sc["verdicts"].get("ok", 0)
    verdict = "passed" if sc["passed"] else "not passed"
    body = f'<div class="frame"><iframe src="{html.escape(c.audit_url)}"></iframe></div>'
    return page(
        f"Hand audit before training: {ok} of {sc['sample']} ok, {verdict}",
        body,
        num=c.num,
        total=c.total,
        source=f"ggh audit serve (127.0.0.1) over {c.paths.audit_sample}",
        sub="Cited sentences highlighted. It checked each citation set as a whole; the model "
        "later learned to cite single aim sentences.",
        extra_css=AUDIT_CSS,
    )


@dataclass
class Screen:
    slug: str
    title: str
    render: Callable[[Ctx], str]
    caption: str
    live: bool = False  # needs the local audit server while captured


def screens() -> list[Screen]:
    """The walkthrough, in order. The audit page is served live from committed data."""
    return [
        Screen("title", "Overview", screen_title, "Tagline and v2's headline numbers."),
        Screen("results", "Results", screen_results, "v2 run table, pre-registered verdicts."),
        Screen("failure", "Failure diagnosis", screen_failure, "Why v1 lost on grounding."),
        Screen("hf", "Hugging Face", screen_hf, "The released model's page."),
        Screen("demo", "Demo", screen_demo, "The local demo answering a sample question."),
        Screen("dataflow", "Data flow", screen_dataflow, "docs/assets/dataflow.svg."),
        Screen("laptop", "Laptop benchmark", screen_laptop, "Speed and memory on one laptop."),
        Screen("checks", "Checks", screen_checks, "Quantisation parity and paraphrase probe."),
        Screen("doctor", "Doctor", screen_cli_doctor, "`ggh doctor` and `ggh audit score`."),
        Screen(
            "audit",
            "Audit page",
            screen_audit,
            "`ggh audit serve` on 127.0.0.1, the 50-example hand audit.",
            live=True,
        ),
    ]


def render_all(
    root: Path, paths: Paths, out_dir: Path, audit_url: str = "about:blank"
) -> list[tuple[Screen, Path]]:
    """Write each screen as an HTML page, refusing any page that would leak a path or key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = screens()
    ctx = Ctx(root, paths, total=len(ss), audit_url=audit_url)
    done: list[tuple[Screen, Path]] = []
    for i, s in enumerate(ss, 1):
        ctx.num = i
        text = s.render(ctx)
        leaks = find_leaks(DATA_URI.sub("", text))
        if leaks:
            raise SystemExit(f"walkthrough: screen {s.slug} would show {', '.join(leaks)}")
        path = out_dir / f"{i:02d}-{s.slug}.html"
        path.write_text(text, encoding="utf-8")
        done.append((s, path))
    return done


# --------------------------------------------------------------------------- capture


def find_chrome() -> str | None:
    env = os.environ.get("GGH_CHROME")
    if env:
        return env
    for exe in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(exe):
            return found
    for app in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ):
        if Path(app).exists():
            return app
    return None


def require_tools() -> tuple[str, str]:
    chrome, ffmpeg = find_chrome(), shutil.which("ffmpeg")
    missing = []
    if not chrome:
        missing.append("Google Chrome or Chromium (or set GGH_CHROME to its binary)")
    if not ffmpeg:
        missing.append("ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    if missing:
        raise SystemExit("walkthrough: missing " + "; ".join(missing))
    assert chrome and ffmpeg
    return chrome, ffmpeg


def capture(chrome: str, url: str, png: Path, profile: Path) -> None:
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--force-color-profile=srgb",
        "--force-device-scale-factor=1",
        f"--user-data-dir={profile}",
        f"--window-size={WIDTH},{HEIGHT}",
        "--virtual-time-budget=4000",
        f"--screenshot={png}",
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not png.exists():
        raise SystemExit(f"walkthrough: Chrome wrote no screenshot for {url}")


def audit_server(root: Path, paths: Paths, tmp: Path):
    """Serve the audit page from a copy of the committed sample, so no click can write."""
    from ggh.targets import audit

    copy = tmp / "audit-root"
    for rel in (paths.audit_sample, paths.audit_labels):
        dst = copy / audit.SAMPLE.parent / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dst)
    leaks = find_leaks((copy / audit.SAMPLE).read_text(encoding="utf-8"))
    if leaks:
        raise SystemExit(f"walkthrough: the audit sample would show {', '.join(leaks)}")
    srv = audit.server(copy, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def build(root: Path, paths: Paths, out_gif: Path, frames: Path | None) -> None:
    chrome, ffmpeg = require_tools()
    with tempfile.TemporaryDirectory(prefix="ggh-walk-") as t:
        tmp = Path(t)
        frames = frames or tmp / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for old in frames.glob("seq*.png"):
            old.unlink()
        srv = audit_server(root, paths, tmp)
        try:
            host, port = srv.server_address[:2]
            rendered = render_all(root, paths, tmp / "html", f"http://{host!s}:{port}/")
            for i, (s, html_path) in enumerate(rendered, 1):
                capture(chrome, html_path.as_uri(), frames / f"seq{i:02d}.png", tmp / "profile")
                print(f"captured {i:02d} {s.slug}")
        finally:
            srv.shutdown()
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "1/2.5",
                "-i",
                "seq%02d.png",
                "-vf",
                FFMPEG_FILTER,
                "-loop",
                "0",
                str(out_gif.resolve()),
            ],
            cwd=frames,
            check=True,
        )
    shown_path = out_gif.relative_to(root) if out_gif.is_relative_to(root) else out_gif.name
    size = out_gif.stat().st_size / 1e6
    print(f"wrote {shown_path} ({size:.2f} MB, {len(rendered)} screens)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="docs/assets/walkthrough.gif")
    ap.add_argument("--frames", default=None, help="keep the PNG frames in this folder")
    ap.add_argument("--html-only", default=None, help="write the HTML pages here and stop")
    ap.add_argument("--summary", default=None, help="results summary.json (default: v2 main)")
    ap.add_argument("--citation-summary", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    paths = Paths()
    if args.summary:  # e.g. v1's run; the failure screen stays on v1's run
        paths.summary = args.summary
        paths.run = f"{Path(args.summary).parent.as_posix()}/run.json"
    if args.citation_summary:
        paths.citation_summary = args.citation_summary
    if args.html_only:
        for s, p in render_all(root, paths, Path(args.html_only)):
            print(f"{s.slug}: {p.name}")
        return 0
    out = Path(args.out)
    frames = Path(args.frames) if args.frames else None
    build(root, paths, out if out.is_absolute() else root / out, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
