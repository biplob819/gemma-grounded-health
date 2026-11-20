"""The V2.2 headline chart (docs/assets/headline_v2.svg), generated from a committed summary.

Three panels share one row per model (base, v1, v2): the correct-and-grounded headline,
citation precision and the aim-citation rate, each a point with its 95% bootstrap interval as
EvalKit reports it, under the plan's one judge (OWNER D-57: GPT-6.1 Sol, claims by GPT-6 Luna).
Every number is read from `summary.json` and printed beside its mark, so the chart never relies
on colour alone. Light and dark steps come from the dataviz reference palette (validated).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

MODELS = (
    ("gemma-e2b-base-q4", "Base"),
    ("gemma-e2b-grounded-q4", "v1"),
    ("gemma-e2b-grounded-v2-q4", "v2"),
)
PANELS = (
    ("headline", "Correct and fully grounded", "higher is better"),
    ("citation_precision", "Cited sentences that support a claim", "higher is better"),
    ("aim_citation_rate", "Answers citing an aim sentence", "unsupporting; lower is better"),
)
# categorical slots 1-3, light and dark (dataviz reference palette, validated)
COLOURS = (("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"))
W, PANEL_W, LABEL_W, ROW_H, TOP = 960, 270, 60, 34, 100


def _x(q: float, left: float, right: float) -> float:
    return left + q * (right - left)


def values(summary: dict[str, Any], judge: str) -> dict[str, dict[str, dict[str, float]]]:
    by = {m["model"]: m for m in summary["models"]}
    return {
        model: {metric: dict(by[model][metric][judge]) for metric, _, _ in PANELS}
        for model, _ in MODELS
    }


def svg(summary: dict[str, Any], judge: str = "gpt-6.1-sol", run_id: str = "") -> str:
    vals = values(summary, judge)
    height = TOP + ROW_H * len(MODELS) + 58
    css = [
        ".s{fill:#fcfcfb}.t{fill:#0b0b0b}.m{fill:#52514e}.g{stroke:#e1e0d9}.ax{stroke:#c3c2b7}",
        *(f".c{i}{{stroke:{lt};fill:{lt}}}" for i, (lt, _) in enumerate(COLOURS)),
        "@media (prefers-color-scheme:dark){.s{fill:#1a1a19}.t{fill:#f0efec}.m{fill:#c3c2b7}"
        ".g{stroke:#2c2c2a}.ax{stroke:#383835}"
        + "".join(f".c{i}{{stroke:{dk};fill:{dk}}}" for i, (_, dk) in enumerate(COLOURS))
        + "}",
        "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" role="img" aria-labelledby="t d">',
        '<title id="t">Base, v1 and v2 on 1,000 PubMedQA questions</title>',
        '<desc id="d">'
        + escape(
            "; ".join(
                f"{label}: "
                + ", ".join(
                    f"{name.lower()} {vals[m][k]['point']:.3f} "
                    f"[{vals[m][k]['lo']:.3f}, {vals[m][k]['hi']:.3f}]"
                    for k, name, _ in PANELS
                )
                for m, label in MODELS
            )
        )
        + "</desc>",
        f"<style>{''.join(css)}</style>",
        f'<rect class="s" width="{W}" height="{height}" rx="8"/>',
        '<text class="t" x="20" y="30" font-size="17" font-weight="600">'
        "v2 fixed the citations, not the headline</text>",
        '<text class="m" x="20" y="50" font-size="12.5">'
        + escape(
            "1,000 PubMedQA questions. Claims extracted by GPT-6 Luna, judged by GPT-6.1 Sol"
            " (OWNER D-57)."
        )
        + "</text>",
        '<text class="m" x="20" y="66" font-size="12.5">'
        + escape(f"Point and 95% bootstrap interval{'; EvalKit run ' + run_id if run_id else ''}.")
        + "</text>",
    ]
    for p, (metric, name, direction) in enumerate(PANELS):
        x0 = LABEL_W + 20 + p * (PANEL_W + 20)
        plot_l, plot_r = x0 + 8, x0 + PANEL_W - 58
        out.append(
            f'<text class="t" x="{x0}" y="{TOP - 8}" font-size="12.5" font-weight="600">'
            f"{escape(name)}</text>"
        )
        out.append(f'<text class="m" x="{x0}" y="{TOP + 6}" font-size="11">{direction}</text>')
        for tick in (0.0, 0.5, 1.0):
            tx = plot_l + tick * (plot_r - plot_l)
            out.append(
                f'<line class="g" x1="{tx:.1f}" x2="{tx:.1f}" y1="{TOP + 14}" '
                f'y2="{TOP + 14 + ROW_H * len(MODELS)}" stroke-width="1"/>'
            )
            out.append(
                f'<text class="m" x="{tx:.1f}" y="{TOP + 30 + ROW_H * len(MODELS)}" '
                f'font-size="11" text-anchor="middle">{tick:g}</text>'
            )
        for i, (model, label) in enumerate(MODELS):
            v = vals[model][metric]
            y = TOP + 14 + ROW_H * i + ROW_H / 2
            if p == 0:
                out.append(
                    f'<text class="t" x="20" y="{y + 4:.1f}" font-size="13" font-weight="600">'
                    f"{label}</text>"
                )
            lo, hi, pt = (_x(v[k], plot_l, plot_r) for k in ("lo", "hi", "point"))
            out.append(
                f'<line class="c{i}" x1="{lo:.1f}" x2="{hi:.1f}" '
                f'y1="{y:.1f}" y2="{y:.1f}" stroke-width="2" stroke-linecap="round"/>'
            )
            out.append(f'<circle class="c{i}" cx="{pt:.1f}" cy="{y:.1f}" r="5"/>')
            out.append(
                f'<text class="t" x="{plot_r + 10}" y="{y + 4:.1f}" font-size="12" '
                f'font-variant-numeric="tabular-nums">{v["point"]:.3f}</text>'
            )
        out.append(
            f'<line class="ax" x1="{plot_l}" x2="{plot_r}" y1="{TOP + 14 + ROW_H * len(MODELS)}" '
            f'y2="{TOP + 14 + ROW_H * len(MODELS)}" stroke-width="1"/>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write(root: Path, run: str = "main_v2", out: str = "docs/assets/headline_v2.svg") -> Path:
    folder = root / "docs" / "results" / run
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    run_id = json.loads((folder / "run.json").read_text(encoding="utf-8"))["run_id"]
    path = root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg(summary, run_id=run_id), encoding="utf-8")
    return path
