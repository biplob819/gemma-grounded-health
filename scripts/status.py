#!/usr/bin/env python3
"""One-table build status, read live from graph.md and catalog.tsv, plus ledger totals.

Usage:
    python scripts/status.py          # every node: status, rows green, dependencies, ledgers
    python scripts/status.py E3.1     # one node: each capability and its status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness_lib import ROOT, HarnessError, load_catalog, load_graph, node_status

sys.path.insert(0, str(ROOT / "src"))  # ggh.ledger is stdlib-only, so this works before install
from ggh import ledger

LEDGERS = (
    ("API spend", "spend-ledger.jsonl", ledger.SPEND_FIELD, "${:.2f}"),
    ("Kaggle GPU", "gpu-ledger.jsonl", ledger.GPU_FIELD, "{:.2f} h"),
)


def ledger_totals(measurements: Path = ROOT / "docs" / "measurements") -> str:
    """One line with both ledger totals (D-22), validated by ggh.ledger.total.

    Raises ledger.LedgerError for a malformed ledger; callers report it, never print nan."""
    parts = []
    for label, name, field, fmt in LEDGERS:
        path = measurements / name
        n = ledger.count(path)
        parts.append(f"{label}: {fmt.format(ledger.total(path, field))} ({n} entries)")
    return "Ledgers — " + "; ".join(parts)


def render_graph(nodes, caps) -> str:
    lines = [f"{'Node':<6} {'Status':<12} {'Green':>7}  {'Title':<36} Depends on"]
    for n in nodes.values():
        rows = [c for c in caps if c.node == n.id]
        green = sum(c.status == "green" for c in rows)
        st = node_status(n.id, caps)
        ready = st in ("not-started", "spec") and all(
            node_status(d, caps) == "green" for d in n.depends_on
        )
        label = "ready" if ready else st
        lines.append(
            f"{n.id:<6} {label:<12} {green:>3}/{len(rows):<3}  "
            f"{n.title[:36]:<36} {', '.join(n.depends_on) or '—'}"
        )
    total = sum(c.status == "green" for c in caps)
    done = sum(node_status(n, caps) == "green" for n in nodes)
    lines.append(f"\n{done}/{len(nodes)} nodes green, {total}/{len(caps)} capabilities green")
    try:
        lines.append(ledger_totals())
    except ledger.LedgerError as exc:
        lines.append(f"Ledgers — ERROR: {exc}")
    return "\n".join(lines)


def render_node(node_id, nodes, caps) -> str:
    n = nodes[node_id]
    lines = [f"{n.id} — {n.title}  [{node_status(n.id, caps)}]", ""]
    for c in (c for c in caps if c.node == node_id):
        lines.append(f"  {c.cap_id:<7} {c.status:<12} {c.note[:90]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("node", nargs="?")
    args = ap.parse_args(argv)
    try:
        nodes, caps = load_graph(), load_catalog()
    except (HarnessError, OSError) as exc:
        print(f"status: cannot read inputs: {exc}")
        return 1
    if args.node:
        if args.node not in nodes:
            print(f"status: unknown node {args.node}")
            return 1
        print(render_node(args.node, nodes, caps))
    else:
        out = render_graph(nodes, caps)
        print(out)
        if "Ledgers — ERROR" in out:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
