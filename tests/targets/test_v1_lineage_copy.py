"""v1's targets lineage (docs/measurements/lineage/targets.json) pins the drop report v1 was
built with. v2.x rewrote docs/measurements/targets_drop_report.json, so v1's copy is kept under
its own name and must still hash to what the lineage record says."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_v1_drop_report_copy_matches_the_v1_lineage_hash():
    lineage = json.loads(
        (ROOT / "docs/measurements/lineage/targets.json").read_text(encoding="utf-8")
    )
    want = lineage["outputs"]["docs/measurements/targets_drop_report.json"]
    got = hashlib.sha256((ROOT / "docs/measurements/targets_drop_report_v1.json").read_bytes())
    assert got.hexdigest() == want
