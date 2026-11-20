"""UPS-01 to UPS-03 on the committed evidence of the EvalKit contribution (R4.4, OWNER D-64).
No network: the record names the merged PR, the live page and EvalKit's gates."""

import json
from pathlib import Path

from ggh.release import links

ROOT = Path(__file__).resolve().parents[2]
REC = json.loads((ROOT / "docs/measurements/evalkit_upstream_r44.json").read_text("utf-8"))


def test_the_evalkit_pr_was_merged_as_node_b517():
    assert REC["evalkit_node"] == "B5.17" and REC["pr_state"] == "MERGED"
    assert (
        REC["pr"] == links.LINKS["evalkit_pr"][1] == "https://github.com/biplob819/evalkit/pull/1"
    )
    assert REC["merge_commit"] and REC["label_fix_commit"]


def test_the_gemma_report_page_is_live_and_linked():
    assert REC["page_http"] == 200 and "success" in REC["pages_run"]
    assert REC["page"] == links.LINKS["evalkit_report"][1]
    shot = (ROOT / REC["screenshot"]).read_bytes()
    assert shot[:8] == b"\x89PNG\r\n\x1a\n" and len(shot) > 50_000


def test_evalkit_gates_and_verifier_passed_and_both_approvals_are_recorded():
    assert "3275 passed" in REC["evalkit_gates"] and "12/12" in REC["evalkit_gates"]
    assert "PASS" in REC["evalkit_verifier"] and "green" in REC["evalkit_verifier"]
    assert any("EvalKit's own session" in a for a in REC["approvals"])
    assert REC["issue"].endswith("/issues/2")
