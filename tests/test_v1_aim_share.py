"""The case study's v1 training-answer counts trace to committed evidence (P5.1 known issue)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_the_aim_share_is_measured_on_v1s_exact_training_set():
    m = _json("docs/measurements/v1_targets_aim_share.json")
    lineage = _json("docs/measurements/lineage/targets.json")
    assert m["input_sha256"] == lineage["outputs"]["artifacts/targets/train.jsonl"]
    assert m["answers"] == sum(m["answer_labels"].values()) == m["kinds"]["answer"]
    assert m["share"] == round(m["answers_citing_an_aim_sentence"] / m["answers"], 4)


def test_7508_is_the_yes_and_no_quota_of_v1s_training_set():
    quotas = _json("docs/measurements/targets_drop_report_v1.json")["quotas"]["train"]
    m = _json("docs/measurements/v1_targets_aim_share.json")
    assert quotas["yes"] + quotas["no"] == m["answers"] == 7508


def test_case_study_and_data_card_print_the_measured_count():
    m = _json("docs/measurements/v1_targets_aim_share.json")
    n, share = f"{m['answers_citing_an_aim_sentence']:,}", f"{m['share'] * 100:.1f} %"
    for rel in ("docs/case-study.md", "DATA_CARD.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert n in text and share in text and "v1_targets_aim_share.json" in text, rel
        assert "3,785" not in text, rel
