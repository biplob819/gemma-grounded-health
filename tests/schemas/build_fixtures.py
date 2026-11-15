"""Builds tests/fixtures/schemas/*.json: one valid instance per schema (run: python -m ...).

Kept as code so the fixtures are regenerated, not hand-edited; a test checks they match."""

from __future__ import annotations

import json
from pathlib import Path

H = "a" * 64
G = "b" * 40
IV = {"point": 0.5, "lo": 0.4, "hi": 0.6}

DEV = {
    "n": 500,
    "schema_valid": 0.99,
    "schema_valid_unconstrained": 0.9,
    "citation_valid": 0.99,
    "has_citation": 1.0,
    "label_accuracy": {"yes": 0.8, "no": 0.7},
    "mean_words": 42.0,
    "over_80_words": 0.0,
}
TRAIN_RUN = {
    "config_hash": H,
    "data_manifest_hash": H,
    "repo_commit": G,
    "kaggle_kernel_version": "3",
    "gpu": "Tesla T4",
    "wall_clock_s": 9000.0,
    "peak_vram_gb": 11.5,
    "steps": 1250,
    "selected_step": 1000,
    "val_loss_curve": [[250, 1.2], [500, 1.0], [750, 0.95], [1000, 0.94]],
    "dev_metrics": DEV,
}
PARITY = {
    "n": 500,
    "label_agreement": 0.97,
    "citation_exact_match": 0.9,
    "schema_valid": {"q4": 1.0, "bf16": 1.0},
    "judged_headline_delta": {"opus": IV},
}
BENCH = {
    "chip": "Apple M3",
    "ram_gb": 16,
    "os": "macOS 26.1",
    "power": "ac",
    "llama_cpp_commit": "b1234",
    "quant": "Q4_K_M",
    "model_file_mb": 3100.0,
    "n_prompts": 50,
    "repetitions": 3,
    "prompt_tps_p50": 900.0,
    "gen_tps_p50": 45.0,
    "gen_tps_p95": 40.0,
    "ttft_ms_p50": 600,
    "answer_latency_ms_p50": 3500,
    "answer_latency_ms_p95": 5200,
    "peak_rss_mb": 3400.0,
    "llama_bench": {"pp512": 950.0, "tg128": 47.0},
}
ARTIFACT = {
    "kind": "gguf",
    "model": "ft",
    "quant": "Q4_K_M",
    "path": "artifacts/export/ft-Q4_K_M.gguf",
    "sha256": H,
    "llama_cpp_commit": "b1234",
    "parent_key": H,
}

FIXTURES = {
    "TrainExample": {
        "id": "pqa-a-21645374",
        "pmid": "21645374",
        "kind": "answer",
        "split": "train",
        "question": "Does X improve Y?",
        "context": ["Sentence one.", "Sentence two."],
        "prompt": "rendered prompt",
        "target": {"label": "yes", "answer": "It does.", "citations": [1, 2]},
        "source_label": "yes",
        "rule_ids": ["TGT-R2", "TGT-R6"],
        "support": 0.8,
        "nli_entail": 0.9,
    },
    "DropRecord": {"pmid": "123", "reason": "too_long"},
    "DataManifest": {
        "schema_version": "1.0",
        "dataset": "qiaojin/PubMedQA",
        "dataset_revision": "9001f2853fb87cab8d220904e0de81ac6973b318",
        "evalkit_version": "0.1.0+06591f8",
        "template_hash": H,
        "splitter_version": "06591f8",
        "seed": 7,
        "counts": {"train": {"yes": 10, "no": 6}},
        "drops": {"too_long": 2},
        "files": {"artifacts/data/train.jsonl": H},
        "offscope_eval_hash": H,
    },
    "AuditLabel": {"example_id": "pqa-a-1", "content_digest": H, "verdict": "ok", "note": ""},
    "RecipeConfig": None,  # loaded from configs/recipe.yaml instead
    "DevMetrics": DEV,
    "TrainRun": TRAIN_RUN,
    "ExportArtifact": ARTIFACT,
    "ParityReport": PARITY,
    "LaptopBench": BENCH,
    "BehaviourRow": {
        "rule": "B3",
        "model": "gemma-e2b-grounded-q4",
        "metric": "maybe_recall",
        "value": IV,
        "n": 110,
        "source": "ggh:b3",
    },
    "ResultsBundle": {
        "schema_version": "1.0",
        "evalkit_run_ids": ["r-1"],
        "evalkit_version": "0.1.0+06591f8",
        "judge_mode": "dual",
        "calibration_status": "calibrated",
        "gemma_family_check": {"opus": 0.9, "sol": 0.85},
        "summary_paths": ["docs/results/summary.json"],
        "primary": {"ft_minus_base:opus": IV},
        "behaviour": [
            {
                "rule": "B1",
                "model": "m",
                "metric": "schema_valid",
                "value": 0.99,
                "n": 1000,
                "source": "evalkit:n_errors",
            }
        ],
        "parity": PARITY,
        "bench": BENCH,
        "train": TRAIN_RUN,
        "audit_pass_rate": 0.94,
        "artifacts": [ARTIFACT],
        "lineage_ok": True,
    },
}


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "fixtures" / "schemas"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in FIXTURES.items():
        if data is not None:
            (out / f"{name}.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
