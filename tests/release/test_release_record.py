"""HUB-02 to HUB-05 on the committed evidence of the real push (R4.2, 2026-11-20/21): the plan,
the download-back record and the Ollama answer. No network."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "docs/measurements"
OWNER = "biplob123"
MAIN = f"{OWNER}/gemma-4-e2b-grounded-health"
GGUF = f"{MAIN}-GGUF"
# R4.5 regenerated both cards (known issue R4.2) and re-uploaded them under OWNER D-65; the record
# holds the downloaded-back hashes, so nothing awaits re-upload
AWAITING_REUPLOAD: set[str] = set()


def _json(name):
    return json.loads((M / name).read_text(encoding="utf-8"))


def test_every_planned_file_was_downloaded_back_with_its_planned_hash():
    plan, rec = _json("release_plan_v2.json"), _json("release_record_v2.json")
    planned = {f"{f['repo']}/{f['path_in_repo']}": f["sha256"] for f in plan["files"]}
    got = {d["file"]: d["sha256"] for d in rec["downloads"]}
    assert plan["problems"] == [] and rec["problems"] == []
    assert set(got) == set(planned) and all(d["ok"] for d in rec["downloads"])
    same = {k: v for k, v in planned.items() if k not in AWAITING_REUPLOAD}
    assert {k: got[k] for k in same} == same
    for k in AWAITING_REUPLOAD:  # the plan already holds the regenerated card's hash
        assert planned[k] != got[k], f"{k} was re-uploaded: record it and empty the set"
    card = hashlib.sha256((ROOT / "MODEL_CARD.md").read_bytes()).hexdigest()
    assert planned[f"{MAIN}/README.md"] == card
    assert rec["release"] == "v2" and rec["tag"] == "v2.0.0"


def test_the_main_repo_holds_weights_adapter_card_bundle_lineage_and_legal_files():
    rec = _json("release_record_v2.json")
    main = {
        d["file"].removeprefix(MAIN + "/")
        for d in rec["downloads"]
        if d["file"].startswith(MAIN + "/")
    }
    need = {"model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja",
            "adapter/adapter_model.safetensors", "adapter/adapter_config.json", "README.md",
            "results_bundle.json", "lineage.json", "LICENSE", "NOTICE"}  # fmt: skip
    assert need <= main


def test_the_gguf_repo_holds_both_quants_with_the_export_hashes():
    rec = _json("release_record_v2.json")
    arts = {a["path"]: a["sha256"] for a in _json("export_artifacts_v2.json")}
    got = {d["file"].removeprefix(GGUF + "/"): d["sha256"] for d in rec["downloads"]
           if d["file"].startswith(GGUF + "/")}  # fmt: skip
    assert got["gemma-4-e2b-grounded-health-Q4_K_M.gguf"] == arts["artifacts/gguf/ft2-Q4_K_M.gguf"]
    assert got["gemma-4-e2b-grounded-health-Q8_0.gguf"] == arts["artifacts/gguf/ft2-Q8_0.gguf"]
    assert {"README.md", "LICENSE", "NOTICE"} <= set(got)


def test_ollama_answered_one_sample_with_a_parsed_label_and_citations():
    o = _json("release_record_v2.json")["ollama"]
    assert o["ref"] == f"hf.co/{GGUF}:Q4_K_M" and "--think=false" in o["command"]
    assert o["ok"] is True and o["label"] in {"yes", "no", "maybe"} and o["citations"]
    answer = json.loads(o["answer"])
    assert answer["label"] == o["label"] and answer["citations"] == list(o["citations"])


def test_the_first_ollama_attempt_and_its_fix_are_recorded():
    first = _json("release_record_v2.json")["ollama_first_attempt"]
    assert first["answer"].startswith("<|channel>thought") and first["label"] is None
