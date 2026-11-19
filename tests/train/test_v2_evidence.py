"""V2.1: v2's (`ft2`) committed local checks — dev parity of the Q4_K_M GGUF against the merged
bf16 model, the laptop benchmark beside v1's, and the read sample (VTG-06). These check the
committed files (CI-safe); dev_eval is consulted only where it exists locally."""

import gzip
import json
from pathlib import Path

import pytest

from ggh.data.build import read_jsonl
from ggh.export import parity
from ggh.schemas import LaptopBench, ParityReport
from ggh.serve import bench as B
from ggh.serve import llama
from ggh.train.devrun import DEV_EVAL
from ggh.versions import VERSIONS

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / VERSIONS["ft2"].measurements
RUN_STATE = (ROOT / "specs/V2.1-retrain/RUN-STATE.md").read_text(encoding="utf-8")


def _lines(path: Path) -> list[dict]:
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if path.suffix == ".gz" else raw.decode("utf-8")
    return [json.loads(x) for x in text.splitlines() if x]


DEV = _lines(M / "dev_generations.jsonl.gz")
SAMPLE = _lines(M / "dev_generations_sample.jsonl")
V1_SAMPLE = _lines(ROOT / "docs/measurements/dev_generations_sample.jsonl")


# ---- dev parity ----

REPORT, GENS, PSAMPLE = (ROOT / p for p in parity.paths("ft2"))


def test_v2_parity_on_500_dev_prompts_recomputes_and_its_fail_is_recorded():
    rep = ParityReport.model_validate_json(REPORT.read_text(encoding="utf-8"))
    rows = _lines(GENS)
    assert rep.n == len(rows) == 500
    # v1's bars, unchanged: Q4_K_M v2 misses the agreement bar; schema validity passes
    assert parity.failures(rep) == [f"label agreement {rep.label_agreement:.3f} < 0.95"]
    assert "**FAIL** on the 0.95 agreement bar" in RUN_STATE
    assert parity.build_report([r["bf16"] for r in rows], [r["q4_k_m"] for r in rows]) == rep
    assert [r["pmid"] for r in rows] == [r["pmid"] for r in DEV]
    assert _lines(PSAMPLE) == rows[::25]
    for r in rows:
        assert "<|channel>" not in r["q4_k_m"] and r["q4_k_m"].startswith('{"label": ')
    flips = sum(json.loads(r["bf16"])["label"] != json.loads(r["q4_k_m"])["label"] for r in rows)
    assert f"{rep.label_agreement:.3f}" in RUN_STATE and f"{flips} of 500" in RUN_STATE


def test_the_q8_0_diagnostic_is_against_the_same_bf16_replies_and_recomputes():
    rows, q8 = _lines(GENS), _lines(M / "parity_q8_0_generations.jsonl.gz")
    rep = ParityReport.model_validate_json((M / "parity_q8_0.json").read_text(encoding="utf-8"))
    assert [r["pmid"] for r in q8] == [r["pmid"] for r in rows]
    assert parity.build_report([r["bf16"] for r in rows], [r["q8_0"] for r in q8]) == rep
    assert parity.passes(rep) and f"agrees on {rep.label_agreement:.3f}" in RUN_STATE


def test_v1_parity_files_are_untouched():
    v1 = ParityReport.model_validate_json((ROOT / parity.REPORT).read_text(encoding="utf-8"))
    assert v1.n == 500 and f"{v1.label_agreement:.3f}" == "0.956"


# ---- laptop benchmark ----

BENCH = ROOT / B.V2_OUT


@pytest.mark.skipif(not BENCH.exists(), reason="v2 laptop benchmark not recorded")
def test_v2_bench_ran_the_protocol_beside_v1_in_one_session():
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    res = {k: LaptopBench.model_validate(v) for k, v in data["variants"].items()}
    assert set(res) == {f"{m}-{q}" for m, q in B.V2_VARIANTS}
    assert data["protocol"] == B.PROTOCOL and data["measured_at"].endswith("Z")
    man = json.loads((M / "gguf_ft2_manifest.json").read_text(encoding="utf-8"))
    for k, r in res.items():
        assert r.n_prompts == 50 and r.repetitions == 3 and r.power == "ac"
        assert (
            r.llama_cpp_commit == llama.TAG and len(data["details"][k]["llama_bench_rounds"]) == 3
        )
        if k.startswith("ft2-"):
            assert r.model_file_mb == round(man["files"][f"{k}.gguf"]["bytes"] / 2**20, 1)
    for on in ("tg128", "request"):
        for q, g in B.per_token_gap(res, on, ("ft", "ft2")).items():
            assert f"{100 * g:.1f} % ({q}" in RUN_STATE


# ---- read the sample (VTG-06) ----


def test_the_v2_sample_is_every_25th_row_on_the_same_prompts_as_v1s():
    assert DEV[::25] == SAMPLE and len(SAMPLE) == 20
    assert [r["pmid"] for r in SAMPLE] == [r["pmid"] for r in V1_SAMPLE]


def test_the_v2_sample_reading_is_noted():
    ans = [json.loads(r["constrained"]) for r in SAMPLE]
    right = sum(a["label"] == r["label"] for a, r in zip(ans, SAMPLE, strict=True))
    single = sum(len(a["citations"]) == 1 for a in ans)
    first = sum(1 in a["citations"] for a in ans)
    words = [len(a["answer"].split()) for a in ans]
    note = RUN_STATE[RUN_STATE.index("Read the sample (VTG-06") :]
    assert f"{right}/20 labels right" in note and f"{single}/20 cite exactly one" in note
    assert f"{first} of 20 cite sentence [1]" in note
    assert f"{min(words)} to {max(words)} words" in note
    for pmid in ("23472862", "24896137", "20579518"):  # the two misses, the background cite
        assert pmid in note
    if (ROOT / DEV_EVAL).exists():  # gitignored; built locally by `ggh data`
        dev = {str(r["pmid"]): r for r in read_jsonl(ROOT / DEV_EVAL)}
        aim = {"OBJECTIVE", "BACKGROUND"}

        def cites_aim(rows):
            return sum(bool(aim & {dev[r["pmid"]]["sections"][i - 1]
                                   for i in json.loads(r["constrained"])["citations"]})
                       for r in rows)  # fmt: skip

        assert f"{cites_aim(SAMPLE)} of 20 cite an OBJECTIVE or BACKGROUND" in note
        assert f"v1: {cites_aim(V1_SAMPLE)} of 20" in note
        assert f"{cites_aim(DEV)} of 500" in note
