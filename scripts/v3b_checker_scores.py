#!/usr/bin/env python3
"""v3b reward design (OWNER D-70): score the committed dev answers of base, v1 and v2 with three
local grounding checkers, owner-approved downloads at pinned revisions:
cross-encoder/nli-deberta-v3-large (Apache-2.0), lytang/MiniCheck-RoBERTa-Large and
lytang/MiniCheck-Flan-T5-Large (MIT). Evidence is
the answer's cited sentences or the whole abstract. Needs `uv sync --group nli`; MiniCheck weights
are pickled .bin files, loaded by transformers with torch's weights-only loader.
Usage: uv run --group nli python scripts/v3b_checker_scores.py OUT.json
"""

import gzip
import json
import sys
import time

sys.path.insert(0, "src")
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from ggh.rl import reward as R
from ggh.targets.nli import entailment_index

OUT = sys.argv[1]
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


RUNS = ("dev_ground_v1", "dev_ground_v2_sol")
cases = {c["id"]: c for d in RUNS for c in load(f"docs/results/{d}/cases.jsonl.gz")}
names = {
    "gemma-e2b-base-q4": "base",
    "gemma-e2b-grounded-q4": "v1",
    "gemma-e2b-grounded-v2-q4": "v2",
}
items = []
for d in RUNS:
    for r in load(f"docs/results/{d}/records.jsonl.gz"):
        c = cases[r["case_id"]]
        a = R.parse(r["completion"]["raw_text"] or "")
        cites = (
            sorted(set(a.citations))
            if a is not None
            and a.citations
            and all(1 <= i <= len(c["context"]) for i in a.citations)
            else []
        )
        items.append(
            dict(
                case=r["case_id"],
                model=names[r["completion"]["model"]],
                cited=" ".join(c["context"][i - 1] for i in cites) if cites else None,
                abstract=" ".join(c["context"]),
                answer=a.answer if a is not None else None,
            )
        )
pairs = {("cited", i["cited"], i["answer"]) for i in items if i["cited"]} | {
    ("abstract", i["abstract"], i["answer"]) for i in items if i["cited"]
}
pairs = sorted(pairs, key=lambda p: len(p[1]))
CK = {
    "deberta_large": (
        "cross-encoder/nli-deberta-v3-large",
        "bab4bc7178836f731dcfd18c06ca9def0a137712",
        ["model.safetensors", "config.json", "*.json", "spm.model", "*.model"],
    ),
    "minicheck_roberta": (
        "lytang/MiniCheck-RoBERTa-Large",
        "74c8919647e61ed0f71bc177d94f10930f090068",
        ["pytorch_model.bin", "*.json", "merges.txt", "vocab.json"],
    ),
    "minicheck_t5": (
        "lytang/MiniCheck-Flan-T5-Large",
        "96eafd01cee2d16cf81aaa2fb226b14f422a37b3",
        ["pytorch_model.bin", "*.json", "spiece.model"],
    ),
}
res = {}
for name, (rid, rev, pat) in CK.items():
    t0 = time.time()
    path = snapshot_download(rid, revision=rev, allow_patterns=pat)
    print(name, "downloaded", round(time.time() - t0), "s", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    scores = {}
    if name == "minicheck_t5":
        m = AutoModelForSeq2SeqLM.from_pretrained(path).eval().to(dev)
        for b in range(0, len(pairs), 16):
            batch = pairs[b : b + 16]
            enc = tok(
                ["predict: " + tok.eos_token.join([p, h]) for _, p, h in batch],
                max_length=2048,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            dec = torch.zeros((len(batch), 1), dtype=torch.long, device=dev)
            with torch.no_grad():
                lg = m(**enc, decoder_input_ids=dec).logits[:, 0, :]
            pr = torch.softmax(lg[:, [3, 209]].float(), dim=-1)[:, 1].cpu().tolist()
            for p, s in zip(batch, pr, strict=True):
                scores[json.dumps(p)] = s
    else:
        m = AutoModelForSequenceClassification.from_pretrained(path).eval().to(dev)
        idx = entailment_index(m.config.id2label) if name == "deberta_large" else 1
        print(name, "id2label", m.config.id2label, "using", idx, flush=True)
        for b in range(0, len(pairs), 16):
            batch = pairs[b : b + 16]
            enc = tok(
                [p for _, p, _ in batch],
                [h for _, _, h in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                lg = m(**enc).logits
            pr = torch.softmax(lg.float(), dim=-1)[:, idx].cpu().tolist()
            for p, s in zip(batch, pr, strict=True):
                scores[json.dumps(p)] = s
    res[name] = scores
    del m
    if dev == "mps":
        torch.mps.empty_cache()
    print(name, "scored", len(scores), "in", round(time.time() - t0), "s", flush=True)
    for i in items:
        for ev in ("cited", "abstract"):
            i[f"{name}_{ev}"] = (
                res[name][json.dumps((ev, i[ev], i["answer"]))] if i["cited"] else 0.0
            )
    with open(OUT, "w", encoding="utf-8") as f:
        drop = ("cited", "abstract", "answer")
        json.dump([{k: v for k, v in i.items() if k not in drop} for i in items], f)
print("done")
