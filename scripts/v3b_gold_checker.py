#!/usr/bin/env python3
"""V3.3 provenance: scores v2's 2,722 gold train targets with the two v3b checker models (cited
sentences as evidence), from the local Hub cache. Its output was summarised into
docs/measurements/v3/gold_checker.json.
Usage: uv run --group nli python scripts/v3b_gold_checker.py OUT.json
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

dev = "mps" if torch.backends.mps.is_available() else "cpu"
rows = [
    json.loads(line)
    for line in Path("artifacts/targets/train.jsonl").read_text(encoding="utf-8").splitlines()
]
pairs = []
for r in rows:
    t = r["target"]
    cites = sorted(set(t["citations"]))
    pairs.append((" ".join(r["context"][i - 1] for i in cites), t["answer"]))
CK = {
    "minicheck_roberta": (
        "lytang/MiniCheck-RoBERTa-Large",
        "74c8919647e61ed0f71bc177d94f10930f090068",
        1,
    ),
    "deberta_large": (
        "cross-encoder/nli-deberta-v3-large",
        "bab4bc7178836f731dcfd18c06ca9def0a137712",
        1,
    ),
}
out = [{"id": r["id"], "label": r["target"]["label"], "kind": r["kind"]} for r in rows]
for name, (rid, rev, idx) in CK.items():
    path = (
        os.path.expanduser("~/.cache/huggingface/hub/models--")
        + f"{rid.replace('/', '--')}/snapshots/{rev}"
    )
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval().to(dev)
    t0 = time.time()
    for b in range(0, len(pairs), 32):
        bt = pairs[b : b + 32]
        enc = tok(
            [p for p, _ in bt],
            [h for _, h in bt],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            pr = torch.softmax(m(**enc).logits.float(), dim=-1)[:, idx].cpu().tolist()
        for j, s in enumerate(pr):
            out[b + j][name] = round(s, 4)
    print(name, round(time.time() - t0), "s", flush=True)
    del m
Path(sys.argv[1]).write_text(json.dumps(out), encoding="utf-8")
print("done")
