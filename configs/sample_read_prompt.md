# Sample reader (OWNER D-61, behaviour rule B5, Tech Spec §7.6)

You are reading a sample of scored records from an evaluation of a research-question assistant.
The assistant answers a yes/no/maybe question about one biomedical abstract with a label, a
short answer and citations to numbered abstract sentences. An automated pipeline then split
each answer into claims and a judge model marked each claim supported or not by the cited
sentences, and marked each cited sentence as supporting or not. Your job is to find out whether
the numbers this pipeline produces can be trusted, by reading real records the way a careful
human auditor would.

## Your task

Each item has:

- **key**: an opaque id. Copy it exactly.
- **question**, **reference_label** (the dataset's answer) and the assistant's **label**,
  **answer** and **citations**.
- **cited_sentences**: the abstract sentences the answer cited, by number.
- **claims**: the claims extracted from the answer, each with the judge's verdict, its
  rationale, and per cited sentence whether it supports the claim and the sentence's role
  (aim, background, method, result, conclusion).

Give each record one **finding**:

- **ok**: the record reads correctly. The claims faithfully split the answer, and the judge's
  verdicts are ones a careful reader of the cited sentences would agree with (you may disagree
  on a borderline call; that is still ok). Wrong or ungrounded answers are ok if the record
  scores them as wrong or ungrounded.
- **model_fault**: the record is scored correctly, and it shows a notable failure of the
  assistant worth quoting: citing an aim or method sentence for a finding, a claim the cited
  sentences do not support, a label that contradicts its own answer, advice, an answer far past
  80 words, or copying text unrelated to the question. Name the failure in the note.
- **pipeline_fault**: the numbers would be wrong because of the pipeline, not the assistant:
  claims that are not in the answer or that miss a main point of it, a judge verdict that is
  clearly wrong (supported when the cited text plainly does not say it, or unsupported when it
  plainly does), citation support pointing at the wrong sentence, sentence roles clearly
  mislabelled in a way that changes a score, a garbled or truncated answer, or the answer's
  JSON parsed wrongly. Say exactly what is wrong in the note.
- **unclear**: you cannot tell; say why in the note.

## Rules

1. **Read the cited sentences yourself** for every claim before agreeing with a verdict. Use only
   the cited sentences, as the judge was told to.
2. **A judge verdict you would call the other way only on a fine point is ok.** Mark a
   pipeline_fault only when the verdict is clearly wrong.
3. **Prefer the most serious finding:** pipeline_fault over model_fault over ok.
4. **Notes:** one or two sentences, quoting the words that matter. ok needs no note, but a short
   one is welcome.
5. **Judge each record alone** and never guess which model wrote it.

## Output

Return one JSON object per item, one per line (JSON Lines), in the order given, with exactly
these fields:

```
{"key": "<key>", "finding": "ok" | "model_fault" | "pipeline_fault" | "unclear", "note": "<text, may be empty for ok>"}
```

Read every item. Do not skip any, do not add items, and do not change keys.
