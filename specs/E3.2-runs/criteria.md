# E3.2 acceptance criteria

**RUN-01** — `configs/eval_plan.yaml` names the three primary comparisons per judge and the separation rule; its sha256 is committed before the main run starts, and a test checks the file still has that hash.
**RUN-02** — A `--dry-run` estimate is recorded for each paid run before it starts; runs over $25 are covered by an owner approval (D-17 for the main run).
**RUN-03** — The main run finishes on 1,000 PQA-L cases for the four models with both judges' verdicts complete, and its summary, run record and gzipped records are committed.
**RUN-04** — The bf16 fine-tune answers 200 seeded PQA-L cases through an OpenAI-compatible Transformers server with schema-constrained greedy decoding, judged like the main run, and its headline difference against Q4_K_M on the same cases is reported.
**RUN-05** — The four models answer the 20 frozen off-scope prompts, and the records are committed for the owner's B8 review.
**RUN-06** — A paraphrase probe set is built by EvalKit's documented method (D-42), and base and fine-tuned answer both wordings; accuracy on each and the paired difference are reported.
**RUN-07** — The spend ledger holds every run's cost, and the ledger total matches the runs' own cost records.
