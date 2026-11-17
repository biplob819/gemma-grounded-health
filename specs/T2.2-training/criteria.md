# T2.2 acceptance criteria

**TRN-01** — One "Save and Run All" of the committed notebook in a fresh Kaggle session finishes inside one session. Its log and wall clock are stored in `docs/measurements/`, and the run used the committed notebook unchanged (its pinned commit is the one recorded).
**TRN-02** — The loss curve (train and validation, with gradient norms), the config hash, the seed and the data manifest hash are saved with the adapter: in the committed run record keyed by the adapter's frozen hash, which R4.2 publishes beside the adapter. The run loaded the pinned checkpoint, and every loss was finite.
**TRN-03** — Checkpoint selection follows §5.2. Candidates are tried in order of validation loss. The selected one passes all three dev bars (schema validity constrained, has-citation and citation validity, each ≥ 0.98), computed from greedy generations on the 500 `dev_eval` prompts by `dev_metrics`, and the rule is tested on fixtures.
**TRN-04** — `TrainRun` is written and validates. The adapter's hash is recorded before any PQA-L use, and the downloaded adapter matches it.
**TRN-05** — Read the sample: 20 dev generations from the selected checkpoint are read and noted in RUN-STATE.
**TRN-06** — The GPU ledger holds the session's hours, no fewer than its log span.
