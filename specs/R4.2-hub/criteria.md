# R4.2 criteria

**HUB-01** — The release refuses to push unless the v2 lineage chain verifies and the plan has no problems; the committed plan lists every file with its size and sha256.
**HUB-02** — The main repo holds the merged v2 weights, the adapter under `adapter/`, the card as README, the results bundle, the v2 lineage, LICENSE and NOTICE, each equal to its committed record.
**HUB-03** — The GGUF repo holds Q4_K_M and Q8_0 with hashes equal to the export records, a card linking the main repo, LICENSE and NOTICE.
**HUB-04** — Every pushed file is downloaded back from the release tag and its sha256 equals the plan; the record is committed.
**HUB-05** — `ollama run hf.co/biplob123/gemma-4-e2b-grounded-health-GGUF:Q4_K_M` answers one demo sample with valid JSON; the answer is recorded.
