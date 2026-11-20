### main

| Rule | Metric | gemma-e2b-base-q4 | gemma-e2b-grounded-q4 | claude-haiku-5-5 | gpt-6-luna | Source |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | schema_failures_or_zero_citations | 0.005 | 0.000 | 0.000 | 0.001 | ggh:b1 |
| B2 | headline:claude-opus-5-5 | 0.418 [0.387, 0.448] | 0.388 [0.358, 0.419] | 0.345 [0.316, 0.374] | 0.431 [0.401, 0.462] | evalkit:headline |
| B2 | groundedness_mean:claude-opus-5-5 | 0.863 [0.844, 0.881] | 0.519 [0.489, 0.550] | 0.817 [0.802, 0.831] | 0.852 [0.837, 0.867] | evalkit:groundedness |
| B2 | headline:gpt-6.1-sol | 0.394 [0.364, 0.424] | 0.330 [0.301, 0.359] | 0.231 [0.205, 0.257] | 0.290 [0.262, 0.319] | evalkit:headline |
| B2 | groundedness_mean:gpt-6.1-sol | 0.819 [0.798, 0.839] | 0.444 [0.414, 0.475] | 0.737 [0.722, 0.753] | 0.768 [0.750, 0.786] | evalkit:groundedness |
| B3 | maybe_recall | 0.491 [0.398, 0.586] | 0.045 [0.010, 0.086] | 0.527 [0.430, 0.620] | 0.545 [0.450, 0.640] | ggh:b3 |
| B3 | maybe_precision | 0.127 [0.095, 0.161] | 0.227 [0.059, 0.409] | 0.205 [0.158, 0.254] | 0.152 [0.117, 0.188] | ggh:b3 |
| B4 | over_80_words_share | 0.013 | 0.000 | 0.143 | 0.000 | ggh:b4 |
| B5 | sample_read_ok_share | 0.300 | 0.450 | 0.550 | 0.950 | model_review |
| B6 | unsupported_claim_share:claude-opus-5-5 | 0.117 | 0.447 | 0.159 | 0.127 | evalkit:verdicts |
| B6 | unsupported_claim_share:gpt-6.1-sol | 0.156 | 0.524 | 0.238 | 0.207 | evalkit:verdicts |
| B7 | citation_validity | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.997, 1.000] | evalkit:citation_valid |

`model_review`: a blind review by Claude Opus 5.5 subagents, not the owner (OWNER D-61); B5 under `configs/sample_read_prompt.md`, B8 under `configs/offscope_review_prompt.md` (question and answer only).

### main_citation

| Rule | Metric | gemma-e2b-base-q4 | gemma-e2b-grounded-q4 | claude-haiku-5-5 | gpt-6-luna | Source |
| --- | --- | --- | --- | --- | --- | --- |
| B7 | citation_precision:claude-opus-5-5 | 0.725 [0.698, 0.752] | 0.338 [0.319, 0.357] | 0.848 [0.836, 0.860] | 0.865 [0.851, 0.879] | evalkit:citation_support |
| B7 | citation_precision:gpt-6.1-sol | 0.746 [0.721, 0.771] | 0.359 [0.340, 0.379] | 0.882 [0.872, 0.893] | 0.894 [0.881, 0.905] | evalkit:citation_support |
| B7 | aim_citation_rate:claude-opus-5-5 | 0.057 [0.043, 0.072] | 0.557 [0.525, 0.588] | 0.186 [0.162, 0.210] | 0.094 [0.076, 0.113] | evalkit:citation_support |
| B7 | aim_citation_rate:gpt-6.1-sol | 0.142 [0.120, 0.164] | 0.567 [0.536, 0.598] | 0.225 [0.199, 0.251] | 0.116 [0.096, 0.136] | evalkit:citation_support |

### main_v2

| Rule | Metric | gemma-e2b-base-q4 | gemma-e2b-grounded-q4 | gemma-e2b-grounded-v2-q4 | Source |
| --- | --- | --- | --- | --- | --- |
| B1 | schema_failures_or_zero_citations | 0.005 | 0.000 | 0.000 | ggh:b1 |
| B2 | headline:gpt-6.1-sol | 0.404 [0.373, 0.434] | 0.331 [0.302, 0.361] | 0.329 [0.300, 0.358] | evalkit:headline |
| B2 | groundedness_mean:gpt-6.1-sol | 0.840 [0.820, 0.859] | 0.457 [0.427, 0.487] | 0.493 [0.463, 0.523] | evalkit:groundedness |
| B3 | maybe_recall | 0.491 [0.398, 0.586] | 0.045 [0.010, 0.086] | 0.055 [0.018, 0.098] | ggh:b3 |
| B3 | maybe_precision | 0.127 [0.095, 0.161] | 0.227 [0.059, 0.409] | 0.240 [0.087, 0.409] | ggh:b3 |
| B4 | over_80_words_share | 0.013 | 0.000 | 0.000 | ggh:b4 |
| B5 | sample_read_ok_share | 0.250 | 0.450 | 0.550 | model_review |
| B6 | unsupported_claim_share:gpt-6.1-sol | 0.142 | 0.523 | 0.469 | evalkit:verdicts |
| B7 | citation_validity | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | evalkit:citation_valid |
| B7 | citation_precision:gpt-6.1-sol | 0.747 [0.722, 0.772] | 0.362 [0.343, 0.382] | 0.736 [0.708, 0.763] | evalkit:citation_support |
| B7 | aim_citation_rate:gpt-6.1-sol | 0.138 [0.117, 0.161] | 0.560 [0.529, 0.590] | 0.084 [0.067, 0.102] | evalkit:citation_support |

`model_review`: a blind review by Claude Opus 5.5 subagents, not the owner (OWNER D-61); B5 under `configs/sample_read_prompt.md`, B8 under `configs/offscope_review_prompt.md` (question and answer only).

### offscope

| Rule | Metric | gemma-e2b-base-q4 | gemma-e2b-grounded-q4 | claude-haiku-5-5 | gpt-6-luna | Source |
| --- | --- | --- | --- | --- | --- | --- |
| B8 | presort_flagged_share | 0.100 | 0.000 | 0.250 | 0.400 | ggh:b8 |
| B8 | gives_advice_count | 0 | 2 | 1 | 0 | model_review |

`model_review`: a blind review by Claude Opus 5.5 subagents, not the owner (OWNER D-61); B5 under `configs/sample_read_prompt.md`, B8 under `configs/offscope_review_prompt.md` (question and answer only).

### offscope_v2

| Rule | Metric | gemma-e2b-grounded-v2-q4 | Source |
| --- | --- | --- | --- |
| B8 | presort_flagged_share | 0.000 | ggh:b8 |
| B8 | gives_advice_count | 0 | model_review |

`model_review`: a blind review by Claude Opus 5.5 subagents, not the owner (OWNER D-61); B5 under `configs/sample_read_prompt.md`, B8 under `configs/offscope_review_prompt.md` (question and answer only).
