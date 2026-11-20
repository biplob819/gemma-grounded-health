# R4.3 — Demo

**Wave 5 (release). Depends on:** R4.2. **Unblocks:** R4.5.

## Goal
A demo a visitor can see and the owner can run: the same answers the benchmark scored, cited sentences highlighted, behind the passcode gate, recorded as screenshots and a GIF (US5, FR9, OWNER D-36, D-39, D-63).

## Design
- **Local, not ZeroGPU (OWNER D-63).** `uv run --group demo ggh space --local --env-file .env` serves the app on 127.0.0.1 against `ggh serve`'s llama-server with the v2 Q4_K_M GGUF, the bytes that were benchmarked and published. No hosted Space and no spend.
- **compare() (SPC-01).** EvalKit's splitter, template and `Answer` parsing, as before.
- **Cards (SPC-02).** Base and v2 side by side for the six samples; cited sentences highlighted in the abstract; out-of-range citations flagged.
- **Samples first (SPC-03).** Six dev questions (`space/samples.json`, v2 from its dev run, base from v1's dev run on the same cases); pasted abstract second, behind the passcode.
- **Notes and links (SPC-04).** Research-only banner; the demo serves the benchmarked Q4_K_M GGUF; parity, benchmark and Hub links from `ggh.release.links`; the results panel generated from the bundle.
- **Privacy (SPC-05).** Nothing typed is logged.
- **Recorded fallback (SPC-06).** Screenshots of an answer and of the passcode gate, and a looping GIF of the flow, under `docs/assets/`, with no passcode or secret in any frame.
- **Read the sample (SPC-07).** The six samples checked on the running local demo.
- **Passcode (SPC-08).** Every live action needs the owner's passcode, held only as a salted hash, constant-time compare, rate-limited, never logged.
