# R4.3 criteria

**SPC-01** — The demo answers through `compare()` with EvalKit's splitter, template and `Answer` parsing; a live answer from the local server parses like the benchmark's.
**SPC-02** — Cards show the label, answer and cited sentences highlighted in the abstract; out-of-range citations are flagged.
**SPC-03** — The six samples open by default without a passcode; the pasted-abstract path is secondary.
**SPC-04** — The page shows the research-only banner, says the demo serves the benchmarked Q4_K_M GGUF, links parity, benchmark results and both Hub repos, and its results panel is generated from the bundle.
**SPC-05** — Nothing a visitor types reaches any log.
**SPC-06** — The local fallback is recorded (OWNER D-63): committed screenshots of an answer with cited sentences and of the passcode gate, and a looping GIF, with no passcode, token or secret in any frame.
**SPC-07** — The six samples are checked on the running local demo and what they showed is recorded in RUN-STATE.
**SPC-08** — Every live action needs the passcode, checked against a salted hash with a constant-time compare and a rate limit, and never logged.
