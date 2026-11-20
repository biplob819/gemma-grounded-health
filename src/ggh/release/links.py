"""OSS-06, SPC-04: every link the README, the Space footer and the release notes need (OWNER
D-37), in one place. A link that does not exist yet is `None` and renders as a visible
placeholder; R4.2 to R4.5 filled them in as each thing was published. Two R4.5 links point at
pages that go live with an outward step the builder takes after the merge: the v1.0.0 GitHub
release (the tag; R4.5). EvalKit PR #1 merged and its report page is live (R4.4). The write-up
(P5.1) may stay pending. `tests/release/test_links.py` checks every link offline: https on an
allowed host, and every local or repo path exists.
"""

from __future__ import annotations

REPO = "https://github.com/biplob819/gemma-grounded-health"
PENDING = "(link pending)"

# name -> (label, url or None until published)
LINKS: dict[str, tuple[str, str | None]] = {
    "repo": ("GitHub repository", REPO),
    "release": ("GitHub release v1.0.0", f"{REPO}/releases/tag/v1.0.0"),  # R4.5: the tag
    "hf_model": (
        "Hugging Face model",
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health",
    ),
    "hf_gguf": (
        "Hugging Face GGUF",
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health-GGUF",
    ),
    "space": ("Demo (local, recorded; OWNER D-63)", "docs/assets/demo.gif"),
    "evalkit_report": ("EvalKit report page", "https://biplob819.github.io/evalkit/gemma/"),
    "evalkit_pr": ("EvalKit upstream pull request", "https://github.com/biplob819/evalkit/pull/1"),
    "parity": (
        "Parity (Q4_K_M against the merged weights)",
        f"{REPO}/blob/main/docs/measurements/v2/parity.json",
    ),
    "benchmark": ("Benchmark results", f"{REPO}/tree/main/docs/results"),
    "data_card": ("DATA_CARD.md", "DATA_CARD.md"),
    "model_card": ("MODEL_CARD.md", "MODEL_CARD.md"),
    "prd": ("PRD", "Gemma%20Grounded%20Health%20PRD.md"),
    "tech_spec": ("Tech Spec", "Gemma%20Grounded%20Health%20Tech%20Spec.md"),
    "bundle": ("Results bundle", "docs/results/results_bundle.json"),
    "citation": ("CITATION.cff", "CITATION.cff"),
    "writeup": ("Write-up", None),  # P5.1
    "case_study": ("Case study: the v1 grounding failure", "docs/case-study.md"),
    "reproduce": ("Reproduce from scratch", "docs/reproduce.md"),
    "release_notes": ("Release notes v1.0.0", "docs/release_notes_v1.0.0.md"),
    "known_issues": ("Known issues", "docs/known-issues.md"),
    "evalkit": ("EvalKit", "https://github.com/biplob819/evalkit"),
}


def md(name: str) -> str:
    label, url = LINKS[name]
    return f"[{label}]({url})" if url else f"{label} {PENDING}"


def absolute(url: str) -> str:
    """A repo-relative path as its GitHub URL on main (for text read outside the repo, such
    as the GitHub release body); a full URL is returned as it is."""
    if url.startswith(("https://", "http://")):
        return url
    return f"{REPO}/blob/main/{url}"


def md_abs(name: str) -> str:
    label, url = LINKS[name]
    return f"[{label}]({absolute(url)})" if url else f"{label} {PENDING}"


def pending() -> list[str]:
    return [name for name, (_, url) in LINKS.items() if url is None]
