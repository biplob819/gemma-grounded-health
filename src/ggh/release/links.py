"""OSS-06, SPC-04: every link the README, the Space footer and the release notes need (OWNER
D-37), in one place. A link that does not exist yet is `None` and renders as a visible
placeholder; R4.2 to R4.5 filled them in as each thing was published. Two R4.5 links point at
pages that go live with an outward step the builder takes after the merge: the v1.0.0 GitHub
release (the tag; R4.5). EvalKit PR #1 merged and its report page is live (R4.4). The write-up
(P5.1) was added after v1.0.0 was published: the README links it, while the v1.0.0 release notes
keep it as pending, exactly as published (`AFTER_RELEASE`, `md_abs(..., as_released=True)`).
The registry is per GGH version (`links_at`, V3.8): v1.0.0's notes keep v1.0.0's links when the
live registry (`LINKS`, the current `ggh.__version__`) moves to v1.1.0 (v3b, OWNER D-75).
`tests/release/test_links.py` checks every link offline: https on an allowed host, and every
local or repo path exists.
"""

from __future__ import annotations

from ggh import __version__

REPO = "https://github.com/biplob819/gemma-grounded-health"
PENDING = "(link pending)"

# per GGH version: the links that name that release (its tag, its notes, the shipped release's
# parity record); everything else is shared
RELEASE_LINKS: dict[str, dict[str, tuple[str, str | None]]] = {
    "1.0.0": {  # v2 (R4.5)
        "release": ("GitHub release v1.0.0", f"{REPO}/releases/tag/v1.0.0"),
        "parity": (
            "Parity (Q4_K_M against the merged weights)",
            f"{REPO}/blob/main/docs/measurements/v2/parity.json",
        ),
        "release_notes": ("Release notes v1.0.0", "docs/release_notes_v1.0.0.md"),
    },
    "1.1.0": {  # v3b (V3.8, OWNER D-75)
        "release": ("GitHub release v1.1.0", f"{REPO}/releases/tag/v1.1.0"),
        "parity": (
            "Parity (Q4_K_M against the merged weights)",
            f"{REPO}/blob/main/docs/measurements/v3b/parity.json",
        ),
        "release_notes": ("Release notes v1.1.0", "docs/release_notes_v1.1.0.md"),
    },
}


def links_at(version: str) -> dict[str, tuple[str, str | None]]:
    """name -> (label, url or None until published), as at GGH `version`."""
    own = RELEASE_LINKS[version]
    return {
        "repo": ("GitHub repository", REPO),
        "release": own["release"],
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
        "evalkit_pr": (
            "EvalKit upstream pull request",
            "https://github.com/biplob819/evalkit/pull/1",
        ),
        "parity": own["parity"],
        "benchmark": ("Benchmark results", f"{REPO}/tree/main/docs/results"),
        "data_card": ("DATA_CARD.md", "DATA_CARD.md"),
        "model_card": ("MODEL_CARD.md", "MODEL_CARD.md"),
        "prd": ("PRD", "Gemma%20Grounded%20Health%20PRD.md"),
        "tech_spec": ("Tech Spec", "Gemma%20Grounded%20Health%20Tech%20Spec.md"),
        "bundle": ("Results bundle", "docs/results/results_bundle.json"),
        "citation": ("CITATION.cff", "CITATION.cff"),
        "writeup": ("Write-up", "docs/writeup/how-i-built-it.md"),  # P5.1
        "case_study": ("Case study: the v1 grounding failure", "docs/case-study.md"),
        "reproduce": ("Reproduce from scratch", "docs/reproduce.md"),
        "release_notes": own["release_notes"],
        "known_issues": ("Known issues", "docs/known-issues.md"),
        "evalkit": ("EvalKit", "https://github.com/biplob819/evalkit"),
    }


# the live registry: the README and the write-up link the current version's release
LINKS: dict[str, tuple[str, str | None]] = links_at(__version__)


# per version, links added after that release; its published notes show them as they were then
# (pending), so `docs/release_notes_v1.0.0.md` stays the file the release used
AFTER: dict[str, frozenset[str]] = {"1.0.0": frozenset({"writeup"}), "1.1.0": frozenset()}
AFTER_RELEASE: frozenset[str] = AFTER["1.0.0"]


def md(name: str) -> str:
    label, url = LINKS[name]
    return f"[{label}]({url})" if url else f"{label} {PENDING}"


def absolute(url: str) -> str:
    """A repo-relative path as its GitHub URL on main (for text read outside the repo, such
    as the GitHub release body); a full URL is returned as it is."""
    if url.startswith(("https://", "http://")):
        return url
    return f"{REPO}/blob/main/{url}"


def md_abs(name: str, as_released: bool = False, version: str | None = None) -> str:
    """The link with an absolute URL, as at GGH `version` (default: the current one);
    `as_released` renders a link added after that release as its notes showed it (pending)."""
    v = version or __version__
    label, url = links_at(v)[name]
    if as_released and name in AFTER.get(v, frozenset()):
        url = None
    return f"[{label}]({absolute(url)})" if url else f"{label} {PENDING}"


def pending() -> list[str]:
    return [name for name, (_, url) in LINKS.items() if url is None]
