"""OSS-03: CITATION.cff is valid Citation File Format 1.2.0 and matches the release.

cffconvert needs the network for its schema, so this is the schema's structure checked offline:
the required keys, only keys the 1.2.0 schema defines, the value shapes it requires (dates,
URLs, identifier and reference types, person entries), and the version and links of v1.0.0."""

import datetime as dt
import re
import tomllib
from pathlib import Path

import yaml

from ggh import __version__
from ggh.release import links

ROOT = Path(__file__).resolve().parents[2]
CFF = ROOT / "CITATION.cff"

# https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema.json
TOP_KEYS = {
    "abstract", "authors", "cff-version", "commit", "contact", "date-released", "doi",
    "identifiers", "keywords", "license", "license-url", "message", "preferred-citation",
    "references", "repository", "repository-artifact", "repository-code", "title", "type",
    "url", "version",
}  # fmt: skip
REQUIRED = {"authors", "cff-version", "message", "title"}
PERSON_KEYS = {
    "address", "affiliation", "alias", "city", "country", "email", "family-names", "fax",
    "given-names", "name-particle", "name-suffix", "orcid", "post-code", "region", "tel",
    "website",
}  # fmt: skip
ENTITY_KEYS = {"name", "address", "city", "country", "email", "website", "date-start"}
IDENTIFIER_TYPES = {"doi", "url", "swh", "other"}
REFERENCE_TYPES = {
    "art", "article", "audiovisual", "bill", "blog", "book", "catalogue", "conference-paper",
    "conference", "data", "database", "dictionary", "edited-work", "encyclopedia",
    "film-broadcast", "generic", "government-document", "grant", "hearing",
    "historical-work", "legal-case", "legal-rule", "magazine-article", "manual", "map",
    "multimedia", "music", "newspaper-article", "pamphlet", "patent", "personal-communication",
    "proceedings", "report", "serial", "slides", "software", "software-code",
    "software-container", "software-executable", "software-virtual-machine", "sound-recording",
    "standard", "statute", "thesis", "unpublished", "video", "website",
}  # fmt: skip
URL = re.compile(r"^(https|http|ftp|sftp)://.+")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _cff() -> dict:
    return yaml.safe_load(CFF.read_text(encoding="utf-8"))


def _person_problems(where: str, p: dict) -> list[str]:
    if not isinstance(p, dict) or not p:
        return [f"{where}: not a person or entity"]
    keys = set(p)
    if "name" in keys:  # an entity
        return [f"{where}: unknown entity key {k}" for k in keys - ENTITY_KEYS]
    out = [f"{where}: unknown person key {k}" for k in keys - PERSON_KEYS]
    if not ({"family-names", "given-names"} & keys):
        out.append(f"{where}: a person needs family-names or given-names")
    return out


def problems(cff: dict) -> list[str]:
    out = [f"missing {k}" for k in sorted(REQUIRED - set(cff))]
    out += [f"unknown key {k}" for k in sorted(set(cff) - TOP_KEYS)]
    if cff.get("cff-version") != "1.2.0":
        out.append("cff-version must be 1.2.0")
    if cff.get("type", "software") not in {"software", "dataset"}:
        out.append("type must be software or dataset")
    for i, a in enumerate(cff.get("authors") or []):
        out += _person_problems(f"authors[{i}]", a)
    if not cff.get("authors"):
        out.append("authors is empty")
    for k in ("repository-code", "url", "repository", "repository-artifact", "license-url"):
        if k in cff and not URL.match(str(cff[k])):
            out.append(f"{k} is not a URL")
    if "date-released" in cff:
        d = cff["date-released"]
        if not (isinstance(d, dt.date) or (isinstance(d, str) and DATE.match(d))):
            out.append("date-released must be YYYY-MM-DD")
    if "version" in cff and not isinstance(cff["version"], str | int | float):
        out.append("version must be a string or number")
    for i, ident in enumerate(cff.get("identifiers") or []):
        if ident.get("type") not in IDENTIFIER_TYPES or not ident.get("value"):
            out.append(f"identifiers[{i}] needs a type in {sorted(IDENTIFIER_TYPES)} and a value")
        elif ident["type"] == "url" and not URL.match(ident["value"]):
            out.append(f"identifiers[{i}] value is not a URL")
        extra = set(ident) - {"type", "value", "description"}
        out += [f"identifiers[{i}]: unknown key {k}" for k in extra]
    if not all(isinstance(k, str) for k in cff.get("keywords") or []):
        out.append("keywords must be strings")
    for i, r in enumerate(cff.get("references") or []):
        if r.get("type") not in REFERENCE_TYPES:
            out.append(f"references[{i}] type {r.get('type')!r} is not a CFF reference type")
        if not r.get("title") or not r.get("authors"):
            out.append(f"references[{i}] needs a title and authors")
        for j, a in enumerate(r.get("authors") or []):
            out += _person_problems(f"references[{i}].authors[{j}]", a)
    return out


def test_citation_file_is_valid_cff_1_2_0():
    assert problems(_cff()) == []


def test_validator_catches_broken_files():
    good = _cff()
    assert problems({k: v for k, v in good.items() if k != "authors"})
    assert problems({**good, "colour": "blue"})
    assert problems({**good, "date-released": "21/11/2026"})
    assert problems({**good, "identifiers": [{"type": "web", "value": "x"}]})
    assert problems({**good, "authors": [{"orcid-id": "x"}]})


def test_version_and_links_match_the_release():
    cff = _cff()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert str(cff["version"]) == "1.1.0" == __version__ == pyproject["project"]["version"]
    assert cff["repository-code"] == links.REPO
    values = {i["value"] for i in cff["identifiers"]}
    for name in ("release", "hf_model", "hf_gguf"):
        assert links.LINKS[name][1] in values, name
    assert str(links.LINKS["release"][1]).endswith(f"/releases/tag/v{cff['version']}")
    assert cff["license"] == "Apache-2.0"
