import pytest

import secret_scan

FAKES = {
    "anthropic key": "sk-ant-" + "a1B2" * 10,
    "openai key": "sk-proj-" + "Z9y8" * 10,
    "aws access key": "AKIA" + "ABCDEFGHIJKLMNOP",
    "github token": "ghp_" + "x" * 36,
    "pypi token": "pypi-" + "A" * 60,
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "hugging face token": "hf_" + "Ab3d" * 9,
    "kaggle.json credentials": '{"username": "someone", "key": "' + "0a1b2c3d" * 4 + '"}',
    "kaggle key": "KAGGLE_KEY=" + "0a1b2c3d" * 4,
    "kaggle api token": "KGAT_" + "a1B2c3D4" * 4,
}
MORE = {
    "github fine-grained token": "github_pat_" + "A1b2" * 15,
    "aws access key (session)": "ASIA" + "ABCDEFGHIJKLMNOP",
    "aws secret key": "aws_secret_access_key = " + "a/B+" * 10,
    "encrypted private key": "-----BEGIN ENCRYPTED " + "PRIVATE KEY-----",
    "dsa private key": "-----BEGIN DSA " + "PRIVATE KEY-----",
    "pgp private key": "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----",
}


def test_self_test_passes():
    assert secret_scan.self_test() == 0


@pytest.mark.parametrize("kind", sorted(FAKES))
def test_detects_each_kind(tmp_path, kind):
    f = tmp_path / "leak.txt"
    f.write_text(f"value = {FAKES[kind]}\n", encoding="utf-8")
    hits = secret_scan.scan([f])
    assert hits and kind in hits[0]


@pytest.mark.parametrize("label", sorted(MORE))
def test_detects_formats_found_by_the_verifier(tmp_path, label):
    f = tmp_path / ".env.example"
    f.write_text(f"{MORE[label]}\n", encoding="utf-8")
    assert secret_scan.scan([f]), label


def test_non_utf8_file_is_still_scanned(tmp_path):
    f = tmp_path / "latin1.txt"
    f.write_bytes(b"caf\xe9 " + FAKES["anthropic key"].encode() + b"\n")
    assert secret_scan.scan([f])


def test_scan_is_deterministic(tmp_path):
    for i, kind in enumerate(sorted(FAKES)):
        (tmp_path / f"f{i}.txt").write_text(FAKES[kind] + "\n", encoding="utf-8")
    files = sorted(tmp_path.iterdir())
    assert secret_scan.scan(files) == secret_scan.scan(files)
    assert len(secret_scan.scan(files)) == len(FAKES)


def test_env_var_reference_is_clean(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text('key = os.environ["OPENAI_API_KEY"]\n', encoding="utf-8")
    assert secret_scan.scan([f]) == []


@pytest.mark.safety
def test_live_repo_has_no_secrets():
    assert secret_scan.scan(secret_scan.candidate_files(secret_scan.ROOT)) == []


@pytest.mark.parametrize(
    "line",
    [
        "KAGGLE_KEY=" + "0a1b2c3d" * 4,
        "export KAGGLE_KEY='" + "0a1b2c3d" * 4 + "'",
        'os.environ["KAGGLE_KEY"] = "' + "0a1b2c3d" * 4 + '"',
        '{"key": "' + "0a1b2c3d" * 4 + '", "username": "someone"}',
        'os.environ.setdefault("KAGGLE_KEY", "' + "0a1b2c3d" * 4 + '")',
        '{\n  "username": "someone",\n  "key": "' + "0a1b2c3d" * 4 + '"\n}',
    ],
)
def test_detects_kaggle_key_formats(tmp_path, line):
    f = tmp_path / "x.txt"
    f.write_text(line + "\n", encoding="utf-8")
    assert any("kaggle" in h for h in secret_scan.scan([f])), line


def test_cache_key_hex_is_not_a_kaggle_key(tmp_path):
    f = tmp_path / "manifest.json"
    f.write_text('{"key": "' + "0a1b2c3d" * 4 + '", "stage": "data"}\n', encoding="utf-8")
    assert secret_scan.scan([f]) == []


@pytest.mark.parametrize("suffix", sorted(secret_scan.SKIP_SUFFIXES))
def test_model_binaries_are_skipped(tmp_path, suffix):
    f = tmp_path / f"weights{suffix}"
    f.write_text("sk-ant-" + "a1B2" * 10 + "\n", encoding="utf-8")
    assert secret_scan.scan([f]) == []


def test_manifest_with_username_and_nested_key_is_not_flagged(tmp_path):
    f = tmp_path / "manifest.json"
    f.write_text(
        '{"username": "builder", "cache": {"key": "' + "0a1b2c3d" * 4 + '"}, "stage": "x"}\n',
        encoding="utf-8",
    )
    assert secret_scan.scan([f]) == []


def test_fallback_outside_git_skips_tool_caches(tmp_path):
    for d in (".pytest_cache", ".hypothesis", ".venv-py311", "src"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.txt").write_text("x\n", encoding="utf-8")
    files = secret_scan.candidate_files(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["src/f.txt"]


HEX = "0a1b2c3d" * 4


@pytest.mark.parametrize(
    "text",
    [
        'os.environ.setdefault(\n    "KAGGLE_KEY",\n    "' + HEX + '",\n)\n',
        '{"cell": "creds = {\\"username\\": \\"me\\", \\"key\\": \\"' + HEX + '\\"}"}\n',
        "url = 'https://x/?t=x_hf_" + "Ab3d" * 9 + "'\n",
        "tok = ghp_" + "x" * 36 + "_suffix\n",
    ],
)
def test_detects_round3_formats(tmp_path, text):
    f = tmp_path / "x.ipynb"
    f.write_text(text, encoding="utf-8")
    assert secret_scan.scan([f]), text


def test_fallback_skips_any_virtualenv_and_tool_dirs(tmp_path):
    for d in ("venv", "env", ".tox", ".nox", ".mypy_cache", ".ipynb_checkpoints", "myenv", "src"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "myenv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    files = secret_scan.candidate_files(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["src/f.txt"]


def _nb(*cells: list[str]) -> str:
    import json

    return json.dumps(
        {"cells": [{"cell_type": "code", "source": src} for src in cells], "nbformat": 4}, indent=1
    )


@pytest.mark.parametrize(
    "notebook",
    [
        _nb(
            [
                "%%writefile ~/.kaggle/kaggle.json\n",
                "{\n",
                '  "username": "me",\n',
                '  "key": "' + HEX + '"\n',
                "}\n",
            ]
        ),
        _nb(
            [
                "import os\n",
                "os.environ.setdefault(\n",
                '    "KAGGLE_KEY",\n',
                '    "' + HEX + '",\n',
                ")\n",
            ]
        ),
    ],
)
def test_notebook_cells_are_scanned_as_joined_source(tmp_path, notebook):
    f = tmp_path / "train.ipynb"
    f.write_text(notebook, encoding="utf-8")
    hits = secret_scan.scan([f])
    assert hits and str(f) in hits[0]  # found in the raw JSON or in the joined cell


@pytest.mark.parametrize(
    "line",
    [
        "x_sk-proj-" + "Z9y8" * 10,
        "x_AKIA" + "ABCDEFGHIJKLMNOP",
        "AKIA" + "ABCDEFGHIJKLMNOP" + "_x",
        "x_KGAT_" + "a1B2c3D4" * 4,
        "x_pypi-" + "A" * 60,
        "x_github_pat_" + "A1b2" * 15,
        "KAGGLE_KEY=\\\n" + HEX,
        "creds = {'username': 'me', 'key': '" + HEX + "'}",
    ],
)
def test_glued_and_alternate_forms_are_caught(tmp_path, line):
    f = tmp_path / "x.sh"
    f.write_text(line + "\n", encoding="utf-8")
    assert secret_scan.scan([f]), line


def test_fallback_scans_nested_env_and_dist_folders_in_source(tmp_path):
    for d in ("env", "src/env", "src/dist", "dist"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "f.py").write_text("x\n", encoding="utf-8")
    files = sorted(
        p.relative_to(tmp_path).as_posix() for p in secret_scan.candidate_files(tmp_path)
    )
    assert files == ["src/dist/f.py", "src/env/f.py"]


@pytest.mark.parametrize(
    "text",
    [
        'os.environ["KAGGLE_KEY"] = \\\n    "' + HEX + '"\n',
        "os.environ['KAGGLE_KEY'] = \\\n    '" + HEX + "'\n",
        'os.environ["KAGGLE_KEY"] = (\n    "' + HEX + '"\n)\n',
        'KAGGLE_KEY = (\n    "' + HEX + '"\n)\n',
        'KAGGLE_KEY: str = "' + HEX + '"\n',
    ],
)
def test_wrapped_and_typed_kaggle_assignments(tmp_path, text):
    for name, body in (("a.py", text), ("b.ipynb", _nb(text.splitlines(keepends=True)))):
        f = tmp_path / name
        f.write_text(body, encoding="utf-8")
        assert secret_scan.scan([f]), (name, text)


@pytest.mark.parametrize(
    "text",
    [
        '{"msg": "line\\nhf_' + "Ab3d" * 9 + '"}\n',
        '{"msg": "col\\tghp_' + "x" * 36 + '"}\n',
    ],
)
def test_tokens_after_string_escapes(tmp_path, text):
    f = tmp_path / "x.json"
    f.write_text(text, encoding="utf-8")
    assert secret_scan.scan([f]), text


def test_notebook_output_text_is_scanned(tmp_path):
    """A kaggle.json printed over several output lines is found in the decoded output (the
    `:cell2:` label proves it came from the decoded cell, not the raw fallback)."""
    import json

    lines = ["{\n", '  "username": "me",\n', '  "key": "' + HEX + '"\n', "}\n"]
    for kind, output in (
        ("stream", {"output_type": "stream", "text": lines}),
        ("html", {"output_type": "display_data", "data": {"text/html": lines}}),
    ):
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["x = 1"]},
                {"cell_type": "code", "source": ["print(c)"], "outputs": [output]},
            ]
        }
        f = tmp_path / f"{kind}.ipynb"
        f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        hits = secret_scan.scan([f])
        assert hits and ":cell2:" in hits[0], (kind, hits)


@pytest.mark.parametrize(
    "body",
    [
        '{"cells": 5}',
        '{"cells": null}',
        '{"cells": [{"source": [1, 2]}]}',
        "[" * 50000 + "]" * 50000,
    ],
)
def test_malformed_notebooks_do_not_crash(tmp_path, body):
    f = tmp_path / "bad.ipynb"
    f.write_text(body, encoding="utf-8")
    assert secret_scan.scan([f]) == []


@pytest.mark.parametrize(
    "text",
    [
        'os.environ.setdefault("KAGGLE_KEY", value="' + HEX + '")\n',
        'os.environ.get("KAGGLE_KEY", default="' + HEX + '")\n',
        'KAGGLE_KEY: Final = "' + HEX + '"\n',
        'os.environ["KAGGLE_KEY"] = (  # kaggle api key\n    "' + HEX + '"\n)\n',
        'os.environ.setdefault(\n    "KAGGLE_KEY",\n    value="' + HEX + '",\n)\n',
    ],
)
def test_kaggle_keyword_and_commented_forms(tmp_path, text):
    f = tmp_path / "k.py"
    f.write_text(text, encoding="utf-8")
    assert secret_scan.scan([f]), text


@pytest.mark.parametrize("esc", ["\\f", "\\b", "\\u000b", "\\u000a"])
def test_tokens_after_every_json_escape(tmp_path, esc):
    for token in ("hf_" + "Ab3d" * 9, "ghp_" + "x" * 36, "KGAT_" + "a1B2c3D4" * 4):
        f = tmp_path / "e.json"
        f.write_text('{"m": "a' + esc + token + '"}\n', encoding="utf-8")
        assert secret_scan.scan([f]), (esc, token)


def test_kaggle_name_far_from_any_hex_is_not_flagged(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "Set KAGGLE_KEY in Kaggle Secrets; never commit it.\n\n"
        + "Some unrelated line of prose that is long enough to be far away.\n" * 3
        + "Checksum: "
        + HEX
        + "\n",
        encoding="utf-8",
    )
    assert secret_scan.scan([f]) == []


WRAPPED = 'os.environ["KAGGLE_KEY"] = (\n    "' + HEX + '"\n)\n'


def test_allow_marker_on_another_line_does_not_hide_a_real_secret(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("# KAGGLE_KEY  # secret-scan: allow\n" + WRAPPED, encoding="utf-8")
    assert secret_scan.scan([f])
    f.write_text(
        'os.environ["KAGGLE_KEY"] = (  # secret-scan: allow\n    "' + HEX + '"\n)\n',
        encoding="utf-8",
    )
    assert secret_scan.scan([f]) == []  # the marker on the match's own line still works


@pytest.mark.parametrize("sep", ["\x0c", "\u2028", "\x85"])
def test_odd_line_separators_do_not_shift_the_allow_window(tmp_path, sep):
    f = tmp_path / "b.py"
    f.write_text("x = 1" + sep + "\n# note  # secret-scan: allow\n" + WRAPPED, encoding="utf-8")
    assert secret_scan.scan([f]), repr(sep)
    allowed = (
        "x = 1"
        + sep
        + '\nos.environ.setdefault(\n    "KAGGLE_KEY",\n    "'
        + HEX
        + '",  # secret-scan: allow\n)\n'
    )
    f.write_text(allowed, encoding="utf-8")
    assert not any("multi-line" in h for h in secret_scan.scan([f])), repr(sep)


def test_multiline_kaggle_reaches_three_breaks_but_not_four(tmp_path):
    near = "KAGGLE_KEY = (\n\n\n    '" + HEX + "')\n"  # value 3 line breaks after the name
    far = "KAGGLE_KEY note\n\n\n\nmd5: " + HEX + "\n"  # 4 line breaks
    f = tmp_path / "c.py"
    f.write_text(near, encoding="utf-8")
    assert secret_scan.scan([f])
    f.write_text(far, encoding="utf-8")
    assert secret_scan.scan([f]) == []
    f.write_text(
        "Set KAGGLE_KEY in Kaggle Secrets.\n" + "\n" * 15 + "md5: " + HEX + "\n", encoding="utf-8"
    )
    assert secret_scan.scan([f]) == []


def test_same_line_and_multiline_length_limits(tmp_path):
    f = tmp_path / "d.py"
    f.write_text("KAGGLE_KEY" + "=" * 60 + HEX + "\n", encoding="utf-8")  # 60-char gap: caught
    assert secret_scan.scan([f])
    f.write_text("KAGGLE_KEY" + "=" * 90 + HEX + "\n", encoding="utf-8")  # past both limits
    assert secret_scan.scan([f]) == []
    f.write_text(
        "KAGGLE_KEY = (\n" + " " * 70 + "'" + HEX + "')\n", encoding="utf-8"
    )  # 80 across lines
    assert secret_scan.scan([f])


def test_one_kaggle_match_cannot_swallow_a_later_secret(tmp_path):
    f = tmp_path / "e.py"
    f.write_text(
        "# KAGGLE_KEY is read below  # secret-scan: allow\nx = 1\n" + WRAPPED, encoding="utf-8"
    )
    assert secret_scan.scan([f])


def test_traceback_output_and_null_cells(tmp_path):
    import json

    lines = ["{\n", '  "username": "me",\n', '  "key": "' + HEX + '"\n', "}\n"]
    nb = {
        "cells": [
            None,
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [{"output_type": "error", "traceback": lines}],
            },
        ]
    }
    f = tmp_path / "t.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    hits = secret_scan.scan([f])
    assert any(":cell2:" in h for h in hits), hits


def test_bom_notebook_is_parsed(tmp_path):
    import json

    lines = ["{\n", '  "username": "me",\n', '  "key": "' + HEX + '"\n', "}\n"]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [{"output_type": "stream", "text": lines}],
            }
        ]
    }
    f = tmp_path / "bom.ipynb"
    f.write_text("\ufeff" + json.dumps(nb, indent=1), encoding="utf-8")
    assert any(":cell1:" in h for h in secret_scan.scan([f]))


@pytest.mark.parametrize("gap", [61, 70, 80])
def test_same_line_values_past_60_chars_are_not_flagged(tmp_path, gap):
    f = tmp_path / "s.py"
    f.write_text("KAGGLE_KEY" + "=" * gap + HEX + "\n", encoding="utf-8")
    assert secret_scan.scan([f]) == []


def test_realistic_sentence_with_distant_checksum_is_not_flagged(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "Set KAGGLE_KEY in Kaggle Secrets; the file below has checksum of course md5 " + HEX + "\n",
        encoding="utf-8",
    )
    assert secret_scan.scan([f]) == []


@pytest.mark.parametrize("compact", [True, False])
def test_notebook_break_limit_applies_to_decoded_text(tmp_path, compact):
    import json

    far = "KAGGLE_KEY note\n\n\n\nmd5: " + HEX
    for source in (far, far.splitlines(keepends=True)):
        nb = {"cells": [{"cell_type": "markdown", "source": source}], "metadata": {}}
        f = tmp_path / "far.ipynb"
        f.write_text(json.dumps(nb) if compact else json.dumps(nb, indent=1), encoding="utf-8")
        assert secret_scan.scan([f]) == [], (compact, type(source))


def test_notebook_metadata_and_cells_are_judged_independently(tmp_path):
    import json

    token = "hf_" + "Ab3d" * 9
    nb = {
        "cells": [
            {"cell_type": "code", "source": "x = 1  # secret-scan: allow"},
            {"cell_type": "code", "source": "y = 2"},
        ],
        "metadata": {"widgets": {"state": "token " + token}},
    }
    f = tmp_path / "m.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")  # compact: one raw line
    hits = secret_scan.scan([f])
    assert any(":meta:" in h and "hugging face" in h for h in hits), hits


def test_traceback_lines_keep_their_breaks(tmp_path):
    import json

    tb = ["Traceback (most recent call last)", "KAGGLE_KEY was printed", "", "", "", "md5 " + HEX]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [{"output_type": "error", "traceback": tb}],
            }
        ]
    }
    f = tmp_path / "tb.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")
    assert secret_scan.scan([f]) == []  # 4 line breaks between name and value


def test_unparseable_notebook_falls_back_to_a_raw_scan(tmp_path):
    f = tmp_path / "broken.ipynb"
    f.write_text("not json at all, but token hf_" + "Ab3d" * 9 + "\n", encoding="utf-8")
    assert secret_scan.scan([f])


TOKEN = "sk-ant-" + "a1B2" * 10


@pytest.mark.parametrize(
    "where, cell",
    [
        (
            "application/json output",
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {"output_type": "display_data", "data": {"application/json": {"token": TOKEN}}}
                ],
            },
        ),
        (
            "kaggle.json as JSON output",
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "data": {"application/json": {"username": "me", "key": HEX}},
                    }
                ],
            },
        ),
        (
            "cell metadata",
            {"cell_type": "code", "source": "x", "metadata": {"note": "KEY=" + TOKEN}},
        ),
        (
            "output metadata",
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": "ok"},
                        "metadata": {"u": TOKEN},
                    }
                ],
            },
        ),
        (
            "error evalue",
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "E",
                        "evalue": "bad " + TOKEN,
                        "traceback": [],
                    }
                ],
            },
        ),
        (
            "attachment",
            {
                "cell_type": "markdown",
                "source": "x",
                "attachments": {"a.txt": {"text/plain": TOKEN}},
            },
        ),
    ],
)
def test_every_string_in_a_cell_is_scanned(tmp_path, where, cell):
    import json

    f = tmp_path / "n.ipynb"
    f.write_text(json.dumps({"cells": [cell], "metadata": {}}, indent=1), encoding="utf-8")
    hits = secret_scan.scan([f])
    assert any(":cell1:" in h for h in hits), (where, hits)


AWS_SECRET = "a/B+" * 10
PLACEMENTS = {
    "json list of objects": lambda: {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"application/json": [{"token": TOKEN}]},
                    }
                ],
            }
        ]
    },
    "json list with KAGGLE_KEY": lambda: {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"application/json": [{"KAGGLE_KEY": HEX}]},
                    }
                ],
            }
        ]
    },
    "json list of strings": lambda: {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"application/json": ["KAGGLE_KEY", HEX]},
                    }
                ],
            }
        ]
    },
    "papermill parameters": lambda: {
        "cells": [],
        "metadata": {"papermill": {"parameters": {"KAGGLE_KEY": HEX}}},
    },
    "aws secret in notebook metadata": lambda: {
        "cells": [],
        "metadata": {"aws_secret_access_key": AWS_SECRET},
    },
    "aws secret in output metadata": lambda: {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": "ok"},
                        "metadata": {"aws_secret_access_key": AWS_SECRET},
                    }
                ],
            }
        ]
    },
    "KAGGLE_KEY in cell metadata": lambda: {
        "cells": [{"cell_type": "code", "source": "x", "metadata": {"KAGGLE_KEY": HEX}}]
    },
    "kaggle.json in widget state": lambda: {
        "cells": [],
        "metadata": {"widgets": {"state": {"username": "me", "key": HEX}}},
    },
    "svg output": lambda: {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/svg+xml": "<svg><!-- " + TOKEN + " --></svg>"},
                    }
                ],
            }
        ]
    },
}


@pytest.mark.parametrize("name", sorted(PLACEMENTS))
def test_decoded_notebook_scan_catches_what_a_raw_scan_catches(tmp_path, name):
    import json

    body = json.dumps(PLACEMENTS[name](), indent=1)
    f = tmp_path / "p.ipynb"
    f.write_text(body, encoding="utf-8")
    raw = secret_scan._scan_text(str(f), body)
    decoded = secret_scan.scan([f])
    assert raw, f"fixture {name} should be visible to a raw scan"
    assert decoded, f"decoded scan missed {name}, which a raw scan catches"


def _deep(n: int) -> dict:
    node: dict = {"leaf": 1}
    for _ in range(n):
        node = {"d": node}
    return node


def test_unrenderably_deep_notebook_falls_back_to_raw_and_still_catches(tmp_path):
    import json
    import sys

    nb = {
        "cells": [],
        "metadata": {"papermill": {"parameters": {"KAGGLE_KEY": HEX}}, "z": _deep(1200)},
    }
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(20000)
    try:
        body = json.dumps(nb, indent=1)
    finally:
        sys.setrecursionlimit(old)
    f = tmp_path / "deep.ipynb"
    f.write_text(body, encoding="utf-8")
    assert secret_scan.scan([f])


def test_printed_value_is_paired_with_the_source_that_read_it(tmp_path):
    import json

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": 'os.environ["KAGGLE_KEY"]',
                "outputs": [
                    {"output_type": "execute_result", "data": {"text/plain": "'" + HEX + "'"}}
                ],
            }
        ]
    }
    f = tmp_path / "printed.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert any(":cell1:" in h for h in secret_scan.scan([f]))


def test_mime_types_are_case_insensitive_and_plus_json_is_json(tmp_path):
    import json

    for data in (
        {"image/SVG+XML": "<svg>" + TOKEN + "</svg>"},
        {"application/vnd.custom+json": ["KAGGLE_KEY", HEX]},
    ):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x",
                    "outputs": [{"output_type": "display_data", "data": data}],
                }
            ]
        }
        f = tmp_path / "mime.ipynb"
        f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        assert secret_scan.scan([f]), data


def test_pixel_images_are_not_scanned(tmp_path):
    import json

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {"output_type": "display_data", "data": {"image/png": "iVBOR" + TOKEN}}
                ],
            }
        ]
    }
    f = tmp_path / "png.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert secret_scan.scan([f]) == []


def test_allow_marker_in_metadata_does_not_cover_a_sibling_key(tmp_path):
    import json

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "metadata": {"note": "fixture # secret-scan: allow", "token": TOKEN},
            }
        ]
    }
    f = tmp_path / "allow.ipynb"
    f.write_text(
        json.dumps(nb), encoding="utf-8"
    )  # compact on disk; rendered indented when decoded
    assert any(":cell1:" in h for h in secret_scan.scan([f]))


def test_zero_gap_kaggle_key(tmp_path):
    f = tmp_path / "z.txt"
    f.write_text("KAGGLE_KEY" + HEX + "\n", encoding="utf-8")
    assert secret_scan.scan([f])


def test_plus_json_lists_are_data_not_lines(tmp_path):
    """Joined as lines, the allow marker would share a line with the token and hide it."""
    import json

    data = {"application/vnd.custom+json": ["fixture  # secret-scan: allow", TOKEN]}
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [{"output_type": "display_data", "data": data}],
            }
        ]
    }
    f = tmp_path / "pj.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert any(":cell1:" in h for h in secret_scan.scan([f]))


def _nbfile(tmp_path, nb, name="n.ipynb", raw=None):
    import json

    f = tmp_path / name
    f.write_text(raw if raw is not None else json.dumps(nb, indent=1), encoding="utf-8")
    return f


def test_pairing_covers_every_output_and_tracebacks(tmp_path):
    src = 'os.environ["KAGGLE_KEY"]'
    second = {
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "outputs": [
                    {"output_type": "stream", "text": "loading\n"},
                    {"output_type": "execute_result", "data": {"text/plain": "'" + HEX + "'"}},
                ],
            }
        ]
    }
    tb = {
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "E",
                        "evalue": "x",
                        "traceback": ["'" + HEX + "'"],
                    }
                ],
            }
        ]
    }
    ev = {
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "outputs": [{"output_type": "error", "ename": "E", "evalue": HEX, "traceback": []}],
            }
        ]
    }
    for name, nb in (("second", second), ("traceback", tb), ("evalue", ev)):
        assert secret_scan.scan([_nbfile(tmp_path, nb, f"{name}.ipynb")]), name


def test_pairing_keeps_blank_lines_so_far_values_stay_unflagged(tmp_path):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "KAGGLE_KEY note\n\n\n\n",
                "outputs": [{"output_type": "stream", "text": "md5 " + HEX + "\n"}],
            }
        ]
    }
    assert secret_scan.scan([_nbfile(tmp_path, nb)]) == []


def test_duplicate_notebook_keys_fall_back_to_a_raw_scan(tmp_path):
    raw = (
        '{"cells": [{"cell_type": "code", "source": "KAGGLE_KEY = \\"'
        + HEX
        + '\\"", "source": "x"}]}\n'
    )
    assert secret_scan.scan([_nbfile(tmp_path, None, raw=raw)])


def test_two_distinct_secrets_in_one_cell_are_both_reported(tmp_path):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["a = 'hf_" + "Ab3d" * 9 + "'\n", "b = 'hf_" + "Zz9y" * 9 + "'\n"],
            }
        ]
    }
    hits = [h for h in secret_scan.scan([_nbfile(tmp_path, nb)]) if "hugging face" in h]
    assert len({h.split(": possible")[0] for h in hits}) == 2, hits


def test_distinct_secrets_sharing_a_line_number_in_one_cell_are_both_reported(tmp_path):
    import json

    a, b = "hf_" + "Ab3d" * 9, "hf_" + "Zz9y" * 9
    rest = {"cell_type": "code", "metadata": {"note": b}}
    line_b = json.dumps(rest, indent=1).split("\n").index(f'  "note": "{b}"') + 1
    source = ["\n"] * (line_b - 1) + ["x = '" + a + "'\n"]  # a on the same line number as b
    nb = {"cells": [{"cell_type": "code", "source": source, "metadata": {"note": b}}]}
    hits = [h for h in secret_scan.scan([_nbfile(tmp_path, nb)]) if "hugging face" in h]
    assert len({h for h in hits}) == 1 and len(hits) == 2, hits  # same label, both reported


def test_two_tokens_on_one_line_are_two_hits(tmp_path):
    f = tmp_path / "two.py"
    f.write_text("toks = ['hf_" + "Ab3d" * 9 + "', 'hf_" + "Zz9y" * 9 + "']\n", encoding="utf-8")
    assert len(secret_scan.scan([f])) == 2


A32 = "0123456789abcdef" * 2
B32 = "fedcba9876543210" * 2


@pytest.mark.parametrize(
    "text, expected",
    [
        ('KAGGLE_KEY="' + A32 + '"; OLD_KAGGLE_KEY=\\\n"' + B32 + '"\n', 2),
        (
            'os.environ["HF_TOKEN"] = "hf_'
            + "Ab3d" * 9
            + '"; os.environ["KAGGLE_KEY"] = (\n    "'
            + A32
            + '")\n',
            2,
        ),
        ('KAGGLE_KEY = (\n"' + A32 + '"); OLD_KAGGLE_KEY = (\n"' + B32 + '")\n', 2),
        ('KAGGLE_KEY="' + A32 + '"\n', 1),
        ('os.environ.setdefault(\n    "KAGGLE_KEY",\n    "' + A32 + '",\n)\n', 1),
    ],
)
def test_every_secret_is_reported_once(tmp_path, text, expected):
    f = tmp_path / "k.py"
    f.write_text(text, encoding="utf-8")
    assert len(secret_scan.scan([f])) == expected, secret_scan.scan([f])


def test_same_secret_in_line_and_multiline_form_is_one_report(tmp_path):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "E",
                        "evalue": "bad KAGGLE_KEY:\n" + A32,
                        "traceback": [],
                    }
                ],
            }
        ]
    }
    hits = secret_scan.scan([_nbfile(tmp_path, nb)])
    assert len(hits) == 1, hits


def test_secret_identity_comes_from_each_patterns_own_value():
    core = secret_scan._secret_core
    kk = secret_scan.PATTERNS["kaggle key"].search('KAGGLE_KEY="' + A32.upper() + '"')
    assert kk and core(kk) == A32
    _, kj = secret_scan.FILE_PATTERNS["kaggle.json credentials"]
    m = kj.search('{"key": "' + B32 + '", "username": "kaggle-' + A32 + '"}')
    assert m and core(m) == B32  # the key, never a hex that happens to sit in the username
    hf = secret_scan.PATTERNS["hugging face token"].search("hf_" + "Ab3d" * 9)
    assert hf and core(hf) == "hf_" + "Ab3d" * 9


def test_kaggle_json_key_is_not_confused_with_a_username_hex(tmp_path):
    f = tmp_path / "g.py"
    f.write_text(
        'KAGGLE_KEY = "'
        + A32
        + '"\nx = {"key": "'
        + B32
        + '", "username": "kaggle-'
        + A32
        + '"}\n',
        encoding="utf-8",
    )
    assert len(secret_scan.scan([f])) == 2


def test_same_value_wrapped_and_inline_is_reported_once(tmp_path):
    f = tmp_path / "j.py"
    f.write_text(
        'KAGGLE_KEY = (\n    "' + A32 + '")\n\nOTHER_KAGGLE_KEY = "' + A32 + '"\n', encoding="utf-8"
    )
    assert len(secret_scan.scan([f])) == 1


PEM_A = "-----BEGIN " + "PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCaaaa"
PEM_B = "-----BEGIN " + "PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCbbbb"


@pytest.mark.parametrize("one_line", [False, True])
def test_two_different_private_keys_in_one_cell_are_both_reported(tmp_path, one_line):
    import json

    def sa(pem: str) -> str:
        return '{"type": "service_account", "private_key": "' + pem + '"}'

    source = sa(PEM_A) + (" " if one_line else "\n") + sa(PEM_B) + "\n"
    nb = {"cells": [{"cell_type": "code", "source": source}]}
    f = tmp_path / "sa.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert len([h for h in secret_scan.scan([f]) if "private key" in h]) == 2


def test_two_pem_files_worth_of_keys_in_one_cell(tmp_path):
    import json

    def pem(body: str) -> str:
        return "-----BEGIN RSA " + "PRIVATE KEY-----\n" + body + "\n-----END RSA PRIVATE KEY-----\n"

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": pem("MIIEowIBAAKCAQEAaaaa") + pem("MIIEowIBAAKCAQEAbbbb"),
            }
        ]
    }
    f = tmp_path / "pem.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert len([h for h in secret_scan.scan([f]) if "private key" in h]) == 2


KJ_LINES = [
    "%%writefile ~/.kaggle/kaggle.json\n",
    "{\n",
    '  "username": "me",\n',
    '  "key": "' + A32 + '"\n',
    "}\n",
]


def test_pretty_kaggle_json_in_a_raw_scanned_notebook(tmp_path):
    import json

    cell = json.dumps({"cell_type": "code", "source": KJ_LINES}, indent=1)
    raw = '{"cells": [' + cell + '], "metadata": {}, "metadata": {}}\n'  # duplicate key: raw scan
    f = tmp_path / "dup.ipynb"
    f.write_text(raw, encoding="utf-8")
    assert secret_scan.scan([f])


def test_pretty_kaggle_json_held_as_an_escaped_string(tmp_path):
    import json

    pretty = json.dumps({"username": "me", "key": A32}, indent=2)
    f = tmp_path / "esc.py"
    f.write_text("CREDS = " + json.dumps(pretty) + "\n", encoding="utf-8")
    assert secret_scan.scan([f])


def test_uppercase_kaggle_json_key(tmp_path):
    f = tmp_path / "upper.json"
    f.write_text('{"username": "me", "key": "' + A32.upper() + '"}\n', encoding="utf-8")
    assert secret_scan.scan([f])


def test_keys_with_a_long_common_prefix_are_told_apart(tmp_path):
    import json

    prefix = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZWQyNTUxOQAAACA"

    def key(tail: str) -> str:
        return (
            "-----BEGIN OPENSSH "
            + "PRIVATE KEY-----\n"
            + prefix
            + "\n"
            + tail * 40
            + "\n"
            + "-----END OPENSSH PRIVATE KEY-----\n"
        )

    nb = {"cells": [{"cell_type": "code", "source": key("Aa") + key("Bb")}]}
    f = tmp_path / "ssh.ipynb"
    f.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    assert len([h for h in secret_scan.scan([f]) if "private key" in h]) == 2


def test_double_escaped_kaggle_json_in_a_raw_scanned_notebook(tmp_path):
    import json

    src = 'CREDS = "{\\"username\\": \\"me\\", \\"key\\": \\"' + A32 + '\\"}"\n'
    cell = json.dumps({"cell_type": "code", "source": [src]}, indent=1)
    f = tmp_path / "dbl.ipynb"
    f.write_text('{"cells": [' + cell + '], "metadata": {}, "metadata": {}}\n', encoding="utf-8")
    assert secret_scan.scan([f])
