"""Settings from the process environment, else from the repo-root `.env`.

Seam the implementation must honour (pinned by these tests):

* `horsetrack.core.env.repo_root() -> Path` is a module-level function, found by walking up from
  the package to the directory holding pyproject.toml.
* `get_setting` and `horsetrack.core.paths` must call it through the module at call time
  (`env.repo_root()`, not a name imported by value), so that
  `monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: fake_root)` redirects both the
  `.env` lookup and the "inside the repo" check.
* Tests never read the developer's real `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from horsetrack.core import env

NAME = "HORSETRACK_TEST_SETTING"


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.delenv(NAME, raising=False)
    return root


def test_repo_root_is_the_directory_holding_pyproject():
    root = env.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "horsetrack" / "core" / "env.py").is_file()


def test_unset_everywhere_gives_none(fake_root):
    assert env.get_setting(NAME) is None
    (fake_root / ".env").write_text("OTHER=1\n")
    assert env.get_setting(NAME) is None


def test_value_comes_from_the_environment(fake_root, monkeypatch):
    monkeypatch.setenv(NAME, "from-env")
    assert env.get_setting(NAME) == "from-env"


def test_value_falls_back_to_dot_env(fake_root):
    (fake_root / ".env").write_text(f"{NAME}=from-file\n")
    assert env.get_setting(NAME) == "from-file"


def test_environment_wins_over_dot_env(fake_root, monkeypatch):
    (fake_root / ".env").write_text(f"{NAME}=from-file\n")
    monkeypatch.setenv(NAME, "from-env")
    assert env.get_setting(NAME) == "from-env"


def test_dot_env_comments_and_blank_lines_are_skipped(fake_root):
    (fake_root / ".env").write_text(
        "\n"
        "# a comment\n"
        f"# {NAME}=commented-out\n"
        "\n"
        "   \n"
        "OTHER=other\n"
        f"{NAME}=real\n"
        "# trailing comment\n"
    )
    assert env.get_setting(NAME) == "real"
    assert env.get_setting("OTHER") == "other"


def test_a_commented_out_line_alone_does_not_define_the_setting(fake_root):
    (fake_root / ".env").write_text(f"# {NAME}=nope\n")
    assert env.get_setting(NAME) is None


def test_only_the_first_equals_splits_and_names_match_exactly(fake_root):
    (fake_root / ".env").write_text(f"{NAME}_EXTRA=wrong\n{NAME}=a=b=c\n")
    assert env.get_setting(NAME) == "a=b=c"
    assert env.get_setting(f"{NAME}_EXTRA") == "wrong"


# .env values: one pair of matching surrounding quotes is stripped, as python-dotenv does.
# (raw .env text after the `=`, value the setting must read back as)
QUOTE_CASES = [
    ('"abc"', "abc"),
    ("'abc'", "abc"),
    ("abc", "abc"),
    (
        '"Mozilla/5.0 (X11; Linux) Gecko/2010 Firefox/115.0"',
        "Mozilla/5.0 (X11; Linux) Gecko/2010 Firefox/115.0",
    ),
    ("'two words here'", "two words here"),
    ('"a=b=c"', "a=b=c"),
    ('a"b', 'a"b'),
    ("a'b", "a'b"),
    ('"a"b"', 'a"b'),
    ('"it\'s"', "it's"),
    ("'say \"hi\"'", 'say "hi"'),
    ('"abc', '"abc'),
    ('abc"', 'abc"'),
    ("'abc", "'abc"),
    ("abc'", "abc'"),
    ("\"abc'", "\"abc'"),
    ("'abc\"", "'abc\""),
    ('""x""', '"x"'),
    ("''x''", "'x'"),
    ('"', '"'),
    ("'", "'"),
]


@pytest.mark.parametrize(("raw", "expected"), QUOTE_CASES)
def test_dot_env_value_loses_one_pair_of_surrounding_quotes(fake_root, raw, expected):
    (fake_root / ".env").write_text(f"{NAME}={raw}\n")
    assert env.get_setting(NAME) == expected


def test_quote_stripping_applies_per_line_and_leaves_other_settings_alone(fake_root):
    (fake_root / ".env").write_text(f'FIRST="one"\n{NAME}=\'two\'\nTHIRD=three\nFOURTH="a\'b"\n')
    assert env.get_setting("FIRST") == "one"
    assert env.get_setting(NAME) == "two"
    assert env.get_setting("THIRD") == "three"
    assert env.get_setting("FOURTH") == "a'b"


def test_quoted_user_agent_reads_back_without_quotes_for_the_header(fake_root):
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    (fake_root / ".env").write_text(f'HORSETRACK_USER_AGENT="{ua}"\n')
    assert env.get_setting("HORSETRACK_USER_AGENT") == ua


@pytest.mark.parametrize("raw", ['"quoted"', "'quoted'", 'a"b'])
def test_process_environment_wins_and_is_returned_verbatim(fake_root, monkeypatch, raw):
    (fake_root / ".env").write_text(f'{NAME}="from-file"\n')
    monkeypatch.setenv(NAME, raw)
    assert env.get_setting(NAME) == raw
    # the file is never copied into the process environment
    monkeypatch.delenv(NAME)
    assert env.get_setting(NAME) == "from-file"
    assert NAME not in os.environ
