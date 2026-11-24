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
