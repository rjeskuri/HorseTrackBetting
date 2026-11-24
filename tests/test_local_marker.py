"""The `local` marker and the `real_fixtures` fixture for tests that need real scraped pages.

Each test runs a small inner pytest session (pytester) against the project's own pytest config
([tool.pytest.ini_options] from pyproject.toml) and the project's own tests/conftest.py, both
copied into the inner project, so the shipped fixture and marker registration are what is tested.
"""

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from horsetrack.core import env

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
VAR = "HORSETRACK_DATA_DIR"

INNER_LOCAL_TEST = textwrap.dedent(
    """
    import os
    from pathlib import Path

    import pytest


    @pytest.mark.local
    def test_real_page(real_fixtures):
        expected = Path(os.environ["HORSETRACK_DATA_DIR"]).resolve() / "fixtures"
        assert real_fixtures == expected
        assert (real_fixtures / "page.html").read_text() == os.environ["EXPECTED_PAGE"]
    """
)


def _inner_project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inner project with the repo's pytest options and conftest, isolated from the .env."""
    options = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    # The inner project has its own tests; path settings of this repo do not apply to it.
    options = {k: v for k, v in options.items() if k not in ("testpaths", "pythonpath")}
    lines = ["[tool.pytest.ini_options]"]
    for key, value in options.items():
        if isinstance(value, list):
            items = ", ".join(f'"{v}"' for v in value)
            lines.append(f"{key} = [{items}]")
        else:
            lines.append(f'{key} = "{value}"')
    pytester.makepyprojecttoml("\n".join(lines) + "\n")
    shutil.copy(ROOT / "tests" / "conftest.py", pytester.path / "conftest.py")
    # A repo-root .env must not supply the variable: point the repo root at an empty directory.
    fake_root = pytester.path.parent / "fake-repo-root"
    fake_root.mkdir(exist_ok=True)
    monkeypatch.setattr(env, "repo_root", lambda: fake_root)


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest("-p", "no:cacheprovider", "--import-mode=importlib", *args)


def test_real_fixtures_skips_when_the_data_dir_is_unset(pytester, monkeypatch):
    _inner_project(pytester, monkeypatch)
    pytester.makepyfile(test_inner=INNER_LOCAL_TEST)
    monkeypatch.setenv("EXPECTED_PAGE", "unused")
    for value in (None, "", "   "):
        if value is None:
            monkeypatch.delenv(VAR, raising=False)
        else:
            monkeypatch.setenv(VAR, value)
        for extra in ((), ("-m", "local")):
            result = _run(pytester, *extra)
            result.assert_outcomes(skipped=1)
            assert result.ret == 0
            result.stdout.fnmatch_lines(["SKIPPED*HORSETRACK_DATA_DIR*"])


def test_real_fixtures_skips_when_the_fixtures_directory_is_missing(
    pytester, monkeypatch, tmp_path
):
    _inner_project(pytester, monkeypatch)
    pytester.makepyfile(test_inner=INNER_LOCAL_TEST)
    monkeypatch.setenv("EXPECTED_PAGE", "unused")
    absent_data_dir = tmp_path / "no-such-data-dir"
    data_dir_without_fixtures = tmp_path / "data"
    data_dir_without_fixtures.mkdir()
    (data_dir_without_fixtures / "raw").mkdir()
    for data_dir in (absent_data_dir, data_dir_without_fixtures):
        monkeypatch.setenv(VAR, str(data_dir))
        result = _run(pytester, "-m", "local")
        result.assert_outcomes(skipped=1)
        assert result.ret == 0
        result.stdout.fnmatch_lines(["SKIPPED*fixtures*"])


def test_real_fixtures_returns_the_directory_when_present(pytester, monkeypatch, tmp_path):
    _inner_project(pytester, monkeypatch)
    pytester.makepyfile(test_inner=INNER_LOCAL_TEST)
    for name, content in (("alpha", "<html>one</html>"), ("beta-data", "<html>two</html>")):
        fixtures = tmp_path / name / "fixtures"
        fixtures.mkdir(parents=True)
        (fixtures / "page.html").write_text(content)
        monkeypatch.setenv(VAR, str(tmp_path / name))
        monkeypatch.setenv("EXPECTED_PAGE", content)
        for extra in ((), ("-m", "local")):
            result = _run(pytester, *extra)
            result.assert_outcomes(passed=1)
            assert result.ret == 0
    # The marker is the selector: deselecting `local` leaves nothing to run.
    assert _run(pytester, "-m", "not local").ret == 5


def test_unknown_markers_fail_collection(pytester, monkeypatch):
    _inner_project(pytester, monkeypatch)
    monkeypatch.delenv(VAR, raising=False)
    pytester.makepyfile(
        test_known="""
        import pytest

        @pytest.mark.local
        def test_ok():
            pass
        """
    )
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    assert result.ret == 0

    for name in ("not_registered", "slow_typo"):
        pytester.makepyfile(
            test_unknown=f"""
            import pytest

            @pytest.mark.{name}
            def test_bad():
                pass
            """
        )
        result = _run(pytester, "test_unknown.py")
        assert result.ret != 0
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines([f"*'{name}' not found in `markers` configuration option*"])
