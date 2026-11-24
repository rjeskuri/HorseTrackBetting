"""The data directory and the on-disk layout under it (kept from HKTrack).

Seam the implementation must honour: `horsetrack.core.env.repo_root()` (see test_env.py).
`horsetrack.core.paths` must call `env.get_setting(...)` and `env.repo_root()` through the module
so these tests can monkeypatch `horsetrack.core.env.repo_root` and the HORSETRACK_DATA_DIR variable.

Layout pinned here (D = data_dir()): raw/<source>/<page_type>/<yyyy>/<key>.html.gz,
raw/<source>/manifest.sqlite, interim/<source>, processed/<source>,
odds_snapshots/<source>/<date>/<venue>.jsonl.gz, logs/<country>, reports, fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from horsetrack.core import paths

VAR = "HORSETRACK_DATA_DIR"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.delenv(VAR, raising=False)
    return root


@pytest.fixture
def D(tmp_path, monkeypatch, fake_root) -> Path:
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def test_data_dir_comes_from_the_environment(tmp_path, monkeypatch, fake_root):
    for name in ("one", "two"):
        target = tmp_path / name
        monkeypatch.setenv(VAR, str(target))
        assert paths.data_dir() == target.resolve()


def test_data_dir_falls_back_to_the_dot_env_file(tmp_path, fake_root):
    target = tmp_path / "from-file"
    (fake_root / ".env").write_text(f"# note\n\n{VAR}={target}\n")
    assert paths.data_dir() == target.resolve()


def test_environment_wins_over_dot_env(tmp_path, monkeypatch, fake_root):
    (fake_root / ".env").write_text(f"{VAR}={tmp_path / 'file'}\n")
    monkeypatch.setenv(VAR, str(tmp_path / "env"))
    assert paths.data_dir() == (tmp_path / "env").resolve()


def test_tilde_is_expanded(tmp_path, monkeypatch, fake_root):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(VAR, "~/x")
    assert paths.data_dir() == (home / "x").resolve()
    (fake_root / ".env").write_text(f"{VAR}=~/y\n")
    monkeypatch.delenv(VAR)
    assert paths.data_dir() == (home / "y").resolve()


def test_relative_segments_are_resolved(tmp_path, monkeypatch, fake_root):
    monkeypatch.setenv(VAR, str(tmp_path / "a" / ".." / "b"))
    assert paths.data_dir() == (tmp_path / "b").resolve()


def test_unset_data_dir_raises_data_dir_not_set(fake_root):
    with pytest.raises(paths.DataDirNotSet) as exc:
        paths.data_dir()
    assert issubclass(paths.DataDirNotSet, RuntimeError)
    assert "\n" not in str(exc.value)
    assert VAR in str(exc.value)


def test_unset_raises_from_every_helper(fake_root):
    for call in (
        lambda: paths.raw_dir("jra"),
        lambda: paths.manifest_path("jra"),
        lambda: paths.raw_page_path("jra", "results", 2025, "k"),
        lambda: paths.interim_dir("jra"),
        lambda: paths.processed_dir("jra"),
        lambda: paths.odds_snapshot_path("jra", "2026-11-22", "KYO"),
        lambda: paths.logs_dir("japan"),
        lambda: paths.reports_dir(),
        lambda: paths.fixtures_dir(),
    ):
        with pytest.raises(paths.DataDirNotSet):
            call()


def test_a_data_dir_inside_the_repo_is_refused(monkeypatch, fake_root):
    for inside in (fake_root / "data", fake_root / "a" / "b", fake_root):
        monkeypatch.setenv(VAR, str(inside))
        with pytest.raises(paths.DataDirInsideRepo) as exc:
            paths.data_dir()
        assert issubclass(paths.DataDirInsideRepo, RuntimeError)
        assert "\n" not in str(exc.value)


def test_inside_repo_is_refused_when_it_comes_from_dot_env_or_a_symlink(
    tmp_path, monkeypatch, fake_root
):
    (fake_root / ".env").write_text(f"{VAR}={fake_root / 'data'}\n")
    with pytest.raises(paths.DataDirInsideRepo):
        paths.data_dir()

    (fake_root / ".env").unlink()
    link = tmp_path / "link"
    link.symlink_to(fake_root, target_is_directory=True)
    monkeypatch.setenv(VAR, str(link / "data"))
    with pytest.raises(paths.DataDirInsideRepo):
        paths.data_dir()


def test_a_sibling_sharing_the_repo_name_prefix_is_allowed(tmp_path, monkeypatch, fake_root):
    sibling = tmp_path / "repo-data"
    monkeypatch.setenv(VAR, str(sibling))
    assert paths.data_dir() == sibling.resolve()
    monkeypatch.setenv(VAR, str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()


def test_layout_matches_hktrack(D):
    assert paths.raw_page_path("jra", "results", 2025, "202508040611") == (
        D / "raw" / "jra" / "results" / "2025" / "202508040611.html.gz"
    )
    assert paths.odds_snapshot_path("jra", "2026-11-22", "KYO") == (
        D / "odds_snapshots" / "jra" / "2026-11-22" / "KYO.jsonl.gz"
    )
    assert paths.manifest_path("jra") == D / "raw" / "jra" / "manifest.sqlite"
    assert paths.processed_dir("hkjc") == D / "processed" / "hkjc"
    assert paths.fixtures_dir() == D / "fixtures"


def test_layout_varies_with_every_argument(D):
    assert paths.raw_dir("nar") == D / "raw" / "nar"
    assert paths.manifest_path("hkjc") == D / "raw" / "hkjc" / "manifest.sqlite"
    assert paths.interim_dir("jrdb") == D / "interim" / "jrdb"
    assert paths.processed_dir("jra") == D / "processed" / "jra"
    assert paths.logs_dir("japan") == D / "logs" / "japan"
    assert paths.logs_dir("hongkong") == D / "logs" / "hongkong"
    assert paths.reports_dir() == D / "reports"
    assert paths.raw_page_path("nar", "entries", 2019, "abc") == (
        D / "raw" / "nar" / "entries" / "2019" / "abc.html.gz"
    )
    assert paths.raw_page_path("jra", "results", 2025, "x") != paths.raw_page_path(
        "jra", "results", 2024, "x"
    )
    assert paths.odds_snapshot_path("nar", "2025-01-05", "OOI") == (
        D / "odds_snapshots" / "nar" / "2025-01-05" / "OOI.jsonl.gz"
    )


def test_helpers_return_paths_and_follow_the_data_dir(tmp_path, monkeypatch, fake_root):
    for name in ("d1", "d2"):
        monkeypatch.setenv(VAR, str(tmp_path / name))
        got = paths.raw_dir("jra")
        assert isinstance(got, Path)
        assert got == (tmp_path / name).resolve() / "raw" / "jra"


def test_year_is_zero_padded_to_four_digits(D):
    assert paths.raw_page_path("jra", "results", 987, "k").parent.name == "0987"


def test_env_example_documents_the_variable():
    lines = (REPO / ".env.example").read_text().splitlines()
    assert f"{VAR}=~/Data/horsetrack" in lines


def test_no_path_in_src_is_hard_coded_to_a_home_directory():
    patterns = [
        r"/Users/",
        r"/home/",
        r"Path\.home\(",
        r"expanduser\(\s*[\"']~",
        r"[\"']~/",
    ]
    offenders = []
    for py in sorted((REPO / "src").rglob("*.py")):
        text = py.read_text()
        for pat in patterns:
            if re.search(pat, text):
                offenders.append(f"{py.relative_to(REPO)}: {pat}")
    assert not offenders, offenders
