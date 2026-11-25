"""Ticket #74: ``write_report`` keeps licensed-source outputs out of the repository.

Contract pinned here (the coder implements what these tests call):

- ``write_report(path, text, *, sources, repo_root=None) -> Path`` writes ``text`` as UTF-8 to
  ``path`` atomically (a temporary file then ``os.replace``: a failed write leaves the previous
  file and no stray file) and returns the path written.
- When any source is in ``WITHHELD_SOURCES`` (a frozenset holding at least ``jrdb`` and
  ``jravan``; any spelling of them counts) and the path is anywhere inside ``repo_root``, after
  ``..``, symlinks, relative paths and, on a case-insensitive file system, letter case, it raises
  ``LicensedOutputInRepo`` (a RuntimeError) and leaves the repository as it was: no file, no
  directory, no temporary file.
- Outside the repository it writes whatever the sources; inside it writes when no source is
  withheld.
- ``sources`` is keyword-only and required: missing, positional or None is a TypeError.
- ``repo_root=None`` means the project root (``horsetrack.core.env.repo_root()``, the directory
  holding pyproject.toml, #15), called through the module so tests can point it at ``tmp_path``.

Every path is under ``tmp_path``; the repository is a synthetic directory with a pyproject.toml.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from horsetrack.core.eval.reports import WITHHELD_SOURCES, LicensedOutputInRepo, write_report

TEXT = "# E77 report\n\nlog loss 0.123456 [0.083, 0.164], n = 4321\n勝率 0.25\n"
LICENSED = [
    pytest.param(("jrdb",), id="jrdb"),
    pytest.param(("netkeiba", "jravan"), id="netkeiba-jravan"),
]


@pytest.fixture
def repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    (root / "docs" / "experiments").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n', encoding="utf-8")
    (root / "docs" / "experiments" / "README.md").write_text("committed\n", encoding="utf-8")
    return root


@pytest.fixture
def outside(tmp_path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    return path


def tree(root: Path) -> dict[str, bytes | None]:
    """Every file (its bytes) and directory (None) under ``root``."""
    return {
        str(p.relative_to(root)): (None if p.is_dir() else p.read_bytes())
        for p in sorted(root.rglob("*"))
    }


# --------------------------------------------------------------------------- inside the repo


def _at_root(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    return repo / "report.md", repo


def _in_docs(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    return repo / "docs" / "experiments" / "report.md", repo


def _in_new_directories(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    return repo / "new" / "deeper" / "report.md", repo


def _dotdot_within(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    return repo / "docs" / ".." / "report.md", repo


def _dotdot_into(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    return tmp / "outside" / ".." / "repo" / "docs" / "report.md", repo


def _symlinked_directory(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    link = tmp / "outside" / "to-repo"
    link.symlink_to(repo, target_is_directory=True)
    return link / "docs" / "report.md", repo


def _symlinked_root(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    link = tmp / "repo-link"
    link.symlink_to(repo, target_is_directory=True)
    return repo / "docs" / "report.md", link


def _relative_path(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    mp.chdir(repo)
    return Path("docs") / "report.md", repo


def _relative_dotdot(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    mp.chdir(tmp / "outside")
    return Path("..") / "repo" / "report.md", repo


def _relative_root(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    mp.chdir(tmp)
    return repo / "docs" / "report.md", Path("repo")


def _letter_case(tmp: Path, repo: Path, mp) -> tuple[Path, Path]:
    variant = tmp / "REPO"
    if not variant.exists():
        pytest.skip("case-sensitive file system: REPO is another directory")
    return variant / "docs" / "report.md", repo


INSIDE = [
    pytest.param(_at_root, id="repo-root"),
    pytest.param(_in_docs, id="docs"),
    pytest.param(_in_new_directories, id="new-directories"),
    pytest.param(_dotdot_within, id="dotdot-within"),
    pytest.param(_dotdot_into, id="dotdot-into"),
    pytest.param(_symlinked_directory, id="symlinked-directory"),
    pytest.param(_symlinked_root, id="symlinked-repo-root"),
    pytest.param(_relative_path, id="relative-path"),
    pytest.param(_relative_dotdot, id="relative-dotdot"),
    pytest.param(_relative_root, id="relative-repo-root"),
    pytest.param(_letter_case, id="letter-case"),
]


@pytest.mark.parametrize("sources", LICENSED)
@pytest.mark.parametrize("place", INSIDE)
def test_a_licensed_report_inside_the_repo_is_refused(
    tmp_path, repo, outside, monkeypatch, place, sources
):
    assert issubclass(LicensedOutputInRepo, RuntimeError)
    assert isinstance(WITHHELD_SOURCES, frozenset)
    assert {"jrdb", "jravan"} <= WITHHELD_SOURCES
    path, root = place(tmp_path, repo, monkeypatch)
    repo_before, outside_before = tree(repo), tree(outside)

    with pytest.raises(LicensedOutputInRepo):
        write_report(path, TEXT, sources=sources, repo_root=root)

    assert tree(repo) == repo_before, "a refused report changed the repository"
    assert tree(outside) == outside_before


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param(("JRDB",), id="upper-case"),
        pytest.param(("netkeiba", "JraVan"), id="mixed-case"),
        pytest.param((" jrdb",), id="leading-space"),
        pytest.param("jrdb", id="bare-string"),
    ],
)
def test_a_licensed_source_in_another_spelling_is_refused_inside_the_repo(repo, sources):
    before = tree(repo)
    with pytest.raises((LicensedOutputInRepo, ValueError, TypeError)):
        write_report(
            repo / "docs" / "experiments" / "report.md", TEXT, sources=sources, repo_root=repo
        )
    assert tree(repo) == before


# --------------------------------------------------------------------------- written


def _outside_plain(tmp: Path, repo: Path, mp) -> tuple[Path, Path, Path]:
    target = tmp / "outside" / "report.md"
    return target, repo, target


def _prefix_sibling(tmp: Path, repo: Path, mp) -> tuple[Path, Path, Path]:
    sibling = tmp / "repo-reports"
    sibling.mkdir()
    return sibling / "report.md", repo, sibling / "report.md"


def _dotdot_out_of(tmp: Path, repo: Path, mp) -> tuple[Path, Path, Path]:
    return (
        repo / "docs" / ".." / ".." / "outside" / "report.md",
        repo,
        tmp / "outside" / "report.md",
    )


def _relative_outside(tmp: Path, repo: Path, mp) -> tuple[Path, Path, Path]:
    mp.chdir(tmp / "outside")
    return Path("report.md"), repo, tmp / "outside" / "report.md"


OUTSIDE = [
    pytest.param(_outside_plain, id="outside"),
    pytest.param(_prefix_sibling, id="sibling-sharing-the-name-prefix"),
    pytest.param(_dotdot_out_of, id="dotdot-out-of-the-repo"),
    pytest.param(_relative_outside, id="relative-outside"),
]


@pytest.mark.parametrize("sources", LICENSED)
@pytest.mark.parametrize("place", OUTSIDE)
def test_a_licensed_report_outside_the_repo_is_written(
    tmp_path, repo, outside, monkeypatch, place, sources
):
    path, root, target = place(tmp_path, repo, monkeypatch)
    repo_before = tree(repo)

    written = write_report(path, TEXT, sources=sources, repo_root=root)

    assert isinstance(written, Path)
    assert written.resolve() == target.resolve()
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert sorted(p.name for p in target.parent.iterdir()) == ["report.md"], "a stray file"
    assert tree(repo) == repo_before


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param((), id="no-sources"),
        pytest.param(("netkeiba",), id="netkeiba"),
        pytest.param(("jra", "nar"), id="jra-nar"),
    ],
)
def test_an_unlicensed_report_inside_the_repo_is_written(repo, sources):
    target = repo / "docs" / "experiments" / "report.md"

    written = write_report(target, TEXT, sources=sources, repo_root=repo)

    assert written.resolve() == target.resolve()
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert sorted(p.name for p in target.parent.iterdir()) == ["README.md", "report.md"]


def test_a_failed_write_leaves_the_previous_report(repo, outside):
    target = outside / "report.md"
    target.write_text("old report\n", encoding="utf-8")

    write_report(target, TEXT, sources=("jrdb",), repo_root=repo)
    assert target.read_bytes() == TEXT.encode("utf-8")

    # A lone surrogate cannot be encoded: the write fails part-way through.
    with pytest.raises(UnicodeError):
        write_report(target, "half written \ud800 report\n", sources=("jrdb",), repo_root=repo)
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert os.listdir(outside) == ["report.md"], "the failed write left a temporary file"


def test_the_default_repo_root_is_the_project_root(repo, outside, monkeypatch):
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: repo)
    before = tree(repo)

    with pytest.raises(LicensedOutputInRepo):
        write_report(repo / "docs" / "experiments" / "report.md", TEXT, sources=("jrdb",))
    assert tree(repo) == before

    written = write_report(outside / "report.md", TEXT, sources=("jrdb",))
    assert written.resolve() == (outside / "report.md").resolve()
    assert (outside / "report.md").read_bytes() == TEXT.encode("utf-8")


def test_sources_are_required(repo, outside):
    target = outside / "report.md"

    with pytest.raises(TypeError):
        write_report(target, TEXT, repo_root=repo)
    with pytest.raises(TypeError):
        write_report(target, TEXT, ("jrdb",), repo)  # sources is keyword-only
    with pytest.raises(TypeError):
        write_report(target, TEXT, sources=None, repo_root=repo)

    assert not target.exists()
    assert os.listdir(outside) == []
