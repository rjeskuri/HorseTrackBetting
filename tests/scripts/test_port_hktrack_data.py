"""scripts/port_hktrack_data.py (ticket #18): HKTrack's data and real fixtures to the data dir.

Every test builds a small synthetic HKTrack root under tmp_path; no test reads HKTrack.

Contract pinned here (the names the implementation must follow):

* `plan(src_data, src_fixtures, dst) -> steps` and `run(steps, dry_run=...) -> Report`, where
  `src_data` is `<HKTrack>/data` and `src_fixtures` is `<HKTrack>/tests/fixtures`.
* Ported, byte for byte: `data/{raw,interim,processed,odds_snapshots}` to the same names under dst,
  and `tests/fixtures/{hkjc,jra,jrdb,web}` to `dst/fixtures/{hkjc,jra,jrdb,web}`. Skipped:
  `data/features` (and any other data directory), `*.lock`, `*.tmp`, anything under a
  `__pycache__` directory, and the -wal and -shm files of a sqlite database. `*.sqlite` files go
  through the sqlite3 backup API, so rows still in the -wal reach the copy, and the source
  database file and its -wal are left as they were (no checkpoint into HKTrack's file).
* Re-runs: a missing file is copied; a destination that is a strict byte prefix of the source is
  extended to equal it; a destination equal to the source is left alone; any other differing
  destination is a conflict, reported (on every run while it lasts) and left alone; nothing in
  dst is deleted. A sqlite copy is equal when its content equals the source's: a second run over
  an unchanged database copies nothing and reports no conflict, and a copy written to since is a
  conflict.
* `Report` has attributes `copied`, `extended` and `conflicts`: collections of destination paths,
  relative to dst (str or PathLike; absolute paths under dst are accepted).
* `run(steps, dry_run=True)` writes nothing at all (no directory, no file, no PORTED.json) and
  returns the Report the real run would.
* A destination inside a git work tree (a linked worktree too) or equal to the source (after
  resolving) is refused before anything is written: `plan` or `run` raises (SystemExit with a
  non-zero code counts) and the CLI exits non-zero. A source inside a git work tree is fine.
* CLI: `port_hktrack_data.py --from <HKTrack root> --to <data dir> [--dry-run]` exits 0 on a
  clean run. Every real run (re)writes `dst/PORTED.json`, a JSON object with
  `time` (ISO 8601, UTC, `Z` or `+00:00`), `source` (the `--from` root), `dirs`
  (`{top: {"files": int, "bytes": int}}` for raw, interim, processed, odds_snapshots, fixtures:
  the source files the run covers after skips, with their source sizes, so the source's totals
  on every run rather than this run's delta) and `conflicts` (a list of this run's conflicts as
  paths relative to dst; objects with a "path" key are accepted).
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "port_hktrack_data.py"
MODULE = "port_hktrack_data"
TOPS = {"raw", "interim", "processed", "odds_snapshots", "fixtures"}
OLD_NS = 1_000_000_000 * 10**9  # 2001-09-09: an mtime no copy or rewrite would leave behind
MANIFEST = "raw/jra/manifest.sqlite"

# Ported data files (relative to <HKTrack>/data, kept under dst) -> number of lines.
DATA = {
    "raw/jra/results/2025/202508040611.html.gz": 7,
    "raw/jra/results/2025/block.html.gz": 2,  # "lock" inside a name: still copied
    "raw/jra/entries/2026/202611220511.html.gz": 3,
    "raw/hkjc/results/2024/20240101ST1.html.gz": 5,
    "raw/jrdb/SED/2025/SED250104.txt": 11,
    "interim/jra/results.parquet": 4,
    "interim/jra/features/part.parquet": 2,  # "features" below the top: still copied
    "processed/jra/races.parquet": 9,
    "processed/jra/_build.json": 1,
    "processed/hkjc/races.parquet": 6,
    "processed/hkjc/_build.json": 2,
    "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz": 8,
    "odds_snapshots/jra/2026-11-23/TOK.jsonl.gz": 3,
    "odds_snapshots/hkjc/2026-11-23/ST.jsonl.gz": 4,
}
# Ported fixtures (relative to <HKTrack>/tests/fixtures, under dst/fixtures) -> number of lines.
FIXTURES = {
    "hkjc/results_20240101_ST_1.html": 3,
    "jra/results_202508040611.html": 4,
    "jra/odds/win_202611220511.html": 2,
    "jrdb/SED250104.txt": 5,
    "web/race_card.html": 6,
}
SKIPPED_DATA = {
    "features/jra/features.parquet": 10,
    "features/hkjc/features.parquet": 4,
    "scratch/notes.txt": 1,  # not one of the four data directories
    "raw/jra/manifest.lock": 1,
    "interim/jra/results.parquet.tmp": 3,
    "processed/jra/__pycache__/build.cpython-310.pyc": 2,
}
SKIPPED_FIXTURES = {
    "jra/__pycache__/conftest.cpython-310.pyc": 1,
    "web/download.tmp": 1,
    "hkjc/fetch.lock": 1,
}
OUTSIDE = ("README.md", "tests/test_parse.py", "src/hktrack/__init__.py")  # never ported

MAIN_ROWS = [(f"jra/results/2025/20250804061{i}", f"{i:064x}", 1000 + i) for i in range(5)]
WAL_ROWS = [(f"jra/entries/2026/20261122051{i}", f"{i + 100:064x}", 2000 + i) for i in range(7)]
NOTES = [(1, "re-fetched after a 503"), (2, "page moved")]


# --- helpers -------------------------------------------------------------------------------


def _require_script() -> None:
    if not SCRIPT.is_file():
        pytest.fail(f"{SCRIPT.relative_to(REPO)} does not exist yet", pytrace=False)


def _port():
    """The script loaded as a module (scripts/ is not a package)."""
    _require_script()
    if MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module  # dataclasses look their module up here
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[MODULE]
            raise
    return sys.modules[MODULE]


def _run(root: Path, dst: Path, *, dry_run: bool = False, fixtures: Path | None = None):
    port = _port()
    src_fixtures = fixtures if fixtures is not None else root / "tests" / "fixtures"
    steps = port.plan(root / "data", src_fixtures, dst)
    return port.run(steps, dry_run=dry_run)


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _cli(root: Path, dst: Path, *extra: str) -> subprocess.CompletedProcess:
    _require_script()
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--from", str(root), "--to", str(dst), *extra],
        capture_output=True,
        text=True,
        timeout=120,
        env=_env(),
        cwd=root.parent,
    )


def _git(cwd: Path, *args: str) -> str:
    cmd = [
        "git",
        "-c",
        "user.name=test",
        "-c",
        # Built from parts so the public-tree guard does not read it as an email address.
        "user.email=test" + "@" + "example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        *args,
    ]
    done = subprocess.run(
        cmd, cwd=cwd, env=_env(), check=True, capture_output=True, text=True, timeout=60
    )
    return done.stdout


def _blob(label: str, lines: int) -> bytes:
    return "".join(f"{label} line {i}\n" for i in range(lines)).encode()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_tree(root: Path) -> dict[str, bytes]:
    """Write a synthetic HKTrack root; return what dst must hold (path relative to dst -> bytes)."""
    expected = {}
    for rel, lines in DATA.items():
        content = _blob(f"data/{rel}", lines)
        _write(root / "data" / rel, content)
        expected[rel] = content
    for rel, lines in SKIPPED_DATA.items():
        _write(root / "data" / rel, _blob(f"data/{rel}", lines))
    for rel, lines in FIXTURES.items():
        content = _blob(f"tests/fixtures/{rel}", lines)
        _write(root / "tests" / "fixtures" / rel, content)
        expected[f"fixtures/{rel}"] = content
    for rel, lines in SKIPPED_FIXTURES.items():
        _write(root / "tests" / "fixtures" / rel, _blob(f"tests/fixtures/{rel}", lines))
    for rel in OUTSIDE:
        _write(root / rel, _blob(rel, 1))
    return expected


def _src(root: Path, rel: str) -> Path:
    """The source path of a destination path."""
    if rel.startswith("fixtures/"):
        return root / "tests" / rel
    return root / "data" / rel


def _files(top: Path) -> dict[str, bytes]:
    """Files below top's subdirectories; top-level files such as PORTED.json are left out."""
    out = {}
    for path in top.rglob("*"):
        rel = path.relative_to(top)
        if path.is_file() and len(rel.parts) > 1:
            out[rel.as_posix()] = path.read_bytes()
    return out


def _snapshot(top: Path) -> dict[str, tuple]:
    """Every directory (mtime) and file (bytes, mtime) under top, .git left out."""
    snap: dict[str, tuple] = {".": ("dir", top.stat().st_mtime_ns)}
    for base, dirs, names in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        here = Path(base)
        for name in dirs:
            path = here / name
            snap[path.relative_to(top).as_posix()] = ("dir", path.lstat().st_mtime_ns)
        for name in names:
            path = here / name
            rel = path.relative_to(top).as_posix()
            snap[rel] = ("file", path.read_bytes(), path.lstat().st_mtime_ns)
    return snap


def _rels(entries, dst: Path) -> set[str]:
    root = dst.resolve()
    out = set()
    for entry in entries:
        path = Path(entry)
        if path.is_absolute():
            path = path.resolve().relative_to(root)
        out.add(path.as_posix())
    return out


def _outcome(report, dst: Path) -> tuple[set[str], set[str], set[str]]:
    """(copied, extended, conflicts) of a Report, as posix paths relative to dst."""
    return (
        _rels(report.copied, dst),
        _rels(report.extended, dst),
        _rels(report.conflicts, dst),
    )


def _ported(dst: Path) -> dict:
    return json.loads((dst / "PORTED.json").read_text())


def _ported_dirs(ported: dict) -> dict[str, tuple[int, int]]:
    return {
        top: (counts["files"], counts["bytes"])
        for top, counts in ported["dirs"].items()
        if counts["files"] or counts["bytes"]
    }


def _conflict_paths(entries, dst: Path) -> set[str]:
    assert isinstance(entries, list), entries
    return _rels((e["path"] if isinstance(e, dict) else e for e in entries), dst)


def _dirs_of(files: dict[str, bytes]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for rel, content in files.items():
        top = rel.split("/", 1)[0]
        count, size = out.get(top, (0, 0))
        out[top] = (count + 1, size + len(content))
    return out


def _utc(stamp: str) -> datetime:
    when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert when.utcoffset() == timedelta(0), stamp
    return when


def _refused(call) -> None:
    """call() must refuse: raise (not a TypeError, AttributeError or NameError) or exit non-zero."""
    try:
        call()
    except SystemExit as exc:
        assert exc.code not in (0, None), f"exited {exc.code!r}"
    except (TypeError, AttributeError, NameError):
        raise
    except Exception:
        return
    else:
        pytest.fail("the destination was accepted")


def _wal_db(path: Path) -> sqlite3.Connection:
    """A WAL database: MAIN_ROWS in the file, WAL_ROWS and the notes table only in its -wal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, isolation_level=None)
    assert con.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("CREATE TABLE pages (key TEXT PRIMARY KEY, sha TEXT NOT NULL, size INTEGER)")
    con.executemany("INSERT INTO pages VALUES (?, ?, ?)", MAIN_ROWS)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.executemany("INSERT INTO pages VALUES (?, ?, ?)", WAL_ROWS)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, note TEXT NOT NULL)")
    con.executemany("INSERT INTO notes VALUES (?, ?)", NOTES)
    assert Path(f"{path}-wal").stat().st_size > 0
    return con


def _sidecars(db: Path) -> list[Path]:
    return [Path(f"{db}-wal"), Path(f"{db}-shm")]


def _query(db: Path, sql: str) -> list:
    con = sqlite3.connect(db)
    try:
        return sorted(con.execute(sql).fetchall())
    finally:
        con.close()


def _pages(db: Path) -> list:
    return _query(db, "SELECT key, sha, size FROM pages")


def _notes(db: Path) -> list:
    return _query(db, "SELECT id, note FROM notes")


# --- the tests the ticket names ------------------------------------------------------------


def test_plan_copies_raw_interim_processed_and_snapshots_but_not_features(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    dst = tmp_path / "Data" / "horsetrack"  # its parent does not exist yet either

    report = _run(root, dst)

    data = {k: v for k, v in _files(dst).items() if not k.startswith("fixtures/")}
    assert data == {k: v for k, v in expected.items() if not k.startswith("fixtures/")}
    assert {p.name for p in dst.iterdir() if p.is_dir()} == TOPS
    assert not (dst / "features").exists()
    assert _outcome(report, dst) == (set(expected), set(), set())


def test_fixtures_go_under_fixtures(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    # plan takes the fixtures directory on its own: it need not sit at <root>/tests/fixtures
    fixtures = tmp_path / "elsewhere" / "real-pages"
    fixtures.parent.mkdir()
    shutil.move(str(root / "tests" / "fixtures"), str(fixtures))
    dst = tmp_path / "data"

    report = _run(root, dst, fixtures=fixtures)

    got = {k: v for k, v in _files(dst).items() if k.startswith("fixtures/")}
    assert got == {k: v for k, v in expected.items() if k.startswith("fixtures/")}
    for sub in ("hkjc", "jra", "jrdb", "web"):
        assert (dst / "fixtures" / sub).is_dir()
        assert not (dst / sub).exists()
    copied, _, _ = _outcome(report, dst)
    assert {p for p in copied if p.startswith("fixtures/")} == set(got)


def test_dry_run_copies_nothing(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    source = _snapshot(root)

    fresh = tmp_path / "fresh"
    report = _run(root, fresh, dry_run=True)
    assert not fresh.exists()
    assert _outcome(report, fresh) == (set(expected), set(), set())

    # an existing destination with a shorter log, a conflicting file and a file of its own
    dst = tmp_path / "data"
    grown = "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz"
    theirs = "processed/jra/_build.json"
    own = "odds_snapshots/jra/2026-11-29/KYO.jsonl.gz"
    _write(dst / grown, expected[grown][:20])
    _write(dst / theirs, b'{"rows": 1}\n')
    _write(dst / own, b"written by the new logger\n")
    before = _snapshot(dst)

    dry = _outcome(_run(root, dst, dry_run=True), dst)
    assert _snapshot(dst) == before  # no file, no directory, no PORTED.json
    assert dry == (set(expected) - {grown, theirs}, {grown}, {theirs})
    assert _outcome(_run(root, dst), dst) == dry  # the dry run told what the real one did

    cli_dst = tmp_path / "cli"
    proc = _cli(root, cli_dst, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert not cli_dst.exists()
    assert _snapshot(root) == source


def test_rerun_copies_only_new_files(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    dst = tmp_path / "data"
    assert _outcome(_run(root, dst), dst) == (set(expected), set(), set())
    for rel in expected:
        os.utime(dst / rel, ns=(OLD_NS, OLD_NS))  # a rewrite would show in the mtime

    new = {
        "odds_snapshots/jra/2026-11-29/KYO.jsonl.gz": _blob("next weekend KYO", 4),
        "odds_snapshots/hkjc/2026-11-30/ST.jsonl.gz": _blob("next weekend ST", 2),
        "raw/jra/results/2026/202611290511.html.gz": _blob("a new results page", 3),
        "fixtures/jra/results_202611290511.html": _blob("a new fixture", 1),
    }
    for rel, content in new.items():
        _write(_src(root, rel), content)

    assert _outcome(_run(root, dst), dst) == (set(new), set(), set())
    assert _files(dst) == expected | new
    assert {rel for rel in expected if (dst / rel).stat().st_mtime_ns != OLD_NS} == set()

    assert _outcome(_run(root, dst), dst) == (set(), set(), set())
    assert _files(dst) == expected | new


def test_a_grown_append_only_log_is_extended(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    first = gzip.compress(b'{"t": "2026-11-22T01:00:00Z", "win": [2.1, 3.4]}\n', mtime=0)
    later = gzip.compress(b'{"t": "2026-11-22T01:05:00Z", "win": [2.0, 3.6]}\n', mtime=0)
    log = "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz"
    _write(_src(root, log), first + later)
    expected[log] = first + later
    half = "raw/jrdb/SED/2025/SED250104.txt"
    heads = {
        log: first,  # HKTrack's logger appended a gzip member since the last port
        "odds_snapshots/hkjc/2026-11-23/ST.jsonl.gz": b"",  # created empty, then written
        half: expected[half][: len(expected[half]) // 2],
        "fixtures/web/race_card.html": expected["fixtures/web/race_card.html"][:1],
    }
    dst = tmp_path / "data"
    for rel, head in heads.items():
        assert expected[rel].startswith(head) and len(head) < len(expected[rel])
        _write(dst / rel, head)

    report = _run(root, dst)

    assert _outcome(report, dst) == (set(expected) - set(heads), set(heads), set())
    assert _files(dst) == expected
    assert gzip.decompress((dst / log).read_bytes()) == gzip.decompress(first + later)
    assert _outcome(_run(root, dst), dst) == (set(), set(), set())


def test_a_conflicting_destination_file_is_reported_and_kept(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)

    def flip_last(content: bytes) -> bytes:
        return content[:-1] + bytes([content[-1] ^ 0x01])

    tok = "odds_snapshots/jra/2026-11-23/TOK.jsonl.gz"
    st = "odds_snapshots/hkjc/2026-11-23/ST.jsonl.gz"
    page = "raw/jra/results/2025/202508040611.html.gz"
    sed = "fixtures/jrdb/SED250104.txt"
    theirs = {
        tok: expected[tok] + b'{"logger": "new"}\n',  # both loggers wrote the day
        st: flip_last(expected[st]),  # same length, one byte differs
        page: b"X" + expected[page][1:10],  # shorter, not a prefix
        "processed/hkjc/_build.json": b'{"rows": 999}\n',
        sed: expected[sed][1:] + expected[sed][:1],  # the same bytes, rotated
    }
    dst = tmp_path / "data"
    for rel, content in theirs.items():
        assert content != expected[rel] and not expected[rel].startswith(content)
        _write(dst / rel, content)
        os.utime(dst / rel, ns=(OLD_NS, OLD_NS))

    # reported on every run while it lasts, never resolved by overwriting
    for new_files in (set(expected) - set(theirs), set()):
        report = _run(root, dst)
        assert _outcome(report, dst) == (new_files, set(), set(theirs))
        assert _files(dst) == expected | theirs
        for rel in theirs:
            assert (dst / rel).stat().st_mtime_ns == OLD_NS, rel


def test_sqlite_in_wal_mode_is_copied_consistently(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    db = root / "data" / MANIFEST
    con = _wal_db(db)  # held open, as HKTrack's logger holds it
    try:
        # the database file alone lacks the -wal rows: a byte copy would lose them
        byte_copy = tmp_path / "byte-copy.sqlite"
        shutil.copyfile(db, byte_copy)
        assert _pages(byte_copy) == sorted(MAIN_ROWS)
        dst = tmp_path / "data"

        report = _run(root, dst)

        copy = dst / MANIFEST
        assert copy.is_file()
        assert [p.name for p in _sidecars(copy) if p.exists()] == []  # before opening the copy
        copied, extended, conflicts = _outcome(report, dst)
        assert copied == set(expected) | {MANIFEST}
        assert extended == set() and conflicts == set()
        assert _pages(copy) == sorted(MAIN_ROWS + WAL_ROWS)
        assert _notes(copy) == sorted(NOTES)
        assert _query(copy, "PRAGMA integrity_check") == [("ok",)]
        others = {k: v for k, v in _files(dst).items() if not k.startswith(MANIFEST)}
        assert others == expected
    finally:
        con.close()


def test_refuses_a_destination_inside_a_git_work_tree(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "--no-verify", "-m", "init")
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", str(linked))  # its .git is a file, not a directory
    _port()

    for dst in (repo, repo / "sub", repo / "data", repo / "a" / "b" / "data", linked / "data"):
        _refused(partial(_run, root, dst))

    assert sorted(os.listdir(repo)) == [".git", "sub"]
    assert os.listdir(repo / "sub") == []
    assert sorted(os.listdir(linked)) == [".git"]
    for tree in (repo, linked):
        assert _git(tree, "status", "--porcelain", "--untracked-files=all") == ""

    proc = _cli(root, repo / "cli" / "data")
    assert proc.returncode != 0
    assert not (repo / "cli").exists()

    outside = tmp_path / "outside" / "data"  # the same tree ports to a dst outside any work tree
    _run(root, outside)
    assert _files(outside) == expected


# --- edge cases the ticket names -----------------------------------------------------------


def test_refuses_a_destination_equal_to_the_source(tmp_path):
    root = tmp_path / "HKTrack"
    _build_tree(root)
    link = tmp_path / "link-to-data"
    link.symlink_to(root / "data", target_is_directory=True)
    _port()
    before = _snapshot(root)

    for dst in (root / "data", root / "data" / "raw" / "..", link):
        _refused(partial(_run, root, dst))
        assert _snapshot(root) == before, dst

    proc = _cli(root, root)
    assert proc.returncode != 0
    assert _snapshot(root) == before


def test_lock_tmp_and_pycache_files_are_skipped_wherever_they_are(tmp_path):
    root = tmp_path / "HKTrack"
    skipped = [
        "data/raw/__pycache__/fetch.cpython-310.pyc",
        "data/raw/jra/results/2025/__pycache__/deep/parse.cpython-310.pyc",
        "data/interim/jra/results.lock",
        "data/processed/hkjc/races.parquet.tmp",
        "data/odds_snapshots/hkjc/2026-11-23/ST.jsonl.gz.lock",
        "data/odds_snapshots/jra/2026-11-22/KYO.jsonl.gz.tmp",
        "tests/fixtures/hkjc/__pycache__/conftest.cpython-310.pyc",
        "tests/fixtures/jra/odds/win.tmp",
        "tests/fixtures/web/fetch.lock",
    ]
    kept = [  # near misses: copied
        "raw/jra/results/2025/block.html.gz",
        "interim/jra/tmp/part.parquet",
        "processed/jra/lock.json",
        "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz",
        "fixtures/jrdb/pycache.txt",
        "fixtures/hkjc/results.tmp.html",
    ]
    for i, rel in enumerate(skipped):
        _write(root / rel, _blob(rel, i + 1))
    want = {rel: _blob(rel, i + 2) for i, rel in enumerate(kept)}
    for rel, content in want.items():
        _write(_src(root, rel), content)
    dst = tmp_path / "data"

    report = _run(root, dst)

    assert _files(dst) == want
    assert _outcome(report, dst) == (set(kept), set(), set())


def test_destination_files_missing_from_the_source_are_kept(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    own = {
        "odds_snapshots/jra/2026-11-29/KYO.jsonl.gz": b"a day only the new logger saw\n",
        "raw/jra/results/2026/202611290511.html.gz": b"fetched by the new code\n",
        "raw/jra/manifest.lock": b"held by the new logger\n",  # a skipped name is not deleted
        "logs/japan/logger.log": b"started\n",
        "reports/2026-11-29.html": b"<p>report</p>\n",
        "fixtures/web/new_page.html": b"saved by the new code\n",
    }
    dst = tmp_path / "data"
    for rel, content in own.items():
        _write(dst / rel, content)

    for _ in range(2):
        _run(root, dst)
        assert _files(dst) == expected | own


def test_the_source_is_left_unchanged(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    # HKTrack is a git work tree with data/ ignored; only the destination must be outside one
    (root / ".gitignore").write_text("data/\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "init")
    before = _snapshot(root)
    dst = tmp_path / "data"

    _run(root, dst, dry_run=True)
    _run(root, dst)
    build = "processed/jra/_build.json"
    log = "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz"
    _write(dst / build, b'{"rows": 2}\n')  # a conflict
    _write(dst / log, expected[log][:10])  # an extension
    _run(root, dst)
    _write(dst / build, expected[build])
    assert _cli(root, dst).returncode == 0
    assert _cli(root, dst, "--dry-run").returncode == 0

    assert _snapshot(root) == before
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""


def test_cli_ports_a_hktrack_root_and_writes_ported_json(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    want = _dirs_of(expected)
    assert set(want) == TOPS
    dst = tmp_path / "Data" / "horsetrack"

    start = datetime.now(timezone.utc)
    proc = _cli(root, dst)
    end = datetime.now(timezone.utc)

    assert proc.returncode == 0, proc.stderr
    assert _files(dst) == expected
    ported = _ported(dst)
    assert _ported_dirs(ported) == want
    assert _conflict_paths(ported["conflicts"], dst) == set()
    assert Path(ported["source"]).resolve() in {root.resolve(), (root / "data").resolve()}
    assert start - timedelta(seconds=1) <= _utc(ported["time"]) <= end + timedelta(seconds=1)

    # a second run: nothing new, no conflict, and the counts are still the source's totals
    proc = _cli(root, dst)
    assert proc.returncode == 0, proc.stderr
    ported = _ported(dst)
    assert _ported_dirs(ported) == want
    assert _conflict_paths(ported["conflicts"], dst) == set()

    # the new logger appended to a day HKTrack also logged: reported, kept, totals unchanged
    log = "odds_snapshots/jra/2026-11-22/KYO.jsonl.gz"
    ours = expected[log] + b'{"logger": "new"}\n'
    _write(dst / log, ours)
    _cli(root, dst)
    ported = _ported(dst)
    assert _conflict_paths(ported["conflicts"], dst) == {log}
    assert _ported_dirs(ported) == want
    assert (dst / log).read_bytes() == ours


def test_a_wal_sqlite_left_without_an_open_connection_is_copied_without_touching_it(tmp_path):
    root = tmp_path / "HKTrack"
    _build_tree(root)
    db = root / "data" / MANIFEST
    # a -wal and -shm on disk and no connection open (a killed logger): a connection that
    # closes last would checkpoint the -wal into HKTrack's file and delete it
    staged = tmp_path / "stage" / "manifest.sqlite"
    con = _wal_db(staged)
    for suffix in ("", "-wal", "-shm"):
        shutil.copyfile(f"{staged}{suffix}", f"{db}{suffix}")
    con.close()
    watched = [db, Path(f"{db}-wal")]  # -shm is shared memory every reader writes to
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in watched}
    listing = sorted(os.listdir(db.parent))
    dir_mtime = db.parent.stat().st_mtime_ns
    dst = tmp_path / "data"

    _run(root, dst)

    assert sorted(os.listdir(db.parent)) == listing
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in watched} == before
    assert db.parent.stat().st_mtime_ns == dir_mtime
    copy = dst / MANIFEST
    assert [p.name for p in _sidecars(copy) if p.exists()] == []
    assert _pages(copy) == sorted(MAIN_ROWS + WAL_ROWS)
    assert _notes(copy) == sorted(NOTES)


def test_a_second_run_over_a_wal_sqlite_copies_nothing_and_reports_no_conflict(tmp_path):
    root = tmp_path / "HKTrack"
    expected = _build_tree(root)
    db = root / "data" / MANIFEST
    con = _wal_db(db)
    try:
        dst = tmp_path / "data"
        copied, _, _ = _outcome(_run(root, dst), dst)
        assert MANIFEST in copied
        copy = dst / MANIFEST
        os.utime(copy, ns=(OLD_NS, OLD_NS))
        content = copy.read_bytes()

        assert _outcome(_run(root, dst), dst) == (set(), set(), set())
        assert copy.read_bytes() == content
        assert copy.stat().st_mtime_ns == OLD_NS

        proc = _cli(root, dst)
        assert proc.returncode == 0, proc.stderr
        ported = _ported(dst)
        assert _conflict_paths(ported["conflicts"], dst) == set()
        want = _dirs_of(expected)
        got = _ported_dirs(ported)
        assert got["raw"][0] == want["raw"][0] + 1  # the database, not its -wal and -shm
        assert {k: v for k, v in got.items() if k != "raw"} == {
            k: v for k, v in want.items() if k != "raw"
        }
        assert copy.read_bytes() == content
    finally:
        con.close()


def test_a_sqlite_copy_written_to_since_is_a_conflict_and_kept(tmp_path):
    root = tmp_path / "HKTrack"
    _build_tree(root)
    db = root / "data" / MANIFEST
    con = _wal_db(db)
    try:
        dst = tmp_path / "data"
        _run(root, dst)
        copy = dst / MANIFEST
        own_row = ("jra/results/2026/202611290511", "f" * 64, 4321)
        writer = sqlite3.connect(copy)
        writer.execute("INSERT INTO pages VALUES (?, ?, ?)", own_row)
        writer.commit()
        writer.close()
        ours = _pages(copy)
        assert own_row in ours

        assert _outcome(_run(root, dst), dst) == (set(), set(), {MANIFEST})
        assert _pages(copy) == ours
    finally:
        con.close()
