"""The incremental parse cache (#47). Ported from HKTrack src/hktrack/store/cache.py.

Seams pinned here:

* `horsetrack.core.store.cache.cache_path(source)` is
  `paths.interim_dir(source) / "parse_cache.sqlite"`.
* `ParseCache(path)` (a context manager; `close()`) creates the file and its parent, and keeps one
  row per (page_type, key): `index()` -> {(page_type, key): (sha256, parser_version,
  key_mismatch)}, `put(results)` stores the ParseResults that carry a payload (replacing the
  page's row) and skips the rest, `payloads(page_type)` yields (key, sha256, parser_version,
  payload) in key order.
* `CACHE_FORMAT` is the SQLite `user_version`; a file at any other version is emptied on open.
  It is read when a cache opens, so a bumped module constant takes effect on the next open.
* HKTrack's files are format 1 with the DDL copied verbatim below; they open and serve unchanged.

The build (a later port) parses a page again when the cache's (sha256, parser_version) for it
differs from the manifest's sha256 and the parser's version now. `_build` below is that rule,
copied from HKTrack's store/build.py `_stale_jobs`: what it can skip depends only on what the
cache persisted.
"""

from __future__ import annotations

import json
import sqlite3
import zlib

import pytest
from parse_fakes import SOURCE, YEAR, FakeParseAdapter, body, expected_payload_tables, register
from parse_fakes import write_page as _write

from horsetrack.core import paths

# isort: split
# A block of its own: while the modules are missing, they are the import error a red run reports.
from horsetrack.core.store import cache, parse_worker

VAR = "HORSETRACK_DATA_DIR"

# Verbatim from HKTrack src/hktrack/store/cache.py (CACHE_FORMAT = 1).
HKTRACK_CACHE_FORMAT = 1
HKTRACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS parsed (
    page_type      TEXT NOT NULL,
    key            TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    key_mismatch   INTEGER NOT NULL DEFAULT 0,
    n_rows         INTEGER NOT NULL DEFAULT 0,
    payload        BLOB NOT NULL,
    PRIMARY KEY (page_type, key)
)
"""
HKTRACK_SQL_PUT = "INSERT OR REPLACE INTO parsed VALUES (?, ?, ?, ?, ?, ?, ?)"


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def _job(page_type, key, sha, version):
    url = f"https://site.test/{page_type}/{key}"
    return parse_worker.ParseJob(SOURCE, page_type, key, YEAR, url, None, None, sha, version)


def _build(manifest: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    """One incremental build over `manifest` ((page_type, key) -> sha256): parse what is stale,
    cache the results; returns the pages it sent to the parser, sorted."""
    with cache.ParseCache(cache.cache_path(SOURCE)) as pc:
        index = pc.index()
        jobs = []
        for (page_type, key), sha in sorted(manifest.items()):
            version = parse_worker.parser_version(SOURCE, page_type)
            if index.get((page_type, key), ("", ""))[:2] == (sha, version):
                continue
            jobs.append(_job(page_type, key, sha, version))
        pc.put([parse_worker.parse_job(job) for job in jobs])
    return [(job.page_type, job.key) for job in jobs]


class Site:
    """The raw cache plus its manifest: page bodies by (page_type, key)."""

    def __init__(self) -> None:
        self.tables: dict[tuple[str, str], dict] = {}
        self.manifest: dict[tuple[str, str], str] = {}

    def write(self, page_type: str, key: str, tables: dict, mismatch: bool = False) -> None:
        self.tables[(page_type, key)] = tables
        self.manifest[(page_type, key)] = _write(SOURCE, page_type, key, body(tables, mismatch))


def _cached_tables(page_type: str) -> dict[str, dict]:
    """key -> the data tables (metadata tables dropped) cached for `page_type`."""
    with cache.ParseCache(cache.cache_path(SOURCE)) as pc:
        return {
            key: {
                n: t
                for n, t in parse_worker.decode_tables(payload).items()
                if not n.startswith("_")
            }
            for key, _sha, _version, payload in pc.payloads(page_type)
        }


def _index():
    with cache.ParseCache(cache.cache_path(SOURCE)) as pc:
        return pc.index()


def test_a_page_is_parsed_again_only_when_its_sha_or_parser_version_changes(data, monkeypatch):
    fake = register(monkeypatch, FakeParseAdapter(SOURCE, {"results": 1, "card": 7}))
    path = cache.cache_path(SOURCE)
    assert path == paths.interim_dir(SOURCE) / "parse_cache.sqlite"
    assert path == data / "interim" / SOURCE / "parse_cache.sqlite"

    site = Site()
    keys = {"results": ["R3", "R1", "R2"], "card": ["C2", "C1"]}  # written out of key order
    for page_type, names in keys.items():
        for i, key in enumerate(names):
            site.write(page_type, key, {"rows": [{"k": key, "i": j} for j in range(i + 1)]}, i == 1)

    # First build: everything is new.
    every = sorted(site.manifest)
    assert _build(site.manifest) == every
    assert path.is_file()
    assert sorted(fake.parsed_keys()) == every
    versions = {"results": "1", "card": "7"}
    assert _index() == {
        (pt, key): (sha, versions[pt], key == keys[pt][1])
        for (pt, key), sha in site.manifest.items()
    }
    for page_type, names in keys.items():
        cached = _cached_tables(page_type)
        assert list(cached) == sorted(names)  # payloads come in key order
        for key in names:
            assert cached[key] == expected_payload_tables(site.tables[(page_type, key)])

    # Nothing changed: nothing is parsed, however often the build runs.
    for _ in range(3):
        assert _build(site.manifest) == []
    # Rewriting a page with the same bytes is no change either.
    site.write("results", "R2", site.tables[("results", "R2")], False)
    assert _build(site.manifest) == []
    assert len(fake.pages) == len(every)

    # One page's bytes change: only that page is parsed, and its row is replaced.
    site.write("results", "R1", {"rows": [{"k": "R1", "i": 99}, {"k": "R1", "i": 100}]})
    assert _build(site.manifest) == [("results", "R1")]
    assert _cached_tables("results")["R1"] == expected_payload_tables(
        site.tables[("results", "R1")]
    )
    assert _index()[("results", "R1")] == (site.manifest[("results", "R1")], "1", False)
    assert len(_index()) == len(every)

    # A parser version bump re-parses every page of that type, and only those.
    fake.versions["card"] = 8
    assert _build(site.manifest) == [("card", "C1"), ("card", "C2")]
    assert {k: v[1] for k, v in _index().items()} == {
        k: ("8" if k[0] == "card" else "1") for k in site.manifest
    }
    assert _build(site.manifest) == []

    # A version change in either direction counts, and a new page is parsed on its own.
    fake.versions["results"] = 0
    site.write("card", "C9", {"rows": []})
    assert _build(site.manifest) == [("card", "C9")] + [
        ("results", k) for k in sorted(keys["results"])
    ]
    assert _build(site.manifest) == []
    assert {k: v[:2] for k, v in _index().items()} == {
        k: (sha, "8" if k[0] == "card" else "0") for k, sha in site.manifest.items()
    }


def test_failed_parses_are_never_cached(data, monkeypatch):
    fake = register(monkeypatch, FakeParseAdapter(SOURCE, {"results": 2}))
    site = Site()
    for key in ("G1", "G2", "B1", "B2", "G3"):
        site.write("results", key, {"rows": [{"k": key}]})
    fake.broken.update({"B1": ValueError("bad"), "B2": KeyError("odds")})
    site.manifest[("results", "U1")] = "f" * 64  # in the manifest, missing from the raw cache
    _write(SOURCE, "results", "U2", b"not the body the manifest recorded")
    site.manifest[("results", "U2")] = "e" * 64

    # Failures and unreadable pages leave no row, so every build tries them again.
    for attempt in range(1, 4):
        tried = _build(site.manifest)
        expected = (
            ["B1", "B2", "G1", "G2", "G3", "U1", "U2"] if attempt == 1 else ["B1", "B2", "U1", "U2"]
        )
        assert tried == [("results", k) for k in expected]
        assert sorted(_index()) == [("results", k) for k in ("G1", "G2", "G3")]
        assert sorted(_cached_tables("results")) == ["G1", "G2", "G3"]

    # The parser is fixed: the very next build picks the pages up, with no flag.
    fake.broken.clear()
    assert _build(site.manifest) == [("results", k) for k in ("B1", "B2", "U1", "U2")]
    assert sorted(_index()) == [("results", k) for k in ("B1", "B2", "G1", "G2", "G3")]

    # A cached page whose new body fails keeps its old row: the failure never overwrites it,
    # and since the row is for the old sha256 the page stays due for parsing.
    old = _index()[("results", "G2")]
    site.write("results", "G2", {"rows": [{"k": "G2", "new": True}]})
    fake.broken["G2"] = RuntimeError("new layout")
    for _ in range(2):
        assert ("results", "G2") in _build(site.manifest)
        assert _index()[("results", "G2")] == old
    assert _cached_tables("results")["G2"] == expected_payload_tables({"rows": [{"k": "G2"}]})

    # put() itself drops every result without a payload, whatever its status.
    ok_payload = zlib.compress(b'{"rows":{"columns":["k"],"rows":[["P1"]]}}')
    result = parse_worker.ParseResult
    results = [
        result("card", "P1", "a" * 64, "2", parse_worker.STATUS_OK, False, 1, None, ok_payload),
        result("card", "P2", "b" * 64, "2", parse_worker.STATUS_FAILED, False, 0, "E: x", None),
        result("card", "P3", "c" * 64, "2", parse_worker.STATUS_UNREADABLE, False, 0, "gone", None),
    ]
    with cache.ParseCache(cache.cache_path(SOURCE)) as pc:
        pc.put(results)
        pc.put([])
        assert {k: v for k, v in pc.index().items() if k[0] == "card"} == {
            ("card", "P1"): ("a" * 64, "2", False)
        }
        assert [row[:3] for row in pc.payloads("card")] == [("P1", "a" * 64, "2")]


def _seed(pc, page_type, keys, sha="s", version="1"):
    rows = [
        parse_worker.ParseResult(
            page_type, k, f"{sha}{k}", version, parse_worker.STATUS_OK, False, 1, None,
            zlib.compress(f'{{"t":{{"columns":["k"],"rows":[["{k}"]]}}}}'.encode()),
        )
        for k in keys
    ]  # fmt: skip
    pc.put(rows)


def _user_version(path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_a_cache_format_bump_drops_the_cache(data, monkeypatch):
    path = cache.cache_path(SOURCE)
    base = cache.CACHE_FORMAT
    assert isinstance(base, int)

    with cache.ParseCache(path) as pc:
        _seed(pc, "results", ["A", "B", "C"])
    assert _user_version(path) == base

    # Same format: the rows stay.
    with cache.ParseCache(path) as pc:
        assert sorted(pc.index()) == [("results", k) for k in "ABC"]

    # Each new format starts empty, stamps the file, and keeps what it then caches.
    for bump in (base + 1, base + 2, base + 10, base):  # back to an old format counts as a change
        monkeypatch.setattr(cache, "CACHE_FORMAT", bump)
        with cache.ParseCache(path) as pc:
            assert pc.index() == {}
            assert list(pc.payloads("results")) == []
            assert list(pc.payloads("card")) == []
            fresh = [f"K{bump}-{i}" for i in range(bump % 3 + 1)]
            _seed(pc, "card", fresh, sha="t")
        assert _user_version(path) == bump
        with cache.ParseCache(path) as pc:
            assert sorted(pc.index()) == [("card", k) for k in sorted(fresh)]

    # A file stamped with a format this code does not know (or none) is dropped too.
    monkeypatch.setattr(cache, "CACHE_FORMAT", base)
    for stamp in (0, base + 99):
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {stamp}")
        conn.commit()
        conn.close()
        with cache.ParseCache(path) as pc:
            assert pc.index() == {}
            _seed(pc, "results", ["Q"])
        assert _user_version(path) == base


def _hktrack_payload(tables: dict) -> bytes:
    """HKTrack's encode_tables output for plain JSON rows: zlib (level 3) of compact JSON."""
    text = json.dumps(tables, separators=(",", ":"), ensure_ascii=False)
    return zlib.compress(text.encode("utf-8"), 3)


def test_opens_an_hktrack_parse_cache(data):
    path = cache.cache_path(SOURCE)
    path.parent.mkdir(parents=True)
    hk_rows = [
        ("results", "2025-03-02_ST_R2", "1" * 64, "4", 0, 2,
         {"runners": {"columns": ["no", "horse"], "rows": [[1, "A"], [2, "B"]]}}),
        ("results", "2025-03-02_ST_R1", "2" * 64, "4", 1, 1,
         {"runners": {"columns": ["no"], "rows": [[7]]},
          "_page": {"columns": ["key_mismatch"], "rows": [[True]]}}),
        ("meeting", "2025-03-02_ST", "3" * 64, "0", 0, 0,
         {"meetings": {"columns": [], "rows": []}}),
    ]  # fmt: skip

    # The file exactly as HKTrack's ParseCache writes it.
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(HKTRACK_SCHEMA)
        conn.execute(f"PRAGMA user_version = {HKTRACK_CACHE_FORMAT}")
        conn.executemany(HKTRACK_SQL_PUT, [(*row[:6], _hktrack_payload(row[6])) for row in hk_rows])
    table_info = conn.execute("PRAGMA table_info(parsed)").fetchall()
    conn.close()

    with cache.ParseCache(path) as pc:
        assert pc.index() == {
            (pt, key): (sha, version, bool(mismatch))
            for pt, key, sha, version, mismatch, _, _ in hk_rows
        }
        served = list(pc.payloads("results"))
        assert [row[:3] for row in served] == [
            ("2025-03-02_ST_R1", "2" * 64, "4"),
            ("2025-03-02_ST_R2", "1" * 64, "4"),
        ]  # key order
        assert parse_worker.decode_tables(served[0][3]) == hk_rows[1][6]
        assert parse_worker.decode_tables(served[1][3]) == hk_rows[0][6]
        [(key, _, _, payload)] = pc.payloads("meeting")
        assert (key, parse_worker.decode_tables(payload)) == ("2025-03-02_ST", hk_rows[2][6])

        # A row this code adds lands in the same table, in HKTrack's column order.
        new_payload, n_rows, mismatch = parse_worker.encode_tables({"runners": [{"no": 3}] * 3})
        pc.put(
            [
                parse_worker.ParseResult(
                    "results", "2025-03-09_HV_R1", "9" * 64, "5", parse_worker.STATUS_OK,
                    mismatch, n_rows, None, new_payload,
                )
            ]
        )  # fmt: skip

    # The file is still one HKTrack opens as its own: same format, same table, all rows.
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == HKTRACK_CACHE_FORMAT
        assert conn.execute("PRAGMA table_info(parsed)").fetchall() == table_info
        rows = conn.execute("SELECT * FROM parsed ORDER BY page_type, key").fetchall()
    finally:
        conn.close()
    assert [r[:6] for r in rows] == [
        ("meeting", "2025-03-02_ST", "3" * 64, "0", 0, 0),
        ("results", "2025-03-02_ST_R1", "2" * 64, "4", 1, 1),
        ("results", "2025-03-02_ST_R2", "1" * 64, "4", 0, 2),
        ("results", "2025-03-09_HV_R1", "9" * 64, "5", 0, 3),
    ]
    assert rows[3][6] == new_payload
