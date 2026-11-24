"""The parse worker (#47): one raw page in, one compact payload out, through `get_adapter(source)`.

Ported from HKTrack src/hktrack/store/parse_worker.py. Seams pinned here:

* `horsetrack.core.store.parse_worker` exposes ParseJob, ParseResult, parse_job, parser_version,
  encode_tables, decode_tables and STATUS_OK / STATUS_FAILED / STATUS_UNREADABLE.
* `ParseJob` fields, in order: (source, page_type, key, year, url, final_url, fetched_utc, sha256,
  parser_version); `parse_job` also accepts the plain tuple (jobs cross process boundaries).
* `ParseResult` fields: page_type, key, sha256, parser_version, status, key_mismatch, n_rows,
  error, payload.
* The page is read from `paths.raw_page_path(source, page_type, year, key)` (gzip) and must hash to
  the job's sha256. Payload = zlib(JSON) of {table: {"columns": [...], "rows": [[...]]}};
  datetimes become UTC ISO strings. Tables named "_..." are metadata: not counted in n_rows, and
  a truthy `key_mismatch` in them sets the result's key_mismatch.
* `parse_job` never raises: a parser error is STATUS_FAILED with a one-line error (at most 300
  characters) naming the exception type; a missing, corrupt or wrong-sha page is
  STATUS_UNREADABLE and never reaches the adapter. Neither carries a payload.
"""

from __future__ import annotations

import gzip
import json
import zlib
from datetime import timedelta

import pytest
from parse_fakes import OTHER, SOURCE, YEAR, FakeParseAdapter, body, register, write_page

from horsetrack.core import paths

# isort: split
# A block of its own: while the module is missing, it is the import error a red run reports.
from horsetrack.core.store import parse_worker

VAR = "HORSETRACK_DATA_DIR"


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def _job(source, page_type, key, sha, version="1", *, fetched=None, url=None, year=YEAR):
    url = url if url is not None else f"https://site.test/{source}/{page_type}/{key}"
    return parse_worker.ParseJob(source, page_type, key, year, url, None, fetched, sha, version)


def _decode_independently(payload: bytes):
    return json.loads(zlib.decompress(payload).decode("utf-8"))


# Per source: key -> (page_type, tables, key_mismatch, fetched_utc as the manifest stores it)
PAGES = {
    SOURCE: {
        "20260301A01": (
            "results",
            {"runners": [{"horse": "H1", "pos": 1}, {"horse": "H2", "pos": 2, "time": 71.4}]},
            False,
            "2026-03-01T09:30:00",
        ),
        "20260301A02": ("results", {"runners": []}, False, "2026-03-01T18:30:00+09:00"),
        "20260302B01": (
            "card",
            {
                "races": [{"race": 1, "dist": 1200}],
                "entries": [{"no": 1}, {"no": 2, "jockey": "J"}, {"jockey": "K", "no": 3}],
            },
            True,
            None,
        ),
    },
    OTHER: {
        "M-77": (
            "results",
            {"runners": [{"horse": "Z9", "pos": 4}]},
            False,
            "2026-04-05T10:02:03+09:00",
        ),
        "M-78": ("results", {"dividends": [{"pool": "WIN", "pays": 340}] * 2}, True, None),
    },
}

EXPECTED_UTC = {
    "2026-03-01T09:30:00": "2026-03-01T09:30:00+00:00",
    "2026-03-01T18:30:00+09:00": "2026-03-01T09:30:00+00:00",
    "2026-04-05T10:02:03+09:00": "2026-04-05T01:02:03+00:00",
    None: None,
}


def test_worker_parses_with_the_registered_adapter(data, monkeypatch):
    fakes = {
        SOURCE: register(monkeypatch, FakeParseAdapter(SOURCE, {"results": 3, "card": "2b"})),
        OTHER: register(monkeypatch, FakeParseAdapter(OTHER, {"results": 11, "card": None})),
    }

    # parser_version comes from the adapter's hook, as a string; None means "no parser".
    assert parse_worker.parser_version(SOURCE, "results") == "3"
    assert parse_worker.parser_version(SOURCE, "card") == "2b"
    assert parse_worker.parser_version(OTHER, "results") == "11"
    assert parse_worker.parser_version(OTHER, "card") is None

    for source, pages in PAGES.items():
        for key, (page_type, tables, mismatch, fetched) in pages.items():
            raw = body(tables, mismatch)
            sha = write_page(source, page_type, key, raw)
            version = parse_worker.parser_version(source, page_type)
            url = f"https://{source}.test/{key}"
            job = _job(source, page_type, key, sha, version, fetched=fetched, url=url)
            before = {s: len(f.pages) for s, f in fakes.items()}

            # A plain tuple is as good as a ParseJob (spawned workers get plain tuples).
            result = parse_worker.parse_job(tuple(job) if key.endswith("2") else job)

            # The registered adapter for this source parsed it, and no other adapter did.
            assert len(fakes[source].pages) == before[source] + 1
            for other, fake in fakes.items():
                if other != source:
                    assert len(fake.pages) == before[other]
            page = fakes[source].pages[-1]
            assert (page.source, page.page_type, page.key, page.url) == (
                source,
                page_type,
                key,
                url,
            )
            assert page.body == raw

            assert result.status == parse_worker.STATUS_OK
            assert (result.page_type, result.key) == (page_type, key)
            assert (result.sha256, result.parser_version) == (sha, version)
            assert result.error is None
            assert result.key_mismatch is mismatch
            assert result.n_rows == sum(len(rows) for rows in tables.values())

            decoded = _decode_independently(result.payload)
            assert parse_worker.decode_tables(result.payload) == decoded
            assert set(decoded) == set(tables) | {"_page"}
            for name, rows in tables.items():
                columns = list(dict.fromkeys(c for row in rows for c in row))
                assert decoded[name] == {
                    "columns": columns,
                    "rows": [[row.get(c) for c in columns] for row in rows],
                }
            meta = dict(zip(decoded["_page"]["columns"], decoded["_page"]["rows"][0], strict=True))
            assert meta["key"] == key
            assert meta["n_bytes"] == len(raw)
            assert meta["key_mismatch"] is mismatch
            # The manifest's fetch time reaches the adapter as a UTC datetime, and the payload
            # stores it as a UTC ISO string.
            assert meta["fetched_utc"] == EXPECTED_UTC[fetched]
            if fetched is not None:
                assert page.fetched_utc.utcoffset() == timedelta(0)


FAILURES = {
    "K-value": ValueError("bad row in table"),
    "K-key": KeyError("odds"),
    "K-zero": ZeroDivisionError("division by zero"),
    "K-lines": RuntimeError("line one\n\n  line two\tthree"),
    "K-long": ValueError("long " + "x" * 2000),
}


def test_worker_reports_a_failure_without_raising(data, monkeypatch):
    fake = register(monkeypatch, FakeParseAdapter(SOURCE, {"results": 5}))
    fake.broken.update(FAILURES)

    results = {}
    for key, exc in FAILURES.items():
        sha = write_page(SOURCE, "results", key, body({"runners": [{"horse": key}]}))
        result = results[key] = parse_worker.parse_job(_job(SOURCE, "results", key, sha, "5"))
        assert result.status == parse_worker.STATUS_FAILED, key
        assert (result.page_type, result.key, result.sha256, result.parser_version) == (
            "results", key, sha, "5",
        )  # fmt: skip
        assert result.payload is None
        assert (result.n_rows, result.key_mismatch) == (0, False)
        assert isinstance(result.error, str) and type(exc).__name__ in result.error
        assert "\n" not in result.error and "\t" not in result.error
        assert 0 < len(result.error) <= 300
    assert "line one" in results["K-lines"].error
    assert "line two three" in results["K-lines"].error  # whitespace runs collapse to one space

    # A good page between the failures still parses.
    good_sha = write_page(SOURCE, "results", "K-good", body({"runners": [{"horse": "ok"}]}))
    good = parse_worker.parse_job(_job(SOURCE, "results", "K-good", good_sha, "5"))
    assert (good.status, good.n_rows) == (parse_worker.STATUS_OK, 1)

    # Unreadable pages never reach the adapter.
    calls = len(fake.pages)
    real = body({"runners": [{"horse": "R"}]})
    sha_real = write_page(SOURCE, "results", "U-wrongsha", real)
    write_page(SOURCE, "results", "U-zero", b"")
    zero = paths.raw_page_path(SOURCE, "results", YEAR, "U-zero")
    zero.write_bytes(b"")  # a zero-byte file, not an empty gzip stream
    notgz = paths.raw_page_path(SOURCE, "results", YEAR, "U-notgz")
    notgz.write_bytes(real)  # plain bytes where gzip is expected
    truncated = paths.raw_page_path(SOURCE, "results", YEAR, "U-trunc")
    truncated.write_bytes(gzip.compress(real)[:-12])
    unreadable = {
        "U-missing": "0" * 64,
        "U-wrongsha": sha_real[::-1],  # the file is there but is not the body the manifest recorded
        "U-zero": "",
        "U-notgz": sha_real,
        "U-trunc": sha_real,
    }
    for key, sha in unreadable.items():
        result = parse_worker.parse_job(_job(SOURCE, "results", key, sha, "5"))
        assert result.status == parse_worker.STATUS_UNREADABLE, key
        assert result.payload is None and result.n_rows == 0
        assert isinstance(result.error, str) and result.error
    assert len(fake.pages) == calls

    # A source with no registered adapter is a failed parse, not an exception.
    sha = write_page("nosuchsrc", "results", "N1", real)
    result = parse_worker.parse_job(_job("nosuchsrc", "results", "N1", sha, "1"))
    assert result.status == parse_worker.STATUS_FAILED
    assert result.payload is None and "ValueError" in result.error

    assert (
        len({parse_worker.STATUS_OK, parse_worker.STATUS_FAILED, parse_worker.STATUS_UNREADABLE})
        == 3
    )
