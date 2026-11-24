"""The board page and its live feed served on localhost (ticket #41).

Contract pinned here (`horsetrack.countries.japan.board.server.make_server`):
- `make_server(race_date, *, host='127.0.0.1', port=8765, page_path, min_rebuild_s=5)` returns a
  `ThreadingHTTPServer`, bound but not yet serving; the tests bind port 0, run `serve_forever` in a
  thread and shut it down in teardown.
- Two paths: `/` serves the file at `page_path`; `/feed` serves `build_board(race_date)` as JSON
  with an ETag, and a request whose If-None-Match names the current ETag gets 304 with no body.
  Any other path is 404: no listings, no other files.
- Every response carries Cache-Control no-store, X-Robots-Tag noindex nofollow, Referrer-Policy
  no-referrer and X-Content-Type-Options nosniff.
- The feed is rebuilt at most every `min_rebuild_s`, and only when a log changed; a torn last
  gzip member (a capture being appended) is ignored.
- `horsetrack japan board-serve --date D [--port N] [--host H]` is registered.

All logs are synthetic and written under tmp_path (HORSETRACK_DATA_DIR); no external network.
The race day is in the past so every capture is at or before the real clock's `now`.
"""

from __future__ import annotations

import gzip
import http.client
import json
import re
import threading
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2025, 11, 29)

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


# ---------------------------------------------------------------- helpers


def jst(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hh, mm, ss, tzinfo=JST)


def rid(venue: str, no: int) -> str:
    return f"jra:{DAY.isoformat()}:{venue}:{no:02d}"


def price(race_id: str, pool: str, combination: str, odds: float, at: datetime) -> dict:
    return {
        "snapshot_utc": at.astimezone(UTC).isoformat(),
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": f"{at.hour:02d}時{at.minute:02d}分現在",
        "source_update_utc": at.astimezone(UTC).isoformat(),
    }


def log_path(venue: str) -> Path:
    return paths.odds_snapshot_path("jra", DAY.isoformat(), venue)


def capture(venue: str, at: datetime, win: dict[tuple[int, str], float]) -> None:
    """Append one capture (one gzip member) of WIN odds {(race_no, combination): odds}."""
    rows = [price(rid(venue, no), "WIN", combo, odds, at) for (no, combo), odds in win.items()]
    odds_log.append_capture(log_path(venue), rows)


class Reply:
    def __init__(self, status: int, headers: http.client.HTTPMessage, body: bytes):
        self.status, self.headers, self.body = status, headers, body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def get(server, path: str, headers: dict[str, str] | None = None) -> Reply:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        if (resp.getheader("Content-Encoding") or "").lower() == "gzip":
            body = gzip.decompress(body)
        return Reply(resp.status, resp.headers, body)
    finally:
        conn.close()


def win_odds(board: dict) -> dict[tuple[str, int, str], float]:
    """{(venue, race_no, combination): odds} of every WIN line on the board."""
    out = {}
    for venue in board["venues"]:
        for race in venue["races"]:
            for row in race["pools"].get("WIN", {}).get("rows", []):
                out[(venue["venue"], race["race_no"], row["combination"])] = row["odds"]
    return out


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


@pytest.fixture
def page(tmp_path) -> Path:
    p = tmp_path / "site" / "board.html"
    p.parent.mkdir()
    p.write_text("<!doctype html><title>board</title><p>page-token-7f3a 馬</p>\n", encoding="utf-8")
    return p


@pytest.fixture
def serve(page) -> Iterator[Callable[..., Any]]:
    """Start a server on port 0 in a thread; every server started is shut down in teardown."""
    from horsetrack.countries.japan.board import server as board_server

    started = []

    def start(**kwargs):
        kwargs.setdefault("page_path", page)
        kwargs.setdefault("port", 0)
        srv = board_server.make_server(DAY, **kwargs)
        thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05})
        thread.daemon = True
        thread.start()
        started.append((srv, thread))
        return srv

    yield start
    for srv, thread in started:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------- the page


@pytest.mark.parametrize(
    "html",
    [
        "<!doctype html><title>a</title><main id='board'>alpha-91c2</main>\n",
        "<!doctype html><title>b</title><script>poll('/feed')</script>beta-04d8 東京\n",
    ],
)
def test_root_serves_the_page(serve, tmp_path, html):
    page_path = tmp_path / "elsewhere" / "my-page.html"
    page_path.parent.mkdir()
    page_path.write_text(html, encoding="utf-8")
    srv = serve(page_path=page_path)

    reply = get(srv, "/")

    assert reply.status == 200
    assert reply.body == page_path.read_bytes()
    assert reply.headers["Content-Type"].split(";")[0].strip() == "text/html"


# ---------------------------------------------------------------- the feed


def test_feed_is_json_with_an_etag(serve):
    written = {
        ("TOK", 1, "1"): 3.4,
        ("TOK", 1, "2"): 7.9,
        ("TOK", 11, "10"): 15.5,
        ("KYO", 3, "4"): 2.1,
    }
    for venue in ("TOK", "KYO"):
        win = {(no, c): odds for (v, no, c), odds in written.items() if v == venue}
        capture(venue, jst(9, 30), win)
    srv = serve()

    reply = get(srv, "/feed")

    assert reply.status == 200
    assert reply.headers["Content-Type"].split(";")[0].strip() == "application/json"
    etag = reply.headers["ETag"]
    assert etag and etag.strip('"W/'), f"no usable ETag: {etag!r}"
    board = reply.json()
    assert board["race_date"] == DAY.isoformat()
    assert [v["venue"] for v in board["venues"]] == ["KYO", "TOK"]
    assert win_odds(board) == written
    assert [r["race_id"] for r in board["venues"][1]["races"]] == [rid("TOK", 1), rid("TOK", 11)]


def test_feed_of_a_day_without_logs_is_an_empty_board(serve):
    reply = get(serve(), "/feed")

    assert reply.status == 200
    assert reply.headers["ETag"]
    board = reply.json()
    assert board["race_date"] == DAY.isoformat()
    assert board["venues"] == []


@pytest.mark.parametrize("min_rebuild_s", [5, 0])
def test_same_etag_is_304(serve, min_rebuild_s):
    capture("NAK", jst(10, 0), {(1, "1"): 4.2, (1, "2"): 6.0})
    srv = serve(min_rebuild_s=min_rebuild_s)
    first = get(srv, "/feed")
    etag = first.headers["ETag"]

    again = get(srv, "/feed", {"If-None-Match": etag})

    assert again.status == 304
    assert again.body == b""
    assert again.headers["ETag"] == etag

    other = get(srv, "/feed", {"If-None-Match": '"not-the-current-tag"'})
    assert other.status == 200
    assert other.headers["ETag"] == etag
    assert win_odds(other.json()) == win_odds(first.json())


@pytest.mark.parametrize("venue", ["NAK", "HAN"], ids=["same-venue", "venue-logged-after-start"])
def test_a_new_capture_changes_the_feed_and_its_etag(serve, venue):
    capture("NAK", jst(9, 0), {(2, "1"): 5.0, (2, "3"): 9.0})
    srv = serve(min_rebuild_s=0)
    reply = get(srv, "/feed")
    seen_tags = [reply.headers["ETag"]]
    expected = {("NAK", 2, "1"): 5.0, ("NAK", 2, "3"): 9.0}
    assert win_odds(reply.json()) == expected

    for minute, odds in ((5, 4.4), (10, 3.7), (15, 6.1)):
        capture(venue, jst(9, minute), {(2, "1"): odds})
        expected[(venue, 2, "1")] = odds

        reply = get(srv, "/feed", {"If-None-Match": seen_tags[-1]})

        assert reply.status == 200, f"capture at 09:{minute:02d} not seen: {reply.status}"
        assert win_odds(reply.json()) == expected
        assert reply.headers["ETag"] not in seen_tags
        seen_tags.append(reply.headers["ETag"])


def test_the_feed_is_not_rebuilt_within_min_rebuild_s(serve):
    capture("NAK", jst(9, 0), {(1, "1"): 5.0})
    srv = serve(min_rebuild_s=3600)
    first = get(srv, "/feed")

    capture("NAK", jst(9, 5), {(1, "1"): 2.5})
    again = get(srv, "/feed", {"If-None-Match": first.headers["ETag"]})

    assert again.status == 304
    assert again.headers["ETag"] == first.headers["ETag"]


def test_a_torn_last_member_is_ignored(serve):
    capture("KYO", jst(9, 0), {(5, "1"): 8.0, (5, "2"): 3.3})
    srv = serve(min_rebuild_s=0)
    before = get(srv, "/feed")

    torn = json.dumps(price(rid("KYO", 5), "WIN", "1", 1.9, jst(9, 5))) + "\n"
    member = gzip.compress(torn.encode("utf-8"))
    with open(log_path("KYO"), "ab") as fh:  # the logger caught mid-write
        fh.write(member[: len(member) // 2])

    reply = get(srv, "/feed")

    assert reply.status == 200
    assert win_odds(reply.json()) == {("KYO", 5, "1"): 8.0, ("KYO", 5, "2"): 3.3}
    assert win_odds(before.json()) == win_odds(reply.json())


# ---------------------------------------------------------------- nothing else


def test_any_other_path_is_404(serve, page):
    capture("TOK", jst(9, 0), {(1, "1"): 3.0})
    srv = serve()
    log_rel = log_path("TOK").relative_to(Path(paths.data_dir()))
    others = [
        "/index.html",
        "/board.html",
        f"/{page.name}",
        "/feed/",
        "/feeds",
        "/feed.json",
        "/favicon.ico",
        "/.env",
        "/odds_snapshots/",
        f"/{log_rel.as_posix()}",
        "/../board.html",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/site/",
    ]
    page_text = page.read_bytes()
    for path in others:
        reply = get(srv, path)
        assert reply.status == 404, f"{path} gave {reply.status}"
        assert page_text not in reply.body, f"{path} leaked the page"
        assert b"race_id" not in reply.body, f"{path} leaked feed data"
        assert not re.search(rb"<a\s+href", reply.body, re.I), f"{path} looks like a listing"


def test_default_bind_is_localhost(page):
    import inspect

    from horsetrack.countries.japan.board import server as board_server

    params = inspect.signature(board_server.make_server).parameters
    assert params["host"].default == "127.0.0.1"
    assert params["port"].default == 8765

    srv = board_server.make_server(DAY, port=0, page_path=page)
    try:
        assert srv.server_address[0] == "127.0.0.1"
        assert srv.server_address[1] > 0
    finally:
        srv.server_close()


def test_security_headers_are_sent(serve):
    capture("TOK", jst(9, 0), {(1, "1"): 3.0})
    srv = serve()
    feed = get(srv, "/feed")
    replies = {
        "/": get(srv, "/"),
        "/feed": feed,
        "/feed 304": get(srv, "/feed", {"If-None-Match": feed.headers["ETag"]}),
        "/nope 404": get(srv, "/nope"),
    }
    assert [r.status for r in replies.values()] == [200, 200, 304, 404]
    for name, reply in replies.items():
        for header, value in SECURITY_HEADERS.items():
            assert (reply.headers[header] or "").strip().lower() == value, (
                f"{name}: {header} is {reply.headers[header]!r}"
            )
        robots = re.split(r"[,\s]+", (reply.headers["X-Robots-Tag"] or "").strip().lower())
        assert {"noindex", "nofollow"} <= set(robots), f"{name}: X-Robots-Tag {robots}"


# ---------------------------------------------------------------- the command


def test_board_serve_command_is_registered():
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "board-serve", "--help"])

    assert result.exit_code == 0, result.output
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    for option in ("--date", "--port", "--host"):
        assert option in text
