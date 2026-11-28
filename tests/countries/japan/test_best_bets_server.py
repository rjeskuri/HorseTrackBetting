"""Ticket #714: `/best-bets?venue=<VENUE>` on the board server.

Contract pinned here (`horsetrack.countries.japan.board.server.make_server`, on top of
test_board_server.py and board/test_server_access.py):

- `/best-bets?venue=<VENUE>` answers 200 with JSON: `best_bets.build_best_bets` of the day for that
  venue as of the server's clock (`feed.utc_now`), so its `race` is the venue's next race and moves
  to the following race at the post time (the delayed post time when the post-time sidecar says
  so). The rows are the private leaderboard JSON at `paths.reports_dir() /
  "leaderboard-japan.json"` (leaderboard_build.PRIVATE_JSON), read at start and read again when the
  file changes; the server's `models` give the nowcast.
- With a token it is gated like every other path: without the token an empty 404, with `?t=` (beside
  `venue=`) or the cookie it is served.

The day, rows and nowcast are synthetic (best_bets_fakes.py); the expected bets are worked out by
hand there. The clock is faked through `feed.datetime`, as test_board_server.py does.
"""

from __future__ import annotations

import gzip
import http.client
import json
import os
import secrets
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from horsetrack.core import paths
from horsetrack.core.eval.leaderboard import to_json
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules

# isort: split
import best_bets_fakes as fx

UTC = fx.UTC
WEB = Path(__file__).resolve().parents[3] / "src/horsetrack/countries/japan/board/web"
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
ACCESS = secrets.token_urlsafe(24)  # a fresh one per run, as board-serve makes


class Reply:
    def __init__(self, status: int, headers: http.client.HTTPMessage, body: bytes):
        self.status, self.headers, self.body = status, headers, body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def get(server, path: str, headers: dict[str, str] | None = None) -> Reply:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        if (resp.getheader("Content-Encoding") or "").lower() == "gzip":
            body = gzip.decompress(body)
        return Reply(resp.status, resp.headers, body)
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


class _Clock:
    def __init__(self, at: datetime):
        self.at = at

    def set(self, at: datetime) -> None:
        self.at = at


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    from horsetrack.countries.japan.board import feed as board_feed

    state = _Clock(fx.jst(9, 0))

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            at = state.at
            return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)

    monkeypatch.setattr(board_feed, "datetime", FakeDateTime)
    return state


@pytest.fixture
def serve() -> Iterator[Callable[..., Any]]:
    from horsetrack.countries.japan.board import server as board_server

    started = []

    def start(**kwargs):
        kwargs.setdefault("page_path", WEB / "board.html")
        kwargs.setdefault("port", 0)
        kwargs.setdefault("models", fx.make_models())
        kwargs.setdefault("flag_rules", RULES)
        kwargs.setdefault("min_rebuild_s", 0)
        srv = board_server.make_server(fx.DAY, **kwargs)
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


def leaderboard_path() -> Path:
    return paths.reports_dir() / "leaderboard-japan.json"


def write_leaderboard(rows) -> None:
    path = leaderboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(to_json(rows), encoding="utf-8")
    stamp = max(path.stat().st_mtime_ns, before + 5_000_000_000)
    os.utime(path, ns=(stamp, stamp))


def best(srv, venue: str, extra: str = "", headers=None) -> Reply:
    return get(srv, f"/best-bets?venue={venue}{extra}", headers)


def check(reply: Reply, venue, no, checkpoint, frozen, board, post=None) -> None:
    where = f"{venue} {no} T-{checkpoint}"
    assert reply.status == 200, (where, reply.status, reply.body[:200])
    assert reply.headers["Content-Type"].split(";")[0].strip() == "application/json", where
    got = reply.json()
    assert got["race"]["race_id"] == fx.rid(venue, no), where
    post = post or fx.POSTS[(venue, no)]
    assert datetime.fromisoformat(got["race"]["post_utc"]) == post, where
    assert got["checkpoint_min"] == checkpoint, where
    assert got["frozen"] is frozen, where
    fx.check_bets(got["best_bets"], board, checkpoint, where)
    assert {u["option_id"] for u in got["unmatched"]} == fx.unmatched_ids(checkpoint), where


def test_each_venue_gets_its_next_race_and_switches_at_the_post(serve, clock):
    fx.write_day()
    write_leaderboard(fx.all_rows())
    srv = serve()
    for venue, now, no, checkpoint, frozen, board in sorted(fx.TIMELINE, key=lambda c: c[1]):
        clock.set(now)
        check(best(srv, venue), venue, no, checkpoint, frozen, board)


def test_a_delayed_post_moves_the_switch(serve, clock):
    fx.write_day()
    write_leaderboard(fx.all_rows())
    srv = serve()
    clock.set(fx.jst(10, 19))
    check(best(srv, "KYO"), "KYO", 3, 2, True, "B")
    sidecar = paths.odds_snapshot_path("jra", fx.DAY.isoformat(), "KYO").with_name("KYO.races.json")
    posts = {3: fx.DELAYED_KYO3.astimezone(UTC), 4: fx.POSTS[("KYO", 4)].astimezone(UTC)}
    odds_log.write_post_times(sidecar, posts, fx.jst(10, 19).astimezone(UTC))
    for venue, now, no, checkpoint, frozen, board in fx.DELAYED_TIMELINE:
        clock.set(now)
        post = fx.DELAYED_KYO3 if (venue, no) == ("KYO", 3) else None
        check(best(srv, venue), venue, no, checkpoint, frozen, board, post)


def test_a_changed_leaderboard_file_is_read_again(serve, clock):
    fx.write_day()
    write_leaderboard(fx.all_rows())
    srv = serve()
    clock.set(fx.jst(9, 59))  # TOK 1, T-10, board A
    reply = best(srv, "TOK")
    check(reply, "TOK", 1, 10, False, "A")
    first = reply.json()["best_bets"]
    assert (first[0]["pool"], first[0]["combination"]) == ("WIN", "2")
    flipped = [
        replace(r, shrunk_per_100=200.0) if r.rule == fx.QUIN12 and r.shrunk_per_100 else r
        for r in fx.all_rows()
    ]
    write_leaderboard(flipped)
    again = best(srv, "TOK").json()["best_bets"]
    assert {(e["pool"], e["combination"]) for e in again[:2]} == {
        ("QUINELLA", "2,3"),
        ("QUINELLA", "2,4"),
    }
    assert again[0]["shrunk_per_100"] == pytest.approx(200.0)


def test_best_bets_is_gated_by_the_token(serve, clock):
    fx.write_day()
    write_leaderboard(fx.all_rows())
    srv = serve(token=ACCESS)
    clock.set(fx.jst(9, 54))
    denied = best(srv, "TOK")
    assert denied.status == 404 and denied.body == b""
    other = ACCESS[:-1] + ("A" if ACCESS[-1] != "A" else "B")
    wrong = best(srv, "TOK", "&t=" + other)
    assert wrong.status == 404 and wrong.body == b""
    granted = best(srv, "TOK", "&t=" + ACCESS)
    check(granted, "TOK", 1, 10, False, "A")
    cookie = granted.headers["Set-Cookie"].split(";")[0]
    check(best(srv, "KYO", headers={"Cookie": cookie}), "KYO", 3, 10, False, "A")
