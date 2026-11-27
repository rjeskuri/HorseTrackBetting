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
import math
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
from horsetrack.countries.japan.board.flags import FlagRules

# isort: split
from shadow_fakes import (
    IDS,
    WIN,
    Race,
    SyntheticDay,
    all_keys,
    expected_values,
    fake_models,
    instant,
)

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


# ---------------------------------------------------------------- the shadow page and feed (#121)
#
# `make_server(..., models=None, flag_rules=None)` also serves `/shadow` (board/web/shadow.html),
# `/shadow.js` (the script shadow.html loads beside it) and `/shadow-feed` (`build_shadow` of the
# day as JSON) with the same headers, ETag, 304 and rebuild throttle as /feed. The models are the
# fakes of shadow_fakes.py; the race day is in the past, so every race is after its post.

WEB = Path(__file__).resolve().parents[3] / "src/horsetrack/countries/japan/board/web"
SHADOW_RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
SHADOW_BASE = (1.25, 2.9, 4.6, 7.3, 11.8, 26.0, 64.0, 152.0)


def shadow_day() -> tuple[SyntheticDay, list[Race]]:
    races = [
        Race(DAY, "TOK", 1, jst(10, 5).astimezone(UTC), SHADOW_BASE),
        Race(DAY, "TOK", 2, jst(10, 40).astimezone(UTC), SHADOW_BASE[:6]),
        Race(DAY, "KYO", 3, jst(11, 15).astimezone(UTC), SHADOW_BASE),
    ]
    day = SyntheticDay(DAY, races)
    for r in races:
        day.capture(r, r.post - timedelta(minutes=6), 1.0)
        day.capture(r, r.post - timedelta(minutes=1), 1.04)
    return day, races


def test_shadow_path_serves_the_shadow_page(serve):
    srv = serve(page_path=WEB / "board.html", models=fake_models(), flag_rules=SHADOW_RULES)

    page = get(srv, "/shadow")
    script = get(srv, "/shadow.js")
    board = get(srv, "/")

    assert page.status == 200
    assert page.body == (WEB / "shadow.html").read_bytes()
    assert page.headers["Content-Type"].split(";")[0].strip() == "text/html"
    assert script.status == 200, "shadow.html loads shadow.js beside it"
    assert script.body == (WEB / "shadow.js").read_bytes()
    assert script.headers["Content-Type"].split(";")[0].strip() == "text/javascript"
    assert board.body == (WEB / "board.html").read_bytes()
    for reply in (page, script):
        for header, value in SECURITY_HEADERS.items():
            assert (reply.headers[header] or "").strip().lower() == value
    for path in ("/shadow/", "/shadow-feed/", "/shadow.json", "/shadows", "/shadow-feeds"):
        assert get(srv, path).status == 404, path


def test_shadow_feed_has_an_etag_and_304(serve):
    day, races = shadow_day()
    srv = serve(models=fake_models(), flag_rules=SHADOW_RULES, min_rebuild_s=3600)
    live = serve(models=fake_models(), flag_rules=SHADOW_RULES, min_rebuild_s=0)

    first = get(srv, "/shadow-feed")

    assert first.status == 200
    assert first.headers["Content-Type"].split(";")[0].strip() == "application/json"
    etag = first.headers["ETag"]
    assert etag and etag.strip('"W/'), f"no usable ETag: {etag!r}"
    feed = first.json()
    assert feed["race_date"] == DAY.isoformat()
    assert feed["models"] == IDS
    got = {r["race_id"]: r for r in feed["races"]}
    assert set(got) == {r.race_id for r in races}
    for r in races:
        assert got[r.race_id]["tau_bucket_min"] == 0  # the day is past: every race is after post
        want = expected_values(r, 1.04, 0)
        assert set(got[r.race_id]["pools"][WIN]) == set(want[WIN])
        for combination, values in want[WIN].items():
            entry = got[r.race_id]["pools"][WIN][combination]
            assert math.isclose(entry["value_mean"], values["value_mean"], rel_tol=1e-9)
            assert "stale" in entry["flags"]

    again = get(srv, "/shadow-feed", {"If-None-Match": etag})
    assert again.status == 304
    assert again.body == b""
    assert again.headers["ETag"] == etag
    other = get(srv, "/shadow-feed", {"If-None-Match": '"not-the-current-tag"'})
    assert other.status == 200
    assert other.headers["ETag"] == etag
    zipped = get(srv, "/shadow-feed", {"Accept-Encoding": "gzip"})
    assert zipped.status == 200
    assert zipped.json()["races"] == feed["races"]
    for name, reply in {"200": first, "304": again}.items():
        for header, value in SECURITY_HEADERS.items():
            assert (reply.headers[header] or "").strip().lower() == value, (name, header)
        robots = re.split(r"[,\s]+", (reply.headers["X-Robots-Tag"] or "").strip().lower())
        assert {"noindex", "nofollow"} <= set(robots), name

    # the rebuild throttle of /feed: a new capture is seen at min_rebuild_s 0, not within 3600 s
    live_tag = get(live, "/shadow-feed").headers["ETag"]
    late = races[0].post + timedelta(minutes=3)
    day.capture(races[0], late, 1.2)

    fresh = get(live, "/shadow-feed", {"If-None-Match": live_tag})
    assert fresh.status == 200
    assert fresh.headers["ETag"] != live_tag
    race = next(r for r in fresh.json()["races"] if r["race_id"] == races[0].race_id)
    assert instant(race["board_capture_utc"]) == late
    want = expected_values(races[0], 1.2, 0)[WIN]["1"]["value_mean"]
    assert math.isclose(race["pools"][WIN]["1"]["value_mean"], want, rel_tol=1e-9)

    held = get(srv, "/shadow-feed", {"If-None-Match": etag})
    assert held.status == 304
    assert held.headers["ETag"] == etag


def test_the_plain_feed_has_no_value_fields(serve):
    from horsetrack.countries.japan.board.feed import build_board

    day, races = shadow_day()
    srv = serve(models=fake_models(), flag_rules=SHADOW_RULES)

    reply = get(srv, "/feed")
    shadow = get(srv, "/shadow-feed")

    assert reply.status == 200 and shadow.status == 200
    plain = reply.json()
    named = sorted(
        {
            key
            for key in all_keys(plain)
            for word in ("prob", "value", "bet", "stake", "kelly", "wager", "flag", "tau")
            if word in key.lower()
        }
    )
    assert named == [], named
    board = build_board(DAY)
    plain.pop("generated_utc")
    board.pop("generated_utc")
    assert plain == json.loads(json.dumps(board))  # /feed is #40's board, unchanged
    assert {r.race_id for r in races} == {
        race["race_id"] for venue in plain["venues"] for race in venue["races"]
    }


# ---------------------------------------------------------------- the clock's cutoffs (#498)
#
# The T-20 and T-2 references follow the clock as well as the logs: a reference is null until
# post - 20 / post - 2 min has passed, then it is the last priced capture at or before that
# instant. A feed built just before a cutoff must not be served after it until a log changes.
# The clock seam assumed: the `datetime` of the feed module, which `feed.utc_now` (and so
# `build_board`'s default `now`) reads; the server takes its now from `feed.utc_now` too.


class _Clock:
    def __init__(self, at: datetime):
        self.at = at

    def set(self, at: datetime) -> None:
        self.at = at


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    from horsetrack.countries.japan.board import feed as board_feed

    state = _Clock(jst(9, 0))

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            at = state.at
            return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)

    monkeypatch.setattr(board_feed, "datetime", FakeDateTime)
    return state


# venue, race, post (JST), WIN odds of runners 1 and 2 at post-45m, post-20m-10s, post-2m-10s
CUTOFF_RACES = {
    ("TOK", 1): (jst(9, 50), ((6.0, 3.0), (5.0, 3.2), (4.0, 3.6))),
    ("KYO", 3): (jst(10, 12), ((8.0, 2.5), (7.0, 2.6), (5.6, 2.9))),
}


def log_cutoff_day() -> None:
    for venue in sorted({v for v, _ in CUTOFF_RACES}):
        posts = {no: post for (v, no), (post, _) in CUTOFF_RACES.items() if v == venue}
        sidecar = log_path(venue).with_name(f"{venue}.races.json")
        odds_log.write_post_times(
            sidecar, {no: p.astimezone(UTC) for no, p in posts.items()}, jst(6, 0)
        )
    offsets = (
        timedelta(minutes=45),
        timedelta(minutes=20, seconds=10),
        timedelta(minutes=2, seconds=10),
    )
    captures = []
    for (venue, no), (post, odds) in CUTOFF_RACES.items():
        for offset, (one, two) in zip(offsets, odds, strict=True):
            captures.append((post - offset, venue, {(no, "1"): one, (no, "2"): two}))
    for at, venue, win in sorted(captures, key=lambda c: c[0]):
        capture(venue, at, win)


def win_line(board: dict, venue: str, no: int, combination: str) -> dict:
    v = next(v for v in board["venues"] if v["venue"] == venue)
    race = next(r for r in v["races"] if r["race_no"] == no)
    return next(r for r in race["pools"]["WIN"]["rows"] if r["combination"] == combination)


@pytest.mark.parametrize("cutoff_min", [20, 2], ids=["T-20", "T-2"])
@pytest.mark.parametrize("race", sorted(CUTOFF_RACES), ids=lambda r: f"{r[0]}{r[1]}")
def test_a_poll_after_a_cutoff_fills_its_reference_without_a_log_write(
    serve, clock, race, cutoff_min
):
    venue, no = race
    post, (_, at_t20, at_t2) = CUTOFF_RACES[race]
    cutoff = post - timedelta(minutes=cutoff_min)
    log_cutoff_day()
    clock.set(cutoff - timedelta(seconds=5))
    srv = serve(min_rebuild_s=0)

    before = get(srv, "/feed")
    assert before.status == 200
    etag = before.headers["ETag"]
    for i, combination in enumerate(("1", "2")):
        line = win_line(before.json(), venue, no, combination)
        if cutoff_min == 20:
            assert line["t20_odds"] is None, (combination, line)
        else:
            assert line["t20_odds"] == at_t20[i], (combination, line)
        assert line["t2_odds"] is None, (combination, line)
        assert line["move_t20_t2_pct"] is None, (combination, line)

    clock.set(cutoff - timedelta(seconds=2))  # no cutoff passed, no log changed
    held = get(srv, "/feed", {"If-None-Match": etag})
    assert held.status == 304, "nothing changed: the poll stays a 304"

    clock.set(cutoff + timedelta(seconds=5))  # the cutoff passed; still no log write
    after = get(srv, "/feed", {"If-None-Match": etag})

    assert after.status == 200, f"the T-{cutoff_min} column stays empty after its cutoff"
    assert after.headers["ETag"] != etag
    for i, combination in enumerate(("1", "2")):
        line = win_line(after.json(), venue, no, combination)
        assert line["t20_odds"] == at_t20[i], (combination, line)
        if cutoff_min == 2:
            assert line["t2_odds"] == at_t2[i], (combination, line)
            want = round((at_t2[i] / at_t20[i] - 1) * 100, 1)
            assert line["move_t20_t2_pct"] == pytest.approx(want, abs=0.05), (combination, line)
        else:
            assert line["t2_odds"] is None, (combination, line)


@pytest.mark.parametrize("cutoff_min", [10, 5], ids=["T-10", "T-5"])
@pytest.mark.parametrize("race", sorted(CUTOFF_RACES), ids=lambda r: f"{r[0]}{r[1]}")
def test_a_poll_after_t10_or_t5_fills_its_reference_without_a_log_write(
    serve, clock, race, cutoff_min
):
    """#499: the T-10 and T-5 references fill at their cutoffs as T-20 and T-2 do, without a log
    write (the latest capture at or before both is the post - 20m10s one)."""
    venue, no = race
    post, (_, at_t20, _) = CUTOFF_RACES[race]
    cutoff = post - timedelta(minutes=cutoff_min)
    field = f"t{cutoff_min}_odds"
    log_cutoff_day()
    clock.set(cutoff - timedelta(seconds=5))
    srv = serve(min_rebuild_s=0)

    before = get(srv, "/feed")
    assert before.status == 200
    etag = before.headers["ETag"]
    for combination in ("1", "2"):
        line = win_line(before.json(), venue, no, combination)
        assert line[field] is None, (combination, line)

    clock.set(cutoff + timedelta(seconds=5))  # the cutoff passed; still no log write
    after = get(srv, "/feed", {"If-None-Match": etag})

    assert after.status == 200, f"the T-{cutoff_min} column stays empty after its cutoff"
    assert after.headers["ETag"] != etag
    for i, combination in enumerate(("1", "2")):
        line = win_line(after.json(), venue, no, combination)
        assert line[field] == at_t20[i], (combination, line)
