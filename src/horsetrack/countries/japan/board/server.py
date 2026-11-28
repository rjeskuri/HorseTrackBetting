"""Serve the board page and its live feed on localhost (ticket #41), and the shadow page and its
feed beside them (ticket #121).

Seven paths only: `/` (the page at `page_path`, read on every request), `/board.js` (the page's
script, `board.js` beside `page_path`, ticket #42), `/feed` (`build_board` of the day as JSON,
docs/specs/board-feed.md) with an ETag, so the page's poll costs a 304 when nothing changed;
`/shadow` and `/shadow.js` (`shadow.html` and `shadow.js` beside `page_path`, ticket #122),
`/shadow-feed` (`build_shadow` of the day, with the same ETag, gzip and throttle as /feed) and
`/best-bets?venue=<VENUE>` (ticket #714: `best_bets.build_best_bets` of that venue's next race as
of `feed.utc_now`, from the private leaderboard JSON `leaderboard-japan.json` under the reports
dir, read at start and again when it changes; rebuilt at most every `min_rebuild_s` per venue).
Anything else is 404: no listings, no other files. Every response carries
Cache-Control no-store, X-Robots-Tag noindex nofollow, Referrer-Policy no-referrer,
X-Content-Type-Options nosniff and X-Frame-Options DENY.

The feed is rebuilt at most every `min_rebuild_s`, and only when a file of the day folder (an odds
log or a post-time sidecar) appeared, went away, or changed size or mtime, or when a race's
post - 20, 10, 5 or 2 min passed (its T-20 / T-10 / T-5 / T-2 reference fills then, #498, #499;
the clock is
`feed.utc_now`). A log being appended
reads safely: `odds_log` counts whole gzip members only, so a torn last member is ignored. A build
that fails keeps the last good feed. The shadow feed is rebuilt at most every `min_rebuild_s` too,
whether or not a file changed (its buckets and stale flags follow the clock), and its body and
ETag change only when something other than `generated_utc` and `minutes_to_post` did. The default
bind is 127.0.0.1; the board has no login.

With `--tunnel` (ticket #505) the server is exposed to a phone through a cloudflared quick tunnel
(`tunnel.quick_tunnel`), whose URL is public, so every path needs an access token: a fresh one per
run, held in memory until the server is bound and the tunnel up, then kept in
$HORSETRACK_DATA_DIR/board/access-token (`write_access_token`, mode 0600; a start that fails leaves
the file as it was), given as `?t=<token>` on first load, after which an HttpOnly, SameSite=Strict
cookie (Secure behind https) carries it on the page's fetches. Missing or wrong, it is the empty
404 of an unknown path; the check holds for 127.0.0.1 too, since the tunnel connects from there.
The phone URL is printed once and the token appears nowhere else. Without `--tunnel` nothing
changes. docs/specs/board-feed.md.

    horsetrack japan board-serve --date 2026-11-29 [--port 8765] [--host 127.0.0.1] [--page P]
        [--from-vm ALIAS] [--tunnel]
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import signal
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from datetime import date, datetime
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.board import feed as board_feed
from horsetrack.countries.japan.board.best_bets import LeaderboardFile, build_best_bets
from horsetrack.countries.japan.board.feed import build_board
from horsetrack.countries.japan.board.flags import FlagRules, load_flag_rules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.board.snapshot import jra_day_dir
from horsetrack.countries.japan.board.tunnel import TunnelError, quick_tunnel
from horsetrack.countries.japan.board.tunnel import parse_tunnel_url as parse_tunnel_url
from horsetrack.countries.japan.logger.pull_vm import vm_day_dir
from horsetrack.countries.japan.models.v0.loader import load_shadow_models
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.research.leaderboard_build import PRIVATE_JSON

log = logging.getLogger(__name__)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PAGE = Path(__file__).with_name("web") / "board.html"  # written by ticket #42
HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
COOKIE_NAME = "board_access"
_TOKEN_CHARS = re.compile(r"[A-Za-z0-9_-]+")  # secrets.token_urlsafe's alphabet
_QUERY = re.compile(r"\?[^\s\"']*")

Fingerprint = tuple[tuple[str, int, int], ...]


def _next_cutoff(board: dict[str, Any], now: datetime) -> datetime | None:
    """The first post - 20, 10, 5 or 2 min of a race on `board` still after `now`."""
    cutoffs = [
        datetime.fromisoformat(race["post_utc"]) - offset
        for venue in board["venues"]
        for race in venue["races"]
        if race.get("post_utc")
        for offset in board_feed.CHECKPOINTS
    ]
    return min((at for at in cutoffs if at > now), default=None)


class Feed:
    """The day's board as JSON bytes and its ETag, rebuilt only when the day folder changed."""

    name = "board feed"

    def __init__(self, race_date: date, min_rebuild_s: float, day_dir: Path | None = None):
        self.race_date, self.min_rebuild_s, self.day_dir = race_date, min_rebuild_s, day_dir
        self._lock = threading.Lock()
        self._checked: float | None = None
        self._fingerprint: Fingerprint | None = None
        self._body: bytes | None = None
        self._gzipped: bytes | None = None
        self._etag: str | None = None
        self._next_cutoff: datetime | None = None

    def _day_files(self) -> Fingerprint:
        day_dir = jra_day_dir(self.race_date, self.day_dir)
        if not day_dir.is_dir():
            return ()
        out = []
        for p in day_dir.iterdir():
            try:
                st = p.stat()
            except OSError:  # gone between the listing and the stat
                continue
            if p.is_file():
                out.append((p.name, st.st_size, st.st_mtime_ns))
        return tuple(sorted(out))

    def current(self, *, gzipped: bool = False) -> tuple[bytes | None, str | None]:
        """The feed body (gzip-compressed when asked) and its ETag; (None, None) before any
        build succeeded."""
        with self._lock:
            now = time.monotonic()
            if self._checked is None or now - self._checked >= self.min_rebuild_s:
                self._checked = now
                self._refresh()
            if self._body is not None and gzipped and self._gzipped is None:
                self._gzipped = gzip.compress(self._body, 6)
            return (self._gzipped if gzipped else self._body), self._etag

    def _build(self) -> dict[str, Any] | None:
        """The new feed, or None when no file changed and no checkpoint cutoff passed since the
        last one (a reference fills at its cutoff without a log write)."""
        fingerprint = self._day_files()
        now = board_feed.utc_now()
        due = self._next_cutoff is not None and now >= self._next_cutoff
        if fingerprint == self._fingerprint and self._body is not None and not due:
            return None
        board = build_board(self.race_date, now=lambda: now, day_dir=self.day_dir)
        self._fingerprint, self._next_cutoff = fingerprint, _next_cutoff(board, now)
        return board

    def _refresh(self) -> None:
        try:
            built = self._build()
        except Exception:  # keep the last good feed; try again on the next check
            log.exception("%s for %s: build failed", self.name, self.race_date)
            return
        if built is None:
            return
        body = json.dumps(built, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._body, self._gzipped = body, None
        self._etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'


def _without_clock(shadow: dict[str, Any]) -> dict[str, Any]:
    """The shadow feed less the fields that move with the clock alone."""
    races = [{k: v for k, v in r.items() if k != "minutes_to_post"} for r in shadow["races"]]
    return {**{k: v for k, v in shadow.items() if k != "generated_utc"}, "races": races}


class ShadowFeed(Feed):
    """The day's shadow values and flags, rebuilt on every check: buckets and stale flags follow
    the clock. The values cache and the log readers persist between builds."""

    name = "shadow feed"

    def __init__(
        self,
        race_date: date,
        min_rebuild_s: float,
        models: ShadowModels | None,
        flag_rules: FlagRules,
        day_dir: Path | None = None,
    ):
        super().__init__(race_date, min_rebuild_s, day_dir)
        self.models, self.flag_rules = models, flag_rules
        self._cache: dict = {}
        self._shown: dict[str, Any] | None = None

    def _build(self) -> dict[str, Any] | None:
        shadow = build_shadow(
            self.race_date,
            models=self.models,
            flag_rules=self.flag_rules,
            cache=self._cache,
            day_dir=self.day_dir,
        )
        shown = _without_clock(shadow)
        if shown == self._shown and self._body is not None:
            return None
        self._shown = shown
        return shadow


_VENUE = re.compile(r"[A-Z]{2,4}")  # a JRA venue code (TOK, KYO, ...)


class BestBets:
    """Each venue's best bets as JSON bytes, rebuilt at most every `min_rebuild_s` per venue (its
    race, checkpoint and freeze follow the clock). The leaderboard file is read again when it
    changes; the log readers persist between builds. A build that fails keeps the last body."""

    def __init__(
        self,
        race_date: date,
        min_rebuild_s: float,
        models: ShadowModels | None,
        day_dir: Path | None = None,
    ):
        self.race_date, self.min_rebuild_s, self.day_dir = race_date, min_rebuild_s, day_dir
        self.models = models
        self._leaderboard: LeaderboardFile | None = None  # resolved at the first request
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._bodies: dict[str, tuple[float, bytes]] = {}

    def current(self, venue: str) -> bytes | None:
        """The venue's body; None before any build of it succeeded."""
        with self._lock:
            now = time.monotonic()
            last = self._bodies.get(venue)
            if last is not None and now - last[0] < self.min_rebuild_s:
                return last[1]
            try:
                if self._leaderboard is None:
                    self._leaderboard = LeaderboardFile(paths.reports_dir() / PRIVATE_JSON)
                built = build_best_bets(
                    self.race_date,
                    venue=venue,
                    leaderboard=self._leaderboard.rows(),
                    models=self.models,
                    now=board_feed.utc_now,
                    day_dir=self.day_dir,
                    cache=self._cache,
                )
            except Exception:  # keep the last good body; try again on the next request
                log.exception("best bets for %s %s: build failed", self.race_date, venue)
                return None if last is None else last[1]
            body = json.dumps(built, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._bodies[venue] = (now, body)
            return body


def _etag_matches(header: str | None, etag: str) -> bool:
    # a proxy that re-compresses hands the tag back weak: W/"..."
    asked = {t.strip().removeprefix("W/") for t in (header or "").split(",")}
    return etag in asked or "*" in asked


def _fresh_token() -> str:
    return secrets.token_urlsafe(24)


def write_access_token(token: str | None = None) -> str:
    """`token` (URL-safe characters), or a fresh one when None, written alone to
    $HORSETRACK_DATA_DIR/board/access-token with mode 0600 (replacing the last run's) and
    returned. The `board/` folder is resolved, a symlink followed, and DataDirInsideRepo raised
    before any file is created when it lies in the repository."""
    if token is None:
        token = _fresh_token()
    elif not _TOKEN_CHARS.fullmatch(token):
        raise ValueError("token: a non-empty string of [A-Za-z0-9_-], or None for a fresh one")
    folder = paths.outside_repo(paths.data_dir() / "board", what=f"{paths.DATA_DIR_VAR}/board")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "access-token"
    fd, tmp = tempfile.mkstemp(prefix=".access-token.", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            os.fchmod(fh.fileno(), 0o600)
            fh.write(token + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return token


def _is_token(candidate: str, secret: bytes) -> bool:
    # bytes, so a non-ASCII candidate compares (False) instead of raising
    return hmac.compare_digest(candidate.encode("utf-8", "replace"), secret)


def _cookie_values(headers: Message, name: str) -> list[str]:
    """The values of every cookie called `name` in the request's Cookie headers."""
    values = []
    for header in headers.get_all("Cookie") or []:
        for pair in header.split(";"):
            key, sep, value = pair.strip().partition("=")
            if sep and key == name:
                values.append(value)
    return values


def _redact(text: str, token: str | None) -> str:
    """`text` with every query string and the token blanked, for the request log."""
    text = _QUERY.sub("?[redacted]", text)
    return text if token is None else text.replace(token, "[redacted]")


def _make_handler(
    page_path: Path,
    feed: Feed,
    shadow_feed: ShadowFeed,
    best_bets: BestBets,
    token: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    pages = {"/": page_path, "/shadow": page_path.with_name("shadow.html")}
    scripts = {"/board.js": "board.js", "/shadow.js": "shadow.js"}
    feeds = {"/feed": feed, "/shadow-feed": shadow_feed}
    best_path = "/best-bets"
    known = {*pages, *scripts, *feeds, best_path}
    secret = None if token is None else token.encode("ascii")

    class Handler(BaseHTTPRequestHandler):
        grant: str | None = None  # the Set-Cookie of a request let in by its ?t=

        def _send(
            self,
            status: int,
            body: bytes = b"",
            ctype: str | None = None,
            extra: dict | None = None,
        ) -> None:
            self.send_response(status)
            headers = {**HEADERS, **(extra or {})}
            if self.grant is not None and status != 404:
                headers["Set-Cookie"] = self.grant
            for key, value in headers.items():
                self.send_header(key, value)
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _not_found(self) -> None:
            if secret is None:
                return self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return self._send(404)  # empty: a missing token and a missing page look the same

        def _wants_gzip(self) -> bool:
            return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

        def _admit(self, path: str, query: str) -> bool:
            """With a token: whether the request's ?t= or cookie holds it and `path` exists;
            a right ?t= also grants the cookie."""
            self.grant = None
            if secret is None:
                return True
            asked = [v for k, v in parse_qsl(query, keep_blank_values=True) if k == "t"]
            by_query = any(_is_token(v, secret) for v in asked)
            cookies = _cookie_values(self.headers, COOKIE_NAME)
            if not (by_query or any(_is_token(v, secret) for v in cookies)) or path not in known:
                return False
            if by_query:
                proto = (self.headers.get("X-Forwarded-Proto") or "").split(",")[0]
                secure = "; Secure" if proto.strip().lower() == "https" else ""
                self.grant = f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict{secure}"
            return True

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path, _, query = self.path.partition("?")
            if not self._admit(path, query):
                return self._not_found()
            if path in pages:
                try:
                    html = pages[path].read_bytes()
                except OSError:
                    return self._send(503, b"page not built\n", "text/plain; charset=utf-8")
                return self._send(200, html, "text/html; charset=utf-8")
            if path in scripts:
                try:
                    script = page_path.with_name(scripts[path]).read_bytes()
                except OSError:
                    return self._not_found()
                return self._send(200, script, "text/javascript; charset=utf-8")
            if path in feeds:
                gzipped = self._wants_gzip()
                body, etag = feeds[path].current(gzipped=gzipped)
                if body is None or etag is None:
                    return self._send(503, b"no feed yet\n", "text/plain; charset=utf-8")
                if _etag_matches(self.headers.get("If-None-Match"), etag):
                    return self._send(304, extra={"ETag": etag})
                extra = {"ETag": etag, "Vary": "Accept-Encoding"}
                if gzipped:
                    extra["Content-Encoding"] = "gzip"
                return self._send(200, body, "application/json; charset=utf-8", extra)
            if path == best_path:
                asked = [v for k, v in parse_qsl(query) if k == "venue"]
                if len(asked) != 1 or not _VENUE.fullmatch(asked[0]):
                    return self._send(400, b"venue=<VENUE> needed\n", "text/plain; charset=utf-8")
                body = best_bets.current(asked[0])
                if body is None:
                    return self._send(503, b"no best bets yet\n", "text/plain; charset=utf-8")
                gzipped = self._wants_gzip()
                extra = {"Vary": "Accept-Encoding"}
                if gzipped:
                    body, extra["Content-Encoding"] = gzip.compress(body, 6), "gzip"
                return self._send(200, body, "application/json; charset=utf-8", extra)
            return self._not_found()

        do_HEAD = do_GET  # noqa: N815 (http.server API)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (http.server API)
            line = _redact(format % args, token)
            sys.stderr.write(f"{self.log_date_time_string()} {line}\n")

    return Handler


def make_server(
    race_date: date,
    *,
    host: str = LOCALHOST,
    port: int = DEFAULT_PORT,
    page_path: Path,
    min_rebuild_s: float = 5,
    models: ShadowModels | None = None,
    flag_rules: FlagRules | None = None,
    day_dir: Path | None = None,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """A server for the board of `race_date`, bound to (host, port) but not yet serving. Without
    `models` the shadow feed's values are null; `flag_rules` defaults to the shipped rules. Both
    feeds read the odds logs from `day_dir` when given (a pulled VM folder), else the Mac's own.
    With `token` (URL-safe characters) every path needs it, as `?t=` or the cookie that sets;
    anything else is an empty 404. `/best-bets` reads the private leaderboard JSON under the
    reports dir (`leaderboard_build.PRIVATE_JSON`), resolved at its first request."""
    if token is not None and not _TOKEN_CHARS.fullmatch(token):
        raise ValueError("token: a non-empty string of [A-Za-z0-9_-], or None for no token")
    feed = Feed(race_date, min_rebuild_s, day_dir)
    rules = load_flag_rules() if flag_rules is None else flag_rules
    shadow_feed = ShadowFeed(race_date, min_rebuild_s, models, rules, day_dir)
    best_bets = BestBets(race_date, min_rebuild_s, models, day_dir)
    handler = _make_handler(Path(page_path), feed, shadow_feed, best_bets, token)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def board_serve_command(
    race_date: str = typer.Option(..., "--date", help="Race day to serve (YYYY-MM-DD)."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to listen on."),
    host: str = typer.Option(LOCALHOST, "--host", help="Address to bind; keep 127.0.0.1."),
    page: str | None = typer.Option(
        None,
        "--page",
        help="The board page served at / (default: web/board.html beside server.py).",
    ),
    from_vm: str | None = typer.Option(
        None,
        "--from-vm",
        help="Read the day's odds logs pulled from this VM (ssh alias; `pull-vm --every`) from "
        "$HORSETRACK_DATA_DIR/vm/<alias>/ instead of the Mac's own odds_snapshots/.",
    ),
    tunnel: bool = typer.Option(
        False,
        "--tunnel",
        help="Expose the pages to a phone through a cloudflared quick tunnel (cloudflared on "
        "PATH). Every path then needs a fresh access token, kept in "
        "$HORSETRACK_DATA_DIR/board/access-token; the phone URL holding it is printed once.",
    ),
) -> None:
    """Serve the board page at / and the day's live feed at /feed, and the shadow page at /shadow
    and its feed at /shadow-feed."""
    try:
        day = date.fromisoformat(race_date)
    except ValueError as exc:
        raise typer.BadParameter(f"{race_date!r} is not a date (YYYY-MM-DD)") from exc
    try:
        paths.data_dir()
    except (paths.DataDirNotSet, paths.DataDirInsideRepo) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    day_dir = None
    if from_vm is not None:
        try:
            day_dir = vm_day_dir(from_vm, day)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--from-vm") from exc
    if tunnel and host != LOCALHOST:  # the tunnel connects to 127.0.0.1
        raise typer.BadParameter(f"--tunnel serves on {LOCALHOST} only", param_hint="--host")
    if host != LOCALHOST:
        typer.echo(
            f"warning: binding {host}, not {LOCALHOST}: the board is reachable from other "
            "machines and has no login",
            err=True,
        )
    page_path = DEFAULT_PAGE if page is None else Path(page).expanduser()
    if not page_path.is_file():
        typer.echo(f"warning: no page at {page_path} yet: / answers 503 until it exists", err=True)
    try:  # once, at start: a pointer that names a broken fit stops here, as in shadow-day
        models = load_shadow_models()
    except Exception as exc:
        typer.echo(f"error: shadow models: {' '.join(str(exc).split())[:300]}", err=True)
        raise typer.Exit(1) from exc
    fits = ",".join(f"{k}={v}" for k, v in models.ids.items()) if models else "none (values null)"
    # in memory until the tunnel is up: a start that fails keeps the running one's token file
    token = _fresh_token() if tunnel else None
    try:
        server = make_server(
            day,
            host=host,
            port=port,
            page_path=page_path,
            models=models,
            day_dir=day_dir,
            token=token,
        )
    except OSError as exc:  # the port is taken, most often by an earlier board-serve
        typer.echo(f"error: cannot listen on {host}:{port}: {exc}", err=True)
        raise typer.Exit(1) from exc
    bound_host, bound_port = server.server_address[:2]
    typer.echo(f"serving the board of {day} on http://{bound_host}:{bound_port}/")
    typer.echo(f"shadow page on http://{bound_host}:{bound_port}/shadow, models={fits}")
    try:
        with ExitStack() as stack:
            if tunnel:
                # SIGTERM (and SIGINT, even if inherited as ignored) stops like Ctrl-C, so the
                # tunnel below is closed; the old handlers come back on the way out
                for sig in (signal.SIGINT, signal.SIGTERM):
                    stack.callback(signal.signal, sig, signal.signal(sig, _interrupt))
                url = stack.enter_context(quick_tunnel(bound_port))
                try:
                    write_access_token(token)
                except (OSError, paths.DataDirInsideRepo) as exc:
                    typer.echo(f"error: access token: {exc}", err=True)
                    raise typer.Exit(1) from exc
                typer.echo(f"open on the phone: {url}/shadow?t={token}")
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except TunnelError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        server.server_close()


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt
