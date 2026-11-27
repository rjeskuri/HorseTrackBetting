"""Serve the board page and its live feed on localhost (ticket #41), and the shadow page and its
feed beside them (ticket #121).

Six paths only: `/` (the page at `page_path`, read on every request), `/board.js` (the page's
script, `board.js` beside `page_path`, ticket #42), `/feed` (`build_board` of the day as JSON,
docs/specs/board-feed.md) with an ETag, so the page's poll costs a 304 when nothing changed;
`/shadow` and `/shadow.js` (`shadow.html` and `shadow.js` beside `page_path`, ticket #122) and
`/shadow-feed` (`build_shadow` of the day, with the same ETag, gzip and throttle as /feed).
Anything else is 404: no listings, no other files. Every response carries
Cache-Control no-store, X-Robots-Tag noindex nofollow, Referrer-Policy no-referrer and
X-Content-Type-Options nosniff.

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

    horsetrack japan board-serve --date 2026-11-29 [--port 8765] [--host 127.0.0.1] [--page P]
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sys
import threading
import time
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.board import feed as board_feed
from horsetrack.countries.japan.board.feed import build_board
from horsetrack.countries.japan.board.flags import FlagRules, load_flag_rules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.board.snapshot import jra_day_dir
from horsetrack.countries.japan.logger.pull_vm import vm_day_dir
from horsetrack.countries.japan.models.v0.loader import load_shadow_models
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

log = logging.getLogger(__name__)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PAGE = Path(__file__).with_name("web") / "board.html"  # written by ticket #42
HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}

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


def _etag_matches(header: str | None, etag: str) -> bool:
    # a proxy that re-compresses hands the tag back weak: W/"..."
    asked = {t.strip().removeprefix("W/") for t in (header or "").split(",")}
    return etag in asked or "*" in asked


def _make_handler(
    page_path: Path, feed: Feed, shadow_feed: ShadowFeed
) -> type[BaseHTTPRequestHandler]:
    pages = {"/": page_path, "/shadow": page_path.with_name("shadow.html")}
    scripts = {"/board.js": "board.js", "/shadow.js": "shadow.js"}
    feeds = {"/feed": feed, "/shadow-feed": shadow_feed}

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            status: int,
            body: bytes = b"",
            ctype: str | None = None,
            extra: dict | None = None,
        ) -> None:
            self.send_response(status)
            for key, value in {**HEADERS, **(extra or {})}.items():
                self.send_header(key, value)
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _wants_gzip(self) -> bool:
            return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]
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
                    return self._send(404, b"not found\n", "text/plain; charset=utf-8")
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
            return self._send(404, b"not found\n", "text/plain; charset=utf-8")

        do_HEAD = do_GET  # noqa: N815 (http.server API)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (http.server API)
            sys.stderr.write(f"{self.log_date_time_string()} {format % args}\n")

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
) -> ThreadingHTTPServer:
    """A server for the board of `race_date`, bound to (host, port) but not yet serving. Without
    `models` the shadow feed's values are null; `flag_rules` defaults to the shipped rules. Both
    feeds read the odds logs from `day_dir` when given (a pulled VM folder), else the Mac's own."""
    feed = Feed(race_date, min_rebuild_s, day_dir)
    rules = load_flag_rules() if flag_rules is None else flag_rules
    shadow_feed = ShadowFeed(race_date, min_rebuild_s, models, rules, day_dir)
    server = ThreadingHTTPServer((host, port), _make_handler(Path(page_path), feed, shadow_feed))
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
    server = make_server(
        day, host=host, port=port, page_path=page_path, models=models, day_dir=day_dir
    )
    bound_host, bound_port = server.server_address[:2]
    typer.echo(f"serving the board of {day} on http://{bound_host}:{bound_port}/")
    typer.echo(f"shadow page on http://{bound_host}:{bound_port}/shadow, models={fits}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
