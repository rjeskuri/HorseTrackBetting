"""Serve the board page and its live feed on localhost (ticket #41).

Two paths only: `/` (the page at `page_path`, read on every request) and `/feed` (`build_board` of
the day as JSON, docs/specs/board-feed.md) with an ETag, so the page's poll costs a 304 when
nothing changed. Anything else is 404: no listings, no other files. Every response carries
Cache-Control no-store, X-Robots-Tag noindex nofollow, Referrer-Policy no-referrer and
X-Content-Type-Options nosniff.

The feed is rebuilt at most every `min_rebuild_s`, and only when a file of the day folder (an odds
log or a post-time sidecar) appeared, went away, or changed size or mtime. A log being appended
reads safely: `odds_log` counts whole gzip members only, so a torn last member is ignored. A build
that fails keeps the last good feed. The default bind is 127.0.0.1; the board has no login.

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
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.board.feed import build_board

log = logging.getLogger(__name__)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PAGE = Path(__file__).with_name("board.html")  # rendered by ticket #42
HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}

Fingerprint = tuple[tuple[str, int, int], ...]


class Feed:
    """The day's board as JSON bytes and its ETag, rebuilt only when the day folder changed."""

    def __init__(self, race_date: date, min_rebuild_s: float):
        self.race_date, self.min_rebuild_s = race_date, min_rebuild_s
        self._lock = threading.Lock()
        self._checked: float | None = None
        self._fingerprint: Fingerprint | None = None
        self._body: bytes | None = None
        self._gzipped: bytes | None = None
        self._etag: str | None = None

    def _day_files(self) -> Fingerprint:
        day_dir = paths.odds_snapshot_path("jra", self.race_date.isoformat(), "_").parent
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

    def _refresh(self) -> None:
        try:
            fingerprint = self._day_files()
            if fingerprint == self._fingerprint and self._body is not None:
                return
            board = build_board(self.race_date)
        except Exception:  # keep the last good feed; try again on the next check
            log.exception("board feed for %s: build failed", self.race_date)
            return
        body = json.dumps(board, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._fingerprint, self._body, self._gzipped = fingerprint, body, None
        self._etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'


def _etag_matches(header: str | None, etag: str) -> bool:
    # a proxy that re-compresses hands the tag back weak: W/"..."
    asked = {t.strip().removeprefix("W/") for t in (header or "").split(",")}
    return etag in asked or "*" in asked


def _make_handler(page_path: Path, feed: Feed) -> type[BaseHTTPRequestHandler]:
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
            if path == "/":
                try:
                    html = page_path.read_bytes()
                except OSError:
                    return self._send(503, b"page not built\n", "text/plain; charset=utf-8")
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/feed":
                gzipped = self._wants_gzip()
                body, etag = feed.current(gzipped=gzipped)
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
) -> ThreadingHTTPServer:
    """A server for the board of `race_date`, bound to (host, port) but not yet serving."""
    feed = Feed(race_date, min_rebuild_s)
    server = ThreadingHTTPServer((host, port), _make_handler(Path(page_path), feed))
    server.daemon_threads = True
    return server


def board_serve_command(
    race_date: str = typer.Option(..., "--date", help="Race day to serve (YYYY-MM-DD)."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to listen on."),
    host: str = typer.Option(LOCALHOST, "--host", help="Address to bind; keep 127.0.0.1."),
    page: str | None = typer.Option(
        None, "--page", help="The board page served at / (default: board.html beside server.py)."
    ),
) -> None:
    """Serve the board page at / and the day's live feed at /feed."""
    try:
        day = date.fromisoformat(race_date)
    except ValueError as exc:
        raise typer.BadParameter(f"{race_date!r} is not a date (YYYY-MM-DD)") from exc
    try:
        paths.data_dir()
    except (paths.DataDirNotSet, paths.DataDirInsideRepo) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if host != LOCALHOST:
        typer.echo(
            f"warning: binding {host}, not {LOCALHOST}: the board is reachable from other "
            "machines and has no login",
            err=True,
        )
    page_path = DEFAULT_PAGE if page is None else Path(page).expanduser()
    if not page_path.is_file():
        typer.echo(f"warning: no page at {page_path} yet: / answers 503 until it exists", err=True)
    server = make_server(day, host=host, port=port, page_path=page_path)
    bound_host, bound_port = server.server_address[:2]
    typer.echo(f"serving the board of {day} on http://{bound_host}:{bound_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
