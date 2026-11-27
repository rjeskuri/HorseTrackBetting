"""Token-gated access and a cloudflared quick tunnel for board-serve (ticket #505).

Contract pinned here (`horsetrack.countries.japan.board.server`):

The access token
- `write_access_token(token: str | None = None) -> str`: writes `token` as given, or, when None (the
  default), a fresh `secrets.token_urlsafe(24)` (32 characters of [A-Za-z0-9_-]) on every call, and
  returns what it wrote. It is written alone (a trailing newline allowed) to `paths.data_dir() /
  "board" / "access-token"`, the `board/` folder created when missing, with mode 0600 whatever the
  umask and whatever mode an earlier file had; it replaces the earlier token. A data dir inside the
  repository raises `paths.DataDirInsideRepo` and nothing is written. So does a `board/` that
  resolves inside the repository (a symlink into the checkout): the token folder is resolved and
  checked before any file, temporary or final, is created, and nothing is written there. No other
  file is left in `board/`.

The gate: `make_server(..., token: str | None = None)`
- `token=None`, the default: today's behaviour, unchanged. No token is needed, no cookie is set and
  a `?t=` is ignored.
- `token=<str>`: every path, HEAD as GET, needs the token, also from 127.0.0.1 (the tunnel connects
  from localhost: one code path). It is accepted as the query parameter `t` (`?t=<token>`) or as the
  cookie the server sets. A request with the right `?t=` is served as usual, and its response
  carries one `Set-Cookie: <name>=<value>; HttpOnly; SameSite=Strict` (a Path, if given, is `/`),
  with `Secure` exactly when the request's `X-Forwarded-Proto` is `https` (cloudflared sends it). A
  later request carrying that cookie is served without `?t=`, also beside other cookies (Cloudflare
  sets its own). The cookie's name and value are the server's choice: the tests read them from
  Set-Cookie.
- Missing, wrong (same length, other length, empty, non-ASCII), under another query parameter, in
  the path, a wrong cookie value or the right value under another cookie name: 404 with an empty
  body, identical (status, every header but Date, body) to the 404 of an unknown path, with no
  Set-Cookie. With a token, an unknown path is that same empty 404, with or without the token. The
  gate comes first: a page that would answer 503 is a 404 to a request without the token.
- Every candidate is compared with `hmac.compare_digest`, looked up at call time (`import hmac`, or
  a module-level `compare_digest` imported from hmac), and its answer alone decides.
- `token=""` raises ValueError.
- The token never appears in what the server writes to stdout, stderr or the logging system: the
  request log line of `GET /shadow?t=<token>` included.

Headers
- Every response, with or without a token, 200, 304, 404 and 503 alike, carries `X-Frame-Options:
  DENY` beside the four headers it has today.

The tunnel
- `parse_tunnel_url(output: str) -> str | None`: the quick-tunnel URL
  `https://<name>.trycloudflare.com` (no trailing slash) in cloudflared's output, else None, under
  either banner ("Your quick Tunnel has been created!" or "Your protected quick Tunnel has been
  created!"). `api.trycloudflare.com`, which cloudflared's error lines name, and the other
  Cloudflare URLs of its banner are never it.
- `quick_tunnel(port: int, *, timeout_s: float = 30.0)`, a context manager. On entry it finds
  `cloudflared` on PATH at call time, starts `cloudflared tunnel --url http://127.0.0.1:<port>
  --no-autoupdate` (exactly that argv after the executable), reads the child's output (cloudflared
  prints its banner on stderr) until `parse_tunnel_url` finds the URL and then on until the first
  registered connection, a line holding `Registered tunnel connection`, and only then yields the
  URL. `Unregistered tunnel connection`, `Registered session`, `Register tunnel error from server
  side` and the other lines of a failing connection are not a registration. It keeps draining the
  child's output afterwards, so cloudflared never blocks on a full pipe. `TunnelError` (exported by
  the module, its message naming cloudflared) when cloudflared is not on PATH, when the child exits
  before printing a URL or before a connection registered, or when the URL and a registered
  connection did not both come within `timeout_s`; in each case no child is left running. On exit,
  normal or by an exception (KeyboardInterrupt included, which propagates), the child is terminated
  and, if it is still running a few seconds later (at most 5), killed, so the exit takes under 10 s
  even for a child that ignores SIGTERM.

The command
- `horsetrack japan board-serve --date D --tunnel [--port N] [--page P]` makes a token in memory,
  serves with `make_server(..., token=<it>)`, opens `quick_tunnel` on the bound port (so `--port 0`
  works), and only once the server is bound and the tunnel is up (a connection registered) writes
  the token to the token file (`write_access_token`), before it prints exactly one line holding the
  token: the phone URL `https://<name>.trycloudflare.com/shadow?t=<token>` (path `/shadow`, `/` or
  empty, query only `t`; stdout or stderr). The token file then holds the token of that URL. The
  token appears nowhere else in its stdout or stderr, nor in any file under the data dir but the
  token file. On SIGINT or SIGTERM it stops within 15 s and the cloudflared child is gone. A start
  that fails, the port taken, cloudflared missing from PATH, or cloudflared exiting before its URL
  or before a connection registered, exits non-zero, prints no phone URL, leaves no cloudflared
  running and leaves the token file as it was: an earlier one byte for byte, none when there was
  none. Without cloudflared on PATH it names cloudflared on stderr. Without `--tunnel` nothing
  changes: no token, no token file, no cloudflared started.

No real cloudflared ever runs: a fake `cloudflared` (a sh wrapper exec-ing this interpreter on a
small script) is the only one on a PATH of it and /usr/bin:/bin. It records its argv and pid, prints
a realistic quick-tunnel banner and, by default, a registered connection on stderr, then sleeps;
a test can make it print more later, pause, leave a marker file, or exit. Ports are 0, unused, or
held by the test; no external network.
"""

from __future__ import annotations

import gzip
import hmac
import http.client
import inspect
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from horsetrack.core import env, paths

DAY = date(2025, 11, 29)
PS = shutil.which("ps") or "/bin/ps"
SYSTEM_PATH = "/usr/bin:/bin"
SIX_PATHS = ("/", "/shadow", "/board.js", "/shadow.js", "/feed", "/shadow-feed")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32}")


def server_module():
    from horsetrack.countries.japan.board import server as board_server

    return board_server


# ---------------------------------------------------------------- HTTP helpers


@dataclass
class Reply:
    status: int
    headers: http.client.HTTPMessage
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def cookies(self) -> list[str]:
        return self.headers.get_all("Set-Cookie") or []


def request_at(
    port: int,
    path: str,
    headers: dict[str, Any] | None = None,
    *,
    method: str = "GET",
    host: str = "127.0.0.1",
) -> Reply:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request(method, path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        if body and (resp.getheader("Content-Encoding") or "").lower() == "gzip":
            body = gzip.decompress(body)
        return Reply(resp.status, resp.headers, body)
    finally:
        conn.close()


def get(server, path: str, headers: dict[str, Any] | None = None, *, method: str = "GET") -> Reply:
    host, port = server.server_address[:2]
    return request_at(port, path, headers, method=method, host=host)


def signature(reply: Reply, *, drop: tuple[str, ...] = ("date",)) -> tuple:
    """What a client can tell a response by: status, headers (bar `drop`) and body."""
    headers = sorted((k.lower(), v) for k, v in reply.headers.items() if k.lower() not in drop)
    return reply.status, headers, reply.body


@dataclass
class Cookie:
    name: str
    value: str
    attrs: dict[str, str]

    @property
    def pair(self) -> str:
        return f"{self.name}={self.value}"


def the_cookie(reply: Reply) -> Cookie:
    """The one Set-Cookie of `reply`, its attributes keyed in lower case."""
    cookies = reply.cookies()
    assert len(cookies) == 1, f"want exactly one Set-Cookie, got {cookies}"
    first, *rest = (part.strip() for part in cookies[0].split(";"))
    name, sep, value = first.partition("=")
    assert sep and name and value, f"Set-Cookie without name=value: {cookies[0]!r}"
    attrs = {}
    for part in rest:
        key, _, val = part.partition("=")
        attrs[key.strip().lower()] = val.strip()
    return Cookie(name.strip(), value.strip(), attrs)


def fresh_token() -> str:
    return secrets.token_urlsafe(24)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


@pytest.fixture
def site(tmp_path) -> Path:
    """board.html, shadow.html and their scripts, each with its own content."""
    d = tmp_path / "site"
    d.mkdir()
    (d / "board.html").write_text("<!doctype html><title>b</title><p>board-5c1e 馬</p>\n", "utf-8")
    (d / "shadow.html").write_text("<!doctype html><title>s</title><p>shadow-9d07</p>\n", "utf-8")
    (d / "board.js").write_text("poll('feed'); // board-js-31aa\n", "utf-8")
    (d / "shadow.js").write_text("poll('shadow-feed'); // shadow-js-c4f2\n", "utf-8")
    return d


@pytest.fixture
def serve(site) -> Iterator[Callable[..., Any]]:
    """Start a server on port 0 in a thread; every server started is shut down in teardown."""
    started = []

    def start(**kwargs):
        kwargs.setdefault("page_path", site / "board.html")
        kwargs.setdefault("port", 0)
        srv = server_module().make_server(DAY, **kwargs)
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


def expect_served(path: str, reply: Reply, site: Path) -> None:
    """`reply` is what the open server gives for `path` today."""
    assert reply.status == 200, f"{path}: {reply.status}"
    files = {
        "/": "board.html",
        "/shadow": "shadow.html",
        "/board.js": "board.js",
        "/shadow.js": "shadow.js",
    }
    if path in files:
        assert reply.body == (site / files[path]).read_bytes(), path
    else:
        assert reply.json()["race_date"] == DAY.isoformat(), path


# ---------------------------------------------------------------- the gate: token required

TOKENS = [
    pytest.param("random", id="random-token"),
    pytest.param("Zq-_0aB9xY-_kLmN0pQrStUvWx-_12Ab", id="fixed-token"),
]


@pytest.fixture(params=TOKENS)
def token(request) -> str:
    value = fresh_token() if request.param == "random" else request.param
    assert TOKEN_RE.fullmatch(value)
    return value


def test_right_query_token_serves_every_path_and_sets_a_strict_httponly_cookie(serve, site, token):
    srv = serve(token=token)

    for path in SIX_PATHS:
        reply = get(srv, f"{path}?t={token}")

        expect_served(path, reply, site)
        cookie = the_cookie(reply)
        assert "httponly" in cookie.attrs, f"{path}: {cookie}"
        assert cookie.attrs.get("samesite", "").lower() == "strict", f"{path}: {cookie}"
        assert cookie.attrs.get("path", "/") == "/", f"{path}: {cookie}"
        assert "secure" not in cookie.attrs, f"{path}: Secure over plain http: {cookie}"


@pytest.mark.parametrize(
    ("forwarded", "secure"),
    [({"X-Forwarded-Proto": "https"}, True), ({"X-Forwarded-Proto": "http"}, False), ({}, False)],
    ids=["via-https-tunnel", "forwarded-http", "direct-http"],
)
def test_the_cookie_is_secure_exactly_when_the_request_came_over_https(
    serve, token, forwarded, secure
):
    srv = serve(token=token)

    reply = get(srv, f"/shadow?t={token}", forwarded)

    assert reply.status == 200
    cookie = the_cookie(reply)
    assert ("secure" in cookie.attrs) is secure, cookie
    assert "httponly" in cookie.attrs and cookie.attrs.get("samesite", "").lower() == "strict"


def test_the_cookie_alone_then_serves_every_path(serve, site, token):
    srv = serve(token=token)
    cookie = the_cookie(get(srv, f"/shadow?t={token}", {"X-Forwarded-Proto": "https"}))

    for path in SIX_PATHS:
        for header in (cookie.pair, f"__cf_bm=a1b2c3; {cookie.pair}; theme=dark"):
            reply = get(srv, path, {"Cookie": header})

            expect_served(path, reply, site)

        head = get(srv, path, {"Cookie": cookie.pair}, method="HEAD")
        assert head.status == 200, f"HEAD {path}: {head.status}"
        assert head.body == b""


def test_without_the_token_every_path_is_an_empty_404_like_an_unknown_path(serve, token):
    srv = serve(token=token)
    reference = get(srv, "/no-such-path")
    assert reference.status == 404
    assert reference.body == b""
    assert reference.cookies() == []
    other = fresh_token()
    assert len(other) == len(token) and other != token

    asks = {
        "no token": "",
        "wrong, same length": f"?t={other}",
        "wrong, one shorter": f"?t={token[:-1]}",
        "wrong, one longer": f"?t={token}x",
        "wrong, reversed": f"?t={token[::-1]}",
        "empty": "?t=",
        "non-ascii": "?t=%E9%A6%AC" + token[3:],
        "another parameter": f"?token={token}",
        "upper-case name": f"?T={token}",
        "a longer name": f"?xt={token}",
    }
    for path in SIX_PATHS:
        for name, query in asks.items():
            reply = get(srv, path + query)

            assert signature(reply) == signature(reference), f"{path} {name}: {reply.status}"

        head = get(srv, path, method="HEAD")
        assert head.status == 404 and head.body == b"", f"HEAD {path}: {head.status}"
        assert signature(head) == signature(get(srv, "/no-such-path", method="HEAD"))

    for path in (f"/{token}", f"/shadow/{token}", f"/t={token}"):
        assert signature(get(srv, path)) == signature(reference), path


def test_a_wrong_cookie_is_an_empty_404(serve, token):
    srv = serve(token=token)
    reference = get(srv, "/no-such-path")
    cookie = the_cookie(get(srv, f"/?t={token}"))
    other = fresh_token()

    wrong = {
        "other value, same length": f"{cookie.name}={other}",
        "value plus a character": f"{cookie.name}={cookie.value}x",
        "value less a character": f"{cookie.name}={cookie.value[:-1]}",
        "empty value": f"{cookie.name}=",
        "the value under another name": f"decoy={cookie.value}",
        "the value under a longer name": f"x{cookie.name}={cookie.value}",
        "non-ascii value": f"{cookie.name}=".encode("ascii") + "馬".encode(),
    }
    for path in SIX_PATHS:
        for name, header in wrong.items():
            reply = get(srv, path, {"Cookie": header})

            assert signature(reply) == signature(reference), f"{path} {name}: {reply.status}"


def test_with_a_token_an_unknown_path_is_the_same_empty_404(serve, token):
    srv = serve(token=token)
    reference = get(srv, "/no-such-path")
    cookie = the_cookie(get(srv, f"/shadow?t={token}"))

    for path in ("/index.html", "/feed/", "/shadow.json", "/.env", "/odds_snapshots/"):
        with_cookie = get(srv, path, {"Cookie": cookie.pair})
        assert signature(with_cookie, drop=("date", "set-cookie")) == signature(
            reference, drop=("date", "set-cookie")
        ), path
        with_query = get(srv, f"{path}?t={token}")
        assert with_query.status == 404 and with_query.body == b"", path


def test_the_gate_comes_before_a_missing_page(serve, tmp_path, token):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "board.html").write_text("<p>only the board</p>\n", "utf-8")
    srv = serve(page_path=bare / "board.html", token=token)
    reference = get(srv, "/no-such-path")

    assert signature(get(srv, "/shadow")) == signature(reference)
    assert get(srv, f"/shadow?t={token}").status == 503
    assert get(srv, f"/?t={token}").status == 200


def test_requests_from_localhost_are_gated_too(serve, token):
    srv = serve(token=token)
    assert srv.server_address[0] == "127.0.0.1"

    for headers in ({}, {"Host": "localhost"}, {"Host": f"127.0.0.1:{srv.server_address[1]}"}):
        reply = get(srv, "/shadow", headers)
        assert reply.status == 404 and reply.body == b"", headers


def test_an_empty_token_is_refused(site):
    with pytest.raises(ValueError):
        server_module().make_server(DAY, port=0, page_path=site / "board.html", token="")


def test_the_token_is_compared_with_hmac_compare_digest(serve, monkeypatch):
    board_server = server_module()
    real = hmac.compare_digest
    calls = []
    forced: dict[str, bool | None] = {"answer": None}

    def spy(a, b):
        calls.append((a, b))
        return real(a, b) if forced["answer"] is None else forced["answer"]

    monkeypatch.setattr(hmac, "compare_digest", spy)
    if hasattr(board_server, "compare_digest"):
        monkeypatch.setattr(board_server, "compare_digest", spy)
    token = fresh_token()
    srv = serve(token=token)

    first = get(srv, f"/shadow?t={token}")
    assert first.status == 200
    assert calls, "the right ?t= was not compared with hmac.compare_digest"
    cookie = the_cookie(first)
    before = len(calls)
    assert get(srv, "/feed", {"Cookie": cookie.pair}).status == 200
    assert len(calls) > before, "the cookie was not compared with hmac.compare_digest"

    forced["answer"] = False  # its answer alone decides: no other comparison lets these in
    assert get(srv, f"/shadow?t={token}").status == 404
    assert get(srv, "/feed", {"Cookie": cookie.pair}).status == 404

    forced["answer"] = True
    other = fresh_token()
    assert len(other) == len(token) and other != token
    assert get(srv, f"/shadow?t={other}").status == 200


def test_the_token_is_never_logged(serve, capfd, caplog, token):
    caplog.set_level(logging.DEBUG)
    srv = serve(token=token)

    cookie = the_cookie(get(srv, f"/shadow?t={token}", {"X-Forwarded-Proto": "https"}))
    for path in SIX_PATHS:
        get(srv, f"{path}?t={token}")
        get(srv, f"{path}?t={token}", method="HEAD")
        get(srv, path, {"Cookie": cookie.pair})
        get(srv, f"{path}?t={token}x")
    get(srv, f"/no-such-path?t={token}")

    out, err = capfd.readouterr()
    assert token not in out
    assert token not in err, [line for line in err.splitlines() if token in line]
    assert token not in caplog.text


# ---------------------------------------------------------------- without a token: unchanged


def test_without_a_token_every_path_is_open_and_no_cookie_is_set(serve, site):
    params = inspect.signature(server_module().make_server).parameters
    assert params["token"].default is None
    srv = serve(token=None)
    default = serve()

    for server in (srv, default):
        for path in SIX_PATHS:
            for query in ("", f"?t={fresh_token()}"):
                reply = get(server, path + query)

                expect_served(path, reply, site)
                assert reply.cookies() == [], f"{path}{query}: {reply.cookies()}"
                assert reply.headers["X-Frame-Options"] == "DENY", path
        assert get(server, "/no-such-path").status == 404


# ---------------------------------------------------------------- headers


def test_x_frame_options_deny_on_every_response(serve, tmp_path, token):
    bare = tmp_path / "bare"
    bare.mkdir()
    open_srv = serve(page_path=bare / "board.html")  # no page yet: / is 503
    gated = serve(token=token)
    cookie = the_cookie(get(gated, f"/?t={token}"))
    etag = get(gated, "/feed", {"Cookie": cookie.pair}).headers["ETag"]
    open_etag = get(open_srv, "/feed").headers["ETag"]

    replies = {
        "open 503": (get(open_srv, "/"), 503),
        "open 200 feed": (get(open_srv, "/feed"), 200),
        "open 304": (get(open_srv, "/feed", {"If-None-Match": open_etag}), 304),
        "open 404": (get(open_srv, "/no-such-path"), 404),
        "gated 200 ?t=": (get(gated, f"/shadow?t={token}"), 200),
        "gated 200 cookie": (get(gated, "/shadow.js", {"Cookie": cookie.pair}), 200),
        "gated 304": (get(gated, "/feed", {"Cookie": cookie.pair, "If-None-Match": etag}), 304),
        "gated 404 no token": (get(gated, "/shadow"), 404),
        "gated 404 wrong token": (get(gated, f"/feed?t={fresh_token()}"), 404),
        "gated 404 unknown path": (get(gated, "/nope", {"Cookie": cookie.pair}), 404),
        "gated HEAD 404": (get(gated, "/", method="HEAD"), 404),
    }
    for name, (reply, status) in replies.items():
        assert reply.status == status, f"{name}: {reply.status}"
        assert (reply.headers["X-Frame-Options"] or "").strip().upper() == "DENY", name
        assert (reply.headers["Cache-Control"] or "").strip().lower() == "no-store", name
        assert (reply.headers["Referrer-Policy"] or "").strip().lower() == "no-referrer", name
        assert (reply.headers["X-Content-Type-Options"] or "").strip().lower() == "nosniff", name
        assert "noindex" in (reply.headers["X-Robots-Tag"] or "").lower(), name


# ---------------------------------------------------------------- the token file


def token_file(data_dir: Path) -> Path:
    return data_dir / "board" / "access-token"


def test_write_access_token_writes_a_fresh_0600_token_each_run(data_dir):
    old_umask = os.umask(0)  # a permissive umask must not loosen the file
    try:
        tokens = []
        for _ in range(4):
            token = server_module().write_access_token()

            assert isinstance(token, str) and TOKEN_RE.fullmatch(token), token
            path = token_file(data_dir)
            assert path.read_text("utf-8").strip() == token
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)
            tokens.append(token)
    finally:
        os.umask(old_umask)
    assert len(set(tokens)) == len(tokens), "a token was reused"
    assert sorted(p.name for p in (data_dir / "board").iterdir()) == ["access-token"]


def test_write_access_token_tightens_and_replaces_an_earlier_file(data_dir):
    path = token_file(data_dir)
    path.parent.mkdir()
    path.write_text("old-token-0000000000000000000000\n", "utf-8")
    path.chmod(0o644)

    token = server_module().write_access_token()

    assert path.read_text("utf-8").strip() == token != "old-token-0000000000000000000000"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_access_token_writes_a_given_token_as_it_is(data_dir):
    path = token_file(data_dir)
    path.parent.mkdir()
    path.write_text("old-token-0000000000000000000000\n", "utf-8")
    path.chmod(0o644)
    old_umask = os.umask(0)
    try:
        given = ["Zq-_0aB9xY-_kLmN0pQrStUvWx-_12Ab", fresh_token(), fresh_token()]
        for token in given:
            assert server_module().write_access_token(token) == token

            assert path.read_text("utf-8").strip() == token
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)
        fresh = server_module().write_access_token(None)
    finally:
        os.umask(old_umask)
    assert TOKEN_RE.fullmatch(fresh) and fresh not in given, fresh
    assert path.read_text("utf-8").strip() == fresh
    assert sorted(p.name for p in path.parent.iterdir()) == ["access-token"]


def test_the_token_file_is_never_written_inside_the_repo(monkeypatch):
    inside = env.repo_root() / f".token-test-{secrets.token_hex(4)}"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(inside))
    try:
        with pytest.raises(paths.DataDirInsideRepo):
            server_module().write_access_token()
        assert not inside.exists()
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_a_board_folder_linked_into_the_repo_is_refused(data_dir):
    root = env.repo_root().resolve()
    target = root / f".token-link-test-{secrets.token_hex(4)}"
    target.mkdir()
    try:
        assert target.resolve().is_relative_to(root) and not data_dir.resolve().is_relative_to(root)
        (data_dir / "board").symlink_to(target, target_is_directory=True)

        write = server_module().write_access_token
        for call in (lambda: write(), lambda: write(fresh_token())):  # generated, then given
            with pytest.raises(paths.DataDirInsideRepo):
                call()

            assert sorted(p.name for p in target.iterdir()) == [], "a file was written in the repo"
    finally:
        shutil.rmtree(target, ignore_errors=True)
    assert not target.exists()


# ---------------------------------------------------------------- cloudflared's output

STAMP = "2026-11-28T00:41:07Z"


REGISTERED = (
    f"{STAMP} INF Registered tunnel connection connIndex=0 "
    "connection=0c5f3e8a-7d1b-4c2e-8f90-1a2b3c4d5e6f event=0 ip=198.41.200.13 "
    "location=nrt01 protocol=quic\n"
)
# a connection that does not come up: none of these lines is a registered connection
NOT_REGISTERED = (
    f'{STAMP} ERR Unable to establish connection with Cloudflare edge error="TLS handshake '
    'timeout" connIndex=0 event=0 ip=198.41.200.13\n'
    f'{STAMP} ERR Register tunnel error from server side error="Unauthorized: Failed to get '
    'tunnel" connIndex=0 event=0 ip=198.41.200.13\n'
    f"{STAMP} DBG Registered session sessionID=4f2a9c1e0b7d4e6a8c3f5b1d2e9a7c40\n"
    f"{STAMP} INF Unregistered tunnel connection connIndex=0 event=0 ip=198.41.200.13\n"
    f'{STAMP} ERR Serve tunnel error error="connection with edge closed" connIndex=0 event=0 '
    "ip=198.41.200.13\n"
    f"{STAMP} INF Retrying connection in up to 2s connIndex=0 event=0 ip=198.41.200.13\n"
)


def banner(
    name: str | None, port: int = 8765, *, registered: bool = True, protected: bool = False
) -> str:
    """cloudflared's quick-tunnel banner as it prints it on stderr, ending on its first registered
    connection unless `registered` is False; without the URL box when `name` is None; the
    protected-tunnel wording of the box when `protected`."""
    lines = [
        f"{STAMP} INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare "
        "account, is a quick way to experiment and try it out. However, be aware that these "
        "account-less Tunnels have no uptime guarantee, are subject to the Cloudflare Online "
        "Services Terms of Use (https://www.cloudflare.com/website-terms/), and Cloudflare "
        "reserves the right to investigate your use of Tunnels for violations of such terms. If "
        "you intend to use Tunnels in production you should use a pre-created named tunnel by "
        "following: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps",
        f"{STAMP} INF Requesting new quick Tunnel on trycloudflare.com...",
    ]
    if name is not None:
        visit = "Your protected quick Tunnel" if protected else "Your quick Tunnel"
        visit += " has been created! Visit it at (it may take some time to be reachable):"
        width = max(92, len(visit) + 4)
        url = f"https://{name}.trycloudflare.com"
        lines += [
            f"{STAMP} INF +{'-' * width}+",
            f"{STAMP} INF |  {visit:<{width - 2}}|",
            f"{STAMP} INF |  {url:<{width - 2}}|",
            f"{STAMP} INF +{'-' * width}+",
        ]
    lines += [
        f"{STAMP} INF Cannot determine default configuration path. No file [config.yml "
        "config.yaml] in [~/.cloudflared ~/.cloudflare-warp ~/cloudflare-warp /etc/cloudflared "
        "/usr/local/etc/cloudflared]",
        f"{STAMP} INF Version 2025.11.1 (Checksum 5d0b2c7e)",
        f"{STAMP} INF GOOS: darwin, GOVersion: go1.24.9, GoArch: arm64",
        f"{STAMP} INF Settings: map[ha-connections:1 no-autoupdate:true protocol:quic "
        f"url:http://127.0.0.1:{port}]",
        f"{STAMP} INF cloudflared will not automatically update when run from the shell. To "
        "enable auto-updates, run cloudflared as a service: "
        "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/run-tunnel/"
        "as-a-service/",
        f"{STAMP} INF Generated Connector ID: 6b0f4f4e-2d1c-4a8e-9a57-0e5d9f3c2b71",
        f"{STAMP} INF Initial protocol quic",
        f"{STAMP} INF ICMP proxy will use 192.0.2.10 as source for IPv4",
        f"{STAMP} INF Starting metrics server on 127.0.0.1:20241/metrics",
    ]
    return "\n".join(lines) + "\n" + (REGISTERED if registered else "")


OFFLINE = (
    f"{STAMP} INF Requesting new quick Tunnel on trycloudflare.com...\n"
    'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": dial tcp: '
    "lookup api.trycloudflare.com: no such host\n"
)
RATE_LIMITED = (
    f"{STAMP} INF Requesting new quick Tunnel on trycloudflare.com...\n"
    f"{STAMP} ERR Error unmarshaling QuickTunnel response: error code: 1015 "
    "error=\"invalid character 'e' looking for beginning of value\" "
    'status_code="429 Too Many Requests"\n'
    "failed to unmarshal quick Tunnel: invalid character 'e' looking for beginning of value\n"
)


def random_name() -> str:
    return f"brisk-{secrets.token_hex(3)}-meadow-{secrets.randbelow(90) + 10}"


@pytest.mark.parametrize(
    "name",
    ["seasonal-deck-organisms-sf", "ring-hd-stuffed-revision", "x7-q2-delta-9", "random"],
)
def test_parse_tunnel_url_finds_the_quick_tunnel_url(name):
    if name == "random":
        name = random_name()
    want = f"https://{name}.trycloudflare.com"
    parse = server_module().parse_tunnel_url

    for protected in (False, True):
        text = banner(name, port=secrets.randbelow(40000) + 20000, protected=protected)
        assert parse(text) == want, f"protected={protected}"
        assert parse(OFFLINE + text) == want, "api.trycloudflare.com is not the tunnel"
        only_box = [line for line in text.splitlines() if "trycloudflare.com" in line]
        assert parse("\n".join(only_box)) == want, f"protected={protected}"


@pytest.mark.parametrize(
    "output",
    [banner(None), OFFLINE, RATE_LIMITED, "", "https://trycloudflare.com\n"],
    ids=["banner-without-box", "offline", "rate-limited", "empty", "bare-domain"],
)
def test_parse_tunnel_url_is_none_without_a_tunnel_url(output):
    assert server_module().parse_tunnel_url(output) is None


# ---------------------------------------------------------------- a fake cloudflared

FAKE_SCRIPT = """\
import json, os, signal, sys, time

CONFIG = json.loads({config!r})
pid = os.getpid()
record = os.path.join(CONFIG["records"], str(pid) + ".json")
with open(record + ".tmp", "w") as fh:
    json.dump({{"argv": sys.argv[1:], "pid": pid}}, fh)
os.replace(record + ".tmp", record)
if CONFIG["ignore_term"]:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stderr.write(CONFIG["banner"])
sys.stderr.flush()
for kind, arg in CONFIG["later"]:
    if kind == "write":
        sys.stderr.write(arg)
        sys.stderr.flush()
    elif kind == "sleep":
        time.sleep(arg)
    elif kind == "mark":
        open(os.path.join(CONFIG["records"], str(pid) + "." + arg), "w").close()
if CONFIG["flood"]:
    line = ("2026-11-28T01:02:03Z INF Registered tunnel connection connIndex=1 event=0 "
            "ip=198.41.200.13 location=nrt01 protocol=quic\\n")
    written = 0
    while written < CONFIG["flood"]:
        sys.stderr.write(line)
        written += len(line)
    sys.stderr.flush()
    open(os.path.join(CONFIG["records"], str(pid) + ".drained"), "w").close()
if CONFIG["then"] == "exit":
    sys.exit(1)
time.sleep(CONFIG["life_s"])
"""


def alive(pid: int) -> bool:
    """The process runs (a zombie counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run([PS, "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@dataclass
class FakeCloudflared:
    bin_dir: Path
    records: Path
    seen: set[int] = field(default_factory=set)

    def install(
        self,
        banner: str,
        *,
        later: tuple[tuple[str, Any], ...] = (),
        then: str = "sleep",
        life_s: float = 20,
        ignore_term: bool = False,
        flood: int = 0,
    ) -> None:
        """`banner` on stderr at once, then the `later` steps in order: ("write", text),
        ("sleep", seconds) or ("mark", name), a file `marked(pid, name)` sees; then `then`."""
        config = {
            "records": str(self.records),
            "banner": banner,
            "later": [list(step) for step in later],
            "then": then,
            "life_s": life_s,
            "ignore_term": ignore_term,
            "flood": flood,
        }
        script = self.bin_dir.parent / "fake_cloudflared.py"
        script.write_text(FAKE_SCRIPT.format(config=json.dumps(config)), "utf-8")
        wrapper = self.bin_dir / "cloudflared"
        wrapper.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -S "
            f'{shlex.quote(str(script))} "$@"\n',
            "utf-8",
        )
        wrapper.chmod(0o755)

    def runs(self) -> list[dict]:
        out = []
        for p in sorted(self.records.glob("*.json")):
            run = json.loads(p.read_text("utf-8"))
            self.seen.add(run["pid"])
            out.append(run)
        return out

    def the_run(self, timeout: float = 10) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.runs():
            time.sleep(0.02)
        runs = self.runs()
        assert len(runs) == 1, f"want one cloudflared started, got {runs}"
        return runs[0]

    def drained(self, pid: int) -> bool:
        return self.marked(pid, "drained")

    def marked(self, pid: int, name: str) -> bool:
        return (self.records / f"{pid}.{name}").exists()


@pytest.fixture
def fake(tmp_path, monkeypatch) -> Iterator[FakeCloudflared]:
    """A fake `cloudflared`, alone on PATH with /usr/bin:/bin; whatever it left running is killed
    in teardown."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    records = tmp_path / "cloudflared-runs"
    records.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{SYSTEM_PATH}")
    fake = FakeCloudflared(bin_dir, records)
    yield fake
    fake.runs()
    for pid in fake.seen:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


PORTS = [pytest.param(18423, id="port-18423"), pytest.param(51007, id="port-51007")]


@pytest.mark.parametrize("port", PORTS)
def test_quick_tunnel_starts_cloudflared_and_yields_its_url(fake, port):
    name = random_name()
    fake.install(banner(name, port))

    with server_module().quick_tunnel(port, timeout_s=10) as url:
        assert url == f"https://{name}.trycloudflare.com"
        run = fake.the_run()
        assert run["argv"] == ["tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
        assert alive(run["pid"]), "the tunnel died while the server runs"

    assert gone_within(run["pid"], 5), "cloudflared outlived the server"


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_the_tunnel_is_stopped_when_the_server_fails(fake, error):
    fake.install(banner(random_name(), 18423))

    with pytest.raises(error):
        with server_module().quick_tunnel(18423, timeout_s=10):
            pid = fake.the_run()["pid"]
            assert alive(pid)
            raise error("the server stopped")

    assert gone_within(pid, 5), f"cloudflared outlived a {error.__name__}"


def test_a_child_that_ignores_sigterm_is_killed(fake):
    fake.install(banner(random_name(), 18423), ignore_term=True, life_s=25)

    with server_module().quick_tunnel(18423, timeout_s=10):
        pid = fake.the_run()["pid"]
        started = time.monotonic()

    assert time.monotonic() - started < 10, "stopping the tunnel waited on a stubborn child"
    assert gone_within(pid, 2)


def test_the_child_output_is_drained_after_the_url(fake):
    fake.install(banner(random_name(), 18423), flood=512 * 1024)

    with server_module().quick_tunnel(18423, timeout_s=10):
        pid = fake.the_run()["pid"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not fake.drained(pid):
            time.sleep(0.05)

        assert fake.drained(pid), "cloudflared blocked writing its log: nobody reads the pipe"
        assert alive(pid)


def test_no_url_within_the_timeout_is_a_tunnel_error(fake):
    board_server = server_module()
    fake.install(banner(None, 18423), life_s=20)
    started = time.monotonic()

    with pytest.raises(board_server.TunnelError) as caught:
        with board_server.quick_tunnel(18423, timeout_s=1.0):
            pytest.fail("no URL was printed, yet the tunnel was entered")

    assert time.monotonic() - started < 8
    assert "cloudflared" in str(caught.value).lower()
    assert gone_within(fake.the_run()["pid"], 5), "a child without a URL was left running"


@pytest.mark.parametrize("output", [OFFLINE, RATE_LIMITED], ids=["offline", "rate-limited"])
def test_cloudflared_exiting_without_a_url_is_a_tunnel_error(fake, output):
    board_server = server_module()
    fake.install(output, then="exit")
    started = time.monotonic()

    with pytest.raises(board_server.TunnelError) as caught:
        with board_server.quick_tunnel(18423, timeout_s=30):
            pytest.fail("cloudflared exited, yet the tunnel was entered")

    assert time.monotonic() - started < 10, "an exited child was waited on until the timeout"
    assert "cloudflared" in str(caught.value).lower()


AFTER_THE_URL = [
    pytest.param("", id="silent"),
    pytest.param(NOT_REGISTERED, id="failing-connection"),
]


@pytest.mark.parametrize("after", AFTER_THE_URL)
def test_cloudflared_exiting_after_its_url_but_before_a_connection_is_a_tunnel_error(fake, after):
    board_server = server_module()
    fake.install(banner(random_name(), 18423, registered=False) + after, then="exit")
    started = time.monotonic()

    with pytest.raises(board_server.TunnelError) as caught:
        with board_server.quick_tunnel(18423, timeout_s=30):
            pytest.fail(
                "cloudflared exited before a connection registered, yet the tunnel was entered"
            )

    assert time.monotonic() - started < 10, "an exited child was waited on until the timeout"
    assert "cloudflared" in str(caught.value).lower()
    assert gone_within(fake.the_run()["pid"], 5)


@pytest.mark.parametrize("after", AFTER_THE_URL)
def test_no_registered_connection_within_the_timeout_is_a_tunnel_error(fake, after):
    board_server = server_module()
    fake.install(banner(random_name(), 18423, registered=False) + after, life_s=20)
    started = time.monotonic()

    with pytest.raises(board_server.TunnelError) as caught:
        with board_server.quick_tunnel(18423, timeout_s=1.0):
            pytest.fail("no connection registered, yet the tunnel was entered")

    assert time.monotonic() - started < 8, "the wait for a connection outlasted timeout_s"
    assert "cloudflared" in str(caught.value).lower()
    assert gone_within(fake.the_run()["pid"], 5), "a child without a connection was left running"


@pytest.mark.parametrize("before", AFTER_THE_URL)
def test_the_url_is_yielded_only_once_a_connection_registered(fake, before):
    name = random_name()
    fake.install(
        banner(name, 18423, registered=False),
        later=(("write", before), ("sleep", 0.8), ("mark", "registering"), ("write", REGISTERED)),
    )

    with server_module().quick_tunnel(18423, timeout_s=10) as url:
        pid = fake.the_run()["pid"]
        assert fake.marked(pid, "registering"), "the URL came before a connection registered"
        assert url == f"https://{name}.trycloudflare.com"
        assert alive(pid)

    assert gone_within(pid, 5)


def test_cloudflared_missing_from_path_is_a_clear_error(tmp_path, monkeypatch):
    board_server = server_module()
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(board_server.TunnelError) as caught:
        with board_server.quick_tunnel(18423, timeout_s=5):
            pytest.fail("no cloudflared on PATH, yet the tunnel was entered")

    assert "cloudflared" in str(caught.value).lower()


# ---------------------------------------------------------------- the command, as a process

CLI = "from horsetrack.cli import app; app()"


class Command:
    """`horsetrack japan board-serve ...` as a child process, its output read as it comes."""

    def __init__(self, args: list[str], *, cwd: Path):
        process_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", CLI, "japan", "board-serve", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=process_env,
            cwd=cwd,
        )
        self.out: list[str] = []
        self.err: list[str] = []
        self.readers = [
            threading.Thread(target=self._read, args=(self.proc.stdout, self.out), daemon=True),
            threading.Thread(target=self._read, args=(self.proc.stderr, self.err), daemon=True),
        ]
        for reader in self.readers:
            reader.start()

    @staticmethod
    def _read(stream, into: list[str]) -> None:
        for line in stream:
            into.append(line)

    def text(self) -> str:
        return "".join(self.out) + "".join(self.err)

    def wait_for_line(self, wanted: Callable[[str], bool], timeout: float = 40) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in [*self.out, *self.err]:
                if wanted(line):
                    return line
            if self.proc.poll() is not None:
                time.sleep(0.2)  # let the readers catch the last lines
                pytest.fail(f"board-serve exited {self.proc.returncode}:\n{self.text()}")
            time.sleep(0.05)
        pytest.fail(f"no such line within {timeout} s:\n{self.text()}")

    def finish(self, timeout: float) -> int | None:
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        for reader in self.readers:
            reader.join(timeout=5)
        return code

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture
def command(tmp_path) -> Iterator[Callable[[list[str]], Command]]:
    started: list[Command] = []

    def start(args: list[str]) -> Command:
        cmd = Command(args, cwd=tmp_path)
        started.append(cmd)
        return cmd

    yield start
    for cmd in started:
        cmd.kill()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_board_serve_tunnel_prints_one_phone_url_and_stops_the_tunnel(
    fake, command, site, data_dir, sig
):
    name = random_name()
    host = f"{name}.trycloudflare.com"
    fake.install(banner(name), life_s=60)
    old = fresh_token()  # the last run's token, which this run replaces
    token_file(data_dir).parent.mkdir()
    token_file(data_dir).write_text(old + "\n", "utf-8")
    cmd = command(
        ["--date", DAY.isoformat(), "--port", "0", "--page", str(site / "board.html"), "--tunnel"]
    )

    line = cmd.wait_for_line(lambda text: host in text and "t=" in text)
    token = token_file(data_dir).read_text("utf-8").strip()  # written before the URL is printed
    assert TOKEN_RE.fullmatch(token) and token != old, token
    assert stat.S_IMODE(token_file(data_dir).stat().st_mode) == 0o600
    run = fake.the_run()
    target = re.fullmatch(
        r"http://127\.0\.0\.1:(\d+)", run["argv"][2] if len(run["argv"]) > 2 else ""
    )
    assert target, run["argv"]
    port = int(target.group(1))
    assert port > 0, "the tunnel must point at the bound port"
    assert run["argv"] == ["tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]

    phone = re.search(r"https://[^\s<>()\[\]'\"]+", line)
    assert phone, line
    parts = urlsplit(phone.group(0))
    assert (parts.scheme, parts.netloc) == ("https", host), phone.group(0)
    assert parts.path in ("", "/", "/shadow"), phone.group(0)
    assert parse_qs(parts.query) == {"t": [token]}, phone.group(0)

    first = request_at(port, f"{parts.path or '/'}?{parts.query}", {"X-Forwarded-Proto": "https"})
    assert first.status == 200
    cookie = the_cookie(first)
    assert "secure" in cookie.attrs and "httponly" in cookie.attrs
    assert request_at(port, "/shadow-feed", {"Cookie": cookie.pair}).status == 200
    assert request_at(port, "/shadow.js", {"Cookie": cookie.pair}).status == 200
    assert request_at(port, "/shadow").status == 404
    assert request_at(port, f"/shadow?t={fresh_token()}").status == 404

    cmd.proc.send_signal(sig)
    assert cmd.finish(timeout=15) is not None, f"board-serve did not stop on {sig.name}"
    assert gone_within(run["pid"], 5), f"cloudflared outlived board-serve after {sig.name}"

    output = cmd.text()
    holding = [text for text in output.splitlines() if token in text]
    assert len(holding) == 1, holding
    assert output.count(token) == 1, "the token is printed beyond the phone URL"
    assert phone.group(0) in holding[0]
    for p in data_dir.rglob("*"):
        if p.is_file() and p != token_file(data_dir):
            assert token.encode("ascii") not in p.read_bytes(), f"the token is in {p}"


def test_board_serve_tunnel_without_cloudflared_fails_clearly(
    command, site, data_dir, tmp_path, monkeypatch
):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    cmd = command(
        ["--date", DAY.isoformat(), "--port", "0", "--page", str(site / "board.html"), "--tunnel"]
    )

    code = cmd.finish(timeout=30)

    assert code is not None, "board-serve kept running without a tunnel"
    assert code != 0
    assert "cloudflared" in "".join(cmd.err).lower(), cmd.text()
    assert not token_file(data_dir).exists(), "a token file for a run that never served"


PHONE_URL = re.compile(r"trycloudflare\.com[^\s]*\?t=")


@pytest.mark.parametrize(
    "failure",
    ["port-taken", "no-cloudflared", "exits-before-url", "exits-before-a-connection"],
)
def test_a_failed_start_leaves_the_token_file_as_it_was(
    fake, command, site, data_dir, tmp_path, monkeypatch, failure
):
    path = token_file(data_dir)
    path.parent.mkdir()
    prior = (fresh_token() + "\n").encode("ascii")  # what an earlier run, still serving, wrote
    path.write_bytes(prior)
    path.chmod(0o600)
    port, holder = "0", None
    if failure == "port-taken":  # a second launch on a port the first one holds
        fake.install(banner(random_name()))
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = str(holder.getsockname()[1])
    elif failure == "no-cloudflared":
        empty = tmp_path / "empty-bin"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
    elif failure == "exits-before-url":
        fake.install(OFFLINE, then="exit")
    else:
        fake.install(banner(random_name(), registered=False) + NOT_REGISTERED, then="exit")
    try:
        cmd = command(
            [
                "--date",
                DAY.isoformat(),
                "--port",
                port,
                "--page",
                str(site / "board.html"),
                "--tunnel",
            ]
        )
        code = cmd.finish(timeout=20)
    finally:
        if holder is not None:
            holder.close()

    assert code is not None, f"board-serve kept running after a failed start:\n{cmd.text()}"
    assert code != 0, cmd.text()
    assert path.read_bytes() == prior, "a failed start replaced the token a running tunnel serves"
    assert sorted(p.name for p in path.parent.iterdir()) == ["access-token"]
    assert not PHONE_URL.search(cmd.text()), f"a phone URL for a tunnel that is down:\n{cmd.text()}"
    if failure == "port-taken":
        assert fake.runs() == [], "cloudflared started for a server that never bound"
    for run in fake.runs():
        assert gone_within(run["pid"], 5), "a cloudflared child outlived the failed start"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_board_serve_without_tunnel_needs_no_token(fake, command, site, data_dir):
    fake.install(banner(random_name()))
    port = free_port()
    cmd = command(
        ["--date", DAY.isoformat(), "--port", str(port), "--page", str(site / "board.html")]
    )

    deadline = time.monotonic() + 40
    reply = None
    while time.monotonic() < deadline and cmd.proc.poll() is None:
        try:
            reply = request_at(port, "/shadow")
            break
        except OSError:
            time.sleep(0.1)
    assert reply is not None, f"board-serve never answered:\n{cmd.text()}"

    expect_served("/shadow", reply, site)
    assert reply.cookies() == []
    assert reply.headers["X-Frame-Options"] == "DENY"
    expect_served("/shadow-feed", request_at(port, "/shadow-feed"), site)
    assert not token_file(data_dir).exists(), "a token file without --tunnel"
    assert fake.runs() == [], "cloudflared started without --tunnel"

    cmd.proc.send_signal(signal.SIGINT)
    assert cmd.finish(timeout=15) is not None
