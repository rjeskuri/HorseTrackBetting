"""`horsetrack japan jrdb fetch` (ticket #718): JRDB's race-day member files, fetched nightly
with the owner's login into the layout `horsetrack japan jrdb build` reads.

Everything here is offline and synthetic. The member site is a fake behind httpx.MockTransport,
wired in through `Fetcher.__init__` (so a request made any other way meets the socket guard and
fails); the keychain is a fake `security` behind subprocess.Popen; time is one fake clock; every
archive is a zip or an lzh-like body of made-up bytes built here. No JRDB byte, record or value is
in this file.

Contract pinned here (the coder follows it). Module `horsetrack.countries.japan.jrdb.fetch`, its
command registered as `jrdb.command("fetch")` in `horsetrack.cli`.

CLI         `horsetrack japan jrdb fetch [--date YYYY-MM-DD] [--families KYI,CYB] [--dry-run]
            [--retry-missing] [--print-launchd]`.
            --date: the race day. Default: the next calendar day in Asia/Tokyo at the time of
            the run (the eve rule: a run at 20:00 JST fetches tomorrow's files).
            --families: comma-separated upper-case JRDB family codes. Default: the day-before
            pre-race set, holding at least KYI, CYB, CHA, KKA, KAB and BAC, as zip deliveries only.
            A date that does not parse or an unknown family code: non-zero exit, nothing sent,
            nothing written.
Seam        `fetch._utc_now()` returns the aware UTC now (the default date, the closed hours,
            fetched_utc, checked_utc, the recheck rule), looked up at call time. Monotonic time
            and sleeping are the Fetcher's `clock` and `sleep` (the tests inject fakes through
            `Fetcher.__init__`); construct the Fetcher per run.
URLs        https://jrdb.com. A zip family: /member/datazip/<Fam>/<yyyy>/<FAM><yymmdd>.zip (Fam is
            the code capitalised, Kyi; yyyy and yymmdd from the race date). An lzh-only family
            (SEC at least, as the owner's backfill holds it): /member/data/<Fam>/<FAM><yymmdd>.lzh.
Storage     raw/jrdb/<FAM>/<FAM><yymmdd>.zip, the bytes as served. Unpacking: every .txt member of
            a zip is extracted under its base name into the same folder
            (raw/jrdb/KYI/KYI261205.txt), and nothing is ever written outside that folder whatever
            the member names say. An .lzh is stored as served and not unpacked (no decoder); that
            is not a failure. Nothing else is written under raw/jrdb but manifest.json (no
            core.ingest gz cache copy of an archive). `tables.collect` reads the result, each
            delivery once, and no family folder the build reads ends up holding only .lzh (which
            would stop the build).
Manifest    raw/jrdb/manifest.json, one JSON object. Each fetched archive appends one entry to
            "files": {file (relative to raw/jrdb), url_path, size, sha256, fetched_utc,
            published_utc}. fetched_utc is the request time, ISO 8601 ending +00:00;
            published_utc is the response's Last-Modified the same way, or null/absent when the
            response has none. Existing entries stay unchanged and in order ahead of the new ones;
            every other key is kept unchanged. An archive is complete on disk before its entry is
            written; the manifest is replaced atomically. A manifest that does not read (empty,
            not JSON, not an object, "files" not a list): non-zero exit, nothing sent, the file
            left byte for byte.
Held        a file whose url_path or file is in "files", or whose archive is on disk, is not
            requested.
Missing     a 404 or 410: recorded in "missing" as {url_path, status, checked_utc}, named in the
            output, and the run goes on with the other files (exit 0 when nothing else failed). A
            later run asks for a missing file again only once at least MIN_RECHECK has passed
            since its checked_utc (1 h <= MIN_RECHECK <= 24 h, the coder's constant), or with
            --retry-missing; a missing file that is then fetched leaves "missing".
Stops       401, 403, 429, any 5xx, a transport error (refused connection, body cut short), or a
            200 whose body does not start like its format (zip PK\\x03\\x04; lzh -lh?- at offset
            2): that file is requested once (no retry), nothing more is sent, a line starts with
            ERROR and never echoes the response body, the exit is non-zero, and nothing partial is
            stored. Archives fetched before the stop stay stored and listed.
Politeness  every request through `horsetrack.core.ingest.fetcher.Fetcher`, with the User-Agent of
            HORSETRACK_USER_AGENT; request starts >= 2.0 s apart; no request from 08:00
            (inclusive) to 19:00 (exclusive) Asia/Tokyo on a Saturday or a Sunday, when JRDB closes
            downloads (carried over from HKTrack's downloader and pinned here): a run starting then
            sends nothing.
Credentials the user id is HORSETRACK_JRDB_USER and the password HORSETRACK_JRDB_PASSWORD, each
            read with `core.env.get_setting` (process environment, else the repo-root .env); blank
            is missing. Else the macOS keychain's generic-password item of service `jrdb` (account
            = the user id): `security find-generic-password -s jrdb` prints its attributes
            ("acct"), with -w the password on stdout, with -g the password on stderr as
            `password: "..."`. Call it on any platform; a FileNotFoundError or a non-zero exit
            means no keychain item. No credential value on any subprocess argv. A missing user id
            or password: non-zero exit, a message naming both HORSETRACK_JRDB_USER and
            HORSETRACK_JRDB_PASSWORD, nothing sent. HTTP Basic auth (user:password) goes only to
            https://jrdb.com, never to another host (a redirect). Neither value appears in the
            output, the logs, exception text, the manifest, any written file or any argv.
Dry run     --dry-run resolves the credentials (missing: as above), prints the files it would
            fetch, each with its URL path, sends no request and writes nothing.
LaunchAgent --print-launchd prints only a plist (plistlib), with no credential value, and sends and
            writes nothing (no credentials needed). ProgramArguments, the first an absolute path,
            run `... japan jrdb fetch` (no fixed date, no --dry-run) and then
            `... japan jrdb build`. StartCalendarInterval (one dict or a list of them): 20:00
            Asia/Tokyo expressed in the Mac's local zone at print time
            (`datetime.now().astimezone()`, TZ honoured), Minute 0.

Every expected value (URL paths, sha256, member bytes, times) is worked out here from the rule
above and the bytes this file serves, never by calling the code under test.
"""

from __future__ import annotations

import base64
import email.utils
import errno
import hashlib
import importlib
import io
import json
import logging
import os
import plistlib
import random
import re
import shlex
import socket
import subprocess
import time
import traceback
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
import pytest
from typer.testing import CliRunner, Result

from horsetrack.cli import app
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.countries.japan.jrdb import tables

MODULE = "horsetrack.countries.japan.jrdb.fetch"
HOST = "jrdb.com"
FOREIGN_HOST = "mirror.example.net"
LOOKALIKE_HOST = "jrdb.com.mirror.example.net"
TOKYO = ZoneInfo("Asia/Tokyo")
USER_AGENT = "horsetrack-tests/1.0 (jrdb fetch)"
USER_VAR, PASSWORD_VAR = "HORSETRACK_JRDB_USER", "HORSETRACK_JRDB_PASSWORD"
KEYCHAIN_SERVICE = "jrdb"
MIN_GAP_S = 2.0
PRE_RACE_SET = ("KYI", "CYB", "CHA", "KKA", "KAB", "BAC")
BUILD_FAMILIES = ["KYI", "CYB", "JOA", "TYB", "SED", "SKB", "UKC"]  # the families the build reads
LZH_FAMILIES = {"SEC"}
ECHO = "ECHO-7Q2-BODY"  # in every error body the fake site sends: the output must never show it
_REAL_SLEEP = time.sleep


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@dataclass(frozen=True)
class Creds:
    user: str
    password: str
    core: str  # a distinctive part of the password: a cut-down copy is a leak too

    @property
    def basic(self) -> str:
        return "Basic " + _b64(f"{self.user}:{self.password}")

    def forms(self) -> list[str]:
        """The ways the pair can leak: plain, its core, URL-quoted, base64, in a Basic header."""
        pw = self.password
        return [self.user, pw, self.core, quote(pw, safe=""), _b64(pw), self.basic[6:]]


# Distinctive values, each with a ':' (legal in a Basic password) and a '%' (must not be decoded).
ENV = Creds("tq-member-0417", "pX9:Wombat-7731%2Fq!", "Wombat-7731")
DOTENV = Creds("rk-member-2290", "Lq3:Quokka-4410%7Cz?", "Quokka-4410")
KEYCHAIN = Creds("vb-member-6613", "Zt8:Numbat-9052%3Dk~", "Numbat-9052")
DECOY = Creds("dc-member-5150", "dK2:Echidna-8086%5Ex^", "Echidna-8086")  # must never be used
ALL_CREDS = (ENV, DOTENV, KEYCHAIN, DECOY)
# Environment variables a careless implementation might read: they hold the decoy.
DECOY_ENV = {
    "JRDB_USER": DECOY.user,
    "JRDB_PASSWORD": DECOY.password,
    "JRDB_PASS": DECOY.password,
    "HKTRACK_JRDB_ACCOUNT": DECOY.user,
    "HORSETRACK_JRDB_PASS": DECOY.password,
    "HORSETRACK_JRDB_SECRET": DECOY.password,
}
DECOY_ITEM = ("jrdb-decoy", DECOY.user, DECOY.password)  # a keychain item of another service

FRI_EVE = datetime(2026, 12, 4, 20, 0, tzinfo=TOKYO).astimezone(timezone.utc)  # Fri 20:00 JST
RACE_DAY = date(2026, 12, 5)  # the Saturday after it


# ---------------------------------------------------------------- names and synthetic bytes


def stem(family: str, day: date) -> str:
    return f"{family}{day:%y%m%d}"


def zip_path(family: str, day: date) -> str:
    return f"/member/datazip/{family.capitalize()}/{day:%Y}/{stem(family, day)}.zip"


def lzh_path(family: str, day: date) -> str:
    return f"/member/data/{family.capitalize()}/{stem(family, day)}.lzh"


def url_path(family: str, day: date) -> str:
    return lzh_path(family, day) if family in LZH_FAMILIES else zip_path(family, day)


_ALPHABET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ "


def member_bytes(name: str) -> bytes:
    """Made-up fixed-width records, distinct per name and of varying size (not JRDB data)."""
    rng = random.Random(f"member:{name}")
    width, rows = 24 + rng.randrange(48), 2 + rng.randrange(30)
    lines = ("".join(rng.choice(_ALPHABET) for _ in range(width)) + "\r\n" for _ in range(rows))
    return "".join(lines).encode("ascii")


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buf.getvalue()


def lzh_bytes(name: str) -> bytes:
    """An LHA-like header (method id -lh5- at offset 2) and made-up bytes."""
    rng = random.Random(f"lzh:{name}")
    return b"\x24\x1f-lh5-" + bytes(rng.randrange(256) for _ in range(200 + rng.randrange(800)))


@dataclass
class Delivery:
    family: str
    day: date
    url_path: str
    file: str  # relative to raw/jrdb
    body: bytes
    members: dict[str, bytes]
    last_modified: datetime | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


_ZIP_URL = re.compile(r"/member/datazip/([A-Z][a-z]+)/([0-9]{4})/([A-Z]+)([0-9]{6})\.zip")
_LZH_URL = re.compile(r"/member/data/([A-Z][a-z]+)/([A-Z]+)([0-9]{6})\.lzh")


# ---------------------------------------------------------------- fakes


class FakeClock:
    """One fake time line: `monotonic` and `utc_now` move together, only when someone sleeps or a
    request takes its (fake) transfer time."""

    def __init__(self, start_utc: datetime) -> None:
        self.t0 = self.t = 5000.0
        self.start_utc = start_utc
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def utc_now(self) -> datetime:
        return self.start_utc + timedelta(seconds=self.t - self.t0)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def set_utc(self, utc: datetime) -> None:
        self.start_utc = utc - timedelta(seconds=self.t - self.t0)


@dataclass
class Hit:
    n: int
    mono: float
    utc: datetime
    scheme: str
    host: str
    path: str
    authorization: str | None
    user_agent: str | None
    header_lines: list[str]


class BrokenStream(httpx.SyncByteStream):
    """A body that stops half-way with a connection reset."""

    def __init__(self, head: bytes) -> None:
        self.head = head

    def __iter__(self) -> Iterator[bytes]:
        yield self.head
        raise httpx.ReadError("[Errno 54] Connection reset by peer")


def echo_body(status: int, request: httpx.Request) -> str:
    """An error page that echoes the credentials it was sent, as some servers do."""
    auth = request.headers.get("authorization", "")
    decoded = base64.b64decode(auth[6:]).decode("utf-8", "replace") if auth[:6] == "Basic " else ""
    return f"<html><body>{status} {ECHO}: you sent {auth} = {decoded}</body></html>"


Answer = Callable[[httpx.Request], httpx.Response]


def status_answer(status: int) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": "1"} if status in (429, 503) else {}
        if status == 401:
            headers["WWW-Authenticate"] = 'Basic realm="JRDB"'
        return httpx.Response(status, headers=headers, text=echo_body(status, request))

    return answer


def connect_refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


def login_page(request: httpx.Request) -> httpx.Response:
    page = f"<!DOCTYPE html><html><body>member login {ECHO}</body></html>"
    return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text=page)


def cut_body(body: bytes) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        headers = {"Content-Type": "application/octet-stream", "Content-Length": str(len(body))}
        return httpx.Response(200, headers=headers, stream=BrokenStream(body[: len(body) // 2]))

    return answer


def redirect_to(url: str) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": url})

    return answer


class FakeJrdb:
    """The member site on https://jrdb.com (and whatever foreign host a redirect names).

    jrdb.com: a forced answer for a path wins; else a request without the expected Basic header gets
    a 401 that echoes it; else a published delivery (or, with `publish_all`, any day-file URL of the
    pinned shape) is a 200; else a 404. Every request is recorded on the fake clock and takes
    `transfer_s` fake seconds."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.deliveries: dict[str, Delivery] = {}
        self.forced: dict[str, Answer] = {}
        self.foreign: dict[str, bytes] = {}  # path on any other host -> body served with a 200
        self.publish_all = False
        self.expected_auth = ENV.basic
        self.hits: list[Hit] = []
        self.transfer_s = 0.25

    def publish(
        self,
        family: str,
        day: date,
        *,
        last_modified: datetime | None = None,
        members: dict[str, bytes] | None = None,
        lzh: bool | None = None,
    ) -> Delivery:
        name = stem(family, day)
        if family in LZH_FAMILIES if lzh is None else lzh:
            path = lzh_path(family, day)
            found = Delivery(family, day, path, f"{family}/{name}.lzh", lzh_bytes(name), {})
        else:
            members = members if members is not None else {f"{name}.txt": member_bytes(name)}
            path = zip_path(family, day)
            found = Delivery(family, day, path, f"{family}/{name}.zip", zip_bytes(members), members)
        found.last_modified = last_modified
        self.deliveries[path] = found
        return found

    def _generated(self, path: str) -> Delivery | None:
        """Any day-file URL of the pinned shapes, zip or lzh, for any family."""
        if match := _ZIP_URL.fullmatch(path):
            folder, year, family, digits = match.groups()
        elif match := _LZH_URL.fullmatch(path):
            folder, family, digits = match.groups()
            year = None
        else:
            return None
        try:
            day = datetime.strptime(digits, "%y%m%d").date()
        except ValueError:
            return None
        if folder != family.capitalize() or (year is not None and year != f"{day:%Y}"):
            return None
        return self.publish(family, day, lzh=year is None)

    def handler(self, request: httpx.Request) -> httpx.Response:
        hit = Hit(
            n=len(self.hits) + 1,
            mono=self.clock.monotonic(),
            utc=self.clock.utc_now(),
            scheme=request.url.scheme,
            host=request.url.host,
            path=request.url.path,
            authorization=request.headers.get("authorization"),
            user_agent=request.headers.get("user-agent"),
            header_lines=[f"{k}: {v}" for k, v in request.headers.items()],
        )
        self.hits.append(hit)
        try:
            return self._answer(hit, request)
        finally:
            self.clock.advance(self.transfer_s)

    def _answer(self, hit: Hit, request: httpx.Request) -> httpx.Response:
        if hit.host != HOST:
            body = self.foreign.get(hit.path)
            if body is None:
                return httpx.Response(404, text="not found")
            return httpx.Response(200, headers={"Content-Type": "application/zip"}, content=body)
        if hit.path in self.forced:
            return self.forced[hit.path](request)
        if hit.authorization != self.expected_auth:
            headers = {"WWW-Authenticate": 'Basic realm="JRDB"'}
            return httpx.Response(401, headers=headers, text=echo_body(401, request))
        found = self.deliveries.get(hit.path)
        if found is None and self.publish_all:
            found = self._generated(hit.path)
        if found is None:
            return httpx.Response(404, text=f"<html>404 Not Found {ECHO}</html>")
        headers = {"Content-Type": "application/octet-stream"}
        if found.last_modified is not None:
            headers["Last-Modified"] = email.utils.format_datetime(found.last_modified, usegmt=True)
        return httpx.Response(200, headers=headers, content=found.body)


NOT_FOUND = (
    "security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.\n"
)


def flag_value(argv: list[str], flag: str) -> str | None:
    for i, token in enumerate(argv[:-1]):
        if token == flag:
            return argv[i + 1]
    return None


class FakeProcess:
    """Enough of a Popen for subprocess.run / check_output / communicate."""

    def __init__(self, argv: list[str], kwargs: dict[str, Any], code: int, out: str, err: str):
        self.args = argv
        self.pid = 4242
        self.returncode: int | None = None
        self._code = code
        text = any(kwargs.get(k) for k in ("text", "universal_newlines", "encoding", "errors"))

        def pipe(data: str, wanted: bool) -> io.IOBase | None:
            if not wanted:
                return None
            return io.StringIO(data) if text else io.BytesIO(data.encode("utf-8"))

        merged = kwargs.get("stderr") == subprocess.STDOUT
        self.stdin = None
        self.stdout = pipe(out + (err if merged else ""), kwargs.get("stdout") == subprocess.PIPE)
        self.stderr = pipe(err, kwargs.get("stderr") == subprocess.PIPE)

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[Any, Any]:
        self.returncode = self._code
        out = self.stdout.read() if self.stdout else None
        err = self.stderr.read() if self.stderr else None
        return out, err

    def poll(self) -> int:
        self.returncode = self._code
        return self._code

    def wait(self, timeout: float | None = None) -> int:
        return self.poll()

    def kill(self) -> None:
        pass

    terminate = kill

    def send_signal(self, sig: int) -> None:
        pass

    def __enter__(self) -> FakeProcess:
        return self

    def __exit__(self, *exc: object) -> bool:
        for stream in (self.stdout, self.stderr):
            if stream is not None:
                stream.close()
        return False


class FakeKeychain:
    """Stands in for macOS `security` (subprocess.Popen is replaced; the real keychain is never
    read). `items` are (service, account, password); a missing -s or -a matches any, as the real
    tool does. `failure`: "absent" (the tool is not installed) or "locked" (non-zero exit).
    `argvs` records the argv of every subprocess started during the test."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []
        self.failure: str | None = None
        self.argvs: list[list[str]] = []

    def popen(self, real_popen: Callable[..., Any]) -> Callable[..., Any]:
        keychain = self

        def fake_popen(args: Any, *a: Any, **kw: Any) -> Any:
            argv = shlex.split(args) if isinstance(args, str) else [os.fsdecode(x) for x in args]
            keychain.argvs.append(argv)
            if not argv or os.path.basename(argv[0]) != "security":
                return real_popen(args, *a, **kw)
            if keychain.failure == "absent":
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", argv[0])
            return FakeProcess(argv, kw, *keychain.answer(argv))

        return fake_popen

    def answer(self, argv: list[str]) -> tuple[int, str, str]:
        if self.failure == "locked":
            return 36, "", "security: SecKeychainSearchCopyNext: User interaction is not allowed.\n"
        if "find-generic-password" not in argv:
            return 1, "", "security: unexpected command in a test\n"
        service, account = flag_value(argv, "-s"), flag_value(argv, "-a")
        found = [
            (s, a, p) for s, a, p in self.items if service in (None, s) and account in (None, a)
        ]
        if not found:
            return 44, "", NOT_FOUND
        item_service, item_account, password = found[0]
        if "-w" in argv:
            return 0, password + "\n", ""
        out = (
            'keychain: "/private/var/tmp/test.keychain-db"\nversion: 512\nclass: "genp"\n'
            f'attributes:\n    0x00000007 <blob>="{item_service}"\n'
            f'    "acct"<blob>="{item_account}"\n    "svce"<blob>="{item_service}"\n'
        )
        return 0, out, (f'password: "{password}"\n' if "-g" in argv else "")


@dataclass
class World:
    tmp: Path
    root: Path  # the fake repo root (its .env is the only one read)
    data: Path
    clock: FakeClock
    site: FakeJrdb
    keychain: FakeKeychain
    caplog: pytest.LogCaptureFixture
    monkeypatch: pytest.MonkeyPatch
    extra_forms: list[str] = field(default_factory=list)

    @property
    def raw(self) -> Path:
        return self.data / "raw" / "jrdb"

    @property
    def manifest_path(self) -> Path:
        return self.raw / "manifest.json"

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def fetch(self, *args: str) -> Result:
        return CliRunner().invoke(app, ["japan", "jrdb", "fetch", *args])

    def hits_on(self, path: str) -> list[Hit]:
        return [h for h in self.site.hits if h.host == HOST and h.path == path]

    def set_env(self, user: str | None, password: str | None) -> None:
        for var, value in ((USER_VAR, user), (PASSWORD_VAR, password)):
            if value is None:
                self.monkeypatch.delenv(var, raising=False)
            else:
                self.monkeypatch.setenv(var, value)

    def write_dotenv(self, user: str | None, password: str | None) -> None:
        lines = ["# synthetic .env for a test"]
        lines += [f"{USER_VAR}={user}"] if user is not None else []
        lines += [f"{PASSWORD_VAR}={password}"] if password is not None else []
        (self.root / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_sleep_forbidden(seconds: float) -> None:
    if seconds > 0.05:
        raise AssertionError(f"real time.sleep({seconds}): pace requests on the Fetcher's sleep")
    _REAL_SLEEP(seconds)


def _network_forbidden(*args: object, **kwargs: object) -> None:
    raise RuntimeError("real network access in an offline test (a request outside the Fetcher?)")


@pytest.fixture
def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> World:
    module = importlib.import_module(MODULE)  # red until #718 lands: the module does not exist yet

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)  # never the real .env
    data = tmp_path / "data"
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    monkeypatch.setenv("HORSETRACK_USER_AGENT", USER_AGENT)
    monkeypatch.setenv(USER_VAR, ENV.user)
    monkeypatch.setenv(PASSWORD_VAR, ENV.password)
    for name, value in DECOY_ENV.items():
        monkeypatch.setenv(name, value)

    clock = FakeClock(FRI_EVE)
    monkeypatch.setattr(module, "_utc_now", clock.utc_now)  # raising=True: the seam must exist
    monkeypatch.setattr(time, "sleep", _real_sleep_forbidden)

    site = FakeJrdb(clock)
    original = Fetcher.__init__

    def offline_init(self: Fetcher, source: str, *, http: Any, **kwargs: Any) -> None:
        kwargs.update(
            transport=httpx.MockTransport(site.handler), sleep=clock.sleep, clock=clock.monotonic
        )
        original(self, source, http=http, **kwargs)

    monkeypatch.setattr(Fetcher, "__init__", offline_init)

    keychain = FakeKeychain()
    monkeypatch.setattr(subprocess, "Popen", keychain.popen(subprocess.Popen))
    shim = tmp_path / "bin"  # belt and braces: a `security` on PATH that never reads a keychain
    shim.mkdir()
    (shim / "security").write_text(
        "#!/bin/sh\necho 'stub security: Popen was bypassed' >&2\nexit 97\n"
    )
    (shim / "security").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ.get('PATH', '')}")

    for target, attr in (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    ):
        monkeypatch.setattr(target, attr, _network_forbidden)
    caplog.set_level(logging.DEBUG)
    return World(tmp_path, root, data, clock, site, keychain, caplog, monkeypatch)


@pytest.fixture
def local_zone() -> Iterator[Callable[[str], None]]:
    """Sets the process's local time zone (TZ + tzset) and restores it afterwards."""
    saved = os.environ.get("TZ")

    def set_zone(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield set_zone
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


# ---------------------------------------------------------------- checks


def files_under(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def error_lines(result: Result) -> list[str]:
    return [line for line in result.output.splitlines() if line.strip().startswith("ERROR")]


def assert_utc_near(text: Any, when: datetime, slack_s: float = 5.0) -> None:
    assert isinstance(text, str) and text.endswith("+00:00"), text
    parsed = datetime.fromisoformat(text)
    assert parsed.utcoffset() == timedelta(0), text
    assert abs((parsed - when).total_seconds()) <= slack_s, (text, when.isoformat())


def assert_entries_match_disk(world: World, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        path = world.raw / entry["file"]
        assert path.is_file(), f"the manifest lists {entry['file']}, which is not on disk"
        body = path.read_bytes()
        assert entry["size"] == len(body), entry["file"]
        assert entry["sha256"] == hashlib.sha256(body).hexdigest(), entry["file"]


def assert_no_leak(world: World, *results: Result) -> None:
    """No credential form in the output, exception text, logs, any subprocess argv or any file
    written under the test's tmp dir (but the .env and the `security` stub the test wrote)."""
    places: dict[str, str] = {"logs": world.caplog.text}
    for i, result in enumerate(results):
        places[f"output of run {i}"] = result.output
        exc = result.exception
        if exc is not None:
            text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            places[f"exception of run {i}"] = text
    for argv in world.keychain.argvs:
        places[f"argv of {os.path.basename(argv[0]) if argv else '?'}"] = "\0".join(argv)
    skip = {world.root / ".env", world.tmp / "bin" / "security"}
    for path in sorted(world.tmp.rglob("*")):
        if path.is_file() and path not in skip:
            places[str(path.relative_to(world.tmp))] = path.read_bytes().decode("latin-1")
    forms = [form for creds in ALL_CREDS for form in creds.forms()] + world.extra_forms
    for form in forms:
        for where, text in places.items():
            assert form not in text, f"a credential value ({form[:3]}...) appears in {where}"


def eve_of(day: date) -> datetime:
    """20:00 JST the day before `day`, in UTC."""
    return datetime.combine(day - timedelta(days=1), dtime(20, 0), TOKYO).astimezone(timezone.utc)


# ---------------------------------------------------------------- DoD 1: fetch, unpack, manifest


@pytest.mark.parametrize(
    ("day", "families"),
    [
        pytest.param(RACE_DAY, ("KYI", "CYB", "CHA"), id="2026-12-05"),
        pytest.param(date(2027, 1, 9), ("BAC", "KAB", "UKC", "KKA"), id="2027-01-09"),
    ],
)
def test_a_fetch_stores_unpacks_and_records_each_file(
    world: World, day: date, families: tuple[str, ...]
) -> None:
    world.clock.set_utc(eve_of(day))
    published = {}
    for i, family in enumerate(families):
        # Distinct Last-Modified times per family; the last family's response has none.
        when = None
        if i < len(families) - 1:
            hours, seconds = timedelta(hours=7 + i), timedelta(seconds=37 * i + 11)
            when = datetime.combine(day - timedelta(days=1), dtime(0, 0), timezone.utc) + hours
            when += seconds
        published[family] = world.site.publish(family, day, last_modified=when)

    result = world.fetch("--date", day.isoformat(), "--families", ",".join(families))

    assert result.exit_code == 0, result.output
    manifest = world.manifest()
    by_path = {entry["url_path"]: entry for entry in manifest["files"]}
    assert len(manifest["files"]) == len(families) == len(by_path)
    expected_files = {"manifest.json"}
    for family, delivery in published.items():
        hits = world.hits_on(delivery.url_path)
        assert len(hits) == 1, f"{delivery.url_path}: {len(hits)} requests"
        assert (world.raw / delivery.file).read_bytes() == delivery.body
        for name, data in delivery.members.items():
            assert (world.raw / family / name).read_bytes() == data, f"{name} not unpacked"
        entry = by_path[delivery.url_path]
        assert entry["file"] == delivery.file
        assert entry["size"] == len(delivery.body)
        assert entry["sha256"] == delivery.sha256
        assert_utc_near(entry["fetched_utc"], hits[0].utc)
        if delivery.last_modified is None:
            assert entry.get("published_utc") is None
        else:
            assert_utc_near(entry["published_utc"], delivery.last_modified, slack_s=0)
        expected_files |= {delivery.file} | {f"{family}/{name}" for name in delivery.members}
    assert files_under(world.raw) == expected_files

    collected = tables.collect(list(families))  # what `jrdb build` reads
    assert collected.lzh_only == []
    for family, delivery in published.items():
        found = collected.deliveries[family]
        assert [d.name for d in found] == [stem(family, day)]
        assert found[0].data == next(iter(delivery.members.values()))
    assert_no_leak(world, result)


def test_a_fetch_appends_to_the_manifest_and_keeps_every_other_entry_and_key(
    world: World,
) -> None:
    old_files = []
    for i in range(40):
        family = ("KYI", "SED", "CYB", "UKC")[i % 4]
        day = date(2026, 10, 1) + timedelta(days=i)
        old_files.append(
            {
                "file": f"{family}/{stem(family, day)}.zip",
                "url_path": zip_path(family, day),
                "size": 1000 + 17 * i,
                "sha256": hashlib.sha256(f"old-{i}".encode()).hexdigest(),
                "fetched_utc": f"2026-11-{10 + i % 15:02d}T14:30:{i % 60:02d}+00:00",
            }
        )
    old_files.append(
        {"file": "SEC/SEC261114.lzh", "url_path": "/member/data/Sec/SEC261114.lzh", "size": 77,
         "sha256": "ab" * 32, "fetched_utc": "2026-11-19T14:38:16+00:00"}
    )  # fmt: skip
    old = {
        "files": old_files,
        "missing": [
            {"url_path": zip_path("CHA", date(2026, 10, 3)), "status": 404},
            {"url_path": zip_path("KKA", date(2026, 10, 4)), "status": 410},
        ],
        "index": {
            "indexes": [
                {"url_path": "/member/datazip/Kyi/", "fetched_utc": "2026-11-19T14:36:47+00:00"}
            ],
            "archives": [f"https://{HOST}{zip_path('KYI', date(2026, 10, 1))}"],
        },
        "note": {"written_by": "an earlier backfill", "kept": True},
    }
    world.raw.mkdir(parents=True)
    world.manifest_path.write_text(json.dumps(old, indent=1), encoding="utf-8")
    days = (RACE_DAY, RACE_DAY + timedelta(days=1))
    published = [world.site.publish(f, d) for d in days for f in ("KYI", "CYB")]

    first = world.fetch("--date", days[0].isoformat(), "--families", "KYI,CYB")
    world.clock.set_utc(eve_of(days[1]))
    second = world.fetch("--date", days[1].isoformat(), "--families", "KYI,CYB")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    new = world.manifest()
    assert new["files"][: len(old_files)] == old_files  # unchanged, in order, ahead of the new ones
    added = new["files"][len(old_files) :]
    assert sorted(e["url_path"] for e in added) == sorted(d.url_path for d in published)
    assert_entries_match_disk(world, added)
    for entry in old["missing"]:
        assert entry in new["missing"]
    assert new["index"] == old["index"]
    assert new["note"] == old["note"]
    assert set(old) <= set(new)
    collected = tables.collect(["KYI", "CYB"])
    for family in ("KYI", "CYB"):
        assert [d.name for d in collected.deliveries[family]] == [stem(family, d) for d in days]
    assert_no_leak(world, first, second)


@pytest.mark.parametrize("held_by", ["an earlier run", "a manifest entry", "the file on disk"])
def test_a_file_already_held_is_never_requested_again(world: World, held_by: str) -> None:
    kyi = world.site.publish("KYI", RACE_DAY)
    cyb = world.site.publish("CYB", RACE_DAY)
    if held_by == "an earlier run":
        earlier = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI,CYB")
        assert earlier.exit_code == 0, earlier.output
        held_paths = [kyi.url_path, cyb.url_path]
    else:
        held = zip_bytes({f"{stem('KYI', RACE_DAY)}.txt": b"HELD COPY\r\n"})
        (world.raw / "KYI").mkdir(parents=True)
        (world.raw / kyi.file).write_bytes(held)
        if held_by == "a manifest entry":
            entry = {"file": kyi.file, "url_path": kyi.url_path, "size": len(held),
                     "sha256": hashlib.sha256(held).hexdigest(),
                     "fetched_utc": "2026-12-04T09:00:00+00:00"}  # fmt: skip
            world.manifest_path.write_text(json.dumps({"files": [entry], "missing": []}))
        held_paths = [kyi.url_path]
    before_hits = len(world.site.hits)
    before = {p: (world.raw / p).read_bytes() for p in (kyi.file,)}
    before_files = world.manifest()["files"] if world.manifest_path.exists() else []

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI,CYB")

    assert result.exit_code == 0, result.output
    new_hits = world.site.hits[before_hits:]
    assert not [h for h in new_hits if h.path in held_paths], "a held file was requested again"
    if held_by == "an earlier run":
        assert new_hits == []
        assert world.manifest()["files"] == before_files  # no duplicate entry
    else:
        assert [h.path for h in new_hits] == [cyb.url_path]
    assert {p: (world.raw / p).read_bytes() for p in before} == before  # not rewritten
    assert_no_leak(world, result)


def test_the_default_families_are_the_pre_race_zip_set_the_build_can_read(world: World) -> None:
    world.site.publish_all = True  # every day-file URL of the pinned shape is published

    result = world.fetch("--date", RACE_DAY.isoformat())

    assert result.exit_code == 0, result.output
    requested = {h.path for h in world.site.hits}
    for family in PRE_RACE_SET:
        assert zip_path(family, RACE_DAY) in requested, f"{family} is not in the default set"
    manifest = world.manifest()
    assert len(manifest["files"]) == len(world.site.hits), "a default request was not a day file"
    assert_entries_match_disk(world, manifest["files"])
    collected = tables.collect(BUILD_FAMILIES)
    assert collected.lzh_only == [], "a build family folder holds only .lzh: the build would stop"
    assert [d.name for d in collected.deliveries["KYI"]] == [stem("KYI", RACE_DAY)]
    assert_no_leak(world, result)


def test_an_lzh_only_family_is_stored_as_served_and_not_unpacked(world: World) -> None:
    sec = world.site.publish("SEC", RACE_DAY)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "SEC")

    assert result.exit_code == 0, result.output
    assert len(world.hits_on(lzh_path("SEC", RACE_DAY))) == 1
    assert (world.raw / sec.file).read_bytes() == sec.body
    (entry,) = world.manifest()["files"]
    assert (entry["file"], entry["url_path"]) == (sec.file, sec.url_path)
    assert (entry["size"], entry["sha256"]) == (len(sec.body), sec.sha256)
    assert files_under(world.raw) == {"manifest.json", sec.file}


def test_zip_member_names_cannot_write_outside_the_family_folder(world: World) -> None:
    members = {
        "../../escape-a.txt": b"ESCAPE A\r\n",
        "nested/../../../escape-b.txt": b"ESCAPE B\r\n",
        f"{stem('KYI', RACE_DAY)}.txt": member_bytes("KYI-trav"),
    }
    world.site.publish("KYI", RACE_DAY, members=members)

    world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI")

    written = files_under(world.tmp) - {"repo/.env", "bin/security"}
    outside = {p for p in written if not p.startswith("data/raw/jrdb/KYI/")}
    assert outside <= {"data/raw/jrdb/manifest.json"}, outside


# ---------------------------------------------------------------- missing files


@pytest.mark.parametrize("status", [404, 410])
def test_a_file_not_published_is_recorded_missing_and_the_others_still_come(
    world: World, status: int
) -> None:
    kyi, cha = world.site.publish("KYI", RACE_DAY), world.site.publish("CHA", RACE_DAY)
    cyb_path = zip_path("CYB", RACE_DAY)
    world.site.forced[cyb_path] = status_answer(status)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI,CYB,CHA")

    assert result.exit_code == 0, result.output
    for path in (kyi.url_path, cyb_path, cha.url_path):
        assert len(world.hits_on(path)) == 1, path
    manifest = world.manifest()
    assert sorted(e["url_path"] for e in manifest["files"]) == sorted([kyi.url_path, cha.url_path])
    assert_entries_match_disk(world, manifest["files"])
    (missing,) = [m for m in manifest["missing"] if m["url_path"] == cyb_path]
    assert missing["status"] == status
    assert_utc_near(missing["checked_utc"], world.hits_on(cyb_path)[0].utc)
    assert stem("CYB", RACE_DAY) in result.output
    assert ECHO not in result.output
    assert files_under(world.raw / "CYB") == set()
    assert_no_leak(world, result)


@pytest.mark.parametrize("asked_again", ["a day later", "with --retry-missing"])
def test_a_missing_file_is_asked_for_again_only_after_the_recheck_interval_or_on_request(
    world: World, asked_again: str
) -> None:
    world.site.publish("KYI", RACE_DAY)
    cha_path = zip_path("CHA", RACE_DAY)
    args = ["--date", RACE_DAY.isoformat(), "--families", "KYI,CHA"]
    first = world.fetch(*args)  # CHA is not published yet: 404
    assert first.exit_code == 0, first.output
    cha = world.site.publish("CHA", RACE_DAY)  # published since

    world.clock.set_utc(FRI_EVE + timedelta(minutes=30))
    soon = world.fetch(*args)
    assert soon.exit_code == 0, soon.output
    assert len(world.hits_on(cha_path)) == 1, "a missing file was asked for again within 30 min"
    assert not (world.raw / cha.file).exists()

    hits_before = len(world.site.hits)
    if asked_again == "a day later":
        world.clock.set_utc(FRI_EVE + timedelta(hours=25))
        later = world.fetch(*args)
    else:
        later = world.fetch(*args, "--retry-missing")

    assert later.exit_code == 0, later.output
    assert [h.path for h in world.site.hits[hits_before:]] == [cha_path]
    assert (world.raw / cha.file).read_bytes() == cha.body
    manifest = world.manifest()
    assert cha_path in {e["url_path"] for e in manifest["files"]}
    assert cha_path not in {m.get("url_path") for m in manifest["missing"]}
    assert_no_leak(world, first, soon, later)


# ---------------------------------------------------------------- stops: no retry, no partial file


STOPS = [
    pytest.param(status_answer(401), id="401"),
    pytest.param(status_answer(403), id="403"),
    pytest.param(status_answer(429), id="429"),
    pytest.param(status_answer(500), id="500"),
    pytest.param(status_answer(503), id="503"),
    pytest.param(connect_refused, id="connection refused"),
    pytest.param("cut", id="body cut short"),
    pytest.param(login_page, id="200 that is not a zip"),
]


@pytest.mark.parametrize("answer", STOPS)
def test_a_refusal_or_failure_stops_the_run_after_one_request(world: World, answer: Any) -> None:
    families = ("KYI", "CYB", "CHA")
    for family in families:
        delivery = world.site.publish(family, RACE_DAY)
        world.site.forced[delivery.url_path] = (
            cut_body(delivery.body) if answer == "cut" else answer
        )

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", ",".join(families))

    assert result.exit_code != 0, result.output
    assert len(world.site.hits) == 1, f"{len(world.site.hits)} requests: no retry, no next file"
    assert error_lines(result), result.output
    assert ECHO not in result.output, "the response body was echoed"
    assert files_under(world.raw) <= {"manifest.json"}, "a partial or refused file was kept"
    if world.manifest_path.exists():
        assert world.manifest()["files"] == []
    assert_no_leak(world, result)


def test_files_fetched_before_a_stop_stay_stored_and_listed(world: World) -> None:
    families = ("KYI", "CYB", "CHA", "KKA")
    published = {f: world.site.publish(f, RACE_DAY) for f in families}
    cha_path = published["CHA"].url_path
    world.site.forced[cha_path] = status_answer(503)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", ",".join(families))

    assert result.exit_code != 0, result.output
    assert error_lines(result), result.output
    paths = [h.path for h in world.site.hits]
    assert paths.count(cha_path) == 1
    assert paths[-1] == cha_path, "a request was sent after the 503"
    before = [f for f, d in published.items() if d.url_path in paths and f != "CHA"]
    manifest = world.manifest() if world.manifest_path.exists() else {"files": []}
    listed = {e["url_path"] for e in manifest["files"]}
    assert listed == {published[f].url_path for f in before}
    assert_entries_match_disk(world, manifest["files"])
    for family, delivery in published.items():
        stored = (world.raw / delivery.file).exists()
        assert stored == (family in before), family
    assert_no_leak(world, result)


@pytest.mark.parametrize(
    "body",
    [b"", b"{not json", b"[]", b'{"files": {}}', b'{"files": [{"file": "KYI/KYI261128.zip"'],
    ids=["empty", "not json", "a list", "files not a list", "cut short"],
)
def test_an_unreadable_manifest_is_left_alone_and_nothing_is_sent(
    world: World, body: bytes
) -> None:
    world.site.publish("KYI", RACE_DAY)
    world.raw.mkdir(parents=True)
    world.manifest_path.write_bytes(body)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI")

    assert result.exit_code != 0, result.output
    assert error_lines(result), result.output
    assert world.site.hits == []
    assert world.manifest_path.read_bytes() == body
    assert files_under(world.raw) == {"manifest.json"}


# ---------------------------------------------------------------- politeness through core.ingest


def test_requests_are_spaced_and_carry_the_ingest_user_agent(world: World) -> None:
    families = ("KYI", "CYB", "CHA", "KKA", "KAB")
    for family in families:
        world.site.publish(family, RACE_DAY)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", ",".join(families))

    assert result.exit_code == 0, result.output
    hits = world.site.hits
    assert len(hits) == len(families)
    gaps = [hits[i + 1].mono - hits[i].mono for i in range(len(hits) - 1)]
    assert min(gaps) >= MIN_GAP_S - 1e-6, gaps
    assert {h.user_agent for h in hits} == {USER_AGENT}
    assert {(h.scheme, h.host) for h in hits} == {("https", HOST)}


CLOSED_HOURS = [
    # (JST wall time of the run, requests expected for one file)
    pytest.param(datetime(2026, 12, 5, 7, 58, tzinfo=TOKYO), 1, id="Sat 07:58"),
    pytest.param(datetime(2026, 12, 5, 8, 0, tzinfo=TOKYO), 0, id="Sat 08:00"),
    pytest.param(datetime(2026, 12, 5, 13, 30, tzinfo=TOKYO), 0, id="Sat 13:30"),
    pytest.param(datetime(2026, 12, 5, 18, 59, tzinfo=TOKYO), 0, id="Sat 18:59"),
    pytest.param(datetime(2026, 12, 5, 19, 0, tzinfo=TOKYO), 1, id="Sat 19:00"),
    pytest.param(datetime(2026, 12, 6, 8, 0, tzinfo=TOKYO), 0, id="Sun 08:00"),
    pytest.param(datetime(2026, 12, 6, 21, 15, tzinfo=TOKYO), 1, id="Sun 21:15"),
    pytest.param(datetime(2026, 12, 4, 12, 0, tzinfo=TOKYO), 1, id="Fri 12:00"),
    pytest.param(datetime(2026, 12, 9, 12, 0, tzinfo=TOKYO), 1, id="Wed 12:00"),
]


@pytest.mark.parametrize(("local", "expected"), CLOSED_HOURS)
def test_no_request_in_jrdbs_closed_hours(world: World, local: datetime, expected: int) -> None:
    day = date(2026, 12, 6)
    kyi = world.site.publish("KYI", day)
    world.clock.set_utc(local.astimezone(timezone.utc))

    result = world.fetch("--date", day.isoformat(), "--families", "KYI")

    assert len(world.site.hits) == expected, result.output
    if expected:
        assert result.exit_code == 0, result.output
        assert (world.raw / kyi.file).read_bytes() == kyi.body
    else:
        assert files_under(world.raw / "KYI") == set()


# ---------------------------------------------------------------- DoD 2: credentials


CREDENTIAL_SOURCES = [
    pytest.param(
        (ENV.user, ENV.password), (DOTENV.user, DOTENV.password), [KEYCHAIN],
        (ENV.user, ENV.password), id="the environment first",
    ),
    pytest.param(
        (None, None), (DOTENV.user, DOTENV.password), [KEYCHAIN],
        (DOTENV.user, DOTENV.password), id="then the repo-root .env",
    ),
    pytest.param(
        (None, None), None, [KEYCHAIN], (KEYCHAIN.user, KEYCHAIN.password), id="then the keychain"
    ),
    pytest.param(
        (ENV.user, None), None, [Creds(ENV.user, KEYCHAIN.password, KEYCHAIN.core)],
        (ENV.user, KEYCHAIN.password), id="user from the environment, password from the keychain",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("env", "dotenv", "items", "expected"), CREDENTIAL_SOURCES)
def test_credentials_come_from_the_environment_the_env_file_or_the_keychain(
    world: World,
    env: tuple[str | None, str | None],
    dotenv: tuple[str, str] | None,
    items: list[Creds],
    expected: tuple[str, str],
) -> None:
    world.set_env(*env)
    if dotenv is not None:
        world.write_dotenv(*dotenv)
    # A keychain item of another service comes first: a lookup without `-s jrdb` finds it.
    world.keychain.items = [DECOY_ITEM] + [(KEYCHAIN_SERVICE, c.user, c.password) for c in items]
    want = Creds(expected[0], expected[1], expected[1][4:12])
    world.site.expected_auth = want.basic
    world.extra_forms = want.forms()
    kyi = world.site.publish("KYI", RACE_DAY)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI")

    assert result.exit_code == 0, result.output
    (hit,) = world.site.hits
    assert hit.authorization == want.basic
    assert (world.raw / kyi.file).read_bytes() == kyi.body
    assert_no_leak(world, result)  # also: no credential value on the `security` argv


MISSING_CREDENTIALS = [
    pytest.param((None, None), None, None, [], id="nowhere"),
    pytest.param((None, None), None, "absent", [], id="no security tool"),
    pytest.param((None, None), None, "locked", [], id="keychain locked"),
    pytest.param((ENV.user, None), None, None, [], id="user id only"),
    pytest.param((None, None), (None, DOTENV.password), None, [], id="password only, in .env"),
    pytest.param(("  ", ""), None, None, [], id="blank values"),
    pytest.param((None, None), None, None, ["--dry-run"], id="a dry run too"),
]


@pytest.mark.parametrize(("env", "dotenv", "failure", "extra"), MISSING_CREDENTIALS)
def test_a_missing_credential_names_both_variables_and_sends_nothing(
    world: World,
    env: tuple[str | None, str | None],
    dotenv: tuple[str | None, str | None] | None,
    failure: str | None,
    extra: list[str],
) -> None:
    world.set_env(*env)
    if dotenv is not None:
        world.write_dotenv(*dotenv)
    world.keychain.items = [DECOY_ITEM]
    world.keychain.failure = failure
    world.site.publish("KYI", RACE_DAY)

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI", *extra)

    assert result.exit_code != 0, result.output
    assert USER_VAR in result.output and PASSWORD_VAR in result.output, result.output
    assert world.site.hits == []
    assert files_under(world.data) == set()
    assert_no_leak(world, result)


def test_basic_auth_never_leaves_the_jrdb_origin(world: World) -> None:
    for family, host in (("KYI", FOREIGN_HOST), ("CYB", LOOKALIKE_HOST)):
        delivery = world.site.publish(family, RACE_DAY)
        world.site.forced[delivery.url_path] = redirect_to(f"https://{host}{delivery.url_path}")
        world.site.foreign[delivery.url_path] = delivery.body

    result = world.fetch("--date", RACE_DAY.isoformat(), "--families", "KYI,CYB")

    forms = [form for creds in ALL_CREDS for form in creds.forms()]
    for hit in world.site.hits:
        if hit.host != HOST:
            assert hit.authorization is None, f"credentials sent to {hit.host}"
            for line in hit.header_lines:
                assert not any(form in line for form in forms), f"a credential sent to {hit.host}"
    assert_no_leak(world, result)


# ---------------------------------------------------------------- DoD 3: the dry run


DRY_RUNS = [
    pytest.param(RACE_DAY, ("KYI", "CYB", "UKC"), id="2026-12-05 KYI,CYB,UKC"),
    pytest.param(date(2027, 1, 9), ("BAC", "KAB"), id="2027-01-09 BAC,KAB"),
    pytest.param(RACE_DAY, ("SEC",), id="an lzh family"),
]


@pytest.mark.parametrize(("day", "families"), DRY_RUNS)
def test_a_dry_run_lists_the_files_it_would_fetch_and_sends_and_writes_nothing(
    world: World, day: date, families: tuple[str, ...]
) -> None:
    world.clock.set_utc(eve_of(day))
    for family in families:
        world.site.publish(family, day)
    world.raw.mkdir(parents=True)
    world.manifest_path.write_text(json.dumps({"files": [], "missing": [], "index": {}}))
    before = tree(world.data)

    result = world.fetch("--dry-run", "--date", day.isoformat(), "--families", ",".join(families))

    assert result.exit_code == 0, result.output
    for family in families:
        assert url_path(family, day) in result.output, result.output
    assert world.site.hits == []
    assert tree(world.data) == before
    assert_no_leak(world, result)


@pytest.mark.parametrize(
    ("now_utc", "expected"),
    [
        pytest.param(datetime(2026, 12, 4, 11, 0, tzinfo=timezone.utc), date(2026, 12, 5),
                     id="Fri 20:00 JST"),
        pytest.param(datetime(2026, 12, 4, 16, 0, tzinfo=timezone.utc), date(2026, 12, 6),
                     id="Sat 01:00 JST, still Friday in UTC"),
        pytest.param(datetime(2026, 12, 31, 12, 0, tzinfo=timezone.utc), date(2027, 1, 1),
                     id="New Year's Eve 21:00 JST"),
    ],
)  # fmt: skip
def test_the_default_date_is_tomorrow_in_tokyo(
    world: World, now_utc: datetime, expected: date
) -> None:
    world.clock.set_utc(now_utc)

    result = world.fetch("--dry-run", "--families", "KYI")

    assert result.exit_code == 0, result.output
    assert zip_path("KYI", expected) in result.output, result.output
    for other in (expected - timedelta(days=1), expected + timedelta(days=1)):
        assert zip_path("KYI", other) not in result.output
    assert world.site.hits == []


@pytest.mark.parametrize(
    "args",
    [
        ["--date", "2026-13-01"],
        ["--date", "5 Dec 2026"],
        ["--date", RACE_DAY.isoformat(), "--families", "XYZ"],
        ["--date", RACE_DAY.isoformat(), "--families", "../KYI"],
        ["--date", RACE_DAY.isoformat(), "--families", "KYI/../../x"],
    ],
    ids=["no month 13", "not ISO", "unknown family", "a path", "a traversal"],
)
def test_bad_arguments_send_and_write_nothing(world: World, args: list[str]) -> None:
    world.site.publish_all = True

    result = world.fetch(*args)

    assert result.exit_code != 0, result.output
    assert world.site.hits == []
    assert files_under(world.data) == set()


# ---------------------------------------------------------------- the nightly LaunchAgent


@pytest.mark.parametrize(
    "zone", [pytest.param("Asia/Tokyo", id="Mac on JST"), pytest.param("UTC", id="Mac on UTC")]
)
def test_print_launchd_schedules_the_fetch_then_the_build_at_20_jst(
    world: World, local_zone: Callable[[str], None], zone: str
) -> None:
    local_zone(zone)
    expected = datetime.combine(RACE_DAY, dtime(20, 0), TOKYO).astimezone(ZoneInfo(zone))
    world.keychain.items = [(KEYCHAIN_SERVICE, KEYCHAIN.user, KEYCHAIN.password)]

    result = world.fetch("--print-launchd")

    assert result.exit_code == 0, result.output
    plist = plistlib.loads(result.stdout.encode("utf-8"))
    intervals = plist["StartCalendarInterval"]
    intervals = [intervals] if isinstance(intervals, dict) else intervals
    assert intervals, plist
    assert {(i["Hour"], i["Minute"]) for i in intervals} == {(expected.hour, 0)}
    args = plist["ProgramArguments"]
    assert all(isinstance(a, str) for a in args) and Path(args[0]).is_absolute(), args
    joined = " ".join(args)
    assert "jrdb fetch" in joined and "jrdb build" in joined, joined
    assert joined.index("jrdb fetch") < joined.index("jrdb build"), joined
    assert not re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", joined), "a fixed date in the job"
    assert "--dry-run" not in joined
    assert world.site.hits == []
    assert files_under(world.data) == set()

    world.set_env(None, None)  # no credentials anywhere: the plist still prints
    world.keychain.items = []
    bare = world.fetch("--print-launchd")
    assert bare.exit_code == 0, bare.output
    assert plistlib.loads(bare.stdout.encode("utf-8"))["ProgramArguments"] == args
    assert_no_leak(world, result, bare)


def test_fetch_is_registered_with_its_options() -> None:
    result = CliRunner().invoke(app, ["japan", "jrdb", "fetch", "--help"])

    assert result.exit_code == 0, result.output
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    for option in ("--date", "--families", "--dry-run", "--retry-missing", "--print-launchd"):
        assert option in text, option
