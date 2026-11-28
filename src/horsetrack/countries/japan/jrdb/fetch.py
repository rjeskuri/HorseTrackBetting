"""`horsetrack japan jrdb fetch` (ticket #718): one race day's JRDB member files, fetched with the
owner's login into raw/jrdb/<FAM>/, where `horsetrack japan jrdb build` reads them. Spec:
docs/specs/jrdb-fetch.md.

A zip family's day file is /member/datazip/<Fam>/<yyyy>/<FAM><yymmdd>.zip, stored as served with
its .txt members extracted beside it under their base names; an lzh-only family's (SEC) is
/member/data/<Fam>/<FAM><yymmdd>.lzh, stored as served. Every request goes through
`core.ingest.fetcher.Fetcher.request` (never the cache), one at a time, starts >= MIN_GAP_S apart,
no retry, Basic auth on https://jrdb.com only, none from 08:00 to 19:00 JST on a Saturday or
Sunday. A file held (a manifest entry by url_path or file, or the archive on disk) is never asked
for again; a 404/410 goes to the manifest's "missing" and is asked for again only after
MIN_RECHECK or with --retry-missing. Any other refusal or failure stops the run after that one
request. The manifest (raw/jrdb/manifest.json) is kept as parsed and rewritten atomically after
each change, every key and entry kept. The login: the account from HORSETRACK_JRDB_ACCOUNT
(environment, else the repo-root .env), the password only from the macOS keychain item of
service jrdb and that account.

Seam, looked up at call time: `_utc_now()`, the aware UTC now.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import plistlib
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import typer

from horsetrack.core import env, paths
from horsetrack.core.ingest.fetcher import CircuitOpen, Fetcher, FetchError, OriginAuth
from horsetrack.core.ingest.types import FetchTask

SOURCE = "jrdb"
ORIGIN = "https://jrdb.com"
TOKYO = ZoneInfo("Asia/Tokyo")
ACCOUNT_VAR = "HORSETRACK_JRDB_ACCOUNT"  # the password: only the keychain item
KEYCHAIN_SERVICE = "jrdb"
SECURITY = "/usr/bin/security"  # absolute: no `security` earlier on PATH sees the lookup
MIN_GAP_S = 2.0
MIN_RECHECK = timedelta(hours=1)
MISSING_STATUSES = (404, 410)
CLOSED_WEEKDAYS = (5, 6)  # Saturday, Sunday
CLOSED_HOURS = (dtime(8, 0), dtime(19, 0))  # start inclusive, end exclusive, Tokyo time
CLOSED = "Saturday and Sunday 08:00-19:00 JST"
# The eve set, fetched by default; then the zip families fetched only when named.
DEFAULT_FAMILIES = ("BAC", "KAB", "KYI", "CYB", "CHA", "KKA", "UKC")
ZIP_FAMILIES = (*DEFAULT_FAMILIES, "TYB", "SED", "SKB")
LZH_FAMILIES = ("SEC",)  # no zip delivery and no reader: stored as served
LABEL = "horsetrack.japan.jrdb-fetch"
EVE = dtime(20, 0)  # Tokyo time
EVE_WEEKDAYS = (4, 5, 6)  # Friday, Saturday, Sunday: the eves of Saturday, Sunday, Monday racing
EXIT_ERROR = 1

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MEMBER = re.compile(r"[A-Za-z0-9_-]+\.txt", re.IGNORECASE)
_NO_ACCOUNT = (
    f"no JRDB account: set {ACCOUNT_VAR} (environment or the repo-root .env) to the member "
    "account; nothing sent"
)
_NO_PASSWORD = (
    f"no JRDB password: no readable keychain item of service {KEYCHAIN_SERVICE} for the account "
    f"in {ACCOUNT_VAR} (no item, no `security` tool, or a locked keychain); add it with "
    f"`security add-generic-password -s {KEYCHAIN_SERVICE} -a <account> -w` (it prompts for the "
    "password); nothing sent"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Stop(Exception):
    """Ends the run with an ERROR line; the message never holds a credential or a body."""


@dataclass(frozen=True)
class Item:
    """One family's day file."""

    family: str
    day: date

    @property
    def name(self) -> str:
        return f"{self.family}{self.day:%y%m%d}"

    @property
    def is_zip(self) -> bool:
        return self.family not in LZH_FAMILIES

    @property
    def file(self) -> str:
        """Relative to raw/jrdb."""
        return f"{self.family}/{self.name}{'.zip' if self.is_zip else '.lzh'}"

    @property
    def url_path(self) -> str:
        folder = self.family.capitalize()
        if self.is_zip:
            return f"/member/datazip/{folder}/{self.day:%Y}/{self.name}.zip"
        return f"/member/data/{folder}/{self.name}.lzh"


def closed(now_utc: datetime) -> bool:
    local = now_utc.astimezone(TOKYO)
    return local.weekday() in CLOSED_WEEKDAYS and CLOSED_HOURS[0] <= local.time() < CLOSED_HOURS[1]


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")


def _jst(when: datetime) -> str:
    return f"{when.astimezone(TOKYO):%a %Y-%m-%d %H:%M} JST"


def _parse_utc(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        when = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _published(response: httpx.Response) -> str | None:
    """Last-Modified as ISO UTC, or None when absent or unreadable."""
    try:
        when = parsedate_to_datetime(response.headers.get("last-modified", ""))
    except (TypeError, ValueError, IndexError):
        return None
    return _iso(when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc))


# ------------------------------------------------------------------ credentials


def _keychain_password(account: str) -> str | None:
    """The password of the keychain item of service jrdb and `account`, or None (no tool, no item,
    a locked keychain). Both streams are captured and stdin is closed; only the one trailing
    newline of `security -w` is dropped. The password is on no argv."""
    argv = [SECURITY, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"]
    try:
        done = subprocess.run(argv, capture_output=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL, timeout=120, check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    secret = done.stdout[:-1] if done.stdout.endswith("\n") else done.stdout
    return secret or None


def credentials() -> tuple[str, str]:
    """(account, password): the account from HORSETRACK_JRDB_ACCOUNT (environment, else the .env;
    stripped, blank is missing), the password only from the keychain item of service jrdb and that
    account. Stop when either is missing; no password is read from a setting."""
    account = (env.get_setting(ACCOUNT_VAR) or "").strip()
    if not account:
        raise Stop(_NO_ACCOUNT)
    password = _keychain_password(account)
    if password is None:
        raise Stop(_NO_PASSWORD)
    return account, password


# ------------------------------------------------------------------ files and the manifest


def _write_atomic(path: Path, data: bytes) -> None:
    """A temp file beside `path`, fsynced, then os.replace: readers see the old file or the new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_manifest(path: Path) -> dict[str, Any]:
    """The manifest as parsed (a new one when there is none); Stop when it does not read."""
    if not path.exists():
        return {"files": [], "missing": []}
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        data = None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("files"), list)
        or not isinstance(data.get("missing", []), list)
    ):
        raise Stop(f"{path} is not a readable manifest (a JSON object whose \"files\" is a list): "
                   "fix or move it; nothing sent")  # fmt: skip
    data.setdefault("missing", [])
    return data


def _save(path: Path, manifest: dict[str, Any]) -> None:
    _write_atomic(path, json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))


def _set_missing(manifest: dict[str, Any], url_path: str, status: int | None, when: str) -> None:
    """Record `url_path` missing with `status` (one entry, updated in place), or with status None
    drop every entry of it."""
    kept, found = [], False
    for entry in manifest["missing"]:
        if isinstance(entry, dict) and entry.get("url_path") == url_path:
            if status is not None and not found:
                entry.update(status=status, checked_utc=when)
                kept.append(entry)
                found = True
            continue
        kept.append(entry)
    if status is not None and not found:
        kept.append({"url_path": url_path, "status": status, "checked_utc": when})
    manifest["missing"] = kept


def _held(item: Item, manifest: dict[str, Any], raw: Path) -> bool:
    entries = [e for e in manifest["files"] if isinstance(e, dict)]
    return (
        any(e.get("url_path") == item.url_path or e.get("file") == item.file for e in entries)
        or (raw / item.file).exists()
    )


def _recheck_after(item: Item, manifest: dict[str, Any], now: datetime) -> datetime | None:
    """When a missing `item` may be asked for again, if that is still ahead; else None."""
    checked = [
        _parse_utc(e.get("checked_utc"))
        for e in manifest["missing"]
        if isinstance(e, dict) and e.get("url_path") == item.url_path
    ]
    recent = [c for c in checked if c is not None and timedelta(0) <= now - c < MIN_RECHECK]
    return max(recent) + MIN_RECHECK if recent else None


def plan(
    items: list[Item], manifest: dict[str, Any], raw: Path, now: datetime, retry_missing: bool
) -> tuple[list[Item], list[str]]:
    """The items a run requests, and a line for each one it leaves out."""
    todo, skipped = [], []
    for item in items:
        if _held(item, manifest, raw):
            skipped.append(f"  held    {item.name}")
        elif not retry_missing and (after := _recheck_after(item, manifest, now)):
            skipped.append(f"  missing {item.name}: asked again after {_jst(after)} "
                           "or with --retry-missing")  # fmt: skip
        else:
            todo.append(item)
    return todo, skipped


# ------------------------------------------------------------------ one file


def _txt_members(body: bytes) -> list[tuple[str, bytes]]:
    """(base name, bytes) of each safe .txt member of a zip that opens and passes every CRC. A
    member whose name is absolute or climbs out ('..') stays in the zip, not extracted."""
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            if archive.testzip() is not None:
                raise zipfile.BadZipFile("a member fails its CRC")
            out, seen = [], set()
            for info in archive.infolist():
                parts = info.filename.replace("\\", "/").split("/")
                base = parts[-1]
                unsafe = parts[0] == "" or ".." in parts or ":" in info.filename
                if unsafe or not _MEMBER.fullmatch(base) or base.lower() in seen:
                    continue
                seen.add(base.lower())
                out.append((base, archive.read(info)))
            return out
    except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, NotImplementedError,
            ValueError, OSError):  # fmt: skip
        raise Stop("the body does not open as a zip (or fails a CRC)") from None


def _looks_like(item: Item, body: bytes) -> bool:
    """A zip starts with a local file header; an lzh has its method id (-lh?-) at offset 2."""
    if item.is_zip:
        return body[:4] == b"PK\x03\x04"
    return len(body) >= 7 and body[2:5] == b"-lh" and body[6:7] == b"-"


def _on_origin(url: str | None) -> bool:
    if not url:
        return True
    own, other = httpx.URL(ORIGIN), httpx.URL(url)
    return (other.scheme, other.host, other.port) == (own.scheme, own.host, own.port)


def _http(requests: int) -> dict[str, Any]:
    """The Fetcher settings: no retry, and a breaker that never counts the 404s of one run out
    (every other failure ends the run anyway)."""
    rate = {"requests_per_second": 1.0 / MIN_GAP_S, "jitter": 0.0}
    breaker = {"stop_on_status": [401, 403, 429], "max_consecutive_failures": requests + 1}
    return {"hosts": {"default": rate}, "max_retries": 0, "timeout_s": 60,
            "circuit_breaker": breaker}  # fmt: skip


def _get(fetcher: Fetcher, item: Item, auth: OriginAuth) -> httpx.Response | int:
    """The 200 response for `item`, or its missing status (404/410). Stop for anything else."""
    task = FetchTask(SOURCE, item.family, item.name, ORIGIN + item.url_path, item.day.year)
    where = f"GET {item.url_path}"
    off = f"{where}: redirected off {ORIGIN}; stopped"
    try:
        response = fetcher.request(task, auth=auth)
    except CircuitOpen as exc:
        if exc.host not in (None, httpx.URL(ORIGIN).host):
            raise Stop(off) from None
        raise Stop(f"{where}: HTTP {exc.http_status or '-'}; stopped, no retry") from None
    except FetchError as exc:
        if not _on_origin(exc.final_url):
            raise Stop(off) from None
        if exc.http_status in MISSING_STATUSES:
            return exc.http_status
        found = re.match(r"[A-Za-z_]+", str(exc))  # the exception type: never the server's words
        why = f"HTTP {exc.http_status}" if exc.http_status else (found[0] if found else "failed")
        raise Stop(f"{where}: {why}; stopped, no retry") from None
    if not _on_origin(str(response.url)):
        raise Stop(off)
    if response.status_code != 200 or not _looks_like(item, response.content):
        kind = "zip" if item.is_zip else "lzh"
        raise Stop(f"{where}: HTTP {response.status_code}, the body is not an {kind} archive "
                   "(a login or error page?); stopped")  # fmt: skip
    return response


def _store(
    item: Item, response: httpx.Response, raw: Path, fetched: datetime
) -> tuple[dict[str, Any], int]:
    """The manifest entry and the number of members unpacked. The members, then the archive, are
    on disk before the entry exists; a zip that does not open stores nothing."""
    body = response.content
    members = _txt_members(body) if item.is_zip else []
    for name, data in members:
        _write_atomic(raw / item.family / name, data)
    _write_atomic(raw / item.file, body)
    entry = {"file": item.file, "url_path": item.url_path, "size": len(body),
             "sha256": hashlib.sha256(body).hexdigest(), "fetched_utc": _iso(fetched),
             "published_utc": _published(response)}  # fmt: skip
    return entry, len(members)


def _fetch(
    todo: list[Item], manifest: dict[str, Any], manifest_path: Path, auth: OriginAuth
) -> tuple[int, int]:
    """Request each item in turn, saving the manifest after each answer; (fetched, missing)."""
    raw = manifest_path.parent
    fetched = missing = 0
    with Fetcher(SOURCE, http=_http(len(todo))) as fetcher:
        for item in todo:
            when = _utc_now()
            if closed(when):
                typer.echo(f"stopped at {_jst(when)}: JRDB's closed hours ({CLOSED}); the rest "
                           "after 19:00 JST")  # fmt: skip
                break
            try:
                got = _get(fetcher, item, auth)
                if isinstance(got, int):
                    _set_missing(manifest, item.url_path, got, _iso(when))
                    _save(manifest_path, manifest)
                    missing += 1
                    typer.echo(f"  missing {item.name}: HTTP {got} (asked again after "
                               f"{_jst(when + MIN_RECHECK)} or with --retry-missing)")  # fmt: skip
                    continue
                entry, members = _store(item, got, raw, when)
            except Stop as stop:
                raise Stop(f"{stop} ({fetched} fetched and {missing} missing before)") from None
            manifest["files"].append(entry)
            _set_missing(manifest, item.url_path, None, "")
            _save(manifest_path, manifest)
            fetched += 1
            typer.echo(f"  fetched {item.file} ({entry['size']} bytes, {members} members "
                       f"unpacked, published {entry['published_utc'] or 'unknown'})")  # fmt: skip
    return fetched, missing


def run(day: date, families: list[str], *, dry_run: bool, retry_missing: bool) -> None:
    """Fetch (or with dry_run list) the day files of `families` for `day`, echoing as it goes;
    Stop on the first refusal or failure."""
    now = _utc_now()
    if closed(now) and not dry_run:
        typer.echo(f"nothing sent: it is {_jst(now)}, inside JRDB's closed hours ({CLOSED}); "
                   "run again after 19:00 JST")  # fmt: skip
        return
    manifest_path = paths.raw_dir(SOURCE) / "manifest.json"
    manifest = load_manifest(manifest_path)
    auth = OriginAuth.basic(ORIGIN, *credentials())
    items = [Item(family, day) for family in families]
    todo, skipped = plan(items, manifest, manifest_path.parent, now, retry_missing)
    if dry_run:
        typer.echo(f"jrdb fetch {day}, dry run: {len(todo)} to fetch, {len(skipped)} left out; "
                   "nothing sent, nothing written")  # fmt: skip
        for item in todo:
            typer.echo(f"  fetch   {item.name}  {item.url_path}")
    else:
        typer.echo(f"jrdb fetch {day}: {len(todo)} to fetch, {len(skipped)} left out")
    for line in skipped:
        typer.echo(line)
    if dry_run:
        if closed(now):
            typer.echo(f"note: it is {_jst(now)}, inside JRDB's closed hours ({CLOSED}): "
                       "a real run now sends nothing")  # fmt: skip
        return
    fetched, missing = _fetch(todo, manifest, manifest_path, auth) if todo else (0, 0)
    typer.echo(f"jrdb fetch {day}: {fetched} fetched, {missing} missing; manifest {manifest_path}")


# ------------------------------------------------------------------ the LaunchAgent


def launch_agent() -> dict[str, Any]:
    """The nightly job: the fetch (tomorrow's files), then the build only when the fetch exits 0;
    at 20:00 Tokyo time on Friday, Saturday and Sunday, in the Mac's local zone now."""
    horsetrack = str(Path(sys.executable).parent / "horsetrack")
    fetch = shlex.join([horsetrack, "japan", "jrdb", "fetch"])
    build = shlex.join([horsetrack, "japan", "jrdb", "build"])
    local = datetime.now().astimezone()
    today = local.astimezone(TOKYO).date()
    intervals = []
    for day in (today + timedelta(days=n) for n in range(7)):
        if day.weekday() in EVE_WEEKDAYS:
            at = datetime.combine(day, EVE, TOKYO).astimezone(local.tzinfo)
            intervals.append({"Weekday": (at.weekday() + 1) % 7, "Hour": at.hour,
                              "Minute": at.minute})  # fmt: skip
    logs = paths.logs_dir("japan")
    return {
        "Label": LABEL,
        "ProgramArguments": ["/bin/sh", "-c", f"{fetch} && {build}"],
        "StartCalendarInterval": sorted(intervals, key=lambda i: i["Weekday"]),
        "RunAtLoad": False,
        "WorkingDirectory": str(env.repo_root()),
        "StandardOutPath": str(logs / "jrdb-fetch.out.log"),
        "StandardErrorPath": str(logs / "jrdb-fetch.err.log"),
    }


# ------------------------------------------------------------------ the command


def _families(text: str | None) -> list[str]:
    if text is None:
        return list(DEFAULT_FAMILIES)
    wanted = list(dict.fromkeys(f.strip().upper() for f in text.split(",") if f.strip()))
    known = (*ZIP_FAMILIES, *LZH_FAMILIES)
    if not wanted or any(f not in known for f in wanted):
        raise typer.BadParameter(f"known families: {', '.join(known)}", param_hint="--families")
    return wanted


def _day(text: str | None) -> date:
    if text is None:
        return _utc_now().astimezone(TOKYO).date() + timedelta(days=1)
    try:
        if not _DATE.fullmatch(text):
            raise ValueError(text)
        return date.fromisoformat(text)
    except ValueError:
        raise typer.BadParameter("a date as YYYY-MM-DD", param_hint="--date") from None


def fetch_command(
    race_date: str | None = typer.Option(
        None, "--date", help="Race day (YYYY-MM-DD); default: tomorrow in Asia/Tokyo."
    ),
    families: str | None = typer.Option(
        None,
        "--families",
        help=f"JRDB families, e.g. KYI,CYB; default: {','.join(DEFAULT_FAMILIES)}.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List the files a run would fetch; send and write nothing."
    ),
    retry_missing: bool = typer.Option(
        False, "--retry-missing", help="Ask again for files recorded missing, however recently."
    ),
    print_launchd: bool = typer.Option(
        False, "--print-launchd", help="Print the nightly LaunchAgent plist (fetch, then build)."
    ),
) -> None:
    """Fetch a race day's JRDB member files into raw/jrdb/<FAM>/ (unpacked) and the manifest."""
    try:
        if print_launchd:
            typer.echo(plistlib.dumps(launch_agent()).decode("utf-8"), nl=False)
            return
        run(_day(race_date), _families(families), dry_run=dry_run, retry_missing=retry_missing)
    except Stop as stop:
        typer.echo(f"ERROR {stop}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except (OSError, ValueError, RuntimeError) as exc:  # the data dir, a write, the User-Agent
        typer.echo(f"ERROR {type(exc).__name__}: {' '.join(str(exc).split())[:300]}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
