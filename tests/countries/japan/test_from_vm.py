"""`--from-vm HOST` on `shadow-day` and `board-serve`: read the pulled VM day folder (ticket #236).

Contract pinned here:
- `horsetrack japan shadow-day ... --from-vm HOST` and `horsetrack japan board-serve ... --from-vm
  HOST` read every odds log of the day (venue discovery, log tailing, post-time sidecars, the
  board feed, the shadow feed and the feed's change fingerprint) from
  `$HORSETRACK_DATA_DIR/vm/<HOST>/odds_snapshots/jra/<date>/`, the layout the nightly pull keeps.
  Outputs stay on the Mac's normal paths (the shadow log, the shadow-day detail log). Nothing
  VM-pulled is written into the Mac's own `odds_snapshots/`, and nothing is written into `vm/`.
- HOST is validated like pull-vm's `--host`: an address (`1.2.3.4`, `user@host`) is refused with a
  non-zero exit before anything runs. Without `--from-vm` the read path is the Mac's own folder.
- board-serve is driven through the CLI here: `ThreadingHTTPServer.serve_forever` is replaced by a
  driver that serves the requests of a client thread, so no port is left open. The server
  module's `time.monotonic` is faked so every request re-checks the day folder.

All logs are synthetic, written under tmp_path (HORSETRACK_DATA_DIR); no network, no real rsync.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import types
from datetime import date, datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.shadow import day as shadow_day
from horsetrack.countries.japan.shadow import log as shadow_log
from horsetrack.countries.japan.shadow.log import ShadowLog, shadow_log_path

# isort: split
from shadow_fakes import IDS, Race, capture_rows, fake_models, figures

UTC = timezone.utc
DAY = date(2026, 11, 22)  # development days, before the holdout start
DAY2 = date(2026, 11, 23)
BASE = (1.25, 3.4, 6.2, 12.5, 38.0, 85.0)
WIN_KEYS = 6 + 6 + 15  # WIN, PLACE and Wide keys of a six-runner race
GRID = 6
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    monkeypatch.setattr(shadow_log, "is_holdout_day", lambda country, day: False)
    return d


def vm_folder(host: str, day: date) -> Path:
    return paths.data_dir() / "vm" / host / "odds_snapshots" / "jra" / day.isoformat()


def mac_folder(day: date) -> Path:
    return paths.data_dir() / "odds_snapshots" / "jra" / day.isoformat()


def post_of(day: date, hour: int = 1) -> datetime:
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC)


def race_at(day: date, venue: str, no: int, hour: int = 1) -> Race:
    return Race(day, venue, no, post_of(day, hour), BASE)


def write_day(folder: Path, races: list[Race], *, every_min: bool = True) -> None:
    """Captures a minute apart from T-75 to T+1 and the post-time sidecars, into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for venue in sorted({r.venue for r in races}):
        posts = {r.no: r.post for r in races if r.venue == venue}
        sidecar = folder / f"{venue}.races.json"
        odds_log.write_post_times(sidecar, posts, min(posts.values()) - timedelta(hours=3))
    for race in races:
        for m in range(75, -2, -1):
            at = race.post - timedelta(minutes=m, seconds=30)
            factor = 1.0 + 0.004 * (m % 23)
            odds_log.append_capture(
                folder / f"{race.venue}.jsonl.gz", capture_rows(race, at, factor)
            )


def listing(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }


def run_cli(*args: str):
    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", *args])


def plain(result) -> str:
    return ANSI.sub("", result.output)


# ------------------------------------------------------------------ shadow-day


def shadow_records(day: date) -> list[dict]:
    return ShadowLog(shadow_log_path(day), day=day).read()


@pytest.mark.parametrize("extra", [[], ["--venues", "TOK"]], ids=["discover", "venues"])
def test_shadow_day_reads_the_vm_folder_and_not_the_macs_own(data_dir, monkeypatch, extra):
    # the VM logged TOK race 1; the Mac's own folder holds a different TOK race and a KYO race
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    write_day(mac_folder(DAY), [race_at(DAY, "TOK", 5), race_at(DAY, "KYO", 1)])
    before_vm, before_mac = listing(data_dir / "vm"), listing(data_dir / "odds_snapshots")
    monkeypatch.setattr(shadow_day, "utc_now", lambda: post_of(DAY, 3))

    result = run_cli("shadow-day", "--date", DAY.isoformat(), "--once", "--from-vm", "vm1", *extra)

    assert result.exit_code == 0, plain(result)
    records = shadow_records(DAY)
    assert {r["race_id"] for r in records} == {f"jra:{DAY}:TOK:01"}
    assert len(records) == WIN_KEYS * GRID  # every grid point, every key, from the VM's rows
    assert {r["status"] for r in records} == {"ok"}  # the sidecar's post times came from the VM too
    assert shadow_log_path(DAY).is_relative_to(data_dir) and not shadow_log_path(
        DAY
    ).is_relative_to(data_dir / "vm")
    assert (paths.logs_dir("japan") / f"shadow-day-{DAY.isoformat()}.log").is_file()
    # nothing pulled is copied to the Mac's folder, nothing is written into the VM's
    assert listing(data_dir / "vm") == before_vm
    assert listing(data_dir / "odds_snapshots") == before_mac


def test_shadow_day_reads_each_hosts_own_folder(data_dir, monkeypatch):
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    write_day(vm_folder("vm2", DAY), [race_at(DAY, "KYO", 3), race_at(DAY, "KYO", 4, hour=2)])
    monkeypatch.setattr(shadow_day, "utc_now", lambda: post_of(DAY, 5))

    result = run_cli("shadow-day", "--date", DAY.isoformat(), "--once", "--from-vm", "vm2")

    assert result.exit_code == 0, plain(result)
    assert {r["race_id"] for r in shadow_records(DAY)} == {f"jra:{DAY}:KYO:03", f"jra:{DAY}:KYO:04"}


def test_shadow_day_without_from_vm_keeps_reading_the_macs_folder(data_dir, monkeypatch):
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    write_day(vm_folder("vm1", DAY2), [race_at(DAY2, "TOK", 1)])
    write_day(mac_folder(DAY2), [race_at(DAY2, "TOK", 5), race_at(DAY2, "KYO", 1)])
    monkeypatch.setattr(shadow_day, "utc_now", lambda: post_of(DAY2, 3))

    via_vm = run_cli("shadow-day", "--date", DAY.isoformat(), "--once", "--from-vm", "vm1")
    default = run_cli("shadow-day", "--date", DAY2.isoformat(), "--once")

    assert via_vm.exit_code == 0, plain(via_vm)  # the control: the option exists and works
    assert default.exit_code == 0, plain(default)
    assert {r["race_id"] for r in shadow_records(DAY)} == {f"jra:{DAY}:TOK:01"}
    assert {r["race_id"] for r in shadow_records(DAY2)} == {
        f"jra:{DAY2}:TOK:05",
        f"jra:{DAY2}:KYO:01",
    }


@pytest.mark.parametrize("host", ["1.2.3.4", "user@vm1", "192.168.0.10", "vm1:/etc", "-oProxy=x"])
def test_shadow_day_refuses_an_address_before_anything_runs(data_dir, monkeypatch, host):
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    monkeypatch.setattr(shadow_day, "utc_now", lambda: post_of(DAY, 3))

    result = run_cli("shadow-day", "--date", DAY.isoformat(), "--once", "--from-vm", host)

    assert result.exit_code != 0
    assert "no such option" not in plain(result).lower()
    assert not shadow_log_path(DAY).exists()
    assert not paths.logs_dir("japan").exists()


# ------------------------------------------------------------------ board-serve


def get(server, path: str) -> tuple[int, dict | None]:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, (json.loads(body) if resp.status == 200 else None)
    finally:
        conn.close()


@pytest.fixture
def serve_cli(monkeypatch, tmp_path):
    """`serve(args, steps, n)`: run `board-serve` on port 0; `steps(server)` makes `n` requests in
    a client thread while the server (in the main thread) handles them one by one."""
    from horsetrack.countries.japan.board import server as board_server

    page = tmp_path / "site" / "board.html"
    page.parent.mkdir()
    page.write_text("<!doctype html><title>board</title>\n", encoding="utf-8")
    ticks = iter(range(0, 10**9, 1000))
    monkeypatch.setattr(board_server, "time", types.SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(board_server, "load_shadow_models", lambda: fake_models())
    failures: list[BaseException] = []

    def serve(args: list[str], steps, n: int):
        def serve_forever(self, poll_interval=0.5):
            self.timeout = 10

            def client():
                try:
                    steps(self)
                except BaseException as exc:  # reported in the test thread
                    failures.append(exc)

            thread = threading.Thread(target=client, daemon=True)
            thread.start()
            for _ in range(n):
                self.handle_request()
            thread.join(timeout=20)

        monkeypatch.setattr(ThreadingHTTPServer, "serve_forever", serve_forever)
        result = run_cli("board-serve", "--port", "0", "--page", str(page), *args)
        assert not failures, failures
        return result

    return serve


def venues_of(feed: dict) -> list[str]:
    return [v["venue"] for v in feed["venues"]]


def win_odds(feed: dict, venue: str, race_no: int) -> dict[str, float]:
    for v in feed["venues"]:
        for race in v["races"]:
            if v["venue"] == venue and race["race_no"] == race_no:
                return {r["combination"]: r["odds"] for r in race["pools"]["WIN"]["rows"]}
    raise AssertionError(f"no {venue} race {race_no} on the board")


def test_board_serve_builds_both_feeds_from_the_vm_folder(data_dir, serve_cli):
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1), race_at(DAY, "TOK", 2, hour=2)])
    write_day(mac_folder(DAY), [race_at(DAY, "KYO", 7)])
    seen: dict[str, dict] = {}

    def steps(server):
        for name in ("/feed", "/shadow-feed"):
            status, body = get(server, name)
            assert status == 200, (name, status)
            seen[name] = body

    result = serve_cli(["--date", DAY.isoformat(), "--from-vm", "vm1"], steps, 2)

    assert result.exit_code == 0, plain(result)
    assert venues_of(seen["/feed"]) == ["TOK"]  # the VM's venue, not the Mac's KYO
    nos = [r["race_no"] for v in seen["/feed"]["venues"] for r in v["races"]]
    assert sorted(nos) == [1, 2]
    assert {r["race_id"] for r in seen["/shadow-feed"]["races"]} == {
        f"jra:{DAY}:TOK:01",
        f"jra:{DAY}:TOK:02",
    }
    assert seen["/shadow-feed"]["models"] == IDS


def test_board_serve_without_from_vm_keeps_reading_the_macs_folder(data_dir, serve_cli):
    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    write_day(mac_folder(DAY), [race_at(DAY, "KYO", 7)])
    seen: dict[str, dict] = {}

    def steps(server):
        seen["vm"] = get(server, "/feed")[1]

    def steps_default(server):
        seen["default"] = get(server, "/feed")[1]

    via_vm = serve_cli(["--date", DAY.isoformat(), "--from-vm", "vm1"], steps, 1)
    default = serve_cli(["--date", DAY.isoformat()], steps_default, 1)

    assert via_vm.exit_code == 0, plain(via_vm)  # the control: the option exists and works
    assert default.exit_code == 0, plain(default)
    assert venues_of(seen["vm"]) == ["TOK"]
    assert venues_of(seen["default"]) == ["KYO"]


def test_board_serve_notices_a_new_pull_into_the_vm_folder(data_dir, serve_cli):
    """The feed's change fingerprint watches the VM's day folder: a capture the loop pulls in
    after the first request shows up in the next feed, and the Mac's folder never moves."""
    race = race_at(DAY, "TOK", 1)
    write_day(vm_folder("vm1", DAY), [race])
    write_day(mac_folder(DAY), [race_at(DAY, "KYO", 7)])
    mac_before = listing(mac_folder(DAY))
    at = race.post + timedelta(minutes=3)
    seen: dict[str, dict] = {}

    def steps(server):
        seen["first"] = get(server, "/feed")[1]
        seen["shadow_first"] = get(server, "/shadow-feed")[1]
        odds_log.append_capture(vm_folder("vm1", DAY) / "TOK.jsonl.gz", capture_rows(race, at, 1.5))
        seen["second"] = get(server, "/feed")[1]
        seen["shadow_second"] = get(server, "/shadow-feed")[1]

    result = serve_cli(["--date", DAY.isoformat(), "--from-vm", "vm1"], steps, 4)

    assert result.exit_code == 0, plain(result)
    old, new = (figures(race, f)[0] for f in (1.0 + 0.004 * ((-1) % 23), 1.5))
    assert win_odds(seen["first"], "TOK", 1)["1"] == old[1]
    assert win_odds(seen["second"], "TOK", 1)["1"] == new[1]
    assert win_odds(seen["second"], "TOK", 1)["2"] == new[2]
    shadow_new = next(r for r in seen["shadow_second"]["races"] if r["race_id"] == race.race_id)
    shadow_old = next(r for r in seen["shadow_first"]["races"] if r["race_id"] == race.race_id)
    assert shadow_new["board_capture_utc"] != shadow_old["board_capture_utc"]
    assert listing(mac_folder(DAY)) == mac_before


@pytest.mark.parametrize("host", ["1.2.3.4", "user@vm1", "192.168.0.10", "vm1:/etc"])
def test_board_serve_refuses_an_address_before_serving(data_dir, monkeypatch, host):
    from horsetrack.countries.japan.board import server as board_server

    write_day(vm_folder("vm1", DAY), [race_at(DAY, "TOK", 1)])
    made: list[tuple] = []

    def refuse(*args, **kwargs):
        made.append((args, kwargs))
        raise RuntimeError("no server may start")

    monkeypatch.setattr(board_server, "make_server", refuse)

    result = run_cli("board-serve", "--port", "0", "--date", DAY.isoformat(), "--from-vm", host)

    assert result.exit_code != 0
    assert "no such option" not in plain(result).lower()
    assert made == []
