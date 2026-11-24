"""`horsetrack scrape ...` (plan, run, status, expand, retry-failed), driven end to end through the
root CLI with typer's CliRunner. Ported from HKTrack tests/unit/ingest/test_cli_scrape.py (the same
16 tests), on a fake source with a fake adapter and a fake site. No network, no real sleeping.

Seams the implementation must honour (pinned here):

* `horsetrack.core.ingest.cli` is a module; the root app mounts its group as `scrape`, so
  `horsetrack scrape <command> --source/-s SOURCE ...` runs it.
* `cli.source_config(source) -> Mapping`: the source's config as a plain mapping (for jra, the
  parsed `countries/japan/config/jra.yaml`), looked up through the module attribute at call time,
  so a test can replace it. The CLI reads from it:
    - `http`: the block in the yaml shape (`default`, optional `hosts`, `circuit_breaker`,
      optional `quiet_hours`); `hosts[<host>]` sets that host's rate, `default` every other host;
    - `timezone`: the source's local zone ("today" for `plan` is local to the source);
    - `backfill.start_date` (a date or an ISO string): `plan`'s default `--start`.
* Adapters come from `horsetrack.core.ingest.adapters.get_adapter` (its `ADAPTERS` registry).
  With exactly one source registered, `--source` defaults to it.
* Options as in HKTrack: `plan --start/--end YYYY-MM-DD`; `run --max-requests N --max-minutes M
  --page-types a,b`; `expand --page-types`; `retry-failed --page-types --max-attempts`.
* Output is a few lines. `run` exits 0 on a clean stop, 2 when the circuit breaker opened, 1 on
  anything else; every error is one line starting with `ERROR`. A crash leaves its traceback in a
  `scrape-crash-*.log` under `$HORSETRACK_DATA_DIR/logs/` (any subfolder).
* `status`: one row per page type (`page_type` header, then pending fetched empty abandoned
  failed total), a `total` row, `%` done and an `ETA` at the `http.default` rate
  ("<x.x> h", "<r.r> req/s"); never more than 20 lines, extra page types folded into an `other`
  row.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path

import pytest
from fakes import BASE, SOURCE, FakeAdapter, FakeClock, FakeSite, make_task
from scrape_fakes import lines, offline_env, register_adapter
from typer.testing import CliRunner

from horsetrack.core.ingest import adapters
from horsetrack.core.ingest.queue import FetchQueue
from horsetrack.core.paths import manifest_path

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a red
# run reports.
from horsetrack.core.ingest import cli as cli_module

# isort: split
from horsetrack.cli import app

cli = CliRunner()


def _config(rps: float = 1.0) -> dict:
    """The fake source's config, in the shape of a country yaml (no user agent: it comes from
    the environment)."""
    return {
        "source": SOURCE,
        "timezone": "Asia/Tokyo",
        "backfill": {"start_date": date(2012, 3, 4)},
        "http": {
            "default": {"requests_per_second": rps, "jitter": 0.3},
            "circuit_breaker": {"stop_on_status": [403, 429], "max_consecutive_failures": 5},
        },
    }


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def small_site(site: FakeSite) -> FakeSite:
    """One fixture month -> two meetings -> three horses (one shared between meetings)."""
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:2026-09-16_R1\nmeeting:2026-09-20_R1\n")
    site.add(f"{BASE}/results/2026-09-16_R1", "MARKER\nhorse:H1\nhorse:H2\n")
    site.add(f"{BASE}/results/2026-09-20_R1", "MARKER\nhorse:H2\nhorse:H3\n")
    for horse in ("H1", "H2", "H3"):
        site.add(f"{BASE}/horse/{horse}", f"MARKER profile {horse}")
    return site


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def config() -> dict:
    """Mutable per test; the CLI sees a copy of its state at each call."""
    return _config()


@pytest.fixture(autouse=True)
def cli_env(tmp_path, monkeypatch, site, clock, adapter, config) -> Path:
    data = offline_env(monkeypatch, tmp_path, site, clock)
    register_adapter(monkeypatch, SOURCE, adapter)
    real = cli_module.source_config

    def fake_source_config(source: str):
        if source == SOURCE:
            return copy.deepcopy(config)
        return real(source)

    monkeypatch.setattr(cli_module, "source_config", fake_source_config)
    return data


def _invoke(*args: str):
    return cli.invoke(app, ["scrape", *args])


def _queue() -> FetchQueue:
    return FetchQueue(manifest_path(SOURCE))


def _plan_september():
    return _invoke("plan", "-s", SOURCE, "--start", "2026-09-01", "--end", "2026-09-30")


def test_plan_enqueues_fixture_tasks(adapter):
    result = _plan_september()
    assert result.exit_code == 0, result.output
    assert adapter.planned == [(date(2026, 9, 1), date(2026, 9, 30))]
    assert len(lines(result)) <= 3 and "1" in result.stdout
    with _queue() as q:
        assert q.counts() == {("fixture", "pending"): 1}

    # idempotent; a different range reaches the adapter as given
    adapter.months = ["2026-09", "2026-10"]
    result = _invoke("plan", "--source", SOURCE, "--start", "2026-08-15", "--end", "2026-10-31")
    assert result.exit_code == 0, result.output
    assert adapter.planned[-1] == (date(2026, 8, 15), date(2026, 10, 31))
    with _queue() as q:
        assert q.counts() == {("fixture", "pending"): 2}


def test_plan_defaults_come_from_config(adapter, config):
    result = _invoke("plan", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    ((start, end),) = adapter.planned
    assert start == date(2012, 3, 4)
    assert abs(end - date.today()) <= timedelta(days=1)

    config["backfill"]["start_date"] = "2015-06-07"  # an ISO string works as well as a yaml date
    result = _invoke("plan", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert adapter.planned[-1][0] == date(2015, 6, 7)

    del config["backfill"]
    result = _invoke("plan", "-s", SOURCE)
    assert result.exit_code != 0
    assert [ln.startswith("ERROR") for ln in lines(result)] == [True]
    assert len(adapter.planned) == 2


def test_plan_rejects_a_bad_date(adapter):
    result = _invoke("plan", "-s", SOURCE, "--start", "yesterday")
    assert result.exit_code != 0
    result = _invoke("plan", "-s", SOURCE, "--start", "2026-09-30", "--end", "2026-09-01")
    assert result.exit_code != 0
    assert adapter.planned == []
    if manifest_path(SOURCE).exists():
        with _queue() as q:
            assert q.counts() == {}


def test_run_clean_stop_exits_zero_with_a_short_summary(small_site):
    _plan_september()
    result = _invoke("run", "-s", SOURCE)

    assert result.exit_code == 0, result.output
    out = lines(result)
    assert 1 <= len(out) <= 8
    assert not any(ln.startswith("ERROR") for ln in out)
    assert "queue empty" in result.stdout
    assert "MARKER" not in result.stdout
    with _queue() as q:
        assert q.counts() == {
            ("fixture", "fetched"): 1,
            ("results", "fetched"): 2,
            ("horse", "fetched"): 3,
        }
    assert small_site.total_hits == 6


@pytest.mark.parametrize("status", [403, 429])
def test_run_circuit_open_exits_two_with_exactly_one_error_line(site, status):
    site.add(f"{BASE}/fixture/2026-09", "denied", status=status)
    _plan_september()
    result = _invoke("run", "-s", SOURCE)

    assert result.exit_code == 2, result.output
    out = lines(result)
    assert len(out) <= 8
    errors = [ln for ln in out if ln.startswith("ERROR")]
    assert len(errors) == 1 and str(status) in errors[0]
    assert site.total_hits == 1


def test_run_budget_and_page_type_options(small_site):
    _plan_september()

    result = _invoke("run", "-s", SOURCE, "--max-requests", "2")
    assert result.exit_code == 0, result.output
    assert "max_requests" in result.stdout
    assert small_site.total_hits == 2

    result = _invoke("run", "-s", SOURCE, "--page-types", "fixture, results", "--max-minutes", "5")
    assert result.exit_code == 0, result.output
    assert "queue empty" in result.stdout
    with _queue() as q:
        assert q.counts() == {
            ("fixture", "fetched"): 1,
            ("results", "fetched"): 2,
            ("horse", "pending"): 3,
        }
    assert small_site.total_hits == 3


def test_run_with_unknown_source_prints_one_error_line(site):
    result = _invoke("run", "-s", "nosuch")
    assert result.exit_code == 1, result.output
    assert [ln.startswith("ERROR") for ln in lines(result)] == [True]
    assert not manifest_path("nosuch").exists()
    assert site.total_hits == 0


def test_run_refuses_to_start_next_to_a_running_backfill(small_site):
    fcntl = pytest.importorskip("fcntl")
    _plan_september()
    lock = manifest_path(SOURCE).with_name("manifest.sqlite.lock")
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # what a live `scrape run` holds
        result = _invoke("run", "-s", SOURCE)
    assert result.exit_code == 1, result.output
    assert [ln.startswith("ERROR already running") for ln in lines(result)] == [True]
    assert small_site.total_hits == 0


def test_run_crash_prints_one_error_line_and_saves_the_traceback(small_site, monkeypatch, cli_env):
    _plan_september()

    def broken_mark(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(FetchQueue, "mark", broken_mark)
    result = _invoke("run", "-s", SOURCE)

    assert result.exit_code == 1, result.output
    out = lines(result)
    assert len(out) == 1 and out[0].startswith("ERROR") and "No space left" in out[0]
    (crash,) = (cli_env / "logs").rglob("scrape-crash-*.log")
    assert "Traceback" in crash.read_text()


@pytest.mark.parametrize(("rps", "eta"), [(1.0, "2.0 h"), (2.0, "1.0 h")])
def test_status_counts_totals_percent_and_eta(small_site, config, rps, eta):
    config["http"]["default"]["requests_per_second"] = rps
    _plan_september()
    _invoke("run", "-s", SOURCE, "--max-requests", "3")
    with _queue() as q:
        q.enqueue([make_task("sectional", f"s{i:05d}") for i in range(7197)])

    result = _invoke("status", "-s", SOURCE)

    assert result.exit_code == 0, result.output
    out = lines(result)
    assert len(out) <= 20
    text = result.stdout
    rows = {ln.split()[0]: ln.split()[1:] for ln in out}
    assert rows["page_type"] == ["pending", "fetched", "empty", "abandoned", "failed", "total"]
    assert rows["fixture"] == ["0", "1", "0", "0", "0", "1"]
    assert rows["results"] == ["0", "2", "0", "0", "0", "2"]
    assert rows["horse"] == ["3", "0", "0", "0", "0", "3"]
    assert rows["sectional"] == ["7197", "0", "0", "0", "0", "7197"]
    assert rows["total"] == ["7200", "3", "0", "0", "0", "7203"]
    assert "%" in text
    # 7200 pending at the configured default rate
    assert "ETA" in text and eta in text and f"{rps:.1f} req/s" in text


def test_status_stays_within_20_lines_with_many_page_types():
    with _queue() as q:
        q.enqueue([make_task(f"type{i:02d}", f"k{j}") for i in range(40) for j in range(i + 1)])
    result = _invoke("status", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    out = lines(result)
    assert len(out) <= 20
    total_row = next(ln for ln in out if ln.lower().startswith("total"))
    assert str(sum(range(1, 41))) in total_row.split()
    assert any("other" in ln for ln in out)


def test_status_without_a_manifest_is_a_hint_not_a_crash():
    result = _invoke("status", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert len(lines(result)) == 1 and "plan" in result.stdout
    assert not manifest_path(SOURCE).exists()


def test_source_defaults_to_the_only_registered_one(monkeypatch):
    # several registered (jra, hkjc and the fake): --source is needed
    assert len(adapters.ADAPTERS) > 1
    result = _invoke("status")
    assert result.exit_code != 0
    assert not manifest_path(SOURCE).exists()

    for other in [name for name in adapters.ADAPTERS if name != SOURCE]:
        monkeypatch.delitem(adapters.ADAPTERS, other)
    result = _invoke("status")
    assert result.exit_code == 0, result.output
    assert SOURCE in result.stdout


def test_retry_failed_resets_failed_tasks(site):
    site.add(f"{BASE}/results/k1", "gone", status=404)
    with _queue() as q:
        q.enqueue([make_task("results", "k1")])
    _invoke("run", "-s", SOURCE)
    with _queue() as q:
        assert q.counts() == {("results", "failed"): 1}

    result = _invoke("retry-failed", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert "1" in result.stdout and len(lines(result)) == 1
    with _queue() as q:
        assert q.counts() == {("results", "pending"): 1}


def test_expand_enqueues_missing_followups_offline(small_site, adapter):
    adapter.follow_horses = False
    _plan_september()
    _invoke("run", "-s", SOURCE)
    hits = small_site.total_hits
    with _queue() as q:
        assert ("horse", "pending") not in q.counts()

    adapter.follow_horses = True
    result = _invoke("expand", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert "3" in result.stdout and len(lines(result)) <= 2
    assert small_site.total_hits == hits
    with _queue() as q:
        assert q.counts()[("horse", "pending")] == 3


def test_nothing_is_written_outside_the_data_dir(small_site, cli_env, tmp_path):
    _plan_september()
    result = _invoke("run", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert manifest_path(SOURCE).is_relative_to(cli_env)
    assert list((cli_env / "logs").rglob("scrape-*.log"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data", "repo"]
    assert list((tmp_path / "repo").iterdir()) == []
