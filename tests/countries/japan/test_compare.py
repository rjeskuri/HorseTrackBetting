"""Comparing a day's odds snapshot log with HKTrack's (ticket #28).

Contract pinned here (`horsetrack.countries.japan.logger.compare`):
- `compare_logs(ours, theirs) -> Comparison` is pure: rows as `odds_log.read_rows` returns them, not
  mutated. HKTrack's known bugs are normalised first: their `source_update_utc` more than 1 h after
  their `snapshot_utc` is moved back one day; a `status_raw` other than "SCR" counts as null.
- `Comparison` attributes:
  - `races_ours`, `races_theirs`, `races_only_ours`, `races_only_theirs`: race ids (set-like).
  - `pools_ours`, `pools_theirs`, `pools_only_ours`, `pools_only_theirs`: `(race_id, pool)`
    (set-like); plain set differences, so a missing race's pools are listed too.
  - `snapshots_ours`, `snapshots_theirs`: `{(race_id, pool): distinct snapshot_utc count}`.
  - `compared`: their price rows (pool-total "__pool__" rows are not in the odds agreement).
  - `matched`: those with our row of the same race, pool, combination and (normalised)
    `source_update_utc` captured within 20 s (inclusive) of theirs; the nearest such row is used.
  - `equal`: matched rows whose odds, odds_hi (absent == null) and normalised status agree.
  - `mismatches`: the first 20 matched-but-unequal rows in their log order; each has `race_id`,
    `pool`, `combination`, `ours` and `theirs` (the two rows).
- `render(comparison) -> str` is markdown; it reports and gives no pass/fail verdict.
- `horsetrack japan compare-logs --date D --venue V --theirs PATH [--ours PATH] [--out PATH]`:
  ours defaults to the day's log under the data dir; the report goes to --out, else stdout; exit 0
  whatever the differences.

All rows are synthetic.
"""

# The new module is first-party once it exists; until then ruff cannot place it (I001).
from __future__ import annotations  # noqa: I001

import copy
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.logger.compare import Comparison, compare_logs, render

UTC = timezone.utc
DAY = "2026-11-28"
T0 = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)  # 14:00 in Tokyo
POOL = "__pool__"
MISSING = object()


def race(no: int, venue: str = "TOK") -> str:
    return f"jra:{DAY}:{venue}:{no:02d}"


def iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def price(
    combination: str,
    odds,
    *,
    at: datetime,
    pool: str = "WIN",
    race_no: int = 11,
    su=MISSING,
    status=MISSING,
    odds_hi=MISSING,
) -> dict:
    """A price row as the log holds it (ISO UTC text)."""
    row = {
        "snapshot_utc": iso(at),
        "race_id": race(race_no),
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": ("SCR" if odds is None else "14時00分現在") if status is MISSING else status,
        "source_update_utc": iso(at - timedelta(seconds=12) if su is MISSING else su),
    }
    if odds_hi is not MISSING:
        row["odds_hi"] = odds_hi
    return row


def total(pool: str, investment: float, *, at: datetime, race_no: int = 11) -> dict:
    return {
        "snapshot_utc": iso(at),
        "race_id": race(race_no),
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "investment": investment,
        "status_raw": None,
        "source_update_utc": iso(at - timedelta(seconds=12)),
    }


POOLS = {
    "WIN": ["1", "2", "3", "4"],
    "PLACE": ["1", "2", "3", "4"],
    "QUINELLA": ["1-2", "1-3", "2-3"],
    "QUINELLA_PLACE": ["1-2", "1-3", "2-3"],
}


def day(race_nos: list[int], captures: int, pools: dict[str, list[str]] = POOLS) -> list[dict]:
    """A synthetic day: every race and pool in every capture, one minute apart."""
    rows = []
    for c in range(captures):
        at = T0 + timedelta(minutes=c)
        for r in race_nos:
            for pool, combos in pools.items():
                for i, combo in enumerate(combos):
                    if pool == "WIN" and combo == "4" and r == race_nos[-1]:
                        rows.append(price(combo, None, at=at, pool=pool, race_no=r))
                        continue
                    odds = round(1.5 + r + i * 1.7 + c * 0.1, 1)
                    extra = {}
                    if pool in ("PLACE", "QUINELLA_PLACE"):
                        extra["odds_hi"] = round(odds + 0.8, 1)
                    rows.append(price(combo, odds, at=at, pool=pool, race_no=r, **extra))
                rows.append(total(pool, 1000.0 * (r + c + 1), at=at, race_no=r))
    return rows


def mismatch_keys(result) -> list[tuple[str, str, str]]:
    return [(m.race_id, m.pool, m.combination) for m in result.mismatches]


# ---------------------------------------------------------------- full agreement


@pytest.mark.parametrize("race_nos,captures", [([11], 1), ([1, 5, 11], 3), ([2, 3], 7)])
def test_identical_logs_agree_fully(race_nos, captures):
    rows = day(race_nos, captures)
    ours, theirs = copy.deepcopy(rows), copy.deepcopy(rows)
    result = compare_logs(ours, theirs)

    assert isinstance(result, Comparison)
    assert ours == rows and theirs == rows, "compare_logs does not mutate its inputs"

    races = {race(r) for r in race_nos}
    pools = {(race(r), p) for r in race_nos for p in POOLS}
    assert set(result.races_ours) == races and set(result.races_theirs) == races
    assert set(result.pools_ours) == pools and set(result.pools_theirs) == pools
    assert set(result.races_only_ours) == set() and set(result.races_only_theirs) == set()
    assert set(result.pools_only_ours) == set() and set(result.pools_only_theirs) == set()

    counts = {key: captures for key in pools}
    assert dict(result.snapshots_ours) == counts
    assert dict(result.snapshots_theirs) == counts

    price_rows = sum(1 for r in rows if r["combination"] != POOL)
    assert result.compared == price_rows
    assert result.matched == price_rows
    assert result.equal == price_rows
    assert list(result.mismatches) == []


# ---------------------------------------------------------------- HKTrack's bugs


def test_hktrack_day_off_source_update_is_normalised():
    # Early captures: 08:55 JST on race day is 23:55 UTC the day before. HKTrack dated the page's
    # printed time by the race day, so its source_update_utc lands about a day after its capture.
    snap = datetime(2026, 11, 27, 23, 55, tzinfo=UTC)
    day_ = timedelta(days=1)
    cases = [  # (combination, their source_update_utc, ours, matched)
        ("1", snap - timedelta(minutes=5) + day_, snap - timedelta(minutes=5), True),
        ("2", snap + timedelta(minutes=70), snap + timedelta(minutes=70) - day_, True),
        (
            "3",
            snap + timedelta(hours=23, minutes=59),
            snap + timedelta(hours=23, minutes=59) - day_,
            True,
        ),
        # not more than 1 h after the capture: left as it is
        ("4", snap + timedelta(minutes=50), snap + timedelta(minutes=50), True),
        ("5", snap + timedelta(hours=1), snap + timedelta(hours=1), True),
        ("6", snap - timedelta(seconds=12), snap - timedelta(seconds=12), True),
        ("7", None, None, True),
        # the shift is applied, not the source update ignored: ours equal to their raw value
        ("8", snap - timedelta(minutes=5) + day_, snap - timedelta(minutes=5) + day_, False),
        ("9", snap + timedelta(minutes=50), snap + timedelta(minutes=50) - day_, False),
    ]
    theirs, ours = [], []
    for i, (combo, their_su, our_su, _) in enumerate(cases):
        odds = 2.0 + i
        theirs.append(price(combo, odds, at=snap, su=their_su))
        ours.append(price(combo, odds, at=snap + timedelta(seconds=3), su=our_su, status=None))

    result = compare_logs(ours, theirs)
    expected = sum(1 for case in cases if case[3])
    assert result.compared == len(cases)
    assert result.matched == expected
    assert result.equal == expected
    assert list(result.mismatches) == []


def test_hktrack_time_text_status_is_not_a_mismatch():
    cases = [  # (combination, odds theirs, status theirs, odds ours, status ours, equal)
        ("1", 3.4, "14時00分現在", 3.4, None, True),
        ("2", 8.0, "9時05分現在", 8.0, None, True),
        ("3", 12.5, "最終オッズ", 12.5, None, True),
        ("4", 5.1, "", 5.1, None, True),
        ("5", 6.6, None, 6.6, None, True),
        ("6", None, "SCR", None, "SCR", True),
        # SCR is a status: it must agree
        ("7", None, "SCR", None, None, False),
        ("8", 4.2, "14時00分現在", 4.2, "SCR", False),
    ]
    theirs, ours = [], []
    for combo, t_odds, t_status, o_odds, o_status, _ in cases:
        theirs.append(price(combo, t_odds, at=T0, status=t_status))
        ours.append(price(combo, o_odds, at=T0 + timedelta(seconds=2), status=o_status))

    result = compare_logs(ours, theirs)
    assert result.compared == len(cases)
    assert result.matched == len(cases)
    assert result.equal == sum(1 for case in cases if case[5])
    assert sorted(m.combination for m in result.mismatches) == [c[0] for c in cases if not c[5]]


# ---------------------------------------------------------------- differences


def test_a_missing_race_and_pool_are_reported():
    three = {p: POOLS[p] for p in ("WIN", "PLACE", "QUINELLA")}
    theirs = day([10, 11], 2, pools=three) + day([12], 2, pools={"WIN": ["1", "2"], "PLACE": ["1"]})
    ours = [
        r
        for r in day([10, 11], 2, pools=three)
        if not (r["race_id"] == race(11) and r["pool"] == "QUINELLA")
    ]
    ours += day([10], 2, pools={"QUINELLA_PLACE": ["1-2"]}) + day([13], 1, pools={"WIN": ["1"]})

    result = compare_logs(ours, theirs)

    assert set(result.races_theirs) == {race(10), race(11), race(12)}
    assert set(result.races_ours) == {race(10), race(11), race(13)}
    assert set(result.races_only_theirs) == {race(12)}
    assert set(result.races_only_ours) == {race(13)}
    assert set(result.pools_only_theirs) == {
        (race(11), "QUINELLA"),
        (race(12), "WIN"),
        (race(12), "PLACE"),
    }
    assert set(result.pools_only_ours) == {(race(10), "QUINELLA_PLACE"), (race(13), "WIN")}

    # Rows we lack are unmatched, not mismatches.
    missing = sum(
        1
        for r in theirs
        if r["combination"] != POOL
        and (r["race_id"] == race(12) or (r["race_id"] == race(11) and r["pool"] == "QUINELLA"))
    )
    their_prices = sum(1 for r in theirs if r["combination"] != POOL)
    assert result.compared == their_prices
    assert result.matched == their_prices - missing
    assert result.equal == result.matched
    assert list(result.mismatches) == []


def test_a_changed_price_on_a_matched_row_is_a_mismatch():
    cases = [  # (pool, combination, theirs (odds, odds_hi), ours (odds, odds_hi), equal)
        ("WIN", "1", (3.4, MISSING), (3.5, MISSING), False),
        ("WIN", "2", (8.0, MISSING), (8.0, MISSING), True),
        ("WIN", "3", (None, MISSING), (6.1, MISSING), False),
        ("WIN", "5", (4.4, MISSING), (None, MISSING), False),
        ("PLACE", "1", (1.2, 1.5), (1.2, 1.6), False),
        ("PLACE", "2", (2.0, 2.9), (2.0, 2.9), True),
        ("PLACE", "3", (1.1, 1.4), (1.3, 1.4), False),
        ("PLACE", "4", (1.8, None), (1.8, MISSING), True),  # absent odds_hi reads as null
        ("QUINELLA", "1-2", (12.5, MISSING), (12.5, MISSING), True),
        ("QUINELLA", "2-3", (40.2, MISSING), (41.0, MISSING), False),
    ]
    theirs, ours = [], []
    for pool, combo, (t_odds, t_hi), (o_odds, o_hi), _ in cases:
        theirs.append(price(combo, t_odds, at=T0, pool=pool, odds_hi=t_hi, status=None))
        ours.append(
            price(combo, o_odds, at=T0 + timedelta(seconds=4), pool=pool, odds_hi=o_hi, status=None)
        )

    result = compare_logs(ours, theirs)
    unequal = [(race(11), pool, combo) for pool, combo, _, _, eq in cases if not eq]
    assert result.compared == len(cases)
    assert result.matched == len(cases)
    assert result.equal == len(cases) - len(unequal)
    assert mismatch_keys(result) == unequal, "in their log order"
    by_combo = {(m.pool, m.combination): m for m in result.mismatches}
    for pool, combo, (t_odds, _), (o_odds, _), eq in cases:
        if not eq:
            assert by_combo[(pool, combo)].theirs["odds"] == t_odds
            assert by_combo[(pool, combo)].ours["odds"] == o_odds


def test_only_the_first_20_mismatches_are_kept():
    n = 25
    theirs = [price(str(i), 2.0 + i, at=T0, status=None) for i in range(1, n + 1)]
    theirs.insert(3, price("99", 7.7, at=T0, status=None))  # equal, among the changed ones
    ours = [
        price(str(i), 2.5 + i, at=T0 + timedelta(seconds=1), status=None) for i in range(n, 0, -1)
    ]
    ours.append(price("99", 7.7, at=T0 + timedelta(seconds=1), status=None))

    result = compare_logs(ours, theirs)
    assert result.compared == n + 1
    assert result.matched == n + 1
    assert result.equal == 1
    assert [m.combination for m in result.mismatches] == [str(i) for i in range(1, 21)]


def test_rows_more_than_20_s_apart_are_not_matched():
    offsets = [0, 5, -12, 19, 20, -20, 21, -21, 45, -300, 3600]
    theirs, ours = [], []
    for i, off in enumerate(offsets):
        combo = str(i + 1)
        su = T0 - timedelta(seconds=30)  # the page has not changed between the two captures
        theirs.append(price(combo, 3.0 + i, at=T0, su=su, status=None))
        ours.append(price(combo, 3.0 + i, at=T0 + timedelta(seconds=off), su=su, status=None))

    # Several of our captures in the window: the nearest one is the match.
    su = T0 - timedelta(seconds=30)
    theirs.append(price("A", 5.0, at=T0, su=su, status=None))
    ours.append(price("A", 9.9, at=T0 + timedelta(seconds=15), su=su, status=None))
    ours.append(price("A", 5.0, at=T0 - timedelta(seconds=4), su=su, status=None))
    theirs.append(price("B", 6.0, at=T0, su=su, status=None))
    ours.append(price("B", 6.0, at=T0 - timedelta(seconds=18), su=su, status=None))
    ours.append(price("B", 6.5, at=T0 + timedelta(seconds=2), su=su, status=None))
    # Same capture time, other combination or pool: not a match.
    theirs.append(price("C", 4.0, at=T0, su=su, status=None))
    ours.append(price("C", 4.0, at=T0, su=su, status=None, pool="PLACE"))
    theirs.append(price("D", 4.0, at=T0, su=su, status=None))
    ours.append(price("D-1", 4.0, at=T0, su=su, status=None))

    result = compare_logs(ours, theirs)
    within = sum(1 for off in offsets if abs(off) <= 20)
    assert result.compared == len(offsets) + 4
    assert result.matched == within + 2
    assert result.equal == within + 1
    assert mismatch_keys(result) == [(race(11), "WIN", "B")]


def test_snapshot_counts_per_race_and_pool():
    def captures(race_no: int, pool: str, minutes: list[int], combos=("1", "2")) -> list[dict]:
        rows = []
        for m in minutes:
            at = T0 + timedelta(minutes=m)
            rows += [price(c, 2.0 + m, at=at, pool=pool, race_no=race_no) for c in combos]
            rows.append(total(pool, 500.0 + m, at=at, race_no=race_no))
        return rows

    ours = (
        captures(11, "WIN", [0, 1, 2])
        + captures(11, "QUINELLA", [0, 2], combos=("1-2", "1-3", "2-3"))
        + captures(12, "WIN", [5])
        + captures(12, "PLACE", [1, 2, 3, 4, 5, 6], combos=("1",))
    )
    theirs = (
        captures(11, "WIN", [0, 1, 2, 3])
        + captures(11, "QUINELLA", [0, 3], combos=("1-2",))
        + captures(12, "PLACE", [1, 2, 3, 4, 5], combos=("1", "2", "3"))
        + captures(10, "WIN", [9])
    )

    result = compare_logs(ours, theirs)
    assert dict(result.snapshots_ours) == {
        (race(11), "WIN"): 3,
        (race(11), "QUINELLA"): 2,
        (race(12), "WIN"): 1,
        (race(12), "PLACE"): 6,
    }
    assert dict(result.snapshots_theirs) == {
        (race(11), "WIN"): 4,
        (race(11), "QUINELLA"): 2,
        (race(12), "PLACE"): 5,
        (race(10), "WIN"): 1,
    }


# ---------------------------------------------------------------- the report and the command


def _differing_logs() -> tuple[list[dict], list[dict]]:
    theirs = day([3, 7], 3) + day([9], 2, pools={"WIN": ["1", "2"]})
    ours = [
        r for r in day([3, 7], 3) if not (r["race_id"] == race(7) and r["pool"] == "QUINELLA_PLACE")
    ]
    for r in ours:
        if r["race_id"] == race(3) and r["pool"] == "WIN" and r["combination"] == "1":
            r["odds"] = round(r["odds"] + 0.3, 1)
    return ours, theirs


def test_render_reports_counts_and_differences_without_a_verdict():
    ours, theirs = _differing_logs()
    result = compare_logs(ours, theirs)
    # Independent counts: 3 changed WIN "1" rows; QUINELLA_PLACE of race 7 and race 9 missing.
    their_prices = sum(1 for r in theirs if r["combination"] != POOL)
    missing = 3 * 3 + 2 * 2
    assert (result.compared, result.matched, result.equal) == (
        their_prices,
        their_prices - missing,
        their_prices - missing - 3,
    )

    text = render(result)
    assert isinstance(text, str) and text.strip()
    for figure in (their_prices, their_prices - missing, their_prices - missing - 3):
        assert re.search(rf"(?<![\d.]){figure}(?![\d.])", text), figure
    assert race(9) in text
    assert "QUINELLA_PLACE" in text
    assert not re.search(r"\b(pass|passed|fail|failed|verdict)\b", text, re.I)
    assert render(compare_logs(theirs, theirs)) != text


def test_compare_logs_command_reads_the_day_log_by_default(tmp_path, monkeypatch):
    from horsetrack.cli import app
    from horsetrack.core import paths

    data = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    ours, theirs = _differing_logs()
    default_ours = paths.odds_snapshot_path("jra", DAY, "TOK")
    assert odds_log.append_capture(default_ours, ours) == len(ours)
    their_path = tmp_path / "hktrack" / "odds_snapshots" / "jra" / DAY / "TOK.jsonl.gz"
    assert odds_log.append_capture(their_path, theirs) == len(theirs)
    their_bytes = their_path.read_bytes()
    out = tmp_path / "report" / "compare.md"
    out.parent.mkdir()

    runner = CliRunner()
    base = ["japan", "compare-logs", "--date", DAY, "--venue", "TOK", "--theirs", str(their_path)]
    result = runner.invoke(app, [*base, "--out", str(out)])
    assert result.exit_code == 0, result.output  # differences are reported, not judged
    report = out.read_text()
    assert race(9) in report and "QUINELLA_PLACE" in report
    assert their_path.read_bytes() == their_bytes, "their log is read only"

    # Without --out the report goes to stdout; --ours overrides the default.
    same = tmp_path / "copy" / "TOK.jsonl.gz"
    odds_log.append_capture(same, theirs)
    result = runner.invoke(app, [*base, "--ours", str(same)])
    assert result.exit_code == 0, result.output
    assert result.output.strip()
    assert result.output != report

    missing = runner.invoke(app, [*base[:-1], str(tmp_path / "nope.jsonl.gz")])
    assert missing.exit_code != 0


def test_compare_logs_reads_logs_through_odds_log(tmp_path: Path):
    """Rows round-tripped through the log compare as the in-memory rows do."""
    rows = day([4], 2)
    path = tmp_path / "TOK.jsonl.gz"
    odds_log.append_capture(path, rows)
    result = compare_logs(odds_log.read_rows(path), rows)
    price_rows = sum(1 for r in rows if r["combination"] != POOL)
    assert (result.compared, result.matched, result.equal) == (price_rows,) * 3
