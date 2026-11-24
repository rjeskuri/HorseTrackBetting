"""The shadow feed: each race's shadow values at its bucket and its avoid flags (ticket #121).

Contract pinned here (`horsetrack.countries.japan.board.shadow_feed.build_shadow`):

`build_shadow(race_date, *, venues=None, now=utc_now, models, flag_rules, n_draws=400, seed=0,
cache=None) -> dict` (the ticket's signature plus `cache`, see below):
- Races are those logged in `odds_snapshots/jra/<date>/<VENUE>.jsonl.gz` (only `venues` when
  given); post times come from `post_times_for` (the `<VENUE>.races.json` sidecar here).
- Per race: the latest board as of now() (`board_at(rows, race_id, now)`: a row captured after
  now() is not on it), minutes_to_post = (post - now) in minutes, tau_bucket_min =
  `bucket_for(minutes_to_post, load_decision_grid())` (the shipped D2 grid 60, 30, 16, 5, 2, 1;
  0 at or after the post), and `shadow_values(board, tau_min=tau_bucket_min, ...)`.
- Feed: {"race_date", "generated_utc", "models": models.ids or None, "races": [{"race_id",
  "venue", "race_no", "post_utc", "minutes_to_post", "tau_bucket_min", "board_capture_utc",
  "pools": {"WIN": {"7": {"value_mean", "value_q05", "value_q95", "flags"}}, "PLACE": {"7":
  {"value_at_low", "value_at_high", "flags"}}, "QUINELLA_PLACE": {"3,7": {...}}}}]}. Every WIN
  runner, PLACE runner and Wide pair on the board has an entry; flags are the sorted list of
  `avoid_flags(board, rules=flag_rules, now_utc=now())` for the key ([] when none).
- Without models, every value is None, the flags are still there and "models" is None.
- `cache`: a mutable mapping the caller keeps between builds (the server's shadow feed keeps one).
  A race whose (race_id, board capture_utc, bucket) was computed into that mapping is not
  recomputed: shadow_values (and so the nowcast) is not called for it. Flags read the clock and are
  current on every build.
- No key anywhere in the feed names a stake, bet, Kelly, amount or wager.

The models are fakes (shadow_fakes.py): the fake nowcast records every call, the spy on the bucket
used and on the races computed. Expected values are worked out by hand from the synthetic prices.
Every row is synthetic and written under tmp_path; nothing waits on a real clock.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.board.snapshot import board_at

# isort: split
from shadow_fakes import (
    BANNED_KEY_WORDS,
    IDS,
    PLACE,
    WIDE,
    WIN,
    Race,
    SyntheticDay,
    all_keys,
    expected_values,
    fake_models,
    instant,
)

UTC = timezone.utc
DAY = date(2026, 11, 21)
T0 = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)  # 11:00 in Tokyo
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
N_DRAWS = 24
EIGHT = (1.25, 2.9, 4.6, 7.3, 11.8, 26.0, 64.0, 152.0)
SIX = (1.8, 3.4, 6.2, 12.5, 38.0, 85.0)
WIN_FIELDS = ("value_mean", "value_q05", "value_q95")
RANGE_FIELDS = ("value_at_low", "value_at_high")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def race(venue: str, no: int, minutes: float, base=EIGHT) -> Race:
    return Race(DAY, venue, no, T0 + timedelta(minutes=minutes), base)


def build(now: datetime, models, **options) -> dict:
    settings = dict(now=lambda: now, models=models, flag_rules=RULES, n_draws=N_DRAWS, seed=3)
    settings.update(options)
    return build_shadow(DAY, **settings)


def by_id(feed: dict) -> dict[str, dict]:
    return {r["race_id"]: r for r in feed["races"]}


def close(got, want) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


def expected_flags(day: SyntheticDay, r: Race, now: datetime) -> dict[tuple[str, str], list]:
    board = board_at(day.rows, r.race_id, now, post_utc=r.post)
    return {k: list(v) for k, v in avoid_flags(board, rules=RULES, now_utc=now).items()}


def assert_race(
    got: dict, day: SyntheticDay, r: Race, *, now, factor, tau, capture_at, valued=True
) -> None:
    """One race of the feed against the capture at `factor` read at bucket `tau`."""
    where = r.race_id
    assert got["venue"] == r.venue, where
    assert got["race_no"] == r.no, where
    assert instant(got["post_utc"]) == r.post, where
    assert got["tau_bucket_min"] == tau, where
    assert instant(got["board_capture_utc"]) == capture_at, where
    want = expected_values(r, factor, tau)
    flags = expected_flags(day, r, now)
    assert set(got["pools"]) == {WIN, PLACE, WIDE}, where
    for pool, entries in want.items():
        assert set(got["pools"][pool]) == set(entries), (where, pool)
        fields = WIN_FIELDS if pool == WIN else RANGE_FIELDS
        for combination, values in entries.items():
            entry = got["pools"][pool][combination]
            for name in fields:
                expected = values[name] if valued else None
                assert close(entry[name], expected), (where, pool, combination, name, entry)
            assert entry["flags"] == flags.get((pool, combination), []), (
                where,
                pool,
                combination,
            )


# ---------------------------------------------------------------- every race


def test_every_race_has_values_and_flags():
    tok1, tok2, kyo5 = race("TOK", 1, 3.5), race("TOK", 2, 25.0, SIX), race("KYO", 5, 70.0)
    day = SyntheticDay(DAY, [tok1, tok2, kyo5])
    for r in (tok1, tok2):
        day.capture(r, T0 - timedelta(minutes=10), 1.0)
        day.capture(r, T0 - timedelta(seconds=20), 1.03)
    day.capture(tok1, T0 + timedelta(seconds=30), 7.0)  # after now(): not on the board
    day.capture(kyo5, T0 - timedelta(minutes=3), 0.98)  # KYO's logger stopped: stale
    models = fake_models()

    feed = build(T0, models)

    assert feed["race_date"] == DAY.isoformat()
    assert feed["models"] == IDS
    races = by_id(feed)
    assert set(races) == {tok1.race_id, tok2.race_id, kyo5.race_id}
    latest = T0 - timedelta(seconds=20)
    assert_race(races[tok1.race_id], day, tok1, now=T0, factor=1.03, tau=5, capture_at=latest)
    assert_race(races[tok2.race_id], day, tok2, now=T0, factor=1.03, tau=30, capture_at=latest)
    assert_race(
        races[kyo5.race_id],
        day,
        kyo5,
        now=T0,
        factor=0.98,
        tau=60,
        capture_at=T0 - timedelta(minutes=3),
    )
    for r, minutes in ((tok1, 3.5), (tok2, 25.0), (kyo5, 70.0)):
        assert math.isclose(races[r.race_id]["minutes_to_post"], minutes, abs_tol=0.05)

    seen = {
        flag
        for r in races.values()
        for entries in r["pools"].values()
        for entry in entries.values()
        for flag in entry["flags"]
    }
    assert {"floor", "tail", "tail_member", "stale"} <= seen, seen  # the fixture exercises them
    kyo_flags = [e["flags"] for e in races[kyo5.race_id]["pools"][WIN].values()]
    assert all("stale" in f for f in kyo_flags)
    assert not any("stale" in e["flags"] for e in races[tok1.race_id]["pools"][WIN].values())

    only_kyo = build(T0, fake_models(), venues=["KYO"])
    assert [r["race_id"] for r in only_kyo["races"]] == [kyo5.race_id]


# ---------------------------------------------------------------- the bucket


BUCKETS = [  # (minutes to post, bucket on the D2 grid 60, 30, 16, 5, 2, 1)
    (95.0, 60),
    (60.0, 60),
    (45.0, 60),
    (30.0, 30),
    (17.0, 30),
    (16.0, 16),
    (5.5, 16),
    (3.5, 5),
    (2.25, 5),
    (2.0, 2),
    (1.5, 2),
    (0.25, 1),
]


def test_values_use_the_bucket_at_or_above_minutes_to_post():
    races = [race("NAK", no, minutes) for no, (minutes, _) in enumerate(BUCKETS, 1)]
    day = SyntheticDay(DAY, races)
    for r in races:
        day.capture(r, T0 - timedelta(minutes=4), 1.0)
        day.capture(r, T0 - timedelta(seconds=15), 0.96)
    models = fake_models()

    feed = build(T0, models)

    got = by_id(feed)
    assert set(got) == {r.race_id for r in races}
    for r, (minutes, tau) in zip(races, BUCKETS, strict=True):
        assert models.nowcast.taus(r.race_id) == {tau}, (minutes, models.nowcast.taus(r.race_id))
        assert math.isclose(got[r.race_id]["minutes_to_post"], minutes, abs_tol=0.05)
        assert_race(
            got[r.race_id],
            day,
            r,
            now=T0,
            factor=0.96,
            tau=tau,
            capture_at=T0 - timedelta(seconds=15),
        )


def test_after_the_post_the_bucket_is_zero():
    gone = race("HAN", 1, -0.5)  # post 30 s ago
    long_gone = race("HAN", 2, -12.0, SIX)
    upcoming = race("HAN", 3, 3.5)
    day = SyntheticDay(DAY, [gone, long_gone, upcoming])
    for r in (gone, long_gone):
        day.capture(r, r.post - timedelta(minutes=3), 1.0)
    upcoming_at = T0 - timedelta(seconds=20)  # on the board at now(), and fresh
    day.capture(upcoming, upcoming_at, 1.0)
    day.capture(gone, gone.post + timedelta(seconds=10), 1.08)  # the board after the post
    day.capture(long_gone, long_gone.post + timedelta(minutes=2), 0.93)
    models = fake_models()

    feed = build(T0, models)

    got = by_id(feed)
    for r, factor, at in (
        (gone, 1.08, gone.post + timedelta(seconds=10)),
        (long_gone, 0.93, long_gone.post + timedelta(minutes=2)),
    ):
        assert models.nowcast.taus(r.race_id) == {0}, r.race_id
        assert got[r.race_id]["minutes_to_post"] <= 0
        assert_race(got[r.race_id], day, r, now=T0, factor=factor, tau=0, capture_at=at)
    assert models.nowcast.taus(upcoming.race_id) == {5}
    assert_race(
        got[upcoming.race_id],
        day,
        upcoming,
        now=T0,
        factor=1.0,
        tau=5,
        capture_at=upcoming_at,
    )


# ---------------------------------------------------------------- without models


def test_without_models_values_are_null_and_flags_remain():
    tok1, kyo2 = race("TOK", 1, 3.5), race("KYO", 2, 40.0, SIX)
    day = SyntheticDay(DAY, [tok1, kyo2])
    day.capture(tok1, T0 - timedelta(seconds=25), 1.0)
    day.capture(kyo2, T0 - timedelta(minutes=2), 1.1)  # stale at T0

    feed = build(T0, None)

    assert feed["models"] is None
    got = by_id(feed)
    assert set(got) == {tok1.race_id, kyo2.race_id}
    assert_race(
        got[tok1.race_id],
        day,
        tok1,
        now=T0,
        factor=1.0,
        tau=5,
        capture_at=T0 - timedelta(seconds=25),
        valued=False,
    )
    assert_race(
        got[kyo2.race_id],
        day,
        kyo2,
        now=T0,
        factor=1.1,
        tau=60,
        capture_at=T0 - timedelta(minutes=2),
        valued=False,
    )
    flagged = [
        entry["flags"]
        for r in got.values()
        for entries in r["pools"].values()
        for entry in entries.values()
        if entry["flags"]
    ]
    assert len(flagged) > 10


# ---------------------------------------------------------------- no stake words


@pytest.mark.parametrize("with_models", [True, False], ids=["models", "no-models"])
def test_no_key_names_a_stake(with_models):
    races = [race("TOK", 1, 3.5), race("TOK", 2, 25.0, SIX), race("KYO", 1, -4.0)]
    day = SyntheticDay(DAY, races)
    for r in races:
        day.capture(r, T0 - timedelta(seconds=30), 1.0)

    feed = build(T0, fake_models() if with_models else None)

    assert len(feed["races"]) == 3
    keys = all_keys(feed)
    assert {"value_mean", "value_at_low", "flags", "tau_bucket_min"} <= set(keys)
    named = sorted({k for k in keys for word in BANNED_KEY_WORDS if word in k.lower()})
    assert named == [], named


# ---------------------------------------------------------------- the cache


def test_only_changed_races_are_recomputed():
    tok1 = race("TOK", 1, 10.5)  # bucket 16 throughout
    tok2 = race("TOK", 2, 40.0)  # bucket 60 throughout
    kyo1 = race("KYO", 1, 16.75)  # bucket 30, then 16 once 45 s have passed
    day = SyntheticDay(DAY, [tok1, tok2, kyo1])
    for r in (tok1, tok2, kyo1):
        day.capture(r, T0 - timedelta(minutes=5), 1.0)
        day.capture(r, T0 - timedelta(seconds=5), 1.02)
    models = fake_models()
    calls = models.nowcast.calls
    cache: dict = {}
    start = T0 - timedelta(seconds=5)

    first = build(T0, models, cache=cache)

    assert models.nowcast.races_called() == {tok1.race_id, tok2.race_id, kyo1.race_id}
    for r, tau in ((tok1, 16), (tok2, 60), (kyo1, 30)):
        assert_race(by_id(first)[r.race_id], day, r, now=T0, factor=1.02, tau=tau, capture_at=start)

    # the same capture and bucket everywhere: nothing is recomputed, the values are the same
    mark = len(calls)
    again = build(T0, models, cache=cache)
    assert models.nowcast.races_called(mark) == set()
    assert by_id(again) == by_id(first)

    # a new capture of TOK 1 only; every bucket unchanged
    new_at = T0 + timedelta(seconds=25)
    day.capture(tok1, new_at, 1.11)
    now1 = T0 + timedelta(seconds=30)
    mark = len(calls)
    second = build(now1, models, cache=cache)
    assert models.nowcast.races_called(mark) == {tok1.race_id}
    got = by_id(second)
    assert_race(got[tok1.race_id], day, tok1, now=now1, factor=1.11, tau=16, capture_at=new_at)
    for r, tau in ((tok2, 60), (kyo1, 30)):
        assert_race(got[r.race_id], day, r, now=now1, factor=1.02, tau=tau, capture_at=start)

    # no capture; KYO 1 crosses into bucket 16, and the other boards turn stale (65 s old)
    now2 = T0 + timedelta(seconds=60)
    mark = len(calls)
    third = build(now2, models, cache=cache)
    assert models.nowcast.races_called(mark) == {kyo1.race_id}
    assert models.nowcast.taus(kyo1.race_id) == {30, 16}
    got = by_id(third)
    assert_race(got[kyo1.race_id], day, kyo1, now=now2, factor=1.02, tau=16, capture_at=start)
    assert_race(got[tok2.race_id], day, tok2, now=now2, factor=1.02, tau=60, capture_at=start)
    assert_race(got[tok1.race_id], day, tok1, now=now2, factor=1.11, tau=16, capture_at=new_at)
    assert all("stale" in e["flags"] for e in got[tok2.race_id]["pools"][WIN].values())
    assert not any("stale" in e["flags"] for e in got[tok1.race_id]["pools"][WIN].values())
