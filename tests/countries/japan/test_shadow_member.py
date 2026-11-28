"""The horse member in the shadow feed (ticket #717): `p_member` and the p/q mark per race.

Contract pinned here (`horsetrack.countries.japan.board.shadow_feed.build_shadow`, one new
keyword, `member_inputs`):
- The race's segment is its `race_class` on the meeting's stored card
  (`snapshot.card_race_classes`: `processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`).
- `member_inputs(race_id) -> {horse number: {member column: value}} | None` is the race's
  per-runner member inputs on race day (#438's `feature_columns` names; a column left out or NaN
  is a missing value), None when the day has none for the race. The real JRDB rows come from
  #718; the tests pass a callable over synthetic dicts. Left out (None), no race has inputs.
- When `models.member` names the race's segment (DEBUT or MAIDEN; models/v0/loader.py), the race
  carries `p_member`: {horse: P(win)} over the runners on the board as of now (scratched runners
  have no key), the member fit's prediction at q = the board's 1/O_WIN normalised, summing to 1;
  and `member`: {"segment": the segment, "k": the pointer's k for it, "status": None, "marked":
  the horses (texts) whose p_member / q is at least k}.
- Without that day's JRDB rows (`member_inputs` None or returning None for the race, or no runner
  of the race with a JRDB member value, `workouts_jrdb__*`), `p_member` is None and `member` is
  {"segment", "k", "status": "no JRDB data", "marked": []}: no figure.
- Any other race (another class, a null class, no card, or a segment the pointer does not name)
  has `p_member` None and `member` None, the keys present whenever the models carry a member.
  Models without a member keep #506's race keys exactly (test_shadow_p_model.py).
- Display only: values, flags, p_model and the feed's `models` are exactly what they are without
  the member. A rebuild through the cache follows a new capture and newly arrived inputs.

Expected probabilities come from #438's own `fit_offset` / `predict_offset` on the training rows
(member_synth.py), never from the code under test. All data is synthetic, under tmp_path.
"""

from __future__ import annotations

import dataclasses
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.models.v0.loader import load_shadow_models
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

# isort: split
import member_synth as ms
from shadow_fakes import (
    BANNED_KEY_WORDS,
    IDS,
    PLACE,
    WIN,
    FakeNowcast,
    FakeOrdering,
    FakeWin,
    Race,
    SyntheticDay,
    all_keys,
    capture_rows,
    figures,
    log_path,
)

UTC = timezone.utc
DAY = date(2026, 11, 21)
T0 = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)  # 11:00 in Tokyo
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
N_DRAWS = 8
NO_JRDB = "no JRDB data"
KS = {"DEBUT": 1.1, "MAIDEN": 1.5}
TOK_CLASSES = {1: "DEBUT", 2: "MAIDEN", 3: "C1", 4: None, 5: "DEBUT", 6: "MAIDEN", 7: "G3"}
FIT_ROWS = ms.synthetic_rows(ms.weekend_days(date(2026, 5, 2), date(2026, 11, 15)), seed=1717)
DESIGN_FITS = {s: ms.offset_fit(FIT_ROWS, s, DAY) for s in ms.SEGMENTS}  # #438's own


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def field(n: int, seed: int) -> tuple[float, ...]:
    rng = random.Random(seed)
    odds = [round(1.5 + 1.9 * i**1.3 + rng.random(), 1) for i in range(n)]
    rng.shuffle(odds)
    return tuple(odds)


def race(venue: str, no: int, minutes: float, n: int) -> Race:
    return Race(DAY, venue, no, T0 + timedelta(minutes=minutes), field(n, 100 * no + len(venue)))


def write_card(venue: str, classes: dict[int, str | None]) -> None:
    path = paths.processed_dir("jra") / "upcoming" / f"jra_{DAY}_{venue}" / "races.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    nos = sorted(classes)
    pq.write_table(
        pa.table(
            {
                "race_id": pa.array([f"jra:{DAY}:{venue}:{no:02d}" for no in nos], pa.string()),
                "race_no": pa.array(nos, pa.int64()),
                "race_class": pa.array([classes[no] for no in nos], pa.string()),
                "post_time_local": pa.array(
                    [f"{DAY}T{9 + no:02d}:30:00+09:00" for no in nos], pa.string()
                ),
            }
        ),
        path,
    )


def inputs(r: Race, seed: int, *, jrdb_missing=(), no_jrdb=False) -> dict[int, dict[str, float]]:
    """Race-day member inputs of every horse drawn: the JRDB column left out for `jrdb_missing`
    (and for everyone with `no_jrdb`), the late move NaN for one horse."""
    rng = np.random.default_rng(seed)
    out = {}
    for h in range(1, len(r.base) + 1):
        row = {f: float(rng.normal(0.0, ms.SCALE[f])) for f in ms.FEATURES}
        if no_jrdb or h in jrdb_missing:
            del row[ms.JRDB_FEATURE]
        out[h] = row
    out[len(r.base)][ms.LATE_FEATURE] = float("nan")
    return out


# ---------------------------------------------------------------- the day

RACES = {
    "TOK1": race("TOK", 1, 3.5, 10),  # DEBUT
    "TOK2": race("TOK", 2, 25.0, 12),  # MAIDEN
    "TOK3": race("TOK", 3, 40.0, 9),  # C1
    "TOK4": race("TOK", 4, 55.0, 8),  # the card leaves its class null
    "TOK5": race("TOK", 5, 70.0, 11),  # DEBUT, no inputs for the day
    "TOK6": race("TOK", 6, 85.0, 14),  # MAIDEN, inputs without any JRDB column
    "TOK7": race("TOK", 7, 95.0, 16),  # G3
    "KYO1": race("KYO", 1, 12.0, 13),  # no card at all
}
SEGMENT_OF = {"TOK1": "DEBUT", "TOK2": "MAIDEN", "TOK5": "DEBUT", "TOK6": "MAIDEN"}
WITH_FIGURES = ("TOK1", "TOK2")
NONE_RACES = ("TOK3", "TOK4", "TOK7", "KYO1")


def day_inputs() -> dict[str, dict | None]:
    r = RACES
    return {
        r["TOK1"].race_id: inputs(r["TOK1"], 1, jrdb_missing=(2,)),
        r["TOK2"].race_id: inputs(r["TOK2"], 2),
        r["TOK3"].race_id: inputs(r["TOK3"], 3),
        r["TOK4"].race_id: inputs(r["TOK4"], 4),
        r["TOK5"].race_id: None,
        r["TOK6"].race_id: inputs(r["TOK6"], 6, no_jrdb=True),
        r["TOK7"].race_id: inputs(r["TOK7"], 7),
        r["KYO1"].race_id: inputs(r["KYO1"], 8),
    }


def setup_day() -> SyntheticDay:
    write_card("TOK", TOK_CLASSES)
    day = SyntheticDay(DAY, list(RACES.values()))
    for r in RACES.values():
        day.capture(r, T0 - timedelta(seconds=25), 1.0)
    return day


def fakes() -> ShadowModels:
    return ShadowModels(
        win=FakeWin(), ordering=FakeOrdering(), nowcast=FakeNowcast(), ids=dict(IDS)
    )


FITS: dict = {}


def member_fit(segment: str):
    from horsetrack.countries.japan.models.v0.member import fit_member

    if segment not in FITS:
        FITS[segment] = fit_member(FIT_ROWS, segment=segment, today=DAY, features=list(ms.FEATURES))
    return FITS[segment]


def member_models(ks: dict[str, float] = KS) -> ShadowModels:
    """The fake value models with the member the pointer names, loaded by load_shadow_models."""
    folder = ms.default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    entry = {}
    for segment, k in ks.items():
        name = f"member-{segment.lower()}.json"
        (folder / name).write_text(member_fit(segment).to_json(), encoding="utf-8")
        entry[segment] = {"fit": name, "k": k}
    pointer, _ = ms.write_pointer(folder, entry)
    loaded = load_shadow_models(pointer)
    assert loaded is not None and set(loaded.member) == set(ks)
    base = fakes()
    return dataclasses.replace(
        loaded, win=base.win, ordering=base.ordering, nowcast=base.nowcast, ids=base.ids
    )


def build(now: datetime, models, **options) -> dict:
    settings = dict(now=lambda: now, models=models, flag_rules=RULES, n_draws=N_DRAWS)
    settings.update(options)
    return build_shadow(DAY, seed=3, **settings)


def by_id(feed: dict) -> dict[str, dict]:
    return {r["race_id"]: r for r in feed["races"]}


def provider(table: dict[str, dict | None]):
    return lambda race_id: table.get(race_id)


def expected(r: Race, factor: float, table: dict, segment: str, k: float, out=()):
    """(p_member, marked) of the capture of `r` at `factor`, the runners `out` scratched: #438's
    prediction at the board's q, and the horses with p / q >= k."""
    win, _, _ = figures(r, factor)
    horses = sorted(h for h in win if h not in out)
    q = ms.market_q([win[h] for h in horses])
    rows = table[r.race_id]
    x = {f: [rows.get(h, {}).get(f, math.nan) for h in horses] for f in ms.FEATURES}
    p = ms.expected_member_p(DESIGN_FITS[segment], q, x)
    p_member = {str(h): float(v) for h, v in zip(horses, p, strict=True)}
    marked = {str(h) for h, pv, qv in zip(horses, p, q, strict=True) if pv / qv >= k}
    return p_member, marked


def check_figures(got: dict, want_p: dict, want_marked: set, segment: str, k: float, where):
    assert isinstance(got.get("p_member"), dict), (where, got.get("p_member"))
    assert set(got["p_member"]) == set(want_p), (where, sorted(got["p_member"]))
    for h, p in want_p.items():
        assert math.isclose(got["p_member"][h], p, rel_tol=1e-6, abs_tol=1e-9), (where, h)
    assert math.isclose(math.fsum(got["p_member"].values()), 1.0, rel_tol=1e-9), where
    member = got.get("member")
    assert isinstance(member, dict), (where, member)
    assert member["segment"] == segment, where
    assert member["k"] == pytest.approx(k), where
    assert member["status"] is None, (where, member["status"])
    assert set(member["marked"]) == want_marked, (where, sorted(member["marked"]), want_marked)


def _check_fixture() -> None:
    """The member moves off the market, marks some horses but not all, and the k matters."""
    table = day_inputs()
    for name in WITH_FIGURES:
        r, segment = RACES[name], SEGMENT_OF[name]
        p, marked = expected(r, 1.0, table, segment, KS[segment])
        win, _, _ = figures(r, 1.0)
        q = ms.market_q([win[h] for h in sorted(win)])
        assert max(abs(p[str(h)] - qv) for h, qv in zip(sorted(win), q, strict=True)) > 1e-3
        assert 0 < len(marked) < len(win), (name, marked)
        other = KS["MAIDEN" if segment == "DEBUT" else "DEBUT"]
        assert expected(r, 1.0, table, segment, other)[1] != marked, name


_check_fixture()


# ---------------------------------------------------------------- tests


def test_debut_and_maiden_races_carry_the_member_and_its_mark():
    setup_day()
    table = day_inputs()

    got = by_id(build(T0, member_models(), member_inputs=provider(table)))

    assert set(got) == {r.race_id for r in RACES.values()}
    for name in WITH_FIGURES:
        r, segment = RACES[name], SEGMENT_OF[name]
        want_p, want_marked = expected(r, 1.0, table, segment, KS[segment])
        check_figures(got[r.race_id], want_p, want_marked, segment, KS[segment], name)
    for name in NONE_RACES:
        race_ = got[RACES[name].race_id]
        assert "p_member" in race_ and race_["p_member"] is None, (name, race_.get("p_member"))
        assert "member" in race_ and race_["member"] is None, (name, race_.get("member"))


@pytest.mark.parametrize(
    "ks", [{"DEBUT": 1.05, "MAIDEN": 1.9}, {"DEBUT": 2.0, "MAIDEN": 1.2}], ids=str
)
def test_the_mark_follows_the_pointers_k(ks):
    setup_day()
    table = day_inputs()

    got = by_id(build(T0, member_models(ks), member_inputs=provider(table)))

    for name in WITH_FIGURES:
        r, segment = RACES[name], SEGMENT_OF[name]
        want_p, want_marked = expected(r, 1.0, table, segment, ks[segment])
        check_figures(got[r.race_id], want_p, want_marked, segment, ks[segment], name)


def test_without_the_days_jrdb_rows_the_race_says_so_and_shows_no_figure():
    setup_day()
    table = day_inputs()

    for label, options in (
        ("inputs given", {"member_inputs": provider(table)}),
        ("no inputs at all", {}),
    ):
        got = by_id(build(T0, member_models(), **options))
        missing = ("TOK5", "TOK6") if label == "inputs given" else tuple(SEGMENT_OF)
        for name in missing:
            race_ = got[RACES[name].race_id]
            assert race_["p_member"] is None, (label, name, race_["p_member"])
            member = race_["member"]
            assert isinstance(member, dict), (label, name, member)
            assert member["status"] == NO_JRDB, (label, name, member)
            assert member["segment"] == SEGMENT_OF[name], (label, name)
            assert member["k"] == pytest.approx(KS[SEGMENT_OF[name]]), (label, name)
            assert not member["marked"], (label, name)
        for name in NONE_RACES:
            assert got[RACES[name].race_id]["member"] is None, (label, name)


def test_a_segment_the_pointer_does_not_name_has_no_member():
    setup_day()
    table = day_inputs()

    got = by_id(build(T0, member_models({"DEBUT": 1.3}), member_inputs=provider(table)))

    r = RACES["TOK1"]
    want_p, want_marked = expected(r, 1.0, table, "DEBUT", 1.3)
    check_figures(got[r.race_id], want_p, want_marked, "DEBUT", 1.3, "TOK1")
    for name in ("TOK2", "TOK6", *NONE_RACES):  # the maiden races too now
        race_ = got[RACES[name].race_id]
        assert race_["p_member"] is None and race_["member"] is None, name


def test_the_member_changes_no_value_flag_or_model_probability():
    setup_day()
    table = day_inputs()

    plain = build(T0, fakes())
    with_member = build(T0, member_models(), member_inputs=provider(table))

    assert with_member["models"] == plain["models"]
    plain_races, member_races = by_id(plain), by_id(with_member)
    assert set(plain_races) == set(member_races)
    for race_id, race_ in plain_races.items():
        assert "p_member" not in race_ and "member" not in race_, race_id  # #506's keys only
        other = member_races[race_id]
        assert other["pools"] == race_["pools"], race_id
        assert other["p_model"] == race_["p_model"], race_id
        rest = {k: v for k, v in other.items() if k not in ("p_member", "member")}
        assert rest == race_, race_id
    assert any(v["pools"][WIN] for v in plain_races.values())  # the fixture: values to compare
    for key in all_keys(with_member):
        assert not any(word in key.lower() for word in BANNED_KEY_WORDS), key


def capture_with_scratches(day: SyntheticDay, r: Race, at: datetime, factor: float, out) -> None:
    """A capture where the runners `out` show SCR on WIN and PLACE and their Wide pairs are gone."""
    rows = []
    for row in capture_rows(r, at, factor):
        runners = {int(part) for part in row["combination"].split(",")}
        if not runners & set(out):
            rows.append(row)
        elif row["pool"] in (WIN, PLACE):
            scratched = {k: v for k, v in row.items() if k != "odds_hi"}
            rows.append({**scratched, "odds": None, "status_raw": "SCR"})
    odds_log.append_capture(log_path(day.day, r.venue), rows)
    day.rows.extend(rows)


def test_a_scratched_runner_has_no_member_probability():
    day = setup_day()
    table = day_inputs()
    r = RACES["TOK2"]
    out = (3, 8)
    capture_with_scratches(day, r, T0 - timedelta(seconds=5), 1.0, out)

    got = by_id(build(T0, member_models(), member_inputs=provider(table)))

    want_p, want_marked = expected(r, 1.0, table, "MAIDEN", KS["MAIDEN"], out=out)
    assert not {"3", "8"} & set(want_p)
    check_figures(got[r.race_id], want_p, want_marked, "MAIDEN", KS["MAIDEN"], "scratched")


def test_a_rebuild_follows_a_new_capture_and_newly_arrived_inputs():
    day = setup_day()
    table = day_inputs()
    models = member_models()
    cache: dict = {}

    first = by_id(build(T0, models, member_inputs=provider(table), cache=cache))
    r1, r5 = RACES["TOK1"], RACES["TOK5"]
    want_p, want_marked = expected(r1, 1.0, table, "DEBUT", KS["DEBUT"])
    check_figures(first[r1.race_id], want_p, want_marked, "DEBUT", KS["DEBUT"], "first")
    assert first[r5.race_id]["member"]["status"] == NO_JRDB

    moved = Race(r1.day, r1.venue, r1.no, r1.post, field(len(r1.base), 9917))  # a new crowd
    day.capture(moved, T0 + timedelta(seconds=25), 1.0)
    table[r5.race_id] = inputs(r5, 55)  # the day's rows for TOK 5 arrive
    second = by_id(
        build(T0 + timedelta(seconds=30), models, member_inputs=provider(table), cache=cache)
    )

    want_p, want_marked = expected(moved, 1.0, table, "DEBUT", KS["DEBUT"])
    check_figures(second[r1.race_id], want_p, want_marked, "DEBUT", KS["DEBUT"], "new capture")
    old_p, _ = expected(r1, 1.0, table, "DEBUT", KS["DEBUT"])
    assert max(abs(old_p[h] - want_p[h]) for h in want_p) > 1e-3  # the fixture: it moved
    want_p, want_marked = expected(r5, 1.0, table, "DEBUT", KS["DEBUT"])
    check_figures(second[r5.race_id], want_p, want_marked, "DEBUT", KS["DEBUT"], "rows arrived")
