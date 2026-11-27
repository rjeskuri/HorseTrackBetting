"""Ticket #418: the population that reaches a model after the point-in-time joins
(`horsetrack.countries.japan.research.joined_coverage`).

Contract pinned here (the coder implements what these tests call):

`joined_coverage(schedule, sources) -> JoinedCoverage`, pure, on in-memory frames:
- `schedule`: one row per scheduled runner, columns `race_id`, `horse_no`, `race_date` (a date or
  anything `pd.to_datetime` reads), `race_class` (the card class: DEBUT, MAIDEN, C1, ..., G1).
  A race whose `race_date` is a holdout day (`is_holdout_day("japan", d)`, D14) is in no count,
  and neither is any source row of it: the report is a development report.
- `sources`: `{source name: frame}`, each frame one row per runner the source holds, columns
  `race_id`, `horse_no`, `forecast` (float, NaN = no forecast for the runner after the joins) and
  `won` (0/1, missing = no result).
- Every scheduled race passes, per source, the stages `STAGES = ("scheduled", "matched",
  "forecastable", "scored")` in order and drops at the first it fails, with one reason:
  matched fails with `"duplicate_key"` when a (race_id, horse_no) of the race occurs more than once
  in the source, else with `"unmatched"` when a scheduled runner of the race has no source row;
  forecastable fails with `"no_forecast"` when a runner's forecast is missing; scored fails with
  `"no_result"` when the race has no result (its `won` values are missing).
- `cov.races(source, stage, *, era=None, race_class=None) -> int`: the races that reached
  `stage` (so `scheduled >= matched >= forecastable >= scored`), optionally of one era and one
  card class. The era of a race is `PRICE_ERA` from D10's price era start
  (`models.v0.store.PRICE_ERA_START`, 2014-06-07) on, `PRE_PRICE_ERA` before it.
- `cov.drops(source, reason, *, era=None, race_class=None) -> int`: the races dropped for it.
- `cov.duplicate_rows(source)`: source rows whose key occurs more than once in it (every copy);
  `cov.unmatched_scheduled(source)`: scheduled runner keys with no source row;
  `cov.unmatched_source(source)`: source rows whose key is not scheduled.
- `cov.sources`: the source names.

Rendering (D21, as `research.coverage` does):
- `render_summary(cov) -> str`, the committed view: a licensed source (`reports.is_withheld((s,))`:
  jravan, JRDB) shows `coverage.WITHHELD` and no figure of it; the public ones show their
  denominators and drop reasons.
- `render_report(cov) -> str`, the private view, every figure and every drop reason.
- `write_private_report(cov, path=None) -> Path`: `render_report(cov)` through `write_report(...,
  sources=cov.sources)`, by default to `paths.reports_dir() / REPORT_NAME`; inside the repository
  it raises `LicensedOutputInRepo` and writes nothing.

`first_starts(runs, *, store_start=None) -> DataFrame` (the history gaps of #377/#403): `runs` has
`horse_id`, `race_date`, `race_class`, `is_scratched`. One row per horse with a start (a row not
scratched), columns `horse_id`, `first_start` (date of its earliest start) and `history`:
`VERIFIED_DEBUT` when that start is in a DEBUT (新馬) race; otherwise `STORE_START` when it falls
before one year after `store_start` (default: the earliest `race_date` in `runs`, scratched rows
included), so earlier runs may predate the store; otherwise `UNVERIFIED` (a maiden debut or a
horse with runs outside JRA, #377).

Every value is synthetic and in memory or under tmp_path; no test reads real data.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import random
import uuid
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.research import coverage as cv
from horsetrack.countries.japan.research import joined_coverage as jc

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGES = ("scheduled", "matched", "forecastable", "scored")
REASONS = ("duplicate_key", "unmatched", "no_forecast", "no_result")
CLASSES = ("DEBUT", "MAIDEN", "C1", "C2", "OP", "G1")
VENUE = "QQV"  # a made-up venue code, so a race id never matches a real one
HOLDOUT_A, HOLDOUT_B = date(2026, 11, 29), date(2026, 12, 1)
DEV_LATE = date(2026, 11, 28)


def race_id(day: date, n: int) -> str:
    return f"jra:{day.isoformat()}:{VENUE}:{n:02d}"


def era_of(day: date) -> str:
    return jc.PRICE_ERA if day >= PRICE_ERA_START else jc.PRE_PRICE_ERA


# ------------------------------------------------------------------------------------------------
# Synthetic schedules and source rows


def schedule_rows(races: list[tuple[str, date, str, int]]) -> pd.DataFrame:
    """(race_id, race_date, race_class, runners) -> one row per scheduled runner."""
    rows = [
        {"race_id": rid, "horse_no": no, "race_date": day, "race_class": cls}
        for rid, day, cls, n in races
        for no in range(1, n + 1)
    ]
    return pd.DataFrame(rows)


def source_rows(races: list[tuple[str, date, str, int]], issues: dict[str, str]) -> pd.DataFrame:
    """The source's rows for the races, with one issue per race id (absent = clean):
    `absent` (no row), `duplicate_key`, `unmatched` (its last runner missing), `no_forecast`
    (the first runner's forecast NaN), `no_result` (every `won` missing)."""
    rows = []
    for rid, _day, _cls, n in races:
        issue = issues.get(rid)
        if issue == "absent":
            continue
        for no in range(1, n + 1):
            if issue == "unmatched" and no == n:
                continue
            row = {
                "race_id": rid,
                "horse_no": no,
                "forecast": 1.0 / n,
                "won": float(no == 1),
            }
            if issue == "no_forecast" and no == 1:
                row["forecast"] = np.nan
            if issue == "no_result":
                row["won"] = np.nan
            rows.append(row)
            if issue == "duplicate_key" and no == 1:
                rows.append(dict(row))
    return pd.DataFrame(rows, columns=["race_id", "horse_no", "forecast", "won"])


def orphan_rows(k: int) -> pd.DataFrame:
    """k source rows of a race nobody scheduled."""
    rid = race_id(date(2019, 2, 3), 12)
    return pd.DataFrame(
        {"race_id": [rid] * k, "horse_no": list(range(1, k + 1)), "forecast": 0.5, "won": 0.0}
    )


# A hand-built card: one issue per race in the jra store, another pattern in the jravan store.
D_PRE = date(2013, 5, 5)
D_EDGE_BEFORE = PRICE_ERA_START - timedelta(days=1)
D_EDGE = PRICE_ERA_START
D_2015 = date(2015, 3, 1)
D_2020 = date(2020, 6, 7)
CARD = [
    (race_id(D_PRE, 1), D_PRE, "DEBUT", 3),
    (race_id(D_EDGE_BEFORE, 2), D_EDGE_BEFORE, "C1", 2),
    (race_id(D_EDGE, 3), D_EDGE, "C1", 4),
    (race_id(D_2015, 4), D_2015, "MAIDEN", 3),
    (race_id(D_2015, 5), D_2015, "C1", 3),
    (race_id(D_2020, 6), D_2020, "DEBUT", 2),
    (race_id(D_2020, 7), D_2020, "G1", 5),
    (race_id(D_2020, 8), D_2020, "MAIDEN", 3),
]
R = [r[0] for r in CARD]
JRA_ISSUES = {
    R[3]: "duplicate_key",
    R[4]: "unmatched",
    R[5]: "no_forecast",
    R[6]: "no_result",
}
JRAVAN_ISSUES = {
    R[0]: "absent",
    R[2]: "no_forecast",
    R[4]: "duplicate_key",
    R[7]: "duplicate_key",
}


def card_coverage(jravan_issues: dict[str, str] | None = None, *, orphans: int = 2):
    jra = pd.concat([source_rows(CARD, JRA_ISSUES), orphan_rows(orphans)], ignore_index=True)
    jravan = source_rows(CARD, JRAVAN_ISSUES if jravan_issues is None else jravan_issues)
    return jc.joined_coverage(schedule_rows(CARD), {"jra": jra, "jravan": jravan})


def test_the_card_gives_each_source_its_stage_counts_and_drop_reasons() -> None:
    cov = card_coverage()
    assert set(cov.sources) == {"jra", "jravan"}
    expected = {
        "jra": {"scheduled": 8, "matched": 6, "forecastable": 5, "scored": 4},
        "jravan": {"scheduled": 8, "matched": 5, "forecastable": 4, "scored": 4},
    }
    for source, stages in expected.items():
        for stage, n in stages.items():
            assert cov.races(source, stage) == n, (source, stage)
    drops = {
        "jra": {"duplicate_key": 1, "unmatched": 1, "no_forecast": 1, "no_result": 1},
        "jravan": {"duplicate_key": 2, "unmatched": 1, "no_forecast": 1, "no_result": 0},
    }
    for source, reasons in drops.items():
        for reason, n in reasons.items():
            assert cov.drops(source, reason) == n, (source, reason)


def test_the_card_counts_duplicate_and_unmatched_keys() -> None:
    cov = card_coverage()
    # jra: R[3]'s runner 1 twice; R[4] lacks its last runner; two orphan rows
    assert cov.duplicate_rows("jra") == 2
    assert cov.unmatched_scheduled("jra") == 1
    assert cov.unmatched_source("jra") == 2
    # jravan: R[4] and R[7] each repeat runner 1; R[0] (3 runners) is absent
    assert cov.duplicate_rows("jravan") == 4
    assert cov.unmatched_scheduled("jravan") == 3
    assert cov.unmatched_source("jravan") == 0
    # the orphan count follows the input
    assert card_coverage(orphans=5).unmatched_source("jra") == 5


def test_the_card_splits_by_era_at_the_price_era_start_and_by_card_class() -> None:
    cov = card_coverage()
    pre, price = jc.PRE_PRICE_ERA, jc.PRICE_ERA
    assert pre != price
    # R[0] (2013) and R[1] (the day before 2014-06-07) are before it; R[2] (on it) is in it
    assert cov.races("jra", "scheduled", era=pre) == 2
    assert cov.races("jra", "scheduled", era=price) == 6
    assert cov.races("jravan", "scored", era=pre) == 1  # R[0] absent, R[1] scored
    assert cov.races("jravan", "scored", era=price) == 3  # R[3], R[5], R[6]
    assert cov.drops("jravan", "no_forecast", era=price) == 1
    assert cov.drops("jravan", "no_forecast", era=pre) == 0
    assert cov.drops("jravan", "unmatched", era=pre) == 1
    # by card class, and by era and class together
    assert cov.races("jra", "scheduled", race_class="C1") == 3
    assert cov.races("jra", "scored", race_class="C1") == 2  # R[1], R[2]; R[4] unmatched
    assert cov.races("jra", "scored", race_class="DEBUT") == 1  # R[0]; R[5] no forecast
    assert cov.races("jravan", "scored", race_class="DEBUT") == 1  # R[5]; R[0] absent
    assert cov.races("jra", "scheduled", era=pre, race_class="C1") == 1
    assert cov.races("jra", "scored", era=price, race_class="C1") == 1
    assert cov.drops("jravan", "duplicate_key", race_class="MAIDEN") == 1
    assert cov.drops("jravan", "duplicate_key", race_class="C1") == 1
    assert cov.races("jra", "scheduled", race_class="G2") == 0


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_a_generated_schedule_gives_counts_that_follow_its_issues(seed: int) -> None:
    rng = random.Random(seed)
    start = date(2011, 1, 2)
    days = sorted({start + timedelta(days=7 * rng.randrange(0, 52 * 14)) for _ in range(30)})
    races = []
    for day in days:
        for n in range(1, rng.randrange(1, 3) + 1):
            races.append((race_id(day, n), day, rng.choice(CLASSES), rng.randrange(1, 6)))
    kinds = [None, None, "absent", "duplicate_key", "unmatched", "no_forecast", "no_result"]
    issues = {"jra": {}, "jravan": {}}
    for source in issues:
        for rid, _d, _c, n in races:
            kind = rng.choice(kinds)
            if kind == "unmatched" and n == 1:
                kind = "absent"  # a one-runner race missing its runner has no row at all
            if kind is not None:
                issues[source][rid] = kind
    orphans = {"jra": rng.randrange(0, 4), "jravan": rng.randrange(0, 4)}
    frames = {
        s: pd.concat([source_rows(races, issues[s]), orphan_rows(orphans[s])], ignore_index=True)
        for s in issues
    }
    cov = jc.joined_coverage(schedule_rows(races), frames)

    reason_of = {"absent": "unmatched"}
    for source in issues:
        reached: Counter = Counter()
        dropped: Counter = Counter()
        dup_rows = unmatched = 0
        for rid, day, cls, n in races:
            kind = issues[source].get(rid)
            reason = reason_of.get(kind, kind)
            cut = STAGES.index("scored") + 1 if reason is None else {
                "duplicate_key": 1, "unmatched": 1, "no_forecast": 2, "no_result": 3
            }[reason]  # fmt: skip
            for stage in STAGES[:cut]:
                for key in ((None, None), (era_of(day), None), (None, cls), (era_of(day), cls)):
                    reached[(stage, *key)] += 1
            if reason is not None:
                for key in ((None, None), (era_of(day), None), (None, cls), (era_of(day), cls)):
                    dropped[(reason, *key)] += 1
            dup_rows += 2 * (kind == "duplicate_key")
            unmatched += {"absent": n, "unmatched": 1}.get(kind, 0)
        for stage in STAGES:
            for era in (None, jc.PRE_PRICE_ERA, jc.PRICE_ERA):
                for cls in (None, *CLASSES):
                    got = cov.races(source, stage, era=era, race_class=cls)
                    assert got == reached[(stage, era, cls)], (source, stage, era, cls)
        for reason in REASONS:
            for era in (None, jc.PRE_PRICE_ERA, jc.PRICE_ERA):
                for cls in (None, *CLASSES):
                    got = cov.drops(source, reason, era=era, race_class=cls)
                    assert got == dropped[(reason, era, cls)], (source, reason, era, cls)
        assert cov.duplicate_rows(source) == dup_rows, source
        assert cov.unmatched_scheduled(source) == unmatched, source
        assert cov.unmatched_source(source) == orphans[source], source
        # every scheduled race reaches the end or drops once
        total = cov.races(source, "scored") + sum(cov.drops(source, r) for r in REASONS)
        assert total == cov.races(source, "scheduled") == len(races)


def test_holdout_days_are_in_no_count() -> None:
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    assert not is_holdout_day("japan", DEV_LATE)
    card = [
        (race_id(DEV_LATE, 1), DEV_LATE, "C1", 2),
        (race_id(HOLDOUT_A, 1), HOLDOUT_A, "C1", 2),
        (race_id(HOLDOUT_B, 2), HOLDOUT_B, "MAIDEN", 3),
    ]
    # the holdout races would drop at every stage if they were counted
    jra = source_rows(card, {card[1][0]: "duplicate_key", card[2][0]: "unmatched"})
    cov = jc.joined_coverage(schedule_rows(card), {"jra": jra})
    assert cov.races("jra", "scheduled") == 1
    assert cov.races("jra", "scored") == 1
    assert cov.races("jra", "scheduled", race_class="MAIDEN") == 0
    assert sum(cov.drops("jra", r) for r in REASONS) == 0
    assert cov.duplicate_rows("jra") == 0
    assert cov.unmatched_scheduled("jra") == 0
    assert cov.unmatched_source("jra") == 0  # a holdout race's rows are not orphans either


# ------------------------------------------------------------------------------------------------
# The aggregate report (D21)


def test_the_committed_summary_carries_no_licensed_figure() -> None:
    base = card_coverage()
    other = card_coverage({R[1]: "no_result", R[5]: "unmatched"})
    assert other.races("jravan", "scored") != base.races("jravan", "scored")
    summary = jc.render_summary(base)
    assert cv.WITHHELD in summary
    # the jravan figures changed and the committed view did not
    assert jc.render_summary(other) == summary
    # the public source's figures are in it: they change it
    assert jc.render_summary(card_coverage(orphans=7)) != summary
    for reason in REASONS:
        assert reason in summary, reason


def test_the_private_report_holds_every_figure_and_drop_reason() -> None:
    base = card_coverage()
    report = jc.render_report(base)
    for reason in REASONS:
        assert reason in report, reason
    assert cv.WITHHELD not in report
    assert jc.render_report(card_coverage({R[1]: "no_result", R[5]: "unmatched"})) != report
    assert jc.render_report(card_coverage(orphans=7)) != report


def test_the_private_report_is_written_outside_the_repo_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    cov = card_coverage()

    written = jc.write_private_report(cov)
    assert written == paths.reports_dir() / jc.REPORT_NAME
    assert written.read_text(encoding="utf-8") == jc.render_report(cov)

    elsewhere = tmp_path / "elsewhere" / "joined.md"
    assert jc.write_private_report(cov, elsewhere) == elsewhere
    assert elsewhere.read_text(encoding="utf-8") == jc.render_report(cov)

    target = REPO_ROOT / f".test-joined-coverage-{uuid.uuid4().hex[:12]}" / "joined.md"
    try:
        with pytest.raises(LicensedOutputInRepo):
            jc.write_private_report(cov, target)
        assert not target.parent.exists()
    finally:
        if target.exists():
            target.unlink()
        if target.parent.exists():
            target.parent.rmdir()

    # the default path refused too when the reports folder is the repository
    reports = paths.reports_dir()
    (reports / jc.REPORT_NAME).unlink()
    monkeypatch.setattr(env, "repo_root", lambda: reports)
    with pytest.raises(LicensedOutputInRepo):
        jc.write_private_report(cov)
    assert list(reports.iterdir()) == []


# ------------------------------------------------------------------------------------------------
# History: verified debuts against first observed starts that may miss earlier runs


def runs_frame(rows: list[tuple[str, date, str, bool]], seed: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["horse_id", "race_date", "race_class", "is_scratched"])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # order is no cue


RUNS = [
    # h1: first start a DEBUT race, years after the store starts
    ("h1", date(2015, 7, 4), "DEBUT", False),
    ("h1", date(2015, 9, 1), "MAIDEN", False),
    # h2: a DEBUT race in the store's first year is still a verified debut
    ("h2", date(2010, 3, 6), "DEBUT", False),
    ("h2", date(2010, 5, 2), "C1", False),
    # h3: a C1 start in the store's first year: earlier runs may predate the store
    ("h3", date(2010, 4, 10), "C1", False),
    ("h3", date(2011, 6, 1), "C2", False),
    # h4: first observed start a maiden race long after the store starts
    ("h4", date(2016, 2, 7), "MAIDEN", False),
    # h5: scratched from a DEBUT race, first start a maiden race: not verified
    ("h5", date(2016, 6, 5), "DEBUT", True),
    ("h5", date(2016, 7, 3), "MAIDEN", False),
    # h6: an open-class first start (an NAR or overseas transfer, #377)
    ("h6", date(2018, 10, 7), "OP", False),
    ("h6", date(2018, 8, 5), "C2", True),
    # h7: never started
    ("h7", date(2019, 1, 6), "DEBUT", True),
]


@pytest.mark.parametrize(
    ("store_start", "expected"),
    [
        (
            None,  # the earliest race day in the runs, 2010-03-06
            {"h1": "V", "h2": "V", "h3": "S", "h4": "U", "h5": "U", "h6": "U"},
        ),
        (
            date(2005, 1, 1),  # a store much older than every first start
            {"h1": "V", "h2": "V", "h3": "U", "h4": "U", "h5": "U", "h6": "U"},
        ),
        (
            date(2015, 10, 1),  # a store that starts later: h4, h5 fall in its first year
            {"h1": "V", "h2": "V", "h3": "S", "h4": "S", "h5": "S", "h6": "U"},
        ),
    ],
)
def test_first_starts_tell_verified_debuts_from_possibly_incomplete_history(
    store_start: date | None, expected: dict[str, str]
) -> None:
    labels = {"V": jc.VERIFIED_DEBUT, "S": jc.STORE_START, "U": jc.UNVERIFIED}
    assert len(set(labels.values())) == 3
    for seed in (0, 1):
        out = jc.first_starts(runs_frame(RUNS, seed), store_start=store_start)
        assert len(out) == len(expected)
        got = dict(zip(out["horse_id"], out["history"], strict=True))
        assert got == {h: labels[v] for h, v in expected.items()}, store_start
        first = {
            h: pd.Timestamp(d).date()
            for h, d in zip(out["horse_id"], out["first_start"], strict=True)
        }
        assert first == {
            "h1": date(2015, 7, 4),
            "h2": date(2010, 3, 6),
            "h3": date(2010, 4, 10),
            "h4": date(2016, 2, 7),
            "h5": date(2016, 7, 3),
            "h6": date(2018, 10, 7),
        }
        counts = Counter(out["history"])
        assert counts == Counter(labels[v] for v in expected.values())
