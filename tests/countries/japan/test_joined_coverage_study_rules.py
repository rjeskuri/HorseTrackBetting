"""Ticket #418, round 2 findings B and C: `horsetrack japan joined-coverage` counts the member
source under the #87 study's own store and member rules.

Contract pinned here, through the CLI on a synthetic store (`meta_store_fakes`), reading the
`JoinedCoverage` the command hands to `research.joined_coverage.write_private_report`:

B. The member source (`MEMBER_SOURCE`) is scored only for a race the study scores:
   - a race `studies.meta.store_frame` drops (here: its winner has no Win price, so no priced
     winner and no valid q) is not member-scored: it drops with the reason `"not_in_store_frame"`,
     even when the member file has every runner;
   - a race the study uses only to train, before the first scored fold's test_start (the member
     file's seed season 2023, before `studies.meta.SCORED_FROM`), is not member-scored: it drops
     with the reason `"train_only"`;
   - both are in `REASONS`; every scheduled race is scored or dropped for exactly one reason.
C. Member rows are keyed on the file's own (race_id, horse_id): two member rows of scratched
   horses in one race (no starter, so no horse number) are not a duplicate key; they count as
   source rows that are not scheduled (`unmatched_source`) and do not drop the race. A key that
   does repeat in the member file still drops its race as `"duplicate_key"`.

The races before the member file's first day drop as `"unmatched"`, as round 1 pinned.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from meta_store_fakes import (
    Store,
    add_race_day,
    build_store,
    member_rows,
    write_members,
    write_store,
)
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.countries.japan.research import joined_coverage as jc

MEMBER_START = date(2023, 1, 10)  # the seed season 2023 only trains; 2024 is scored
SCORED_FROM = date(2024, 1, 1)
NEW_REASONS = ("not_in_store_frame", "train_only")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    seen: list[jc.JoinedCoverage] = []
    real = jc.write_private_report

    def spy(cov, *args, **kwargs):
        seen.append(cov)
        return real(cov, *args, **kwargs)

    monkeypatch.setattr(jc, "write_private_report", spy)
    return {"seen": seen, "members": tmp_path / "members"}


def races_by_span(store: Store) -> tuple[list[str], list[str], list[str]]:
    """The race ids before the member file, in its seed season, and in its scored seasons."""
    days = pd.to_datetime(store.races["race_date"])
    ids = store.races["race_id"]
    before = list(ids[days < pd.Timestamp(MEMBER_START)])
    seed = list(ids[(days >= pd.Timestamp(MEMBER_START)) & (days < pd.Timestamp(SCORED_FROM))])
    scored = list(ids[days >= pd.Timestamp(SCORED_FROM)])
    return before, seed, scored


def run(env: dict, store: Store, members: pd.DataFrame) -> jc.JoinedCoverage:
    write_store(store)
    path = write_members(env["members"] / "m.parquet", members)
    result = CliRunner().invoke(app, ["japan", "joined-coverage", "--members", str(path)])
    assert result.exit_code == 0, result.output
    assert env["seen"], "the command wrote no private report"
    return env["seen"][-1]


def every_race_once(cov: jc.JoinedCoverage, source: str) -> None:
    total = cov.races(source, "scheduled")
    assert cov.races(source, "scored") + sum(cov.drops(source, r) for r in jc.REASONS) == total


def test_the_member_source_scores_only_the_races_the_study_scores(env: dict) -> None:
    for reason in NEW_REASONS:
        assert reason in jc.REASONS, reason
    store = build_store()
    before, seed, scored = races_by_span(store)
    assert before and seed and len(scored) > 3
    # two scored races and one seed race lose their winner's price: store_frame drops them
    unpriced = [scored[1], scored[-2], seed[4]]
    runners = store.runners.copy()
    winner = runners["race_id"].isin(unpriced) & runners["finish_pos"].eq(1)
    runners.loc[winner, "win_odds"] = float("nan")
    store = Store(runners=runners, races=store.races)

    cov = run(env, store, member_rows(store, MEMBER_START))
    s = jc.MEMBER_SOURCE
    assert cov.races(s, "scheduled") == len(store.races)
    assert cov.drops(s, "unmatched") == len(before)
    assert cov.drops(s, "not_in_store_frame") + cov.drops(s, "train_only") == len(seed) + 2
    assert cov.drops(s, "not_in_store_frame") >= 2  # the two scored races, whatever the seed one
    assert cov.drops(s, "train_only") >= len(seed) - 1
    assert cov.races(s, "scored") == len(scored) - 2
    for reason in ("duplicate_key", "no_forecast", "no_result"):
        assert cov.drops(s, reason) == 0, reason
    every_race_once(cov, s)


def test_scratched_member_rows_do_not_drop_their_race_but_a_repeated_key_does(env: dict) -> None:
    store = build_store()
    _, seed, scored = races_by_span(store)
    scratched_race, repeated_race = scored[3], scored[9]
    runners = store.runners.copy()
    losers = runners.index[runners["race_id"].eq(scratched_race) & runners["finish_pos"].ne(1)]
    runners.loc[losers[:2], "is_scratched"] = True
    store = Store(runners=runners, races=store.races)
    members = member_rows(store, MEMBER_START)  # still holds the two scratched horses' rows
    copy = members.loc[members["race_id"].eq(repeated_race)].iloc[[0]]
    members = pd.concat([members, copy], ignore_index=True)

    cov = run(env, store, members)
    s = jc.MEMBER_SOURCE
    assert cov.drops(s, "duplicate_key") == 1, "only the repeated (race_id, horse_id) is one"
    assert cov.duplicate_rows(s) == 2
    assert cov.unmatched_source(s) == 2  # the scratched horses' rows: not starters
    assert cov.races(s, "scored") == len(scored) - 1  # the scratched horses' race is scored
    assert cov.drops(s, "train_only") == len(seed)
    every_race_once(cov, s)


# ------------------------------------------------------------------------------------------------
# The owner's restart instructions (2026-11-27), fix 5: the member source's stages come from the
# study's own sets. The scheduled denominator is the full schedule; `not_in_store_frame` is a race
# `studies.meta.store_frame` drops; `unmatched` and `no_forecast` look only at the runner keys
# `store_frame` keeps; `train_only` is a race in the remapped training indices of the scored folds;
# `outside_folds` (new) a race in neither their training nor their test indices; `scored` a race in
# their test indices.


def reasons(cov: jc.JoinedCoverage, source: str) -> dict[str, str | None]:
    """Per scheduled race, the reason it drops for `source` (None: scored)."""
    frame = cov.races_frame
    column = frame[f"reason:{source}"]
    return {r: None if pd.isna(x) else x for r, x in zip(frame["race_id"], column, strict=True)}


def with_odds(store: Store, picked, value: float = float("nan")) -> Store:
    runners = store.runners.copy()
    runners.loc[picked(runners), "win_odds"] = value
    return Store(runners=runners, races=store.races)


def first_loser(store: Store, race: str) -> int:
    rows = store.runners
    return int(rows.index[rows["race_id"].eq(race) & rows["finish_pos"].ne(1)][0])


def test_a_race_the_store_frame_drops_is_not_in_store_frame_whatever_its_member_rows(
    env: dict,
) -> None:
    store = build_store()
    before, seed, scored = races_by_span(store)
    full_scored, gap_scored, pre_member, full_seed = scored[2], scored[6], before[-3], seed[5]
    unpriced = [full_scored, gap_scored, pre_member, full_seed]  # their winner has no Win price
    store = with_odds(store, lambda r: r["race_id"].isin(unpriced) & r["finish_pos"].eq(1))
    runners = store.runners
    winner = runners.loc[runners["race_id"].eq(gap_scored) & runners["finish_pos"].eq(1)]
    gap = frozenset({(gap_scored, winner["horse_id"].iloc[0])})  # the file lacks that winner

    cov = run(env, store, member_rows(store, MEMBER_START, drop=gap))
    s = jc.MEMBER_SOURCE
    why = reasons(cov, s)
    for race in unpriced:
        assert why[race] == "not_in_store_frame", (race, why[race])
    assert cov.races(s, "scheduled") == len(store.races)
    assert cov.drops(s, "not_in_store_frame") == len(unpriced)
    assert cov.drops(s, "unmatched") == len(before) - 1
    assert cov.races(s, "scored") == len(scored) - 2
    assert cov.drops(s, "train_only") == len(seed) - 1
    every_race_once(cov, s)


@pytest.mark.parametrize("gap", ["no_member_row", "null_member_forecast"])
def test_an_unpriced_loser_without_a_member_forecast_does_not_drop_its_race(
    env: dict, gap: str
) -> None:
    store = build_store()
    before, seed, scored = races_by_span(store)
    scored_race, seed_race = scored[4], seed[7]
    losers = [first_loser(store, scored_race), first_loser(store, seed_race)]
    store = with_odds(store, lambda r: r.index.isin(losers))  # store_frame drops these runners
    keys = {(store.runners.at[i, "race_id"], store.runners.at[i, "horse_id"]) for i in losers}
    if gap == "no_member_row":
        members = member_rows(store, MEMBER_START, drop=frozenset(keys))
    else:
        members = member_rows(store, MEMBER_START)
        hit = [(r, h) in keys for r, h in zip(members["race_id"], members["horse_id"], strict=True)]
        members.loc[hit, "p_lgbm_no_jrdb_oof"] = float("nan")

    cov = run(env, store, members)
    s = jc.MEMBER_SOURCE
    why = reasons(cov, s)
    assert why[scored_race] is None, why[scored_race]  # matched, forecastable and scored
    assert why[seed_race] == "train_only", why[seed_race]
    assert cov.races(s, "scheduled") == len(store.races)
    assert cov.drops(s, "unmatched") == len(before)
    assert cov.drops(s, "no_forecast") == 0
    assert cov.races(s, "scored") == len(scored)
    assert cov.drops(s, "train_only") == len(seed)
    every_race_once(cov, s)


def q1_store() -> Store:
    """The fixture store cut at 2024-03-31, plus a priced 2023-12-30 race day and a priced
    2024-01-05 one: the only scored fold, 2024-Q1, tests from 2024-01-05 and trains to 2023-12-29
    (the 1-week embargo), so the 2023-12-30 races are in neither its training nor its test rows,
    and no later fold trains on them."""
    store = build_store()
    keep = pd.to_datetime(store.races["race_date"]) <= pd.Timestamp("2024-03-31")
    ids = set(store.races.loc[keep, "race_id"])
    store = Store(
        runners=store.runners.loc[store.runners["race_id"].isin(ids)].reset_index(drop=True),
        races=store.races.loc[keep].reset_index(drop=True),
    )
    return add_race_day(add_race_day(store, date(2023, 12, 30)), date(2024, 1, 5))


def test_a_race_between_the_training_cutoff_and_the_test_start_is_outside_folds(env: dict) -> None:
    assert "outside_folds" in jc.REASONS
    store = q1_store()
    days = pd.to_datetime(store.races["race_date"])
    ids = store.races["race_id"]
    before = set(ids[days < pd.Timestamp(MEMBER_START)])
    gap = set(ids[days == pd.Timestamp("2023-12-30")])
    seed = set(ids[(days >= pd.Timestamp(MEMBER_START)) & (days < pd.Timestamp(SCORED_FROM))]) - gap
    scored = set(ids[days >= pd.Timestamp(SCORED_FROM)])
    assert len(gap) == 2 and seed and len(scored) > 2
    members = member_rows(store, MEMBER_START)  # every runner from 2023-01-10, the gap day's too
    season = members["fold_season"].str[:4].astype(int)
    members["train_end"] = pd.to_datetime((season - 1).astype(str) + "-12-20")  # out of sample

    cov = run(env, store, members)
    s = jc.MEMBER_SOURCE
    why = reasons(cov, s)
    expected = {
        **{r: "unmatched" for r in before},
        **{r: "train_only" for r in seed},  # the seed season: in 2024-Q1's training rows
        **{r: "outside_folds" for r in gap},
        **{r: None for r in scored},
    }
    assert set(why) == set(expected)
    wrong = {r: (why[r], want) for r, want in expected.items() if why[r] != want}
    assert not wrong, dict(list(wrong.items())[:6])
    assert cov.races(s, "scheduled") == len(store.races)
    assert cov.drops(s, "outside_folds") == len(gap)
    assert cov.races(s, "scored") == len(scored)
    every_race_once(cov, s)
