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
from meta_store_fakes import Store, build_store, member_rows, write_members, write_store
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
