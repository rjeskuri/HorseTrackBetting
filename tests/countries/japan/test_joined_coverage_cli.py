"""Ticket #418, round 1 finding 4 (DoD 1): the aggregate joined-coverage report from the store.

Contract pinned here: `horsetrack japan joined-coverage --members PATH`, registered on the `japan`
group of `horsetrack.cli.app`.

- It builds the schedule from the store's development runners (`studies.meta.read_development`,
  `store_frame`: price era, holdout days dropped) and takes the #87 member file
  (`studies.meta.read_members`) as a source beside the store's own, then runs
  `research.joined_coverage.joined_coverage`.
- It writes the private report through `write_private_report` to its default,
  `paths.reports_dir() / joined_coverage.REPORT_NAME`, outside the repository: every source's stage
  counts and drop reasons, and the history labels of `first_starts` on the store's runs with the
  number of horses under each (`VERIFIED_DEBUT`, `STORE_START`, `UNVERIFIED`).
- It prints only the committed summary (`render_summary`): the member source is JRDB-derived
  (D21), so it shows `coverage.WITHHELD` and no figure of it. Exit code 0.

Every value is synthetic, under a tmp HORSETRACK_DATA_DIR (`meta_store_fakes`).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from meta_store_fakes import build_store, member_rows, write_members, write_store
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.countries.japan.research import coverage as cv
from horsetrack.countries.japan.research import joined_coverage as jc

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBER_START = date(2023, 1, 10)
STAGES = ("scheduled", "matched", "forecastable", "scored")
REASONS = ("duplicate_key", "unmatched", "no_forecast", "no_result")


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    store = build_store()
    write_store(store)
    return {"store": store, "members": tmp_path / "members"}


def runners_of(store, rid: str) -> list[tuple[str, str]]:
    rows = store.runners.loc[store.runners["race_id"] == rid]
    return list(zip(rows["race_id"], rows["horse_id"], strict=True))


def member_race_ids(store) -> list[str]:
    days = pd.to_datetime(store.races["race_date"])
    return list(store.races.loc[days >= pd.Timestamp(MEMBER_START), "race_id"])


def invoke(members: Path):
    return CliRunner().invoke(app, ["japan", "joined-coverage", "--members", str(members)])


def has_line(text: str, *, label: str, count: int) -> bool:
    """Whether a line of `text` carries `label` and the number `count` (as rendered, 1,234)."""
    number = re.compile(rf"(?<![\d,.]){re.escape(f'{count:,}')}(?![\d,]|\.\d)")
    return any(label in line and number.search(line) for line in text.splitlines())


def test_joined_coverage_writes_the_private_report_and_prints_only_the_summary(data: dict) -> None:
    store = data["store"]
    in_span = member_race_ids(store)
    # one runner of one member-span race lacks a member row: that race drops as unmatched
    drop_a = frozenset(runners_of(store, in_span[3])[2:3])
    members_a = write_members(
        data["members"] / "a.parquet", member_rows(store, MEMBER_START, drop=drop_a)
    )

    result = invoke(members_a)
    assert result.exit_code == 0, result.output

    report_path = paths.reports_dir() / jc.REPORT_NAME
    assert report_path.is_file()
    assert not report_path.resolve().is_relative_to(REPO_ROOT.resolve())
    report = report_path.read_text(encoding="utf-8")

    races = len(store.races)
    before_span = races - len(in_span)
    assert before_span > 0
    # every source schedules every development race of the store
    for stage in STAGES:
        assert stage in report, stage
    for reason in REASONS:
        assert reason in report, reason
    assert has_line(report, label="scheduled", count=races)
    # the member source: the races before its first day and the one with a missing runner
    assert has_line(report, label="unmatched", count=before_span + 1)

    # the printed summary withholds the member source: no figure of it changes the output
    assert cv.WITHHELD in result.output
    assert not has_line(result.output, label="unmatched", count=before_span + 1)
    drop_b = frozenset(
        key for rid in (in_span[0], in_span[7], in_span[20]) for key in runners_of(store, rid)[:1]
    )
    members_b = write_members(
        data["members"] / "b.parquet", member_rows(store, MEMBER_START, drop=drop_b, seed=5)
    )
    other = invoke(members_b)
    assert other.exit_code == 0, other.output
    assert other.output == result.output
    report_b = report_path.read_text(encoding="utf-8")
    assert report_b != report
    assert has_line(report_b, label="unmatched", count=before_span + 3)


def test_the_private_report_counts_first_starts_by_history_label(data: dict) -> None:
    store = data["store"]
    members = write_members(data["members"] / "m.parquet", member_rows(store, MEMBER_START))
    result = invoke(members)
    assert result.exit_code == 0, result.output
    report = (paths.reports_dir() / jc.REPORT_NAME).read_text(encoding="utf-8")

    # each runner of the fixture is a horse's only start; the store starts on its first race day
    runs = store.runners.merge(store.races[["race_id", "race_class"]], on="race_id")
    days = pd.to_datetime(runs["race_date"])
    first_year = days < days.min() + pd.DateOffset(years=1)
    debut = runs["race_class"].eq("DEBUT")
    counts = {
        jc.VERIFIED_DEBUT: int(debut.sum()),
        jc.STORE_START: int((~debut & first_year).sum()),
        jc.UNVERIFIED: int((~debut & ~first_year).sum()),
    }
    assert len(set(counts.values())) == 3 and min(counts.values()) > 0
    for label, n in counts.items():
        assert has_line(report, label=label, count=n), (label, n)
    assert sum(counts.values()) == runs["horse_id"].nunique()
