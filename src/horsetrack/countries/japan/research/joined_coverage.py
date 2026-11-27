"""The population that reaches a model after the point-in-time joins (ticket #418).

The #374 inventory (`research.coverage`) measures each column on its own; this counts races, per
source, through the joins a model needs. `joined_coverage(schedule, sources)` is pure, on frames:

- `schedule`: one row per scheduled runner (`race_id`, `horse_no`, `race_date`, `race_class`).
  A race on a holdout day (`is_holdout_day("japan", d)`, D14) is in no count, and neither is a
  source row of it, nor one whose race id names a holdout day: this is a development report.
- `sources`: `{name: frame}`, one row per runner the source holds after the joins (`race_id`,
  `horse_no`, `forecast`, NaN when the runner has none, and `won`, missing when no result), and
  optionally `horse_id`: the source's own key, on which its repeats are counted (a row whose
  `horse_no` is missing, not a scheduled starter, is then a source row that is not scheduled).
- `study`: optionally `{name: StudyRaces}`, the races a study keeps for a source: a race outside
  its store frame or not scored by it is not scored here either.

Every scheduled race passes, per source, `STAGES` in order and drops at the first it fails, with
one reason (`REASONS`): matched fails on `duplicate_key` (a key of the race occurs more than once
in the source), else on `unmatched` (a scheduled runner has no source row); forecastable fails,
for a source with a `StudyRaces`, on `not_in_store_frame` (the study's store frame drops the race,
so the study forecasts none of it), else on `no_forecast` (a scheduled runner's forecast is
missing); scored fails on `no_result` (a scheduled runner's `won` is missing, or no runner won; a
dead heat is scored), else, with a `StudyRaces`, on `train_only` (the study only trains on the
race: its scored folds test none of it). Counts split by era
(`PRICE_ERA` from D10's `PRICE_ERA_START` on, `PRE_PRICE_ERA` before) and by card class. Key
counts: source rows whose key repeats (every copy), scheduled keys with no source row, and source
rows whose key is not scheduled.

`render_summary` is the committed view: a licensed source (`reports.is_withheld((s,))`: JRA-VAN,
JRDB) shows `coverage.WITHHELD` and no figure (D21). `render_report` is the private view with every
figure, written by `write_private_report` through `write_report`, never into the repository.

`first_starts(runs)` labels each horse's first observed start (the history gaps of #377/#403):
a `VERIFIED_DEBUT` in a DEBUT (新馬) race; a `STORE_START` in the store's first year, so earlier
runs may predate the store; otherwise `UNVERIFIED` (a maiden debut, or runs outside JRA).

`horsetrack japan joined-coverage --members PATH` (`joined_coverage_command`) runs it on the store:
the schedule is the store's development starters of the price era (`studies.meta.read_development`),
the sources the store's final Win price (`jra`) and the #87 member file (`MEMBER_SOURCE`,
`studies.meta.read_members`), the latter under the #87 study's own rules (`member_study_races`:
its `store_frame`, scored folds and level-2 population); the private report adds `first_starts`
over every store run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from horsetrack.core import paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_withheld, write_report
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research.coverage import WITHHELD
from horsetrack.countries.japan.studies import meta as study

COUNTRY = "japan"
REPORT_NAME = "joined-coverage.md"
STAGES = ("scheduled", "matched", "forecastable", "scored")
REASONS = (  # in the order a race is tested: the first that fails is its reason
    "duplicate_key",
    "unmatched",
    "not_in_store_frame",
    "no_forecast",
    "no_result",
    "train_only",
)
PRICE_ERA = "price_era"
PRE_PRICE_ERA = "pre_price_era"
ERAS = (PRE_PRICE_ERA, PRICE_ERA)
KEY = ["race_id", "horse_no"]
DEBUT = "DEBUT"
VERIFIED_DEBUT = "verified_debut"
STORE_START = "store_start"
UNVERIFIED = "unverified"
HISTORY_LABELS = (VERIFIED_DEBUT, STORE_START, UNVERIFIED)
MEMBER_SOURCE = "hktrack_members_jrdb"  # the #87 member file: JRDB-derived, so withheld (D21)

# the last stage a race reaches when it drops for a reason; a race that drops for none is scored
_REACHED = {
    "duplicate_key": 0,
    "unmatched": 0,
    "not_in_store_frame": 1,
    "no_forecast": 1,
    "no_result": 2,
    "train_only": 2,
    None: 3,
}
_RACE_ID_DAY = re.compile(r"^[^:]*:(\d{4}-\d{2}-\d{2}):")  # `<source>:<YYYY-MM-DD>:...`


@dataclass(frozen=True)
class StudyRaces:
    """The races of a study's store frame and the races it scores (a subset)."""

    in_store_frame: frozenset[str]
    scored: frozenset[str]


@dataclass(frozen=True, eq=False)
class JoinedCoverage:
    """Per scheduled development race its era and class (`races_frame`), and per source the
    reason it dropped (None: scored) and the key counts."""

    races_frame: pd.DataFrame  # race_id, era, race_class, then one reason column per source
    keys: Mapping[str, Mapping[str, int]]  # source -> duplicate_rows, unmatched_*

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.keys)

    def _rows(self, era: str | None, race_class: str | None) -> pd.DataFrame:
        rows = self.races_frame
        if era is not None:
            rows = rows.loc[rows["era"] == era]
        if race_class is not None:
            rows = rows.loc[rows["race_class"] == race_class]
        return rows

    def _reason(self, source: str) -> str:
        if source not in self.keys:
            raise KeyError(f"no source {source!r}; the sources are {', '.join(self.sources)}")
        return f"reason:{source}"

    def races(
        self, source: str, stage: str, *, era: str | None = None, race_class: str | None = None
    ) -> int:
        """The races that reached `stage` (cumulative), optionally of one era and card class."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
        column = self._reason(source)
        reached = self._rows(era, race_class)[column].map(lambda r: _REACHED[r])
        return int((reached >= STAGES.index(stage)).sum())

    def drops(
        self, source: str, reason: str, *, era: str | None = None, race_class: str | None = None
    ) -> int:
        """The races dropped for `reason`, optionally of one era and card class."""
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}, not {reason!r}")
        column = self._reason(source)
        return int((self._rows(era, race_class)[column] == reason).sum())

    def duplicate_rows(self, source: str) -> int:
        self._reason(source)
        return self.keys[source]["duplicate_rows"]

    def unmatched_scheduled(self, source: str) -> int:
        self._reason(source)
        return self.keys[source]["unmatched_scheduled"]

    def unmatched_source(self, source: str) -> int:
        self._reason(source)
        return self.keys[source]["unmatched_source"]

    @property
    def classes(self) -> list[str]:
        return sorted(self.races_frame["race_class"].astype(str).unique())


def _check_columns(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{what} has no column(s): {', '.join(missing)}")


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEY])


def _holdout_id(race_id: object) -> bool:
    """Whether a race id names a holdout day (`<source>:<YYYY-MM-DD>:...`)."""
    match = _RACE_ID_DAY.match(str(race_id))
    if match is None:
        return False
    try:
        day = date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return is_holdout_day(COUNTRY, day)


def joined_coverage(
    schedule: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    *,
    study: Mapping[str, StudyRaces] | None = None,
) -> JoinedCoverage:
    """Each source's races through `STAGES` and its key counts (see the module doc)."""
    study = study or {}
    _check_columns(schedule, ["race_id", "horse_no", "race_date", "race_class"], "the schedule")
    days = pd.to_datetime(schedule["race_date"])
    if days.isna().any():
        raise ValueError("a scheduled runner has no race_date")
    schedule = schedule.assign(race_date=days.dt.date)
    holdout_days = {d for d in schedule["race_date"].unique() if is_holdout_day(COUNTRY, d)}
    holdout_ids = set(schedule.loc[schedule["race_date"].isin(holdout_days), "race_id"])
    schedule = schedule.loc[~schedule["race_id"].isin(holdout_ids)]

    races = schedule.groupby("race_id", sort=True).agg(
        race_date=("race_date", "first"), race_class=("race_class", "first")
    )
    races["era"] = [
        PRICE_ERA if d >= PRICE_ERA_START else PRE_PRICE_ERA for d in races["race_date"]
    ]
    scheduled = schedule.drop_duplicates(KEY)
    scheduled_keys = _keys(scheduled)

    frame = races[["era", "race_class"]].copy()
    keys: dict[str, dict[str, int]] = {}
    for name, rows in sources.items():
        _check_columns(rows, ["race_id", "horse_no", "forecast", "won"], f"source {name!r}")
        named = {r for r in rows["race_id"].unique() if _holdout_id(r)}
        rows = rows.loc[~rows["race_id"].isin(holdout_ids | named)]
        reason, counts = _source(scheduled, scheduled_keys, races.index, rows, study.get(name))
        frame[f"reason:{name}"] = reason
        keys[name] = counts
    return JoinedCoverage(races_frame=frame.reset_index(), keys=keys)


def _source(
    scheduled: pd.DataFrame,
    scheduled_keys: pd.MultiIndex,
    race_ids: pd.Index,
    rows: pd.DataFrame,
    study: StudyRaces | None,
) -> tuple[pd.Series, dict[str, int]]:
    """Per scheduled race the reason it drops in `rows` (None: scored), and the key counts."""
    row_keys = _keys(rows)
    own_key = ["race_id", "horse_id"] if "horse_id" in rows.columns else KEY
    repeated = rows.duplicated(own_key, keep=False).to_numpy()
    in_schedule = row_keys.isin(scheduled_keys)
    missing = ~scheduled_keys.isin(row_keys)

    # the source row of each scheduled runner, where it has exactly one
    single = rows.loc[in_schedule & ~repeated]
    no_forecast = set(single.loc[single["forecast"].isna(), "race_id"])
    winners = set(single.loc[single["won"].eq(True), "race_id"])
    no_result = set(single.loc[single["won"].isna(), "race_id"]) | (
        set(single["race_id"]) - winners
    )
    failing = {
        "duplicate_key": set(rows.loc[repeated & rows["race_id"].isin(race_ids), "race_id"]),
        "unmatched": set(scheduled.loc[missing, "race_id"]),
        "no_forecast": no_forecast,
        "no_result": no_result,
        "not_in_store_frame": set(),
        "train_only": set(),
    }
    if study is not None:
        failing["not_in_store_frame"] = set(race_ids) - study.in_store_frame
        failing["train_only"] = set(study.in_store_frame) - study.scored
    reason = pd.Series([None] * len(race_ids), index=race_ids, dtype=object)
    for why in reversed(REASONS):  # the first failing stage wins
        reason[reason.index.isin(list(failing[why]))] = why
    counts = {
        "duplicate_rows": int(repeated.sum()),
        "unmatched_scheduled": int(missing.sum()),
        "unmatched_source": int((~in_schedule).sum()),
    }
    return reason, counts


# ------------------------------------------------------------------------------------------------
# Rendering (D21)

_INTRO = (
    "Each scheduled development race (holdout days, D14, are in no count) passes, per source, the "
    "stages scheduled, matched, forecastable and scored, and drops at the first it fails: "
    "`duplicate_key` (a runner key repeats in the source) or `unmatched` (a scheduled runner has "
    "no source row) at matched, `not_in_store_frame` (for a source a study reads: the study's "
    "store frame drops the race) or `no_forecast` (a runner has no forecast after the joins) at "
    "forecastable, `no_result` (a runner has no result, or no runner won) or `train_only` (the "
    "study only trains on the race) at scored. Eras split "
    f"at D10's price era start, {PRICE_ERA_START.isoformat()}. Built by "
    "`research.joined_coverage` (#418)."
)


def _cell(n: int) -> str:
    return f"{n:,}"


def _source_lines(cov: JoinedCoverage, s: str) -> list[str]:
    lines = ["| stage | all | " + " | ".join(ERAS) + " |", "|" + " --- |" * (len(ERAS) + 2)]
    for stage in STAGES:
        figures = [cov.races(s, stage, era=e) for e in (None, *ERAS)]
        lines.append(f"| {stage} | " + " | ".join(_cell(n) for n in figures) + " |")
    for why in REASONS:
        figures = [cov.drops(s, why, era=e) for e in (None, *ERAS)]
        lines.append(f"| drop: {why} | " + " | ".join(_cell(n) for n in figures) + " |")
    lines += [
        "",
        f"Keys: {_cell(cov.duplicate_rows(s))} duplicate source rows, "
        f"{_cell(cov.unmatched_scheduled(s))} scheduled runners with no source row, "
        f"{_cell(cov.unmatched_source(s))} source rows not scheduled.",
        "",
        "| card class | " + " | ".join(STAGES) + " | " + " | ".join(REASONS) + " |",
        "|" + " --- |" * (1 + len(STAGES) + len(REASONS)),
    ]
    for cls in cov.classes:
        figures = [cov.races(s, stage, race_class=cls) for stage in STAGES]
        figures += [cov.drops(s, why, race_class=cls) for why in REASONS]
        lines.append(f"| {cls} | " + " | ".join(_cell(n) for n in figures) + " |")
    return lines


def render_summary(cov: JoinedCoverage) -> str:
    """The committed view: no figure of a licensed source (D21)."""
    lines = ["# Joined forecast coverage", "", _INTRO, ""]
    for s in cov.sources:
        lines.append(f"## {s}")
        lines.append("")
        lines += [WITHHELD] if is_withheld((s,)) else _source_lines(cov, s)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report(cov: JoinedCoverage, history: pd.DataFrame | None = None) -> str:
    """The private view: every figure, the licensed ones included, then, given `first_starts`'
    `history`, the horses under each history label. Never inside the repository."""
    lines = [
        "# Joined forecast coverage: private report",
        "",
        _INTRO,
        "It holds JRA-VAN and JRDB figures (D21): keep it under $HORSETRACK_DATA_DIR, never in "
        "git, a ticket or a PR.",
        "",
    ]
    for s in cov.sources:
        lines.append(f"## {s}" + (" (licensed)" if is_withheld((s,)) else ""))
        lines.append("")
        lines += _source_lines(cov, s)
        lines.append("")
    if history is not None:
        counts = history["history"].value_counts()
        lines += [
            "## First observed starts by history label",
            "",
            f"Horses by the label of their first observed start in the store (`first_starts`): "
            f"`{VERIFIED_DEBUT}` in a DEBUT race, `{STORE_START}` in the store's first year "
            f"(earlier runs may predate the store), `{UNVERIFIED}` otherwise (#377, #403).",
            "",
            "| history | horses |",
            "| --- | --- |",
            *(f"| {label} | {_cell(int(counts.get(label, 0)))} |" for label in HISTORY_LABELS),
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_private_report(
    cov: JoinedCoverage, path: Path | str | None = None, *, history: pd.DataFrame | None = None
) -> Path:
    """`render_report(cov, history)` through `write_report`, by default to the reports folder."""
    target = paths.reports_dir() / REPORT_NAME if path is None else Path(path)
    return write_report(target, render_report(cov, history), sources=cov.sources)


# ------------------------------------------------------------------------------------------------
# History: verified debuts against first observed starts


def first_starts(runs: pd.DataFrame, *, store_start: date | None = None) -> pd.DataFrame:
    """One row per horse with a start (a row not scratched): `horse_id`, `first_start` (the day
    of its earliest start) and `history` (see the module doc). `store_start` defaults to the
    earliest `race_date` in `runs`, scratched rows included."""
    _check_columns(runs, ["horse_id", "race_date", "race_class", "is_scratched"], "runs")
    days = pd.to_datetime(runs["race_date"])
    if days.isna().any():
        raise ValueError("a run has no race_date")
    start = days.min() if store_start is None else pd.Timestamp(store_start)
    store_year_end = start + pd.DateOffset(years=1)
    scratched = runs["is_scratched"].fillna(False).astype(bool)
    starts = runs.assign(race_date=days).loc[~scratched]
    first = starts.sort_values(["horse_id", "race_date"], kind="stable").drop_duplicates("horse_id")
    history = [
        VERIFIED_DEBUT if cls == DEBUT else STORE_START if day < store_year_end else UNVERIFIED
        for cls, day in zip(first["race_class"], first["race_date"], strict=True)
    ]
    return pd.DataFrame(
        {
            "horse_id": first["horse_id"].to_numpy(),
            "first_start": first["race_date"].dt.date.to_numpy(),
            "history": history,
        }
    )


# ------------------------------------------------------------------------------------------------
# The command: the store and the #87 member file


def store_sources(
    runners: pd.DataFrame, races: pd.DataFrame, members: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The schedule, the store's starters (scratched runners left out), and two sources on it:
    `jra`, the final Win price as the forecast, and `MEMBER_SOURCE`, the #87 member file keyed to
    the store's horse numbers and keyed on the file's own (race_id, horse_id) (its forecast
    missing when any member's is; a member row that is not a store starter keeps no horse number).
    `won` is the store's: missing for a race not run, otherwise whether the runner finished
    first."""
    scratched = runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    starters = runners.loc[~scratched].merge(
        races[["race_id", "race_class", "status"]], on="race_id", how="left"
    )
    first = starters["finish_pos"].eq(1).to_numpy(dtype=bool, na_value=False)
    won = np.where(starters["status"].eq("run").to_numpy(dtype=bool, na_value=False), first, np.nan)
    schedule = starters[["race_id", "horse_no", "race_date", "race_class"]]
    store = starters[["race_id", "horse_no"]].assign(
        forecast=1.0 / clean_odds(starters["win_odds"].to_numpy()), won=won
    )
    member_cols = [col for col, _ in study.MEMBERS.values()]
    numbers = starters[["race_id", "horse_id", "horse_no"]].assign(won=won)
    member = (
        members[["race_id", "horse_id"]]
        .assign(forecast=members[member_cols].mean(axis=1, skipna=False).to_numpy())
        .merge(numbers, on=["race_id", "horse_id"], how="left", validate="many_to_one")
    )
    return schedule, {"jra": store, MEMBER_SOURCE: member[[*KEY, "horse_id", "forecast", "won"]]}


def member_study_races(
    runners: pd.DataFrame, races: pd.DataFrame, members: pd.DataFrame
) -> StudyRaces:
    """The #87 study's races for the member file, by its own code: `studies.meta.store_frame`,
    its scored folds (`study_folds`) and its level-2 population (`level2_population`), whose
    scored folds' test races are the ones it scores. A (race_id, horse_id) the file repeats is
    left out of the population here (its race drops as `duplicate_key` first)."""
    frame, _ = study.store_frame(runners, races)
    _, scored = study.study_folds(frame, load_country_scheme(COUNTRY))
    unique = members.loc[~members.duplicated(["race_id", "horse_id"], keep=False)]
    level2 = study.level2_population(frame, scored, unique, study.race_calendar())
    tested = [np.asarray(f.test_idx, dtype=int) for f in level2.folds]
    rows = np.unique(np.concatenate(tested)) if tested else np.array([], dtype=int)
    return StudyRaces(
        in_store_frame=frozenset(frame["race_id"]),
        scored=frozenset(level2.frame["race_id"].iloc[rows]),
    )


def read_runs() -> pd.DataFrame:
    """Every development run in the store (holdout days dropped, no price read), for
    `first_starts`: race_id, race_date, horse_id, is_scratched and the race's race_class."""
    runners, races = (
        read_store_table(
            name, columns=columns, start=date.min, end=date.max, touches_price=False
        ).to_pandas(date_as_object=False)
        for name, columns in (
            ("runners", ["race_id", "race_date", "horse_id", "is_scratched"]),
            ("races", ["race_id", "race_class"]),
        )
    )
    return runners.merge(races, on="race_id", how="left")


def joined_coverage_command(
    members: Annotated[
        Path,
        typer.Option("--members", help="The #87 member file (JRDB-derived: withheld, D21)."),
    ],
) -> None:
    """Count each source's development races through the joins, the store's and the #87 member
    file's: the private report, with every figure and the first starts by history label, goes to
    <reports dir>/joined-coverage.md; only the summary, which withholds licensed figures, prints."""
    try:
        runners, races = study.read_development(
            [*study.RUNNER_COLUMNS, "horse_no"], [*study.RACE_COLUMNS, "race_class"]
        )
        member_rows = study.read_members(members)
        schedule, sources = store_sources(runners, races, member_rows)
        rules = {MEMBER_SOURCE: member_study_races(runners, races, member_rows)}
        cov = joined_coverage(schedule, sources, study=rules)
        write_private_report(cov, history=first_starts(read_runs()))
    except (paths.DataDirNotSet, LicensedOutputInRepo, ValueError, KeyError, OSError) as exc:
        typer.echo(f"ERROR joined-coverage failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(render_summary(cov), nl=False)
