"""The population that reaches a model after the point-in-time joins (ticket #418).

The #374 inventory (`research.coverage`) measures each column on its own; this counts races, per
source, through the joins a model needs. `joined_coverage(schedule, sources)` is pure, on frames:

- `schedule`: one row per scheduled runner (`race_id`, `horse_no`, `race_date`, `race_class`).
  A race on a holdout day (`is_holdout_day("japan", d)`, D14) is in no count, and neither is a
  source row of it, nor one whose race id names a holdout day: this is a development report.
- `sources`: `{name: frame}`, one row per runner the source holds after the joins (`race_id`,
  `horse_no`, `forecast`, NaN when the runner has none, and `won`, missing when no result).

Every scheduled race passes, per source, `STAGES` in order and drops at the first it fails, with
one reason (`REASONS`): matched fails on `duplicate_key` (a key of the race occurs more than once
in the source), else on `unmatched` (a scheduled runner has no source row); forecastable fails on
`no_forecast` (a scheduled runner's forecast is missing); scored fails on `no_result` (a scheduled
runner's `won` is missing). Counts split by era (`PRICE_ERA` from D10's `PRICE_ERA_START` on,
`PRE_PRICE_ERA` before) and by card class. Key counts: source rows whose key repeats (every copy),
scheduled keys with no source row, and source rows whose key is not scheduled.

`render_summary` is the committed view: a licensed source (`reports.is_withheld((s,))`: JRA-VAN,
JRDB) shows `coverage.WITHHELD` and no figure (D21). `render_report` is the private view with every
figure, written by `write_private_report` through `write_report`, never into the repository.

`first_starts(runs)` labels each horse's first observed start (the history gaps of #377/#403):
a `VERIFIED_DEBUT` in a DEBUT (新馬) race; a `STORE_START` in the store's first year, so earlier
runs may predate the store; otherwise `UNVERIFIED` (a maiden debut, or runs outside JRA).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import is_withheld, write_report
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.research.coverage import WITHHELD

COUNTRY = "japan"
REPORT_NAME = "joined-coverage.md"
STAGES = ("scheduled", "matched", "forecastable", "scored")
REASONS = ("duplicate_key", "unmatched", "no_forecast", "no_result")
PRICE_ERA = "price_era"
PRE_PRICE_ERA = "pre_price_era"
ERAS = (PRE_PRICE_ERA, PRICE_ERA)
KEY = ["race_id", "horse_no"]
DEBUT = "DEBUT"
VERIFIED_DEBUT = "verified_debut"
STORE_START = "store_start"
UNVERIFIED = "unverified"

# the last stage a race reaches when it drops for a reason; a race that drops for none is scored
_REACHED = {"duplicate_key": 0, "unmatched": 0, "no_forecast": 1, "no_result": 2, None: 3}
_RACE_ID_DAY = re.compile(r"^[^:]*:(\d{4}-\d{2}-\d{2}):")  # `<source>:<YYYY-MM-DD>:...`


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


def joined_coverage(schedule: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> JoinedCoverage:
    """Each source's races through `STAGES` and its key counts (see the module doc)."""
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
        reason, counts = _source(scheduled, scheduled_keys, races.index, rows)
        frame[f"reason:{name}"] = reason
        keys[name] = counts
    return JoinedCoverage(races_frame=frame.reset_index(), keys=keys)


def _source(
    scheduled: pd.DataFrame,
    scheduled_keys: pd.MultiIndex,
    race_ids: pd.Index,
    rows: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """Per scheduled race the reason it drops in `rows` (None: scored), and the key counts."""
    row_keys = _keys(rows)
    repeated = rows.duplicated(KEY, keep=False).to_numpy()
    in_schedule = row_keys.isin(scheduled_keys)
    missing = ~scheduled_keys.isin(row_keys)

    # the source row of each scheduled runner, where it has exactly one
    single = rows.loc[in_schedule & ~repeated]
    no_forecast = set(single.loc[single["forecast"].isna(), "race_id"])
    no_result = set(single.loc[single["won"].isna(), "race_id"])
    failing = {
        "duplicate_key": set(rows.loc[repeated & rows["race_id"].isin(race_ids), "race_id"]),
        "unmatched": set(scheduled.loc[missing, "race_id"]),
        "no_forecast": no_forecast,
        "no_result": no_result,
    }
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
    "no source row) at matched, `no_forecast` (a runner has no forecast after the joins) at "
    "forecastable, `no_result` (a runner has no result) at scored. Eras split at D10's price era "
    f"start, {PRICE_ERA_START.isoformat()}. Built by `research.joined_coverage` (#418)."
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


def render_report(cov: JoinedCoverage) -> str:
    """The private view: every figure, the licensed ones included. Never inside the repository."""
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
    return "\n".join(lines).rstrip() + "\n"


def write_private_report(cov: JoinedCoverage, path: Path | str | None = None) -> Path:
    """`render_report(cov)` through `write_report`, by default to the reports folder."""
    target = paths.reports_dir() / REPORT_NAME if path is None else Path(path)
    return write_report(target, render_report(cov), sources=cov.sources)


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
