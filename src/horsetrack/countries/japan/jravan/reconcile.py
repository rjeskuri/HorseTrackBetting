"""The jravan store against the netkeiba (`jra`) store, race by race (`horsetrack japan jravan
reconcile`, #373).

`reconcile(start, end)` works one calendar year at a time. It reads the `races` and `runners` of
both stores through `read_store_table(..., source=..., touches_price=False)`, so holdout days are
never returned and rows before D10's price era are read (a data check the owner asked for; no
model or fit uses it), and the `dividends` of the race ids those reads returned, the id filter
pushed into the scan. Per year it counts:

- races by `race_id`: in both stores or in one only; of the matched races, those whose
  `RACE_FIELDS` are equal, `post_time_local` compared as minutes of the day;
- runners by (race_id, horse_no), the same way over `RUNNER_FIELDS`, the `TENTHS` fields compared
  after rounding to the tenth;
- dividends per pool by (race_id, pool, combination), and of the matched rows the equal ones;
- matched races whose runners match one to one with every Win odds equal (`win_odds_agree`), and
  whose dividend rows match one to one with every dividend equal (`payouts_agree`);
- rows whose key occurs more than once in one store (`<table>.<source>_duplicate`); they are still
  matched row by row.

Equal means both values equal or both null. A match rate is `both` over the union (`both +
jravan_only + jra_only`); an equal or agree rate is over the matched count of its table. Every
figure comes from a licensed source (D21): `summary()` (the CLI's text) holds overall counts and
rates only, and `to_markdown()`, the private report with up to `MAX_IDS` race ids per failing
check, goes to `<reports dir>/jravan-reconcile.md` through `write_report`, never the repository.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import typer

from horsetrack.core import paths
from horsetrack.core.eval.reports import LicensedOutputInRepo, write_report
from horsetrack.countries.japan.jravan.dividends import POOLS
from horsetrack.countries.japan.models.v0.store import read_store_table

SOURCES = ("jravan", "jra")
DEFAULT_START = date(2010, 1, 1)
DEFAULT_END = date(2026, 11, 23)
_START_OPTION = datetime(DEFAULT_START.year, DEFAULT_START.month, DEFAULT_START.day)
_END_OPTION = datetime(DEFAULT_END.year, DEFAULT_END.month, DEFAULT_END.day)
RACE_FIELDS = ("distance_m", "surface", "field_size", "going", "post_time_local")
RUNNER_FIELDS = (
    "horse_id", "finish_pos", "is_scratched", "finish_time_s", "win_odds", "win_fav_rank",
    "jockey_code", "horse_wt_kg",
)  # fmt: skip
TENTHS = ("finish_time_s", "win_odds")  # published to the tenth: compared at it
POOL_ORDER = tuple(pool.code for pool in POOLS)
RACE_KEY = ["race_id"]
RUNNER_KEY = ["race_id", "horse_no"]
DIVIDEND_KEY = ["race_id", "pool", "combination"]
PRESENCE = {"both": "both", "left_only": "jravan_only", "right_only": "jra_only"}
MAX_IDS = 20
REPORT_NAME = "jravan-reconcile.md"
REPORT_SOURCES = ("jravan", "jra")
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})")


@dataclass
class Reconciliation:
    """Per-check counts by calendar year of the race, and the race ids failing each check."""

    start: date
    end: date
    by_year: dict[int, dict[str, int]]
    failing: dict[str, int] = field(default_factory=dict)  # check -> races failing it
    examples: dict[str, list[str]] = field(default_factory=dict)  # up to MAX_IDS, by race id

    @property
    def overall(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for counts in self.by_year.values():
            for key, n in counts.items():
                total[key] = total.get(key, 0) + n
        return total

    def summary(self) -> str:
        """The overall counts and rates; no race id and no value of any field."""
        o = self.overall
        lines = [
            f"jravan vs jra store, {self.start.isoformat()} to {self.end.isoformat()} (both "
            "inclusive, holdout days not read)",
            "match = in both / (in both + jravan only + jra only); equal and agree rates are of "
            "those in both",
        ]
        for table in ("races", "runners"):
            lines.append(_presence_line(table, o, f"{table}."))
            fields = RACE_FIELDS if table == "races" else RUNNER_FIELDS
            both = f"{table}.both"
            for f in fields:
                lines.append(f"  {f} equal: {_of(o, f'{table}.equal.{f}', both)}")
            if table == "races":
                lines.append(f"  every Win odds agree: {_of(o, 'races.win_odds_agree', both)}")
                lines.append(f"  every payout agrees: {_of(o, 'races.payouts_agree', both)}")
        for pool in _pools(o):
            p = f"dividends.{pool}."
            equal = _of(o, p + "equal", p + "both")
            lines.append(f"{_presence_line(f'dividends {pool}', o, p)}; equal: {equal}")
        repeats = "; ".join(
            f"{table} " + " ".join(f"{_n(o.get(f'{table}.{s}_duplicate', 0))} {s}" for s in SOURCES)
            for table in ("races", "runners", "dividends")
        )
        lines.append(f"rows whose key repeats in one store: {repeats}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The private report: counts and rates overall and by year, and up to `MAX_IDS` race ids
        failing each check."""
        o, years = self.overall, sorted(self.by_year)
        keys = _ordered_keys(o)
        out = [
            "# jravan vs jra store reconciliation",
            "",
            f"Range {self.start.isoformat()} to {self.end.isoformat()}, both inclusive, by the "
            "calendar year of the race; holdout days are never read. Built on JRA-VAN DataLab "
            "(licensed, D21): keep it under `$HORSETRACK_DATA_DIR`, never in the repository.",
            "",
            "Match rate: in both over the union (in both + jravan only + jra only). Equal and "
            "agree rates: over the matched count of the table (races, runners, or the pool's "
            "dividends). Equal means both values equal or both null; `post_time_local` compares "
            "as minutes of the day, `finish_time_s` and `win_odds` at the tenth. "
            "`<table>.<source>_duplicate` counts the rows whose key repeats in that store.",
            "",
            "## Counts",
            "",
            "| Check | Overall | " + " | ".join(str(y) for y in years) + " |",
            "|---|" + "---:|" * (len(years) + 1),
        ]
        for key in keys:
            row = [_n(o.get(key, 0)), *(_n(self.by_year[y].get(key, 0)) for y in years)]
            out.append(f"| `{key}` | " + " | ".join(row) + " |")
        out += [
            "",
            "## Rates",
            "",
            "| Rate | Overall | " + " | ".join(str(y) for y in years) + " |",
            "|---|" + "---:|" * (len(years) + 1),
        ]
        for name, num, den in _rate_keys(o):
            row = [_rate(o, num, den), *(_rate(self.by_year[y], num, den) for y in years)]
            out.append(f"| `{name}` | " + " | ".join(row) + " |")
        out += ["", f"## Races failing each check (up to {MAX_IDS}, by race id)", ""]
        failing = [key for key in keys if self.failing.get(key)]
        if not failing:
            out.append("None: every check holds for every race.")
        for key in failing:
            ids = self.examples.get(key, [])
            out.append(f"- `{key}`: {_n(self.failing[key])} races; {', '.join(ids)}")
        return "\n".join(out) + "\n"


def reconcile(start: date = DEFAULT_START, end: date = DEFAULT_END) -> Reconciliation:
    """Compare the jravan and jra stores' races, runners and dividends over `start <= race_date
    <= end`, one calendar year at a time; holdout days are never read."""
    by_year: dict[int, dict[str, int]] = {}
    failing: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for lo, hi in _years(start, end):
        counts, bad = _compare_year(lo, hi)
        if any(counts.values()):
            by_year[lo.year] = dict(counts)
        for key, ids in bad.items():
            failing[key] += len(ids)
            listed = examples.setdefault(key, [])
            listed += sorted(ids)[: MAX_IDS - len(listed)]
    return Reconciliation(start, end, by_year, dict(failing), examples)


# ------------------------------------------------------------------------------------------------
# One year


def _years(start: date, end: date) -> Iterator[tuple[date, date]]:
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def _compare_year(lo: date, hi: date) -> tuple[Counter[str], dict[str, set[str]]]:
    counts: Counter[str] = Counter()
    bad: dict[str, set[str]] = {}

    def fail(key: str, ids) -> None:
        ids = set(ids)
        if ids:
            bad.setdefault(key, set()).update(ids)

    frames = _read("races", RACE_KEY + list(RACE_FIELDS), lo, hi)
    races = _merge("races", frames, RACE_KEY, counts, fail)
    _presence("races.", races, counts, fail)
    matched = races[races["_merge"] == "both"]
    for f in RACE_FIELDS:
        _equal_count(f"races.equal.{f}", matched, f, counts, fail)
    matched_ids = set(matched["race_id"])

    frames = _read("runners", RUNNER_KEY + list(RUNNER_FIELDS), lo, hi)
    runners = _merge("runners", frames, RUNNER_KEY, counts, fail)
    _presence("runners.", runners, counts, fail)
    both = runners[runners["_merge"] == "both"]
    for f in RUNNER_FIELDS:
        _equal_count(f"runners.equal.{f}", both, f, counts, fail)
    odds_bad = set(runners.loc[runners["_merge"] != "both", "race_id"])
    odds_bad |= set(both.loc[~_equal(both, "win_odds"), "race_id"])
    _agree("races.win_odds_agree", matched_ids, odds_bad, counts, fail)

    ids = sorted(set(races["race_id"]))
    frames = {s: _read_dividends(s, ids) for s in SOURCES}
    dividends = _merge("dividends", frames, DIVIDEND_KEY, counts, fail)
    pools = sorted(set(dividends["pool"]), key=_pool_rank)
    for pool in pools:
        rows = dividends[dividends["pool"] == pool]
        prefix = f"dividends.{pool}."
        _presence(prefix, rows, counts, fail)
        _equal_count(prefix + "equal", rows[rows["_merge"] == "both"], "dividend", counts, fail)
    pay_bad = set(dividends.loc[dividends["_merge"] != "both", "race_id"])
    paired = dividends[dividends["_merge"] == "both"]
    pay_bad |= set(paired.loc[~_equal(paired, "dividend"), "race_id"])
    _agree("races.payouts_agree", matched_ids, pay_bad, counts, fail)
    return counts, bad


def _read(name: str, columns: Sequence[str], lo: date, hi: date) -> dict[str, pd.DataFrame]:
    return {
        s: read_store_table(
            name, columns=columns, start=lo, end=hi, touches_price=False, source=s
        ).to_pandas()
        for s in SOURCES
    }


def _read_dividends(source: str, race_ids: list[str]) -> pd.DataFrame:
    """The source's dividend rows of `race_ids` only (the filter is pushed into the scan)."""
    dataset = ds.dataset(paths.processed_dir(source) / "dividends.parquet", format="parquet")
    wanted = ds.field("race_id").isin(pa.array(race_ids, pa.string()))
    return dataset.to_table(columns=DIVIDEND_KEY + ["dividend"], filter=wanted).to_pandas()


def _merge(table, frames, key, counts, fail) -> pd.DataFrame:
    """The two stores' rows outer-joined on `key` (`_merge`: both, left_only = jravan, right_only
    = jra), after counting the rows whose key repeats within a store."""
    for s in SOURCES:
        repeated = frames[s].duplicated(key, keep=False)
        counts[f"{table}.{s}_duplicate"] += int(repeated.sum())
        fail(f"{table}.{s}_duplicate", frames[s].loc[repeated, "race_id"])
    return frames["jravan"].merge(
        frames["jra"], on=key, how="outer", suffixes=("_jravan", "_jra"), indicator=True
    )


def _presence(prefix, merged, counts, fail) -> None:
    for state, check in PRESENCE.items():
        rows = merged[merged["_merge"] == state]
        counts[prefix + check] += len(rows)
        if check != "both":
            fail(prefix + check, rows["race_id"])


def _equal_count(key, matched, f, counts, fail) -> None:
    equal = _equal(matched, f)
    counts[key] += int(equal.sum())
    fail(key, matched.loc[~equal, "race_id"])


def _agree(key, matched_ids, bad_ids, counts, fail) -> None:
    counts[key] += len(matched_ids - bad_ids)
    fail(key, matched_ids & bad_ids)


def _equal(matched: pd.DataFrame, f: str) -> np.ndarray:
    """Per row, whether `f` is equal in both stores (both null counts as equal)."""
    a, b = (_comparable(f, matched[f"{f}_{s}"]) for s in SOURCES)
    a_null, b_null = a.isna().to_numpy(), b.isna().to_numpy()
    same = np.asarray(a.to_numpy(dtype=object) == b.to_numpy(dtype=object), dtype=bool)
    return (a_null & b_null) | (~a_null & ~b_null & same)


def _comparable(f: str, values: pd.Series) -> pd.Series:
    if f in TENTHS:
        return np.rint(values.astype(float) * 10)
    if f == "post_time_local":
        return values.map(_minutes)
    return values


def _minutes(value):
    """'9:50' and '09:50' -> 590; a null stays null; any other text is compared as written."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = _CLOCK.fullmatch(str(value).strip())
    return int(match[1]) * 60 + int(match[2]) if match else str(value)


# ------------------------------------------------------------------------------------------------
# Text


def _pool_rank(pool: str) -> tuple[int, str]:
    return (POOL_ORDER.index(pool) if pool in POOL_ORDER else len(POOL_ORDER), pool)


def _pools(counts: Mapping[str, int]) -> list[str]:
    pools = {k.split(".")[1] for k in counts if k.startswith("dividends.") and k.count(".") == 2}
    return sorted(pools, key=_pool_rank)


def _ordered_keys(counts: Mapping[str, int]) -> list[str]:
    keys = []
    for table, fields in (("races", RACE_FIELDS), ("runners", RUNNER_FIELDS)):
        keys += [f"{table}.{c}" for c in PRESENCE.values()]
        keys += [f"{table}.equal.{f}" for f in fields]
        if table == "races":
            keys += ["races.win_odds_agree", "races.payouts_agree"]
        keys += [f"{table}.{s}_duplicate" for s in SOURCES]
    for pool in _pools(counts):
        keys += [f"dividends.{pool}.{c}" for c in (*PRESENCE.values(), "equal")]
    keys += [f"dividends.{s}_duplicate" for s in SOURCES]
    return keys + sorted(set(counts) - set(keys))


def _rate_keys(counts: Mapping[str, int]) -> list[tuple[str, str | tuple[str, ...], str]]:
    """(rate name, numerator key, denominator key or keys summed)."""
    rates: list[tuple[str, str | tuple[str, ...], str]] = []
    for table, fields in (("races", RACE_FIELDS), ("runners", RUNNER_FIELDS)):
        rates.append((f"{table}.match", f"{table}.both", _union(f"{table}.")))
        rates += [(f"{table}.equal.{f}", f"{table}.equal.{f}", f"{table}.both") for f in fields]
        if table == "races":
            rates += [(f"races.{k}", f"races.{k}", "races.both")
                      for k in ("win_odds_agree", "payouts_agree")]  # fmt: skip
    for pool in _pools(counts):
        p = f"dividends.{pool}."
        rates += [(p + "match", p + "both", _union(p)), (p + "equal", p + "equal", p + "both")]
    return rates


def _union(prefix: str) -> tuple[str, ...]:
    return tuple(prefix + c for c in PRESENCE.values())


def _value(counts: Mapping[str, int], keys: str | tuple[str, ...]) -> int:
    return sum(counts.get(k, 0) for k in ((keys,) if isinstance(keys, str) else keys))


def _rate(counts: Mapping[str, int], num: str, den: str | tuple[str, ...]) -> str:
    d = _value(counts, den)
    return f"{100 * counts.get(num, 0) / d:.2f}%" if d else "n/a"


def _of(counts: Mapping[str, int], num: str, den: str) -> str:
    return f"{_n(counts.get(num, 0))} of {_n(_value(counts, den))} ({_rate(counts, num, den)})"


def _presence_line(label: str, counts: Mapping[str, int], prefix: str) -> str:
    both, jv, nk = (_n(counts.get(prefix + c, 0)) for c in PRESENCE.values())
    match = _rate(counts, prefix + "both", _union(prefix))
    return f"{label}: {both} in both; {jv} jravan only; {nk} jra only; match {match}"


def _n(value: int) -> str:
    return f"{value:,}"


# ------------------------------------------------------------------------------------------------
# Command


def reconcile_command(
    start: Annotated[
        datetime, typer.Option("--start", formats=["%Y-%m-%d"], help="First race day (inclusive).")
    ] = _START_OPTION,
    end: Annotated[
        datetime, typer.Option("--end", formats=["%Y-%m-%d"], help="Last race day (inclusive).")
    ] = _END_OPTION,
) -> None:
    """Reconcile the jravan store with the netkeiba (jra) store: print the overall counts and
    rates; the detail by year, with failing race ids, goes to <reports dir>/jravan-reconcile.md."""
    try:
        rec = reconcile(start.date(), end.date())
        write_report(paths.reports_dir() / REPORT_NAME, rec.to_markdown(), sources=REPORT_SOURCES)
    except (paths.DataDirNotSet, LicensedOutputInRepo, ValueError, KeyError, OSError) as exc:
        _die(f"jravan reconcile failed: {type(exc).__name__}: {exc}")
    typer.echo(rec.summary())
    typer.echo(f"report: $HORSETRACK_DATA_DIR/reports/{REPORT_NAME}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
