"""The WIN5 carryover and its pool expected return (ticket #135).

- Pool arithmetic: `pool_ev` is the gross return per 1 yen of the whole pool,
  `(pool * (1 - takeout) + carryover) / pool`; `chased_pool_ev` the same on the pool a carryover
  draws (`P0 + chase * C`); `break_even_carryover` where the chased figure reaches 1. These take
  WIN5's standard 30% unless told otherwise.
- The day's takeout and the cap: `win5_takeout(day)` is WIN5's takeout on a day (None before
  2014-06-07), the boost calendar (countries/japan/takeout.py) applied; `capped_payout` splits the
  pool after takeout plus the carryover into the payout and the carryover out under the 6億円 per
  100-yen cap (`PAYOUT_CAP_YEN`); `payout_mismatches` names the history days whose money does not
  add up at their own takeout.
- History: one row per WIN5 day (pool, carryover in and out, winning tickets, the largest dividend,
  the payout over every winning combination and their number), from results pages:
  `parse_results_page` (one table, columns named by its header) and `parse_day_page` (netkeiba's
  per-day WIN5 page, labelled cells; a dead-heat day lists several winning combinations).
  `build_history` merges them; `write_history` keeps the table at
  `processed_dir('jra') / 'win5.parquet'`, never in the tree. `append_history` grows an existing
  table with new days (the run-by-run growth of the sample, #325): it keeps every old day, adds the
  new ones and refuses a day whose figures changed. `bin_counts` reports the development days per
  carryover bin of a study spec, by the scorer's own binning and without reading a holdout day's
  outcomes.
- Live: `live_pool_ev` reads the WIN5 `__pool__` rows of the odds log (docs/specs/odds-log.md).
- The study (docs/specs/win5-carryover-study.md, spec config/win5_study.yaml): `score_study` scores
  the frozen spec (D18) on development days only (D14): `day_value`, the pool-level return on
  `net_pool` (the pool net of refunds, owner answer B), by bin of the carryover over the
  break-even carryover, with a day bootstrap lower bound per bin.
  The `horsetrack japan win5-study` commands live in `win5_study.py`.

Scraped HTML is untrusted data: it is parsed, never followed.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from horsetrack.core.config import load_source
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry, _spec_hash
from horsetrack.core.paths import processed_dir
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.takeout import Boost, BoostCalendar, combine_boosts, shipped_boosts

TAKEOUT = 0.30
PAYOUT_CAP_YEN = 600_000_000  # per 100-yen ticket; the excess carries over, even on a hit
POOL = "WIN5"
WIN5_START = date(2011, 4, 24)
SOURCE = "jra"
COUNTRY = "japan"
HISTORY_NAME = "win5.parquet"
FIGURES = (
    "race_date",
    "pool_yen",
    "carryover_in_yen",
    "carryover_out_yen",
    "winning_tickets",
    "dividend_yen",
    "payout_yen",
    "winning_combinations",
)
COLUMNS = (*FIGURES, "hit", "pool_ev")
COVERAGE = "pool-level"
NET_POOL = "net-of-refunds"


class NotPreRegistered(RuntimeError):
    """The study's spec is not the one frozen for its family in the registry."""


# ---------------------------------------------------------------- pool arithmetic


def _number(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} is missing")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a number: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} is not finite: {value!r}")
    return out


def _pool(value: Any, name: str = "pool") -> float:
    out = _number(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _non_negative(value: Any, name: str) -> float:
    out = _number(value, name)
    if out < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return out


def _takeout(value: Any) -> float:
    out = _number(value, "takeout")
    if not 0 <= out < 1:
        raise ValueError(f"takeout must be in [0, 1), got {value!r}")
    return out


def pool_ev(pool_yen: Any, carryover_yen: Any, *, takeout: float = TAKEOUT) -> float:
    """The gross return per 1 yen of the whole pool: (pool * (1 - takeout) + carryover) / pool."""
    pool = _pool(pool_yen)
    carry = _non_negative(carryover_yen, "carryover")
    t = _takeout(takeout)
    return (pool * (1 - t) + carry) / pool


def chased_pool_ev(
    base_pool_yen: Any, carryover_yen: Any, *, chase: float, takeout: float = TAKEOUT
) -> float:
    """The pool EV on the pool the carryover drew, P0 + chase * C: (1 - t) + C / (P0 + k * C)."""
    base = _pool(base_pool_yen, "base pool")
    carry = _non_negative(carryover_yen, "carryover")
    k = _non_negative(chase, "chase")
    return pool_ev(base + k * carry, carry, takeout=takeout)


def break_even_carryover(
    base_pool_yen: Any, *, takeout: float = TAKEOUT, chase: float = 0.0
) -> float:
    """The carryover at which the chased pool EV is 1: t * P0 / (1 - t * k); inf when t * k >= 1."""
    base = _pool(base_pool_yen, "base pool")
    t = _takeout(takeout)
    k = _non_negative(chase, "chase")
    if t * k >= 1 - 1e-12:  # 0.3 * (10 / 3) may round just below 1
        return math.inf
    return t * base / (1 - t * k)


def _boosts_win5(entry: Boost, day: date) -> bool:
    """Whether a calendar entry changes WIN5 on `day`: the WIN5 pool spans races and venues, so an
    entry counts when its pools name WIN5, or when it names no pools and covers the whole day (no
    venue, race or age restriction)."""
    if not entry.date <= day <= (entry.until or entry.date):
        return False
    if entry.pools is not None:
        return POOL in entry.pools
    return entry.venue is None and entry.race_no is None and entry.age is None


def win5_takeout(day: date, *, calendar: BoostCalendar | None = None) -> float | None:
    """WIN5's takeout on `day` (1 - its payout rate): jra.yaml's WIN5 rate with the boost calendar
    applied as `takeout.race_takeout` combines entries; None before 2014-06-07, where no single
    rate is known. `calendar` None means the shipped takeout_boosts.yaml."""
    base = load_source(COUNTRY, SOURCE).pools[POOL].takeout_on(day)
    if base is None:
        return None
    cal = calendar if calendar is not None else shipped_boosts()
    rate, _ = combine_boosts(1 - base, {e.kind for e in cal.entries if _boosts_win5(e, day)})
    return round(1 - rate, 6)


def capped_payout(
    pool_yen: Any,
    carryover_yen: Any,
    winning_tickets: Any,
    *,
    takeout: float = TAKEOUT,
    cap_yen: float = PAYOUT_CAP_YEN,
) -> tuple[float, float]:
    """(payout, carryover out) of a day: net = pool * (1 - takeout) + carryover; no ticket: (0,
    net); else the winners share net up to `cap_yen` per 100-yen ticket and the excess carries
    over, even on a hit. Each dividend's rounding down to 10 yen is not modelled."""
    pool = _pool(pool_yen)
    carry = _non_negative(carryover_yen, "carryover")
    tickets = _non_negative(winning_tickets, "winning tickets")
    cap = _pool(cap_yen, "cap")
    net = pool * (1 - _takeout(takeout)) + carry
    if tickets == 0:
        return 0.0, net
    payout = min(net, tickets * cap)
    return payout, net - payout


def payout_mismatches(table: pd.DataFrame, *, calendar: BoostCalendar | None = None) -> list[date]:
    """The days of a history table whose money does not add up, sorted: |payout + carryover out -
    (pool * (1 - win5_takeout(day)) + carryover in)| above 10 yen per winning ticket (10 yen on a
    no-hit day), the rounding of each dividend down to 10 yen per 100-yen ticket. Days with an
    unknown takeout (before 2014-06-07) are skipped."""
    step = load_source(COUNTRY, SOURCE).pools[POOL].dividend_step
    bad = []
    for row in table.to_dict("records"):
        day = _as_date(row["race_date"])
        takeout = win5_takeout(day, calendar=calendar)
        if takeout is None:
            continue
        net = float(row["pool_yen"]) * (1 - takeout) + float(row["carryover_in_yen"])
        paid = float(row["payout_yen"]) + float(row["carryover_out_yen"])
        if abs(paid - net) > step * max(int(row["winning_tickets"]), 1):
            bad.append(day)
    return sorted(bad)


# ---------------------------------------------------------------- parsing

_GROUP = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_YEN = re.compile(rf"(?:(?P<oku>{_GROUP})億)?(?:(?P<man>{_GROUP})万)?(?P<one>{_GROUP})?円")
_DATE = re.compile(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日")
_TICKETS = re.compile(rf"({_GROUP})票")
NO_WINNER = "該当なし"
_HEADERS = {
    "開催日": "race_date",
    "発売金額": "pool_yen",
    "前回からのキャリーオーバー": "carryover_in_yen",
    "的中票数": "winning_tickets",
    "払戻金": "dividend_yen",
    "次回へのキャリーオーバー": "carryover_out_yen",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def parse_yen(text: str) -> int:
    """A printed yen amount: `312,450,000円` or `2億6,716万3,960円`."""
    m = _YEN.fullmatch(_clean(text)) if isinstance(text, str) else None
    if m is None or not any(m.group(g) for g in ("oku", "man", "one")):
        raise ValueError(f"not a yen amount: {text!r}")
    part = {g: int((m.group(g) or "0").replace(",", "")) for g in ("oku", "man", "one")}
    return part["oku"] * 100_000_000 + part["man"] * 10_000 + part["one"]


def _tickets(text: str) -> int:
    m = _TICKETS.fullmatch(_clean(text))
    if m is None:
        raise ValueError(f"not a ticket count: {text!r}")
    return int(m.group(1).replace(",", ""))


def _race_date(text: str) -> date:
    m = _DATE.fullmatch(_clean(text))
    if m is None:
        raise ValueError(f"not a race date: {text!r}")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _dividend(text: str) -> int | None:
    cleaned = _clean(text)
    if cleaned in (NO_WINNER, "-円", "-"):
        return None
    return parse_yen(cleaned)


def _row(cells: dict[str, str]) -> dict[str, Any]:
    return {
        "race_date": _race_date(cells["race_date"]),
        "pool_yen": parse_yen(cells["pool_yen"]),
        "carryover_in_yen": parse_yen(cells["carryover_in_yen"]),
        "carryover_out_yen": parse_yen(cells["carryover_out_yen"]),
        "winning_tickets": _tickets(cells["winning_tickets"]),
        "dividend_yen": _dividend(cells["dividend_yen"]),
    }


def parse_results_page(body: bytes) -> list[dict[str, Any]]:
    """One row per day of a results table (`table.win5_results`), its columns found by header."""
    doc = BeautifulSoup(body, "html.parser")
    table = doc.select_one("table.win5_results")
    if table is None:
        raise ValueError("no WIN5 results table on the page")
    head = next((tr for tr in table.find_all("tr") if tr.find("th") is not None), None)
    names = [_clean(th.get_text()) for th in head.find_all("th")] if head is not None else []
    where = {_HEADERS[n]: i for i, n in enumerate(names) if n in _HEADERS}
    missing = sorted(n for n, key in _HEADERS.items() if key not in where)
    if missing:
        raise ValueError(f"WIN5 results table without the columns {missing}")
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        if len(cells) != len(names):
            raise ValueError(f"a WIN5 results row with {len(cells)} cells, not {len(names)}")
        rows.append(_row({key: cells[i].get_text() for key, i in where.items()}))
    return rows


def parse_day_page(body: bytes, race_date: date) -> dict[str, Any]:
    """netkeiba's WIN5 page of one day (`/top/win5.html?date=YYYYMMDD`) as one history row. The
    figures are labelled cells (th then td); a carryover in or out the page does not print is 0.
    A dead-heat day lists several winning combinations, each with its 払戻金 and 的中票数, paired in
    page order: `winning_tickets` is their total, `payout_yen` the sum of dividend x tickets,
    `winning_combinations` how many have tickets and `dividend_yen` the largest dividend (None when
    nobody hit). ValueError for a page without the day's result (sales, winning tickets and
    dividend) or whose dividends and ticket counts do not pair up."""
    doc = BeautifulSoup(body, "html.parser")
    found: dict[str, list[str]] = {}
    for th in doc.select("table.Win5_Table th"):
        td = th.find_next_sibling("td")
        if td is not None:
            found.setdefault(_clean(th.get_text()), []).append(td.get_text())
    need = ("発売金額", "的中票数", "払戻金")
    if any(label not in found for label in need):
        raise ValueError(f"WIN5 page of {race_date}: no result (labels {sorted(found)})")
    dividends, counts = found["払戻金"], found["的中票数"]
    if len(dividends) != len(counts):
        raise ValueError(
            f"WIN5 page of {race_date}: {len(dividends)} dividends for {len(counts)} ticket counts"
        )
    hits = []
    for paid, count in zip(dividends, counts, strict=True):
        tickets, dividend = _tickets(count), _dividend(paid)
        if tickets == 0:
            continue
        if dividend is None:
            raise ValueError(f"WIN5 page of {race_date}: {tickets} winning tickets, no dividend")
        hits.append((dividend, tickets))
    carry_in = found.get("先週までのキャリーオーバー")
    carry_out = found.get("次週へのキャリーオーバー")
    return {
        "race_date": race_date,
        "pool_yen": parse_yen(found["発売金額"][0]),
        "carryover_in_yen": 0 if carry_in is None else parse_yen(carry_in[0]),
        "carryover_out_yen": 0 if carry_out is None else parse_yen(carry_out[0]),
        "winning_tickets": sum(n for _, n in hits),
        "dividend_yen": max(d for d, _ in hits) if hits else None,
        "payout_yen": sum(d * n for d, n in hits),
        "winning_combinations": len(hits),
    }


def parse_year_list(body: bytes) -> list[date]:
    """The WIN5 days a netkeiba year list (`/top/win5_results.html?year=YYYY`) links, sorted."""
    doc = BeautifulSoup(body, "html.parser")
    days = set()
    for a in doc.find_all("a", href=True):
        m = re.search(r"win5\.html\?date=([0-9]{8})(?:$|&)", a["href"])
        if m:
            days.add(datetime.strptime(m.group(1), "%Y%m%d").date())
    return sorted(days)


# ---------------------------------------------------------------- the history table


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):  # pandas Timestamp included
        return value.date()
    if isinstance(value, date):
        return value
    raise ValueError(f"not a date: {value!r}")


def build_history(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """One row per race_date, ascending; identical duplicates collapse, conflicting ones raise. A
    row without `payout_yen` and `winning_combinations` (a results-page row) is one combination:
    dividend x tickets, and 1 combination with a hit (0 and 0 without)."""
    by_day: dict[date, dict[str, Any]] = {}
    for row in rows:
        row = dict(row)
        hit = row["winning_tickets"] > 0
        if "payout_yen" not in row:  # not evaluated for a row that has it (a dividend may be None)
            row["payout_yen"] = row["dividend_yen"] * row["winning_tickets"] if hit else 0
        row.setdefault("winning_combinations", 1 if hit else 0)
        clean = {key: row[key] for key in FIGURES}
        clean["race_date"] = _as_date(clean["race_date"])
        seen = by_day.get(clean["race_date"])
        if seen is not None and seen != clean:
            raise ValueError(f"WIN5 day {clean['race_date'].isoformat()} given with other figures")
        by_day[clean["race_date"]] = clean
    days = [by_day[d] for d in sorted(by_day)]
    table = pd.DataFrame(days, columns=list(FIGURES))
    for column in ("pool_yen", "carryover_in_yen", "carryover_out_yen", "winning_tickets",
                   "payout_yen", "winning_combinations"):  # fmt: skip
        table[column] = table[column].astype("int64")
    table["dividend_yen"] = pd.array(
        [d["dividend_yen"] for d in days], dtype="Int64"
    )  # None when nobody hit
    table["hit"] = (table["winning_tickets"] > 0).astype(bool)
    table["pool_ev"] = [pool_ev(d["pool_yen"], d["carryover_in_yen"]) for d in days]
    table["pool_ev"] = table["pool_ev"].astype("float64")
    return table[list(COLUMNS)]


def _table_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of a history table as plain figures (Python ints, None for no dividend)."""
    rows = []
    for record in table.to_dict("records"):
        row: dict[str, Any] = {"race_date": _as_date(record["race_date"])}
        for key in FIGURES:
            if key != "race_date":
                value = record[key]
                row[key] = None if value is None or pd.isna(value) else int(value)
        rows.append(row)
    return rows


def append_history(
    existing: pd.DataFrame | None, rows: Iterable[dict[str, Any]]
) -> tuple[pd.DataFrame, list[date]]:
    """Grow a history table (#325): `build_history` over the rows of `existing` and the new ones,
    so every old day stays whether or not `rows` has it, and a day given with other figures than
    the table holds raises ValueError naming it. Returns the table and the days that were not in
    `existing`, ascending. `existing` is not changed."""
    old = _table_rows(existing) if existing is not None else []
    have = {r["race_date"] for r in old}
    new = list(rows)
    table = build_history(old + new)
    added = sorted({_as_date(r["race_date"]) for r in new} - have)
    return table, added


def _dates(table: pd.DataFrame) -> list[date]:
    return [_as_date(v) for v in table["race_date"]]


def history_counts(table: pd.DataFrame) -> dict[str, Any]:
    """Aggregates only: days, carryover days, no-winner days, first and last day."""
    dates = _dates(table)
    return {
        "days": len(table),
        "carryover_days": int((table["carryover_in_yen"] > 0).sum()),
        "no_winner_days": int((table["winning_tickets"] == 0).sum()),
        "first_day": min(dates) if dates else None,
        "last_day": max(dates) if dates else None,
    }


def carryover_chain_breaks(table: pd.DataFrame, unread: Iterable[date] = ()) -> list[date]:
    """The days that break the carryover chain, sorted, each once: a no-winner day with no
    carryover out, or a day whose carryover in is not the previous row's carryover out (the first
    row is not checked on that rule). A winner day with a carryover out (the payout cap) is not a
    break. Catches carryover labels the day-page parser no longer finds (read as 0). `unread` are
    WIN5 days the table lacks on purpose (locked holdout days): the day after one is not checked
    against the row before it, the link between them having never been read."""
    breaks: set[date] = set()
    previous_out: int | None = None
    rows = list(
        zip(
            _dates(table),
            table["winning_tickets"],
            table["carryover_in_yen"],
            table["carryover_out_yen"],
            strict=True,
        )
    )
    rows += [(_as_date(d), None, None, None) for d in unread]
    for day, tickets, carry_in, carry_out in sorted(rows, key=lambda r: r[0]):
        if tickets is None:
            previous_out = None
            continue
        if tickets == 0 and not carry_out > 0:
            breaks.add(day)
        if previous_out is not None and carry_in != previous_out:
            breaks.add(day)
        previous_out = carry_out
    return sorted(breaks)


def missing_days(table: pd.DataFrame, expected_days: Iterable[date]) -> list[date]:
    """The expected days without a row, sorted."""
    have = set(_dates(table))
    return sorted({_as_date(d) for d in expected_days} - have)


def history_path() -> Path:
    return processed_dir(SOURCE) / HISTORY_NAME


def write_history(table: pd.DataFrame) -> Path:
    """Write the table to `processed_dir('jra') / 'win5.parquet'` (atomically) and return it."""
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_history() -> pd.DataFrame:
    table = pd.read_parquet(history_path())
    table["race_date"] = [_as_date(v) for v in table["race_date"]]
    return table


# ---------------------------------------------------------------- live


def live_pool_ev(
    rows: Iterable[dict[str, Any]], race_date: date, *, now: datetime
) -> dict[str, Any] | None:
    """The day's WIN5 pool total and carryover as of `now`, from the odds log's WIN5 `__pool__`
    rows (race_id `jra:<date>:WIN5`) available at or before now; None when there is none. A row is
    available at its fetched_utc, else its snapshot_utc (as the board reads rows). Without a
    carryover figure the pool EV takes the carryover as 0 and `carryover` stays None."""
    race_id = f"{SOURCE}:{race_date.isoformat()}:WIN5"
    now_utc = now.astimezone(timezone.utc)
    seen = []
    for n, row in enumerate(rows):
        if row.get("race_id") != race_id or row.get("combination") != odds_log.POOL_TOTAL:
            continue
        snap = odds_log._instant(row.get("snapshot_utc"))
        fetched = odds_log._instant(row.get("fetched_utc"))
        available = fetched if fetched is not None else snap
        if snap is None or available is None or available > now_utc:
            continue
        seen.append((available, snap, n, row))
    if not seen:
        return None
    seen.sort(key=lambda e: e[:3])

    def latest(field: str) -> float | None:
        values = [row.get(field) for *_, row in seen if row.get(field) is not None]
        return float(values[-1]) if values else None

    investment, carryover = latest("investment"), latest("carryover")
    ev = None
    if investment is not None:
        try:
            ev = pool_ev(investment, 0.0 if carryover is None else carryover)
        except ValueError:  # a zero or broken figure: no EV rather than no feed
            ev = None
    return {
        "race_date": race_date.isoformat(),
        "capture_utc": seen[-1][1].isoformat(),
        "investment": investment,
        "carryover": carryover,
        "pool_ev": ev,
    }


# ---------------------------------------------------------------- the study


def _int(spec: dict[str, Any], key: str) -> int:
    value = spec.get(key)
    if type(value) is not int:
        raise ValueError(f"the spec's {key} must be an integer, got {value!r}")
    return value


def _study_params(spec: dict[str, Any]) -> dict[str, Any]:
    """The figures the scorer reads from a study spec; ValueError for a spec it cannot score."""
    if not isinstance(spec, dict):
        raise ValueError(f"a study spec is a mapping, got {type(spec).__name__}")
    if spec.get("coverage") != COVERAGE:
        coverage = spec.get("coverage")
        raise ValueError(f"the scorer knows coverage {COVERAGE!r} only, got {coverage!r}")
    if spec.get("pool") != NET_POOL:
        raise ValueError(f"the scorer knows pool {NET_POOL!r} only, got {spec.get('pool')!r}")
    edges = spec.get("ratio_edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("the spec names no ratio_edges")
    values = [_non_negative(e, "ratio edge") for e in edges]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"ratio_edges must be strictly increasing, got {edges!r}")
    level = _number(spec.get("one_sided_level"), "one_sided_level")
    if not 0 < level < 1:
        raise ValueError(f"one_sided_level must be in (0, 1), got {level!r}")
    n_boot = _int(spec, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot!r}")
    break_even = _number(spec.get("break_even_takeout"), "break_even_takeout")
    if not 0 < break_even < 1:
        raise ValueError(f"break_even_takeout must be in (0, 1), got {break_even!r}")
    _pool(spec.get("payout_cap_yen"), "payout_cap_yen")
    seed = _int(spec, "seed")
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed!r}")
    return {
        "edges": list(edges),
        "level": level,
        "n_boot": n_boot,
        "seed": seed,
        "break_even": break_even,
        "comparator": _number(spec.get("comparator"), "comparator"),
    }


def check_study_spec(spec: dict[str, Any]) -> None:
    """ValueError unless `score_study` can score `spec` (checked before it is frozen)."""
    _study_params(spec)


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["race_date"] = _as_date(out["race_date"])
    div = out.get("dividend_yen")
    out["dividend_yen"] = None if div is None or pd.isna(div) else int(div)
    return out


def net_pool(row: dict[str, Any], *, calendar: BoostCalendar | None = None) -> float:
    """The WIN5 pool of a day net of refunds (owner, #135, 2026-11-25, answer B). netkeiba's
    発売金額 (pool_yen) is gross sales, and on some days the payout falls short of what it implies
    (hypothesis, unverified: refunds of combinations with a scratched horse). From 2014-06-07 the
    pool is the one the money identity implies, (payout_yen + carryover_out_yen -
    carryover_in_yen) / (1 - win5_takeout(day)), a boosted day at its own rate; pool_yen does not
    enter it. Before 2014-06-07 (takeout unknown) it is pool_yen, the gross sales. ValueError when
    the implied pool is not positive."""
    day = _as_date(row["race_date"])
    takeout = win5_takeout(day, calendar=calendar)
    if takeout is None:
        return _pool(row["pool_yen"])
    paid = (
        _non_negative(row["payout_yen"], "payout")
        + _non_negative(row["carryover_out_yen"], "carryover out")
        - _non_negative(row["carryover_in_yen"], "carryover in")
    )
    pool = paid / (1 - takeout)
    if not pool > 0:
        raise ValueError(f"WIN5 day {day.isoformat()}: the money implies a pool of {pool!r} yen")
    return pool


def day_value(row: dict[str, Any]) -> float:
    """The pool-level return of a WIN5 day: the yen paid over every winning combination per yen of
    the net pool (payout_yen / net_pool(row)), 0 on a no-hit day. The realised payout is already
    held to the cap; the carryover a capped day passes on counts on the next day."""
    if int(row["winning_tickets"]) == 0:
        return 0.0
    return float(row["payout_yen"]) / net_pool(row)


def _bin_of(ratio: float, edges: list[float]) -> int | None:
    """(e0, e1) open at its lower edge, then [e_i, e_i+1), then [en, inf); None outside."""
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        if (ratio > lo if i == 0 else ratio >= lo) and (hi is None or ratio < hi):
            return i
    return None


def _development_bins(
    history: pd.DataFrame, p: dict[str, Any]
) -> tuple[list[list[dict[str, Any]]], int, int]:
    """The development days of `history` by carryover bin (`p` from `_study_params`): the rows per
    bin, the holdout days skipped, and the development days with a carryover in and no known
    takeout. A day's ratio is carryover_in / (break_even_takeout x net_pool(row)). A holdout day is
    counted and skipped before any of its outcome figures is read."""
    edges = p["edges"]
    binned: list[list[dict[str, Any]]] = [[] for _ in edges]
    excluded = gross_days = 0
    for record in history.to_dict("records"):
        if is_holdout_day(COUNTRY, _as_date(record["race_date"])):
            excluded += 1
            continue
        row = _plain(record)
        carry_in = float(row["carryover_in_yen"])
        if carry_in > 0 and win5_takeout(row["race_date"]) is None:
            gross_days += 1
        ratio = carry_in / (p["break_even"] * net_pool(row))
        where = _bin_of(ratio, edges)
        if where is not None:
            binned[where].append(row)
    return binned, excluded, gross_days


def bin_counts(history: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """The development days per carryover bin of the study spec, binned as `score_study` bins them
    (#325): {"bins": [{"lo_ratio", "hi_ratio", "n_days"}, ...], "holdout_days_excluded": n}.
    ValueError for a spec the scorer cannot score. No outcome of a holdout day is read."""
    p = _study_params(spec)
    binned, excluded, _ = _development_bins(history, p)
    edges = p["edges"]
    bins = [
        {
            "lo_ratio": lo,
            "hi_ratio": edges[i + 1] if i + 1 < len(edges) else None,
            "n_days": len(binned[i]),
        }
        for i, lo in enumerate(edges)
    ]
    return {"bins": bins, "holdout_days_excluded": excluded}


def _lower_bound(values: list[float], *, n_boot: int, level: float, seed: list[int]) -> float:
    """The (1 - level) quantile of `n_boot` bootstrap means, the days resampled with
    replacement (the WIN5 day is the unit), drawn in blocks to bound memory."""
    data = np.asarray(values, dtype="float64")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    block = max(1, 1_000_000 // len(data))
    for start in range(0, n_boot, block):
        stop = min(n_boot, start + block)
        means[start:stop] = data[rng.integers(0, len(data), size=(stop - start, len(data)))].mean(1)
    return float(np.quantile(means, 1 - level))


def score_study(
    history: pd.DataFrame,
    spec: dict[str, Any],
    *,
    registry: Registry,
    family_id: str,
) -> dict[str, Any]:
    """Score the frozen study on development days. NotPreRegistered unless `spec` is the spec
    frozen for `family_id`; ValueError for a spec the scorer cannot score. Holdout days are dropped
    first and counted. A day's ratio is carryover_in / (break_even_takeout x net_pool(row)), the
    spec's literal rate on every day; `gross_pool_carryover_days` counts the development days with
    a carryover in whose takeout is unknown (before 2014-06-07), scored at gross sales. Each bin
    reports n_days, the mean `day_value`, the lower bound (the
    (1 - one_sided_level) quantile of a day bootstrap of the mean, seeded by the spec's seed and
    the bin's index) and positive_edge, lower bound above the comparator."""
    spec_hash = _spec_hash(spec)
    if registry.frozen_spec_hash(family_id) != spec_hash:
        raise NotPreRegistered(f"family {family_id!r} has no frozen spec with hash {spec_hash}")
    p = _study_params(spec)
    edges = p["edges"]
    binned, excluded, gross_days = _development_bins(history, p)
    values = [[day_value(row) for row in rows] for rows in binned]
    bins = []
    for i, lo in enumerate(edges):
        got = values[i]
        mean = lower = None
        if got:
            mean = sum(got) / len(got)
            lower = _lower_bound(got, n_boot=p["n_boot"], level=p["level"], seed=[p["seed"], i])
        bins.append(
            {
                "lo_ratio": lo,
                "hi_ratio": edges[i + 1] if i + 1 < len(edges) else None,
                "n_days": len(got),
                "mean_value": mean,
                "lower_bound": lower,
                "positive_edge": lower is not None and lower > p["comparator"],
            }
        )
    return {
        "family_id": family_id,
        "spec_hash": spec_hash,
        "holdout_days_excluded": excluded,
        "gross_pool_carryover_days": gross_days,
        "bins": bins,
    }
