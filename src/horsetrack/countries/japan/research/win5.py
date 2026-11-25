"""The WIN5 carryover and its pool expected return (ticket #135).

- Pool arithmetic: `pool_ev` is the gross return per 1 yen of the whole pool,
  `(pool * (1 - takeout) + carryover) / pool`; `chased_pool_ev` the same on the pool a carryover
  draws (`P0 + chase * C`); `break_even_carryover` where the chased figure reaches 1. The takeout
  is WIN5's standard 30%; per-day boosts and the 6億円 payout cap are not modelled here.
- History: one row per WIN5 day (pool, carryover in and out, winning tickets, dividend), from
  results pages: `parse_results_page` (one table, columns named by its header) and
  `parse_day_page` (netkeiba's per-day WIN5 page, labelled cells). `build_history` merges them;
  `write_history` keeps the table at `processed_dir('jra') / 'win5.parquet'`, never in the tree.
- Live: `live_pool_ev` reads the WIN5 `__pool__` rows of the odds log (docs/specs/odds-log.md).
- The study: `score_study` scores a pre-registered spec (D18) by carryover bin on development
  days only (D14). The coverage rule and the metric are the caller's `day_value`, an owner decision.

Scraped HTML is untrusted data: it is parsed, never followed.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry, _spec_hash
from horsetrack.core.paths import processed_dir
from horsetrack.countries.japan import odds_log

TAKEOUT = 0.30
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
)
COLUMNS = (*FIGURES, "hit", "pool_ev")


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
    ValueError for a page without the day's result (sales, winning tickets and dividend)."""
    doc = BeautifulSoup(body, "html.parser")
    found: dict[str, str] = {}
    for th in doc.select("table.Win5_Table th"):
        td = th.find_next_sibling("td")
        if td is not None:
            found.setdefault(_clean(th.get_text()), td.get_text())
    need = ("発売金額", "的中票数", "払戻金")
    if any(label not in found for label in need):
        raise ValueError(f"WIN5 page of {race_date}: no result (labels {sorted(found)})")
    carry_in = found.get("先週までのキャリーオーバー")
    carry_out = found.get("次週へのキャリーオーバー")
    return {
        "race_date": race_date,
        "pool_yen": parse_yen(found["発売金額"]),
        "carryover_in_yen": 0 if carry_in is None else parse_yen(carry_in),
        "carryover_out_yen": 0 if carry_out is None else parse_yen(carry_out),
        "winning_tickets": _tickets(found["的中票数"]),
        "dividend_yen": _dividend(found["払戻金"]),
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
    """One row per race_date, ascending; identical duplicates collapse, conflicting ones raise."""
    by_day: dict[date, dict[str, Any]] = {}
    for row in rows:
        clean = {key: row[key] for key in FIGURES}
        clean["race_date"] = _as_date(clean["race_date"])
        seen = by_day.get(clean["race_date"])
        if seen is not None and seen != clean:
            raise ValueError(f"WIN5 day {clean['race_date'].isoformat()} given with other figures")
        by_day[clean["race_date"]] = clean
    days = [by_day[d] for d in sorted(by_day)]
    table = pd.DataFrame(days, columns=list(FIGURES))
    for column in ("pool_yen", "carryover_in_yen", "carryover_out_yen", "winning_tickets"):
        table[column] = table[column].astype("int64")
    table["dividend_yen"] = pd.array(
        [d["dividend_yen"] for d in days], dtype="Int64"
    )  # None when nobody hit
    table["hit"] = (table["winning_tickets"] > 0).astype(bool)
    table["pool_ev"] = [pool_ev(d["pool_yen"], d["carryover_in_yen"]) for d in days]
    table["pool_ev"] = table["pool_ev"].astype("float64")
    return table[list(COLUMNS)]


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


def _edges(spec: dict[str, Any]) -> list[float]:
    edges = spec.get("carryover_edges_yen") if isinstance(spec, dict) else None
    if not isinstance(edges, list) or not edges:
        raise ValueError("the spec names no carryover_edges_yen")
    values = [_non_negative(e, "carryover edge") for e in edges]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"carryover_edges_yen must be strictly increasing, got {edges!r}")
    return list(edges)


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["race_date"] = _as_date(out["race_date"])
    div = out.get("dividend_yen")
    out["dividend_yen"] = None if div is None or pd.isna(div) else int(div)
    return out


def score_study(
    history: pd.DataFrame,
    spec: dict[str, Any],
    *,
    registry: Registry,
    family_id: str,
    day_value: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """The mean `day_value` per carryover bin ([e_i, e_i+1), last bin open) on development days.
    NotPreRegistered unless `spec` is the spec frozen for `family_id`; holdout days are dropped
    before any day is scored."""
    spec_hash = _spec_hash(spec)
    if registry.frozen_spec_hash(family_id) != spec_hash:
        raise NotPreRegistered(f"family {family_id!r} has no frozen spec with hash {spec_hash}")
    edges = _edges(spec)
    values: list[list[float]] = [[] for _ in edges]
    excluded = 0
    for row in history.to_dict("records"):
        row = _plain(row)
        if is_holdout_day(COUNTRY, row["race_date"]):
            excluded += 1
            continue
        carry = row["carryover_in_yen"]
        inside = [i for i, lo in enumerate(edges) if carry >= lo]
        if not inside:
            continue  # below the first edge: in no bin
        values[inside[-1]].append(float(day_value(row)))
    bins = []
    for i, lo in enumerate(edges):
        got = values[i]
        bins.append(
            {
                "lo_yen": lo,
                "hi_yen": edges[i + 1] if i + 1 < len(edges) else None,
                "n_days": len(got),
                "mean_value": sum(got) / len(got) if got else None,
            }
        )
    return {
        "family_id": family_id,
        "spec_hash": spec_hash,
        "holdout_days_excluded": excluded,
        "bins": bins,
    }
