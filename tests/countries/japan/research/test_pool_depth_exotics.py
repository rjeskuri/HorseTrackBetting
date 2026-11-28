"""Ticket #490: TIERCE (trifecta) and BRACKET_QUINELLA pool depth, with a streaming quantile
(synthetic data only).

Contract pinned here for `horsetrack.countries.japan.research.pool_depth` (the coder implements
what these tests call), beyond #432's (tests/countries/japan/research/test_pool_depth.py):

- `StreamingQuantile()`: an exact quantile that takes its values a chunk at a time.
  `.add(values, counts=None)` takes a 1-d array of floats (NaN ignored, +inf allowed) and, with
  `counts`, a same-length array of non-negative integer multiplicities (value i counted counts[i]
  times; 0 drops it); `.count` is the number of non-NaN values taken (multiplicities summed);
  `.quantiles(qs)` is an ndarray, one value per q: exactly numpy's default (linear) quantile of
  every value taken so far, as if concatenated (+inf sorts last and gives inf wherever the
  interpolation reaches one, as `_quantiles` does); all NaN when nothing was taken.
- `bracket_map(runners) -> DataFrame` with columns race_id, horse_no and bracket (ints): one row
  per runner of the jravan `runners` rows (`schemas.TABLES['runners']`: its `bracket` is 枠番),
  other columns ignored; ValueError when a race gives one horse two brackets.
- `depth_rows(votes, board, races, *, band_edges, brackets=None)`: with `brackets` (a
  `bracket_map`), each BRACKET_QUINELLA row (combination "a,b", brackets a <= b, as the store
  writes 組番) carries `horses`, the horse numbers of brackets a and b, ascending, joined with ","
  ("7,8,9" for "7,8" when bracket 8 holds 8 and 9). A pair is valid when both brackets hold a
  runner and a same-bracket pair ("8,8") holds two or more: an invalid pair gives no row even when
  the board prices it. Other pools' rows are unchanged.
- The spec `config/pool_depth_exotics.yaml` ("pool_depth_exotics", one of `SPECS`, registering
  the `depth` variant): its own family_id (neither #432's v1 nor the co-placer v2), frozen before
  any run, odds bands for TIERCE and BRACKET_QUINELLA at least, date_from not before 2014-06-07,
  plain JSON; `load_spec("pool_depth_exotics")` loads it.
- `run_depth(*, date_from, date_to, out_dir, registry=None, spec_name="pool_depth")`: with
  spec_name "pool_depth_exotics", the spec's pools (TIERCE and BRACKET_QUINELLA included), the
  bracket map read from `processed/jravan/runners.parquet` (development days only, like every
  other store read), cells per pool x class group x odds band exactly as #432's (`depth_table`
  columns; TIERCE and BRACKET_QUINELLA are single-winner pools, so `capacity_{e:g}` is the
  single-winner capacity stake), streamed a year at a time; NotPreRegistered unless that spec is
  frozen under its family; one RunRecord per run in that family, failures included.
- `horsetrack japan pool-depth freeze|run --spec pool_depth_exotics` freeze and run it.

Every board, vote, runner and race here is invented; HORSETRACK_DATA_DIR points at tmp_path.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import copy
import itertools
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import test_pool_depth as tpd

# isort: split
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.registry import Registry, _spec_hash, default_registry
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.research import pool_depth as depth

# #432's fixtures: HORSETRACK_DATA_DIR at tmp_path (autouse), the stand-in holdout, and the
# committed registry summary at tmp_path for the command
data_dir = tpd.data_dir
stand_in_holdout = tpd.stand_in_holdout
summary_folder = tpd.summary_folder

EXOTICS = "pool_depth_exotics"
NEW_POOLS = ("TIERCE", "BRACKET_QUINELLA")
ALL_POOLS = (*tpd.POOLS, "TIERCE")  # the pools make_race can draw; BRACKET_QUINELLA is drawn here
CO_PLACER = ("PLACE", "QUINELLA_PLACE")
BQ = "BRACKET_QUINELLA"
BQ_RATE, BQ_BASE_VOTES = 0.775, 800  # 1 - takeout (config/jra.yaml), and a typical pair's votes
SPEC_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)  # the quantiles #432's cells use
V1_FAMILY = "japan-pool-depth-v1"
V2_FAMILY = "japan-pool-depth-coplacer-v2"


# --- the streaming quantile ----------------------------------------------------------------------

LEVELS = (*SPEC_QUANTILES, 0.0, 1.0, 1 / 3, 0.999)


def chunked(values: np.ndarray, rng: np.random.Generator, pieces: int) -> list[np.ndarray]:
    """`values` cut at `pieces - 1` random points (empty chunks allowed)."""
    cuts = np.sort(rng.integers(0, len(values) + 1, size=max(0, pieces - 1)))
    return np.split(values, cuts)


def datasets() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(490)
    money = np.round(np.exp(rng.normal(math.log(6_000), 1.3, 5_000)), -2)  # yen, many ties
    return {
        "lognormal money": money,
        "few distinct votes": rng.integers(1, 12, 20_000).astype(float) * 100,
        "uniform floats": rng.uniform(-50.0, 50.0, 3_001),
        "one value": np.array([4_200.0]),
        "two values": np.array([300.0, 100.0]),
        "with NaN": np.concatenate([rng.normal(0, 1, 997), [np.nan] * 13]),
        "sorted descending": np.sort(rng.uniform(0, 1e6, 1_000))[::-1].copy(),
    }


@pytest.mark.parametrize("name", list(datasets()))
@pytest.mark.parametrize("pieces", [1, 3, 17])
def test_the_streaming_quantile_equals_numpys_over_chunks(name, pieces):
    """DoD 'the streaming quantile equals numpy's on synthetic data': fed in chunks (empty ones
    included), it gives numpy's linear quantile of every value at once, at the cells' quantiles
    and at the ends; NaN values are not counted."""
    values = datasets()[name]
    rng = np.random.default_rng(len(values) * 31 + pieces)
    stream = depth.StreamingQuantile()
    for chunk in chunked(rng.permutation(values), rng, pieces):
        stream.add(chunk)

    finite = values[~np.isnan(values)]
    want = np.quantile(finite, LEVELS)
    got = np.asarray(stream.quantiles(LEVELS), dtype=float)
    assert got.shape == (len(LEVELS),)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9)
    assert stream.count == len(finite)


def test_the_streaming_quantile_counts_multiplicities():
    """A value with a count stands for that many copies: chunks with and without counts (zero
    counts included) give numpy's quantile of the values repeated; the answer changes with the
    counts, so a quantile of the distinct values alone cannot pass."""
    rng = np.random.default_rng(4901)
    distinct = np.round(np.exp(rng.normal(math.log(2_000), 1.0, 400)), -2)
    counts = rng.integers(0, 60, distinct.size)
    counts[:5] = 0
    plain = rng.uniform(100, 50_000, 700)

    stream = depth.StreamingQuantile()
    half = distinct.size // 2
    stream.add(distinct[:half], counts=counts[:half])
    stream.add(plain)
    stream.add(distinct[half:], counts=counts[half:])

    every = np.concatenate([np.repeat(distinct, counts), plain])
    np.testing.assert_allclose(stream.quantiles(LEVELS), np.quantile(every, LEVELS), rtol=1e-12)
    assert stream.count == int(counts.sum()) + plain.size

    skewed = depth.StreamingQuantile()
    heavy = np.where(distinct == distinct.max(), 10_000, 1)
    skewed.add(distinct, counts=heavy)
    want = np.quantile(np.repeat(distinct, heavy), SPEC_QUANTILES)
    np.testing.assert_allclose(skewed.quantiles(SPEC_QUANTILES), want, rtol=1e-12)
    assert want[0] == distinct.max() != np.quantile(distinct, 0.1)


def linear_with_inf(values: list[float], q: float) -> float:
    """numpy's linear quantile with +inf sorted last, derived here: inf wherever the
    interpolation reaches an inf, the order statistic itself where it lands on one."""
    x = sorted(values)
    h = (len(x) - 1) * q
    lo, hi = math.floor(h), math.ceil(h)
    if lo == hi:
        return x[lo]
    if math.isinf(x[hi]):
        return math.inf
    return x[lo] + (x[hi] - x[lo]) * (h - lo)


def test_the_streaming_quantile_sorts_an_unbounded_capacity_last():
    """An unbounded capacity (+inf) sorts last, as #432's cells take it: the quantiles below the
    inf tail are finite and exact, those reaching it are inf."""
    rng = np.random.default_rng(4902)
    finite = rng.uniform(100, 9_000, 37)
    values = np.concatenate([finite, [np.inf] * 5])
    stream = depth.StreamingQuantile()
    for chunk in chunked(rng.permutation(values), rng, 4):
        stream.add(chunk)

    got = np.asarray(stream.quantiles(LEVELS), dtype=float)
    for q, g in zip(LEVELS, got, strict=True):
        want = linear_with_inf(list(values), q)
        if math.isinf(want):
            assert math.isinf(g) and g > 0, (q, g)
        else:
            assert g == pytest.approx(want, rel=1e-12), q
    assert np.isfinite(got[LEVELS.index(0.5)]) and np.isinf(got[LEVELS.index(1.0)])


def test_an_empty_streaming_quantile_is_nan():
    stream = depth.StreamingQuantile()
    stream.add(np.array([], dtype=float))
    stream.add(np.array([np.nan, np.nan]))
    assert stream.count == 0
    assert np.isnan(np.asarray(stream.quantiles(SPEC_QUANTILES), dtype=float)).all()


# --- synthetic races with a bracket quinella and their runners -----------------------------------


def jra_brackets(n: int) -> dict[int, int]:
    """horse -> bracket by JRA's draw for n <= 16: one horse a bracket up to 8, then the highest
    brackets take two each (n = 9: bracket 8 holds 8 and 9)."""
    if n <= 8:
        return {h: h for h in range(1, n + 1)}
    assert n <= 16
    singles = 16 - n
    out = {h: h for h in range(1, singles + 1)}
    horse = singles + 1
    for bracket in range(singles + 1, 9):
        out[horse], out[horse + 1] = bracket, bracket
        horse += 2
    return out


def members(brackets: dict[int, int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for horse, b in sorted(brackets.items()):
        out.setdefault(b, []).append(horse)
    return out


def valid_pairs(brackets: dict[int, int]) -> list[str]:
    """The bracket pairs sold: two brackets with runners, or one bracket holding two or more."""
    held = members(brackets)
    pairs = []
    for a, b in itertools.combinations_with_replacement(sorted(held), 2):
        if a < b or len(held[a]) >= 2:
            pairs.append(f"{a},{b}")
    return pairs


def pair_horses(brackets: dict[int, int], combination: str) -> str:
    a, b = (int(x) for x in combination.split(","))
    held = members(brackets)
    return ",".join(str(h) for h in sorted({*held[a], *held[b]}))


@dataclass
class Exotic:
    race: tpd.Race
    brackets: dict[int, int]  # horse -> bracket (枠番)


def exotic_race(
    seed: int,
    day: date,
    venue: str,
    race_no: int,
    race_class: str,
    n: int,
    *,
    scale: int = 1,
    pools: tuple[str, ...] = ALL_POOLS,
    brackets: dict[int, int] | None = None,
    phantom: str | None = None,
) -> Exotic:
    """make_race's pools plus a bracket quinella over the valid pairs of `brackets` (JRA's draw
    by default), its money drawn from `seed`; `phantom` adds a priced pair that is not sold."""
    race = tpd.make_race(seed, day, venue, race_no, race_class, n, scale=scale, pools=pools)
    brackets = jra_brackets(n) if brackets is None else brackets
    rng = np.random.default_rng(seed + 49_000)
    combos = valid_pairs(brackets) + ([phantom] if phantom else [])
    raw = np.exp(rng.normal(math.log(BQ_BASE_VOTES), 0.8, len(combos)))
    v = np.maximum(1, np.round(raw)).astype(int)
    total = int(v.sum()) + int(rng.integers(20, 2000))
    assert v.max() / total < 0.8
    race.votes[BQ] = {c: int(x) * scale for c, x in zip(combos, v, strict=True)}
    race.pool_votes[BQ] = total * scale
    race.odds[BQ] = {
        c: max(1.0, math.floor(10 * BQ_RATE * total / int(x)) / 10)
        for c, x in zip(combos, v, strict=True)
    }
    return Exotic(race, brackets)


def runners_schema() -> pa.Schema:
    columns = schemas.TABLES["runners"].columns
    return pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in columns.items()])


def runner_rows(exotic: Exotic) -> list[dict]:
    race = exotic.race
    return [
        {
            "jurisdiction": "jra",
            "race_id": race.race_id,
            "meeting_id": f"jra:{race.day.isoformat()}:{race.venue}",
            "race_date": race.day,
            "venue": race.venue,
            "race_no": race.race_no,
            "horse_id": f"20{race.race_no:02d}{horse:06d}",
            "horse_no": horse,
            "horse_name": f"Synthetic {horse}",
            "bracket": bracket,
            "is_scratched": False,
            "draw": horse,
        }
        for horse, bracket in sorted(exotic.brackets.items())
    ]


def runners_table(exotics: list[Exotic]) -> pa.Table:
    rows = [row for e in exotics for row in runner_rows(e)]
    np.random.default_rng(490).shuffle(rows)  # nothing may rely on the file's row order
    return pa.Table.from_pylist(rows, schema=runners_schema())


def write_exotic_store(data: Path, exotics: list[Exotic]) -> None:
    """#432's store tables and `processed/jravan/runners.parquet`, holdout days included."""
    tpd.write_store(data, [e.race for e in exotics])
    pq.write_table(runners_table(exotics), data / "processed" / "jravan" / "runners.parquet")


# --- the bracket map and the bracket-quinella rows -----------------------------------------------

# A ten-horse field drawn unlike JRA's (brackets 3 and 5 hold two): the map must come from the
# runners, not from the field size.
ODD_DRAW = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 5, 8: 6, 9: 7, 10: 8}


def bracket_exotics() -> list[Exotic]:
    day = date(2019, 6, 2)
    pools = ("WIN", "QUINELLA")
    return [
        exotic_race(61, day, "KYO", 1, "DEBUT", 9, pools=pools, phantom="1,1"),
        exotic_race(62, day, "KYO", 2, "MAIDEN", 12, pools=pools),
        exotic_race(63, day, "KYO", 3, "C2", 10, pools=pools, brackets=ODD_DRAW),
        exotic_race(64, day, "HSN", 4, "G3", 16, pools=pools, scale=4),
    ]


def test_the_bracket_map_comes_from_the_runners():
    """DoD 'bracket-quinella rows map horses to brackets': per race, each runner's 枠番 as the
    runners table holds it, whatever the field size suggests."""
    exotics = bracket_exotics()
    runners = runners_table(exotics).to_pandas()

    out = depth.bracket_map(runners)

    assert {"race_id", "horse_no", "bracket"} <= set(out.columns), list(out.columns)
    got = {(r, int(h), int(b)) for r, h, b in zip(out["race_id"], out["horse_no"], out["bracket"],
                                                   strict=True)}  # fmt: skip
    want = {(e.race.race_id, h, b) for e in exotics for h, b in e.brackets.items()}
    assert got == want
    assert len(out) == len(want)

    clash = runners.copy()
    first = clash.index[0]
    extra = clash.loc[[first]].assign(bracket=int(clash.loc[first, "bracket"]) % 8 + 1)
    with pytest.raises(ValueError):
        depth.bracket_map(pd.concat([clash, extra], ignore_index=True))


BQ_EDGES = {"QUINELLA": (10.0, 50.0), BQ: (5.0, 15.0, 40.0)}


def test_bracket_quinella_rows_name_the_horses_of_each_bracket_pair():
    """DoD 'bracket-quinella rows map horses to brackets': each priced bracket pair is a row with
    its money, pool, class group and band, like any pool, and the horses its two brackets hold
    (same-bracket pairs included); a priced pair that is not sold (1-1 with one horse in bracket
    1) gives no row; the quinella's rows are unchanged."""
    exotics = bracket_exotics()
    races = [e.race for e in exotics]
    frames = {name: t.to_pandas() for name, t in tpd.store_tables(races).items()}
    brackets = depth.bracket_map(runners_table(exotics).to_pandas())

    out = depth.depth_rows(
        frames["pool_votes"], frames["final_odds"], frames["races"], band_edges=BQ_EDGES,
        brackets=brackets,
    )  # fmt: skip

    want = tpd.expected_rows(races, BQ_EDGES)
    phantom = (races[0].race_id, BQ, "1,1")
    keys = list(zip(want["race_id"], want["pool"], want["combination"], strict=True))
    want = want.loc[[k != phantom for k in keys]]
    by_race = {e.race.race_id: e.brackets for e in exotics}
    assert "horses" in out.columns, sorted(out.columns)
    got_keys = list(zip(out["race_id"], out["pool"], out["combination"], strict=True))
    assert len(got_keys) == len(set(got_keys)), "a combination twice"
    got = dict(zip(got_keys, out.to_dict("records"), strict=True))
    want_by = {(r["race_id"], r["pool"], r["combination"]): r for r in want.to_dict("records")}
    assert phantom not in got, "a bracket pair that is not sold gave a row"
    assert set(got) == set(want_by), (
        sorted(set(got) - set(want_by))[:5],
        sorted(set(want_by) - set(got))[:5],
    )
    for key, w in want_by.items():
        g = got[key]
        assert (g["class_group"], g["band"]) == (w["class_group"], w["band"]), (key, g, w)
        assert float(g["combo_yen"]) == pytest.approx(w["combo_yen"]), key
        assert float(g["pool_yen"]) == pytest.approx(w["pool_yen"]), key
        if key[1] == BQ:
            assert g["horses"] == pair_horses(by_race[key[0]], key[2]), (key, g["horses"])

    # spot checks of the map, by hand: JRA's nine-horse draw doubles bracket 8; the odd draw
    # doubles brackets 3 and 5
    nine, odd = races[0].race_id, races[2].race_id
    assert got[(nine, BQ, "7,8")]["horses"] == "7,8,9"
    assert got[(nine, BQ, "8,8")]["horses"] == "8,9"
    assert got[(odd, BQ, "3,5")]["horses"] == "3,4,6,7"
    assert got[(odd, BQ, "3,3")]["horses"] == "3,4"
    assert (odd, BQ, "1,1") not in got and (odd, BQ, "4,4") not in got
    same = [k for k in got if k[1] == BQ and k[2].split(",")[0] == k[2].split(",")[1]]
    assert same, "no same-bracket pair gave a row"


# --- the spec ------------------------------------------------------------------------------------


def exotics_spec() -> dict:
    return copy.deepcopy(dict(load_country_yaml("japan", EXOTICS)))


def exotics_frozen(tmp_path: Path, name: str = "registry") -> tuple[Registry, dict]:
    """A registry under tmp_path with the shipped exotics spec frozen under its family."""
    spec = exotics_spec()
    registry = Registry(tmp_path / f"{name}.jsonl", holdout_log=tmp_path / f"{name}-holdout.jsonl")
    registry.freeze_family(spec["family_id"], spec)
    return registry, spec


def test_the_exotics_spec_registers_tierce_and_the_bracket_quinella(tmp_path):
    """DoD 'the spec is frozen before the run': a spec of its own (a new family, not #432's v1 or
    the co-placer v2) whose cells cover TIERCE and BRACKET_QUINELLA with ascending odds bands,
    from 2014-06-07 at the earliest, registering the depth variant; plain JSON, so it freezes."""
    spec = exotics_spec()
    assert EXOTICS in depth.SPECS and depth.VARIANT_DEPTH in depth.SPECS[EXOTICS]
    family, loaded = depth.load_spec(EXOTICS)
    assert loaded == spec and family == spec["family_id"]
    assert isinstance(family, str) and family.strip()
    assert family not in (V1_FAMILY, V2_FAMILY)
    assert set(NEW_POOLS) <= set(spec["odds_bands"]), sorted(spec["odds_bands"])
    for pool in NEW_POOLS:
        edges = [float(e) for e in spec["odds_bands"][pool]]
        assert edges and edges[0] > 0 and all(a < b for a, b in itertools.pairwise(edges)), pool
    assert date.fromisoformat(spec["date_from"]) >= tpd.PRICE_ERA
    assert [float(e) for e in spec["edges"]] == [0.05, 0.1, 0.2]
    json.dumps(spec, allow_nan=False)
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    assert registry.freeze_family(family, spec) == _spec_hash(spec)


# --- the run -------------------------------------------------------------------------------------

EXOTIC_CLASSES = ("DEBUT", "MAIDEN", "C1", "OP", "G2", "MAIDEN", "DEBUT", "C3", "L")
FIELDS = (9, 12, 10, 16, 11, 9, 13, 10, 14)
# across a year's end: the run reads a year at a time and must merge the years exactly
YEAR_END = (date(2019, 12, 21), date(2019, 12, 28), date(2020, 1, 5), date(2020, 1, 11))
YEAR_END_SPAN = dict(date_from=date(2019, 12, 1), date_to=date(2020, 1, 31))


def exotic_races(days, seeds, *, scale: int = 1, venue: str = "NAK") -> list[Exotic]:
    out = []
    for i, seed in enumerate(seeds):
        out.append(
            exotic_race(
                400 + seed,
                days[i % len(days)],
                venue,
                i // len(days) + 1,
                EXOTIC_CLASSES[seed % len(EXOTIC_CLASSES)],
                FIELDS[seed % len(FIELDS)],
                scale=scale,
                phantom="1,1" if seed == 0 else None,  # bracket 1 holds one horse: not sold
            )
        )
    return out


def expected_cells(exotics: list[Exotic], spec: dict) -> dict[tuple, dict[str, float]]:
    """(pool, class_group, band, quantile) -> what the run's table must give, computed in memory
    from the races: every priced, sold combination of the spec's pools; the single-winner
    capacity for every pool but Place and Wide (TIERCE and BRACKET_QUINELLA included)."""
    rows = tpd.expected_rows([e.race for e in exotics], spec["odds_bands"])
    sold = {(e.race.race_id, c) for e in exotics for c in valid_pairs(e.brackets)}
    keep = [
        p != BQ or (r, c) in sold
        for r, p, c in zip(rows["race_id"], rows["pool"], rows["combination"], strict=True)
    ]
    rows = rows.loc[keep]
    edges, quantiles = spec["edges"], spec["quantiles"]
    out = {}
    for (pool, group, band), cell in rows.groupby(["pool", "class_group", "band"]):
        combo = cell["combo_yen"].to_numpy(dtype=float)
        pools = cell.drop_duplicates("race_id")["pool_yen"].to_numpy(dtype=float)
        pairs = list(zip(cell["pool_yen"], cell["combo_yen"], strict=True))
        for q in quantiles:
            values = {
                "n_combinations": len(cell),
                "n_races": int(cell["race_id"].nunique()),
                "combo_yen": float(np.quantile(combo, float(q))),
                "pool_yen": float(np.quantile(pools, float(q))),
            }
            if pool not in CO_PLACER:
                for e in edges:
                    caps = np.array([tpd.capacity(v, w, float(e)) for v, w in pairs])
                    values[f"capacity_{float(e):g}"] = float(np.quantile(caps, float(q)))
            out[(pool, group, band, round(float(q), 9))] = values
    return out


def test_the_exotics_run_gives_tierce_and_bracket_quinella_cells(tmp_path, data_dir):
    """DoD 'TIERCE and BRACKET_QUINELLA rows per pool x class group x odds band, newcomer and
    maiden included' and 'every run is a registry record': run_depth on the exotics spec, over
    races either side of a year's end, gives every cell an exact in-memory computation gives
    (counts, money, pool and the capacity at each edge), the bracket pair that is not sold left
    out; one scored record in the exotics family; races outside the dates are not read."""
    registry, spec = exotics_frozen(tmp_path)
    inside = exotic_races(YEAR_END, range(9))
    outside = [
        exotic_race(431, date(2019, 11, 30), "NAK", 1, "MAIDEN", 12, scale=50),
        exotic_race(432, date(2020, 2, 1), "NAK", 1, "DEBUT", 10, scale=50),
    ]
    write_exotic_store(data_dir, inside + outside)
    out_dir = data_dir / "reports"

    result = depth.run_depth(**YEAR_END_SPAN, out_dir=out_dir, registry=registry,
                             spec_name=EXOTICS)  # fmt: skip

    edges = spec["edges"]
    tpd.assert_depth_table(result.table, expected_cells(inside, spec), edges)
    for pool in NEW_POOLS:
        part = result.table.loc[result.table["pool"] == pool]
        assert {"newcomer", "maiden"} <= set(part["class_group"]), (pool, set(part["class_group"]))
        caps = part[[f"capacity_{float(e):g}" for e in edges]].to_numpy(dtype=float)
        assert np.isfinite(caps).any(), f"{pool} has no capacity"
    assert result.family_id == spec["family_id"]
    assert result.spec_hash == _spec_hash(spec) == registry.frozen_spec_hash(spec["family_id"])
    report = Path(result.report_path)
    assert report.is_file() and report.resolve().is_relative_to(out_dir.resolve())

    records = registry.records(spec["family_id"])
    assert [(r.variant, r.status) for r in records] == [(depth.VARIANT_DEPTH, "scored")]
    assert tuple(records[0].sources) == tpd.SOURCES
    registry.verify()


def test_the_exotics_run_waits_for_its_freeze_and_records_failures(tmp_path, data_dir):
    """DoD 'the spec is frozen before the run; every run is a registry record; nothing before
    2014-06-07': without the exotics freeze (#432's v1 frozen does not count) the run refuses and
    writes nothing; a start before 2014-06-07 raises; a run that fails after the freeze is a
    failed record in the exotics family."""
    write_exotic_store(data_dir, exotic_races(YEAR_END, range(4)))
    out_dir = data_dir / "reports"
    v1_only, _ = tpd.frozen_registry(tmp_path, "v1-only")
    with pytest.raises(depth.NotPreRegistered):
        depth.run_depth(**YEAR_END_SPAN, out_dir=out_dir, registry=v1_only, spec_name=EXOTICS)
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))

    registry, spec = exotics_frozen(tmp_path)
    with pytest.raises(ValueError):
        depth.run_depth(date_from=date(2014, 6, 6), date_to=date(2020, 1, 31), out_dir=out_dir,
                        registry=registry, spec_name=EXOTICS)  # fmt: skip
    assert not [r for r in registry.records(spec["family_id"]) if r.status == "scored"]

    (data_dir / "processed" / "jravan" / "pool_votes.parquet").unlink()
    with pytest.raises(Exception):  # noqa: B017 - any failure, as long as it is recorded
        depth.run_depth(**YEAR_END_SPAN, out_dir=out_dir, registry=registry, spec_name=EXOTICS)
    assert [r.status for r in registry.records(spec["family_id"])] == ["failed"]
    assert registry.records(V1_FAMILY) == []


def test_the_exotics_run_reads_development_days_only(tmp_path, data_dir, stand_in_holdout,
                                                     monkeypatch):  # fmt: skip
    """DoD 'development days only; nothing before 2014-06-07': holdout races (a thousand times the
    money) and a race before 2014-06-07 sit in every store table, runners included; no row of
    theirs reaches Python and the table is the development races' own."""
    registry, spec = exotics_frozen(tmp_path)
    days = (date(2014, 6, 7), date(2014, 6, 8), *tpd.DEV_DAYS)
    development = [
        exotic_race(500 + i, days[i % len(days)], "KOK", 1 + i // len(days),
                    EXOTIC_CLASSES[i % len(EXOTIC_CLASSES)], FIELDS[i % len(FIELDS)])
        for i in range(8)
    ]  # fmt: skip
    holdout = [
        exotic_race(520 + i, d, "KOK", 1, ("G2", "L", "MAIDEN")[i], 12, scale=1000)
        for i, d in enumerate(tpd.HOLDOUT_DAYS)
    ]
    pre_era = exotic_race(530, date(2014, 6, 1), "KOK", 1, "DEBUT", 10, scale=1000)
    write_exotic_store(data_dir, [*development, *holdout, pre_era])
    spy = tpd.install_row_spy(monkeypatch)

    result = depth.run_depth(**tpd.ELIGIBLE_SPAN, out_dir=data_dir / "reports",
                             registry=registry, spec_name=EXOTICS)  # fmt: skip

    assert spy.rows > 0, "no read went through the watched readers"
    assert not spy.bad, f"rows of ineligible days reached Python: {spy.bad[:5]}"
    tpd.assert_depth_table(result.table, expected_cells(development, spec), spec["edges"])


def test_the_exotics_public_report_carries_no_figures(tmp_path, data_dir):
    """DoD 'no JRA-VAN figure in public': two exotics runs over different data render the same
    public text (identities aside), and no figure of either run's TIERCE or bracket-quinella
    cells is in it."""
    out_dir = data_dir / "reports"
    registry_a, spec = exotics_frozen(tmp_path, "a")
    write_exotic_store(data_dir, exotic_races(YEAR_END, range(9)))
    result_a = depth.run_depth(**YEAR_END_SPAN, out_dir=out_dir, registry=registry_a,
                               spec_name=EXOTICS)  # fmt: skip

    registry_b, _ = exotics_frozen(tmp_path, "b")
    later = (date(2019, 12, 7), date(2019, 12, 14), date(2020, 1, 18), date(2020, 1, 25))
    write_exotic_store(data_dir, exotic_races(later, range(9), scale=3)
                       + exotic_races(later, (1, 4, 7), scale=7, venue="HSN"))  # fmt: skip
    result_b = depth.run_depth(**YEAR_END_SPAN, out_dir=out_dir, registry=registry_b,
                               spec_name=EXOTICS)  # fmt: skip

    text_a = depth.render_public_report(result_a)
    text_b = depth.render_public_report(result_b)
    assert tpd.WITHHELD in text_a
    assert tpd.without_identities(text_a) == tpd.without_identities(text_b), (
        "the public report changes with the data: it carries a figure computed from it"
    )
    columns = ["combo_yen", "pool_yen", *(f"capacity_{float(e):g}" for e in spec["edges"])]
    for result in (result_a, result_b):
        part = result.table.loc[result.table["pool"].isin(NEW_POOLS), columns]
        assert len(part), "no TIERCE or bracket-quinella cell"
        tpd.assert_no_figure_leaks(text_a, part.to_numpy(dtype=float).ravel())


def test_the_exotics_figure_check_holds_against_identities_and_real_figures(tmp_path, data_dir):
    """#749, DoD 'the figure checks cannot fail because a run id or spec hash contains a
    figure's digits, and still fail on a real figure': the exotics report's check passes with a
    run id holding a TIERCE or bracket-quinella figure and fails when that figure is printed."""
    registry, spec = exotics_frozen(tmp_path)
    write_exotic_store(data_dir, exotic_races(YEAR_END, range(9)))
    result = depth.run_depth(**YEAR_END_SPAN, out_dir=data_dir / "reports", registry=registry,
                             spec_name=EXOTICS)  # fmt: skip
    columns = ["combo_yen", "pool_yen", *(f"capacity_{float(e):g}" for e in spec["edges"])]
    part = result.table.loc[result.table["pool"].isin(NEW_POOLS), columns]
    assert len(part), "no TIERCE or bracket-quinella cell"
    tpd.assert_figure_check_is_sound(result, part.to_numpy(dtype=float).ravel())


def test_the_command_freezes_and_runs_the_exotics_spec(data_dir, summary_folder):
    """`pool-depth run --spec pool_depth_exotics` is refused before `pool-depth freeze --spec
    pool_depth_exotics`; then it records one scored depth run in the exotics family alone, writes
    its private report outside the repository and its family's file in the committed summary."""
    write_exotic_store(data_dir, exotic_races(YEAR_END, range(5)))
    family = exotics_spec()["family_id"]
    registry = default_registry("japan")

    early = tpd.invoke_run("--spec", EXOTICS)
    assert early.exit_code != 0
    assert registry.records() == []

    from horsetrack.cli import app

    frozen = CliRunner().invoke(app, ["japan", "pool-depth", "freeze", "--spec", EXOTICS])
    assert frozen.exit_code == 0, tpd.output_of(frozen)
    assert registry.frozen_spec_hash(family) == _spec_hash(exotics_spec())
    assert registry.frozen_spec_hash(V1_FAMILY) is None

    result = tpd.invoke_run("--spec", EXOTICS)

    assert result.exit_code == 0, tpd.output_of(result)
    assert [(r.family_id, r.variant, r.status) for r in registry.records()] == [
        (family, depth.VARIANT_DEPTH, "scored")
    ]
    own = summary_folder / f"{family}.md"
    assert own.is_file() and family in own.read_text(encoding="utf-8")
