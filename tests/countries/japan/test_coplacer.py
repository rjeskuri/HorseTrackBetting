"""Co-placer Place behind heavy favourites: reconciliation and decision-time test (ticket #83).

Contract pinned here (`horsetrack.countries.japan.studies.coplacer`):

The study frame: one row per runner, store column names (docs/specs/store.md), the runners table
joined with the race's `field_size` and the runner's PLACE row of the dividends table:
- `race_id`, `race_date`, `horse_no`, `win_odds` (final, decimal incl. stake), `finish_pos`
  (nullable Int64), `dead_heat` (bool), `is_scratched` (bool): as in the runners table;
- `field_size` (int): as in the races table; places paid follow `jra.yaml` PLACE.places_paid
  (8 or more pay 3, 5 to 7 pay 2, 4 or fewer: Place not sold);
- `place_dividend` (float): the dividends table's `dividend` for pool PLACE and this runner's
  combination, yen per the 100-yen unit stake (`jra.yaml` unit_stake); NaN when it did not pay;
- `place_is_refund` (bool): that dividends row's `is_refund`.
A runner's Place return is `place_dividend / 100` per unit staked, 0 when it did not pay (1.0 is
break-even; ROI = return - 1). The favourite is the race's lowest `win_odds`; a runner's rank is
1 + the number of runners of its race with strictly lower `win_odds` (the favourite is rank 1;
fixtures here never tie co-runners on odds).

- `common_filter(runners, *, min_field)`: keeps whole races only, rows unchanged (same index labels,
  same columns), and drops a race when Place is not sold (field_size < 5), its favourite is not
  unique, a runner with finish_pos <= 3 has dead_heat, any runner has place_is_refund, or
  field_size < min_field (None: every field). A dead heat for 4th keeps the race.
- `odds_ventile(odds, cut_points)`: on the input's index, the number of cut points strictly below
  the odds (bins closed on the right: odds equal to a cut point fall in the lower bin), 0 to
  len(cut_points); missing odds give a missing value.
- `odds_matched_expectation(runners, *, reference)`: on `runners`' index, the mean Place return of
  the `reference` rows in the runner's (odds ventile, places paid) cell. The ventile cut points are
  the reference's `win_odds` quantiles at 0.05, 0.10, ..., 0.95 (numpy's default, linear); places
  paid come from `field_size`. A cell the reference does not fill gives NaN. Columns read:
  win_odds, field_size, place_dividend.
- `coplacer_table(runners, *, metric)`: `runners` is already filtered; one row per (favourite band,
  co-runner rank band) cell holding runners, as columns `fav_band`, `rank_band` (or the index), and
  `n`, `return`, `lo`, `hi`. Favourite bands on the favourite's win_odds: '≤ 1.3', '1.4–1.5',
  '1.6–2.0', '> 2.0' (closed on the right); rank bands '2', '3', '4–5', '6–8', '9+'. The favourite
  itself is not a row's runner. metric 'raw': `return` is the cell's mean Place return;
  'odds_matched': the cell's mean of (Place return - odds_matched_expectation(runner,
  reference=runners)), the reference being the whole input frame, favourites included. `lo <=
  return <= hi` (the interval method and cluster unit are the owner's, not pinned). Any other
  metric raises ValueError.
- `place_payout_scenarios(place_money, pool_total, rate, *, runner, favourite)`: `place_money` maps
  every starter's horse_no to the yen on it in the Place pool; shares are money / pool_total;
  places paid P from the number of starters. Returns the dividend (yen per 100, `jra.yaml` PLACE
  dividend_step 10 and min_dividend 110) of `runner` placing, through
  `core.pricing.stake_back_share_dividend`: 'fav_places' with the favourite among its P - 1
  co-placers, 'fav_misses' with the favourite unplaced. Which other runners fill the remaining
  places is not pinned: every fixture gives all other runners equal money.
- `select_at_t(rows, race_id, *, post_utc, minutes_before=None, ranks=(2, 3)) -> list[int]`: the
  horse numbers whose WIN-odds rank on `board.snapshot.board_at(rows, race_id, t)` is in `ranks`,
  in rank order, with t = post_utc - minutes_before minutes, `minutes_before` defaulting to the
  owner's primary decision time (`snapshot.load_decision_grid().primary`, D2: T-2). Scratched
  runners are not ranked; no board by t gives []. Rows are odds-log rows (docs/specs/odds-log.md).

Expected values are derived here from first principles (explicit band tables, inverse formulas,
numpy quantiles), never from the code under test. Every frame and odds-log row is synthetic and
every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.studies.coplacer import (
    common_filter,
    coplacer_table,
    odds_matched_expectation,
    odds_ventile,
    place_payout_scenarios,
    select_at_t,
)

FAV_BANDS = ("≤ 1.3", "1.4–1.5", "1.6–2.0", "> 2.0")
RANK_BANDS = ("2", "3", "4–5", "6–8", "9+")
UNIT = 100.0
EPS = 1e-9


# ---------------------------------------------------------------------------------------------
# Synthetic study frames and first-principles references
# ---------------------------------------------------------------------------------------------


def places_paid(field: int) -> int:
    return 3 if field >= 8 else 2


def default_dividend(odds: float) -> float:
    return float(max(110, 10 * round(odds * 14)))


def make_race(
    race_id: str,
    odds: list[float],
    finish: list[int],
    *,
    horse_nos: list[int] | None = None,
    dead_heat: tuple[int, ...] = (),
    refund: tuple[int, ...] = (),
    dividends: dict[int, float] | None = None,
) -> pd.DataFrame:
    """One race; runner i has odds[i], finish_pos finish[i] and horse_no horse_nos[i] (1..n by
    default). `dead_heat` and `refund` name horse numbers. Place dividends go to finish_pos <= the
    places paid unless `dividends` (horse_no -> yen) is given; a 4-runner field pays none."""
    n = len(odds)
    horse_nos = list(range(1, n + 1)) if horse_nos is None else horse_nos
    paid = places_paid(n) if n >= 5 else 0
    if dividends is None:
        dividends = {
            h: default_dividend(o)
            for h, o, f in zip(horse_nos, odds, finish, strict=True)
            if f <= paid
        }
    return pd.DataFrame(
        {
            "race_id": race_id,
            "race_date": pd.Timestamp("2025-05-04"),
            "horse_no": horse_nos,
            "win_odds": [float(o) for o in odds],
            "finish_pos": pd.array(finish, dtype="Int64"),
            "dead_heat": [h in dead_heat for h in horse_nos],
            "is_scratched": False,
            "field_size": n,
            "place_dividend": [dividends.get(h, math.nan) for h in horse_nos],
            "place_is_refund": [h in refund for h in horse_nos],
        }
    )


def stack(races: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(races, ignore_index=True)
    frame.index = frame.index * 7 + 1000  # a non-default index, so position and label differ
    return frame


def ref_return(frame: pd.DataFrame) -> pd.Series:
    return frame["place_dividend"].fillna(0.0) / UNIT


def ref_rank(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("race_id")["win_odds"].rank(method="min").astype(int)


def ref_fav_band(odds: float) -> str:
    if odds <= 1.3 + EPS:
        return FAV_BANDS[0]
    if odds <= 1.5 + EPS:
        return FAV_BANDS[1]
    if odds <= 2.0 + EPS:
        return FAV_BANDS[2]
    return FAV_BANDS[3]


def ref_rank_band(rank: int) -> str:
    if rank == 2:
        return "2"
    if rank == 3:
        return "3"
    if rank <= 5:
        return "4–5"
    if rank <= 8:
        return "6–8"
    return "9+"


def ref_ventile(odds: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.array([sum(1 for c in cuts if c < o) for o in odds])


def ref_expectation(runners: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Mean reference Place return per (ventile, places paid), cut points at the 0.05..0.95
    quantiles of the reference's win_odds."""
    cuts = np.quantile(reference["win_odds"].to_numpy(float), np.arange(1, 20) / 20)
    ref = pd.DataFrame(
        {
            "v": ref_ventile(reference["win_odds"].to_numpy(float), cuts),
            "pp": [places_paid(f) for f in reference["field_size"]],
            "ret": ref_return(reference).to_numpy(),
        }
    )
    means = ref.groupby(["v", "pp"])["ret"].mean().to_dict()
    cells = zip(
        ref_ventile(runners["win_odds"].to_numpy(float), cuts),
        [places_paid(f) for f in runners["field_size"]],
        strict=True,
    )
    return pd.Series([means.get(c, math.nan) for c in cells], index=runners.index)


def ref_table(frame: pd.DataFrame, *, metric: str) -> dict[tuple[str, str], tuple[int, float]]:
    """(fav band, rank band) -> (n, mean) over the co-runners of an already filtered frame."""
    rank = ref_rank(frame)
    fav_odds = frame.groupby("race_id")["win_odds"].transform("min")
    value = ref_return(frame)
    if metric == "odds_matched":
        value = value - ref_expectation(frame, frame)
    co = rank > 1
    cells = pd.DataFrame(
        {
            "fb": [ref_fav_band(f) for f in fav_odds[co]],
            "rb": [ref_rank_band(r) for r in rank[co]],
            "value": value[co].to_numpy(),
        }
    )
    grouped = cells.groupby(["fb", "rb"])["value"].agg(["size", "mean"])
    return {k: (int(row["size"]), float(row["mean"])) for k, row in grouped.iterrows()}


def table_cells(table: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """The table's non-empty cells, keyed (fav band, rank band), whether the bands are columns or
    the index."""
    flat = table if "fav_band" in table.columns else table.reset_index()
    out = {}
    for rec in flat.to_dict("records"):
        assert rec["fav_band"] in FAV_BANDS, rec
        assert rec["rank_band"] in RANK_BANDS, rec
        if rec["n"] == 0:
            continue
        key = (rec["fav_band"], rec["rank_band"])
        assert key not in out, f"cell {key} appears twice"
        out[key] = rec
    return out


def population(seed: int, n_races: int) -> pd.DataFrame:
    """Random races, field 4 to 16, favourites across every band, a share of them carrying one
    defect the common filter must drop (dead heat in the top three, a refund, a tied favourite)
    or keep (a dead heat for 4th). Co-runner odds are strictly increasing and never tie."""
    rng = np.random.default_rng(seed)
    fav_grid = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.1, 2.6, 3.2]
    races = []
    for r in range(n_races):
        field = int(rng.integers(4, 17))
        fav = float(rng.choice(fav_grid))
        others = np.round(fav + 0.2 + np.cumsum(0.3 + rng.exponential(4.0, field - 1)), 1)
        odds = [fav, *others.tolist()]
        defect = "none" if field < 6 else str(rng.choice(["none"] * 5 + DEFECTS))
        if defect == "tied_fav":
            odds[1] = fav
        noisy = np.log(odds) + rng.gumbel(size=field)
        finish = (np.argsort(np.argsort(noisy)) + 1).tolist()
        dead_heat: tuple[int, ...] = ()
        if defect == "dh_top3":
            a, b = finish.index(2), finish.index(3)
            finish[b] = 2
            dead_heat = (a + 1, b + 1)
        elif defect == "dh_4th":
            a, b = finish.index(4), finish.index(5)
            finish[b] = 4
            dead_heat = (a + 1, b + 1)
        refund = (int(rng.integers(1, field + 1)),) if defect == "refund" else ()
        paid = places_paid(field) if field >= 5 else 0
        dividends = {
            i + 1: float(max(110, 10 * int((100 + 30 * (o - 1) * rng.uniform(0.5, 1.6)) / 10)))
            for i, (o, f) in enumerate(zip(odds, finish, strict=True))
            if f <= paid
        }
        perm = rng.permutation(field)  # rows not in odds order
        race = make_race(
            f"r{r:04d}:{defect}",
            odds,
            finish,
            dead_heat=dead_heat,
            refund=refund,
            dividends=dividends,
        )
        races.append(race.iloc[perm])
    return stack(races)


DEFECTS = ["dh_top3", "dh_4th", "refund", "tied_fav"]
DROPPING = ("dh_top3", "refund", "tied_fav")


def expected_kept(frame: pd.DataFrame, min_field: int | None) -> set[str]:
    """Races the common filter keeps, from how `population` built them."""
    races = frame.groupby("race_id")["field_size"].first()
    return {
        rid
        for rid, field in races.items()
        if field >= 5
        and rid.split(":")[1] not in DROPPING
        and (min_field is None or field >= min_field)
    }


# ---------------------------------------------------------------------------------------------
# Odds-matched expectation
# ---------------------------------------------------------------------------------------------

LEVELS = [1.2, 1.5, 1.9, 2.4, 3.0, 3.8, 4.7, 5.9, 7.3, 9.0]
LEVELS += [11.2, 14.0, 17.5, 22.0, 27.5, 34.0, 43.0, 55.0, 70.0, 95.0]


def ventile_reference(
    seed: int, m: int = 3, fields: tuple[int, ...] = (7, 8)
) -> tuple[pd.DataFrame, dict[tuple[int, int], float]]:
    """A reference whose 20 odds levels are its ventiles under any quantile method: per field size
    (7 pays 2, 8 pays 3) level 0 has m + 1 rows and every other level m, the same odds multiset in
    both, so each 0.05k quantile falls inside level k - 1. Returns the frame and the planted cell
    means (level, places paid) -> mean Place return."""
    rng = np.random.default_rng(seed)
    rows, means = [], {}
    for field in fields:
        for j, level in enumerate(LEVELS):
            count = m + 1 if j == 0 else m
            dividends = [
                math.nan if rng.random() < 0.4 else float(10 * rng.integers(11, 60))
                for _ in range(count)
            ]
            means[(j, places_paid(field))] = float(np.mean(np.nan_to_num(dividends) / UNIT))
            for d in dividends:
                rows.append({"win_odds": level, "field_size": field, "place_dividend": d})
    frame = pd.DataFrame(rows)
    frame.insert(0, "race_id", [f"ref{i:03d}" for i in range(len(frame))])
    frame.insert(1, "horse_no", 1)
    frame["place_is_refund"] = False
    for method in ("linear", "lower", "higher", "nearest", "midpoint"):
        cuts = np.quantile(frame["win_odds"], np.arange(1, 20) / 20, method=method)
        assert np.allclose(cuts, LEVELS[:19]), method  # the fixture's own premise
    return frame.sample(frac=1.0, random_state=seed), means


def test_odds_matched_expectation_by_ventile_and_places_paid():
    # odds_ventile on given cut points: the count of cut points strictly below, bins right-closed.
    cuts = [1.5, 2.0, 3.0, 5.0, 10.0]
    odds = [1.0, 1.5, 1.500001, 2.0, 2.5, 3.0, 4.9, 5.0, 9.99, 10.0, 10.1, 250.0, math.nan]
    index = pd.Index([40 + 3 * i for i in range(len(odds))])
    got = odds_ventile(pd.Series(odds, index=index), cuts)
    assert list(got.index) == list(index)
    for label, o in zip(index, odds, strict=True):
        if math.isnan(o):
            assert pd.isna(got[label])
        else:
            assert int(got[label]) == sum(1 for c in cuts if c < o), o
    rng = np.random.default_rng(83)
    for _ in range(5):
        cuts = np.sort(rng.uniform(1.0, 60.0, int(rng.integers(3, 20))))
        odds = np.round(rng.uniform(1.0, 80.0, 200), 1)
        got = odds_ventile(pd.Series(odds), cuts.tolist())
        assert got.astype(int).tolist() == ref_ventile(odds, cuts).tolist()

    # The expectation: the reference's mean Place return in the runner's (ventile, places paid).
    for seed in (1, 2, 3):
        reference, means = ventile_reference(seed)
        queries = []
        for j, level in enumerate(LEVELS):
            below = LEVELS[j - 1] if j else 1.0
            for field in (5, 6, 7, 8, 9, 12, 16):
                queries.append((level, field, j))  # on the level: its own ventile
                queries.append(((below + level) / 2, field, j))  # between: the upper ventile
        queries.append((140.0, 10, 19))  # beyond the last cut point
        queries.append((1.0, 6, 0))  # below the first
        runners = pd.DataFrame(
            {
                "race_id": [f"q{i}" for i in range(len(queries))],
                "horse_no": 1,
                "win_odds": [q[0] for q in queries],
                "field_size": [q[1] for q in queries],
                "place_dividend": math.nan,
                "place_is_refund": False,
            },
            index=pd.Index(range(5000, 5000 + 2 * len(queries), 2)),
        )
        got = odds_matched_expectation(runners, reference=reference)
        assert list(got.index) == list(runners.index)
        for label, (o, field, j) in zip(runners.index, queries, strict=True):
            want = means[(j, places_paid(field))]
            assert got[label] == pytest.approx(want, abs=1e-12), (o, field, j)

    # A cell the reference does not fill has no expectation: a reference of 7-runner fields only.
    reference, means = ventile_reference(4, fields=(7,))
    runners = pd.DataFrame(
        {
            "race_id": ["a", "b", "c"],
            "horse_no": 1,
            "win_odds": [LEVELS[5], LEVELS[5], LEVELS[12]],
            "field_size": [10, 6, 5],
            "place_dividend": math.nan,
            "place_is_refund": False,
        }
    )
    got = odds_matched_expectation(runners, reference=reference)
    assert math.isnan(got.iloc[0])  # (ventile 5, 3 paid): no 8+ field in the reference
    assert got.iloc[1] == pytest.approx(means[(5, 2)], abs=1e-12)
    assert got.iloc[2] == pytest.approx(means[(12, 2)], abs=1e-12)


# ---------------------------------------------------------------------------------------------
# The reconciliation table
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 8])
@pytest.mark.parametrize("min_field", [None, 8])
def test_both_definitions_use_the_same_filters(seed, min_field):
    frame = population(seed, 700)
    filtered = common_filter(frame, min_field=min_field)
    assert set(filtered["race_id"]) == expected_kept(frame, min_field)

    raw = table_cells(coplacer_table(filtered, metric="raw"))
    matched = table_cells(coplacer_table(filtered, metric="odds_matched"))

    # The identical runner set: the same cells with the same n, every co-runner counted once.
    assert set(raw) == set(matched)
    assert {k: c["n"] for k, c in raw.items()} == {k: c["n"] for k, c in matched.items()}
    co_runners = int((ref_rank(filtered) > 1).sum())
    assert sum(c["n"] for c in raw.values()) == co_runners

    for metric, cells in (("raw", raw), ("odds_matched", matched)):
        want = ref_table(filtered, metric=metric)
        assert set(cells) == set(want), metric
        for key, (n, mean) in want.items():
            cell = cells[key]
            assert int(cell["n"]) == n, (metric, key)
            assert cell["return"] == pytest.approx(mean, rel=1e-9, abs=1e-12), (metric, key)
            lo, hi = cell["lo"], cell["hi"]
            if n >= 20:
                assert math.isfinite(lo) and math.isfinite(hi), (metric, key)
            if math.isfinite(lo) and math.isfinite(hi):
                assert lo - 1e-12 <= cell["return"] <= hi + 1e-12, (metric, key)

    with pytest.raises(ValueError):
        coplacer_table(filtered, metric="roi")


CO_ODDS = [3.1, 4.2, 5.6, 7.0, 9.4, 12.5, 16.0, 21.0, 28.0, 40.0, 66.0]  # ranks 2 to 12
FAV_CASES = {
    1.1: "≤ 1.3",
    1.3: "≤ 1.3",
    1.4: "1.4–1.5",
    1.5: "1.4–1.5",
    1.6: "1.6–2.0",
    2.0: "1.6–2.0",
    2.1: "> 2.0",
    3.0: "> 2.0",
}
RANK_CASES = {2: "2", 3: "3", 4: "4–5", 5: "4–5", 6: "6–8", 8: "6–8", 9: "9+", 12: "9+"}


def test_band_edges():
    # The explicit band tables agree with the reference mapping used elsewhere in this file.
    assert all(ref_fav_band(o) == band for o, band in FAV_CASES.items())
    assert all(ref_rank_band(r) == band for r, band in RANK_CASES.items())

    rng = np.random.default_rng(1983)
    odds = [None, *CO_ODDS]  # index = rank - 1
    designed = [
        # rank 9 wins, rank 8 second, rank 3 third: '9+' and '6–8' each hold one placer
        {9: 1, 8: 2, 3: 3},
        # rank 2, rank 5, rank 6 place: '2', '4–5' and '6–8'
        {2: 1, 5: 2, 6: 3},
    ]
    for fav, fav_band in FAV_CASES.items():
        odds[0] = fav
        for trial in range(len(designed) + 3):
            if trial < len(designed):
                finish = _complete([designed[trial].get(rank, 0) for rank in range(1, 13)])
            else:
                finish = (rng.permutation(12) + 1).tolist()
            horse_nos = (rng.permutation(12) + 1).tolist()
            race = make_race(f"edge-{fav}-{trial}", odds, finish, horse_nos=horse_nos)
            race = stack([race.iloc[rng.permutation(12)]])
            cells = table_cells(coplacer_table(race, metric="raw"))

            # Every co-runner of this race sits under its favourite's band.
            assert {k[0] for k in cells} == {fav_band}, fav
            # Counts per rank band: ranks 2, 3, 4-5, 6-8, 9-12.
            assert {k[1]: int(c["n"]) for k, c in cells.items()} == {
                "2": 1,
                "3": 1,
                "4–5": 2,
                "6–8": 3,
                "9+": 4,
            }
            # Each rank band's return is the mean over exactly its ranks.
            by_rank = {r + 1: f for r, f in enumerate(finish)}
            for band in RANK_BANDS:
                members = [r for r in range(2, 13) if ref_rank_band(r) == band]
                pays = [
                    default_dividend(odds[r - 1]) / UNIT if by_rank[r] <= 3 else 0.0
                    for r in members
                ]
                got = cells[(fav_band, band)]["return"]
                assert got == pytest.approx(np.mean(pays), abs=1e-12), (fav, trial, band)


def _complete(finish: list[int]) -> list[int]:
    """Fill the zero slots of a partial finish order with the unused positions, in order."""
    used = {f for f in finish if f}
    free = iter(sorted(set(range(1, len(finish) + 1)) - used))
    return [f if f else next(free) for f in finish]


# ---------------------------------------------------------------------------------------------
# The common filter
# ---------------------------------------------------------------------------------------------


def ten(fav: float = 1.3) -> list[float]:
    return [fav, 3.4, 4.8, 6.1, 8.0, 11.5, 15.0, 22.0, 35.0, 60.0]


def test_the_common_filter_drops_dead_heats_and_refunds():
    order10 = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    races = {
        "clean10": make_race("clean10", ten(), order10),
        # dead heat for 3rd: horses 3 and 4 both 3rd
        "dh3rd": make_race("dh3rd", ten(), [1, 2, 3, 3, 5, 6, 7, 8, 9, 10], dead_heat=(3, 4)),
        # dead heat for 1st, the favourite one of the pair
        "dh1st": make_race("dh1st", ten(1.2), [1, 1, 3, 4, 5, 6, 7, 8, 9, 10], dead_heat=(1, 2)),
        # dead heat for 4th, outside the top three: kept
        "dh4th": make_race("dh4th", ten(1.5), [3, 1, 2, 4, 4, 6, 7, 8, 9, 10], dead_heat=(4, 5)),
        # a refunded Place ticket on one runner drops the race
        "refund": make_race("refund", ten(1.4), order10, refund=(7,)),
        # two runners share the lowest odds
        "tiedfav": make_race("tiedfav", [1.6, 1.6, *ten()[2:]], order10),
        # a tie further down the board does not matter
        "tied2nd": make_race("tied2nd", [1.9, 4.0, 4.0, *ten()[3:]], order10),
        "field4": make_race("field4", [1.3, 3.0, 5.0, 9.0], [1, 2, 3, 4]),  # Place not sold
        "field5": make_race("field5", [1.2, 3.0, 5.0, 9.0, 14.0], [2, 1, 3, 4, 5]),
        "field7": make_race(
            "field7", [2.4, 3.0, 5.0, 9.0, 14.0, 20.0, 41.0], [1, 3, 2, 4, 5, 6, 7]
        ),
        "field8": make_race("field8", [1.7, *ten()[1:8]], [1, 2, 3, 4, 5, 6, 7, 8]),
        "dh3rd_f6": make_race(
            "dh3rd_f6", [1.4, 3.0, 5.0, 9.0, 14.0, 20.0], [1, 2, 3, 3, 5, 6], dead_heat=(3, 4)
        ),
    }
    frame = stack(list(races.values())).sample(frac=1.0, random_state=5)
    always = {"clean10", "dh4th", "tied2nd", "field8"}
    cases = {
        None: always | {"field5", "field7"},
        6: always | {"field7"},
        8: always,
        11: set(),
    }
    for min_field, kept in cases.items():
        out = common_filter(frame, min_field=min_field)
        assert set(out["race_id"]) == kept, min_field
        assert list(out.columns) == list(frame.columns)
        want = frame[frame["race_id"].isin(kept)]
        pd.testing.assert_frame_equal(out.sort_index(), want.sort_index())


# ---------------------------------------------------------------------------------------------
# The stake-back-share mechanism
# ---------------------------------------------------------------------------------------------


def ref_dividend(share: float, co: list[float], rate: float) -> float:
    lost = 1.0 - share - sum(co)
    raw = UNIT * rate * (1.0 + lost / ((1 + len(co)) * share))
    return max(110.0, math.floor(raw / 10 + EPS) * 10)


def scenario(field: int, fav: float, runner: float, rate: float) -> tuple[dict, float, dict]:
    """place_money with the favourite on horse 3, the runner on horse 7 (or 2 in a short field)
    and every other starter on equal money; shares fav and runner of a 1,000,000-yen pool."""
    total = 1_000_000.0
    runner_no = 7 if field >= 7 else 2
    other = (1.0 - fav - runner) / (field - 2)
    money = {h: other * total for h in range(1, field + 1)}
    money[3], money[runner_no] = fav * total, runner * total
    p = places_paid(field)
    want = {
        "fav_places": ref_dividend(runner, [fav] + [other] * (p - 2), rate),
        "fav_misses": ref_dividend(runner, [other] * (p - 1), rate),
    }
    return money, total, {"runner": runner_no, "favourite": 3, "want": want}


SCENARIOS = [
    # field, favourite share, runner share, payout rate
    (10, 0.55, 0.08, 0.8),
    (10, 0.40, 0.08, 0.8),
    (14, 0.62, 0.05, 0.8),
    (8, 0.45, 0.12, 0.8),
    (12, 0.50, 0.10, 0.75),
    (6, 0.50, 0.10, 0.8),  # two places paid
    (7, 0.35, 0.20, 0.8),
    (6, 0.50, 0.30, 0.8),  # the favourite placing floors the runner at 110
]


@pytest.mark.parametrize(("field", "fav", "runner", "rate"), SCENARIOS)
def test_stake_back_share_payout_rises_when_a_heavy_favourite_misses(field, fav, runner, rate):
    money, total, args = scenario(field, fav, runner, rate)
    got = place_payout_scenarios(
        money, total, rate, runner=args["runner"], favourite=args["favourite"]
    )
    assert set(got) == {"fav_places", "fav_misses"}
    for key, want in args["want"].items():
        assert got[key] == pytest.approx(want, abs=1e-9), key
    assert got["fav_misses"] > got["fav_places"]


def test_stake_back_share_payout_is_level_when_the_favourite_is_not_heavy():
    """Control: a 'favourite' on the same money as every other runner changes nothing."""
    field, runner, rate = 10, 0.1, 0.8
    money, total, args = scenario(field, (1.0 - runner) / (field - 1), runner, rate)
    got = place_payout_scenarios(money, total, rate, runner=args["runner"], favourite=3)
    assert got["fav_places"] == pytest.approx(got["fav_misses"], abs=1e-9)
    assert got["fav_places"] == pytest.approx(args["want"]["fav_places"], abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Decision-time selection
# ---------------------------------------------------------------------------------------------

UTC = timezone.utc
RACE = "jra:2026-11-29:TOK:11"
POST = datetime(2026, 11, 29, 6, 40, tzinfo=UTC)
FINAL = "最終オッズ"


def log_rows(win: dict[int, float | None], *, fetched: datetime, final: bool = False) -> list:
    """One capture of WIN and PLACE rows (#21 shape), snapshot 45 s before the fetch. PLACE odds
    run in the reverse order of WIN, so a selection that ranked on Place would differ."""
    snap = fetched - timedelta(seconds=45)
    by_win = sorted((h for h, o in win.items() if o is not None), key=lambda h: win[h])
    place = {h: round(1.1 + 0.4 * (len(by_win) - i), 1) for i, h in enumerate(by_win)}
    rows = []
    for pool, prices in (("WIN", win), ("PLACE", place)):
        for horse in win:
            odds = prices.get(horse) if win[horse] is not None else None
            rows.append(
                {
                    "snapshot_utc": snap.isoformat(),
                    "fetched_utc": fetched.isoformat(),
                    "race_id": RACE,
                    "pool": pool,
                    "combination": str(horse),
                    "odds": odds,
                    "status_raw": "SCR" if odds is None else ("" if final else "15時38分現在"),
                    "page_state": FINAL if final else None,
                    "source_update_utc": None if final else snap.isoformat(),
                }
            )
    return rows


def before(minutes: float) -> datetime:
    return POST - timedelta(minutes=minutes)


EARLY = {1: 1.5, 2: 9.0, 3: 12.0, 4: 7.5, 5: 8.0, 6: 3.0, 7: 3.3, 8: 22.0}  # T-10: 6, 7
AT_T = {1: 1.4, 2: 9.0, 3: 6.5, 4: 3.8, 5: 4.2, 6: 30.0, 7: 15.0, 8: 22.0}  # T-2.5: 4, 5
LATE = {1: 1.4, 2: 9.0, 3: 3.5, 4: 4.5, 5: 5.0, 6: 30.0, 7: 15.0, 8: 22.0}  # T-1.5: 3, 4
FINAL_WIN = {1: 1.3, 2: 3.0, 3: 3.9, 4: 5.5, 5: 6.0, 6: 31.0, 7: 16.0, 8: 25.0}  # final: 2, 3


def race_log() -> list[dict]:
    return (
        log_rows(EARLY, fetched=before(10))
        + log_rows(AT_T, fetched=before(2.5))
        # snapshot at T-2.25, before t, but fetched at T-1.5: not available at T-2
        + log_rows(LATE, fetched=before(1.5))
        + log_rows(FINAL_WIN, fetched=POST + timedelta(minutes=1), final=True)
    )


def test_decision_time_selection_uses_the_board_at_t():
    rows = race_log()
    primary = snapshot.load_decision_grid().primary
    assert 1.5 < primary < 10  # the fixture's captures bracket the owner's decision time

    # Default t is the owner's primary decision time: the T-2.5 capture, not the final board.
    assert select_at_t(rows, RACE, post_utc=POST) == [4, 5]
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=primary) == [4, 5]
    final = snapshot.final_board(rows, RACE)
    assert final is not None and sorted(final.win, key=final.win.get)[1:3] == [2, 3]

    # Other decision times read other boards; nothing available after t is used.
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=10) == [6, 7]
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=1) == [3, 4]
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=20) == []
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=2, ranks=(4, 5)) == [3, 2]

    # Rows of another race and of a scratched runner never enter the ranking.
    other = [
        dict(r, race_id="jra:2026-11-29:TOK:12", odds=1.1)
        for r in log_rows(AT_T, fetched=before(2.4))
    ]
    scratched = {**AT_T, 4: None}
    rows = log_rows(EARLY, fetched=before(10)) + log_rows(scratched, fetched=before(3)) + other
    assert select_at_t(rows, RACE, post_utc=POST, minutes_before=2) == [5, 3]
