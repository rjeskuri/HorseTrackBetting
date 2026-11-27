"""Floor, tail and favourite avoidance rules L2, L3 and L7 (#82; 08 §8 A4, §5.1, §4 E3, E7, E12,
E14, R3; 04 §1.3).

Family L, #81's paired design (`studies.avoidance`): each rule against its stated comparator on the
same races. Every option is a ranked row (owner reframe 2026-11-27): each scored variant records
the rule's and the comparator's own mean return per unit staked at final dividends, each with its
meeting-clustered bounds (D7), the paired difference between them with its bounds, n and the
variants searched (the family size). An avoid rule drops nothing from the data: the option with
the flagged tickets (the comparator) and the option without them (the rule) are both rows. There
is no pass/fail here and nothing is adopted here.

- L2 'never bet at the floor' (E14): the base strategy's tickets minus those whose own board price
  is at or below 1.1 (WIN odds, the PLACE or Wide low figure), against the same tickets. A race
  whose every priced ticket is at the floor stays paired, the rule's side returning its stake.
- L3 'avoid the tail' (E12): no WIN or PLACE ticket on a runner whose WIN odds are above a cut-off,
  and no exotic combination holding a runner ranked 9th or worse (13th or worse as a variant) by
  WIN odds, against the same strategy with the tail included.
- L7 'flat-market and handicap favourites' (E3, E7): no favourite Place in the flattest entropy
  quintile; no favourite Win, Place or exotic box in handicaps; against the same tickets in all
  races. A race the rule skips returns its stake.
- The 100-vs-110 check (R3, 04 §1.3): the realised Place dividends of floor-bound placed
  favourites, against the Plus-10 prediction from the race's settled PLACE pool in the jravan
  `pool_votes` table (JRA-VAN H1 votes, #368; `plus10_inputs: jravan_pool_votes`). The votes are a
  label of the settled race, never a selection input, and are read lazily: the row groups whose
  race_id statistics can hold a wanted race, and only the PLACE rows of those races, five
  columns, decoded into Python. Every attempted row (a floor-bound placed favourite) stays, with
  its prediction or why it has none: votes missing or unreadable (a file that cannot be opened is
  recorded in the data build, never fatal), no votes for the race, or a premium PLACE pool whose
  settlement is unresolved (JRA may withhold the premium; #481). Every placed horse of the
  rebuilt races has its dividend reconstructed and compared with the realised one, a diagnostic:
  a miss above the frozen tolerance labels the race's row discrepant and rejects nothing. The
  counts (attempted, available, unavailable, discrepant) come from the rows' final statuses.

Selection: on history the final price (runners.win_odds), labelled LOOK_AHEAD; on logged days the
board at t (odds_timeseries through `read_asof` at the post - 2 min, D2: the post_utc of the
race's captures, else the scheduled post of `board.snapshot.post_times_for`). At t a runner priced
on the board is a runner, whatever the results say later, and one whose latest WIN or PLACE row is
SCR has no price in any pool; a ticket holding a runner the results mark scratched is refunded
(stake back). Cut points (tail odds, ranks, entropy quintiles) come from
the frozen spec, never from the data scored.

Evaluation: each paired comparison has two records. The descriptive one (#81's scheme
`paired_same_races:development`) scores every development race once, the first training span
included; its walk-forward twin (`<variant>:walk_forward`) scores the same per-race returns over
the test windows of `core.eval.folds.walk_forward` under config/folds.yaml's scheme alone,
pooled and per window. The 100-vs-110 check has the descriptive record only.

D21: the votes are JRA-VAN's. The 100-vs-110 records declare the `jravan` source, and their table
goes only to a private report and CSV written through `write_report` with that source (never
inside the repository). The main report and every L3 and L7 record read the jra store alone.

The study (`run_study`; `python -m horsetrack.countries.japan.studies.avoidance_price --run [--out
DIR]`) reads config/avoidance_price.yaml (no defaults: a missing file or key raises
`MissingOwnerDecision` and an unknown key a ValueError, before anything is read), freezes the
family in the run registry (the freeze line is the record of the inputs), reads development rows
only, scores every variant, appends one registry record per variant (failures and 'data pending'
included) and writes the reports through `write_report`.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import _EPS as FLOOR_TOLERANCE
from horsetrack.core.pricing import clean_odds, stake_back_share_dividend
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.jravan import votes as jravan_votes
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.odds_log import POOL_TOTAL
from horsetrack.countries.japan.studies import avoidance, calibration
from horsetrack.countries.japan.studies.avoidance import (
    BOARD_COLUMNS,
    DIVIDEND_COLUMNS,
    FOLD,
    LOOK_AHEAD,
    RUNNER_COLUMNS,
    SCHEME,
    RaceView,
    RuleComparison,
    StudyData,
    Ticket,
    _choice,
    _Context,
    _digest,
    _grid,
    _horses,
    _joined,
    _number,
    _paying,
    _price,
    _priced_sets,
    _score,
    _settle,
    _unit,
)
from horsetrack.countries.japan.studies.calibration import (
    MissingOwnerDecision,
    Study,
    Variant,
    _markdown_table,
    _metric,
    _real,
    normalised_entropy,
)
from horsetrack.countries.japan.takeout import race_takeout

__all__ = [
    "MissingOwnerDecision",
    "OWNER_CHOICES",
    "at_floor",
    "drop_tail",
    "entropy_quintile",
    "place_payout",
    "plus10_withheld",
    "predicted_floor_dividend",
    "read_place_votes",
    "run_study",
    "win_ranks",
    "worst_member_rank",
]

COUNTRY = "japan"
SOURCE = "jra"
VOTES_SOURCE = jravan_votes.SOURCE  # JRA-VAN: withheld under D21 (core.eval.reports)
FLOOR_ODDS = 1.1  # owner, 2026-11-24: a board price at or below 1.1 is at the floor
PLUS10_DIVIDEND = 110  # yen per 100: the floor with Plus-10 (04 §1.3)
WITHHELD_DIVIDEND = 100  # yen per 100: the floor when Plus-10 is withheld
TAILED = ("WIN", "PLACE")  # L3's odds rule: a runner's WIN or PLACE ticket


# --------------------------------------------------------------------------- the library


def at_floor(board_odds: pd.Series) -> pd.Series:
    """True where a board price (WIN odds, a PLACE or Wide low figure) is at or below 1.1, within
    `core.pricing`'s 1e-9; an unpriced figure (missing, non-finite, under 1.0) never is."""
    cleaned = clean_odds(board_odds.to_numpy())
    with np.errstate(invalid="ignore"):
        flags = ~np.isnan(cleaned) & (cleaned <= FLOOR_ODDS + FLOOR_TOLERANCE)
    return pd.Series(flags, index=board_odds.index, dtype=bool)


def _pool_figures(total_payout: float, top_ups: float, sales: float) -> None:
    for name, value in (("total_payout", total_payout), ("top_ups", top_ups), ("sales", sales)):
        if not (_real(value) and value >= 0):
            raise ValueError(f"{name} must be a finite amount, at least 0, got {value!r}")


def plus10_withheld(total_payout: float, top_ups: float, sales: float) -> bool:
    """04 §1.3: Plus-10 is withheld when total payout plus top-ups exceed sales (equal is not)."""
    _pool_figures(total_payout, top_ups, sales)
    return bool(total_payout + top_ups > sales)


def predicted_floor_dividend(total_payout: float, top_ups: float, sales: float) -> int:
    """The floor dividend per 100 yen: 100 when Plus-10 is withheld, else 110."""
    withheld = plus10_withheld(total_payout, top_ups, sales)
    return WITHHELD_DIVIDEND if withheld else PLUS10_DIVIDEND


PLACE_RULE = (
    "JRA's published Place rule (払戻金の計算方法), the stake-back share "
    "(core.pricing.stake_back_share_dividend): with S the pool's tickets net of refunds, P the "
    "placed horses, W_h the tickets on placed horse h and D = S - the placed horses' W, horse h's "
    "payout pool is (W_h + D/P) x R shared over its own tickets, R the Place rate without any "
    "premium (0.80), so its base dividend per 100 yen is R x (W_h + D/P) / W_h x 100 rounded down "
    "to 10 yen, never under 100. The top-ups are 10 yen a ticket on each placed horse at base "
    "100, and predicted_floor_dividend on the base total payout gives those horses 100 or 110. A "
    "premium or ultra_premium PLACE pool (takeout.race_takeout's boost) has no prediction: JRA may "
    "withhold its premium (takeout.boost_may_be_withheld) and the store holds no settlement "
    "signal, so the race's prediction is unavailable with that reason, never a premium inferred "
    "from the realised dividends (verified premium settlement: #481)"
)
DIVIDEND_STEP = 10  # yen per 100: JRA rounds a dividend down to 10 yen


def place_payout(
    votes: Mapping[int, int], placed: Sequence[int], sales: float, payout_rate: float
) -> tuple[dict[int, int], float, float]:
    """JRA's base Place payout, before Plus-10 and any premium (PLACE_RULE, through
    `core.pricing.stake_back_share_dividend`): (each placed horse's base dividend per 100 yen,
    100 at the floor; the base total payout, the sum of base x tickets; the top-ups, 10 yen a
    ticket on each placed horse at base 100), in yen. `votes` maps horses to their 100-yen tickets
    (unplaced horses allowed, not read); `sales` is the pool's sales in yen net of refunds, so the
    tickets on unplaced horses are sales / 100 less the placed horses'. A placed horse without
    tickets, no placed horse, placed tickets above the pool's, or a figure that is not a finite
    amount is a ValueError."""
    _pool_figures(0.0, 0.0, sales)
    if not (_real(payout_rate) and 0.0 < payout_rate <= 1.0):
        raise ValueError(f"payout_rate must be in (0, 1], got {payout_rate!r}")
    placed = list(dict.fromkeys(int(h) for h in placed))
    if not placed:
        raise ValueError("no placed horse")
    tickets = {h: votes.get(h) for h in placed}
    if not all(isinstance(n, (int, np.integer)) and n > 0 for n in tickets.values()):
        raise ValueError(f"a placed horse has no Place tickets: {tickets}")
    unit, top_up = _unit(), PLUS10_DIVIDEND - WITHHELD_DIVIDEND
    pool = sales / unit  # S, the pool's tickets
    if sum(int(n) for n in tickets.values()) > pool + FLOOR_TOLERANCE:
        raise ValueError(f"the placed horses hold more than the pool's {pool:g} tickets: {tickets}")
    shares = {h: int(n) / pool for h, n in tickets.items()}
    paid = {
        h: int(
            stake_back_share_dividend(
                share,
                [s for other, s in shares.items() if other != h],
                rate=payout_rate,
                unit=unit,
                dividend_step=DIVIDEND_STEP,
                min_dividend=WITHHELD_DIVIDEND,
            )
        )
        for h, share in shares.items()
    }
    total = float(sum(paid[h] * int(n) for h, n in tickets.items()))
    top_ups = float(
        sum(top_up * int(n) for h, n in tickets.items() if paid[h] == WITHHELD_DIVIDEND)
    )
    return paid, total, top_ups


@dataclass(frozen=True)
class PlacePool:
    """One race's settled PLACE pool in the jravan pool_votes table: its tickets net of refunds
    (None without a pool row) and each runner's tickets."""

    tickets: int | None
    votes: dict[int, int]


def _row_groups(file: pq.ParquetFile, wanted: Sequence[str]) -> list[int]:
    """The row groups whose race_id statistics can hold one of the sorted `wanted` races."""
    meta, column = file.metadata, file.schema_arrow.get_field_index("race_id")
    groups = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(column).statistics
        if stats is None or not stats.has_min_max:
            groups.append(i)
            continue
        at = bisect.bisect_left(wanted, stats.min)
        if at < len(wanted) and wanted[at] <= stats.max:
            groups.append(i)
    return groups


def read_place_votes(race_ids: Iterable[str], path: Path | None = None) -> dict[str, PlacePool]:
    """The settled PLACE pools of `race_ids` from the jravan pool_votes table (JRA-VAN, withheld
    under D21), read lazily: the row groups whose race_id statistics can hold a wanted race, five
    columns, scanned `VOTES_BATCH` rows at a time with the PLACE rows of those races as the scan's
    filter, so only they are ever decoded into Python. A race on a holdout day is a ValueError
    (D14): no holdout row is asked, and none of a row group mixing them reaches Python."""
    wanted = sorted(set(race_ids))
    holdout = [r for r in wanted if is_holdout_day(COUNTRY, date.fromisoformat(r.split(":")[1]))]
    if holdout:
        raise ValueError(f"{len(holdout)} holdout race(s) asked of pool_votes (D14)")
    if not wanted:
        return {}
    path = votes_path() if path is None else path
    with pq.ParquetFile(path, pre_buffer=False) as file:
        groups = _row_groups(file, wanted)
    scope = (ds.field("pool") == "PLACE") & ds.field("race_id").isin(pa.array(wanted, pa.string()))
    options = ds.ParquetFragmentScanOptions(pre_buffer=False)
    dataset = ds.dataset(path, format=ds.ParquetFileFormat(default_fragment_scan_options=options))
    (fragment,) = dataset.get_fragments()
    table = fragment.subset(row_group_ids=groups).to_table(
        schema=dataset.schema, columns=VOTES_COLUMNS, filter=scope, batch_size=VOTES_BATCH
    )
    pools: dict[str, PlacePool] = {}
    if not table.num_rows:
        return pools
    frame = table.to_pandas()
    for race, rows in frame.groupby("race_id", sort=True):
        total = rows.loc[rows["combination"].eq(jravan_votes.POOL_ROW)]
        tickets = None
        if len(total) == 1 and pd.notna(total["votes"].iloc[0]):
            refund = total["refund_votes"].iloc[0]
            tickets = int(total["votes"].iloc[0]) - (0 if pd.isna(refund) else int(refund))
        runners = rows.loc[rows["combination"].ne(jravan_votes.POOL_ROW) & rows["votes"].notna()]
        pairs = zip(runners["combination"], runners["votes"], strict=True)
        pools[str(race)] = PlacePool(tickets, {int(c): int(v) for c, v in pairs})
    return pools


def votes_path() -> Path:
    return paths.processed_dir(VOTES_SOURCE) / jravan_votes.FILENAME


def _win_prices(view: RaceView) -> dict[int, float]:
    """Each runner priced in the WIN pool of the view's board -> its WIN odds."""
    board = view.board
    combinations = board["combination"].to_numpy()
    win = (board["pool"].to_numpy() == "WIN") & (combinations != POOL_TOTAL)
    odds = clean_odds(board["odds"].to_numpy()[win])
    return {
        int(c): float(o)
        for c, o in zip(combinations[win], odds, strict=True)
        if str(c).isdigit() and not np.isnan(o)
    }


def win_ranks(view: RaceView) -> dict[int, int]:
    """Each runner priced in the view's WIN pool -> 1 + the number of runners priced strictly
    shorter (tied prices share the better rank, as #117's board flags rank them)."""
    prices = _win_prices(view)
    odds = np.array(list(prices.values()), dtype=float)
    return {h: 1 + int(np.sum(odds < o)) for h, o in prices.items()}


def worst_member_rank(combination: str, rank_at_t: dict[int, int]) -> int:
    """The largest rank among the combination's horses; a horse without a rank is a KeyError."""
    return int(max(rank_at_t[h] for h in _horses(combination)))


def drop_tail(view: RaceView, tickets: Iterable[Ticket], *, max_odds: float) -> list[Ticket]:
    """L3's Win/Place rule: the tickets less every WIN or PLACE ticket whose runner's WIN odds on
    the view's board are strictly above `max_odds` (a PLACE ticket is judged by its runner's WIN
    odds); the rest keep their order."""
    if not (_real(max_odds) and max_odds > 1.0):
        raise ValueError(f"max_odds must be decimal odds above 1.0, got {max_odds!r}")
    prices = _win_prices(view)

    def tail(ticket: Ticket) -> bool:
        if ticket.pool not in TAILED:
            return False
        return prices.get(int(ticket.combination), math.nan) > max_odds

    return [t for t in tickets if not tail(t)]


def _cut_points(cut_points: Iterable[float]) -> np.ndarray:
    try:
        cuts = list(cut_points)
    except TypeError:
        cuts = []
    if not (
        len(cuts) == 4
        and all(_real(c) for c in cuts)
        and all(a < b for a, b in itertools.pairwise(cuts))
    ):
        raise ValueError(f"cut points must be four strictly increasing numbers, got {cut_points!r}")
    return np.asarray(cuts, dtype=float)


def entropy_quintile(entropy: float, cut_points: tuple[float, float, float, float]) -> int:
    """1 (the most concentrated market) to 5 (the flattest); a value equal to a cut point goes to
    the upper quintile (#77's binning)."""
    cuts = _cut_points(cut_points)
    if not (_real(entropy) and 0.0 <= entropy <= 1.0):
        raise ValueError(f"entropy must be a normalised entropy in [0, 1], got {entropy!r}")
    return int(np.searchsorted(cuts, float(entropy), side="right")) + 1


# =========================================================================== the study

TICKET = 82
# A changed pre-registration is a new family; the earlier ones keep their records (SUPERSEDES)
FAMILY_ID = "japan-family-l-floor-tail-favourite-v5"
SUPERSEDES = {
    "japan-family-l-floor-tail-favourite": "the review of #82 (2026-11-27): t from the scheduled "
    "post time when the captures carry none; selection at t from the board at t alone, a runner "
    "scratched after t refunded; the 100-vs-110 check's coverage at t and its reconstruction "
    "check with a frozen tolerance; L2's no-bet race kept under stake_back; the realised counts "
    "kept when the votes cannot be read",
    "japan-family-l-floor-tail-favourite-v2": "at t it counted as a runner every horse of a "
    "priced row on the board, and the logged Wide board still prices the pairs of a horse "
    "scratched before t: the runners at t are now those priced in the WIN or PLACE pool",
    "japan-family-l-floor-tail-favourite-v3": "the review of #82, round 2 (2026-11-27): the "
    "100-vs-110 check's Place rule is JRA's published stake-back share "
    "(core.pricing.stake_back_share_dividend) in place of an equal split of the profit, Plus-10 "
    "on the base payout and the 5% premium after it on a premium PLACE pool; eligible "
    "favourites without readable votes leave the prediction unavailable (no check, no cell); a "
    "runner whose latest WIN or PLACE row at t is SCR has no price in any pool at t; a failed "
    "check clears its rows' predictions",
    "japan-family-l-floor-tail-favourite-v4": "fix round 4 of #82 (2026-11-27, the owner's "
    "restart plan): a premium PLACE pool's prediction is unavailable (its settlement unresolved) "
    "in place of a premium taken as paid; the reconstruction check is a diagnostic labelling a "
    "race's row discrepant, never a rejection of the record; the counts come from the rows' "
    "final statuses (attempted, available, unavailable, discrepant); a votes file that cannot be "
    "read is recorded and the run goes on; each paired comparison gains a walk-forward record "
    "over the test windows of config/folds.yaml's scheme",
}
CONFIG_NAME = "avoidance_price"
WALK_FORWARD = "walk_forward"  # E: a walk-forward record's name suffix and fold_scheme prefix
WALK_FORWARD_FOLD = "walk_forward:test_windows"  # its pooled metrics' fold label
STAKE_BACK, NO_BET_RETURN = "stake_back", 1.0  # L2's no-bet race returns its stake, as L7's skip
VALIDATION_TOLERANCE = 10  # yen per 100 staked: one 10-yen rounding step of a Place dividend
POST_TIME = (
    "the post_utc of the race's latest capture carrying one; when no capture carries it, the "
    "scheduled post of board.snapshot.post_times_for(day, venue) (the <VENUE>.races.json sidecar "
    "beside the odds log, else HKTrack's stored card); with neither, the race has no t and is data "
    "pending"
)
BOOTSTRAP_SEED = 82  # reproducibility only; scoring.yaml signs no seed
TAIL_RANKS = (9, 13)  # the request (E12): a runner ranked 9th or worse; 13th or worse a variant
FLATTEST = 5
EXOTIC_POOLS = ("QUINELLA_PLACE", "QUINELLA", "FORECAST", "TRIO", "TIERCE")
ORDERED = ("FORECAST", "TIERCE")
LEGS = {"QUINELLA_PLACE": 2, "QUINELLA": 2, "FORECAST": 2, "TRIO": 3, "TIERCE": 3}
PRICE_POOLS = ("WIN", "PLACE", *EXOTIC_POOLS)
BOARD_POOLS = ("WIN", "PLACE", "QUINELLA_PLACE")
FINAL, AT_T = "final_look_ahead", "board_at_t"
TAGS = (FINAL, AT_T)
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "status", "weight_type", "race_class_raw"]
PLUS10_COLUMNS = [
    "variant", "race_id", "horse_no", "place low at t", "realised", "predicted", "why",
    "diagnostic",
]  # fmt: skip
JRAVAN_VOTES, DATA_PENDING = "jravan_pool_votes", "data_pending"
CELLS = ("100", "110", "above_110")  # the 100-vs-110 table's predicted and realised dividends
VOTES_COLUMNS = ["race_id", "pool", "combination", "votes", "refund_votes"]
VOTES_BATCH = 1 << 20  # rows decoded at a time from pool_votes
OWNER_CHOICES: dict[str, tuple[str, ...]] = {
    "sets": ("every_set",),
    "favourite_ties": ("lower_horse_no", "drop_race"),
    "l7_favourite_boxes": ("top3_box", "none"),
    "l7_skipped_race": ("stake_back",),
    "plus10_inputs": (JRAVAN_VOTES, DATA_PENDING),
}
OWNER_KEYS = {
    "l3_tail_odds": "L3's Win/Place tail cut-offs, a non-empty list of distinct decimal odds above "
    "1.0 ([x] for one), each its own variant: a runner whose WIN odds are strictly above x is in "
    "the tail ('100/1' is 101.0 read as fractional odds, 100.0 as decimal; 50/1 as a variant)",
    "l7_entropy_cut_points": "L7's four entropy cut points, increasing in (0, 1): a race at or "
    "above the last is in the flattest quintile; they bin the normalised entropy of the final Win "
    "odds (history) and of the Win board at t",
    "sets": f"one of {list(OWNER_CHOICES['sets'])}: the base strategy L2 and L3 bet, every runner "
    "(WIN, PLACE) and every combination of the starters in each exotic pool, one unit each; where "
    "L2 reads a set's own price, the sets priced on that pool's board",
    "favourite_ties": f"one of {list(OWNER_CHOICES['favourite_ties'])}: equal shortest Win prices "
    "(L7's favourite and top-3 box, the 100-vs-110 favourite): the lower horse number first, or "
    "the race is not scored when a tie changes the pick",
    "l7_favourite_boxes": f"one of {list(OWNER_CHOICES['l7_favourite_boxes'])}: L7's 'favourite "
    "exotic boxes' in handicaps, the box of the three shortest Win prices in each of "
    f"{list(EXOTIC_POOLS)} (each pool its own variant at both selections), or none",
    "l7_skipped_race": f"one of {list(OWNER_CHOICES['l7_skipped_race'])}: L7 is paired on all "
    "races; a race the rule skips returns its stake (1.0), so the difference there is 1 minus the "
    "favourite ticket's return, and 0 elsewhere",
    "plus10_inputs": f"one of {list(OWNER_CHOICES['plus10_inputs'])}: the 100-vs-110 prediction's "
    f"inputs. {JRAVAN_VOTES}: the per-runner Place votes and the Place pool's sales of the settled "
    "race, from the jravan pool_votes table (JRA-VAN H1, #368; withheld under D21). "
    f"{DATA_PENDING}: no prediction, the realised dividends only",
}
DECISIONS = {
    "D2": "the board at t is read at post - primary_min of config/decision.yaml (T-2)",
    "D3": "every JRA pool is in research scope (amended 2026-11-24, #127); exotics are tier 2",
    "D7": "meeting (venue x day) clusters, 95% one-sided bounds (config/scoring.yaml)",
    "D10": "races from 2014-06-07 only",
    "D14": "holdout days fully locked, outcomes and prices: no holdout row is read",
    "D21": "JRA-VAN-derived figures never enter the repository: the 100-vs-110 table built on "
    "the jravan pool_votes table is in a private report only",
    "ranking": "owner reframe 2026-11-27: every option (each rule and its comparator) is a "
    "ranked row with its mean return, clustered bounds, n and the variants searched; the paired "
    "difference against the stated comparator is reported with its bounds; there is no pass/fail "
    "and nothing is adopted here",
    "floor": "a board price at or below 1.1 is at the floor (owner, 2026-11-24)",
    "handicap": "races.weight_type 'handicap' (calibration's handicap segment, #77)",
    "entropy": "calibration.normalised_entropy, the E3 measure",
}


# --------------------------------------------------------------------------- inputs and spec


def owner_inputs() -> dict:
    """The owner's pre-registered inputs from config/avoidance_price.yaml, validated. The schema
    is closed: an unknown key (`owner_signoff` among them, removed by the owner's reframe of
    2026-11-27: the freeze line in the registry is the record) is a ValueError naming it."""
    where = f"countries/{COUNTRY}/config/{CONFIG_NAME}.yaml"
    try:
        raw, problem = load_country_yaml(COUNTRY, CONFIG_NAME), "lacks"
    except FileNotFoundError:
        raw, problem = {}, "does not exist; it carries"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of {sorted(OWNER_KEYS)}")
    missing = [key for key in OWNER_KEYS if key not in raw]
    if missing:
        needed = "; ".join(f"{key}: {OWNER_KEYS[key]}" for key in missing)
        raise MissingOwnerDecision(f"{where} {problem} the owner decision(s) {needed}")
    unknown = sorted(set(raw) - set(OWNER_KEYS))
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")
    cuts = raw["l7_entropy_cut_points"]
    if not (isinstance(cuts, list) and all(_real(c) and 0.0 < c < 1.0 for c in cuts)):
        raise ValueError(f"l7_entropy_cut_points must be four numbers in (0, 1), got {cuts!r}")
    return {
        "l3_tail_odds": _grid(raw, "l3_tail_odds"),
        "l7_entropy_cut_points": [float(c) for c in _cut_points(cuts)],
        **{key: _choice(raw, key, options) for key, options in OWNER_CHOICES.items()},
    }


def bootstrap_settings() -> dict:
    """D7's bootstrap (config/scoring.yaml, checked by calibration's reader), this study's seed."""
    return {**calibration.bootstrap_settings(), "seed": BOOTSTRAP_SEED}


@dataclass(frozen=True)
class _Run:
    """One variant of the family: its rule, what it computes, its pool, selection and cut."""

    rule: str
    kind: str  # floor, plus10, tail_odds, tail_rank, flat, handicap, box
    pool: str
    tag: str  # FINAL or AT_T
    cut: float | None = None

    @property
    def name(self) -> str:
        middle = {
            "floor": self.pool,
            "plus10": "plus10_100_vs_110",
            "tail_odds": f"{self.pool}:above_{_number(self.cut or 0)}",
            "tail_rank": f"{self.pool}:rank_{_number(self.cut or 0)}_or_worse",
            "flat": f"flattest_quintile:{self.pool}",
            "handicap": f"handicap:{self.pool}",
            "box": f"handicap:{self.pool}_top3_box",
        }[self.kind]
        return f"{self.rule}:{middle}:{self.tag}"

    @property
    def board_pool(self) -> str:
        """The pool whose board at t the selection reads."""
        return self.pool if self.kind in ("floor", "plus10") else "WIN"


def _plan(rules: dict) -> list[_Run]:
    """Every variant of the family, in scoring order (D17: each cut point counts)."""
    runs = [_Run("L2", "floor", "WIN", tag) for tag in TAGS]
    runs += [_Run("L2", "floor", pool, AT_T) for pool in ("PLACE", "QUINELLA_PLACE")]
    runs += [_Run("L2", "plus10", "PLACE", tag) for tag in TAGS]
    l3, l7 = rules["L3"], rules["L7"]
    runs += [
        _Run("L3", "tail_odds", pool, tag, x)
        for x in l3["tail_odds"]
        for pool in TAILED
        for tag in TAGS
    ]
    runs += [
        _Run("L3", "tail_rank", pool, tag, r)
        for r in l3["tail_ranks"]
        for pool in l3["rank_pools"]
        for tag in TAGS
    ]
    runs += [_Run("L7", "flat", "PLACE", tag) for tag in TAGS]
    runs += [_Run("L7", "handicap", pool, tag) for pool in TAILED for tag in TAGS]
    runs += [_Run("L7", "box", pool, tag) for pool in l7["box_pools"] for tag in TAGS]
    return runs


def _rules(owner: dict) -> dict:
    boxes = owner["l7_favourite_boxes"] == "top3_box"
    return {
        "L2": {
            "floor_odds": FLOOR_ODDS,
            "rule": "the base strategy's tickets less those whose own price on the board is at or "
            "below the floor: WIN odds, the PLACE low figure, the Wide low figure",
            "comparator": "the same tickets, those at the floor included (08 §5.1)",
            "no_bet_race": STAKE_BACK,
            "no_bet_rule": "a race whose every ticket the comparator bets is at the floor stays "
            "paired: the rule bets nothing there and its side returns its stake (1.0), as L7's "
            "stake_back; only a race without board data (the comparator bets nothing) drops",
            "pools": {FINAL: ["WIN"], AT_T: list(BOARD_POOLS)},
            "pools_why": "the store holds no final PLACE or Wide board on history",
            "plus10": {
                "rule": "04 §1.3: a dividend that would be 100 yen pays 110 unless total payout "
                "plus top-ups would exceed sales, then 100 (equal is not withheld)",
                FINAL: "the favourite on the final Win odds that placed with a Place dividend at "
                "the floor (100 or 110)",
                AT_T: "the favourite on the Win board at t whose PLACE low figure at t is at or "
                "below the floor, and that placed",
                "coverage": "a race counts once its favourite is priced: at t, once the favourite "
                "at t has a PLACE price at t; with no race counted the check is data pending",
                "inputs": owner["plus10_inputs"],
                "prediction": PLACE_RULE if owner["plus10_inputs"] == JRAVAN_VOTES else "none",
                "validation": (
                    {
                        "rule": "a diagnostic, never a rejection: every placed horse of the "
                        "check's rebuilt races (the floor favourite and the others) has its Place "
                        "dividend reconstructed from the votes (the prediction's rule: the base, "
                        "then Plus-10 on the base total) and compared with the realised one; a "
                        "race with one off by more than the tolerance has its row labelled "
                        "discrepant, and that row keeps its realised dividend and its prediction, "
                        "which the predicted x realised table counts",
                        "tolerance_yen_per_100": VALIDATION_TOLERANCE,
                        "why": "JRA rounds Place dividends down to 10 yen, and a floor horse's "
                        "100 or 110 is the check's own question",
                    }
                    if owner["plus10_inputs"] == JRAVAN_VOTES
                    else "none"
                ),
                "counts": "from the rows' final statuses: attempted (the floor-bound placed "
                "favourites), available (with a prediction), unavailable (without one, each with "
                "its reason) and discrepant (available, the race labelled discrepant)",
                "unavailable": "a row has no prediction when its race cannot be rebuilt: the "
                "pool_votes file missing or unreadable (recorded in its data build, never fatal), "
                "no votes for the race, or a premium PLACE pool whose settlement is unresolved; "
                "eligible favourites none of which has a prediction leave the realised counts "
                "(jra store) scored and no diagnostic or predicted cell; a covered selection with "
                "no eligible favourite is a scored zero, every cell 0",
            },
        },
        "L3": {
            "tail_odds": list(owner["l3_tail_odds"]),
            "tail_ranks": list(TAIL_RANKS),
            "odds_rule": "no WIN or PLACE ticket on a runner whose WIN odds on the board are "
            "strictly above the cut-off; each cut-off its own variant",
            "rank_rule": "no combination holding a runner ranked at or worse than the cut-off on "
            "the Win board (1 + the runners priced strictly shorter); each rank its own variant",
            "rank_pools": list(EXOTIC_POOLS),
            "comparator": "the same strategy with the tail included (08 §5.1)",
            "no_bet_race": "dropped: the rule bets nothing only when every priced runner is in "
            "the tail, which a full board cannot give (its favourite is inside every cut)",
        },
        "L7": {
            "entropy_cut_points": list(owner["l7_entropy_cut_points"]),
            "flattest_quintile": FLATTEST,
            "entropy": "the normalised entropy of the Win odds priced on the board, binned by "
            "entropy_quintile (a value at a cut point goes up)",
            "flat_rule": "no favourite Place in a race of the flattest quintile",
            "handicap_rule": "no favourite Win or Place"
            + (", and no top-3 favourite box, in a handicap" if boxes else " in a handicap"),
            "boxes": owner["l7_favourite_boxes"],
            "box_pools": list(EXOTIC_POOLS) if boxes else [],
            "favourite": "the shortest WIN price among the starters on the board",
            "favourite_ties": owner["favourite_ties"],
            "skipped_race": owner["l7_skipped_race"],
            "comparator": "the same favourite tickets in all races (08 §5.1)",
        },
        "sets": owner["sets"],
        "stake_yen": _unit(),
    }


def _names(rules: dict) -> list[str]:
    """Every record's variant in scoring order: each run's whole-development record and, for a
    paired comparison, its walk-forward record next to it."""
    names = []
    for run in _plan(rules):
        names += [run.name, *([_forward_name(run.name)] if run.kind != "plus10" else [])]
    return names


def variant_names(owner: dict) -> list[str]:
    return _names(_rules(owner))


def _forward_name(name: str) -> str:
    return f"{name}:{WALK_FORWARD}"


def walk_forward_scheme(scheme: FoldScheme) -> str:
    """The walk-forward records' fold_scheme, naming the scheme it scores under."""
    s = scheme
    return (
        f"{WALK_FORWARD}:{s.unit} x{s.test_length} step {s.step}, {s.window}, min_train "
        f"{s.min_train}, embargo {s.embargo_days}d, anchor month {s.anchor_month}"
    )


def build_spec(owner: dict, bootstrap: dict, minutes: int, scheme: FoldScheme) -> dict:
    """The family's pre-registered spec, frozen in the registry before any race is read; `scheme`
    is Japan's walk-forward scheme (config/folds.yaml, core.eval.folds.load_country_scheme)."""
    rules = _rules(owner)
    variants = _names(rules)
    votes = owner["plus10_inputs"] == JRAVAN_VOTES
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "supersedes": dict(SUPERSEDES),
        "hypotheses": {
            "L2": "E14: the return of the base strategy less its tickets at the floor against "
            "the same tickets; R3: the 100-vs-110 check of floor-bound placed favourites",
            "L3": "E12: the return of the strategy less its tail tickets against the strategy "
            "with the tail",
            "L7": "E3, E7: the return with no favourite Place in the flattest-entropy quintile, "
            "and no favourite Win, Place or box in handicaps, against the same tickets in all "
            "races",
        },
        "rules": rules,
        "ranked_rows": "each scored comparison is two ranked options, the rule and its "
        "comparator, each with its mean return per unit staked over the paired races and its "
        "meeting-clustered bounds, and their paired difference with its bounds; every record "
        "carries the family size as variants_searched",
        "difference": "per paired race, the rule's payout / stake minus the comparator's at final "
        "dividends; for L7 a race the rule skips returns its stake (1.0). Its mean over paired "
        "races, with D7's meeting-clustered paired bootstrap (one-sided bounds at the level): "
        "positive is the rule returning more than its comparator",
        "selections": {
            LOOK_AHEAD: "history: the final Win odds (runners.win_odds), so the choice sees the "
            "closing price",
            avoidance.BOARD_AT_T: "odds_timeseries through read_asof at the race's post time - "
            f"{minutes} min, the latest snapshot per (pool, combination)",
        },
        "post_time": POST_TIME,
        "decision_time_min": minutes,
        "board_fields": {
            "odds_timeseries": BOARD_COLUMNS,
            "runners": RUNNER_COLUMNS,
            "races": RACE_COLUMNS,
            "dividends": DIVIDEND_COLUMNS,
        },
        "starters": {
            LOOK_AHEAD: "runners not is_scratched (the results know the scratches)",
            avoidance.BOARD_AT_T: "every runner priced in the WIN or PLACE pool of the board at "
            "t, whatever the results say later; a runner whose latest WIN or PLACE row at t is "
            "SCR is scratched by t and has no price in any pool at t, Wide pairs included, "
            "whatever an older row of another pool says (docs/specs/board-at-t.md)",
            "settlement": "a ticket holding a runner that runners.is_scratched marks is refunded, "
            "stake back, as JRA settles it (the store holds no refund row for a scratch)",
        },
        "races": "development races with status 'run' from 2014-06-07 (D10); holdout days unread",
        "evaluation": {
            "descriptive": f"fold_scheme {SCHEME}: every development race once, the first "
            "training span included; the historical descriptive output (#81's scheme)",
            WALK_FORWARD: {
                "scheme": dataclasses.asdict(scheme),
                "scheme_source": f"countries/{COUNTRY}/config/folds.yaml "
                "(core.eval.folds.load_country_scheme)",
                "fold_scheme": walk_forward_scheme(scheme),
                "calendar": "core.eval.folds.walk_forward over the race days of the development "
                "races read (status 'run')",
                "records": "beside each paired comparison's descriptive record, one named "
                f"'<variant>:{WALK_FORWARD}': the same per-race returns over the races of the "
                f"test windows alone; pooled over them (fold '{WALK_FORWARD_FOLD}') the paired "
                "difference and each side's mean with D7's bounds, n and the variants "
                "searched; per window (fold '<first day>..<last day>') its paired difference "
                "and n. Nothing is fitted, so the scheme only decides which races are scored; "
                "a comparison with no paired race in a window is failed with its reason. The "
                "100-vs-110 check has no walk-forward record",
                "cut_points": "#77's entropy cut points come from 2014-06-07..2019-12-31, the "
                "first window's training span under Japan's scheme, so its test windows are out "
                "of their sample",
            },
        },
        "variants": variants,
        "family_size": len(variants),  # D17: every cut point counts
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "adoption": "none here: options are ranked (owner reframe 2026-11-27)",
        "decisions": DECISIONS,
        "place_votes": (
            {
                "table": f"processed/{VOTES_SOURCE}/{jravan_votes.FILENAME}",
                "columns": VOTES_COLUMNS,
                "rows": "pool PLACE of the development races the check needs, never at t",
                "sales": f"the {jravan_votes.POOL_ROW} row's votes less its refund_votes, x the "
                "unit stake",
                "payout_rate": "takeout.race_takeout(race, 'PLACE')'s payout rate (0.80 for "
                "Place since 2014-06-07); a pool whose boost may be withheld (premium, "
                "ultra_premium) has no prediction (PLACE_RULE)",
                "placed": "the race's paying PLACE rows in the jra dividends table",
                "withheld": "D21: records built on it declare the jravan source",
            }
            if votes
            else None
        ),
        "sources": [SOURCE, VOTES_SOURCE] if votes else [SOURCE],
    }


# --------------------------------------------------------------------------- the pure study

Sides = avoidance.Sides
Paired = avoidance.Paired


VotesReader = Callable[[Iterable[str]], dict[str, PlacePool]]


def _board_horses(combination) -> list[int]:
    """The horses a board row names ([] for a pool total or a row naming none)."""
    parts = str(combination).split(",")
    return [int(p) for p in parts] if all(p.isdigit() for p in parts) else []


def _scratched_at_t(board: pd.DataFrame) -> frozenset[int]:
    """The board-at-t rule (docs/specs/board-at-t.md, `board.snapshot`): a runner whose latest WIN
    or PLACE row at t is SCR is scratched by t, whatever an older row of another pool says."""
    if "status_raw" not in board:
        return frozenset()
    rows = zip(board["pool"], board["combination"], board["status_raw"], strict=True)
    return frozenset(
        h
        for pool, combination, status in rows
        if pool in (snapshot.WIN, snapshot.PLACE)
        and isinstance(status, str)
        and status == snapshot.SCRATCHED
        for h in _board_horses(combination)
    )


def _board_runners(view: RaceView) -> frozenset[int]:
    """Every horse priced in the WIN or PLACE pool of the view's board (the board at t has no
    price for a runner scratched by t, in any pool: `_scratched_at_t`)."""
    board = view.board
    rows = board.loc[board["pool"].isin(TAILED) & board["combination"].ne(POOL_TOTAL)]
    priced = ~np.isnan(clean_odds(rows["odds"].to_numpy()))
    return frozenset(int(c) for c in rows["combination"].to_numpy()[priced] if str(c).isdigit())


def _fold_labels(races: pd.DataFrame, scheme: FoldScheme) -> dict[str, str]:
    """E: each development race in a test window of `core.eval.folds.walk_forward` over the
    races' days -> its window's label, '<first day>..<last day>'."""
    if races.empty:
        return {}
    ids = races["race_id"].to_numpy()
    return {
        ids[i]: f"{fold.test_start.isoformat()}..{fold.test_end.isoformat()}"
        for fold in walk_forward(races["race_date"], scheme=scheme)
        for i in fold.test_idx
    }


class _Ctx(_Context):
    """#81's context over this study's pools, with the handicaps, the races' conditions labels,
    the runners the results mark scratched, the Place votes reader and cached boards (final and
    at t)."""

    def __init__(self, data: StudyData, spec: dict, place_votes: VotesReader | None) -> None:
        super().__init__(data, spec, pools=PRICE_POOLS)
        runners = data.runners.loc[data.runners["race_id"].isin(set(self.race_ids))]
        out = runners.loc[runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
        self.scratched = {
            race: frozenset(int(h) for h in horses)
            for race, horses in out.groupby("race_id")["horse_no"]
        }
        races = self.races.set_index("race_id")
        handicap = races["weight_type"].eq("handicap").to_numpy(dtype=bool, na_value=False)
        self.handicaps = frozenset(races.index[handicap])
        classes = races["race_class_raw"] if "race_class_raw" in races else pd.Series(dtype=object)
        self.class_of = {r: c for r, c in classes.items() if isinstance(c, str)}
        self.unit = _unit()
        self.l7 = spec["rules"]["L7"]
        self.place_votes = place_votes
        self.view_at_t = functools.cache(self._board_at_t)
        self.final_win_view = functools.cache(super().final_win_view)
        forward = spec["evaluation"][WALK_FORWARD]
        self.fold_scheme = forward["fold_scheme"]
        self.fold_of = _fold_labels(self.races, FoldScheme.from_mapping(forward["scheme"]))

    def _board_at_t(self, race_id: str) -> RaceView:
        if self.data.view_at_t is None:
            raise ValueError("no board-at-t reader")
        view = self.data.view_at_t(race_id)
        board = view.board
        kept = board.loc[board["pool"].isin(BOARD_POOLS), ["pool", "combination", "odds"]]
        kept = kept.reset_index(drop=True)
        out = _scratched_at_t(board)
        if out:  # no price in any pool at t, Wide pairs included
            names = [bool(out & set(_board_horses(c))) for c in kept["combination"]]
            kept.loc[np.asarray(names, dtype=bool), "odds"] = np.nan
        return RaceView(race_id, kept, view.selection)

    @functools.cached_property
    def paying(self) -> dict:
        return _paying(self.dividends)

    def runners(self, view: RaceView) -> frozenset[int]:
        """Who may be bet: at the final price the race's starters (look-ahead: the results know
        the scratches); at t every runner priced in the WIN or PLACE pool of the board at t,
        whatever the results say later (a runner scratched after t is refunded when settled)."""
        if view.selection == LOOK_AHEAD:
            return self.starters_of(view.race_id)
        return _board_runners(view)

    def settle(self, race: str, tickets: Iterable[Ticket]) -> tuple[int, float, str | None]:
        """#81's `_settle` (stake, payout, why the race cannot be settled), with every ticket
        holding a runner the results mark scratched refunded, stake back: JRA refunds it, and the
        store holds no refund row for a scratch."""
        out, kept, back = self.scratched.get(race, frozenset()), [], 0
        for ticket in tickets:
            if out and out & set(_horses(ticket.combination)):
                back += int(ticket.stake)
            else:
                kept.append(ticket)
        stake, payout, why = _settle(kept, self.paying.get(race, {}))
        return stake + back, payout + back, why

    def selection(self, run: _Run) -> tuple[str, list[str], Callable[[str], RaceView], str]:
        """(label, races, view, why a variant without a paired race has none)."""
        if run.tag == FINAL:
            return LOOK_AHEAD, self.race_ids, self.final_win_view, "no paired race"
        pending = f"data pending: no {run.board_pool} board at t (none logged, or no post time)"
        return self.at_t, self.logged(run.board_pool), self.view_at_t, pending


def _favourites(view: RaceView, starters: frozenset[int], k: int, ties: str) -> list[int] | None:
    """The k shortest Win prices among the priced starters, the lower horse number first on equal
    prices; None under k or, with ties 'drop_race', when a tie changes the pick."""
    priced = sorted((o, h) for h, o in _win_prices(view).items() if h in starters)
    if len(priced) < k:
        return None
    if ties == "drop_race" and len(priced) > k and priced[k - 1][0] == priced[k][0]:
        return None
    return [h for _, h in priced[:k]]


def _entropy(view: RaceView) -> float:
    odds = list(_win_prices(view).values())
    if len(odds) < 2:
        return math.nan
    race = pd.Series([view.race_id] * len(odds))
    return float(normalised_entropy(pd.Series(odds, dtype=float), race).iloc[0])


def _floor_sides(ctx: _Ctx, pool: str, changed: set) -> Sides:
    """L2: every set of the runners priced in `pool` on the board, less those at the floor; the
    rule's empty list is a no-bet race (every priced set at the floor), not missing data."""

    def comparator(view: RaceView) -> list[Ticket] | None:
        sets = _priced_sets(view, pool, ctx.runners(view))
        return [Ticket(pool, c, ctx.unit) for c, _ in sets] or None

    def rule(view: RaceView) -> list[Ticket] | None:
        sets = _priced_sets(view, pool, ctx.runners(view))
        floor = at_floor(pd.Series([o for _, o in sets], dtype=float)).tolist()
        kept = [Ticket(pool, c, ctx.unit) for (c, _), f in zip(sets, floor, strict=True) if not f]
        if len(kept) < len(sets):
            changed.add(view.race_id)
        return kept

    return rule, comparator


def _tickets_paired(
    ctx: _Ctx, races: Sequence[str], view, sides: Sides, no_bet: float | None = None
) -> Paired:
    """#81's `compare`, settled by `ctx.settle` (a ticket holding a runner scratched after the
    choice is refunded). A race the comparator bets nothing in (no board data) drops; a race the
    rule bets nothing in drops too or, with `no_bet`, stays paired with the rule's side returning
    `no_bet` per unit staked (L2's stake_back: 1.0)."""
    rule, comparator = sides
    returns, reasons = {}, {}
    for race in races:
        seen = view(race)
        ours, theirs = rule(seen), comparator(seen)
        if not theirs:
            reasons[race] = "no priced ticket on the board (the comparator bets nothing)"
            continue
        if not ours and no_bet is None:
            reasons[race] = "the rule does not cover it"
            continue
        settled = [ctx.settle(race, tickets) for tickets in (theirs, *([ours] if ours else []))]
        why = next((w for _, _, w in settled if w), None)
        if why is None and min(s for s, _, _ in settled) <= 0:
            why = "a side stakes nothing"
        if why:
            reasons[race] = why
            continue
        back = [payout / stake for stake, payout, _ in settled]
        returns[race] = (back[1] if ours else float(no_bet), back[0])
    return list(races), returns, reasons


def _tail_sides(ctx: _Ctx, pool: str, max_odds: float, changed: set) -> Sides:
    """L3's odds rule: a `pool` ticket on every runner priced on the Win board, less the tail."""

    def comparator(view: RaceView) -> list[Ticket] | None:
        starters = ctx.runners(view)
        runners = sorted(h for h in _win_prices(view) if h in starters)
        return [Ticket(pool, str(h), ctx.unit) for h in runners] or None

    def rule(view: RaceView) -> list[Ticket] | None:
        tickets = comparator(view) or []
        kept = drop_tail(view, tickets, max_odds=max_odds)
        if len(kept) < len(tickets):
            changed.add(view.race_id)
        return kept or None

    return rule, comparator


def _sets_of(pool: str, n: int) -> int:
    k = LEGS[pool]
    return math.perm(n, k) if pool in ORDERED else math.comb(n, k)


def _rank_paired(
    ctx: _Ctx, pool: str, cut: int, races: Sequence[str], view, changed: set
) -> Paired:
    """L3's rank rule on every combination of the runners, flat: a losing set stakes the same
    and pays nothing on both sides, so per race the returns come from the paying rows and the
    number of sets each side bets (C(n, k), or n!/(n-k)! in an ordered pool). A set holding a
    runner the results mark scratched is refunded on each side that bets it."""
    returns, reasons = {}, {}
    for race in races:
        seen = view(race)
        ranks, starters = win_ranks(seen), ctx.runners(seen)
        out = ctx.scratched.get(race, frozenset()) & starters
        rows = [(c, d, r) for (p, c), (d, r) in ctx.paying.get(race, {}).items() if p == pool]
        kept = {h for h in starters if ranks.get(h, cut) < cut}
        n, m = len(starters), len(kept)
        no_dividend = any(not r and not math.isfinite(d) for _, d, r in rows)
        why = avoidance._first(
            (not rows, f"no {pool} dividend row"),
            (not ranks, "no Win price on the board"),
            (any(h not in ranks for h in starters), "a starter has no Win price on the board"),
            (_sets_of(pool, n) == 0, f"fewer than {LEGS[pool]} starters"),
            (_sets_of(pool, m) == 0, "the rule bets no set"),
            (no_dividend, "a paying row has no dividend"),
        )
        if why:
            reasons[race] = why
            continue
        paid = [(c, d / ctx.unit) for c, d, r in rows if not r and set(_horses(c)) <= starters]
        rule = sum(p for c, p in paid if worst_member_rank(c, ranks) < cut)
        comparator = sum(p for _, p in paid)
        rule += _sets_of(pool, m) - _sets_of(pool, len(kept - out))  # refunded sets: stake back
        comparator += _sets_of(pool, n) - _sets_of(pool, n - len(out))
        if m < n:
            changed.add(race)
        returns[race] = (rule / _sets_of(pool, m), comparator / _sets_of(pool, n))
    return list(races), returns, reasons


def _skip_paired(
    ctx: _Ctx,
    races: Sequence[str],
    view,
    tickets_of: Callable[[RaceView], list[Ticket] | None],
    skips: Callable[[RaceView], bool | None],
    changed: set,
) -> Paired:
    """L7: the comparator's tickets in every race; the rule's are the same except in a race it
    skips, which returns its stake (1.0). A ticket holding a runner scratched after the choice is
    refunded."""
    returns, reasons = {}, {}
    for race in races:
        seen = view(race)
        tickets, skip = tickets_of(seen), skips(seen)
        if not tickets:
            reasons[race] = "no favourite on the board (unpriced, or a tie with drop_race)"
            continue
        if skip is None:
            reasons[race] = "no normalised entropy (under two priced runners)"
            continue
        stake, payout, why = ctx.settle(race, tickets)
        if why:
            reasons[race] = why
            continue
        back = payout / stake
        if skip:
            changed.add(race)
        returns[race] = (1.0 if skip else back, back)
    return list(races), returns, reasons


def _favourite_tickets(ctx: _Ctx, run: _Run) -> Callable[[RaceView], list[Ticket] | None]:
    ties = ctx.l7["favourite_ties"]

    def tickets(view: RaceView) -> list[Ticket] | None:
        k = 3 if run.kind == "box" else 1
        top = _favourites(view, ctx.runners(view), k, ties)
        if top is None:
            return None
        if run.kind != "box":
            return [Ticket(run.pool, str(top[0]), ctx.unit)]
        legs = LEGS[run.pool]
        if run.pool in ORDERED:
            sets = itertools.permutations(top, legs)
        else:
            sets = itertools.combinations(sorted(top), legs)
        return [Ticket(run.pool, _joined(s), ctx.unit) for s in sets]

    return tickets


def _skips(ctx: _Ctx, run: _Run) -> Callable[[RaceView], bool | None]:
    if run.kind != "flat":
        return lambda view: view.race_id in ctx.handicaps
    cuts = ctx.l7["entropy_cut_points"]

    def flattest(view: RaceView) -> bool | None:
        entropy = _entropy(view)
        return None if math.isnan(entropy) else entropy_quintile(entropy, cuts) == FLATTEST

    return flattest


def _cell(dividend: float) -> str:
    """A dividend's cell of the 100-vs-110 table: '100', '110' or 'above_110'."""
    if dividend <= WITHHELD_DIVIDEND:
        return CELLS[0]
    return CELLS[1] if dividend <= PLUS10_DIVIDEND else CELLS[2]


def _placed_horses(paying: Mapping) -> list[int]:
    return sorted(int(c) for (p, c), (_, refund) in paying.items() if p == "PLACE" and not refund)


UNRESOLVED_PREMIUM = (
    "a premium PLACE pool whose premium settlement is unresolved (JRA may withhold it, "
    "takeout.boost_may_be_withheld, and the store holds no settlement signal; #481)"
)


def _reconstruct(ctx: _Ctx, race: str, pool: PlacePool | None) -> tuple[dict | None, str]:
    """({each placed horse: its Place dividend per 100 yen as rebuilt from the race's settled
    PLACE pool (PLACE_RULE: the base, then Plus-10 on the base total)}, ''), or (None, why there
    is none). A race whose PLACE pool the boost calendar gives a premium has none: whether the
    premium was paid is unresolved, and it is never inferred from the realised dividends."""
    placed = _placed_horses(ctx.paying.get(race, {}))
    if pool is None:
        return None, "no PLACE votes for the race"
    if pool.tickets is None or pool.tickets <= 0:
        return None, "no PLACE pool total"
    if any(pool.votes.get(h, 0) <= 0 for h in placed):
        return None, "a placed horse has no PLACE votes"
    if sum(pool.votes[h] for h in placed) > pool.tickets:
        return None, "the placed horses' PLACE votes exceed the pool's"
    try:
        rate = race_takeout(race, "PLACE", race_class_raw=ctx.class_of.get(race))
    except (LookupError, ValueError) as exc:
        return None, f"no Place payout rate ({type(exc).__name__})"
    if rate.boost_may_be_withheld:  # A: no premium engine here (#481), no inference
        return None, UNRESOLVED_PREMIUM
    sales = float(pool.tickets * ctx.unit)
    paid, total, top_ups = place_payout(pool.votes, placed, sales, rate.payout_rate)
    floor = predicted_floor_dividend(total, top_ups, sales)  # on the base payout
    return {h: d if d > WITHHELD_DIVIDEND else floor for h, d in paid.items()}, ""


WITHIN = "within tolerance"


def _diagnostics(
    ctx: _Ctx, run: _Run, rebuilt: Mapping[str, tuple[dict | None, str]]
) -> tuple[list, dict, dict[str, str]]:
    """The reconstruction diagnostic: every placed horse of the rebuilt races, its rebuilt Place
    dividend against the realised one. (metrics, its table row, each rebuilt race -> its label:
    'discrepant: ...' when a placed horse misses by more than the tolerance, else WITHIN). A
    discrepancy labels its race's row; it rejects nothing."""
    tolerance = ctx.spec["rules"]["L2"]["plus10"]["validation"]["tolerance_yen_per_100"]
    misses = {
        race: [abs(d - ctx.paying[race][("PLACE", str(h))][0]) for h, d in paid.items()]
        for race, (paid, _) in rebuilt.items()
        if paid is not None
    }
    labels = {}
    for race, errors in misses.items():
        off = sum(1 for e in errors if e > tolerance + 1e-9)
        labels[race] = WITHIN
        if off:
            labels[race] = (
                f"discrepant: {off} of {len(errors)} placed horses' rebuilt Place dividends miss "
                f"the realised by more than {tolerance} yen per 100 (largest {max(errors):g})"
            )
    errors = [e for found in misses.values() for e in found]
    n, worst = len(errors), max(errors, default=0.0)
    within = sum(1 for e in errors if e <= tolerance + 1e-9)
    metrics = [
        _metric(FOLD, "validation_placed_horses_checked", n, n),
        _metric(FOLD, "validation_within_tolerance", within, n),
        _metric(FOLD, "validation_max_abs_error_yen", worst, n),
    ]
    row = {"variant": run.name, "races checked": len(misses), "placed horses checked": n}
    discrepant = sum(1 for label in labels.values() if label != WITHIN)
    row |= {"within tolerance": within, "races discrepant": discrepant}
    row |= {"largest miss (yen per 100)": worst, "tolerance (yen per 100)": tolerance}
    return metrics, {"plus10_validation": [row]}, labels


def _counts(run: _Run, floor_bound: int, rows: list[dict], prediction: str) -> tuple[list, dict]:
    """B: the check's counts from its rows' final statuses: attempted (the floor-bound placed
    favourites), available (a prediction), unavailable (none) and discrepant (available, its race
    labelled discrepant). (metrics, the private counts row)."""
    attempted = len(rows)
    available = sum(1 for r in rows if r["predicted"] is not None)
    discrepant = sum(1 for r in rows if r["predicted"] is not None and r["diagnostic"] != WITHIN)
    metrics = [
        _metric(FOLD, "predictions_attempted", attempted, attempted),
        _metric(FOLD, "predictions_available", available, attempted),
        _metric(FOLD, "predictions_unavailable", attempted - available, attempted),
        _metric(FOLD, "predictions_discrepant", discrepant, available),
    ]
    row = {"variant": run.name, "floor-bound favourites": floor_bound, "attempted": attempted}
    row |= {"available": available, "unavailable": attempted - available}
    return metrics, row | {"discrepant": discrepant, "prediction": prediction}


def _plus10(ctx: _Ctx, run: _Run, races: Sequence[str], view) -> tuple[list, dict] | None:
    """The 100-vs-110 check: each floor-bound placed favourite's realised Place dividend and, under
    jravan_pool_votes, its predicted dividend from the race's settled PLACE pool (PLACE_RULE),
    cross-tabulated predicted x realised over every available prediction. (metrics, tables); None
    when no race counts (a favourite priced on its board and, at t, with a PLACE price at t): data
    pending, not zero favourites. Every attempted row stays with its status: a prediction, or why
    there is none (the votes missing or unreadable, an unresolved premium, ...); the
    reconstruction diagnostic labels a race's row discrepant and rejects nothing. Eligible
    favourites none of which has a prediction leave it unavailable: the realised counts, no
    diagnostic and no cell. A covered selection with no eligible favourite is a scored zero, every
    cell 0."""
    ties = ctx.l7["favourite_ties"]
    inputs = ctx.spec["rules"]["L2"]["plus10"]["inputs"]
    floor_bound, rows, covered = 0, [], 0
    for race in races:
        seen = view(race)
        top = _favourites(seen, ctx.runners(seen), 1, ties)
        if top is None:
            continue
        horse = str(top[0])
        price = _price(seen.board, "PLACE", horse) if run.tag == AT_T else math.nan
        if run.tag == AT_T and math.isnan(price):
            continue  # no PLACE price for the favourite at t: the race does not count
        covered += 1
        if run.tag == AT_T and not bool(at_floor(pd.Series([price])).iloc[0]):
            continue
        dividend, refund = ctx.paying.get(race, {}).get(("PLACE", horse), (math.nan, False))
        placed = not refund and math.isfinite(dividend)
        if run.tag == FINAL and not (placed and dividend <= PLUS10_DIVIDEND):
            continue
        floor_bound += 1
        if placed:
            rows.append(
                {"variant": run.name, "race_id": race, "horse_no": int(horse)}
                | {"place low at t": price, "realised": dividend}
            )
    if not covered:
        return None
    realised = Counter(_cell(r["realised"]) for r in rows)
    metrics = [_metric(FOLD, "races_counted", covered, len(races))]
    metrics.append(_metric(FOLD, "floor_bound_favourites", floor_bound, floor_bound))
    metrics.append(_metric(FOLD, "placed", len(rows), floor_bound))
    public = []
    for key in CELLS:
        metrics.append(_metric(FOLD, f"realised_{key}", realised[key], len(rows)))
        public.append(
            {"variant": run.name, "floor-bound favourites": floor_bound, "placed": len(rows)}
            | {"realised": key, "favourites": realised[key]}
        )
    tables: dict = {"plus10": public, "plus10_races": rows}
    if inputs != JRAVAN_VOTES:
        for row in rows:
            row |= {"predicted": None, "why": f"data pending ({inputs})", "diagnostic": ""}
        found, counts = _counts(run, floor_bound, rows, "data pending")
        return metrics + found, tables | {"plus10_counts": [counts]}

    if ctx.place_votes is None:
        raise ValueError("plus10_inputs is jravan_pool_votes and no pool_votes reader is given")
    wanted = sorted({r["race_id"] for r in rows})
    try:
        pools, unread = ctx.place_votes(wanted) if wanted else {}, ""
    except Exception as exc:  # an optional label: the realised counts stand without the votes
        pools, unread = {}, f"pool_votes unreadable ({type(exc).__name__})"
    rebuilt = {
        race: (None, unread) if unread else _reconstruct(ctx, race, pools.get(race))
        for race in wanted
    }
    checked, validation, labels = _diagnostics(ctx, run, rebuilt)
    for row in rows:
        paid, why = rebuilt[row["race_id"]]
        row |= {"predicted": None if paid is None else paid[row["horse_no"]], "why": why}
        row["diagnostic"] = labels.get(row["race_id"], "")
    done = [r for r in rows if r["predicted"] is not None]
    reasons = Counter(r["why"] for r in rows if r["predicted"] is None)
    tables["plus10_unpredicted"] = [
        {"variant": run.name, "why": w, "favourites": n} for w, n in reasons.items()
    ]
    if rows and not done:  # eligible favourites, none rebuilt: the prediction is unavailable
        found, counts = _counts(run, floor_bound, rows, "unavailable")
        return metrics + found, tables | {"plus10_counts": [counts]}
    found, counts = _counts(run, floor_bound, rows, "made" if done else "no eligible favourite")
    metrics += found
    tables["plus10_counts"] = [counts]
    if done:
        metrics += checked
        tables |= validation
    cells = Counter((_cell(r["predicted"]), _cell(r["realised"])) for r in done)
    cross = []
    for p, r in itertools.product(CELLS, CELLS):
        metrics.append(_metric(FOLD, f"predicted_{p}_realised_{r}", cells[(p, r)], len(done)))
        cross.append({"variant": run.name, "predicted": p, "realised": r} | {"n": cells[(p, r)]})
    return metrics, tables | {"plus10_cells": cross}


SIDES = {"rule_mean_return": "rule", "comparator_mean_return": "comparator"}
DESCRIPTIVE_ROWS = "descriptive: every development race"
FORWARD_ROWS = "walk-forward: the test windows"


def _evaluated(row: dict, evaluation: str) -> dict:
    """The comparisons row with its evaluation named next to its variant."""
    head = {k: row[k] for k in ("rule", "variant") if k in row}
    return head | {"evaluation": evaluation} | row


def _forward(
    study: Study,
    ctx: _Ctx,
    variant: Variant,
    comparator: str,
    comparison: RuleComparison | None,
    changed: set[str],
    error: str | None = None,
    status: str | None = None,
) -> None:
    """E: a paired comparison's walk-forward record, beside its descriptive one: the same per-race
    returns over the paired races of the walk-forward test windows alone (`ctx.fold_of`). Pooled
    (WALK_FORWARD_FOLD): the paired difference and each side's mean, each with D7's bounds, n,
    the races dropped and changed; per window (its label): the paired difference and n. Without a
    paired race in a window it fails with that reason; when the descriptive record has none
    (`error`), with the descriptive record's error and table `status`."""
    twin = replace(
        variant,
        name=_forward_name(variant.name),
        fold_scheme=ctx.fold_scheme,
        components=variant.components | {"evaluation": WALK_FORWARD},
    )
    row = {"rule": variant.hypothesis_id, "variant": twin.name, "evaluation": FORWARD_ROWS}
    row |= {"selection": variant.components["selection"], "comparator": comparator}
    races = [] if comparison is None else [r for r in comparison.diff.index if r in ctx.fold_of]
    if error is None and not races:
        error = "no paired race in a walk-forward test window"
    if error:
        study.variants.append(replace(twin, error=error))
        study.tables["comparisons"].append(row | {"status": status or error})
        return
    pick = pd.Index(races, name="race_id", dtype=object)
    series = {
        "paired_return_difference": comparison.diff.loc[pick],
        "rule_mean_return": comparison.rule_return.loc[pick],
        "comparator_mean_return": comparison.comparator_return.loc[pick],
    }
    bounds = {name: paired_bootstrap(s, ctx.clusters, **ctx.boot()) for name, s in series.items()}
    n, iv = len(races), bounds["paired_return_difference"]
    dropped = sum(1 for race in comparison.dropped if race in ctx.fold_of)
    k = len(changed & set(races))
    metrics = [
        _metric(WALK_FORWARD_FOLD, name, b.mean, n, b.lo, b.hi) for name, b in bounds.items()
    ]
    metrics += [
        _metric(WALK_FORWARD_FOLD, "dropped_races", dropped, n + dropped),
        _metric(WALK_FORWARD_FOLD, "meetings", iv.n_clusters, n),
        _metric(WALK_FORWARD_FOLD, "races_changed", k, n),
    ]
    diff = series["paired_return_difference"]
    windows = diff.groupby(diff.index.map(ctx.fold_of), sort=True)
    for label, values in windows:
        metrics.append(_metric(label, "paired_return_difference", values.mean(), len(values)))
    study.variants.append(replace(twin, metrics=tuple(metrics)))
    row |= {"races": n, "dropped": dropped, "meetings": iv.n_clusters}
    row |= {"rule return": bounds["rule_mean_return"].mean}
    row |= {"comparator return": bounds["comparator_mean_return"].mean}
    row |= {"paired difference": iv.mean, "lower": iv.lo, "upper": iv.hi, "status": "scored"}
    for name, side in SIDES.items():
        row |= {f"{side} lower": bounds[name].lo, f"{side} upper": bounds[name].hi}
    row |= {"races changed": k, "variants searched": ctx.spec["family_size"]}
    study.tables["comparisons"].append(row | {"windows": windows.ngroups})


def _scored(study: Study, ctx: _Ctx, run: _Run, variant: Variant, compute, pending: str) -> None:
    """#81's `_score`, with each side's own return bounded (the same meeting-clustered bootstrap
    as the difference: two ranked rows) and the races where the rule drops a ticket or skips the
    race; then its walk-forward record (`_forward`)."""
    changed: set[str] = set()
    found: list[RuleComparison] = []

    def counted():
        found.append(compute(changed))
        return found[-1]

    comparator = {
        "floor": "the same tickets, those at the floor included",
        "tail_odds": "the same tickets, the tail included",
        "tail_rank": "every combination, the tail included",
    }.get(run.kind, "the same favourite tickets in all races")
    _score(study, variant, comparator, counted, pending)
    done, rows = study.variants[-1], study.tables["comparisons"]
    rows[-1] = _evaluated(rows[-1], DESCRIPTIVE_ROWS)
    if not found or done.error:
        _forward(study, ctx, variant, comparator, None, changed, done.error, rows[-1]["status"])
        return
    comparison = found[0]
    returns = {"rule_mean_return": comparison.rule_return}
    returns["comparator_mean_return"] = comparison.comparator_return
    bounds = {
        name: paired_bootstrap(series, ctx.clusters, **ctx.boot())
        for name, series in returns.items()
    }
    metrics = [
        _metric(m.fold, m.metric, m.value, m.n, bounds[m.metric].lo, bounds[m.metric].hi)
        if m.metric in bounds
        else m
        for m in done.metrics
    ]
    n = comparison.n_races
    k = len(changed & set(comparison.diff.index))
    metrics.append(_metric(FOLD, "races_changed", k, n))
    study.variants[-1] = replace(done, metrics=tuple(metrics))
    row = study.tables["comparisons"][-1]
    for name, side in SIDES.items():
        row |= {f"{side} lower": bounds[name].lo, f"{side} upper": bounds[name].hi}
    row |= {"races changed": k, "variants searched": ctx.spec["family_size"]}
    _forward(study, ctx, variant, comparator, comparison, changed)


def _score_run(study: Study, ctx: _Ctx, run: _Run) -> None:
    label, races, view, pending = ctx.selection(run)
    params: dict = {"pool": run.pool, "stake_yen": ctx.unit}
    params["variants_searched"] = ctx.spec["family_size"]  # the family size (D17)
    if run.cut is not None:
        params["cut"] = run.cut
    if run.kind in ("flat", "handicap", "box", "plus10"):
        params["favourite_ties"] = ctx.l7["favourite_ties"]
    if run.kind == "flat":
        params["entropy_cut_points"] = ctx.l7["entropy_cut_points"]
    components = {"rule": run.kind, "pool": run.pool, "selection": label}
    variant = Variant(run.rule, run.name, SCHEME, components, params)

    if run.kind == "plus10":
        try:
            found = _plus10(ctx, run, races, view) if races else None
        except Exception as exc:  # every failure is a registry record, never dropped
            study.variants.append(replace(variant, error=f"{type(exc).__name__}: {exc}"))
            return
        if found is None:
            study.variants.append(replace(variant, error=pending))
            return
        metrics, tables = found
        for key, rows in tables.items():
            study.tables.setdefault(key, []).extend(rows)
        study.variants.append(replace(variant, metrics=tuple(metrics)))
        return
    if run.kind in ("floor", "tail_odds"):

        def compute(changed):
            if run.kind == "floor":  # a no-bet race returns its stake (L2's stake_back)
                sides, no_bet = _floor_sides(ctx, run.pool, changed), NO_BET_RETURN
            else:
                sides, no_bet = _tail_sides(ctx, run.pool, run.cut, changed), None
            return ctx.paired(_tickets_paired(ctx, races, view, sides, no_bet), label)

    elif run.kind == "tail_rank":

        def compute(changed):
            return ctx.paired(_rank_paired(ctx, run.pool, run.cut, races, view, changed), label)

    else:

        def compute(changed):
            tickets, skips = _favourite_tickets(ctx, run), _skips(ctx, run)
            return ctx.paired(_skip_paired(ctx, races, view, tickets, skips, changed), label)

    _scored(study, ctx, run, variant, compute, pending)


def compute_study(data: StudyData, spec: dict, place_votes: VotesReader | None = None) -> Study:
    """Every variant of the frozen `spec` on `data`, in `_plan` order. No I/O beyond the
    board-at-t reads of `data.view_at_t` and, for the 100-vs-110 check under jravan_pool_votes,
    the settled PLACE pools of `place_votes`."""
    ctx = _Ctx(data, spec, place_votes)
    study = Study(
        counts={
            "races_read": len(data.races),
            "run_races": len(ctx.race_ids),
            "handicap_races": len(ctx.handicaps),
            "runners_read": len(data.runners),
            "starters": len(ctx.starters),
            "dividend_rows": len(ctx.dividends),
        }
        | {f"races_with_a_logged_{p}_board": len(ctx.logged(p)) for p in BOARD_POOLS}
        | {
            "walk_forward_test_windows": len(set(ctx.fold_of.values())),
            "races_in_walk_forward_test_windows": len(ctx.fold_of),
        }
    )
    for run in _plan(spec["rules"]):
        _score_run(study, ctx, run)
    return study


# --------------------------------------------------------------------------- report and runner

NOTE = (
    "Per paired race: each side's payout / stake at final dividends (L7: a race the rule skips"
    " returns its stake; L2: so does a race whose every ticket is at the floor; a ticket holding a"
    " runner scratched after the choice is refunded). 'rule return' and 'comparator return' are"
    " each side's mean over the paired races and 'paired difference' the mean of rule minus"
    " comparator; each lower and upper is the (1 - level) and level quantile of the bootstrap"
    " over meetings (D7, one-sided bounds). Positive is the rule returning more than its"
    " comparator. 'races changed' counts the paired races where the rule drops a ticket or skips"
    " the race. 'final (look-ahead)' rows"
    " choose by the final price and are look-ahead; 'board at t' rows choose by odds_timeseries at"
    " t. 'evaluation': a descriptive row scores every development race once (the first training"
    " span included: a historical description); its walk-forward row scores the same races'"
    " returns in the test windows of config/folds.yaml's scheme alone ('windows' the test windows"
    " with a paired race; per window in the registry record). Every row is ranked below; nothing"
    " here is adopted."
)
SECTIONS = (
    ("L2", "L2: never bet at the floor (E14)"),
    ("L3", "L3: avoid the tail (E12)"),
    ("L7", "L7: flat-market and handicap favourites (E3, E7)"),
)
PER = 100  # the ranked table's returns are per 100 yen staked


def _ranked(comparisons: list[dict]) -> list[dict]:
    """Each scored comparison as two options (the rule and its comparator), ranked within each
    evaluation (walk-forward, descriptive) and selection by mean return per 100 yen staked,
    highest first."""
    options = []
    for row in comparisons:
        if row.get("status") != "scored":
            continue
        for side, what in (("rule", "the rule's tickets"), ("comparator", row["comparator"])):
            options.append(
                {"evaluation": row["evaluation"], "selection": row["selection"]}
                | {"rule": row["rule"], "variant": row["variant"]}
                | {"option": side, "bets": what, "races": row["races"]}
                | {f"return per {PER} yen": PER * row[f"{side} return"]}
                | {"lower": PER * row[f"{side} lower"], "upper": PER * row[f"{side} upper"]}
                | {"variants searched": row["variants searched"]}
            )
    first = {FORWARD_ROWS: 0, DESCRIPTIVE_ROWS: 1}  # the walk-forward rows lead

    def group(option: dict) -> tuple:
        return first.get(option["evaluation"], 2), option["evaluation"], option["selection"]

    options.sort(key=lambda o: (*group(o), -o[f"return per {PER} yen"], o["variant"]))
    ranked, seen = [], Counter()
    for option in options:
        seen[group(option)] += 1
        ranked.append({"rank": seen[group(option)]} | option)
    return ranked


def _withheld_runs(spec: dict) -> set[str]:
    """The variants built on the jravan votes (D21): the 100-vs-110 check under
    jravan_pool_votes."""
    if spec["rules"]["L2"]["plus10"]["inputs"] != JRAVAN_VOTES:
        return set()
    return {run.name for run in _plan(spec["rules"]) if run.kind == "plus10"}


def render_report(study: Study, spec: dict, about: dict, private: str | None = None) -> str:
    """The main report, on the jra store alone: aggregate tables only (race rows go to the
    companion CSVs). Under jravan_pool_votes the 100-vs-110 prediction is withheld (D21) and the
    report names `private`, the report that holds it, instead."""
    withheld = _withheld_runs(spec)
    lines = [
        f"# Floor, tail and favourite avoidance: Family L rules L2, L3, L7 ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- variants searched (the family size): {spec['family_size']}",
        "- development rows only: holdout days are never read and the holdout is not opened",
        "- every option is a ranked row (owner reframe 2026-11-27): there is no pass/fail and"
        " nothing here is adopted",
        "",
        "## Frozen rules",
        "",
        "```json",
        json.dumps(spec["rules"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Owner decisions used",
        "",
        *(f"- {key}: {value}" for key, value in spec["decisions"].items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    comparisons = study.tables.get("comparisons", [])
    for rule, title in SECTIONS:
        rows = [row for row in comparisons if row["rule"] == rule]
        lines += ["", f"## {title}", "", NOTE, ""]
        lines.append(_markdown_table(rows) if rows else "_None: see the variants below._")
    ranked = _ranked(comparisons)
    lines += [
        "",
        f"## Ranked options (mean return per {PER} yen staked, within each evaluation and"
        " selection)",
        "",
        "Each scored comparison is two rows: the rule's tickets and its comparator's, over the"
        " same paired races. Rows of one selection overlap (one comparator serves several cut"
        " points), so the rank orders the options and is not a test. The walk-forward rows come"
        " first; the descriptive rows (every development race) are the historical description.",
        "",
        _markdown_table(ranked) if ranked else "_None: no comparison was scored._",
        "",
        "## L2: the 100-vs-110 check (Plus-10, 04 §1.3)",
        "",
        "Floor-bound favourites that placed, by realised Place dividend (yen per 100), from the jra"
        " dividends.",
        "",
    ]
    plus10 = study.tables.get("plus10")
    lines.append(_markdown_table(plus10) if plus10 else "_None: no race counted for the check._")
    if withheld:
        lines += [
            "",
            "The predicted dividend of each (100 or 110 under Plus-10), against the realised one,"
            " is built on the JRA-VAN H1 Place votes (jravan pool_votes, #368), and so are the"
            " reconstruction diagnostic (every placed horse of the check's races, its rebuilt"
            " Place dividend against the realised one, within the frozen tolerance) and the counts"
            " of predictions available, unavailable and labelled by the diagnostic. They and the"
            f" records' status are withheld (D21): they are in the private report `{private}`"
            " under the data directory, never in the repository.",
        ]
    else:
        lines += ["", "The prediction is pending: plus10_inputs is data_pending."]
    reasons = study.tables.get("dropped_reasons")
    lines += ["", "## Dropped races by reason", ""]
    lines.append(_markdown_table(reasons) if reasons else "_None._")
    variants = []
    for v in study.variants:
        row = {"rule": v.hypothesis_id, "variant": v.name}
        row["status"] = "failed" if v.error else "scored"
        pending = bool(v.error) and v.error.startswith("data pending")
        if v.name in withheld and not pending:  # its status could tell the check's result (D21)
            row |= {"status": "withheld (D21)", "why": f"see `{private}`"}
        elif v.error:
            row["why"] = v.error
        variants.append(row)
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def render_private(study: Study, spec: dict, about: dict) -> str:
    """The 100-vs-110 table built on the JRA-VAN votes: withheld (D21), written outside the
    repository only."""
    withheld = _withheld_runs(spec)
    status = [
        {"variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"why": v.error} if v.error else {})
        for v in study.variants
        if v.name in withheld
    ]
    lines = [
        f"# The 100-vs-110 check ({COUNTRY}, #{TICKET}): JRA-VAN-derived, withheld (D21)",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- built on the JRA-VAN H1 Place votes (jravan pool_votes, #368): never in the repository",
        "- development races only, read for the settled race (a label), never at t",
        "",
        "## The prediction",
        "",
        PLACE_RULE + ".",
        "",
        "```json",
        json.dumps(spec["place_votes"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Floor-bound placed favourites",
        "",
        "From the rows' final statuses (the private CSV): attempted, the floor-bound placed"
        " favourites; available, those with a prediction; unavailable, those without one (each"
        " with its reason, below); discrepant, the available ones whose race the reconstruction"
        " diagnostic labels discrepant.",
        "",
        _markdown_table(study.tables["plus10_counts"])
        if study.tables.get("plus10_counts")
        else "_None._",
        "",
        "## The reconstruction diagnostic",
        "",
        "Every placed horse of the check's rebuilt races: its Place dividend rebuilt from the votes"
        " (the prediction's rule) against the realised one. A race with a miss above the frozen"
        " tolerance has its row labelled discrepant in the CSV; its prediction stays and is counted"
        " in the table below (a diagnostic, never a rejection).",
        "",
        _markdown_table(study.tables["plus10_validation"])
        if study.tables.get("plus10_validation")
        else "_None._",
        "",
        "## Predicted against realised Place dividend (yen per 100)",
        "",
        _markdown_table(study.tables["plus10_cells"])
        if study.tables.get("plus10_cells")
        else "_None._",
        "",
        "## Not predicted, by reason",
        "",
        _markdown_table(study.tables["plus10_unpredicted"])
        if study.tables.get("plus10_unpredicted")
        else "_None._",
        "",
        "## Variants",
        "",
        _markdown_table(status) if status else "_None._",
        "",
    ]
    return "\n".join(lines)


def _csv(study: Study, key: str, columns: list[str]) -> str:
    return pd.DataFrame(study.tables.get(key, []), columns=columns).to_csv(index=False)


POST_TIMES_BUILD = "board.snapshot.post_times_for (the <VENUE>.races.json sidecar, else the card)"


class _ScheduledPosts:
    """The scheduled post (aware UTC) of a race whose captures carry no post_utc, from
    `board.snapshot.post_times_for` (read once per meeting), or None. A race on a holdout day is
    a ValueError (D14): its card is not read. `timed` and `untimed` are the races asked."""

    def __init__(self) -> None:
        self.meetings: dict[tuple[date, str], dict[int, datetime]] = {}
        self.posts: dict[str, datetime | None] = {}

    @property
    def timed(self) -> set[str]:
        return {race for race, post in self.posts.items() if post is not None}

    @property
    def untimed(self) -> set[str]:
        return {race for race, post in self.posts.items() if post is None}

    def __call__(self, race_id: str) -> datetime | None:
        _, day, venue, race_no = race_id.split(":")
        when = date.fromisoformat(day)
        if is_holdout_day(COUNTRY, when):
            raise ValueError(f"{race_id} is on a holdout day (D14): its post time is not read")
        if (when, venue) not in self.meetings:
            self.meetings[(when, venue)] = snapshot.post_times_for(when, venue)
        post = self.meetings[(when, venue)].get(int(race_no))
        self.posts[race_id] = post
        return post

    def build(self) -> dict[str, str]:
        """The at-t records' post-time input as one data build, '<identity>@<digest>': the
        identity names the source, the digest is of the race -> scheduled post mapping read (None
        for a race the schedule lacks), so a changed or backfilled post time refreshes the same
        build. The coverage counts go in the report, not here."""
        mapping = {
            race: None if post is None else post.astimezone(timezone.utc).isoformat()
            for race, post in sorted(self.posts.items())
        }
        text = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return {f"{SOURCE}_post_times": f"{POST_TIMES_BUILD}@{digest}"}


def _votes_build(path: Path) -> dict[str, str]:
    """The votes' data build, '<path>@<digest>'; a file that is missing or cannot be read for its
    digest is named so, and the run goes on (the prediction is then unavailable)."""
    where = f"processed/{VOTES_SOURCE}/{jravan_votes.FILENAME}"
    if not path.is_file():
        return {f"{VOTES_SOURCE}_pool_votes": f"{where} (missing)"}
    try:
        value = f"{where}@{_digest([path])}"
    except OSError as exc:  # optional input: unreadable is recorded, never fatal
        value = f"{where} (unreadable: {type(exc).__name__})"
    return {f"{VOTES_SOURCE}_pool_votes": value}


def run_study(out_dir: Path | None = None) -> dict:
    """Run the pre-registered study on Japan's development data and write its reports.

    Raises `MissingOwnerDecision` before anything is read when config/avoidance_price.yaml lacks
    an input (a ValueError for an unknown key), `paths.DataDirInsideRepo` for an `out_dir` inside
    the repository, and `FamilyAlreadyFrozen` when the family is frozen with another spec."""
    owner = owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    minutes = int(load_country_yaml(COUNTRY, "decision")["primary_min"])
    spec = build_spec(owner, bootstrap_settings(), minutes, load_country_scheme(COUNTRY))
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any race is read or settled

    posts = _ScheduledPosts()
    data = avoidance.read_development(
        minutes, pools=PRICE_POOLS, race_columns=RACE_COLUMNS, post_time=posts
    )
    builds = avoidance.store_builds(data.board_files)
    withheld = _withheld_runs(spec)
    reader, votes_build = None, {}
    if withheld:
        path = votes_path()
        reader, votes_build = functools.partial(read_place_votes, path=path), _votes_build(path)
    study = compute_study(data, spec, reader)
    del data
    study.counts |= {
        "races_timed_by_the_scheduled_post": len(posts.timed),
        "races_without_any_post_time": len(posts.untimed),
    }
    at_t_builds = {**builds, **posts.build()}

    def record(variant: Variant) -> RunRecord:
        at_t = variant.components["selection"] != LOOK_AHEAD
        made = avoidance._record(
            variant, git_sha, at_t_builds if at_t else builds, family_id=FAMILY_ID, ticket=TICKET
        )
        if variant.name not in withheld:
            return made
        return replace(
            made, sources=(SOURCE, VOTES_SOURCE), data_builds={**made.data_builds, **votes_build}
        )

    existing = registry.records(FAMILY_ID)
    records = [record(v) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = out_dir / f"floor-tail-favourite-{COUNTRY}-{spec_hash[:12]}"
    private = stem.with_name(f"{stem.name}-plus10-{VOTES_SOURCE}.md")
    text = render_report(study, spec, about, private.name if withheld else None)
    written = {"report_path": str(write_report(stem.with_suffix(".md"), text, sources=(SOURCE,)))}
    outputs = [("dropped_races", "dropped", ["variant", "race_id", "reason"], (SOURCE,))]
    if withheld:  # D21: the votes' figures go to private files only, never inside the repository
        both = (SOURCE, VOTES_SOURCE)
        about_private = about | {"store": ", ".join({**builds, **votes_build}.values())}
        text = render_private(study, spec, about_private)
        written["private_report_path"] = str(write_report(private, text, sources=both))
        outputs.append(("plus10_races", f"plus10-{VOTES_SOURCE}", PLUS10_COLUMNS, both))
    else:
        outputs.append(("plus10_races", "plus10", PLUS10_COLUMNS, (SOURCE,)))
    for key, suffix, columns, sources in outputs:
        path = stem.with_name(f"{stem.name}-{suffix}.csv")
        text = _csv(study, key, columns)
        written[f"{suffix.replace('-', '_')}_path"] = str(write_report(path, text, sources=sources))
    return {  # public (printed by main): a withheld variant's status is the private report's (D21)
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        **written,
        "registry_path": str(registry.path),
        "counts": study.counts,
        "records_appended": len(appended),
        "records_already_present": len(records) - len(appended),
        "failed_or_pending_variants": [
            v.name for v in study.variants if v.error and v.name not in withheld
        ],
        "withheld_variants": sorted(withheld),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.avoidance_price"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
