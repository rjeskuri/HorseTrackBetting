"""C2 interactions on #92's Plackett-Luce ability model: jockey, trainer and track (ticket #93,
epic #73; 08 §3.3, §4 E35, E36, §8 B5; 05 §4.1-4.3).

Contract pinned here (`horsetrack.countries.japan.models.interactions`):
- `cell_sizes(frame, *, block) -> pd.DataFrame`: one row per input row with race_id, horse_id and
  n_prior, the number of rows of frame in the row's cell (equal values in every column of
  `block`) whose race_utc is strictly earlier than the row's. The row's own race never counts,
  nor another race at the same race_utc (a tie: two venues at one post time), nor a later race;
  an earlier race of the same day does. Rows are matched on (race_id, horse_id); their order and
  any other columns are not pinned.
- `n_half(tau, info_per_run) -> float`: 1 / (info_per_run * tau^2), the runs a cell needs before
  its estimate is half data, half prior.
- `InteractionBlock(name, cols, kind)`, kind one of:
  - 'variance_component': cols are the cell's keys, one N(0, tau^2) effect per cell, tau fitted
    (horse x jockey = ('horse_id', 'jockey_id'), E35's negative control; trainer x track =
    ('trainer_id', 'venue'), E36);
  - 'structured': cols = (horse column, distance column in metres, surface column); the horse's
    a_h + b_h z(d) + c_h z(d)^2 + e_h 1{surface == 'dirt'}, z a standardisation of distance;
  - 'low_rank': cols = (row entity, column entity); <U_row, W_col> at a rank the coder chooses
    (horse x track <U_h, W_track> = ('horse_id', 'venue'); E36's one-factor east/west term =
    ('trainer_id', 'venue')).
- `fit_with_blocks(frame, base, blocks) -> InteractionFit`: `base` is an unfitted `AbilityModel`
  whose k, discounts, features and entity columns are the model the blocks are added to; the fit
  reads frame's rows only (its training window); other columns of frame are ignored.
  `.tau(name)` is the block's tau as an interval with `.mean` (the point estimate), `.lo` and
  `.hi`, finite with 0 <= lo <= mean <= hi (`core.eval.bootstrap.Interval` has these fields);
  a name the fit has no block for raises KeyError. `.predict(frame)` has one row per input row
  with race_id, horse_id, u and p_ability (the softmax of u within the race), and never reads
  finish_pos.
- `drop_aliased(blocks, frame) -> list[InteractionBlock]`: the blocks frame can identify, in
  their order, the input list unchanged. A block is aliased when its cells are the levels of one
  of its columns: horse x trainer where every horse has one trainer (08 §3.3: aliased with the
  horse). The decision reads frame, never the block's name or column names.
- `transfer_step(frame, *, stable_runs) -> pd.Series`: one bool per row on frame's index, True on
  a run after a stable change of the horse's trainer (frame: race_id, race_utc, horse_id,
  trainer_id). The rule pinned here: a horse's runs are taken in race_utc order, its debut
  trainer is its established trainer, and a change of trainer becomes stable at the run where
  the new trainer has had the horse for `stable_runs` consecutive runs, that run included; the
  step is on from that run while the horse stays with that trainer. A spell with another trainer
  shorter than `stable_runs` (a one-off, then back) never turns it on. A run's value reads the
  horse's runs up to and including that run, never a later one: a run's own trainer is on its
  card, the next run's is not.

- `studies.interactions.member_pool_probs(rows, p_win, ordering, pool) -> pd.DataFrame` (#574):
  a member's per-starter Win probabilities through C5's ordering (`models.exotic.Ordering`) to
  one pool's combination probabilities. `rows` has one row per starter (race_id, horse_no,
  field_size: the runners when sales opened, `races.field_size`), `p_win` one probability per
  row on rows' index (summing to 1 within each race), `pool` one of `models.exotic`'s WIN, PLACE,
  WIDE. The frame has race_id, combination (the store's text: "7" for WIN and PLACE, "3,7"
  ascending for WIDE) and p, one row per combination of each race's starters. WIN gives p_win;
  PLACE's p per race sums to the places paid by field_size (jra.yaml: 2 for 5-7, 3 for 8 and
  more, so a late scratch from 8 to 7 still pays 3), and a race whose PLACE was not sold (4 or
  fewer) has no PLACE rows; WIDE's pairs sum to 3 (any two of the first three, sold from 4).
  Each p equals `ordering.joint_probs` of the race and lies in [0, 1].

Not pinned: the CI's method and level; what tau means for a structured or low-rank block
(only that it is a sane interval); the low-rank rank; the form of z(d); unseen cells; missing ids;
the trainer debut prior (no signature names it); a second change after a stable one and a return
to an earlier trainer (`transfer_step`); out-of-fold ΔLL at the final price through #87's
meta-model, the registry records (including the record of a dropped block with its cell count),
the leaderboard rows and the report, which read the local stores. Expected values are planted or
derived by hand and by brute force, never from the code under test. Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models import exotic
from horsetrack.countries.japan.models.ability import AbilityModel
from horsetrack.countries.japan.models.interactions import (
    InteractionBlock,
    cell_sizes,
    drop_aliased,
    fit_with_blocks,
    n_half,
    transfer_step,
)
from horsetrack.countries.japan.studies import interactions as study

# isort: split
from ability_synth import softmax_within
from interactions_synth import (
    EW_ROLES,
    FIELD_HORSES,
    MIX_TRAIN_RACES,
    OPTIMA,
    SPARSE_ROLES,
    SPARSE_VENUES,
    TT_TRAIN_RACES,
    VENUES,
    after,
    alias_frame,
    aptitude_frame,
    east_west_frame,
    mixed_frame,
    sequence_frame,
    trainer_track_frame,
)

VC, STRUCTURED, LOW_RANK = "variance_component", "structured", "low_rank"
UNIT3 = (1.0, 1.0, 1.0)
PRED_COLUMNS = {"race_id", "horse_id", "u", "p_ability"}


def block(name: str, cols, kind: str) -> InteractionBlock:
    return InteractionBlock(name=name, cols=tuple(cols), kind=kind)


def names(blocks) -> list[str]:
    return [b.name for b in blocks]


def tau_interval(fit, name: str) -> tuple[float, float, float]:
    """(point, lo, hi) of fit.tau(name), checked finite with 0 <= lo <= point <= hi."""
    iv = fit.tau(name)
    point, lo, hi = float(iv.mean), float(iv.lo), float(iv.hi)
    assert all(math.isfinite(x) for x in (point, lo, hi)), (name, point, lo, hi)
    assert 0.0 <= lo <= point <= hi, (name, point, lo, hi)
    return point, lo, hi


def predict_on(fit, frame: pd.DataFrame) -> pd.DataFrame:
    """fit.predict(frame) checked for its shape and keyed back onto frame's (race_id, horse_id)
    rows, in frame's order."""
    out = fit.predict(frame)
    assert isinstance(out, pd.DataFrame)
    assert PRED_COLUMNS <= set(out.columns), sorted(out.columns)
    assert len(out) == len(frame)
    got = out.set_index(["race_id", "horse_id"])
    assert got.index.is_unique
    want = pd.MultiIndex.from_frame(frame[["race_id", "horse_id"]])
    assert set(got.index) == set(want)
    return got.reindex(want)


def check_probabilities(out: pd.DataFrame, race_id) -> None:
    u = out["u"].to_numpy(dtype=float)
    p = out["p_ability"].to_numpy(dtype=float)
    codes = pd.factorize(np.asarray(race_id, dtype=object))[0]
    assert np.isfinite(u).all() and ((p > 0) & (p < 1)).all()
    sums = pd.Series(p).groupby(codes).sum().to_numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    np.testing.assert_allclose(p, softmax_within(u, codes), atol=1e-9)


HJ = block("horse_x_jockey", ("horse_id", "jockey_id"), VC)
HT = block("horse_x_trainer", ("horse_id", "trainer_id"), VC)
TT = block("trainer_x_track", ("trainer_id", "venue"), VC)
APT = block("distance_aptitude", ("horse_id", "distance_m", "surface"), STRUCTURED)
HV = block("horse_x_track", ("horse_id", "venue"), LOW_RANK)
EW = block("trainer_x_track_factor", ("trainer_id", "venue"), LOW_RANK)


# ---------------------------------------------------------------------------------------------
# DoD: the cell-size table counts only runs before each race
# ---------------------------------------------------------------------------------------------

D1 = pd.Timestamp("2024-01-06 01:00", tz="UTC")
HOUR, WEEK = pd.Timedelta(hours=1), pd.Timedelta(days=7)
# Race ids run against time (r9 first, r1 last). r2 is at r9's post time at another venue (a
# tie); r5 is later the same day; r1 a week later. Trainer t1 has two runners in r9.
HAND = pd.DataFrame(
    [
        ("jra:r9", D1, "h1", "j1", "t1", "tokyo"),
        ("jra:r9", D1, "h2", "j2", "t1", "tokyo"),
        ("jra:r9", D1, "h3", "j3", "t2", "tokyo"),
        ("jra:r2", D1, "h4", "j4", "t1", "kyoto"),
        ("jra:r5", D1 + HOUR, "h5", "j1", "t1", "tokyo"),
        ("jra:r5", D1 + HOUR, "h6", "j3", "t2", "tokyo"),
        ("jra:r1", D1 + WEEK, "h1", "j1", "t1", "tokyo"),
        ("jra:r1", D1 + WEEK, "h3", "j2", "t2", "tokyo"),
        ("jra:r1", D1 + WEEK, "h5", "j1", "t1", "tokyo"),
    ],
    columns=["race_id", "race_utc", "horse_id", "jockey_id", "trainer_id", "venue"],
)
HAND_PRIOR = {
    ("horse_id",): [0, 0, 0, 0, 0, 0, 1, 1, 1],
    ("jockey_id",): [0, 0, 0, 0, 1, 1, 2, 1, 2],
    ("trainer_id",): [0, 0, 0, 0, 3, 1, 4, 2, 4],
    ("horse_id", "jockey_id"): [0, 0, 0, 0, 0, 0, 1, 0, 1],
    ("trainer_id", "venue"): [0, 0, 0, 0, 2, 1, 3, 2, 3],
}


def prior_counts(frame: pd.DataFrame, block_cols: tuple[str, ...]) -> np.ndarray:
    """cell_sizes(frame, block=...)'s n_prior, checked for its shape, in frame's row order."""
    out = cell_sizes(frame, block=block_cols)
    assert isinstance(out, pd.DataFrame)
    assert {"race_id", "horse_id", "n_prior"} <= set(out.columns), sorted(out.columns)
    assert len(out) == len(frame)
    got = out.set_index(["race_id", "horse_id"])["n_prior"]
    assert got.index.is_unique
    got = got.reindex(pd.MultiIndex.from_frame(frame[["race_id", "horse_id"]]))
    assert got.notna().all()
    values = got.to_numpy(dtype=float)
    assert (values == np.round(values)).all() and (values >= 0).all()
    return values.astype(np.int64)


def brute_prior(frame: pd.DataFrame, block_cols: tuple[str, ...]) -> np.ndarray:
    """For each row, the rows with the same values in block_cols and a strictly earlier
    race_utc, by a double loop."""
    keys = list(zip(*(frame[c].tolist() for c in block_cols), strict=True))
    times = [pd.Timestamp(t) for t in frame["race_utc"]]
    return np.array(
        [
            sum(1 for j in range(len(keys)) if keys[j] == keys[i] and times[j] < times[i])
            for i in range(len(keys))
        ],
        dtype=np.int64,
    )


def random_card(seed: int) -> pd.DataFrame:
    """48 races over 6 days, 4 post times a day with a race at each of 2 venues at every post time
    (ties), 6 runners from 25 horses and 8 jockeys (distinct within a race), trainers from 5
    (repeating within a race). Race ids are opaque and not in time order; rows are shuffled on
    index labels that are not positions."""
    rng = np.random.default_rng(seed)
    rows = []
    race_no = rng.permutation(48)
    for r in range(48):
        day, slot, venue = r // 8, (r % 8) // 2, ("tokyo", "kyoto")[r % 2]
        when = D1 + pd.Timedelta(days=day) + slot * HOUR
        horses = rng.choice(25, 6, replace=False)
        jockeys = rng.choice(8, 6, replace=False)
        trainers = rng.integers(0, 5, size=6)
        for h, j, t in zip(horses, jockeys, trainers, strict=True):
            rows.append((f"jra:c{race_no[r]:03d}", when, f"h{h:02d}", f"j{j}", f"t{t}", venue))
    frame = pd.DataFrame(
        rows, columns=["race_id", "race_utc", "horse_id", "jockey_id", "trainer_id", "venue"]
    )
    frame = frame.iloc[rng.permutation(len(frame))]
    frame.index = pd.Index(30_000 + rng.permutation(len(frame)), name="row")
    return frame


def test_cell_size_table_counts_prior_runs_only():
    """The hand card: a cell's count at a race is its rows in strictly earlier races only. t1's
    two runners in r9 do not count each other, r2 at r9's post time (another venue) counts
    neither way, r5 later that day counts r9 and r2, and r1 a week later counts all three; the
    count follows the block (horse, jockey, trainer, horse x jockey, trainer x venue). Shuffled,
    the hand card gives the same counts. On a random card of 288 runs with ties at every post
    time, every block's counts are the brute-force counts, and dropping the races from day 4 on
    leaves the counts of the earlier races unchanged: a later race never counts."""
    for cols, want in HAND_PRIOR.items():
        np.testing.assert_array_equal(prior_counts(HAND, cols), want, err_msg=str(cols))
        np.testing.assert_array_equal(brute_prior(HAND, cols), want, err_msg=f"ref {cols}")
    rng = np.random.default_rng(9300)
    order = rng.permutation(len(HAND))
    shuffled = HAND.iloc[order]
    shuffled.index = pd.Index([f"x{i}" for i in rng.permutation(len(HAND))])
    for cols, want in HAND_PRIOR.items():
        expected = np.asarray(want)[order]
        np.testing.assert_array_equal(prior_counts(shuffled, cols), expected, err_msg=str(cols))

    card = random_card(9301)
    cutoff = D1 + pd.Timedelta(days=3)
    early = card[card["race_utc"] < cutoff]
    assert 0 < len(early) < len(card)
    for cols in HAND_PRIOR:
        got = prior_counts(card, cols)
        np.testing.assert_array_equal(got, brute_prior(card, cols), err_msg=str(cols))
        before = got[(card["race_utc"] < cutoff).to_numpy()]
        np.testing.assert_array_equal(prior_counts(early, cols), before, err_msg=str(cols))
    trainer_counts = prior_counts(card, ("trainer_id",))
    assert trainer_counts.max() > 30 and (trainer_counts == 0).any()


# ---------------------------------------------------------------------------------------------
# Outline 1: n_half = 1 / (I tau^2)
# ---------------------------------------------------------------------------------------------


def test_n_half_matches_the_table():
    """tau = 0.1 at I = 0.176 nats per run needs 568 runs (the table's row); over a spread of tau
    and I the value is 1 / (I tau^2): halving tau quadruples it, doubling I halves it."""
    assert abs(n_half(0.1, 0.176) - 568) <= 1
    table = ((0.1, 0.176), (0.05, 0.176), (0.3, 0.176), (0.1, 0.5), (0.2, 0.25), (0.02, 0.03))
    for tau, info in table:
        got = n_half(tau, info)
        assert got == pytest.approx(1.0 / (info * tau * tau), rel=1e-9), (tau, info, got)
    assert n_half(0.05, 0.176) == pytest.approx(4 * n_half(0.1, 0.176), rel=1e-9)
    assert n_half(0.1, 0.352) == pytest.approx(n_half(0.1, 0.176) / 2, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# DoD: each block has tau with a CI, and its standalone ΔLL
# ---------------------------------------------------------------------------------------------

TT_BASE = dict(k=3, discounts=UNIT3, entity_cols=("trainer_id",), features=["x1"])


@pytest.fixture(scope="module")
def planted_tt():
    """A trainer x track tau of 0.3 (interactions_synth.trainer_track_frame), the block fitted on
    the first 7,000 races (about 1,000 runs per cell); the last 2,000 are held out."""
    frame, codes = trainer_track_frame(9302, 0.3)
    in_train = codes < TT_TRAIN_RACES
    train, test = frame[in_train], frame[~in_train]
    fit = fit_with_blocks(train, AbilityModel(**TT_BASE), [TT])
    return train, test, fit


def test_variance_component_recovers_a_planted_tau(planted_tt):
    """With hundreds of runs in each of 48 trainer x venue cells, a planted tau of 0.3 is
    estimated within 0.1, its interval holds 0.3 and its lower bound is clear of 0; with no cell
    effect (tau = 0, about 1,300 runs per cell) the estimate is below 0.05 and the interval's
    upper bound below 0.15."""
    train, _, fit = planted_tt
    per_cell = train.groupby(["trainer_id", "venue"]).size()
    assert len(per_cell) == 48 and per_cell.min() >= 500
    point, lo, hi = tau_interval(fit, TT.name)
    assert abs(point - 0.3) < 0.1, point
    assert lo <= 0.3 <= hi, (lo, hi)
    assert lo > 0.1, lo

    null_frame, _ = trainer_track_frame(9303, 0.0)
    null_fit = fit_with_blocks(null_frame, AbilityModel(**TT_BASE), [TT])
    point0, lo0, hi0 = tau_interval(null_fit, TT.name)
    assert point0 < 0.05, point0
    assert hi0 < 0.15, hi0


def test_a_planted_trainer_track_block_improves_held_out_log_loss(planted_tt):
    """ΔLL standalone, synthetically: on the 2,000 held-out races the block's winner
    log-likelihood beats the same AbilityModel fitted without it on the same training races by
    more than 0.01 nats per race (the reference gain is about 0.025). predict has one row per
    runner, p_ability the softmax of u within race, and never reads finish_pos."""
    train, test, fit = planted_tt
    with_block = predict_on(fit, test.drop(columns=["finish_pos"]))
    check_probabilities(with_block, test["race_id"])
    last = test.groupby("race_id")["finish_pos"].transform("max")
    reversed_order = predict_on(fit, test.assign(finish_pos=last + 1 - test["finish_pos"]))
    np.testing.assert_allclose(
        reversed_order["u"].to_numpy(dtype=float), with_block["u"].to_numpy(dtype=float), atol=1e-12
    )

    base = AbilityModel(**TT_BASE).fit(train).predict(test)
    won = test["finish_pos"].to_numpy() == 1
    ll_with = np.log(with_block["p_ability"].to_numpy(dtype=float)[won]).sum()
    ll_without = np.log(base["p_ability"].to_numpy(dtype=float)[won]).sum()
    per_race = (ll_with - ll_without) / test["race_id"].nunique()
    assert per_race > 0.01, per_race


def test_every_fitted_block_has_a_tau_interval():
    """Horse x jockey (cells of a few runs, no planted effect), trainer x track, the structured
    distance aptitude and the low-rank horse x track, fitted together on 350 races: each has a
    finite tau interval with 0 <= lo <= point <= hi, and a name the fit has no block for raises
    KeyError. predict on the 50 later races has one row per runner, p_ability the softmax of u
    within race, and gives the same u whether finish_pos is dropped or reversed."""
    frame, codes = mixed_frame(9310)
    cells = frame.groupby(["horse_id", "jockey_id"]).size()
    assert cells.median() <= 5 and cells.quantile(0.9) <= 10
    train, test = frame[codes < MIX_TRAIN_RACES], frame[codes >= MIX_TRAIN_RACES]
    base = AbilityModel(
        k=3, discounts=(1.0, 0.5, 0.25), entity_cols=("jockey_id", "trainer_id"), features=["x1"]
    )
    blocks = [HJ, TT, APT, HV]
    fit = fit_with_blocks(train, base, blocks)
    for b in blocks:
        tau_interval(fit, b.name)
    with pytest.raises(KeyError):
        fit.tau("not_a_block")

    out = predict_on(fit, test.drop(columns=["finish_pos"]))
    check_probabilities(out, test["race_id"])
    last = test.groupby("race_id")["finish_pos"].transform("max")
    again = predict_on(fit, test.assign(finish_pos=last + 1 - test["finish_pos"]))
    np.testing.assert_allclose(
        again["u"].to_numpy(dtype=float), out["u"].to_numpy(dtype=float), atol=1e-12
    )


# ---------------------------------------------------------------------------------------------
# DoD: horse x trainer is not fitted where it is aliased with the horse
# ---------------------------------------------------------------------------------------------


def test_aliased_horse_trainer_block_is_dropped():
    """40 horses of 10 runs. Where every horse keeps one trainer, horse x trainer is dropped and
    horse x jockey, trainer x track and the structured aptitude are kept, in order, the input
    list unchanged. Where every horse changed trainer after 5 runs, nothing is dropped. The
    decision reads the data: renamed columns and block names give the same result, and on a card
    where every horse keeps one jockey but changed trainer, horse x jockey is the block dropped."""
    blocks = [HJ, HT, TT, APT]
    constant = alias_frame(9320, trainer_change=False)
    assert constant.groupby("horse_id")["trainer_id"].nunique().max() == 1
    assert constant.groupby("trainer_id")["venue"].nunique().min() > 1
    kept = drop_aliased(blocks, constant)
    assert all(isinstance(b, InteractionBlock) for b in kept)
    assert names(kept) == [HJ.name, TT.name, APT.name]
    assert names(blocks) == [HJ.name, HT.name, TT.name, APT.name]

    changed = alias_frame(9321, trainer_change=True)
    assert changed.groupby("horse_id")["trainer_id"].nunique().min() == 2
    assert names(drop_aliased(blocks, changed)) == names(blocks)

    renamed = constant.rename(columns={"trainer_id": "stable", "horse_id": "runner"})
    by_stable = block("runner_by_stable", ("runner", "stable"), VC)
    stable_track = block("stable_by_course", ("stable", "venue"), VC)
    assert names(drop_aliased([by_stable, stable_track], renamed)) == ["stable_by_course"]
    moved = changed.rename(columns={"trainer_id": "stable", "horse_id": "runner"})
    assert names(drop_aliased([by_stable, stable_track], moved)) == [
        "runner_by_stable",
        "stable_by_course",
    ]

    one_jockey = alias_frame(9322, trainer_change=True, jockey_constant=True)
    assert one_jockey.groupby("horse_id")["jockey_id"].nunique().max() == 1
    assert names(drop_aliased([HJ, HT, TT], one_jockey)) == [HT.name, TT.name]


# ---------------------------------------------------------------------------------------------
# DoD: the transfer step applies only after a stable change
# ---------------------------------------------------------------------------------------------

# trainers run by run, stable_runs, the step run by run (T = on)
TRANSFER_CASES = {
    "debut only": ("A", 2, "F"),
    "one trainer throughout": ("AAAAAA", 2, "FFFFFF"),
    "a one-off blip": ("AAABAA", 2, "FFFFFF"),
    "a blip on the last run": ("AAAB", 2, "FFFF"),
    "blips to two other trainers": ("AABACA", 2, "FFFFFF"),
    "a stable change": ("AAABBB", 2, "FFFFTT"),
    "a change after the debut run": ("ABBB", 2, "FFTT"),
    "a late change": ("AAAAAAAABB", 2, "FFFFFFFFFT"),
    "a stable change at three runs": ("AAABBB", 3, "FFFFFT"),
    "a two-run spell at three runs": ("AAABBA", 3, "FFFFFF"),
    "two-run spells at three runs": ("ABBABBAA", 3, "FFFFFFFF"),
    "a change after the debut at three runs": ("ABBBB", 3, "FFFTT"),
}


def steps(frame: pd.DataFrame, stable_runs: int) -> pd.Series:
    """transfer_step's output checked for its shape: bool per row on frame's index."""
    out = transfer_step(frame, stable_runs=stable_runs)
    assert isinstance(out, pd.Series)
    assert out.index.equals(frame.index)
    values = out.to_numpy()
    assert pd.notna(values).all()
    assert set(np.unique(values.astype(np.int64)).tolist()) <= {0, 1}
    return out.astype(bool)


def check_sequences(sequences: dict[str, str], wants: dict[str, str], stable_runs: int, seed: int):
    frame, where = sequence_frame(sequences, seed)
    got = steps(frame, stable_runs)
    for key, letters in sequences.items():
        run_by_run = "".join("T" if got[where[(key, k)]] else "F" for k in range(len(letters)))
        assert run_by_run == wants[key], (key, letters, stable_runs, run_by_run)


def spells(rng: np.random.Generator, letters: str, lengths: tuple[int, int], count: int) -> str:
    out, last = [], None
    for _ in range(count):
        letter = rng.choice([c for c in letters if c != last])
        out.append(letter * int(rng.integers(lengths[0], lengths[1] + 1)))
        last = letter
    return "".join(out)


def test_transfer_step_applies_only_after_a_stable_change():
    """The hand table (TRANSFER_CASES): a one-off trainer, or a spell shorter than stable_runs,
    never turns the step on; a new trainer kept for stable_runs consecutive runs turns it on from
    the run that completes them. Over every A^m B^n (m 1-5, n 1-6) at stable_runs 2 and 3 the
    step is off for the first m + stable_runs - 1 runs and on after (off throughout when n is
    short); 30 random careers with short spells away from the debut trainer never turn it on.
    As of each run: on 40 random careers, truncating every career at three cut-offs leaves the
    earlier runs' values unchanged."""
    for stable_runs in (2, 3):
        cases = {k: v for k, v in TRANSFER_CASES.items() if v[1] == stable_runs}
        check_sequences(
            {k: v[0] for k, v in cases.items()},
            {k: v[2] for k, v in cases.items()},
            stable_runs,
            seed=9330 + stable_runs,
        )

    for stable_runs in (2, 3):
        sustained, wants = {}, {}
        for m in range(1, 6):
            for n in range(1, 7):
                key = f"A{m}B{n}"
                sustained[key] = "A" * m + "B" * n
                off = m + stable_runs - 1 if n >= stable_runs else m + n
                wants[key] = "F" * off + "T" * (m + n - off)
        check_sequences(sustained, wants, stable_runs, seed=9340 + stable_runs)

    rng = np.random.default_rng(9350)
    for stable_runs in (2, 3):
        blips = {}
        for i in range(30):
            career = "A" * int(rng.integers(1, 4))
            for _ in range(int(rng.integers(1, 4))):
                career += rng.choice(list("BCD")) * int(rng.integers(1, stable_runs))
                career += "A" * int(rng.integers(1, 4))
            blips[f"blips{i}"] = career
        check_sequences(blips, {k: "F" * len(v) for k, v in blips.items()}, stable_runs, 9360)

    careers = {f"c{i}": spells(rng, "ABC", (1, 4), int(rng.integers(2, 6))) for i in range(40)}
    frame, _ = sequence_frame(careers, 9370)
    for stable_runs in (2, 3):
        full = steps(frame, stable_runs)
        assert full.any() and not full.all()
        times = np.sort(frame["race_utc"].unique())
        for cut in (times[len(times) // 4], times[len(times) // 2], times[3 * len(times) // 4]):
            early = frame[frame["race_utc"] <= cut]
            part = steps(early, stable_runs)
            pd.testing.assert_series_equal(part, full.loc[early.index], check_names=False)


# ---------------------------------------------------------------------------------------------
# Outline 1: horse x distance in the structured form
# ---------------------------------------------------------------------------------------------

PROBE_DISTANCES = (1200, 2000, 3000)


def test_structured_distance_aptitude_recovers_a_planted_optimum():
    """Six focal horses with peaked aptitudes (two best at 1,200 m, two at 2,000 m, two at
    3,000 m; -2 per (1,000 m)^2 from the optimum) among 50 flat field horses, 900 races. Rated
    against the field (u less the mean u of the 50 field horses in one probe race per distance),
    a horse best at 2,000 m is rated higher at 2,000 than at 1,200 and at 3,000, and the sprinters
    and stayers are rated highest at their own optimum."""
    frame = aptitude_frame(9380)
    base = AbilityModel(k=3, discounts=UNIT3, entity_cols=(), features=["x1"])
    fit = fit_with_blocks(frame, base, [APT])
    tau_interval(fit, APT.name)

    when = after(frame)
    rows = []
    for h in OPTIMA:
        for d in PROBE_DISTANCES:
            race = f"probe:{h}:{d}"
            for runner in (h, *FIELD_HORSES):
                rows.append((race, when, runner, 0.0, d, "turf", "tokyo"))
    probe = pd.DataFrame(
        rows, columns=["race_id", "race_utc", "horse_id", "x1", "distance_m", "surface", "venue"]
    )
    out = predict_on(fit, probe)
    check_probabilities(out, probe["race_id"])
    u = pd.Series(out["u"].to_numpy(dtype=float), index=out.index)
    rating = {}
    for h in OPTIMA:
        for d in PROBE_DISTANCES:
            race = f"probe:{h}:{d}"
            field_mean = np.mean([u[(race, f)] for f in FIELD_HORSES])
            rating[(h, d)] = u[(race, h)] - field_mean
    for h, best in OPTIMA.items():
        for d in PROBE_DISTANCES:
            if d != best:
                assert rating[(h, best)] > rating[(h, d)], (h, best, d, rating)


# ---------------------------------------------------------------------------------------------
# Outline 1 (E36): the one-factor east/west trainer x track term
# ---------------------------------------------------------------------------------------------


def test_one_factor_trainer_track_term_learns_east_west():
    """Trainers and venues each east or west, the cell effect +0.4 at home and -0.4 away (rank
    one), fitted with the low-rank block alone. Against the two neutral trainers, the double
    difference nakayama minus hanshin of each sparse trainer, who never ran at either, carries
    its region's sign (planted 0.8, asserted above 0.4): a cell the data never saw is predicted
    from the factor, which a per-cell effect cannot do. Each east and west pair's tokyo minus
    kyoto double difference is above 0.8 (planted 1.6)."""
    frame, trainer_of = east_west_frame(9390)
    for role in SPARSE_ROLES:
        ran = set(frame.loc[frame["trainer_id"] == trainer_of[role], "venue"])
        assert ran == set(SPARSE_VENUES), (role, ran)
    base = AbilityModel(k=3, discounts=UNIT3, entity_cols=("trainer_id",), features=["x1"])
    fit = fit_with_blocks(frame, base, [EW])
    tau_interval(fit, EW.name)

    when = after(frame)
    rows = [
        (f"probe:{v}", when, f"probe:{role}", 0.0, trainer_of[role], v)
        for v in VENUES
        for role in EW_ROLES
    ]
    probe = pd.DataFrame(
        rows, columns=["race_id", "race_utc", "horse_id", "x1", "trainer_id", "venue"]
    )
    out = predict_on(fit, probe)
    u = pd.Series(out["u"].to_numpy(dtype=float), index=out.index)

    def at(role: str, venue: str) -> float:
        return float(u[(f"probe:{venue}", f"probe:{role}")])

    def neutral(venue: str) -> float:
        return (at("n1", venue) + at("n2", venue)) / 2.0

    def against_neutral(role: str, venue: str) -> float:
        return at(role, venue) - neutral(venue)

    for role in SPARSE_ROLES:
        dd = against_neutral(role, "nakayama") - against_neutral(role, "hanshin")
        assert EW_ROLES[role] * dd > 0.4, (role, dd)
    for e in ("e1", "e2", "e3", "e4"):
        for w in ("w1", "w2", "w3", "w4"):
            dd = (at(e, "tokyo") - at(w, "tokyo")) - (at(e, "kyoto") - at(w, "kyoto"))
            assert dd > 0.8, (e, w, dd)


# ---------------------------------------------------------------------------------------------
# #574 DoD: each member's abilities through C5's ordering to Place and Wide
# ---------------------------------------------------------------------------------------------

QC_SMALL = {"a2": 0.1, "b2": -0.02, "h2": 0.05, "e2": -0.05}
QC_SMALL |= {"a3": 0.15, "b3": -0.03, "h3": 0.04, "e3": -0.04}
PW_ORDERINGS = {
    "harville": lambda: exotic.Ordering("harville"),
    "henery": lambda: exotic.Ordering("henery"),
    "quadratic_context": lambda: exotic.Ordering("quadratic_context", dict(QC_SMALL)),
}
# (runners when sales opened, starters): Place not sold at 4; two places 5-7; three from 8; a
# late scratch from 8 to 7 still pays three.
PW_FIELDS = [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (12, 12), (18, 18), (8, 7)]


def places_by_hand(field_size: int) -> int:
    return 0 if field_size < 5 else 2 if field_size < 8 else 3


def pw_card(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """One race per PW_FIELDS entry, horse numbers drawn from 1-18 (gaps, as after scratches),
    Win probabilities from seeded strengths; rows shuffled on a non-default index."""
    rng = np.random.default_rng(seed)
    parts = []
    for k, (field, starters) in enumerate(PW_FIELDS):
        numbers = np.sort(rng.choice(np.arange(1, 19), size=starters, replace=False))
        w = np.exp(rng.normal(0.0, 1.2, starters))
        parts.append(
            pd.DataFrame(
                {
                    "race_id": f"jra:2021-05-{k + 1:02d}:05:11",
                    "horse_no": numbers.astype(int),
                    "field_size": field,
                    "p": w / w.sum(),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.iloc[rng.permutation(len(frame))]
    frame.index = pd.Index(700 + 3 * np.arange(len(frame)))
    return frame.drop(columns="p"), frame["p"]


def harville_top3(p: np.ndarray) -> dict[tuple[int, int, int], float]:
    """P(i, j, k the first three in that order) under Harville, by brute force."""
    return {
        (i, j, k): p[i] * p[j] / (1 - p[i]) * p[k] / (1 - p[i] - p[j])
        for i, j, k in itertools.permutations(range(len(p)), 3)
    }


def keyed(out: pd.DataFrame, race_id: str) -> dict[str, float]:
    part = out.loc[out["race_id"].eq(race_id)]
    assert part["combination"].is_unique, race_id
    return dict(zip(part["combination"].astype(str), part["p"].astype(float), strict=True))


@pytest.mark.parametrize("family", sorted(PW_ORDERINGS))
def test_member_place_wide_probabilities_sum_correctly(family):
    """Eight races (4 to 18 runners at sale, one late scratch) through each C5 ordering family:
    WIN reproduces the member's probabilities; PLACE sums to the places paid by the field at
    sale (no rows for a field of 4) and WIDE's pairs to 3, every p in [0, 1], each starter's
    Place at least its Win and its Wide pairs summing to its top-three probability. Every value
    equals the ordering's own joint_probs, and under Harville a brute-force sum over ordered
    first threes, so a member's abilities move every figure."""
    rows, p_win = pw_card(5740 + len(family))
    ordering = PW_ORDERINGS[family]()
    tol = 1e-6 if family == "henery" else 1e-9
    got = {
        pool: study.member_pool_probs(rows, p_win, ordering, pool)
        for pool in (exotic.WIN, exotic.PLACE, exotic.WIDE)
    }
    for pool, out in got.items():
        assert isinstance(out, pd.DataFrame), pool
        assert {"race_id", "combination", "p"} <= set(out.columns), (pool, list(out.columns))
        p = out["p"].to_numpy(dtype=float)
        assert np.isfinite(p).all() and (p >= -tol).all() and (p <= 1 + tol).all(), pool
        assert set(out["race_id"]) <= set(rows["race_id"]), pool

    for race_id, race in rows.groupby("race_id"):
        numbers = race["horse_no"].astype(int).tolist()
        field = int(race["field_size"].iloc[0])
        p = p_win.reindex(race.index).to_numpy(dtype=float)
        single = [str(h) for h in numbers]
        pairs = [f"{a},{b}" for a, b in itertools.combinations(sorted(numbers), 2)]

        win = keyed(got[exotic.WIN], race_id)
        assert sorted(win) == sorted(single), race_id
        np.testing.assert_allclose([win[str(h)] for h in numbers], p, atol=1e-9)

        place = keyed(got[exotic.PLACE], race_id)
        places = places_by_hand(field)
        if places == 0:
            assert place == {}, (race_id, "PLACE is not sold to a field of 4")
        else:
            assert sorted(place) == sorted(single), race_id
            assert sum(place.values()) == pytest.approx(places, abs=1e-6), (race_id, field)
            for h, ph in zip(numbers, p, strict=True):
                assert place[str(h)] >= ph - tol, (race_id, h)
            want = ordering.joint_probs(p, exotic.PLACE, horses=numbers, field_at_sale=field)
            for c, v in want.items():
                assert place[c] == pytest.approx(v, abs=tol), (race_id, c)

        wide = keyed(got[exotic.WIDE], race_id)
        assert sorted(wide) == sorted(pairs), race_id
        assert sum(wide.values()) == pytest.approx(3.0, abs=1e-6), race_id
        want = ordering.joint_probs(p, exotic.WIDE, horses=numbers, field_at_sale=field)
        for c, v in want.items():
            assert wide[c] == pytest.approx(v, abs=tol), (race_id, c)
        t = ordering.trifecta(p)
        in_top3 = t.sum(axis=(1, 2)) + t.sum(axis=(0, 2)) + t.sum(axis=(0, 1))
        for i, h in enumerate(numbers):
            with_h = sum(v for c, v in wide.items() if str(h) in c.split(","))
            assert with_h == pytest.approx(in_top3[i], abs=1e-6), (race_id, h)

        if family == "harville":
            brute = harville_top3(p)
            for i, h in enumerate(numbers):
                if places:
                    hand = sum(v for top, v in brute.items() if i in top[:places])
                    assert place[str(h)] == pytest.approx(hand, abs=1e-9), (race_id, h)
            for a, b in itertools.combinations(range(len(numbers)), 2):
                hand = sum(v for top, v in brute.items() if a in top and b in top)
                key = ",".join(str(x) for x in sorted((numbers[a], numbers[b])))
                assert wide[key] == pytest.approx(hand, abs=1e-9), (race_id, key)
