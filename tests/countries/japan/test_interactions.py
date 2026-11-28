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

import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models import interactions
from horsetrack.countries.japan.models.ability import EB_MAX, TAU2_FLOOR, AbilityModel
from horsetrack.countries.japan.models.interactions import (
    InteractionBlock,
    cell_sizes,
    drop_aliased,
    fit_with_blocks,
    n_half,
    transfer_step,
)

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
    few_run_cells_frame,
    horse_jockey_frame,
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
# #570: the block-Jacobi path and the fixed-point step
# ---------------------------------------------------------------------------------------------

LOCAL_DENSE_MAX = 100  # below the 180 horse x jockey cells, above the 30 to 60 horses of the base
HJ_BASE = dict(k=3, discounts=UNIT3, features=["x1"])
# (seed, planted cell tau, planted horse level sd, races, horses, the base entity): the first
# five are the store-scale shape, `horse_id` among the base entities next to its own horse x
# jockey cells (a clear cell effect with a horse level, no cell effect, a large one on fewer
# races, a strong horse level, twice the horses); the last, jockey_id as the base entity with no
# horse level, is the case with nothing for the entity and the cells to share.
HJ_CASES = [
    (21, 0.4, 0.3, 500, 30, "horse_id"),
    (22, 0.0, 0.3, 500, 30, "horse_id"),
    (23, 0.7, 0.3, 300, 30, "horse_id"),
    (25, 0.4, 0.6, 500, 30, "horse_id"),
    (26, 0.4, 0.3, 500, 60, "horse_id"),
    (21, 0.4, 0.0, 500, 30, "jockey_id"),
]
FLOOR_SD = math.sqrt(TAU2_FLOOR)
SD_TOL = 0.05  # relative, local against dense, of every prior sd that is clear of the floor


def test_block_jacobi_matches_the_dense_fit(monkeypatch):
    """Horse x jockey cells (180 or 360) with the horse level (or the jockey) as the base entity,
    fitted dense, then with DENSE_MAX forced below the block's size so its levels take
    block-Jacobi steps (asserted: the block is local, the base entity is not, the dense fit has no
    local term). (1) At the local fit's own prior variances, the dense MAP agrees with the local
    one to 1e-6 (MAP_TOL is 1e-8): the local MAP is exact. (2) The empirical-Bayes prior sds,
    whose trace and log-determinant use the block-diagonal approximation, are each within 5% of
    the dense fit's (the block's tau and the base entity's sd; where the dense sd is at the floor
    the local one is too), the block's tau within 0.15 of the planted one (0.14 where none is
    planted), and the two fits' utilities within 0.05. A horse level and that horse's own cells
    are confounded (the cells' mean is a horse level), so an approximation that treats the
    entity and the cells as independent misplaces the entity's variance (the horse sd falls to
    the floor from the dense fit's 0.2 to 0.3 and the cells' tau moves 10% or more), which these
    cases catch."""
    taus, clear_horse_sds = {}, 0
    for seed, planted, level_sd, races, horses, entity in HJ_CASES:
        label = (seed, planted, level_sd, races, horses, entity)
        frame = horse_jockey_frame(seed, planted, races, level_sd=level_sd, horses=horses)
        assert frame.groupby(["horse_id", "jockey_id"]).ngroups > LOCAL_DENSE_MAX
        base = dict(HJ_BASE, entity_cols=(entity,))
        dense = fit_with_blocks(frame, AbilityModel(**base), [HJ])
        assert not dense._model.has_local

        with monkeypatch.context() as patch:
            patch.setattr(interactions, "DENSE_MAX", LOCAL_DENSE_MAX)
            local = fit_with_blocks(frame, AbilityModel(**base), [HJ])
        is_local = {t.label: t.local for t in local._model.terms}
        assert is_local == {f"entity:{entity}": False, HJ.name: True}, (label, is_local)
        assert local._model.has_local

        # (1) the MAP at one set of prior variances, dense against block-Jacobi
        same_z = interactions._Model(frame, AbilityModel(**base), [HJ])
        assert not same_z.has_local
        same_z.z = local._model.z.copy()
        same_z.solve(same_z.z, interactions.MAP_TOL)
        exact = interactions.InteractionFit(same_z)
        u_exact = predict_on(exact, frame)["u"].to_numpy(dtype=float)
        u_local = predict_on(local, frame)["u"].to_numpy(dtype=float)
        np.testing.assert_allclose(u_local, u_exact, atol=1e-6)
        np.testing.assert_allclose(local.coef_.to_numpy(), exact.coef_.to_numpy(), atol=1e-6)

        # (2) the fitted prior sds, tau and utilities
        assert set(dense.prior_sd_) == set(local.prior_sd_) == {f"entity:{entity}", HJ.name}
        for name, sd_dense in dense.prior_sd_.items():
            sd_local = local.prior_sd_[name]
            if sd_dense < 2 * FLOOR_SD:
                assert sd_local < 2 * FLOOR_SD, (label, name, sd_local, sd_dense)
            else:
                assert abs(sd_local - sd_dense) < SD_TOL * sd_dense, (
                    label,
                    name,
                    sd_local,
                    sd_dense,
                )
        clear_horse_sds += entity == "horse_id" and dense.prior_sd_["entity:horse_id"] > 0.1
        tau_dense, tau_local = dense.tau_point(HJ.name), local.tau_point(HJ.name)
        taus[seed, entity] = tau_local
        assert abs(tau_local - tau_dense) < SD_TOL * tau_dense, label
        assert abs(tau_local - planted) < 0.15, (label, tau_local, planted)
        u_dense = predict_on(dense, frame)["u"].to_numpy(dtype=float)
        assert np.abs(u_local - u_dense).max() < 0.05, label
        point, lo, hi = tau_interval(local, HJ.name)
        assert lo <= point <= hi and point == pytest.approx(tau_local)
    # the horse-level cases are not vacuous: four have a horse sd clear of the floor (dense)
    assert clear_horse_sds >= 4, clear_horse_sds
    assert taus[22, "horse_id"] < taus[21, "horse_id"] < taus[23, "horse_id"], taus


# (seed, planted tau, races, DENSE_MAX or None): 12 trainers x 4 venues = 48 cells of 6 to 30
# runs. Cases 0 and 3 end with a prior sd clear of the floor, which the fixed point crawls to;
# cases 1, 2 and 4 end at the floor. Cases 2 and 4 are fitted block-Jacobi (48 > 20 cell levels).
WEAK_CASES = [
    (1, 0.0, 60, None),
    (2, 0.05, 60, None),
    (3, 0.0, 60, 20),
    (5, 0.3, 40, None),
    (6, 0.02, 200, 20),
]
WEAK_BASE = dict(k=3, discounts=UNIT3, entity_cols=("trainer_id",), features=["x1"])


def run_weak_fit(monkeypatch, frame, dense_max, start=None):
    """fit_with_blocks on the weak design, with the z and posterior inputs of every empirical-
    Bayes update recorded: [(z, [(m, trace, square) per group])]. Returns (fit, record)."""
    record = []
    original = interactions._Model._posterior

    def recording(self, z, *, trace):
        out = original(self, z, trace=trace)
        if trace:
            record.append((np.array(z, dtype=float), list(out[0])))
        return out

    with monkeypatch.context() as patch:
        patch.setattr(interactions._Model, "_posterior", recording)
        if dense_max is not None:
            patch.setattr(interactions, "DENSE_MAX", dense_max)
        fit = fit_with_blocks(frame, AbilityModel(**WEAK_BASE), [TT], start=start)
    return fit, record


def fixed_point_residuals(record) -> np.ndarray:
    """MacKay's update less z, per empirical-Bayes update and group, from the recorded
    (m, trace, square): ln(square / (m - trace / tau^2)) floored at ln TAU2_FLOOR."""
    floor = math.log(TAU2_FLOOR)
    rows = []
    for z, groups in record:
        row = []
        for g, (m, trace, square) in enumerate(groups):
            effective = m - trace * math.exp(-z[g])
            ratio = square / effective if effective > 0 and square > 0 else 0.0
            row.append((max(math.log(ratio), floor) if ratio > 0 else floor) - z[g])
        rows.append(row)
    return np.array(rows)


def doubled_steps(record) -> int:
    """How many (update, group) pairs took a doubled plain step: the secant of the last two
    updates had no valid slope (dz = 0 or the residual not falling as z rises), the residual kept
    its sign, and the step to the next z was more than the plain residual (the secant's own
    step is never used there)."""
    r = fixed_point_residuals(record)
    z = np.array([z for z, _ in record])
    count = 0
    for t in range(1, len(record) - 1):
        for g in range(r.shape[1]):
            dz, dr = z[t, g] - z[t - 1, g], r[t, g] - r[t - 1, g]
            invalid = dz == 0 or dr * dz >= 0
            if invalid and np.sign(r[t, g]) == np.sign(r[t - 1, g]) and r[t, g] != 0:
                step = z[t + 1, g] - z[t, g]
                if np.sign(step) == np.sign(r[t, g]) and abs(step) > 1.5 * abs(r[t, g]):
                    count += 1
    return count


def test_weak_variance_converges_within_eb_max(monkeypatch):
    """Trainer x venue cells of 6 to 30 runs (40 to 200 races), whose variances are weakly
    identified: planted tau 0, 0.05, 0.3, 0.02, two of them fitted block-Jacobi. Each fit
    succeeds (no RuntimeError) in at most EB_MAX updates, with every prior sd finite and no lower
    than the floor's, and a group the data leaves with no variance sits exactly at the floor.
    Started instead at prior sd 1.0 and then 0.01 (`start=`) it stops at the same prior sds (the
    fixed point, not the path, is the answer), each within 0.2% where it is clear of the floor
    (the runs differ by 0.015% at most; EB_LOCAL_TOL is 1e-4 in ln tau^2). The
    plain step of a group whose secant has no valid slope is doubled in four of the five fits."""
    crawled = 0
    for seed, planted, races, dense_max in WEAK_CASES:
        frame = few_run_cells_frame(seed, planted, races)
        assert frame.groupby(["trainer_id", "venue"]).size().median() <= 30
        fit, record = run_weak_fit(monkeypatch, frame, dense_max)
        label = (seed, planted, races, dense_max)
        assert fit._model.has_local == (dense_max is not None), label
        assert 2 <= len(record) <= EB_MAX, (label, len(record))
        sds = fit.prior_sd_
        assert set(sds) == {"entity:trainer_id", TT.name}
        assert all(math.isfinite(s) and s >= FLOOR_SD * (1 - 1e-9) for s in sds.values()), sds
        crawled += doubled_steps(record) > 0

        tolerance = 2e-3
        for start_sd in (1.0, 0.01):
            start = {"prior_sd": {name: start_sd for name in sds}}
            again, _ = run_weak_fit(monkeypatch, frame, dense_max, start=start)
            for name, sd in sds.items():
                if sd < 2 * FLOOR_SD:
                    assert again.prior_sd_[name] < 2 * FLOOR_SD, (label, name, start_sd)
                else:
                    assert again.prior_sd_[name] == pytest.approx(sd, rel=tolerance), (
                        label,
                        name,
                        start_sd,
                    )
    assert crawled >= 4, crawled
