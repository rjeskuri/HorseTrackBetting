"""#93's trainer debut prior measured by segment at the final price (ticket #573).

Contract pinned here (`horsetrack.countries.japan.studies.debut_prior`, a new module):
- `debut_labels(frame) -> pd.Series`: one bool per row, on frame's index. frame has race_id,
  race_utc (tz-aware) and horse_id. A row is a debut when no row of frame for the same horse_id
  has a strictly earlier race_utc: the row's own race never counts, nor another run at the same
  race_utc (the tie rule of `models.interactions.cell_sizes`), nor a later run. It reads no
  outcome. A horse's runs before frame's first row are unknown: pass the whole history, then cut
  the scored window.
- `label_segments(frame) -> pd.DataFrame`: frame also has race_class (the store's code, may be
  missing). One row per input row, on frame's index and in its order, with `debut` (the runner's
  `debut_labels`), `maiden` (race_class is DEBUT or MAIDEN: a missing class is False) and
  `segment`, one string per race: `<race_class>/debut` when any starter of the race debuts, else
  `<race_class>/seasoned`; a missing class reads `unknown`.
- `score_segments(rows, *, n_boot, level, seed, top=0.99) -> pd.DataFrame`: rows has one row per
  runner of the scored (out-of-fold) races, with race_id, segment, meeting_id, won (bool; a dead
  heat has several winners), win_odds (the final Win odds), paid (the return per unit stake at
  the recorded dividend: 0 on a loser), p_member (the trainer_debut member's probability at the
  final price, stage 2) and p_base (the base member's, same stage), each within-race normalised.
  One row per distinct segment, ranked by delta_ll, highest first, with columns: segment, rank
  (1 for the best, consecutive), races, meetings (distinct meeting_id of the segment's races),
  bets (runners), delta_ll (mean over the segment's races of the per-race gain
  ln p_member(winner) - ln p_base(winner), a dead heat's k winners weighted 1/k each; positive
  when the member is better), delta_ll_lo and delta_ll_hi (`core.eval.bootstrap.paired_bootstrap`
  on the per-race gains over meeting_id clusters, with n_boot, level, seed), tail_bets,
  tail_return and tail_lo, tail_hi (the segment's bets whose claim p_member x win_odds is at or
  above the `top` quantile of the segment's claims (`np.quantile`, linear): flat unit stakes at
  `paid`, returned per 100 staked, with the paired_bootstrap bounds of the per-bet returns x 100
  over meeting_id). A segment with one race still has its row. No segment is dropped.
- `FAMILY_ID`, `TICKET` (573) and `build_spec(bootstrap: dict) -> dict`: the pre-registered spec,
  plain JSON, with family_id, ticket, bootstrap (the dict passed), segments (axes
  ["debut", "maiden", "race_class"] and maiden_classes ["DEBUT", "MAIDEN"]) and tail
  ({"top": 0.99}); `Registry.freeze_family` takes it and a second freeze of an equal spec is a
  no-op.

Not pinned: the CI's method beyond the calls named above; the order of ties in the ranking; the
columns beyond those named; the real run (store reads, #93's out-of-fold export, the registry
records, the report), which needs local data. Expected values are written by hand, planted, or
derived by a brute-force scan; `core.eval.bootstrap.paired_bootstrap` is the one library call the
tests share with the module, to pin the CI's definition.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.studies import debut_prior
from horsetrack.countries.japan.studies.debut_prior import (
    FAMILY_ID,
    TICKET,
    build_spec,
    debut_labels,
    label_segments,
    score_segments,
)

# isort: split
from debut_prior_synth import (
    HAND_SEGMENT,
    SEGMENT_PLAN,
    T0,
    brute_debut,
    brute_segment,
    hand_history,
    random_history,
    scored_rows,
)

BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 400, "seed": 573}


# ---------------------------------------------------------------------------------------------
# Debut labels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_debut_labels_match_a_pairwise_scan(seed):
    frame = random_history(seed)
    got = debut_labels(frame)
    expected = brute_debut(frame)
    assert got.index.equals(frame.index)
    assert got.astype(bool).equals(expected)
    assert 0 < int(got.sum()) < len(frame)  # the table has both labels


def test_debut_label_reads_neither_row_order_nor_outcomes_nor_other_columns():
    frame = hand_history(seed=4)
    labels = debut_labels(frame)
    shuffled = frame.sample(frac=1.0, random_state=9)
    assert debut_labels(shuffled).loc[frame.index].equals(labels)
    noisy = frame.assign(
        finish_pos=frame["finish_pos"].iloc[::-1].to_numpy(),
        trainer_key="t:same",
        extra=np.arange(len(frame)),
    )
    assert debut_labels(noisy).equals(labels)
    bare = frame[["race_id", "race_utc", "horse_id"]]
    assert debut_labels(bare).equals(labels)


def test_a_later_run_added_to_a_random_card_never_changes_an_earlier_label():
    frame = random_history(5)
    labels = debut_labels(frame)
    horse = frame["horse_id"].iloc[0]
    late = pd.DataFrame(
        {
            "race_id": ["jra:late1", "jra:late2"],
            "race_utc": [frame["race_utc"].max() + pd.Timedelta(days=d) for d in (3, 9)],
            "horse_id": [horse, "h:never-seen"],
            "race_class": ["G1", "G1"],
        }
    )
    grown = pd.concat([frame, late], ignore_index=True)
    after = debut_labels(grown)
    assert after.iloc[: len(frame)].to_numpy().tolist() == labels.to_numpy().tolist()
    assert after.iloc[len(frame) :].tolist() == [False, True]


# ---------------------------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------------------------


def test_label_segments_on_the_hand_card():
    frame = hand_history(seed=2)
    out = label_segments(frame)
    assert out.index.equals(frame.index)
    assert {"debut", "maiden", "segment"} <= set(out.columns)
    assert out["debut"].astype(bool).equals(debut_labels(frame).astype(bool))
    for race, expected in HAND_SEGMENT.items():
        rows = out.loc[frame["race_id"] == f"jra:{race}"]
        assert set(rows["segment"]) == {expected}, race
    maiden = out["maiden"].astype(bool)
    assert maiden.equals(frame["race_class"].isin(["DEBUT", "MAIDEN"]))
    assert int(maiden.sum()) == 3 + 3 + 4  # the starters of r1, r2 and r3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_label_segments_match_a_brute_force_cut_on_a_random_card(seed):
    frame = random_history(seed)
    out = label_segments(frame)
    debut = brute_debut(frame)
    assert out.index.equals(frame.index)
    assert out["segment"].tolist() == brute_segment(frame, debut).tolist()
    is_maiden = frame["race_class"].isin(["DEBUT", "MAIDEN"]) & frame["race_class"].notna()
    assert out["maiden"].astype(bool).tolist() == is_maiden.tolist()
    # a race has exactly one segment, and the unknown class is kept, not dropped
    assert out["segment"].groupby(frame["race_id"]).nunique().eq(1).all()
    assert frame["race_class"].isna().any()
    assert out.loc[frame["race_class"].isna(), "segment"].str.startswith("unknown/").all()
    assert set(out["segment"]) >= {"DEBUT/debut", "DEBUT/seasoned", "G1/seasoned", "OP/debut"}


def test_each_axis_moves_the_segment():
    base = pd.DataFrame(
        {
            "race_id": ["jra:p1"] * 2 + ["jra:p2"] * 2,
            "race_utc": [T0] * 2 + [T0 + pd.Timedelta(days=7)] * 2,
            "horse_id": ["h:a", "h:b", "h:a", "h:c"],
            "race_class": ["MAIDEN"] * 4,
        }
    )
    first = label_segments(base)
    assert first.loc[first.index[:2], "segment"].eq("MAIDEN/debut").all()
    assert first.loc[first.index[2:], "segment"].eq("MAIDEN/debut").all()  # h:c debuts in p2
    assert first["maiden"].astype(bool).all()
    # the class: same fields, another class
    other = label_segments(base.assign(race_class="C1"))
    assert other["segment"].str.startswith("C1/").all()
    assert not other["maiden"].astype(bool).any()
    assert label_segments(base.assign(race_class="DEBUT"))["maiden"].astype(bool).all()
    # the debut axis: give h:c an earlier run and p2 has no debut starter
    seasoned = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "race_id": ["jra:p0"],
                    "race_utc": [T0 - pd.Timedelta(days=7)],
                    "horse_id": ["h:c"],
                    "race_class": ["C1"],
                }
            ),
        ],
        ignore_index=True,
    )
    out = label_segments(seasoned)
    p2 = out.loc[seasoned["race_id"] == "jra:p2", "segment"]
    assert set(p2) == {"MAIDEN/seasoned"}
    assert set(out.loc[seasoned["race_id"] == "jra:p1", "segment"]) == {"MAIDEN/debut"}
    assert set(out.loc[seasoned["race_id"] == "jra:p0", "segment"]) == {"C1/debut"}


# ---------------------------------------------------------------------------------------------
# Scoring by segment
# ---------------------------------------------------------------------------------------------


def expected_segment(rows: pd.DataFrame, segment: str, *, top: float, boot: dict) -> dict:
    """Per-segment figures by brute force from the planted rows (never from the module)."""
    sub = rows.loc[rows["segment"] == segment]
    gains, meeting_of = {}, {}
    for race, grp in sub.groupby("race_id", sort=True):
        winners = grp.loc[grp["won"]]
        gains[race] = float(
            np.mean(np.log(winners["p_member"].to_numpy()) - np.log(winners["p_base"].to_numpy()))
        )
        meeting_of[race] = grp["meeting_id"].iloc[0]
    gain = pd.Series(gains, dtype=float)
    iv = paired_bootstrap(gain, pd.Series(meeting_of), **boot)
    claim = (sub["p_member"] * sub["win_odds"]).to_numpy()
    sel = claim >= np.quantile(claim, top)
    ret = pd.Series(100.0 * sub["paid"].to_numpy()[sel])
    tail = paired_bootstrap(ret, pd.Series(sub["meeting_id"].to_numpy()[sel]), **boot)
    return {
        "races": len(gain),
        "meetings": len(set(meeting_of.values())),
        "bets": len(sub),
        "delta_ll": float(gain.mean()),
        "delta_ll_lo": iv.lo,
        "delta_ll_hi": iv.hi,
        "tail_bets": int(sel.sum()),
        "tail_return": float(ret.mean()),
        "tail_lo": tail.lo,
        "tail_hi": tail.hi,
    }


@pytest.mark.parametrize(
    ("seed", "top", "boot"),
    [
        (7, 0.99, {"n_boot": 300, "level": 0.95, "seed": 573}),
        (11, 0.9, {"n_boot": 200, "level": 0.9, "seed": 5}),
        (23, 0.5, {"n_boot": 250, "level": 0.95, "seed": 99}),
    ],
)
def test_one_ranked_row_per_segment_with_ci(seed, top, boot):
    rows = scored_rows(seed, paid_factor=0.9)  # paid is not odds x won: the tail reads `paid`
    out = score_segments(rows, top=top, **boot)
    assert len(out) == len(SEGMENT_PLAN)
    table = out.set_index("segment")
    assert set(table.index) == set(SEGMENT_PLAN)
    assert table.index.is_unique

    for segment in SEGMENT_PLAN:
        want = expected_segment(rows, segment, top=top, boot=boot)
        got = table.loc[segment]
        assert got["races"] == want["races"] == SEGMENT_PLAN[segment][0]
        assert got["meetings"] == want["meetings"]
        assert got["bets"] == want["bets"]
        assert got["tail_bets"] == want["tail_bets"]
        for key in ("delta_ll", "delta_ll_lo", "delta_ll_hi", "tail_return", "tail_lo", "tail_hi"):
            assert got[key] == pytest.approx(want[key], rel=1e-9, abs=1e-9), (segment, key)
        assert got["delta_ll_lo"] <= got["delta_ll"] <= got["delta_ll_hi"]

    # ranked: by delta_ll, highest first, ranks 1..n
    ordered = out.sort_values("rank")
    assert ordered["rank"].tolist() == list(range(1, len(out) + 1))
    assert ordered["delta_ll"].is_monotonic_decreasing
    assert out.loc[out["rank"] == 1, "segment"].iloc[0] == table["delta_ll"].idxmax()


def test_the_planted_lift_ranks_the_debut_segment_first():
    out = score_segments(scored_rows(7), n_boot=200, level=0.95, seed=1)
    first = out.sort_values("rank").iloc[0]
    assert first["segment"] == "DEBUT/debut"
    assert first["delta_ll"] > 0.1
    last = out.sort_values("rank").iloc[-1]
    assert last["segment"] == "MAIDEN/seasoned"
    assert last["delta_ll"] < 0


def test_the_result_does_not_depend_on_row_order_or_mutate_the_input():
    rows = scored_rows(7)
    snapshot = rows.copy(deep=True)
    boot = {"n_boot": 200, "level": 0.95, "seed": 3}
    a = score_segments(rows, **boot)
    pd.testing.assert_frame_equal(rows, snapshot)
    b = score_segments(rows.sample(frac=1.0, random_state=1).reset_index(drop=True), **boot)
    pd.testing.assert_frame_equal(
        a.sort_values("segment").reset_index(drop=True),
        b.sort_values("segment").reset_index(drop=True),
    )


def test_the_seed_and_level_set_the_bounds_not_the_points():
    rows = scored_rows(7)
    a = score_segments(rows, n_boot=300, level=0.95, seed=1).set_index("segment")
    b = score_segments(rows, n_boot=300, level=0.95, seed=2).set_index("segment")
    c = score_segments(rows, n_boot=300, level=0.80, seed=1).set_index("segment")
    for col in ("delta_ll", "tail_return", "races", "bets"):
        pd.testing.assert_series_equal(a[col], b[col])
        pd.testing.assert_series_equal(a[col], c[col])
    assert not np.allclose(a["delta_ll_hi"], b["delta_ll_hi"])
    assert (c["delta_ll_hi"] <= a["delta_ll_hi"] + 1e-12).all()
    assert (c["delta_ll_lo"] >= a["delta_ll_lo"] - 1e-12).all()


def test_tail_top_quantile_changes_the_bets_that_are_counted():
    rows = scored_rows(11)
    boot = {"n_boot": 100, "level": 0.95, "seed": 1}
    wide = score_segments(rows, top=0.5, **boot).set_index("segment")
    narrow = score_segments(rows, top=0.99, **boot).set_index("segment")
    assert (wide["tail_bets"] > narrow["tail_bets"]).all()
    assert (narrow["tail_bets"] >= 1).all()
    default = score_segments(rows, **boot).set_index("segment")
    assert default["tail_bets"].equals(narrow["tail_bets"])


def test_a_hand_worked_two_segment_example():
    """Two segments, hand-computed with `math.log`: A has two races in two meetings, B one."""
    # race, segment, meeting, winner, p_member, p_base (3 runners each), odds, paid
    cards = {
        "a1": ("A", "m1", 0, [0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [2.0, 3.0, 5.0]),
        "a2": ("A", "m2", 1, [0.2, 0.5, 0.3], [0.3, 0.4, 0.3], [4.0, 2.0, 3.0]),
        "b1": ("B", "m3", 2, [0.3, 0.3, 0.4], [0.4, 0.3, 0.3], [3.0, 3.0, 2.0]),
    }
    rows = []
    for race, (segment, meeting, win, pm, pb, odds) in cards.items():
        for k in range(3):
            rows.append(
                {
                    "race_id": race,
                    "segment": segment,
                    "meeting_id": meeting,
                    "won": k == win,
                    "win_odds": odds[k],
                    "paid": odds[k] if k == win else 0.0,
                    "p_member": pm[k],
                    "p_base": pb[k],
                }
            )
    out = score_segments(pd.DataFrame(rows), n_boot=50, level=0.95, seed=0, top=0.99)
    table = out.set_index("segment")
    a = (math.log(0.5 / 0.4) + math.log(0.5 / 0.4)) / 2
    b = math.log(0.4 / 0.3)
    assert table.loc["A", "delta_ll"] == pytest.approx(a, abs=1e-12)
    assert table.loc["B", "delta_ll"] == pytest.approx(b, abs=1e-12)
    assert table.loc["A", "races"] == 2 and table.loc["B", "races"] == 1
    assert table.loc["A", "meetings"] == 2 and table.loc["B", "meetings"] == 1
    assert table.loc["B", "rank"] == 1 and table.loc["A", "rank"] == 2  # b > a
    # claims p x odds: A -> 1.0, 0.9, 1.0 | 0.8, 1.0, 0.9; the top 1% is the claim 1.0 bets,
    # a1/runner 0 (won, pays 2.0), a1/runner 2 (lost), a2/runner 1 (won, pays 2.0)
    claims = [0.5 * 2, 0.3 * 3, 0.2 * 5, 0.2 * 4, 0.5 * 2, 0.3 * 3]
    assert max(claims) == pytest.approx(1.0)
    assert table.loc["A", "tail_bets"] == 3
    assert table.loc["A", "tail_return"] == pytest.approx(100 * (2.0 + 0.0 + 2.0) / 3)
    # B: claims 0.9, 0.9, 0.8; the top is the two 0.9 bets, both losers
    # (runner 2 won, at odds 2.0, with claim 0.8: below the edge)
    assert table.loc["B", "tail_bets"] == 2
    assert table.loc["B", "tail_return"] == pytest.approx(0.0)


def test_a_dead_heat_gains_one_over_k_per_winner_and_a_single_race_segment_keeps_its_row():
    rows = pd.DataFrame(
        {
            "race_id": ["d1"] * 4,
            "segment": ["DEBUT/debut"] * 4,
            "meeting_id": ["m1"] * 4,
            "won": [True, True, False, False],
            "win_odds": [3.0, 5.0, 5.0, 6.0],
            "paid": [3.0, 5.0, 0.0, 0.0],
            "p_member": [0.4, 0.3, 0.2, 0.1],
            "p_base": [0.25, 0.25, 0.25, 0.25],
        }
    )
    out = score_segments(rows, n_boot=20, level=0.95, seed=0)
    assert len(out) == 1
    row = out.iloc[0]
    want = (math.log(0.4 / 0.25) + math.log(0.3 / 0.25)) / 2
    assert row["delta_ll"] == pytest.approx(want, abs=1e-12)
    assert row["races"] == 1 and row["meetings"] == 1 and row["bets"] == 4
    assert row["rank"] == 1
    assert row["delta_ll_lo"] <= row["delta_ll"] <= row["delta_ll_hi"]
    assert row["tail_bets"] == 1  # claims 1.2, 1.5, 1.0, 0.6: the 1.5 (a winner at 5.0) alone
    assert row["tail_return"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------------------------
# The pre-registered spec
# ---------------------------------------------------------------------------------------------


def test_the_spec_is_plain_json_that_freezes_once(tmp_path):
    spec = build_spec(BOOT)
    assert json.loads(json.dumps(spec)) == spec
    assert spec["ticket"] == TICKET == 573
    assert spec["family_id"] == FAMILY_ID
    assert FAMILY_ID not in {"japan-c2-interactions-final-win-v2", "japan-debut-maiden-members-v2"}
    assert spec["bootstrap"] == BOOT
    assert spec["segments"]["axes"] == ["debut", "maiden", "race_class"]
    assert spec["segments"]["maiden_classes"] == ["DEBUT", "MAIDEN"]
    assert spec["tail"] == {"top": 0.99}
    assert debut_prior.build_spec(BOOT) == spec  # deterministic

    registry = Registry(tmp_path / "registry.jsonl", holdout_log=tmp_path / "holdout.jsonl")
    first = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, build_spec(BOOT)) == first
    assert registry.frozen_spec_hash(FAMILY_ID) == first
    other = build_spec({**BOOT, "n_boot": BOOT["n_boot"] + 1})
    assert other != spec  # the bootstrap is part of what is frozen
