"""The paired comparison of two day_sd modes in `track_bias_report` (ticket #256).

Contract pinned here, on top of #138's and #224's (test_track_bias.py and
test_track_bias_day_variance.py, unchanged):

- `track_bias_report(..., versus=...)`: `versus` is any `DaySd` value ("per_level", "pooled",
  None (= pooled), a float, a per-level mapping); its default is the module-level sentinel
  `NO_VERSUS`, because None already means "pooled". Without `versus` the output has no new key.
- With `versus` each fold also fits a prior with `day_sd=versus` on the same `train_end`; the
  "update" variant is built and scored with it (its own ridge conditional logit on the same
  training rows) on exactly the same races. The output gains `ll_update_versus` (mean winner log
  loss per race), `gain_update_vs_versus` (the `_bootstrap` dict of the per-race difference
  ll_update_versus - ll_update, meetings resampled with the same n_boot and seed; positive =
  `day_sd`'s update beats `versus`'s) and `day_sd_versus` (per fold, per level, the versus
  priors' sds, the shape of `day_sd`). Every existing key is identical to a run without `versus`.

Data: synthetic only (day_variance_store.py), under tmp_path; nothing reads $HORSETRACK_DATA_DIR.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date

import pytest

from day_variance_store import planted_history
from handicap_store import patch_holdout, weekend_days
from horsetrack.core.eval.folds import FoldScheme
from horsetrack.countries.japan.handicap import track_bias
from horsetrack.countries.japan.handicap.track_bias_report import NO_VERSUS, track_bias_report

MIDPACK = ("style", "midpack")
LEVELS = [("draw", lv) for lv in ("inner", "middle", "outer")] + [
    ("style", lv) for lv in ("front", "stalker", "midpack", "closer")
]
WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=4, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip
N_BOOT = 50
SEED = 3
GAIN_KEYS = {"mean", "lower_95_one_sided", "ci_95", "races", "meetings"}
NEW_KEYS = {"ll_update_versus", "gain_update_vs_versus", "day_sd_versus"}

SAME_MAPPING = {key: 0.05 for key in LEVELS}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Every report the tests compare, on one small store: [(day_sd, versus)] -> report."""
    days = weekend_days(date(2025, 1, 4), date(2025, 3, 2))
    s, _, _ = planted_history(5, days, target=MIDPACK, sd=0.10, races=3, mode="pinned")
    with pytest.MonkeyPatch.context() as mp:
        s.write(tmp_path_factory.mktemp("store"), mp)
        patch_holdout(mp, [], track_bias)

        def run(day_sd, **kw):
            return track_bias_report(days[0], days[-1], scheme=WEEKLY, n_boot=N_BOOT, seed=SEED,
                                     day_sd=day_sd, **kw)  # fmt: skip

        out = {
            "plain_per_level": run("per_level"),
            "plain_pooled": run("pooled"),
            "plain_small": run(0.01),
            "plain_large": run(0.2),
            "pl_vs_pooled": run("per_level", versus="pooled"),
            "pooled_vs_pl": run("pooled", versus="per_level"),
            "pooled_vs_pooled": run("pooled", versus="pooled"),
            "small_vs_large": run(0.01, versus=0.2),
            "large_vs_small": run(0.2, versus=0.01),
            "pl_vs_none": run("per_level", versus=None),
            "pl_vs_sentinel": run("per_level", versus=NO_VERSUS),
            "float_vs_mapping": run(0.05, versus=SAME_MAPPING),
        }
    return out


def paired(rep: dict) -> dict:
    return rep["gain_update_vs_versus"]


# ---------------------------------------------------------------------------------------------
# Broad: the paired gain is the per-race difference of two runs on the same races
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("paired_run", "own", "other"),
    [
        ("pl_vs_pooled", "plain_per_level", "plain_pooled"),
        ("pooled_vs_pl", "plain_pooled", "plain_per_level"),
        ("small_vs_large", "plain_small", "plain_large"),
        ("large_vs_small", "plain_large", "plain_small"),
    ],
)
def test_paired_gain_is_the_difference_of_the_two_modes_runs(runs, paired_run, own, other):
    rep = runs[paired_run]
    # ll_update_versus is the versus mode's own mean log loss; the report's ll_update is day_sd's
    assert rep["ll_update_versus"] == pytest.approx(runs[other]["ll_update"], abs=1e-9)
    assert rep["ll_update"] == pytest.approx(runs[own]["ll_update"], abs=1e-9)
    want = runs[other]["ll_update"] - runs[own]["ll_update"]
    assert paired(rep)["mean"] == pytest.approx(want, abs=1e-9)
    assert paired(rep)["mean"] == pytest.approx(rep["ll_update_versus"] - rep["ll_update"],
                                                abs=1e-9)  # fmt: skip


@pytest.mark.parametrize(
    ("a", "b"), [("pl_vs_pooled", "pooled_vs_pl"), ("small_vs_large", "large_vs_small")]
)
def test_swapping_the_modes_negates_the_gain_and_its_interval(runs, a, b):
    fwd, rev = paired(runs[a]), paired(runs[b])
    assert fwd["mean"] == pytest.approx(-rev["mean"], abs=1e-12)
    # the same seed draws the same meetings, so the interval is mirrored exactly
    assert fwd["ci_95"][0] == pytest.approx(-rev["ci_95"][1], abs=1e-12)
    assert fwd["ci_95"][1] == pytest.approx(-rev["ci_95"][0], abs=1e-12)
    assert fwd["races"] == rev["races"] and fwd["meetings"] == rev["meetings"]


@pytest.mark.parametrize("name", ["pl_vs_pooled", "small_vs_large"])
def test_two_different_modes_give_a_nonzero_gain_with_a_proper_interval(runs, name):
    rep = runs[name]
    g = paired(rep)
    assert set(g) == GAIN_KEYS
    assert math.isfinite(g["mean"]) and abs(g["mean"]) > 1e-6
    lo, hi = g["ci_95"]
    assert lo < hi
    assert lo <= g["mean"] <= hi
    assert lo <= g["lower_95_one_sided"] <= hi
    assert g["races"] == rep["races"] > 50
    assert g["meetings"] == rep["meetings"] > 1
    assert g["meetings"] == rep["gain_update_vs_prior"]["meetings"]


# ---------------------------------------------------------------------------------------------
# The same mode on both sides is exactly no difference
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["pooled_vs_pooled", "float_vs_mapping"])
def test_the_same_mode_on_both_sides_gives_zero_exactly(runs, name):
    rep = runs[name]
    g = paired(rep)
    assert g["mean"] == 0.0
    assert g["ci_95"] == (0.0, 0.0)
    assert g["lower_95_one_sided"] == 0.0
    assert g["races"] == rep["races"]
    assert g["meetings"] == rep["meetings"]
    assert rep["ll_update_versus"] == pytest.approx(rep["ll_update"], abs=1e-12)


def test_none_as_versus_is_pooled_and_is_not_the_default(runs):
    none, pooled, plain = runs["pl_vs_none"], runs["pl_vs_pooled"], runs["plain_per_level"]
    assert NEW_KEYS <= set(none)
    assert paired(none)["mean"] == pytest.approx(paired(pooled)["mean"], abs=1e-12)
    assert none["ll_update_versus"] == pytest.approx(pooled["ll_update_versus"], abs=1e-12)
    # the sentinel is "no versus": nothing is added
    assert not NEW_KEYS & set(runs["pl_vs_sentinel"])
    assert set(runs["pl_vs_sentinel"]) == set(plain)


# ---------------------------------------------------------------------------------------------
# Output shape: nothing changes without versus; the existing keys do not move with it
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plain_per_level", "plain_pooled", "plain_small"])
def test_without_versus_there_are_no_new_keys(runs, name):
    assert not NEW_KEYS & set(runs[name])


@pytest.mark.parametrize(
    ("paired_run", "plain"),
    [
        ("pl_vs_pooled", "plain_per_level"),
        ("pooled_vs_pl", "plain_pooled"),
        ("small_vs_large", "plain_small"),
    ],
)
def test_existing_keys_are_identical_with_versus(runs, paired_run, plain):
    with_versus, without = runs[paired_run], runs[plain]
    assert NEW_KEYS <= set(with_versus)
    assert set(with_versus) - NEW_KEYS == set(without)
    for key, value in without.items():
        assert with_versus[key] == value, key


@pytest.mark.parametrize(
    ("paired_run", "versus_plain", "own_plain"),
    [
        ("pl_vs_pooled", "plain_pooled", "plain_per_level"),
        ("small_vs_large", "plain_large", "plain_small"),
    ],
)
def test_day_sd_versus_is_the_versus_modes_priors_per_fold_and_level(
    runs, paired_run, versus_plain, own_plain
):
    rep = runs[paired_run]
    assert rep["day_sd_versus"] == runs[versus_plain]["day_sd"]
    assert rep["day_sd"] == runs[own_plain]["day_sd"]
    assert len(rep["day_sd_versus"]) == len(rep["day_sd"]) > 1
    for fold in rep["day_sd_versus"]:
        assert set(fold) == {f"{f}:{lv}" for f, lv in LEVELS}
    # the modes really differ in what they fit
    assert rep["day_sd_versus"] != rep["day_sd"]
