"""E37's run (#538, epic #73; 08 §4 E37, split out of #95): the lineage prior fed into #92's
ability member as an as-of feature, on the netkeiba pedigree.

Contract pinned here, in `horsetrack.countries.japan.studies.lineage` (the module #95 started;
the ticket's outline puts the run there):

- `lineage_feature(runs, pedigree, quarter_starts, **prior) -> pd.Series`: one float per row of
  runs, on runs' index. runs: race_id, race_date (the race day), race_utc (tz-aware UTC post time
  on that day), horse_id, finish_pos (NaN: started, not placed), surface and dist_band (other
  columns ignored, rows in any order). pedigree: `models.lineage.LineagePrior`'s (horse_id,
  sire_id, dam_id and, when given, dam_sire_id). quarter_starts: D9's quarter starts as
  `datetime.date`, ascending, the first on or before every row's race_date. prior: LineagePrior's
  keyword arguments (tau_cell, damsire_weight, and k or discounts when given). A row's quarter is
  the last start on or before its race_date; its value is, when its horse has fewer than
  THIN_STARTS = 30 prior starts (its rows in runs on earlier race days: a debut has 0, and a run
  without a placing is a start), the prior mean of `LineagePrior(**prior).fit(runs, pedigree,
  before_utc=<the start at 00:00 UTC>)` at the row's horse, surface and dist_band (no trainer
  effect), and 0.0 otherwise. Every value therefore reads only runs before its row's quarter, for
  the rows that train a fold and the rows it tests alike: changing a later quarter's outcomes, or
  dropping its runs, leaves it as it was, bit for bit.
- `netkeiba_pedigree(table) -> pd.DataFrame`: the LineagePrior pedigree of a netkeiba
  `processed/jra/horse_pedigree` frame (the store schema, `core.store.schemas.TABLES
  ["horse_pedigree"]`, whose id columns are empty): one row per horse_id with sire_id its
  sire_name_norm and dam_sire_id its dam_sire_name_norm (a missing name stays missing), and a
  dam_id column (values not pinned). It raises ValueError when the table carries any column a
  UKC (JRDB) merge adds, `jrdb.pedigree.OUTPUT_COLUMNS` beyond the store schema, whatever the
  column holds (the netkeiba rows of a merged table are null there).
- `read_netkeiba_pedigree() -> pd.DataFrame`: `netkeiba_pedigree` of
  `$HORSETRACK_DATA_DIR/processed/jra/horse_pedigree.parquet`, so the run refuses the file
  `jrdb.tables` writes back there after a UKC merge.
- `FAMILY_ID` (the netkeiba-only family) reads as kind "member" on the leaderboard
  (`research.leaderboard_backfill.legacy_kind`), and `SOURCES`, its records' sources, hold no
  withheld source (D21): its results can be public.
- `horsetrack japan lineage-study` is registered (`--help` exits 0).

Not pinned: rows before the first quarter start; how and how often the priors are fitted; the
distance bands' cut points, the tau_cell grid, the band schemes and the coverage window (the frozen
spec); the AbilityModel fits, folds and level-2 machinery (#92's); the registry records, the
committed summary and the report text; the optional JRA-VAN pedigree variant (its records must
carry a withheld source; left to review). Expected values come from #95's `LineagePrior` (library
code on main) and hand counts in `lineage_run_synth`, never from the code under test. Every RNG
is seeded; no test reads the local stores.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lineage_run_synth import (
    QUARTER_STARTS,
    RUNNERS,
    STORE_COLUMNS,
    THIN,
    UKC_COLUMNS,
    VETERAN,
    asof_runs,
    expected_feature,
    frame,
    netkeiba_table,
    prior_starts,
    quarter_of,
    quarter_priors,
    ukc_table,
    veteran_runs,
)

from horsetrack.core.eval.reports import is_withheld
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.jrdb import pedigree as jrdb_pedigree
from horsetrack.countries.japan.jrdb import tables as jrdb_tables
from horsetrack.countries.japan.models.lineage import LineagePrior
from horsetrack.countries.japan.research.leaderboard_backfill import legacy_kind
from horsetrack.countries.japan.studies import ability, lineage

PRIOR = {"tau_cell": 0.4, "damsire_weight": 0.25}
# Each row differs from the first in one argument, so a builder that drops one fails its row.
PRIORS = (
    PRIOR,
    {"tau_cell": 0.4, "damsire_weight": 0.0},
    {"tau_cell": 1.5, "damsire_weight": 0.25},
    {"tau_cell": 0.4, "damsire_weight": 0.25, "k": 1},
)


def feature(runs: pd.DataFrame, ped: pd.DataFrame, starts, **prior) -> pd.Series:
    """lineage_feature checked for its shape, aligned on runs' index."""
    got = lineage.lineage_feature(runs, ped, starts, **prior)
    assert isinstance(got, pd.Series)
    assert len(got) == len(runs)
    assert got.index.is_unique
    assert set(got.index) == set(runs.index)
    return got.reindex(runs.index).astype(float)


# ---------------------------------------------------------------------------------------------
# DoD: the as-of lineage features use only runs before their quarter
# ---------------------------------------------------------------------------------------------


def test_feature_is_the_prior_fitted_at_the_start_of_each_rows_quarter():
    """Five quarters of runs (the first with no earlier run), rows shuffled onto a non-default
    index, at four prior settings: every row's value is LineagePrior(**prior) fitted on the runs
    before its quarter's start, read at its horse, surface and band, for a runner with fewer than
    30 prior starts, and 0.0 for the two veterans once they have 30. Each setting gives other
    values, so a builder that ignores tau_cell, damsire_weight or k fails."""
    runs, ped = asof_runs(5383)
    rng = np.random.default_rng(5383)
    shuffled = runs.iloc[rng.permutation(len(runs))].copy()
    shuffled.index = pd.Index([f"run{i:05d}" for i in rng.permutation(len(runs))])
    starts = prior_starts(shuffled).to_numpy()
    unmasked = quarter_priors(shuffled, ped, QUARTER_STARTS, PRIOR).to_numpy()
    assert ((starts >= THIN) & (unmasked != 0)).sum() > 20, "the mask should matter here"

    wants = []
    for prior in PRIORS:
        want = expected_feature(shuffled, ped, QUARTER_STARTS, prior)
        got = feature(shuffled, ped, QUARTER_STARTS, **prior)
        np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-6)
        wants.append(want.to_numpy())
    for i, a in enumerate(wants):
        for b in wants[i + 1 :]:
            assert np.abs(a - b).max() > 1e-3, "the fixture should tell the settings apart"


def reversed_places(runs: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """runs with every placing of `rows` reversed within its race (place p -> RUNNERS + 1 - p);
    a run without a placing keeps none."""
    out = runs.copy()
    out.loc[rows, "finish_pos"] = RUNNERS + 1 - runs.loc[rows, "finish_pos"]
    return out


def test_a_later_quarters_outcomes_leave_every_earlier_rows_feature_unchanged():
    """The leakage test. For each quarter k: reversing every finishing order in quarter k leaves
    the feature of every row of quarters up to k (k's own rows included: they read only earlier
    quarters) exactly as it was, and moves some row of quarter k + 1, so the test can fail; and
    dropping every run from quarter k on leaves every remaining row's feature exactly as it was."""
    runs, ped = asof_runs(5381)
    q = quarter_of(runs["race_date"], QUARTER_STARTS)
    base = feature(runs, ped, QUARTER_STARTS, **PRIOR).to_numpy()
    assert (base[q == 1] != 0).sum() > 100, "the fixture's features should carry the prior"

    for k in range(len(QUARTER_STARTS)):
        changed = feature(reversed_places(runs, q == k), ped, QUARTER_STARTS, **PRIOR).to_numpy()
        np.testing.assert_array_equal(changed[q <= k], base[q <= k], err_msg=f"quarter {k}")
        if k + 1 < len(QUARTER_STARTS):
            later = q == k + 1
            assert (changed[later] != base[later]).any(), f"quarter {k} never reaches {k + 1}"
        if k > 0:
            kept = q < k
            cut = feature(runs.loc[kept], ped, QUARTER_STARTS, **PRIOR).to_numpy()
            np.testing.assert_array_equal(cut, base[kept], err_msg=f"runs from quarter {k} dropped")


# ---------------------------------------------------------------------------------------------
# The thin-history mask: fewer than 30 prior starts, debut included; 0 otherwise
# ---------------------------------------------------------------------------------------------


def test_thin_history_mask_keeps_the_prior_below_30_prior_starts_and_zeroes_it_from_30():
    """A veteran runs 34 Saturdays (its sixth run unplaced, still a start) among debut and
    second-start runners. Its 30th run (29 prior starts) carries the third quarter's prior mean;
    its 31st (30 prior starts) is 0.0 though the same quarter's prior mean there is not 0; a
    debut (0 prior starts) carries its prior; prior starts count earlier race days only, so the
    veteran's early runs are not masked by its later ones."""
    runs, ped, starts = veteran_runs(5382)
    prior = {"tau_cell": 0.5, "damsire_weight": 0.25}
    got = feature(runs, ped, starts, **prior)
    unmasked = quarter_priors(runs, ped, starts, prior)
    n = prior_starts(runs)

    vet = runs.index[runs["horse_id"].eq(VETERAN).to_numpy()]
    assert n.loc[vet].tolist() == list(range(34)), "fixture: one veteran run a race day"
    on = {d.date(): i for i, d in zip(vet, runs.loc[vet, "race_date"], strict=True)}
    at29, at30 = on[date(2024, 7, 27)], on[date(2024, 8, 3)]
    assert (n[at29], n[at30]) == (29, 30)
    assert abs(unmasked[at29]) > 0.1 and abs(unmasked[at30]) > 0.1
    assert got[at29] == pytest.approx(unmasked[at29], abs=1e-6)
    assert got[at30] == 0.0
    for i in vet:
        want = unmasked[i] if n[i] < THIN else 0.0
        assert got[i] == pytest.approx(want, abs=1e-6), (runs.loc[i, "race_date"], n[i])

    debut = n.eq(0).to_numpy() & (quarter_of(runs["race_date"], starts) >= 1)
    assert (unmasked[debut].abs() > 0.1).sum() > 10, "fixture: debuts by a sire with a prior"
    np.testing.assert_allclose(got[debut], unmasked[debut], rtol=0, atol=1e-6)
    want = np.where(n.to_numpy() < THIN, unmasked.to_numpy(), 0.0)
    np.testing.assert_allclose(got.to_numpy(), want, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# The netkeiba-only pedigree: keyed on names, a UKC-merged table refused
# ---------------------------------------------------------------------------------------------

HORSES = [  # (horse_id, sire name, dam name, damsire name); None: missing
    ("2019100001", "Deep Impact", "Dam Alpha", "Sunday Silence"),
    ("2019100002", "Lord Kanaloa", "Dam Beta", "Deep Impact"),
    ("2019100003", "Deep Impact", "Dam Gamma", "Kingmambo"),
    ("2019100004", "Kizuna", "Dam Delta", None),
    ("2019100005", None, "Dam Epsilon", "Lord Kanaloa"),
]
UKC_HORSES = [("2018100009", "Orfevre", "Dam Zeta", "Mejiro McQueen")]


def missing(v) -> bool:
    return v is None or bool(pd.isna(v))


def assert_keyed_on_names(got: pd.DataFrame, table: pd.DataFrame) -> None:
    """One row per horse with sire_id the sire_name_norm and dam_sire_id the
    dam_sire_name_norm, a missing name missing; a pedigree LineagePrior accepts, whose sires are
    those names."""
    assert isinstance(got, pd.DataFrame)
    assert {"horse_id", "sire_id", "dam_id", "dam_sire_id"} <= set(got.columns)
    assert got["horse_id"].is_unique
    assert set(got["horse_id"]) == set(table["horse_id"])
    by = got.set_index("horse_id")
    for row in table.itertuples(index=False):
        for key, name in (("sire_id", row.sire_name_norm), ("dam_sire_id", row.dam_sire_name_norm)):
            value = by.loc[row.horse_id, key]
            assert (missing(value) and name is None) or value == name, (row.horse_id, key, value)

    rows = [
        (f"jra:20240106r{i}", date(2024, 1, 6), i, h, p, "turf", "mile")
        for i in range(2)
        for p, h in enumerate(table["horse_id"], start=1)
    ]
    fitted = LineagePrior(tau_cell=0.5).fit(
        frame(rows), got, before_utc=pd.Timestamp("2024-02-01", tz="UTC")
    )
    names = {n for n in table["sire_name_norm"] if n is not None}
    assert set(fitted.sire_effects_.index) == names


def test_netkeiba_pedigree_keys_sires_and_damsires_on_their_names():
    """The store schema with every id column empty: the sire and the damsire are their
    normalised names, so a stallion that is one horse's sire and another's damsire is one key."""
    table = netkeiba_table(HORSES)
    assert set(table.columns) == set(STORE_COLUMNS)
    assert table["sire_id"].isna().all() and table["dam_sire_id"].isna().all()
    assert_keyed_on_names(lineage.netkeiba_pedigree(table), table)


def ukc_variants() -> dict[str, pd.DataFrame]:
    """Netkeiba tables carrying UKC columns: each UKC column alone (all null, as on a merged
    table's netkeiba rows) and the table `jrdb.pedigree.merge_pedigree` builds."""
    assert set(UKC_COLUMNS) == {
        "sire_line_code",
        "dam_sire_line_code",
        "sire_birth_year",
        "dam_birth_year",
        "dam_sire_birth_year",
        "paternal_grandsire_name_norm",
        "paternal_great_grandsire_name_norm",
    }
    out = {}
    for column in UKC_COLUMNS:
        table = netkeiba_table(HORSES)
        table[column] = None
        out[f"{column} (null)"] = table
    out["merge_pedigree"] = jrdb_pedigree.merge_pedigree(
        ukc_table(UKC_HORSES), netkeiba_table(HORSES)
    )
    return out


@pytest.mark.parametrize("variant", sorted(ukc_variants()))
def test_netkeiba_guard_refuses_a_table_carrying_a_ukc_column(variant):
    table = ukc_variants()[variant]
    with pytest.raises(ValueError):
        lineage.netkeiba_pedigree(table)


def test_the_run_reads_the_store_pedigree_through_the_guard(tmp_path, monkeypatch):
    """`read_netkeiba_pedigree` reads processed/jra/horse_pedigree.parquet under the data dir:
    the netkeiba build (the store's Arrow schema) is keyed on names; the file a UKC merge writes
    back there (`jrdb.tables.PEDIGREE_SCHEMA`) is refused."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    path = tmp_path / "data" / "processed" / "jra" / "horse_pedigree.parquet"
    path.parent.mkdir(parents=True)
    table = netkeiba_table(HORSES)
    pq.write_table(
        pa.Table.from_pandas(table, schema=arrow_schema("horse_pedigree"), preserve_index=False),
        path,
    )
    assert_keyed_on_names(lineage.read_netkeiba_pedigree(), table)

    merged = jrdb_pedigree.merge_pedigree(ukc_table(UKC_HORSES), table)
    frame_ = merged.reindex(columns=jrdb_tables.PEDIGREE_SCHEMA.names)
    pq.write_table(
        pa.Table.from_pandas(frame_, schema=jrdb_tables.PEDIGREE_SCHEMA, preserve_index=False),
        path,
    )
    with pytest.raises(ValueError):
        lineage.read_netkeiba_pedigree()


# ---------------------------------------------------------------------------------------------
# DoD: every variant a ranked member row; only withheld variants withheld
# ---------------------------------------------------------------------------------------------


def test_the_netkeiba_family_is_a_public_member_family():
    """The family (its own, not #92's) reads as kind 'member' on the leaderboard, and its
    records' sources hold no withheld source (D21), so its results can be public."""
    assert lineage.FAMILY_ID not in {ability.FAMILY_ID, ability.TRIAL_FAMILY_ID, ability.SUPERSEDES}
    assert legacy_kind(lineage.FAMILY_ID) == "member"
    assert not is_withheld(lineage.SOURCES)


def test_lineage_study_command_is_registered():
    from typer.testing import CliRunner

    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "lineage-study", "--help"])
    assert result.exit_code == 0, result.output
