"""Japan's fold scheme loaded from config (ticket #70, outline step 2; D9 and the owner's answers).

The scheme lives in the country's packaged config,
`src/horsetrack/countries/<country>/config/folds.yaml` (read through `load_country_yaml`, where
main loads every country config), under a top-level `folds:` section whose keys are exactly the
FoldScheme fields. `load_scheme(path)` reads that section from a file and
`load_country_scheme(country)` from the country's `folds` config; both refuse missing or unknown
keys (through `FoldScheme.from_mapping`). Other top-level sections are left alone.

Japan, signed: quarters anchored on January, one quarter scored per fold, one quarter step, an
expanding window, a 7-day embargo, and 22 quarters of minimum training, so that training runs from
the first data day (2014-06-07) through 2019 and the first scored quarter is 2020-Q1.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.core import config as core_config

# The country loader is reached through the module so that, before it exists, only the tests that
# use it fail (an AttributeError naming it) while the file-loader tests still run.
from horsetrack.core.eval import folds as folds_module
from horsetrack.core.eval.folds import FoldScheme, load_scheme, walk_forward

REPO_ROOT = Path(__file__).resolve().parents[3]
JAPAN_CONFIG = REPO_ROOT / "src" / "horsetrack" / "countries" / "japan" / "config" / "folds.yaml"
OLD_JAPAN_CONFIG = REPO_ROOT / "config" / "japan" / "evaluation.yaml"
JAPAN_DATA_START = date(2014, 6, 7)  # a Saturday
FIRST_SCORED_QUARTER = date(2020, 1, 1)

JAPAN_SCHEME = {
    "unit": "quarter",
    "test_length": 1,
    "step": 1,
    "min_train": 22,
    "window": "expanding",
    "embargo_days": 7,
    "anchor_month": 1,
}


def folds_yaml(fields: dict, extra: str = "") -> str:
    """A config file's text with a `folds:` section, written by hand (no yaml library needed)."""
    lines = ["folds:"] + [f"  {k}: {v}" for k, v in fields.items()]
    return extra + "\n".join(lines) + "\n"


# --- the loader, on varied files ------------------------------------------------------------------

LOADER_CASES = {
    "japan-values": (JAPAN_SCHEME, ""),
    "monthly-rolling": (
        {
            "unit": "month",
            "test_length": 2,
            "step": 3,
            "min_train": 12,
            "window": "rolling",
            "embargo_days": 0,
            "anchor_month": 4,
        },
        "",
    ),
    "yearly-with-other-sections": (
        {
            "unit": "year",
            "test_length": 1,
            "step": 1,
            "min_train": 3,
            "window": "expanding",
            "embargo_days": 14,
            "anchor_month": 9,
        },
        "# evaluation settings\nmetrics:\n  primary: log_loss\n  bootstrap: 1000\n",
    ),
    "weekly": (
        {
            "unit": "week",
            "test_length": 4,
            "step": 4,
            "min_train": 52,
            "window": "rolling",
            "embargo_days": 3,
            "anchor_month": 1,
        },
        "",
    ),
}


@pytest.mark.parametrize("case", LOADER_CASES)
def test_load_scheme_reads_the_folds_section(tmp_path, case):
    fields, extra = LOADER_CASES[case]
    path = tmp_path / "evaluation.yaml"
    path.write_text(folds_yaml(fields, extra))
    assert load_scheme(path) == FoldScheme(**fields)


@pytest.mark.parametrize("missing", list(JAPAN_SCHEME))
def test_load_scheme_refuses_a_missing_field(tmp_path, missing):
    fields = {k: v for k, v in JAPAN_SCHEME.items() if k != missing}
    path = tmp_path / "evaluation.yaml"
    path.write_text(folds_yaml(fields))
    with pytest.raises(ValueError, match=missing):
        load_scheme(path)


def test_load_scheme_refuses_an_unknown_field(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text(folds_yaml({**JAPAN_SCHEME, "season_start": 1}))
    with pytest.raises(ValueError, match="season_start"):
        load_scheme(path)


# --- the country loader, on varied country configs ------------------------------------------------


@pytest.fixture
def countries_dir(tmp_path, monkeypatch):
    """A stand-in for src/horsetrack/countries, so a country's config/folds.yaml can be written."""
    monkeypatch.setattr(core_config, "_COUNTRIES", tmp_path)
    core_config.load_country_yaml.cache_clear()
    yield tmp_path
    core_config.load_country_yaml.cache_clear()


def write_country_folds(countries: Path, country: str, text: str) -> None:
    path = countries / country / "config" / "folds.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text)


@pytest.mark.parametrize("case", LOADER_CASES)
def test_load_country_scheme_reads_the_countrys_folds_config(countries_dir, case):
    fields, extra = LOADER_CASES[case]
    country = f"land-{case}"  # a distinct country per case, so no cached answer can carry over
    write_country_folds(countries_dir, country, folds_yaml(fields, extra))
    # a decoy beside it: the scheme comes from folds.yaml, not another file of the country
    decoy = {**JAPAN_SCHEME, "embargo_days": 99}
    (countries_dir / country / "config" / "evaluation.yaml").write_text(folds_yaml(decoy))
    assert folds_module.load_country_scheme(country) == FoldScheme(**fields)


@pytest.mark.parametrize("missing", ["unit", "embargo_days", "anchor_month"])
def test_load_country_scheme_refuses_a_missing_field(countries_dir, missing):
    country = f"missing-{missing}"
    fields = {k: v for k, v in JAPAN_SCHEME.items() if k != missing}
    write_country_folds(countries_dir, country, folds_yaml(fields))
    with pytest.raises(ValueError, match=missing):
        folds_module.load_country_scheme(country)


def test_load_country_scheme_refuses_an_unknown_field(countries_dir):
    write_country_folds(countries_dir, "unknown", folds_yaml({**JAPAN_SCHEME, "season_start": 1}))
    with pytest.raises(ValueError, match="season_start"):
        folds_module.load_country_scheme("unknown")


BARE_CONFIGS = {
    "empty": "",
    "no-folds-section": "metrics:\n  primary: log_loss\n",
    "folds-not-a-mapping": "folds: quarterly\n",
}


@pytest.mark.parametrize("case", BARE_CONFIGS)
def test_load_country_scheme_refuses_a_config_without_a_folds_mapping(countries_dir, case):
    write_country_folds(countries_dir, f"bare-{case}", BARE_CONFIGS[case])
    with pytest.raises(ValueError, match="folds"):
        folds_module.load_country_scheme(f"bare-{case}")


# --- Japan's signed scheme ------------------------------------------------------------------------


def test_the_japan_config_holds_the_signed_scheme():
    assert JAPAN_CONFIG.is_file(), f"missing {JAPAN_CONFIG.relative_to(REPO_ROOT)}"
    assert load_scheme(JAPAN_CONFIG) == FoldScheme(**JAPAN_SCHEME)


def test_the_japan_scheme_loads_through_the_country_loader():
    assert folds_module.load_country_scheme("japan") == FoldScheme(**JAPAN_SCHEME)


def test_the_japan_scheme_has_one_home():
    # a copy left at the old repo-root path would be a second, silently diverging source of truth
    assert not OLD_JAPAN_CONFIG.exists(), (
        f"{OLD_JAPAN_CONFIG.relative_to(REPO_ROOT)} should have moved to "
        f"{JAPAN_CONFIG.relative_to(REPO_ROOT)}"
    )


def weekly_race_dates(start: date, end: date) -> pd.Series:
    n = (end - start).days // 7 + 1
    return pd.Series(pd.to_datetime([start + timedelta(weeks=i) for i in range(n)]))


@pytest.mark.parametrize("end", [date(2020, 3, 28), date(2023, 6, 24), date(2026, 11, 21)])
def test_japan_scheme_first_scores_2020_q1_after_training_on_2014_to_2019(end):
    dates = weekly_race_dates(JAPAN_DATA_START, end)
    py = [d.date() for d in dates]
    folds = walk_forward(dates, scheme=folds_module.load_country_scheme("japan"))
    assert folds, "the Japan scheme gave no folds"

    first = folds[0]
    assert first.test_start == FIRST_SCORED_QUARTER
    assert first.test_end == date(2020, 3, 31)
    test_dates = [py[i] for i in first.test_idx]
    assert test_dates == [d for d in py if FIRST_SCORED_QUARTER <= d <= date(2020, 3, 31)]

    train_dates = [py[i] for i in first.train_idx]
    assert train_dates[0] == JAPAN_DATA_START  # expanding: from the first data day, nothing earlier
    assert all(JAPAN_DATA_START <= d < FIRST_SCORED_QUARTER for d in train_dates)
    # the 7-day embargo runs back from the first scored race day
    assert max(train_dates) <= min(test_dates) - timedelta(days=7)
    assert train_dates == [d for d in py if d <= min(test_dates) - timedelta(days=7)]

    # every later window is a calendar quarter, one after another, through the last race day
    starts = [f.test_start for f in folds]
    expected, q = [], FIRST_SCORED_QUARTER
    while q <= py[-1]:
        expected.append(q)
        q = date(q.year + (q.month == 10), (q.month + 2) % 12 + 1, 1)
    assert starts == expected
