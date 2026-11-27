"""The board-vs-final run's guards (ticket #103, review round 1): where its outputs may go, E45's
Place and Wide from each race's paid record, and a run whose tables cannot be produced.

Contract pinned here (`horsetrack.countries.japan.studies.board_vs_final`):

- `output_dir(out)`: the run's output directory, resolved. None is `$HORSETRACK_DATA_DIR/reports`;
  a directory given must resolve (`..`, symlinks, the cwd) under the data directory, else
  `OutsideDataDir`; one inside the repository is refused too. `main(["--run", "--out", DIR])`
  refused that way returns 2 before the family is frozen: nothing is written anywhere and the
  registry holds no entry.
- `payout_rules(rows, paid)`: per race of `rows` (one runner each in first, second and third),
  from its non-refund rows of `paid` (the store's `dividends`: race_id, pool, combination
  "4" | "4,7", dividend, is_refund): `places` (the Place places scored, 0 for none), `wide`
  (whether Wide is scored) and each pool's status (`place_status`, `wide_status`): SCORED; NO_RECORD
  for a race without a paid Win (neither pool scored); NOT_PAID for a race without a paid row of
  the pool; MISMATCH when the paid combinations are not those of the result (a combination that
  does not parse included), or Place pays a count JRA never pays. Place's places are the count of
  paid Place horses, never the field of starters: a late scratch from eight to seven still pays
  three.
- `pool_loglik(t, finish, *, places, wide)`: one race's log-likelihood per pool from the trifecta
  array `t`, Place on the given `places` (NaN for 0), Wide only when `wide` (else NaN), Trio always.
- A run (`main(["--run", ...])`) in which a required table cannot be produced, here a JRDB build
  without TYB deliveries: every table's failure is a `failed` registry record (one per price
  fitted on), the fixed-name outputs of the last complete run stay byte-identical, the report goes
  to an attempt-named file alone, and the exit is 1. An identical rerun appends nothing and returns
  the held records' own run ids.

All data is synthetic: no JRDB file or value. Expected values come from the hand-written fixtures
and from sums over the trifecta array written out here, never from the code under test.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval import registry as registry_mod
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.models.exotic import PLACE, TRIO, WIDE, WIN
from horsetrack.countries.japan.studies import board_vs_final as bvf
from horsetrack.countries.japan.studies.board_vs_final import (
    MISMATCH,
    NO_RECORD,
    NOT_PAID,
    SCORED,
    OutsideDataDir,
    cross_apply,
    main,
    output_dir,
    payout_rules,
    pool_loglik,
    post_times,
    read_development,
    write_outputs,
)

FINAL, BOARD = "q_final", "q_board"
ALL_KEYS = {("E2", WIN), ("E45", PLACE), ("E45", WIDE), ("E45", TRIO)}
STATUSES = (SCORED, NO_RECORD, NOT_PAID, MISMATCH)


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """A fresh $HORSETRACK_DATA_DIR, and the committed summary redirected under tmp_path."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    summary = tmp_path / "summary"
    monkeypatch.setattr(bvf, "summary_dir", lambda country: summary)
    monkeypatch.setattr(registry_mod, "summary_dir", lambda country: summary)
    return data


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


# ---------------------------------------------------------------------------------------------
# The output directory
# ---------------------------------------------------------------------------------------------


def outside_of(data: Path, tmp_path: Path, where: str) -> Path:
    if where == "elsewhere":
        return tmp_path / "elsewhere" / "board-vs-final"
    if where == "sibling-sharing-a-prefix":  # a string-prefix check would take it
        return Path(str(data) + "-other") / "reports"
    if where == "escapes-by-dotdot":
        return data / "reports" / ".." / ".." / "escaped"
    if where == "symlink-out-of-the-data-dir":
        target = tmp_path / "target"
        target.mkdir()
        (data / "linked").symlink_to(target, target_is_directory=True)
        return data / "linked" / "board-vs-final"
    raise AssertionError(where)


OUTSIDE = [
    "elsewhere",
    "sibling-sharing-a-prefix",
    "escapes-by-dotdot",
    "symlink-out-of-the-data-dir",
]


@pytest.mark.parametrize("where", OUTSIDE)
def test_an_out_dir_outside_the_data_dir_is_refused(data_dir, tmp_path, where):
    out = outside_of(data_dir, tmp_path, where)
    with pytest.raises(OutsideDataDir):
        output_dir(out)
    assert not out.exists() or files_under(out) == []


def test_an_out_dir_inside_the_repository_is_refused(data_dir):
    inside = env.repo_root() / f".tmp-board-vs-final-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises((paths.DataDirInsideRepo, LicensedOutputInRepo, OutsideDataDir)):
            output_dir(inside)
        assert files_under(inside) == []
    finally:
        shutil.rmtree(inside, ignore_errors=True)


@pytest.mark.parametrize(
    "given",
    ["default", "reports/board-vs-final", "studies/a/b", "relative-to-the-cwd", "symlink-in"],
)
def test_an_out_dir_under_the_data_dir_is_accepted(data_dir, tmp_path, monkeypatch, given):
    if given == "default":
        out, want = None, data_dir / "reports"
    elif given == "relative-to-the-cwd":
        monkeypatch.chdir(data_dir)
        out, want = Path("reports") / "rel", data_dir / "reports" / "rel"
    elif given == "symlink-in":  # spelled outside the data dir, resolving inside it
        (data_dir / "reports").mkdir()
        link = tmp_path / "into-the-data-dir"
        link.symlink_to(data_dir / "reports", target_is_directory=True)
        out, want = link / "bvf", data_dir / "reports" / "bvf"
    else:
        out = want = data_dir / given
    assert Path(output_dir(out)).resolve() == want.resolve()


@pytest.mark.parametrize("where", ["elsewhere", "escapes-by-dotdot", "inside-the-repository"])
def test_the_run_refuses_an_out_dir_outside_the_data_dir_before_the_freeze(
    data_dir, tmp_path, capsys, where
):
    if where == "inside-the-repository":
        out = env.repo_root() / f".tmp-board-vs-final-{uuid.uuid4().hex[:8]}"
    else:
        out = outside_of(data_dir, tmp_path, where)
    try:
        code = main(["--run", "--out", str(out)])
        assert code == 2
        assert files_under(out) == [], "a refused --out is never written"
    finally:
        if where == "inside-the-repository":
            shutil.rmtree(out, ignore_errors=True)
    assert files_under(data_dir) == [], "refused before the freeze: no registry entry, no output"
    assert files_under(tmp_path / "summary") == []
    assert capsys.readouterr().out.strip() == "", "a refused run prints no result"


# ---------------------------------------------------------------------------------------------
# E45's places and Wide from each race's paid record
# ---------------------------------------------------------------------------------------------


def wide_pairs(first3) -> list[str]:
    return [f"{a},{b}" for a, b in itertools.combinations(sorted(first3), 2)]


def paid_rows(win=None, place=(), wide=(), refunds=()) -> list[tuple[str, str, bool]]:
    rows = [] if win is None else [(WIN, str(win), False)]
    rows += [(PLACE, str(h), False) for h in place]
    rows += [(WIDE, c, False) for c in wide]
    rows += [(pool, c, True) for pool, c in refunds]
    return rows


def race(starters, first3, paid, want):
    return {"starters": list(starters), "first3": tuple(first3), "paid": paid, "want": want}


# want: (places, wide, place_status, wide_status)
PAYOUT_RACES = {
    # sold to eight, one scratched late: seven run and Place still pays three
    "late-scratch-8-to-7": race(
        [1, 2, 3, 4, 6, 7, 8],
        (6, 2, 8),
        paid_rows(6, (6, 2, 8), wide_pairs((6, 2, 8)), refunds=[(WIN, "5"), (PLACE, "5")]),
        (3, True, SCORED, SCORED),
    ),
    "seven-at-sale": race(
        range(1, 8),
        (3, 7, 1),
        paid_rows(3, (3, 7), wide_pairs((3, 7, 1))),
        (2, True, SCORED, SCORED),
    ),
    "fourteen-runners": race(
        range(1, 15),
        (10, 4, 13),
        paid_rows(10, (13, 10, 4), wide_pairs((10, 4, 13))),
        (3, True, SCORED, SCORED),
    ),
    "four-runners-place-not-sold": race(
        range(1, 5), (2, 4, 1), paid_rows(2, (), wide_pairs((2, 4, 1))), (0, True, NOT_PAID, SCORED)
    ),
    "wide-not-paid": race(
        range(1, 11), (7, 8, 9), paid_rows(7, (7, 8, 9)), (3, False, SCORED, NOT_PAID)
    ),
    "no-paid-win": race(
        range(1, 10),
        (9, 3, 5),
        paid_rows(None, (9, 3, 5), wide_pairs((9, 3, 5))),
        (0, False, NO_RECORD, NO_RECORD),
    ),
    "win-refunded-only": race(
        range(1, 9),
        (1, 2, 3),
        paid_rows(None, (1, 2, 3), wide_pairs((1, 2, 3)), refunds=[(WIN, "1")]),
        (0, False, NO_RECORD, NO_RECORD),
    ),
    "place-paid-a-fourth-finisher": race(
        range(1, 10),
        (4, 5, 6),
        paid_rows(4, (4, 5, 7), wide_pairs((4, 5, 6))),
        (0, True, MISMATCH, SCORED),
    ),
    "place-one-horse-paid": race(
        range(1, 9),
        (2, 3, 4),
        paid_rows(2, (2,), wide_pairs((2, 3, 4))),
        (0, True, MISMATCH, SCORED),
    ),
    "wide-a-wrong-pair": race(
        range(1, 9),
        (1, 6, 8),
        paid_rows(1, (1, 6, 8), ["1,6", "1,8", "6,7"]),
        (3, False, SCORED, MISMATCH),
    ),
    "wide-a-pair-missing": race(
        range(1, 12),
        (11, 2, 5),
        paid_rows(11, (11, 2, 5), ["2,11", "5,11"]),
        (3, False, SCORED, MISMATCH),
    ),
    "place-unparsable": race(
        range(1, 9),
        (3, 1, 2),
        paid_rows(3, (3, 1, "x2"), wide_pairs((3, 1, 2))),
        (0, True, MISMATCH, SCORED),
    ),
    "wide-unparsable": race(
        range(1, 7),
        (5, 1, 2),
        paid_rows(5, (5, 1), ["1-5", "2,5", "1,2"]),
        (2, False, SCORED, MISMATCH),
    ),
}
ELSEWHERE = "jra:2025-07-06:FUK:12"  # paid rows of a race not in `rows`


def payout_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`rows` as the study frame holds them (race_id, horse_no, finish_pos; one non-finisher) and
    `paid` as the store's dividends table, for PAYOUT_RACES."""
    runners, paid = [], []
    for k, (name, spec) in enumerate(PAYOUT_RACES.items()):
        race_id = f"jra:2025-07-06:FUK:{k + 1:02d}"
        rest = [h for h in spec["starters"] if h not in spec["first3"]]
        position = {h: float(i + 1) for i, h in enumerate([*spec["first3"], *rest])}
        if name == "fourteen-runners":
            position[rest[-1]] = math.nan  # a starter who did not finish
        for h in spec["starters"]:
            runners.append({"race_id": race_id, "horse_no": h, "finish_pos": position[h]})
        for pool, combination, refund in spec["paid"]:
            row = {"race_id": race_id, "pool": pool, "combination": combination}
            paid.append(row | {"dividend": 150.0 + 10 * len(paid), "is_refund": refund})
        paid.append(
            {"race_id": race_id, "pool": TRIO, "combination": "1,2,3", "dividend": 990.0}
            | {"is_refund": False}
        )
    for pool, combination in ((WIN, "1"), (PLACE, "1"), (PLACE, "2"), (PLACE, "3")):
        row = {"race_id": ELSEWHERE, "pool": pool, "combination": combination}
        paid.append(row | {"dividend": 120.0, "is_refund": False})
    rows = pd.DataFrame(runners).astype({"race_id": object, "horse_no": "int64"})
    paid = pd.DataFrame(paid)
    paid.insert(0, "jurisdiction", "jra")
    return rows, paid


def test_e45_scores_place_and_wide_by_each_races_paid_combinations():
    assert len(set(STATUSES)) == 4, "the four statuses are told apart"
    rows, paid = payout_frames()
    got = payout_rules(rows, paid)

    ids = {f"jra:2025-07-06:FUK:{k + 1:02d}": name for k, name in enumerate(PAYOUT_RACES)}
    assert set(map(str, got.index)) == set(ids), "one row per race of `rows`, and only those"
    got.index = got.index.map(str)
    for race_id, name in ids.items():
        places, wide, place_status, wide_status = PAYOUT_RACES[name]["want"]
        row = got.loc[race_id]
        assert int(row["places"]) == places, f"{name}: places {row['places']}, want {places}"
        assert bool(row["wide"]) is wide, f"{name}: wide {row['wide']}, want {wide}"
        assert row["place_status"] == place_status, f"{name}: Place {row['place_status']!r}"
        assert row["wide_status"] == wide_status, f"{name}: Wide {row['wide_status']!r}"


def test_place_places_follow_the_paid_rows_not_the_starters():
    """The same seven-runner race, its finish unchanged, scores two places or three by what paid."""
    first3 = (4, 1, 6)
    order = [*first3, 2, 3, 5, 7]
    rows = pd.DataFrame(
        {"race_id": "x", "horse_no": order, "finish_pos": [float(p) for p in range(1, 8)]}
    ).astype({"race_id": object, "horse_no": "int64"})
    for paid_places, want in (((4, 1), 2), ((4, 1, 6), 3)):
        paid = pd.DataFrame(
            [
                {"race_id": "x", "pool": pool, "combination": c, "is_refund": refund}
                for pool, c, refund in paid_rows(4, paid_places)
            ]
        )
        got = payout_rules(rows, paid)
        assert int(got.loc["x", "places"]) == want, paid_places
        assert got.loc["x", "place_status"] == SCORED


def trifecta(n: int, seed: int) -> np.ndarray:
    """A trifecta array: t[a, b, c] = P(a first, b second, c third), zero unless a, b, c differ."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, n, n))
    for a, b, c in itertools.permutations(range(n), 3):
        t[a, b, c] = rng.uniform(0.05, 1.0)
    return t / t.sum()


def bernoulli(p: float, y: bool) -> float:
    return math.log(p) if y else math.log(1.0 - p)


@pytest.mark.parametrize("n", [5, 7, 10])
@pytest.mark.parametrize("places", [0, 2, 3])
@pytest.mark.parametrize("wide", [True, False])
def test_pool_loglik_scores_the_places_and_wide_it_is_given(n, places, wide):
    t = trifecta(n, seed=10 * n + places)
    rng = np.random.default_rng(n)
    finish = rng.permutation(np.arange(1, n + 1)).astype(float)
    finish[np.flatnonzero(finish == n)] = np.nan  # the last runner did not finish
    top = {int(np.flatnonzero(finish == k)[0]) for k in (1, 2, 3)}
    triples = list(itertools.permutations(range(n), 3))

    got = pool_loglik(t, finish, places=places, wide=wide)

    if places == 0:
        assert math.isnan(got[PLACE]), "no places paid: not scored in Place"
    else:
        in_first = [sum(t[x] for x in triples if i in x[:places]) for i in range(n)]
        placed = [finish[i] <= places for i in range(n)]  # NaN compares False
        want = sum(bernoulli(in_first[i], placed[i]) for i in range(n))
        assert got[PLACE] == pytest.approx(want, rel=1e-9), (n, places)
    if not wide:
        assert math.isnan(got[WIDE]), "Wide not paid: not scored in Wide"
    else:
        want = 0.0
        for i, j in itertools.combinations(range(n), 2):
            both = sum(t[x] for x in triples if i in x and j in x)
            want += bernoulli(both, i in top and j in top)
        assert got[WIDE] == pytest.approx(want, rel=1e-9), n
    want_trio = math.log(sum(t[x] for x in itertools.permutations(sorted(top))))
    assert got[TRIO] == pytest.approx(want_trio, rel=1e-9), n


def test_a_three_place_outcome_on_seven_runners_scores_third_as_placed():
    t, finish = trifecta(7, seed=3), np.array([3.0, 5.0, 1.0, 7.0, 2.0, 4.0, 6.0])
    third = 0  # finished third
    p_top2 = sum(t[x] for x in itertools.permutations(range(7), 3) if third in x[:2])
    p_top3 = sum(t[x] for x in itertools.permutations(range(7), 3) if third in x)
    two = pool_loglik(t, finish, places=2, wide=False)[PLACE]
    three = pool_loglik(t, finish, places=3, wide=False)[PLACE]
    # The third finisher's term moves from ln(1 - P(top 2)) to ln P(top 3); every other runner's
    # term moves from the top-2 to the top-3 chance, unplaced in both.
    others = 0.0
    for i in range(7):
        if i == third:
            continue
        p2 = sum(t[x] for x in itertools.permutations(range(7), 3) if i in x[:2])
        p3 = sum(t[x] for x in itertools.permutations(range(7), 3) if i in x)
        placed2, placed3 = finish[i] <= 2, finish[i] <= 3
        others += bernoulli(p3, placed3) - bernoulli(p2, placed2)
    want = math.log(p_top3) - math.log(1.0 - p_top2) + others
    assert three - two == pytest.approx(want, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# A run whose tables cannot be produced: a JRDB build without TYB deliveries
# ---------------------------------------------------------------------------------------------

STORE_RACES = [
    # race_id, race_date, post_time_local, first three, starters, a scratched runner or None
    ("jra:2025-05-04:TOK:01", date(2025, 5, 4), "10:05", (3, 1, 7), 8, 5),
    ("jra:2025-05-04:TOK:11", date(2025, 5, 4), "15:40", (2, 9, 4), 10, None),
    ("jra:2025-05-11:KYO:02", date(2025, 5, 11), "10:35", (6, 2, 1), 7, None),
    ("jra:2025-05-11:KYO:11", date(2025, 5, 11), "15:35", (1, 8, 3), 9, 4),
]


def write_store(data: Path, *, tyb: str) -> None:
    """A synthetic JRA store under `data`/processed/jra: runners, races, dividends and a
    `jrdb_pre_race` holding KYI's day-before odds but no TYB board: `tyb` "null" (the TYB columns
    all null) or "absent" (no TYB columns)."""
    rng = np.random.default_rng(7)
    store = data / "processed" / "jra"
    store.mkdir(parents=True)
    runners, races, pre, dividends = [], [], [], []
    for race_id, day, post, first3, n, scratched in STORE_RACES:
        meeting = race_id.rsplit(":", 1)[0]
        hh, mm = map(int, post.split(":"))
        published = datetime(day.year, day.month, day.day, hh - 9 + 1, mm, tzinfo=timezone.utc)
        races.append(
            {
                "race_id": race_id,
                "race_date": day,
                "meeting_id": meeting,
                "status": "run",
                "post_time_local": post,
                "going": "good",
                "going_raw": "良",
                "finish_time_s": float(rng.uniform(70, 120)),
                "split_times_s": [12.1, 11.4, 11.9],
                "result_published_utc": published,
                "result_unpublished_utc": None,
            }
        )
        order = [*first3, *[h for h in range(1, n + 1) if h not in first3 and h != scratched]]
        for h in range(1, n + 1):
            out = h == scratched
            runners.append(
                {
                    "race_id": race_id,
                    "race_date": day,
                    "meeting_id": meeting,
                    "horse_no": h,
                    "win_odds": None if out else round(float(rng.uniform(1.5, 80.0)), 1),
                    "finish_pos": None if out else order.index(h) + 1,
                    "dead_heat": False,
                    "is_scratched": out,
                }
            )
            row = {"race_id": race_id, "race_date": day, "horse_no": h}
            pre.append(row | {"kyi_base_odds": round(float(rng.uniform(1.5, 80.0)), 1)})
        paid = paid_rows(first3[0], first3, wide_pairs(first3))
        if scratched is not None:
            paid += [(WIN, str(scratched), True), (PLACE, str(scratched), True)]
        for pool, combination, refund in paid:
            dividends.append(
                {"jurisdiction": "jra", "race_id": race_id, "pool": pool}
                | {"combination": combination, "dividend": 200.0, "is_refund": refund}
            )

    utc = pa.timestamp("us", tz="UTC")
    schemas = {
        "races": {
            "race_id": pa.string(),
            "race_date": pa.date32(),
            "meeting_id": pa.string(),
            "status": pa.string(),
            "post_time_local": pa.string(),
            "going": pa.string(),
            "going_raw": pa.string(),
            "finish_time_s": pa.float64(),
            "split_times_s": pa.list_(pa.float64()),
            "result_published_utc": utc,
            "result_unpublished_utc": utc,
        },
        "runners": {
            "race_id": pa.string(),
            "race_date": pa.date32(),
            "meeting_id": pa.string(),
            "horse_no": pa.int64(),
            "win_odds": pa.float64(),
            "finish_pos": pa.int64(),
            "dead_heat": pa.bool_(),
            "is_scratched": pa.bool_(),
        },
        "jrdb_pre_race": {
            "race_id": pa.string(),
            "race_date": pa.date32(),
            "horse_no": pa.int64(),
            "kyi_base_odds": pa.float64(),
        },
        "dividends": {
            "jurisdiction": pa.string(),
            "race_id": pa.string(),
            "pool": pa.string(),
            "combination": pa.string(),
            "dividend": pa.float64(),
            "is_refund": pa.bool_(),
        },
    }
    if tyb == "null":
        pre = [row | {"tyb_win_odds": None, "tyb_odds_time": None} for row in pre]
        schemas["jrdb_pre_race"] |= {"tyb_win_odds": pa.float64(), "tyb_odds_time": pa.string()}
    for name, rows in (
        ("races", races),
        ("runners", runners),
        ("jrdb_pre_race", pre),
        ("dividends", dividends),
    ):
        schema = pa.schema(list(schemas[name].items()))
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), store / f"{name}.parquet")


def last_complete_outputs(out: Path) -> dict[Path, bytes]:
    """The fixed-name outputs a complete run left in `out` (write_outputs on synthetic tables)."""
    frame = pd.DataFrame(
        [
            {"race_id": f"r{k}", "meeting_id": f"m{k // 2}", "horse_no": h, FINAL: q, BOARD: q}
            for k in range(6)
            for h, q in enumerate((0.5, 0.3, 0.2), start=1)
        ]
    )
    levels = {FINAL: 1.0, BOARD: 2.0}

    def score(model, rows, price):
        value = 0.1 * model + (0.01 if price == BOARD else 0.0)
        return pd.Series(value, index=pd.Index(pd.unique(rows["race_id"]), name="race_id"))

    boot = {"n_boot": 50, "level": 0.95, "seed": 1}
    tables = {
        key: cross_apply(
            frame,
            fit_price=FINAL,
            score_price=BOARD,
            fit=lambda rows, price: levels[price],
            score=score,
            bootstrap=boot,
        )
        for key in sorted(ALL_KEYS)
    }
    written = write_outputs(tables, out_dir=out)
    paths_written = written.values() if isinstance(written, dict) else written
    files = {Path(p).resolve(): Path(p).read_bytes() for p in paths_written}
    assert files and all(path.parent == out.resolve() for path in files)
    return files


def run_ids(registry_path: Path) -> list[str]:
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return [e["run_id"] for e in entries if e.get("kind") == "run"]


def run_cli(argv: list[str], capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("tyb", "out_arg"),
    [
        pytest.param("null", "reports/board-vs-final", id="tyb-columns-null-out-given"),
        pytest.param("absent", None, id="tyb-columns-absent-default-out"),
    ],
)
def test_a_run_without_its_tables_keeps_the_last_outputs_and_exits_non_zero(
    data_dir, tmp_path, capsys, tyb, out_arg
):
    write_store(data_dir, tyb=tyb)
    out = data_dir / (out_arg or "reports")
    out.mkdir(parents=True)
    before = last_complete_outputs(out)
    if tyb == "null":  # the store reads: the run fails on the missing board, not on the read
        data = read_development()
        assert len(data["races"]) == len(STORE_RACES) and len(data["dividends"])
        assert post_times(data["races"]).notna().all()
    argv = ["--run"] + ([] if out_arg is None else ["--out", str(out)])

    code, result = run_cli(argv, capsys)

    assert code == 1, "a run with a table not produced exits non-zero"
    assert {tuple(key) for key in result["failed"]} == ALL_KEYS
    for path, data in before.items():
        assert path.read_bytes() == data, f"{path.name}: the last complete output was touched"
    new = [p.resolve() for p in files_under(out) if p.resolve() not in before]
    assert len(new) == 1, f"one attempt file, got {[p.name for p in new]}"
    attempt = new[0]
    assert "-attempt-" in attempt.name and attempt.suffix == ".md", attempt.name
    assert attempt.stat().st_size > 0
    assert {Path(p).resolve() for p in result["outputs"].values()} == {attempt}

    registry_path = data_dir / "registry" / "japan.jsonl"
    registry = Registry(registry_path, holdout_log=tmp_path / "no-holdout-log.jsonl")
    registry.verify()
    assert registry.frozen_spec_hash(bvf.FAMILY_ID) == result["spec_hash"]
    records = registry.records(bvf.FAMILY_ID)
    assert records and {r.status for r in records} == {"failed"}
    assert all(r.ticket == 103 and "jrdb" in r.sources for r in records)
    fitted = Counter(
        (r.hypothesis_id, r.components["pool"], r.components["fit_price"]) for r in records
    )
    assert {(h, pool) for h, pool, _ in fitted} == ALL_KEYS, "every table not produced is failed"
    want = {(h, pool, price): 1 for h, pool in ALL_KEYS for price in (FINAL, BOARD)}
    assert dict(fitted) == want, "a failed record per table and price fitted on"
    ids = run_ids(registry_path)
    assert len(ids) == len(records) == len(set(ids)) == result["records_appended"]
    assert set(result["record_ids"].values()) == set(ids)

    # The same run again: nothing appended, the held records' ids returned, outputs untouched.
    held = registry_path.read_bytes()
    code, again = run_cli(argv, capsys)
    assert code == 1
    assert registry_path.read_bytes() == held, "an identical rerun appends nothing"
    assert again["records_appended"] == 0
    assert again["record_ids"] == result["record_ids"]
    assert all(i in ids for i in again["record_ids"].values()), "the held records' own run ids"
    for path, data in before.items():
        assert path.read_bytes() == data, f"{path.name}: touched by the rerun"
    assert all("-attempt-" in p.name for p in files_under(out) if p.resolve() not in before)
