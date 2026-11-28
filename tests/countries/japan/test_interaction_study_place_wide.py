"""The C2 interaction members scored on Place and Wide through C5's ordering (ticket #574, a
follow-up of #93; C5 is #99's `models.exotic`).

Contract pinned here (`horsetrack.countries.japan.studies.interactions`):

- `score_place_wide(rows, p, ordering, odds, dividends, spec, *, comparator, fold_scheme)
  -> calibration.Study`, pure (reads no store, writes nothing):
  - `rows`: the scored starters, one row each: race_id, horse_no, field_size (the runners when
    sales opened), finish_pos (1 = first; None or NaN without a placing), meeting_id and fold
    (its scored fold's label, as `studies.meta.Scored.rows` carries it);
  - `p`: member name -> its out-of-fold stage-2 Win probability per row, a Series on rows'
    index summing to 1 within each race; `comparator` is one of its keys;
  - `ordering`: a C5 `models.exotic.Ordering`, which maps each race's Win probabilities to its
    Place and Wide probabilities (`member_pool_probs`, pinned in test_interactions.py);
  - `odds`: the final odds, race_id, pool, combination, odds (decimal, per unit stake); rows of
    another pool or of a race not in `rows` are ignored;
  - `dividends`: the store's dividend rows, `DIVIDEND_COLUMNS` (race_id, pool, combination,
    dividend in yen per 100, is_refund); rows of another pool or race are ignored;
  - `spec["bootstrap"]`: n_boot, level, seed, as the family's spec holds them.
- Its variants, every one with `components["member"]` (a key of `p`) and `components["pool"]`
  (`models.exotic.PLACE` or `WIDE`), with distinct names. For every member of `p` and each pool:
  - exactly one variant carries a `params["leaderboard"]` block of `docs/specs/leaderboard.md`'s
    convention (kind "member", that pool, ci_level the spec's level, n_unit "bets"), scored (no
    error), whose development metrics are the tail return at the final price: `return_per_100`
    is 100 x the realised return per unit stake of the top band of `TAIL_QUANTILES`
    (`core.eval.tail.tail_curve`, clustered on meeting_id, the spec's bootstrap) over a flat bet
    on every combination of the pool with a final odds row in the scored races, its claim the
    member's pool probability x its final odds, settled at its recorded dividend / 100 (0 when
    the pool paid it nothing); lo and hi 100 x the band's bounds, n its bets; `n_meetings` its
    meetings. So the records give each member one ranked Place row and one ranked Wide row
    (`core.eval.leaderboard.rows_from_registry`, `rank`).
  - for every member but the comparator, its variants carry exactly one `delta_ll` metric on
    fold "all_folds" with lo <= value <= hi: the paired log-loss gain of the member's pool
    probabilities over the comparator's on the pool's outcomes (positive: the member better),
    meeting-clustered bounds. A member equal to the comparator gains 0.

Not pinned: the variants' names and hypothesis ids, the leaderboard block's rule, selection and
payout rate, any other metric (per-fold points, other bands), the tables, the log loss's exact
form (only its sign and its zero), a race whose pool has no dividend row, dead heats, refunds,
and `compute_study`'s wiring of this function (its level-1 fits need the stores: the reviewer
checks it). Every frame is synthetic and seeded; the expected tail returns come from
`models.exotic.Ordering.joint_probs` (C5) and `core.eval.tail.tail_curve`, never from the code
under test.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.leaderboard import rank, rows_from_registry
from horsetrack.core.eval.registry import Registry, RunRecord
from horsetrack.core.eval.tail import tail_curve
from horsetrack.countries.japan.models import exotic
from horsetrack.countries.japan.studies import interactions as study

SPEC = {"bootstrap": {"n_boot": 200, "level": 0.95, "seed": 574}}
POOLS = (exotic.PLACE, exotic.WIDE)
PAYOUT = {exotic.PLACE: 0.8, exotic.WIDE: 0.775}
N_RACES = 240
PER_MEETING = 12
COMPARATOR = study.MARKET_ONLY
FAMILY = "japan-c2-interactions-place-wide-test"
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD = hashlib.sha256(b"synthetic place wide").hexdigest()
QC = {"a2": 0.1, "b2": -0.02, "h2": 0.05, "e2": -0.05}
QC |= {"a3": 0.15, "b3": -0.03, "h3": 0.04, "e3": -0.04}
ORDERINGS = {
    "harville": lambda: exotic.Ordering("harville"),
    "quadratic_context": lambda: exotic.Ordering("quadratic_context", dict(QC)),
}


def norm_within(v: np.ndarray, race: np.ndarray) -> np.ndarray:
    total = pd.Series(v).groupby(race).transform("sum").to_numpy()
    return v / total


def card(seed: int) -> dict:
    """N_RACES races of 5-16 starters (horse numbers with gaps), finishing orders drawn by
    Harville from the true Win probabilities, PER_MEETING races a meeting, two folds; the final
    Place and Wide odds from a tempered public (p^0.8) with noise; the paid dividends from the
    odds, floored to 10 yen and at least 110, never the odds themselves."""
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(N_RACES):
        field = int(rng.integers(5, 17))
        numbers = np.sort(rng.choice(np.arange(1, 19), size=field, replace=False))
        w = np.exp(rng.normal(0.0, 1.0, field))
        truth = w / w.sum()
        left, order = list(range(field)), []
        while left:
            q = truth[left] / truth[left].sum()
            order.append(left.pop(int(rng.choice(len(left), p=q))))
        finish = np.empty(field, dtype=int)
        finish[order] = np.arange(1, field + 1)
        parts.append(
            pd.DataFrame(
                {
                    "race_id": f"jra:syn:{r:04d}",
                    "horse_no": numbers.astype(int),
                    "field_size": field,
                    "finish_pos": finish,
                    "meeting_id": f"m{r // PER_MEETING:02d}",
                    "fold": "2021-01-01" if r < N_RACES // 2 else "2021-04-01",
                    "truth": truth,
                }
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    rows.index = pd.Index(10_000 + 7 * np.arange(len(rows)))
    race = rows["race_id"].to_numpy()
    truth = rows["truth"].to_numpy()
    p = {
        "good": pd.Series(truth, index=rows.index),
        COMPARATOR: pd.Series(norm_within(truth**0.5, race), index=rows.index),
        "bad": pd.Series(norm_within(truth**0.1, race), index=rows.index),
    }
    p["copy"] = p[COMPARATOR].copy()
    public = norm_within(truth**0.8, race)

    odds, dividends = [], []
    harville = exotic.Ordering("harville")
    for race_id, part in rows.groupby("race_id", sort=False):
        numbers = part["horse_no"].astype(int).tolist()
        field = int(part["field_size"].iloc[0])
        finish = dict(zip(numbers, part["finish_pos"].astype(int), strict=True))
        pub = public[rows.index.get_indexer(part.index)]
        for pool in POOLS:
            q = harville.joint_probs(pub, pool, horses=numbers, field_at_sale=field)
            paid = exotic.winning_combinations(finish, pool, field_at_sale=field)
            for combination, qc in q.items():
                o = max(1.0, round(PAYOUT[pool] / qc * float(np.exp(rng.normal(0.0, 0.2))), 1))
                odds.append((race_id, pool, combination, o))
                if combination in paid:
                    div = max(110.0, 10.0 * np.floor(o * 100.0 * 0.93 / 10.0))
                    dividends.append((race_id, pool, combination, div, False))
        winner = [str(h) for h, pos in finish.items() if pos == 1]
        odds.append((race_id, exotic.WIN, winner[0], 2.0))
        dividends.append((race_id, exotic.WIN, winner[0], 99_990.0, False))
    # a race that is not scored: its rows must never be read
    for pool, combination in ((exotic.PLACE, "1"), (exotic.PLACE, "2"), (exotic.WIDE, "1,2")):
        odds.append(("jra:syn:extra", pool, combination, 500.0))
        dividends.append(("jra:syn:extra", pool, combination, 99_990.0, False))
    return {
        "rows": rows.drop(columns="truth"),
        "p": p,
        "odds": pd.DataFrame(odds, columns=["race_id", "pool", "combination", "odds"]),
        "dividends": pd.DataFrame(dividends, columns=study.DIVIDEND_COLUMNS),
    }


def expected_tail(data: dict, member: str, pool: str, ordering) -> dict:
    """The top TAIL_QUANTILES band of a flat bet on every priced combination of `pool` in the
    scored races, claim p x final odds, settled at the dividends, built here from C5 and
    tail_curve."""
    rows, p = data["rows"], data["p"][member]
    odds = data["odds"].loc[data["odds"]["pool"].eq(pool)]
    divs = data["dividends"].loc[data["dividends"]["pool"].eq(pool)]
    paid = {(r, c): d / 100.0 for r, c, d in divs[["race_id", "combination", "dividend"]].values}
    meeting = rows.groupby("race_id")["meeting_id"].first()
    probs = {}
    for race_id, part in rows.groupby("race_id"):
        numbers = part["horse_no"].astype(int).tolist()
        field = int(part["field_size"].iloc[0])
        pr = p.reindex(part.index).to_numpy(dtype=float)
        for c, v in ordering.joint_probs(pr, pool, horses=numbers, field_at_sale=field).items():
            probs[(race_id, c)] = v
    scored = odds.loc[odds["race_id"].isin(set(rows["race_id"]))]
    bets = pd.DataFrame(
        {
            "claim": [
                probs[(r, c)] * o for r, c, o in scored[["race_id", "combination", "odds"]].values
            ],
            "ret": [paid.get((r, c), 0.0) for r, c in scored[["race_id", "combination"]].values],
            "meeting_id": meeting.reindex(scored["race_id"]).to_numpy(),
        }
    )
    boot = SPEC["bootstrap"]
    curve = tail_curve(
        bets,
        claim_col="claim",
        return_col="ret",
        cluster_col="meeting_id",
        quantiles=study.TAIL_QUANTILES,
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    top = curve.loc[curve["segment"].eq("all")].iloc[-1]
    return {
        "value": 100.0 * float(top["realised"]),
        "lo": 100.0 * float(top["lo"]),
        "hi": 100.0 * float(top["hi"]),
        "n": int(top["n_bets"]),
        "meetings": int(top["n_clusters"]),
    }


@pytest.fixture(scope="module", params=sorted(ORDERINGS))
def scored(request):
    data = card(574 + len(request.param))
    ordering = ORDERINGS[request.param]()
    out = study.score_place_wide(
        data["rows"],
        data["p"],
        ordering,
        data["odds"],
        data["dividends"],
        SPEC,
        comparator=COMPARATOR,
        fold_scheme="walk_forward:synthetic",
    )
    return data, ordering, out


def by_member_pool(variants) -> dict[tuple[str, str], list]:
    found: dict[tuple[str, str], list] = {}
    for v in variants:
        key = (v.components.get("member"), v.components.get("pool"))
        found.setdefault(key, []).append(v)
    return found


def metric(variant, fold: str, name: str):
    hits = [m for m in variant.metrics if m.fold == fold and m.metric == name]
    assert len(hits) == 1, (variant.name, fold, name, [(m.fold, m.metric) for m in variant.metrics])
    return hits[0]


def leaderboard_variant(variants: list):
    with_block = [v for v in variants if "leaderboard" in v.params]
    assert len(with_block) == 1, [v.name for v in variants]
    return with_block[0]


def delta_ll(variants: list):
    hits = [
        m for v in variants for m in v.metrics if m.fold == "all_folds" and m.metric == "delta_ll"
    ]
    assert len(hits) == 1, [v.name for v in variants]
    return hits[0]


# ---------------------------------------------------------------------------------------------
# DoD: each member has a Place and a Wide ranked row
# ---------------------------------------------------------------------------------------------


def test_each_member_has_one_tail_variant_per_pool_at_the_final_prices(scored):
    """Every member of `p`, the comparator included, gets one leaderboard variant in PLACE and
    one in WIDE, scored, its block the convention's, and its development return the top claim
    band settled at the recorded dividends: equal to the band built here from C5's ordering and
    tail_curve, so each member's own probabilities, the final odds, the dividends (not the odds)
    and only the scored races move it."""
    data, ordering, out = scored
    variants = out.variants
    assert len({v.name for v in variants}) == len(variants), "variant names must be distinct"
    groups = by_member_pool(variants)
    assert {key for key in groups} >= {(m, pool) for m in data["p"] for pool in POOLS}
    for v in variants:
        assert v.components.get("member") in data["p"], v.name
        assert v.components.get("pool") in POOLS, v.name
    seen = set()
    for member in data["p"]:
        for pool in POOLS:
            v = leaderboard_variant(groups[(member, pool)])
            assert v.error is None, (member, pool, v.error)
            block = v.params["leaderboard"]
            assert block["kind"] == "member" and block["pool"] == pool, (member, block)
            assert block["ci_level"] == SPEC["bootstrap"]["level"] and block["n_unit"] == "bets"
            want = expected_tail(data, member, pool, ordering)
            got = metric(v, "development", "return_per_100")
            assert got.value == pytest.approx(want["value"], abs=1e-6), (member, pool)
            assert got.lo == pytest.approx(want["lo"], abs=1e-6), (member, pool)
            assert got.hi == pytest.approx(want["hi"], abs=1e-6), (member, pool)
            assert got.n == want["n"], (member, pool)
            meetings = metric(v, "development", "n_meetings")
            assert meetings.value == want["meetings"], (member, pool)
            seen.add((pool, round(got.value, 6)))
    # the members' claims differ, so their top bands (and returns) are not one constant
    for pool in POOLS:
        assert len({value for p_, value in seen if p_ == pool}) >= 2, pool


def test_the_records_give_each_member_a_ranked_place_and_wide_row(scored, tmp_path, monkeypatch):
    """The variants as registry records read back through the leaderboard: each member's
    leaderboard variant is one measured row in its pool, and `rank` ranks it."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    data, _, out = scored
    registry = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    for v in out.variants:
        registry.append(
            RunRecord(
                hypothesis_id=v.hypothesis_id,
                family_id=FAMILY,
                variant=v.name,
                ticket=574,
                git_sha=GIT_SHA,
                data_builds={"results": BUILD},
                sources=("jra",),
                fold_scheme=v.fold_scheme,
                components=v.components,
                params=dict(v.params) | ({"error": v.error} if v.error else {}),
                metrics=v.metrics,
                status="failed" if v.error else "scored",
            )
        )
    ranked = {r.row.option_id: r for r in rank(rows_from_registry(registry))}
    groups = by_member_pool(out.variants)
    for member in data["p"]:
        for pool in POOLS:
            v = leaderboard_variant(groups[(member, pool)])
            row = ranked.get(f"{FAMILY}/{v.name}")
            assert row is not None, (member, pool, sorted(ranked))
            assert row.row.pool == pool and row.row.measured, (member, pool, row.row)
            assert row.rank is not None, (member, pool)
    for pool in POOLS:
        in_pool = [r for r in ranked.values() if r.row.pool == pool and r.rank is not None]
        assert len(in_pool) == len(data["p"]), pool


# ---------------------------------------------------------------------------------------------
# Outline 2: ΔLL at the final Place and Wide prices, with CI, per member
# ---------------------------------------------------------------------------------------------


def test_delta_ll_against_the_comparator_has_the_planted_sign_in_each_pool(scored):
    """Finishing orders are drawn from `good`'s probabilities; the comparator is them tempered
    (p^0.5), `bad` tempered further (p^0.1) and `copy` the comparator itself. In each pool the
    true member gains with its lower bound above 0, `bad` loses with its upper bound below 0,
    and `copy` gains exactly 0."""
    data, _, out = scored
    groups = by_member_pool(out.variants)
    for pool in POOLS:
        for member in data["p"]:
            if member == COMPARATOR:
                continue
            m = delta_ll(groups[(member, pool)])
            assert m.lo is not None and m.hi is not None, (member, pool)
            assert m.lo <= m.value <= m.hi, (member, pool, m)
            assert m.n > 0, (member, pool)
        good = delta_ll(groups[("good", pool)])
        assert good.lo > 0, (pool, good)
        bad = delta_ll(groups[("bad", pool)])
        assert bad.hi < 0, (pool, bad)
        same = delta_ll(groups[("copy", pool)])
        assert same.value == pytest.approx(0.0, abs=1e-12), (pool, same)
