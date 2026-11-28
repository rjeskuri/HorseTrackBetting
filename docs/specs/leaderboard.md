# Leaderboard: every option, ranked by return per ¥100 at the final price

Code: `src/horsetrack/core/eval/leaderboard.py` (rows, shrinkage, ranking and writers, #449; the
registry reader `rows_from_registry`, #450). Governance: the problem statement as amended
2026-11-27 (D15 to D19, epic #447) and D21.

Every option the project has measured or will measure (race type × pool × combination × rule,
and every model and member) is one row on its country's leaderboard. Nothing is deleted or
excluded: a failed run, a null result and a variant with no return yet are rows too. Nothing is
adopted or rejected by the leaderboard; it orders the options.

## The row (`LeaderboardRow`)

| field | meaning |
| --- | --- |
| `option_id` | unique per leaderboard; from the registry `'<family_id>/<variant>'` or `'<family_id>/<variant>[<segment>]'` |
| `family_id` | the registry family |
| `ticket` | the run's ticket, or None |
| `kind` | one of `rule`, `comparator`, `cell`, `model`, `member`, `other` |
| `pool` | the pari-mutuel pool (e.g. `WIN`, `PLACE`, `EXACTA`), or None |
| `segment` | `all`, or a segment label such as `class:maiden` |
| `rule` | the option in words |
| `selection` | when the selection is known: `decision_time`, `card_only`, `look_ahead`, `unknown` |
| `sources` | the run's data sources |
| `variants_searched` | the family's variants tried (`Registry.family_size`), at least 1 |
| `status` | the run's status (`scored`, `failed`; legacy `rejected` and `adopted` read like `scored`) |
| `return_per_100` | mean return per ¥100 staked at the final dividend (takeout and breakage included), or None when unmeasured |
| `ci_lo`, `ci_hi` | each bound one-sided at `ci_level` (D7 style) |
| `ci_level` | strictly between 0.5 and 1 |
| `se_per_100` | the standard error of the return, when given |
| `n`, `n_unit` | the sample size and its unit: `bets`, `races` or `days` |
| `n_meetings` | the number of race meetings behind the return, when given |
| `payout_rate` | the pool's payout rate (1 - takeout) |
| `shrunk_per_100` | set by `shrink`, never by a reader |
| `licensed` | the run is built on a licensed source (D21) |
| `note` | free text, e.g. why a registry block was refused |
| `descriptive` | a bool, default False: the row is a measurement shown on the board but given no D16 rank (below) |
| `class_group`, `odds_band` | the row's stake-capacity cell beside its `pool`, or None (#491) |
| `capacity` | the cell's capacity stake: (edge, quantile, yen) triples, yen +inf where own money never eats that share of the edge; `capacity_at(edge, quantile=0.5)` reads one |
| `capacity_note` | why the row carries no capacity, or None |

A row is checked on construction (`RowRefused`). A row is *measured* when `return_per_100` is set
and *withheld* when it is marked licensed or its sources name JRDB or JRA-VAN in any spelling
(`core.eval.reports.is_withheld`).

### Descriptive rows

A *descriptive* row holds a measured value that is not a return per ¥100 staked on a defined
ticket, so it cannot be compared with one under D16. The case today is #135's WIN5 carryover
cells (`docs/experiments/win5-carryover-japan.md`, "On the leaderboard"): a cell's value is the
mean share of the day's pool paid back to all its bettors, and the study measured no ticket
strategy, so a cell above 1 must not outrank a ticket rule. A WIN5 row ranks under D16 only on a
walk-forward return per ¥100 staked on a selection rule, a follow-up of epic #447. A descriptive
row keeps its value, bounds and n; `shrink` leaves it out of its pool's estimate and gives it no
shrunk value; `rank` gives it no rank. A row file written before the flag existed reads as not
descriptive.

### Capacity (#491)

Beside each row, whether the flat stake fits its pool before our own money eats half the edge:
the capacity stake of #432's depth table (`docs/specs/pool-depth-prereg.md`) in the row's cell,
pool x class group x odds band, at every edge (5, 10 and 20%) and quantile the table holds.
`countries/japan/research/leaderboard_build.with_capacity` joins the rows to the tables that
`pool-depth run` writes under the reports dir (`pool_depth.read_depth_tables`); a candidate cell
names its cell by its `pool`, `class_group` and `odds_band`. A row with no capacity says why in
`capacity_note`: no depth table yet; no class group or odds band named (a model, a registry rule,
a cell of every class); a co-placer pool (Place, Wide: the table has no single-winner capacity);
no such cell in the table; or a cell with no capacity figure. The capacity never ranks, shrinks
or withholds a row. In the JSON an infinite stake is the string `"inf"`; a row file written
before #491 reads with no capacity.

## Shrinkage and ranking

`shrink` sets `shrunk_per_100` within each pool by DerSimonian-Laird over the measured rows with
a positive standard error that are not descriptive (`se_per_100`, else recovered from the
one-sided bounds at `ci_level`):
each row is pulled toward its pool's own random-effects mean by `tau^2 / (tau^2 + se^2)`. A pool
of one keeps its raw return; a row without a pool or a positive se gets no shrunk value. The
target is the pool's mean, not its payout rate. Rows of one pool overlap (a segment's bets are
also its 'all' row's; variants share races), so the shrinkage is a ranking device, not
inference: no CI, test or claim is built on it. The formulas and their numerically stable form
are in the module docstring.

`rank` orders the measured rows that are not descriptive by shrunk value (else raw return)
descending, ties by raw return descending then option id, ranked 1..k; the measured descriptive
rows follow, unranked, by family id then option id; unmeasured rows come last, unranked, by
family id then option id (an unmeasured descriptive row has no value to show and sorts with
them). Option ids must be unique. Both Markdown texts keep descriptive rows out of the ranked
table (value column "Return per ¥100 staked"), which holds the ranked rows and then the
unmeasured ones that are not descriptive. Every shown descriptive row follows in its own section,
`## Descriptive, not ranked`, in a table whose value column is "Pool value per ¥100
(descriptive)": in `rank`'s order, with its value, CI and n, no shrunk value and "—" for rank.
The public text's `## Withheld (D21)` list stays last. Both headers say descriptive rows are not
ranked; `to_json` and `from_json` carry the flag.

## From the registry (`rows_from_registry`)

`rows_from_registry(registry, *, adapters=None) -> list[LeaderboardRow]` reads the run registry
(`core/eval/registry.py`, #74) through `Registry.records()` and never writes it. The rows come
back unshrunk.

### The convention for new studies

A study that wants measured rows records, per variant:

- `params['leaderboard']`, a mapping with exactly these fields:
  `kind`, `pool`, `rule`, `selection`, `payout_rate`, `ci_level`, `n_unit`, with the
  vocabularies of the row above (`pool` a non-blank string or None, `rule` non-blank,
  `payout_rate` a finite number or None). `params` is part of a variant's identity
  (`Registry.family_size`), which is right: these fields describe the variant.
- Metrics on the fold `development`:
  - `return_per_100` (segment `all`) or `return_per_100[<segment>]`: `value` the mean return per
    ¥100 staked at the final dividend, `lo` and `hi` one-sided bounds at `ci_level`, `n` in
    `n_unit`;
  - optional `se_per_100[<segment>]` (or `se_per_100` for `all`): the return's standard error;
  - optional `n_meetings[<segment>]` (or `n_meetings`): the race meetings behind it, a whole
    number (stored as a float, read back as an int).

  Each `se_per_100` and `n_meetings` metric belongs to the return of the same segment only;
  none is borrowed across segments. Metrics on other folds are per-fold detail and give no row.

### Rows per run and per variant

- A run with `holdout_open` set gives no row. A variant run only on the holdout has none, and a
  holdout run never replaces an earlier development run.
- A run with a well-formed block gives one measured row per development `return_per_100` metric:
  option id `'<family_id>/<variant>'` for `all`, `'<family_id>/<variant>[<segment>]'` otherwise;
  return, bounds and n from the metric; kind, pool, rule, selection, payout rate, CI level and n
  unit from the block.
- Every row carries the family size as `variants_searched`, the run's `ticket`, `sources` and
  `status`, and `licensed = is_licensed_record(record)`, which is exactly what `write_summary`
  withholds (a withheld name among the sources, the data build names or the components).
- A run that gives no row (a failed run, a legacy family whose metrics are not returns, a block
  with no development return) gives one unmeasured row, the option `'<family_id>/<variant>'`:
  kind from its block if valid, else `other`; pool from its block or None; rule its variant
  text; selection `unknown`; status the run's.
- A malformed block (not a mapping, a missing or unknown field, a value outside its vocabulary)
  gives that unmeasured row with the reason in `note`; the rest of the registry still reads.
- Rows resolve per option id: the latest row of each option in append order over the
  non-holdout runs is its row. A rerun replaces only the options it gives; an option a later run
  does not give (a segment the rerun drops) keeps its earlier row. A later failed, unmeasured or
  malformed run's unmeasured row replaces `'<family_id>/<variant>'` alone, so its segment rows
  keep their earlier measured values.

### Adapters

`adapters` maps a family id to `adapter(record, family_size) -> list[LeaderboardRow]`. For a
family with an adapter, the adapter is called with each non-holdout run of the family and the
family size, its rows replace the convention's, and they resolve by the same per-option rule.
Each adapter row's `licensed` is `row.licensed or is_licensed_record(record)`, so an adapter
cannot publish a run `write_summary` withholds. An adapter returning `[]` gives that run the
default unmeasured row, with the reason in `note` when the run's block is malformed. Adapters exist for legacy families recorded before the
convention (#451 backfills them).

## Outputs

- `render_markdown(rows, title=..., public=...)`, `to_json` and `from_json`;
  `write_leaderboard(path, rows, title=..., public=...)` writes through
  `core.eval.reports.write_report`.
- Private: every row, shrunk over all of them, written under `$HORSETRACK_DATA_DIR` (never into
  the repository when any row is withheld).
- Public: `public_view` drops withheld rows and reshrinks the rest over themselves alone, so no
  public shrunk value or rank depends on a licensed row; withheld rows are listed by option id
  and family only. The public variants total still counts withheld families, so the search the
  multiple-testing count discloses is not understated. The public Japan leaderboard is
  `docs/leaderboard/japan.md` (#452).
- The Capacity column: privately the median capacity stake at each edge, or the row's capacity
  note; publicly `withheld (D21)` for every row that carries a capacity (the depth table is
  JRA-VAN), or the note. The flat stake it is read against is `flat_stake_yen` in
  `countries/japan/config/betting.yaml`.

## D21

No JRDB or JRA-VAN figure enters the public repository: no return, bound, se, n, meetings,
payout rate, shrunk value, rule text or note of a withheld row. Licensed metrics stay in the
out-of-repository registry and the private leaderboard. Only an amendment of D21 itself could
change this.
