# N8: the popularity-rank spread scan, pre-registration (ticket #503)

Status: **FROZEN** on 2026-11-27 at commit `c59a484`, before any scored read (it was PROPOSED
until then). All six specs together (`spread_scan.spec_hash()`): sha256
`6f041ec1399564dc0505f7408f205c7d34b6146a13c9b5442e97f8009b08be00`; per family:

| Family | Spec sha256 |
| --- | --- |
| `japan-n8-spread-scan-exacta` | `a380b3aaca6d675cbe9393b780970b8b72a81b0057143a59ffcdfc099b38499b` |
| `japan-n8-spread-scan-quinella` | `f30589a50bf50545eced734496c8fd87e9113ebbeaa280c65db23162421d65aa` |
| `japan-n8-spread-scan-wide` | `e33234f83cf4a7ff6fb43f299e96594072ce2b100e18ba527d634cd5001fab52` |
| `japan-n8-spread-scan-trio` | `fc1bb54bac4fc1752ce7f3f793ac2992b326dc827efa22d52898dc806ecd5611` |
| `japan-n8-spread-scan-trifecta` | `27bc2561847e5437136002010745f343a27dd4d42c1a6de00655f2d2b26126f8` |
| `japan-n8-spread-scan-bracket` | `5b509a0e1d97d78c03e32bb9ab685b748bce98b342813d3475c55eb851c1597c` |

Families: the six `japan-n8-spread-scan-<pool>` families below. The spec is `horsetrack.countries.japan.research.spread_scan.family_spec(<family>)`, plain
JSON; `uv run horsetrack japan spread-scan freeze` freezes each family's spec with
`Registry.freeze_family` in `$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze
cannot be undone). `run_scan` reads no store row unless every family is frozen with exactly its
spec, so any change to the spec after the freeze is a new family, frozen again before it is run.
Under the owner's reframe of 2026-11-27 there is no sign-off gate and no pass/fail verdict: every
cell is a ranked row on the leaderboard (#447).

No figure from the JRA-VAN store appears here or in any public file (D21): counts, rates, spans,
signs, bounds and ranks live in the private report under `$HORSETRACK_DATA_DIR/reports/`.

## Hypothesis

Some pre-registered spread, bought on the T-2 Win popularity ranks, returns more than its stake at
the paid dividends in some cell (shrunk edge above 0). The null, the starting point: no spread
beats the takeout in any cell. The look-ahead variant (ranks from the final Win board) is reported
beside it as a ceiling, never as a strategy.

## The spreads (27) and their families (6)

A spread is a mark card in Win popularity ranks; its id is `<pool>:<rank legs>`. Each distinct
combination of the card is one ¥100 ticket, expanded with #139's `betting.formations` (`box`,
`wheel` with the axis position on an ordered pool, `formation`) on the rank numbers and mapped to
the horses holding those ranks.

| Family | Pool | Spreads |
| --- | --- | --- |
| `japan-n8-spread-scan-exacta` | FORECAST | `exacta:1/2-3`, `exacta:1/2-4`, `exacta:1/2-5` (favourite first); `exacta:2-3/1`, `exacta:2-4/1`, `exacta:2-5/1` (favourite second; 2-4 over the favourite is "the contender over the favourite"); `exacta:box1-3`, `exacta:box1-4` |
| `japan-n8-spread-scan-quinella` | QUINELLA | `quinella:1/2-3`, `quinella:1/2-4`, `quinella:1/2-5`; `quinella:box1-4`; `quinella:box2-5` (against the favourite) |
| `japan-n8-spread-scan-wide` | QUINELLA_PLACE | `wide:1/2-3`, `wide:1/2-4`, `wide:1/2-5`; `wide:box1-4`; `wide:box2-5` |
| `japan-n8-spread-scan-trio` | TRIO | `trio:1/2-5`, `trio:1/2-6` (favourite as axis); `trio:box1-4`, `trio:box1-5`; `trio:1-2/1-4/2-6` (formation) |
| `japan-n8-spread-scan-trifecta` | TIERCE | `trifecta:1/2-4/2-4` (favourite first, 2-4 second and third); `trifecta:1/2-4/2-7`; `trifecta:2-4/1/2-7` (favourite second) |
| `japan-n8-spread-scan-bracket` | BRACKET_QUINELLA | `bracket:1/2-3`: the favourite's bracket with the bracket of rank 2 and with that of rank 3, a same-bracket pair included, each distinct pair once (one or two tickets) |

Family size: 27 spreads, 2 variants per family (`t2`, `lookahead`): 12 registry records per run,
one per (family, variant). Each race is in 8 cells; a spread has at most 78 cell labels (1 'all',
7 class, 4 field, 4 favourite, 3 surface, 3 pool size, 28 class x favourite, 28 class x field), so
a run ranks at most 27 x 78 = 2,106 rows per variant. These counts are the design, not data.

## Ranks, bets and pricing

- **T-2 rank (D2).** At `post_utc - primary_min` of `config/decision.yaml` (2 minutes), each Win
  runner's latest `odds_timeseries` WIN row with `snapshot_utc` at or before that time gives its
  price. A runner whose latest such row has no price, or that `runners.is_scratched` flags, is
  dropped before ranking. Ranks: ascending price, ties by the lower horse number (numerically).
  The `__pool__` row and the other pools are never runners. Exotic boards before the logger have no
  T-2 time series, which is why every spread ranks on the Win board.
- **Look-ahead variant.** The same rule on the final Win board (`final_odds` WIN rows, the same
  drops): a labelled ceiling. Both variants bet the same races and use the same cells.
- **Bet.** Per race each spread is one mark card at ¥100 a ticket; cost = ¥100 x tickets.
- **Pricing.** A ticket returns the dividend of every paying `dividends` row of its pool with its
  combination text (finishing order for FORECAST and TIERCE, ascending otherwise, bracket numbers
  for BRACKET_QUINELLA), so Wide and dead heats can pay several tickets; a ticket with an
  `is_refund` row returns its ¥100 (precedent `studies/avoidance.py`); the jravan `dividends` table
  writes no refund rows (#443). Takeout is in the dividends.
- **No bet** in a race for a spread when the board ranks fewer runners than the spread's highest
  rank (each spread keeps one definition), when the race has no `dividends` row of the spread's
  pool (not sold, e.g. a bracket quinella in a small field, or not settled), or, for
  `bracket:1/2-3`, when a rank 1-3 runner has no bracket. The private report counts each. Since
  both variants bet the same races, each rule is decided once per (race, spread) for both boards:
  a spread is bought in both variants or in neither (made explicit by PR #524's review fixes; the
  frozen spec is unchanged).
- **Race skipped** (both variants): status other than `run`; no T-2 Win board, no T-2 Win pool
  total or no final Win board; segments undefined (`race_cells` refuses).

## Segments and cells

- **Class:** `races.race_class` DEBUT newcomer, MAIDEN maiden, C1/C2/C3 conditions, OP and L open,
  G3, G2 and G1 each a level of its own.
- **Field:** starters (runners not `is_scratched`): le4, 5-10, 11-14, 15-18 (`pool_scan.race_cells`).
- **Favourite strength:** the T-2 favourite's Win price: le1.9, 2.0-2.9, 3.0-4.9, ge5.0, lower edge
  inclusive (both variants: the look-ahead variant changes the ranking only).
- **Surface:** `races.surface`.
- **Pool size:** the T-2 Win `__pool__` investment, in tertiles per calendar year of that year's
  scored races (numpy linear 1/3 and 2/3 quantiles; low below the first edge, mid below the second,
  high from it). A scored race holds at least one bet after the segment and no-bet rules.
- **Cells:** per race 'all', one level of each segment, class x favourite and class x field; a row
  per (variant, spread, cell) with at least one bet.

## Metric, bootstrap and shrinkage

- **rho** (`mean_return`) = total return / total stake over the cell's bets; `n_bets` counts races
  (one card each), `n_tickets` combinations. `payout_rate` is the stake-weighted mean of
  `takeout.race_takeout(race, pool).payout_rate` (the base rate where an age-restricted boost entry
  matches, the jravan class code carrying no age label), descriptive only. `mean_odds` is the mean
  Win price of the variant board's favourite over the bets.
- **Bootstrap (D7).** Meetings (`races.meeting_id`) resampled with multinomial counts, numpy
  `default_rng(seed)`, one draw shared by every (variant, spread, cell); a replicate without the
  cell's meetings is skipped. `se_mean_return` is the replicates' sd of sum(return) / sum(stake);
  `ci_lo` and `ci_hi` their 5% and 95% quantiles (level 0.95 from `config/scoring.yaml`, one-sided
  bounds each). Run of record: seed 20261127, 1,000 replicates.
- **Shrinkage.** A spread's return is not one combination's odds, so #78's likelihood
  (`core.eval.segment_roi`) does not apply; the scan uses N1's (#430) normal empirical-Bayes shrink
  of the realised return, its Place and Wide method. Per (variant, spread): the baseline is
  rho_all of the 'all' cell (not shrunk); a cell's effect e = rho / rho_all with se_e = se /
  rho_all, where se = max(`se_mean_return`, sqrt(s2 / the cell's stake)) and s2 = sum over meetings
  (R_m - rho_all C_m)^2 / sum C_m of the 'all' cell, so a cell with one or two meetings (a
  degenerate bootstrap) is never taken as exact. Per segment kind, tau^2 = max(0, mean over its
  cells of (e - 1)^2 - se_e^2); B = tau^2 / (tau^2 + se_e^2) (0 when both are 0); shrunk rho =
  rho_all (1 + B (e - 1)) with posterior sd rho_all sqrt(B) se_e. 'all' keeps rho_all with sd
  max(`se_mean_return`, sqrt(s2 / its stake)). `shrunk_edge` = shrunk rho - 1;
  `p_return_above_100` = the normal posterior's P(rho > 1).
- **Ranking.** Per variant, by `shrunk_edge` descending, rows without one last, ties in spread then
  segment order; the t2 block first, then the look-ahead block.

## Data rules

- **D10:** nothing before 2014-06-07 is read or scored; an earlier start is clamped.
- **D14:** a holdout day (`is_holdout_day`) is never read: the store readers drop it inside the
  scan (`models.v0.store.read_store_table`, `core.eval.holdout.read_development` for `final_odds`),
  `dividends` (no race_date) is filtered by development race ids, and a holdout day's
  `odds_timeseries` file is never opened. `LockedHoldout` is never opened.
- **Memory:** one calendar year at a time, named columns only; `final_odds` WIN rows only, read by
  byte range.

## Outputs

- Private report `$HORSETRACK_DATA_DIR/reports/spread-scan-japan-<run key>.md`, one per run (the
  run key: the first 16 hex digits of the sha256 of the spec hash, range, seed, replicates, commit
  and store fingerprint; PR #524's review fixes, the first run's report was named by the spec hash
  alone), and every row in the candidate-cell schema (#430's keys plus `spread_id`, `variant`,
  `segment`, `rank`, `n_tickets`, `ci_lo`, `ci_hi`, `p_return_above_100`) in
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/spread-scan.json` (the latest scored run's), both
  through `write_report`.
- One registry record per (family, variant), failures included, hypothesis id `P503`, metrics per
  row (`mean_return[<spread>]` with its bounds and `shrunk_edge[<spread>]`, fold = segment); each
  scored record names its report in `params.report`. A range without a bet is recorded as failed
  and writes neither file.

## Rulings (reversible implementation choices)

- A field smaller than a spread's highest rank: no bet in that race for that spread, counted in
  the private report, so each spread keeps one fixed definition.
- Shrinkage: N1's normal empirical-Bayes shrink of the realised return toward the spread's own
  'all' rho, with the pooled-dispersion floor on se above, because #78's likelihood needs one
  combination's odds per bet.
- Pool size: the Win pool at T-2, the board a bettor sees, in per-year tertiles as N1's.
- Look-ahead variant: the T-2 segments (favourite band, pool size), so it differs from t2 only in
  the ranking.
- `mean_odds`: the variant board's favourite price; `payout_rate`: the stake-weighted race and
  pool payout rate.
- Families: one per pool; hypothesis id `P503` (a ticket-scoped P id, as #283's `P283`).
