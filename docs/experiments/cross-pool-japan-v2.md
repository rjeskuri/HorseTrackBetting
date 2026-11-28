# Experiment report: E25, the T-2 cross-pool signal at the final price (Japan, #562)

Ticket #562: the v2 family of #435's study (epic #73, N4). **Nothing is adopted.** **Every figure is withheld (D21).** The study reads the JRA-VAN time-series boards and the jravan final odds, results and dividends, a licensed source, so this page says what ran and where its figures live. It gives no count, span, coefficient, gain, return, rate, sign or bound derived from that data.

- Registry family `japan-cross-pool-t2-v2`, spec sha256 `86697c3987cebd7a3b1f59bfbad772e36b8a4d18ddbffad455ca62ef82274dc4`, frozen in the registry (`--freeze`) before any scored read. The spec is `docs/specs/cross-pool-prereg.md` (PRE-REGISTERED).
- Family size: 5 pre-registered fit variants. One registry record each: `d161da9abb090ad0`, `dad07deeb223f503`, `e1017ebfe2d5ba7e`, `ea2606f2ff698928`, `db4af658f1c4193f`. Failed: none.
- Private report: `$HORSETRACK_DATA_DIR/reports/cross-pool-japan-86697c3987ce-data-ce44162831c80cde.md`. Candidate cells (every settled selection and N5 band, ranked): `$HORSETRACK_DATA_DIR/reports/candidate-cells/cross-pool-japan-86697c3987ce-data-ce44162831c80cde.json`.
- Data fingerprint `ce44162831c80cde` (a hash of the store's builds and day list, in the private files' names); a rerun on other data writes new files beside these, and a rerun on the same data is refused, never an overwrite.
- Code: `countries/japan/studies/cross_pool_run.py`. Reproduce with `uv run python -m horsetrack.countries.japan.studies.cross_pool_run --run --family v2 --from 2014-06-07 --to 2026-11-27 --seed 435`.
- Holdout: development days only. A holdout day's file is dropped by its name and never opened (D14); `LockedHoldout` is not opened; nothing before 2014-06-07 enters (D10).

## Hypothesis

E25: at t = post - 2 min (D2's primary), the Quinella pool's support against the Win-implied (Harville) support adds to the final price: beta != 0 in u = offset + beta . x, and the walk-forward winner log-loss gain over the offset is positive. Per runner, x_cross (the top-2 support) over the final Win price; per pair (`pair:quinella`), log_ratio (the pair price at t against the Win-implied pair) over the final Quinella price, the winners the pairs the QUINELLA dividends pay. **Null:** beta = 0 and no gain, the starting point.

## Variants tried

| Variant | Features | Offset | Status | Coefficient (95% CI) | ΔLL (one-sided bounds) |
| --- | --- | --- | --- | --- | --- |
| `x_cross:normalised` | x_cross | normalised | scored | withheld (D21) | withheld (D21) |
| `x_cross:calibrated` | x_cross | calibrated | scored | withheld (D21) | withheld (D21) |
| `x_cross+fav:normalised` | x_cross, fav_x_cross | normalised | scored | withheld (D21) | withheld (D21) |
| `x_cross+fav:calibrated` | x_cross, fav_x_cross | calibrated | scored | withheld (D21) | withheld (D21) |
| `pair:quinella` | log_ratio | quinella | scored | withheld (D21) | withheld (D21) |

The gain by class group (newcomer, maiden, conditions, open, graded): withheld (D21).

## Settlement at the final dividend

The planned settlement pools: WIN, QUINELLA and FORECAST (Exacta) for each Win variant and its offset alone (beta = 0: `offset:normalised`, `offset:calibrated`); QUINELLA for the pair variant and its offset alone (`offset:quinella`). Each planned pool's availability in this run is in the private report, withheld (D21). `value` bets every priced combination with claim p x O > 1, `top1` the race's largest claim (equal claims to the lowest final odds, then the lowest combination text). The offsets alone have no registry record: their returns ride as `offset_return[...]` metrics on the records of the variants with the same offset.

Settlement, the N5 curves and the candidate cells are a retrospective diagnostic at the closing (final) prices, not decision-time (T-2) rules: each claim uses the final price twice, in the offset and as the odds, so a late price move can put a combination in or out of a selection after t, and no selection is a bet that could be placed at t.

| Scorer | Pool | Selection | Return per unit stake (bounds) |
| --- | --- | --- | --- |
| `x_cross:normalised` | WIN | value | withheld (D21) |
| `x_cross:normalised` | WIN | top1 | withheld (D21) |
| `x_cross:normalised` | QUINELLA | value | withheld (D21) |
| `x_cross:normalised` | QUINELLA | top1 | withheld (D21) |
| `x_cross:normalised` | FORECAST | value | withheld (D21) |
| `x_cross:normalised` | FORECAST | top1 | withheld (D21) |
| `x_cross:calibrated` | WIN | value | withheld (D21) |
| `x_cross:calibrated` | WIN | top1 | withheld (D21) |
| `x_cross:calibrated` | QUINELLA | value | withheld (D21) |
| `x_cross:calibrated` | QUINELLA | top1 | withheld (D21) |
| `x_cross:calibrated` | FORECAST | value | withheld (D21) |
| `x_cross:calibrated` | FORECAST | top1 | withheld (D21) |
| `x_cross+fav:normalised` | WIN | value | withheld (D21) |
| `x_cross+fav:normalised` | WIN | top1 | withheld (D21) |
| `x_cross+fav:normalised` | QUINELLA | value | withheld (D21) |
| `x_cross+fav:normalised` | QUINELLA | top1 | withheld (D21) |
| `x_cross+fav:normalised` | FORECAST | value | withheld (D21) |
| `x_cross+fav:normalised` | FORECAST | top1 | withheld (D21) |
| `x_cross+fav:calibrated` | WIN | value | withheld (D21) |
| `x_cross+fav:calibrated` | WIN | top1 | withheld (D21) |
| `x_cross+fav:calibrated` | QUINELLA | value | withheld (D21) |
| `x_cross+fav:calibrated` | QUINELLA | top1 | withheld (D21) |
| `x_cross+fav:calibrated` | FORECAST | value | withheld (D21) |
| `x_cross+fav:calibrated` | FORECAST | top1 | withheld (D21) |
| `offset:normalised` | WIN | value | withheld (D21) |
| `offset:normalised` | WIN | top1 | withheld (D21) |
| `offset:normalised` | QUINELLA | value | withheld (D21) |
| `offset:normalised` | QUINELLA | top1 | withheld (D21) |
| `offset:normalised` | FORECAST | value | withheld (D21) |
| `offset:normalised` | FORECAST | top1 | withheld (D21) |
| `offset:calibrated` | WIN | value | withheld (D21) |
| `offset:calibrated` | WIN | top1 | withheld (D21) |
| `offset:calibrated` | QUINELLA | value | withheld (D21) |
| `offset:calibrated` | QUINELLA | top1 | withheld (D21) |
| `offset:calibrated` | FORECAST | value | withheld (D21) |
| `offset:calibrated` | FORECAST | top1 | withheld (D21) |
| `pair:quinella` | QUINELLA | value | withheld (D21) |
| `pair:quinella` | QUINELLA | top1 | withheld (D21) |
| `offset:quinella` | QUINELLA | value | withheld (D21) |
| `offset:quinella` | QUINELLA | top1 | withheld (D21) |

N5 tail-return curves per scorer and pool, by class group: withheld (D21). The ranked candidate cells (`study` `cross-pool/final-price-diagnostic`): withheld (D21). Counts, exclusions and which planned pools were available: withheld (D21), in the private report.

## What the result rules out

withheld (D21): the reading rests on the bounds in the private report.
