# Experiment report: the WIN5 carryover study (Japan, #135, P1)

Ticket #135. D18 positive-edge family: on WIN5 days with a carryover, is the pool-level return at the final dividend
above 1 after the takeout, and how does it depend on carryover size? The spec is
`docs/specs/win5-carryover-study.md`, signed by the owner on #135 (2026-11-25) with every open decision answered,
including the pool (answer B, net of refunds).

**Nothing is adopted, and no betting or staking rule follows from this report.** The coverage scored is the
pool's own return to its bettors, not a ticket anyone can buy. Acting on it is the owner's decision.

Every figure here is an aggregate over the WIN5 history table (netkeiba WIN5 results, one row per WIN5 day). No
licensed source (JRDB, JRA-VAN) enters the study, no per-day row appears here, and no scraped page is in the tree.

- Registry family `win5-carryover-v1`, spec sha256
  `ce33e1bd30e7b6f6a4805a0b7ce7597d5dc4296e8f29e4df439b9d7f56206c45`, frozen with
  `uv run horsetrack japan win5-study freeze` (2026-11-25 06:43:35 UTC) before the study was scored. The spec is
  `src/horsetrack/countries/japan/config/win5_study.yaml`, as committed.
- Scored once with `uv run horsetrack japan win5-study score` (06:43:41 UTC): run id
  `f84043280be6fa46f2e40b1fed3ef5b1af6e14153318b0cc6bd8781c1a5c5e72`, status `scored`, code at `e32af6c`
  (`countries/japan/research/win5.py`, `win5_study.py`).
- Data build: `processed/jra/win5.parquet@60be24d31bd95181a6fb272a6ab24333b7346db2d9571f6f9b2cf9cb380dc655`, the
  backfill of 2026-11-25: 910 WIN5 days from 2011-04-24 to 2026-11-23, 17 with a carryover in.
- The run is listed in `docs/experiments/registry/japan/`.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| (1) Coverage | The pool-level return: a day's value is `payout_yen / net_pool`, every winning combination of a dead heat counted, 0 on a no-hit day. A live top-k ticket rule waits for logged WIN5 legs (#130). | #135, 2026-11-25 |
| (2) Metric | The mean per-day value of a bin against 1.0. | #135 |
| (3) Bins | Carryover in as a multiple of the break-even carryover `0.30 x net pool`: (0, 0.5), [0.5, 1), [1, 2), [2, inf). | #135 |
| (4) Estimand | The 6億円 payout cap is modelled; boosted-takeout days count at their own rate. | #135 |
| (5) Interval and rule | Day-block bootstrap, 10,000 resamples, seed 20261125; a positive edge is a bin whose one-sided 95% lower bound is above 1.0. | #135 |
| (B) Pool | Net of refunds, implied from the money identity from 2014-06-07; gross sales before it (takeout unknown), counted separately. The refund hypothesis is unverified (docs/ledger.md). | #135 |
| D14 | Development days only: holdout days are dropped before scoring. The holdout (A2, from 2026-11-28) was not opened, and the history ends 2026-11-23. | #14; #71 |

## Hypothesis

Written in the spec before scoring. The null is the starting point: no carryover bin's pool-level return is above
1 by its one-sided 95% lower bound. The study is shown wrong for a bin when that bin's lower bound is above 1.0.

What the arithmetic predicts, stated here because it governs how to read the result: from 2014-06-07 the net pool
is implied by the money, so on a day that is won and passes nothing on, the value is exactly
`(1 - t) + t_be x ratio` = `0.70 + 0.30 x ratio` at the standard 30% takeout. A bin above ratio 1 is therefore at
or above 1.0 by construction whenever its days are won; the only empirical content of a positive bin is that its
days were won and none passed a capped excess on. The pre-2014 days follow the same identity with an unknown
effective takeout.

## Variants tried

One variant, one run: coverage `pool-level`, pool `net-of-refunds` (family size 1). No variant failed, and no other
variant was scored. Split: all 910 development days; holdout days excluded: 0.

| Carryover / break-even | Days | Of them gross-pool (pre-2014) | Mean value | One-sided 95% lower bound | Positive edge |
| --- | --- | --- | --- | --- | --- |
| (0, 0.5) | 7 | 2 | 0.816 | 0.802 | no |
| [0.5, 1) | 6 | 3 | 0.949 | 0.930 | no |
| [1, 2) | 4 | 3 | 1.059 | 1.046 | yes, by the pre-registered rule |
| [2, inf) | 0 | 0 | - | - | - (empty) |

- Carryover days: 17 in the whole history, 8 of them before 2014-06-07 and scored at gross sales
  (`gross_pool_carryover_days` = 8); 9 scored on the implied net pool.
- Descriptive counts on the same build, not registry metrics: none of the 17 carryover days was a no-hit day, none
  passed a capped carryover on, and one boosted-takeout day falls in [0.5, 1). Over all 910 days, 11 had no winner.
- No upper bound was pre-registered, so the table gives none.

## Alternatives rejected and why

- **The gross pool (netkeiba's 発売金額) as the pool** (answer A): the money falls short of it on 122 of 743 days
  from 2014-06-07, never over, so the gross pool understates the return on those days. Rejected by the owner before
  scoring (answer B); it was never scored.
- **A buyable ticket rule (top-k per leg)**: needs the WIN5 legs' odds at decision time, which are not logged yet
  (#130). Deferred by the owner's decision (1).
- **Dropping the pre-2014 days** (takeout unknown): the owner kept them at gross sales and asked for their count
  separately. Not run as a variant.
- **A model of the no-hit (rollover) probability**: the bootstrap resamples days and cannot see a risk that never
  occurred in the sample. Modelling it is a new modelling decision, not part of the signed spec; proposed below.

## What this rules out

Very little. By the pre-registered rule, the [1, 2) bin shows a positive edge: mean 1.059, lower bound 1.046, on 4
days. That result is the arithmetic above plus the fact that all 4 days were won:

- 3 of its 4 days are pre-2014 days scored at gross sales with an unknown takeout; 1 is on the net pool.
- The day bootstrap of 4 won days has no no-hit draw to resample, so its lower bound reflects only the spread of
  the carryover ratios. With 0 no-hit days in 4, a one-sided 95% upper bound on the no-hit rate is about 53% (0 in
  17 over all carryover days: about 16%); a no-hit rate above about 5.6% would bring the bin's expected value
  below 1. The interval therefore understates the uncertainty that matters.
- The pool-level return is the average return of the pool's own bettors on the final pool, carryover chasers
  included. It is not the return of any ticket, and it says nothing about which combinations to buy; any stake
  added also enlarges the pool and lowers the ratio.

The bins below break-even have means under 1 (0.816 and 0.949), as `0.70 + 0.30 x ratio` predicts; with no upper
bound pre-registered, they are not shown below 1 by a test. The [2, inf) bin has no days, so nothing is known
about large carryovers. With 17 carryover days in fifteen years, the null result stands for any buyable WIN5 rule:
this study does not show that one exists.

## Follow-ups filed

Filed as `status:proposed` tickets under epic #72, for the owner to promote or decline:

1. #322: pre-register a live WIN5 ticket rule (top-k per leg) on carryover days, scored at the final dividend on
   future carryover days; it needs WIN5 legs logged (#130).
2. #323: estimate the WIN5 rollover (no-hit) risk on carryover days from all development days (by pool size and
   leg favourites), so a carryover day's expected value carries the risk the day bootstrap cannot see.
3. #324: verify the refund hypothesis behind the net pool from a JRA page (返還 on a scratched horse), as the
   ledger's row asks.
4. #325: grow the WIN5 carryover sample, recording each new carryover day's pool-level return as new days run (the
   [1, 2) bin has 1 net-pool day; the [2, inf) bin has none).

## On the leaderboard (2026-11-27)

Under the reframe (epic #447; D16 to D18 as amended 2026-11-27 in #448), the results above are measured results, not
adoption verdicts. Nothing is adopted or rejected: every option, model and member will be a row on the planned
`horsetrack japan leaderboard` (#452; this revision does not have the command yet), ranked by its shrunk return per
¥100 at the final price, and a row with no measured return will be listed after the ranking, never dropped. Where
this report says "not adopted", "rules out" or "rejected", read a measured position. This section reopens and
re-runs nothing.

- Registry family: `win5-carryover-v1`.
- As task 4 (#451, planned) specifies, each carryover-ratio bin's pool-level day value will be a measured WIN5 cell
  row with its one-sided lower bound only, labelled look-ahead, because the net pool is known only after the close.
  The gross-pool carryover-day metric will give no row.
- "Nothing is adopted" and "the null result stands for any buyable WIN5 rule" now read as a measured position: the
  rows are pool-level values, not a buyable ticket rule, and will rank as such. The follow-ups #322 to #325 stay as
  filed.
