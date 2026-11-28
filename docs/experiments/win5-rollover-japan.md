# Experiment report: the WIN5 rollover (no-hit) risk (Japan, #323, P1)

Ticket #323, follow-up of #135 (`docs/experiments/win5-carryover-japan.md`). On WIN5 days with a carryover, how often
does a day have no winning ticket (a rollover), with what interval, and what is each carryover bin's expected value
net of that risk? The spec is `docs/specs/win5-rollover-study.md`, frozen before any scored read.

**Nothing is adopted, and no betting or staking rule follows from this report.** The figures are the pool's own
return to its bettors, not a ticket anyone can buy (the caveat of #135 stands).

Every figure here is an aggregate over the WIN5 history table (netkeiba WIN5 results, one row per WIN5 day). No
licensed source (JRDB, JRA-VAN) enters the study, no per-day row appears here, and no scraped page is in the tree.

- Registry family `win5-rollover-v1`, spec sha256 `5ad78366c7c0618a312b31ce77e35b47db7e7a0d780c6c715a1527e3ae15717a`,
  frozen with `uv run horsetrack japan win5-rollover freeze` before the study was scored. The spec is
  `src/horsetrack/countries/japan/config/win5_rollover.yaml`, as committed.
- Scored once with `uv run horsetrack japan win5-rollover score`: run id
  `77d3b0c0825ed40f268c58e869bc196848b48cfa59d9065813a34577bbec25b9`, status `scored`, code at `854a7ae`
  (`countries/japan/research/win5_rollover.py`).
- Data build: `processed/jra/win5.parquet@60be24d31bd95181a6fb272a6ab24333b7346db2d9571f6f9b2cf9cb380dc655`, the
  same backfill as #135: 910 WIN5 days from 2011-04-24 to 2026-11-23, 17 with a carryover in, 0 holdout days. Holdout
  A2 (from 2026-11-28) was not opened; D14 holds.
- The run is listed in `docs/experiments/registry/japan/win5-rollover-v1.md`.

## Hypothesis

Written in the spec before scoring. The null: a carryover day's expected value is below 1 once rollover risk is
counted. A no-hit day pays 0 and passes the money on, so a bin's expected value is `(1 - q) x won value` with `q` the
no-hit rate and `won value` the mean day value of the bin's won days; it reaches 1 at `q* = 1 - 1 / won value`. The
**pre-registered headline** is the pooled figure: `q` the no-hit rate over all development days, and its value at the
upper bound of `q`. A bin's own-rate figure is secondary, since the bin's own rate on a handful of days carries almost
no information. The study is shown wrong for a bin only if its headline stays above 1 at the upper bound.

## Variants tried

One variant, one run: Clopper-Pearson interval at 95%, bins and net pool as #135 (family size 1). No variant failed
and no other variant was scored. Split: all 910 development days; holdout days excluded: 0.

Overall no-hit rate: **11 of 910 days, 1.21%, 95% interval [0.60%, 2.15%]**. The 17 carryover days contain none of
the 11 (0 of 17); the 893 days without a carryover contain all 11.

| Carryover / break-even | Days | No-hit | No-hit rate [95% interval] | Won value | Break-even no-hit rate | EV at bin's own rate | EV at its upper bound | Pooled EV (headline) | Pooled EV at upper bound (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0, 0.5) | 7 | 0 | 0 [0, 40.96%] | 0.816 | 0 (never above 1) | 0.816 | 0.482 | 0.807 | 0.799 |
| [0.5, 1) | 6 | 0 | 0 [0, 45.93%] | 0.949 | 0 (never above 1) | 0.949 | 0.513 | 0.938 | 0.929 |
| [1, 2) | 4 | 0 | 0 [0, 60.24%] | 1.059 | 5.60% | 1.059 | 0.421 | 1.047 | 1.037 |
| [2, inf) | 0 | - | - | - | - | - | - | - | - |

The pooled columns use `q` = 1.21% and its upper bound 2.15%; the own-rate columns use the bin's own rate (0) and the
bin's own upper bound. The own-rate EV equals #135's mean day value by construction.

## Alternatives rejected and why

- **A rate by pool size or leg favourites** (the ticket's Context mentions pool): not run. With 11 no-hit days in
  total and none among the 17 carryover days, a rate model conditional on pool or favourites has too few events to
  fit; it would be a second modelling decision, not part of this frozen spec. Proposed as a follow-up below.
- **Wilson or a bootstrap interval for the rate**: not run. Zero counts are the case of interest, and the exact
  Clopper-Pearson interval is defined there with a non-trivial upper bound; a bootstrap of 0 of 4 has none.
- **The bin's own rate as the headline**: pre-registered as secondary. Its upper bound (41% to 60%) is wide enough
  to drive every bin's EV below 1, which says only that 4 to 7 days cannot rule out a high no-hit rate.

## What this rules out

Little, and the reading depends on which rate is used:

- **At the pooled rate** (headline), the [1, 2) bin stays above 1: 1.047 at 1.21% and 1.037 at the 2.15% upper bound.
  Its break-even rate, 5.60%, is above the upper bound of the overall interval, so rollover risk at the overall rate
  does not take the bin's EV below 1. The two bins under break-even are below 1 by construction at any rate (their
  won value is under 1, so their break-even no-hit rate is 0).
- **At the bin's own rate**, nothing is shown: 0 of 4 days has an upper bound of 60.24%, which includes the 5.60%
  break-even rate and everything above it. At that bound the bin's EV is 0.421. The sample of 4 days cannot rule
  out a rollover risk that would erase the bin's edge, which is the point of #135's follow-up.
- **Which rate applies is the open question**, and the pooled rate is not a sound stand-in for it. It averages over
  893 days without a carryover, and the days that mattered, those with a carryover, have pools that differ and
  are, by construction, the day after a no-hit or capped day. The 17 carryover days alone (0 no-hit) give an upper
  bound of about 19.5% (`1 - 0.025^(1/17)`, arithmetic from the 17 and 0, not a registry metric), above 5.60%. So
  the bin's break-even rate lies inside the interval of the carryover days and outside the interval of all days: the
  data cannot tell which is right.
- The pooled and own-rate figures are on the same 4 won days of the [1, 2) bin, 3 of them pre-2014 at gross sales
  (#135), so the won value carries the same limits as #135's result: it is `0.70 + 0.30 x ratio` on a won day by the
  money identity, not an independent measurement.
- A no-hit rate above about 5.6% takes the [1, 2) bin's EV below 1, as the ticket stated.

## Leaderboard rows

None added by this report. The no-hit rate is a descriptive rate of a pool event, and the adjusted EV is the pool's
own return, not the return per ¥100 staked on a defined selection rule, so it gets no D16 rank (as the WIN5 cells in
#135's leaderboard section). The run's `no_hit_rate` metrics are in the registry summary for all development days
and each non-empty bin; the [2, inf) bin had no days and wrote no metric, so it is unmeasured.

## Follow-ups filed

No new ticket is filed from here. Candidates for the owner to propose or decline:

1. A rate of no-hit conditional on the WIN5 pool size and the legs' favourites, once WIN5 legs are logged (#130);
   with 11 no-hit days in fifteen years it cannot be fitted on the current history.
2. New carryover days as they run (#325): each adds one day to a bin whose own interval is still above the
   break-even rate. Each 0 of n narrows the upper bound to `1 - 0.025^(1/n)`; 0 of 65 would be below the 5.60% break-even rate.
3. A live ticket rule on carryover days (#322) should carry this risk in its EV: its day-level rollover rate is the
   figure to use, not the pool's.
