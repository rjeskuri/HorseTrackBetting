# N1 pre-registration: the final-board segment scan of six pools (Japan, #430)

**Status: registered and frozen.** Claude registered this spec under the owner's reframe of
2026-11-27 on #430: every cell is ranked, none is closed, there is no owner sign-off, and each
alternative that runs is a counted variant. The spec was frozen in the japan registry
(`default_registry('japan').freeze_family`) before the run read any store row.

- Family `japan-n1-pool-scan`, spec sha256
  `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`.
- The source of truth is `prereg_spec()` in
  `src/horsetrack/countries/japan/research/pool_scan_run.py`, quoted in full below. `run_scan`
  refuses to read the store unless the registry holds exactly that spec frozen for the family, so
  the code cannot run a spec other than the frozen one.
- This file cites no JRA-VAN figure (D21). The dates in it (2014-06-07, 2024-01-01) are
  pre-registration parameters, not measurements.

## In brief

- **Question (E8).** In each of six pools (Win, Place, Quinella, Wide, Exacta, Trio; flat races
  only), which pre-registered cells of the final board does the crowd misprice? Each cell is
  ranked by its shrunk expected return per unit staked at the final price, takeout included. The
  null, the starting point, is that every cell is priced like its pool.
- **Bet.** A cell is a flat box: one unit on every priced combination of the pool's final board in
  the cell's races (and odds band). A combination returns its dividend / 100 when it paid.
- **Cells (106 per pool).** `all`; each odds band; each level of class (newcomer, maiden,
  conditions, open, graded), field size, favourite strength (the WIN favourite's final odds), pool
  size (tertiles per pool and calendar year), surface and venue; then class crossed with pool
  size, with favourite strength and with odds band. The band labels per pool are:

  | Pool | Bands |
  | --- | --- |
  | WIN | lt2, 2-3, 3-5, 5-10, 10-20, 20-50, ge50 |
  | PLACE (low end) | lt1.2, 1.2-1.5, 1.5-2, 2-3, 3-5, 5-10, ge10 |
  | QUINELLA | lt5, 5-10, 10-20, 20-50, 50-100, 100-300, ge300 |
  | QUINELLA_PLACE, Wide (low end) | lt2, 2-5, 5-10, 10-20, 20-50, 50-100, ge100 |
  | FORECAST, Exacta | lt10, 10-20, 20-50, 50-100, 100-300, 300-1000, ge1000 |
  | TRIO | lt10, 10-30, 30-100, 100-300, 300-1000, 1000-3000, ge3000 |

- **Shrinkage.** For the one-winner pools, #78's hierarchical omega: odds-band fixed effects for
  the favourite-longshot bias, and crossed random effects for the six segments and the three
  class interactions. For Place and Wide, whose dividend is not their odds, a normal
  empirical-Bayes shrink of the realised return over each band's baseline.
- **Readings and spans (the variants).** Two omega readings, `omega_full` (the band effects
  included; it ranks the cells) and `segment_effect` (the segment effect alone), on two spans,
  from 2014-06-07 and from 2024-01-01 (reported apart). That makes one registry record per pool,
  reading and span: 24 variants, and the family size is reported on every row as
  `variants_searched`.
- **SPA and StepM.** These run per pool over the pool's cell rules (family
  `japan-n1-pool-scan-cells-<pool>`, one record per cell), with no bet as the benchmark, meetings
  as clusters, block size 12, 1,000 bootstrap draws and StepM at 5%. They are reported beside the
  ranking. They rank nothing and close nothing.
- **Exclusions.** Holdout days (D14), anything before 2014-06-07 (D10), races not run, jump
  races, races with an unknown class, field size, surface or venue, and races with no priced WIN
  favourite. A (race, pool) is also excluded when its payout may be withheld (a premium boost, as
  in #131), when an age-restricted boost matches it (the jravan conditions code cannot show the
  age), or when the pool was not sold.
- **What a result means.** The odds band, the favourite strength and the pool size are read from
  the final board, which is look-ahead for selection (#88). A cell is a calibration target at the
  final price and a candidate for a decision-time rule. It is not a betting rule.

## The frozen spec

```json
{
  "family_id": "japan-n1-pool-scan",
  "ticket": 430,
  "hypothesis_id": "E8",
  "study": "pool-scan",
  "sources": ["jravan"],
  "registration": "registered and frozen by Claude under the owner's reframe of 2026-11-27 on #430 (every cell ranked, none closed, no owner sign-off); every alternative run is a counted variant",
  "hypothesis": "E8: in some pre-registered cell of a JRA pool's final board the hierarchical omega (realised wins over takeout-adjusted implied probability; for Place and Wide the realised return over the payout rate) differs from 1 once the favourite-longshot bias is taken out as odds-band fixed effects, and some cell's expected return per unit staked at the final price reaches break-even (omega at or above 1/payout rate)",
  "null": "every cell is priced like its pool: omega 1, flat rho 1, expected return the payout rate, below break-even. The null is the starting point.",
  "pools": {
    "WIN": "Win",
    "PLACE": "Place (banded and priced at the low end of the final range)",
    "QUINELLA": "Quinella",
    "QUINELLA_PLACE": "Wide (banded and priced at the low end of the final range)",
    "FORECAST": "Exacta",
    "TRIO": "Trio"
  },
  "pools_left_out": "TIERCE (trifecta) and BRACKET_QUINELLA: a follow-up after N1 that runs anyway (owner, 2026-11-27)",
  "span": {
    "rule": "development days from max(date_from, 2014-06-07) (D10) to the run's date_to; holdout days (core.eval.holdout.is_holdout_day, D14) are never read",
    "variants": {
      "from-2014-06-07": "the full span (the ranking, the rows and the candidate file)",
      "from-2024-01-01": "2024-01-01 onward, reported apart: untouched by the earlier pattern search (problem statement 5.1)"
    }
  },
  "store": "processed/jravan: final_odds (the six pools and their __pool__ rows), dividends (filtered by the year's development race ids) and races, read one calendar year at a time through models.v0.store.read_store_table(source='jravan')",
  "exclusions": ["a holdout day (D14) and any day before 2014-06-07 (D10)", "a race whose status is not 'run'", "a jump race (surface 'jump'): flat races only (owner, 2026-11-27)", "a race whose surface is not turf or dirt, whose class is not in pool_scan.CLASS_GROUPS, whose field size is not 1-18 or whose venue is not a JRA venue", "a race with no priced WIN combination (no favourite strength)", "a race and pool whose payout may be withheld (takeout.race_takeout boost_may_be_withheld, a premium boost; as in #131)", "a race and pool an age-restricted boost entry matches: the jravan conditions code cannot show the age restriction, so race_takeout(..., race_class_raw=None) raises RaceConditionsNeeded and the rate is unknown", "a race and pool with no __pool__ total (pool not sold)", "a refunded combination is staked nowhere (pool_scan.combination_returns)"],
  "bet": "flat box: one unit on every priced combination of the pool's final board in the cell; a combination returns its dividend / 100 when it paid, else 0",
  "metric": {
    "flat_rho": "sum(return) / sum(payout rate) over the cell's staked combinations: 1 is priced like the pool; break-even is 1 / payout rate",
    "mean_return": "sum(return) / n_bets: the return per unit staked at the final dividend",
    "payout_rate": "the mean of the race and pool payout rates (takeout.race_takeout(race_id, pool), boosts included) over the cell's bets",
    "se_mean_return": "the D7 meeting bootstrap: n_boot resamples of the span's meetings with replacement, the sd of the ratio sum(return) / sum(bets)"
  },
  "odds_bands": {
    "WIN": [2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
    "PLACE": [1.2, 1.5, 2.0, 3.0, 5.0, 10.0],
    "QUINELLA": [5.0, 10.0, 20.0, 50.0, 100.0, 300.0],
    "QUINELLA_PLACE": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "FORECAST": [10.0, 20.0, 50.0, 100.0, 300.0, 1000.0],
    "TRIO": [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
  },
  "odds_band_rule": "pool_scan.odds_band: [e_i, e_i+1), lower edge inclusive; Place and Wide band on the low end of the final range",
  "segments": {
    "class": "pool_scan.class_group of races.race_class: newcomer (DEBUT), maiden, conditions (C1-C3), open (OP, L), graded (G1-G3)",
    "field": "races.field_size: le4, 5-10, 11-14, 15-18 (pool_scan.race_cells)",
    "fav": "the race's WIN favourite: the lowest priced final WIN odds; le1.9, 2.0-2.9, 3.0-4.9, ge5.0, lower edge inclusive (pool_scan.race_cells)",
    "pool_size": "the pool's __pool__ total; tertiles per pool and calendar year: edges at the 1/3 and 2/3 quantiles (numpy, linear) of that year's scored races of the pool; low below the first edge, mid to the second, high from it (pool_scan.race_cells)",
    "surface": "turf, dirt",
    "venue": "the ten JRA venues",
    "levels": {
      "class": ["newcomer", "maiden", "conditions", "open", "graded"],
      "field": ["le4", "5-10", "11-14", "15-18"],
      "fav": ["le1.9", "2.0-2.9", "3.0-4.9", "ge5.0"],
      "pool_size": ["low", "mid", "high"],
      "surface": ["turf", "dirt"],
      "venue": ["SAP", "HAK", "FKS", "NII", "TOK", "NAK", "CKY", "KYO", "HSN", "KOK"]
    }
  },
  "cells": {
    "rule": "per pool, in this order: 'all'; 'band:<label>' for each odds band; '<segment>:<level>' for each segment and level; then 'class:<group>&<other>:<level>' for other in pool_size, fav and band, every class group and level; a cell with no bets is a row with null figures",
    "crossed_with_class": ["pool_size", "fav", "band"]
  },
  "shrinkage": {
    "WIN, QUINELLA, FORECAST, TRIO": {
      "model": "#78, core.eval.segment_roi.fit_segment_roi: ln omega = mu[odds band] + the sum of crossed random effects, one N(0, tau_c^2) per column, the module's priors (MU_PRIOR_SD 1.0, TAU_PRIOR_SCALE 0.1)",
      "columns": ["class", "field", "fav", "pool_size", "surface", "venue", "class_x_pool_size", "class_x_fav", "class_x_band"],
      "interactions": "class_x_<other> is the level '<class group>|<level>'; a cross cell's query names class, its other level and the interaction",
      "draws": 1000,
      "seed": "numpy SeedSequence([run seed, pool index, span index]).generate_state(1)",
      "grouped_bets": "pool_scan.grouped_bets by race, then per (segment levels, band, takeout, odds group): odds below 10 exact, from 10 on a log grid of ratio 1.1 (group k holds [10 x 1.1^k, 10 x 1.1^(k+1))); a group's odds are its harmonic mean, so its mean implied probability is exact (only losers' terms see the grid)"
    },
    "PLACE, QUINELLA_PLACE": {
      "why": "a Place or Wide dividend is not its odds (it falls in the final low-high range and depends on the co-placers), so #78's likelihood, exact for one-winner pools, would not measure the return; the shrink is on the realised return",
      "model": "normal empirical Bayes per segment family. Band baselines rho_b = sum(return) / sum(rate) of the pool's band b on the span (fixed effects, not shrunk). A cell's effect e = sum(return) / sum(rate x rho_b) over its bets, with its meeting-bootstrap se (rho_b held at its estimate); tau^2 of a family = max(0, mean over its cells with bets of (e - 1)^2 - se^2); B = tau^2 / (tau^2 + se^2) (0 when both are 0); shrunk effect 1 + B (e - 1), sd sqrt(B) se",
      "families": "class, field, fav, pool_size, surface, venue, class&pool_size, class&fav, class&band"
    }
  },
  "readings": {
    "omega_full": "the cell's omega with its band effects: #78's mean over the cell's bets of exp(mu[band] + every random effect of the bet's levels); empirical Bayes: (sum(rate x rho_b) / sum(rate)) x the shrunk effect, and rho itself for 'all' and the band cells. The ranking, the rows and the candidate file use it",
    "segment_effect": "the segment effect alone: #78's exp(sum of the cell's named random effects); empirical Bayes: the shrunk effect. Undefined for 'all' and band cells"
  },
  "bounds": "omega is the posterior mean (#78: of the draws); one-sided 95% bounds each: omega_lo the 5% and omega_hi the 95% posterior quantile (#78 draws; empirical Bayes: omega -/+ 1.6449 sd); P(omega > 1/rate) is the posterior probability that omega x payout_rate exceeds 1 (#78: the share of draws; empirical Bayes: the normal tail)",
  "ranking": "every cell of every pool, full span, omega_full: shrunk_edge = omega x payout_rate - 1 (expected return per unit at the final price, takeout included, minus 1), descending; cells without one last; ties in pool and spec order",
  "bootstrap": {
    "unit": "meeting (races.meeting_id), D7",
    "n_boot": 1000,
    "level": 0.95,
    "draws": "multinomial counts of the span's meetings, shared by the pool's cells",
    "seed": "numpy default_rng(SeedSequence([run seed, pool index, span index]))"
  },
  "spa_stepm": {
    "families": "per pool, <family_id>-cells-<pool>: one record per pre-registered cell, its flat-box rule on the full span, scored with bets and failed without; SPA and StepM test exactly the family's non-failed variants (core.eval.snooping)",
    "loss": "per meeting, the sum over the cell's bets of stake minus return",
    "benchmark": "no bet (loss 0): a rule beats it when it returns more than it stakes",
    "clusters": "meetings in time order, the stationary bootstrap of arch",
    "block_size": 12,
    "n_boot": 1000,
    "stepm_level": 0.05,
    "seed": "the run seed",
    "reported": "beside the ranking; they rank and close nothing"
  },
  "variants": {
    "family": "one record per pool, reading and span, variant '<pool>/<reading>/<span>': 6 x 2 x 2; scored when the pool's shrinkage fitted on the span, failed otherwise; its metrics are every cell's omega with bounds, P(omega > 1/rate), flat rho and mean return",
    "family_size": "registry.family_size(family_id) after the run is variants_searched on every row; each SPA family's size is reported beside its SPA p",
    "failures": "a pool with a data error in any year (pool_scan refuses a malformed board or dividend row) is failed on both spans, its cell rules too; a shrinkage that does not fit fails that pool and span; the private report gives each reason"
  },
  "outputs": {
    "private_report": "$HORSETRACK_DATA_DIR/reports/pool-scan-japan-<spec hash>.md",
    "candidate_cells": "$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json, every cell in rank order in the candidate-cell schema plus omega, omega_lo, omega_hi and variants_searched",
    "public": "docs/experiments/pool-scan-japan.md: the run, no figure (D21)"
  },
  "selection_hygiene": "odds band, favourite strength and pool size are read from the final board: look-ahead for selection (#88). A cell is a calibration target at the final price and a candidate for a decision-time rule, never a betting rule",
  "decisions": {
    "D7": "the meeting is the resampling unit; 95% one-sided",
    "D10": "no price before 2014-06-07",
    "D14": "development days only; LockedHoldout is never opened",
    "D21": "every figure is withheld; outputs through core.eval.reports.write_report"
  }
}
```

## Choices registered here, and the alternatives not run

These were the owner's to decide. The reframe gives them to Claude. Each was decided before any
store row was read, and only the alternatives that run count as variants:

- **Span.** From 2014-06-07 to the run's last day, with 2024-01-01 onward reported apart as a
  variant. The earlier start dates of other studies were not run.
- **Jump races.** Excluded (owner, 2026-11-27: flat races only). They do not form a surface level.
- **Trifecta and bracket quinella.** Left out. They are a follow-up after N1 (owner, 2026-11-27).
- **Odds-band edges per pool.** Those in the table above. No other edges were run.
- **Pool-size tertiles.** Per pool and calendar year, so that the growth of sales over the years
  does not sort the cells by year. A fixed yen basis was not run.
- **Cross cells.** Class crossed with pool size, favourite strength and odds band, because these
  are the owner's axes (maidens and debuts against big graded races, around the favourite, by
  price). Other crosses were not run.
- **Shrinkage for Place and Wide.** The empirical-Bayes shrink of the realised return. #78 at the
  low end of the range was not run, because its omega would describe the placing probability and
  not the return.
- **Odds in the fit.** Exact below 10, and a 1.1 log grid from 10 at the harmonic-mean price.
  Exact odds everywhere were not run, because the long tail of a Trio board makes too many
  grouped rows for the 16 GB limit.
