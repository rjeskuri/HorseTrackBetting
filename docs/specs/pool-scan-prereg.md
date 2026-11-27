# N1 pre-registration: the final-board segment scan of six pools (Japan, #430)

**Status: registered and frozen by Claude, with no owner sign-off, under the owner's reframe of
2026-11-27 on #430.** That comment amends the ticket body. On sign-off it reads: Outline 2,
"Stop with `harness-board needs-input N` for the owner's sign-off. Resume on the answer." →
removed. Claude freezes the spec in the registry. And under Rules: "The cell list and metrics are
pre-registered in a spec the owner approves before the run: a needs-input step" → "Claude
pre-registers the cell list and metrics and freezes them in the registry before the run. No
owner sign-off." On the kill rule it reads: Request, "It names the cells a model should be aimed
at, or closes them." → "It ranks every cell by expected return per ¥100 at the final price
(takeout included, CI beside it). None is closed, and models aim at the top-ranked cells first."
It also removes the kill rule from the Context, and the definition of done becomes "the file
lists every cell, ranked." The frozen spec quotes these lines under `authority`. Every
alternative that runs is a counted variant. The spec was frozen in the japan registry
(`default_registry('japan').freeze_family`) before the run read any store row.

- Family `japan-n1-pool-scan-v2`, spec sha256
  `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6`.
- **It supersedes v1**, family `japan-n1-pool-scan`, spec sha256
  `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`, which was frozen and run on
  2026-11-27. Review of PR #496 found four defects in how v1 scored:
  - it scored a sold board with no dividend row as if every ticket lost;
  - it took a cell's expected return as mean omega times mean payout rate, which is wrong where
    boosted and ordinary rates mix;
  - it checked SPA's columns against every cell any earlier run had scored;
  - it gave no Place or Wide reading for a cell whose odds band returned nothing.

  v2 fixes each one (see `supersedes` in the spec). v1's records stay in the registry,
  superseded, and nothing is deleted. v1's variants count toward variants searched. This file
  quoted v1's spec in full up to commit `39227d5`.
- The source of truth is `prereg_spec()` in
  `src/horsetrack/countries/japan/research/pool_scan_run.py`, quoted in full below. `run_scan`
  refuses to read the store unless the registry holds exactly that spec frozen for the family, so
  the code cannot run a spec other than the frozen one.
- This file cites no JRA-VAN figure (D21). The dates in it (2014-06-07, 2024-01-01) are
  pre-registration parameters, not measurements, and the cell counts are counts of the design.

## In brief

- **Question (E8).** In each of six pools (Win, Place, Quinella, Wide, Exacta, Trio; flat races
  only), which pre-registered cells of the final board does the crowd misprice? Each cell is
  ranked by its shrunk expected return per unit staked at the final price, takeout included.
  None is closed: there is no kill rule (owner's reframe). The null, the starting point, is that
  every cell is priced like its pool.
- **Bet.** A cell is a flat box: one unit on every priced combination of the pool's final board in
  the cell's races (and odds band). A combination returns its dividend / 100 when it paid.
- **Cells (106 per pool).** The list is the same for the ranking, the reports, the candidate file
  and SPA/StepM:
  - `all`, and each odds band (7);
  - each level of class (newcomer, maiden, conditions, open, graded), field size, favourite
    strength (the WIN favourite's final odds), pool size (tertiles per pool and calendar year),
    surface and venue (28);
  - class crossed with pool size (15), with favourite strength (20) and with odds band (35).

  The band labels per pool are:

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
  class interactions. A cell's omega weighs each bet by its payout rate, so omega times the
  cell's mean payout rate is its mean expected return, even where boosted and ordinary races mix.
  For Place and Wide, whose dividend is not their odds, Claude chose a normal empirical-Bayes
  shrink of the realised return over each band's baseline. The ticket left that choice to the
  spec. A cell whose bets all sit in a band that returned nothing keeps the prior effect and reads
  its baseline, 0.
- **Readings and spans (the variants).** Two omega readings on two spans:
  - the readings: `omega_full` (the band effects included; it ranks the cells) and
    `segment_effect` (the segment effect alone);
  - the spans: from 2014-06-07, and from 2024-01-01 (reported apart).

  That makes one registry record per pool, reading and span: 24 variants. Variants searched, on
  every row, is this family's size plus v1's.
- **SPA and StepM.** These run per pool over the pool's cell rules (family
  `japan-n1-pool-scan-v2-cells-<pool>`, one record per cell per run), with these settings:
  - no bet is the benchmark;
  - meetings are the clusters;
  - block size 12, 1,000 bootstrap draws, and StepM at 5%.

  They test this run's cell records, so a narrower rerun is tested on its own cells. They are
  reported beside the ranking. They rank nothing and close nothing.
- **Exclusions.** These races are left out:
  - holdout days (D14), anything before 2014-06-07 (D10), and races not run;
  - jump races;
  - races with an unknown class, field size, surface or venue, and races with no priced WIN
    favourite.

  A (race, pool) is also left out in these cases:
  - its payout may be withheld (a premium boost, as in #131);
  - an age-restricted boost matches it (the jravan conditions code cannot show the age);
  - the pool was not sold;
  - it is unsettled: its board was sold, but the dividends hold no paying row for it. This
    covers a pool not established, a special payout (the jravan dividends builder writes no row
    for either) and dividends not yet delivered. Such a pool is counted, never scored as all-lose.
- **What the store reader guarantees.** `read_store_table` pushes the holdout exclusion into the
  Parquet scan as its filter, so no holdout row is materialized, and `dividends` is filtered
  inside its scan to the development race ids. Parquet prunes row groups by their min/max
  statistics only, and the jravan builders write one row group per year. A year whose row group
  holds holdout days is therefore decoded from disk, and its holdout rows are dropped inside the
  scan. Row groups are not yet isolated by day; proposed #502 does that.
- **What a result means.** The odds band, the favourite strength and the pool size are read from
  the final board, which is look-ahead for selection (#88). A cell is a calibration target at the
  final price and a candidate for a decision-time rule. It is not a betting rule.

## The frozen spec

```json
{
  "family_id": "japan-n1-pool-scan-v2",
  "ticket": 430,
  "hypothesis_id": "E8",
  "study": "pool-scan",
  "sources": ["jravan"],
  "registration": "registered and frozen by Claude in the japan registry before any scored read, with no owner sign-off, under the owner's reframe of 2026-11-27 on #430 ('authority'); every alternative run is a counted variant",
  "authority": {
    "where": "the owner's comment 'Reframe, owner 2026-11-27' on #430, which amends the ticket body; quoted as written, its arrows as ->",
    "sign_off": "Outline 2: \"Stop with `harness-board needs-input N` for the owner's sign-off. Resume on the answer.\" -> removed. Claude freezes the spec in the registry. | Rules: \"The cell list and metrics are pre-registered in a spec the owner approves before the run: a needs-input step\" -> \"Claude pre-registers the cell list and metrics and freezes them in the registry before the run. No owner sign-off.\"",
    "ranking_not_a_kill_rule": "Request: \"It names the cells a model should be aimed at, or closes them.\" -> \"It ranks every cell by expected return per 100 yen at the final price (takeout included, CI beside it). None is closed, and models aim at the top-ranked cells first.\" | Context: \"Kill rule (plan review, section 6...)\" -> removed. | Definition of done: \"the candidate-cell file lists every cell the kill rule leaves open, and is empty when none is open\" -> \"the file lists every cell, ranked.\"",
    "items_for_the_spec": "Context: \"Items for the spec, the owner's to decide\" -> \"Claude decides and registers them; each alternative is a counted variant.\""
  },
  "supersedes": {
    "family_id": "japan-n1-pool-scan",
    "spec_sha256": "74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9",
    "why": "review of PR #496 found that v1 scored a sold board with no dividend row as if every ticket lost, took a cell's expected return as its mean omega times its mean payout rate (wrong when ordinary and boosted rates mix), checked SPA's columns against every cell any earlier run scored, and gave no Place or Wide reading for a cell whose odds band returned nothing; v2 fixes each ('exclusions', 'readings', 'spa_stepm', 'shrinkage'). v1's records stay in the registry, superseded: nothing is deleted, and its variants count toward variants searched ('variants')"
  },
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
  "pools_left_out": "TIERCE (trifecta) and BRACKET_QUINELLA: a follow-up after N1 that runs anyway (owner, 2026-11-27), proposed ticket #501",
  "span": {
    "rule": "development days from max(date_from, 2014-06-07) (D10) to the run's date_to; holdout days (core.eval.holdout.is_holdout_day, D14) are never read",
    "variants": {
      "from-2014-06-07": "the full span (the ranking, the rows and the candidate file)",
      "from-2024-01-01": "2024-01-01 onward, reported apart: untouched by the earlier pattern search (problem statement 5.1)"
    }
  },
  "store": "processed/jravan: final_odds (the six pools and their __pool__ rows), dividends (filtered by the year's development race ids) and races, read one calendar year at a time through models.v0.store.read_store_table(source='jravan')",
  "store_reader_guarantee": "read_store_table pushes the holdout exclusion into the Parquet scan as its filter, so no holdout row is materialized, and the dividends read is filtered inside its scan to the year's development race ids. Parquet prunes row groups by their min/max statistics only, and the jravan builders write one row group per year, so a year whose row group holds holdout days is decoded from disk and its holdout rows are dropped inside the scan; row groups are not yet isolated by day (proposed #502)",
  "exclusions": ["a holdout day (D14) and any day before 2014-06-07 (D10)", "a race whose status is not 'run'", "a jump race (surface 'jump'): flat races only (owner, 2026-11-27)", "a race whose surface is not turf or dirt, whose class is not in pool_scan.CLASS_GROUPS, whose field size is not 1-18 or whose venue is not a JRA venue", "a race with no priced WIN combination (no favourite strength)", "a race and pool whose payout may be withheld (takeout.race_takeout boost_may_be_withheld, a premium boost; as in #131)", "a race and pool an age-restricted boost entry matches: the jravan conditions code cannot show the age restriction, so race_takeout(..., race_class_raw=None) raises RaceConditionsNeeded and the rate is unknown", "a race and pool with no __pool__ total (pool not sold)", "a race and pool sold on the final board with no paying dividends row: not established or a special payout (the jravan dividends builder writes no row for either, jravan/dividends.py) or not yet delivered; it is excluded and counted, never scored as if every ticket lost", "a refunded combination is staked nowhere (pool_scan.combination_returns)"],
  "bet": "flat box: one unit on every priced combination of the pool's final board in the cell; a combination returns its dividend / 100 when it paid, else 0",
  "metric": {
    "flat_rho": "sum(return) / sum(payout rate) over the cell's staked combinations: 1 is priced like the pool; break-even is 1 / payout rate",
    "mean_return": "sum(return) / n_bets: the return per unit staked at the final dividend",
    "payout_rate": "the mean of the race and pool payout rates (takeout.race_takeout(race_id, pool), boosts included) over the cell's bets",
    "expected_return": "the cell's mean over its bets of payout rate x omega: the shrunk expected return per unit at the final price, takeout included; omega_full x payout_rate equals it exactly, the omega being payout-rate weighted ('readings')",
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
    "crossed_with_class": ["pool_size", "fav", "band"],
    "per_pool": "1 + 7 bands + 28 segment levels + 15 class x pool size + 20 class x fav + 35 class x band = 106 cells, the same list for the ranking, the reports, the candidate file and SPA/StepM"
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
      "decided_by": "Claude, under 'authority' (the ticket's Context lists the shrinkage for Place and Wide among the items for the spec); #78's fit on these pools is a registered alternative that is not run ('alternatives_not_run')",
      "why": "a Place or Wide dividend is not its odds (it falls in the final low-high range and depends on the co-placers), so #78's likelihood, exact for one-winner pools, would not measure the return; the shrink is on the realised return",
      "model": "normal empirical Bayes per segment family. Band baselines rho_b = sum(return) / sum(rate) of the pool's band b on the span (fixed effects, not shrunk). A cell's effect e = sum(return) / sum(rate x rho_b) over its bets, with its meeting-bootstrap se (rho_b held at its estimate); tau^2 of a family = max(0, mean over its cells with bets of (e - 1)^2 - se^2); B = tau^2 / (tau^2 + se^2) (0 when both are 0); shrunk effect 1 + B (e - 1), sd sqrt(B) se",
      "zero_return_band": "a band that returned nothing on the span has baseline rho_b 0, its fixed effect at its estimate. A cell with no expected return under the baselines (every bet in such a band, so e is 0 / 0) holds no information on its effect: its shrunk effect is the family's prior, mean 1 and sd tau, its omega_full its baseline 0 with sd 0 (expected return 0), and it enters no family's tau^2. A cell with bets in other bands too has e over those bets alone",
      "families": "class, field, fav, pool_size, surface, venue, class&pool_size, class&fav, class&band"
    }
  },
  "readings": {
    "omega_full": "the cell's omega with its band effects, weighted by payout rate: #78's mean over the cell's bets of exp(mu[band] + every random effect of the bet's levels), each bet weighing by its race and pool payout rate (SegmentROI.omega_draws(..., rate_weighted=True)), so that a draw times the cell's payout_rate is the cell's mean expected return per unit however it mixes ordinary and boosted rates; empirical Bayes: (sum(rate x rho_b) / sum(rate)) x the shrunk effect, and rho itself for 'all' and the band cells. The ranking, the rows and the candidate file use it",
    "segment_effect": "the segment effect alone: #78's exp(sum of the cell's named random effects); empirical Bayes: the shrunk effect. Undefined for 'all' and band cells"
  },
  "bounds": "omega is the posterior mean (#78: of the draws); one-sided 95% bounds each: omega_lo the 5% and omega_hi the 95% posterior quantile (#78 draws; empirical Bayes: omega -/+ 1.6449 sd); P(omega > 1/rate) is the posterior probability that omega x payout_rate exceeds 1 (#78: the share of draws; empirical Bayes: the normal tail): for omega_full, that the cell's mean expected return per unit exceeds 1",
  "ranking": "every cell of every pool, full span, omega_full: shrunk_edge = omega x payout_rate - 1 (expected return per unit at the final price, takeout included, minus 1), descending; cells without one last; ties in pool and spec order. Every cell is ranked and none is closed: no kill rule ('authority')",
  "bootstrap": {
    "unit": "meeting (races.meeting_id), D7",
    "n_boot": 1000,
    "level": 0.95,
    "draws": "multinomial counts of the span's meetings, shared by the pool's cells",
    "seed": "numpy default_rng(SeedSequence([run seed, pool index, span index]))"
  },
  "spa_stepm": {
    "families": "per pool, <family_id>-cells-<pool>: one record per pre-registered cell per run, its flat-box rule on the full span, scored with bets and failed without; SPA and StepM (core.eval.snooping) test exactly this run's non-failed cell records, so a narrower rerun is tested on its own cells and not on every cell an earlier run scored; the family size they report is variants searched ('variants')",
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
    "family_size": "variants searched = registry.family_size(family_id) + registry.family_size(the superseded family) after the run, on every row: v1's variants ran on the same days and count too; each SPA family's size, the same sum over its v2 and v1 SPA families, is reported beside its SPA p",
    "failures": "a pool with a data error in any year (pool_scan refuses a malformed board or dividend row) is failed on both spans, its cell rules too; a shrinkage that does not fit fails that pool and span; the private report gives each reason"
  },
  "alternatives_not_run": {
    "joint_cross": "the full joint cross odds band x class x field x favourite x pool size x surface x venue, 7 x 5 x 4 x 4 x 3 x 2 x 10 = 33600 cells per pool: most would hold a handful of races or none, no cell could show an effect at any useful power, and SPA over that many rules would test nothing. The outline registers task 1's marginal cells plus stated crosses; the crosses run are class x pool size, class x favourite strength and class x odds band, the owner's axes (maidens and debuts against the big graded races, around the favourite, by price), 70 of the 106 cells",
    "segment_roi_78_on_place_and_wide": "#78's fit at the low end of the Place or Wide range: its omega would describe the probability of placing at that price, not the return, which depends on the co-placers ('shrinkage')",
    "kill_rule": "closing a pool with no open cell: removed by the owner's reframe ('authority'); every cell is ranked",
    "exact_odds_in_the_fit": "exact odds for every price: a Trio board's long tail makes too many grouped rows for the 16 GB limit; the 1.1 log grid at the harmonic-mean price keeps each group's mean implied probability exact",
    "pool_size_basis": "a fixed yen basis for the pool-size tertiles: pool totals move with sales from year to year, so a fixed basis could sort the cells by year",
    "jump_races": "jump races as a surface level: flat races only (owner, 2026-11-27)",
    "trifecta_and_bracket_quinella": "in N1: a follow-up after N1 (owner, 2026-11-27), proposed #501"
  },
  "follow_ups": {
    "#501": "Trifecta and bracket quinella on the same scan (proposed)",
    "#502": "date-isolated row groups in the jravan store, so no holdout day is decoded (proposed)",
    "#462": "Family P on development folds: the top-ranked cells as decision-time rules (proposed)",
    "#436": "N6: the A2 holdout power table for the top-ranked cells"
  },
  "outputs": {
    "private_report": "$HORSETRACK_DATA_DIR/reports/pool-scan-japan-<spec hash>.md",
    "candidate_cells": "$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json, every cell in rank order in the candidate-cell schema plus omega, omega_lo, omega_hi and variants_searched",
    "public": "docs/experiments/pool-scan-japan.md: the run, no figure (D21)",
    "registry_summary": "docs/experiments/registry/japan/, by write_summary: each record's E id and variant, its status and values withheld (licensed source)"
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

The ticket made these the owner's to decide. The reframe gives them to Claude ("Claude decides and
registers them; each alternative is a counted variant"). Each was decided before any store row was
read. Only the alternatives that run count as variants. The spec lists the others under
`alternatives_not_run`.

- **Span.** From 2014-06-07 to the run's last day, with 2024-01-01 onward reported apart as a
  variant. The earlier start dates of other studies were not run.
- **Jump races.** Excluded (owner, 2026-11-27: flat races only). They do not form a surface level.
- **Trifecta and bracket quinella.** Left out. They are a follow-up after N1 (owner, 2026-11-27):
  proposed #501.
- **Odds-band edges per pool.** Those in the table above. No other edges were run.
- **Pool-size tertiles.** Per pool and calendar year, because pool totals move with sales from
  year to year, so a fixed yen basis could sort the cells by year. A fixed yen basis was not run.
- **Cross cells, and why not the full joint cross.** The outline asks for task 1's marginal cells
  plus stated crosses. The crosses run are class with pool size, class with favourite strength
  and class with odds band, because these are the owner's axes: maidens and debuts against the big
  graded races, around the favourite, and by price. Together they make 70 of each pool's 106
  cells. The full joint cross of odds band, class, field size, favourite strength, pool size,
  surface and venue is not registered. It would be 33,600 cells per pool, most holding a handful
  of races or none. No such cell could show an effect at any useful power, and SPA over that many
  rules would test nothing. Other crosses were not run.
- **Shrinkage for Place and Wide.** The ticket listed this as an item for the spec, so Claude
  chose it: the empirical-Bayes shrink of the realised return. #78's fit is a registered
  alternative that was not run. A Place or Wide dividend is not its odds: it falls in the final
  low-high range and depends on the co-placers. So #78's omega at the low end of the range would
  describe the probability of placing at that price, not the return.
- **A band that returned nothing (Place and Wide).** Its baseline is 0, its fixed effect at the
  estimate. A cell whose bets all sit in such bands carries no information on its effect, so it
  keeps the family's prior effect, N(1, tau²), and reads its baseline, 0, as `omega_full`.
- **Ranking, not a kill rule.** Every cell is ranked by its shrunk expected return. The owner's
  reframe removed the kill rule and the open/closed verdict.
- **Odds in the fit.** Exact below 10, and a 1.1 log grid from 10 at the harmonic-mean price.
  Exact odds everywhere were not run, because the long tail of a Trio board makes too many
  grouped rows for the 16 GB limit.
- **Variants searched.** This family's size plus that of v1, the family it supersedes. v1's
  variants ran on the same days before v2 was frozen.
