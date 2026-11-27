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

- Family `japan-n1-pool-scan-v4`, spec sha256
  `a166d99f7320b9596ffc31d1db4d1efdd33e0a3c84a2aad0a16c98a92139a91c`.
- **It supersedes v3**, family `japan-n1-pool-scan-v3`, spec sha256
  `785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c`, frozen and run on
  2026-11-27. The third review of PR #496 found one defect in v3, the zero-hit cell rule: a
  Place or Wide cell with an expected return and no hit, in an odds band that pays in other
  cells, has a meeting-bootstrap se of 0, so its shrink weight was 1 and its omega and upper
  bound exactly 0, a certain total loss. v4 gives such a cell the Jeffreys zero-hit se
  (`zero_hit_cell`); nothing changes for a cell that paid. v4 also cites two owner decisions
  that change no behaviour: the D14 boundary of a read (`store_reader_guarantee`,
  `d14_boundary`) and the cross cells being Claude's to register (`cells`, `decided_by`).
- **It also supersedes v2**, family `japan-n1-pool-scan-v2`, spec sha256
  `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6`, frozen and run on
  2026-11-27. The second review of PR #496 found six defects in v2:
  - it scored a race whose paying dividend named a combination off the priced board as if every
    ticket lost;
  - it read a Place or Wide band that paid nothing as a known zero, with no uncertainty;
  - it reported variants searched as the family size;
  - it decoded `final_odds` row groups that span a holdout day before filtering them;
  - it crossed class with three dimensions only;
  - it gave Place and Wide no #78 fit.
- **It also supersedes v1**, family `japan-n1-pool-scan`, spec sha256
  `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`, frozen and run the same
  day. The first review found four defects in v1:
  - it scored a sold board with no dividend row as if every ticket lost;
  - it took a cell's expected return as mean omega times mean payout rate, which is wrong where
    boosted and ordinary rates mix;
  - it checked SPA's columns against every cell any earlier run had scored;
  - it gave no Place or Wide reading for a cell whose odds band returned nothing.

  v2 fixed v1's defects, v3 fixed v2's and v4 fixes v3's (see `supersedes` in the spec). The
  records of v1, v2 and v3 stay in the registry, superseded, and nothing is deleted. Their
  variants count toward variants searched. This file quoted v1's spec in full up to commit
  `39227d5`, v2's up to commit `5977dae` and v3's up to commit `c564dee`.
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
- **Cells (186 per pool).** The list is the same for the ranking, the reports, the candidate file
  and SPA/StepM:
  - `all`, and each odds band (7);
  - each level of class (newcomer, maiden, conditions, open, graded), field size, favourite
    strength (the WIN favourite's final odds), pool size (tertiles per pool and calendar year),
    surface and venue (28);
  - class crossed with each other dimension: odds band (35), field size (20), favourite strength
    (20), pool size (15), surface (10) and venue (50), as `class:<group>&<dimension>:<level>`.

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
  the favourite-longshot bias, and crossed random effects for the six segments and the six class
  interactions. A cell's omega weighs each bet by its payout rate, so omega times the cell's mean
  payout rate is its mean expected return, even where boosted and ordinary races mix.
- **Place and Wide.** Their dividend is not their odds, so two shrinkages run:
  - The ranking's: a normal empirical-Bayes shrink of the realised return over each odds band's
    baseline, which carries its posterior. A band that paid is a gamma with the band's flat rho
    as mean and its meeting-bootstrap sd as sd. A band that paid nothing is not a known zero: it
    takes the Jeffreys posterior of a zero-hit hit rate, Gamma(1/2, n), times its mean odds over
    its mean payout rate. A cell whose bets all sit in such bands keeps the family's prior effect.
    A cell with no hit in bands that pay is not a known zero either: its shrink takes the
    Jeffreys zero-hit se of its effect, sqrt(1/2) times its mean odds over its expected return.
  - Beside it, as variants of their own (`roi78`): #78's fit with the outcome "it paid" and a
    representative price, each combination's low-end odds times its band's realised return per
    unit of low-end odds when it pays. The scaling moves only the band effects.
- **Readings and spans (the variants).** Two omega readings on two spans:
  - the readings: `omega_full` (the band effects included; it ranks the cells) and
    `segment_effect` (the segment effect alone);
  - the spans: from 2014-06-07, and from 2024-01-01 (reported apart).

  That makes one registry record per pool, shrinkage, reading and span: 24 variants, plus 8 for
  Place's and Wide's `roi78`, 32 in all. The family size and variants searched are reported
  apart on every row: variants searched is this family's size plus v1's, v2's and v3's.
- **SPA and StepM.** These run per pool over the pool's cell rules (family
  `japan-n1-pool-scan-v4-cells-<pool>`, one record per cell per run), with these settings:
  - no bet is the benchmark;
  - meetings are the clusters;
  - block size 12, 1,000 bootstrap draws, and StepM at 5%.

  They test this run's cell records, so a narrower rerun is tested on its own cells. They are
  reported beside the ranking, with the SPA family's size and, apart, its variants searched. They
  rank nothing and close nothing.
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
    for either) and dividends not yet delivered. Such a pool is counted, never scored as all-lose;
  - its paying dividend names a combination that is not on the priced, non-refunded board: the
    board lacks it, prints it without a price (`0000`) or cancelled (`****`), or does not have
    that combination at all. It is counted the same way and never fails the pool.
- **What the scan never decodes (D14).** No row group of `final_odds` whose race_date range spans
  a holdout day is decoded. The scan reads each row group's range from the footer statistics,
  without reading a page. It then reads prices only on the day windows no such row group touches,
  each window through `read_store_table`, whose Parquet scan prunes by the same statistics every
  row group outside the window. Each skipped row group is counted, and its development days are
  not scored. The jravan builders write one row group per year, so until #502 rebuilds the store
  with day-isolated row groups, a year whose row group spans a holdout day is skipped whole.
  `races`, which holds no price or result, is read on the same windows, and `dividends` is
  filtered in its scan to the windows' development race ids. Their row groups are filtered, not
  skipped: they do not follow the price windows (`races` is one row group, `dividends` is split
  by row count), so skipping one would drop every window it meets. One that spans a holdout day
  would therefore still be decoded, with its holdout rows dropped inside the scan. The private
  report counts such row groups per table until #502 isolates them too. That meets the D14
  boundary the owner set for study tickets (2026-11-27, on #82): a development read meets D14
  when no holdout value reaches Python, a pyarrow dataset filter dropping holdout days or
  holdout race ids inside the scan. Arrow decoding, in C++ before the filter, the column chunks
  of a row group that mixes development and holdout days is accepted, and isolating holdout days
  at the file or row-group level is #502's scope. Skipping `final_odds`'s spanning row groups
  goes beyond that boundary.
- **What a result means.** The odds band, the favourite strength and the pool size are read from
  the final board, which is look-ahead for selection (#88). A cell is a calibration target at the
  final price and a candidate for a decision-time rule. It is not a betting rule.

## The frozen spec

```json
{
  "family_id": "japan-n1-pool-scan-v4",
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
  "supersedes": [{"family_id": "japan-n1-pool-scan", "spec_sha256": "74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9", "why": "v1. Review of PR #496 found that it scored a sold board with no dividend row as if every ticket lost, took a cell's expected return as its mean omega times its mean payout rate (wrong when ordinary and boosted rates mix), checked SPA's columns against every cell any earlier run scored, and gave no Place or Wide reading for a cell whose odds band returned nothing; v2 fixed each"}, {"family_id": "japan-n1-pool-scan-v2", "spec_sha256": "ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6", "why": "v2. The second review of PR #496 found that it scored a race whose paying dividend named a combination off the priced, non-refunded board as if every ticket lost ('exclusions'), read a Place or Wide band that paid nothing as a known zero with no uncertainty ('shrinkage', 'band_baseline'), reported variants searched as the family size ('variants'), decoded final_odds row groups that span a holdout day before filtering them ('store_reader_guarantee'), crossed class with three dimensions only ('cells') and gave Place and Wide no #78 fit ('shrinkage'); v3 fixed each"}, {"family_id": "japan-n1-pool-scan-v3", "spec_sha256": "785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c", "why": "v3. The third review of PR #496 found the zero-hit cell rule wrong: a Place or Wide cell with an expected return under the point baselines and no hit (effect 0, in a band that pays in other cells) has a meeting-bootstrap se of 0, every resample returning nothing, so with a positive family tau^2 its shrink weight was 1 and its omega and upper bound exactly 0, a certain total loss the data cannot show; v4 gives such a cell the Jeffreys zero-hit se ('shrinkage', 'zero_hit_cell'), and changes nothing for a cell that paid"}],
  "superseded_records": "v1's, v2's and v3's records stay in the registry, superseded: nothing is deleted, and their variants count toward variants searched ('variants')",
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
  "store": "processed/jravan, one calendar year at a time: final_odds (the six pools and their __pool__ rows) and races through models.v0.store.read_store_table(source='jravan'), one read per price window of the year ('store_reader_guarantee'); dividends filtered inside its scan to the development race ids those windows read",
  "store_reader_guarantee": {
    "final_odds": "no row group of final_odds whose race_date [min, max] (its footer statistics, read without decoding a page) spans a holdout day (is_holdout_day, D14) is decoded. The scan reads prices only on the price windows: the days of the run's range that no such row group's [min, max] covers, each window one read_store_table call, whose Parquet scan prunes by the same statistics every row group outside the window and drops holdout days in its filter. Each spanning row group that meets the range is skipped and counted ('row group spans a holdout day (#502)'); its development days, and those of any other row group inside its range, are not scored. A row group without race_date statistics stops the run before any page is read",
    "per_year_groups": "the jravan builders write one final_odds row group per year, so until #502 rebuilds the store with day-isolated row groups a year whose row group spans a holdout day is skipped whole",
    "races_and_dividends": "races (no price or result: class, field size, surface, status) is read on the same windows through read_store_table and dividends (the results) filtered in its scan to the windows' development race ids: their row groups are filtered, not skipped ('alternatives_not_run'), so one that meets a window and spans a holdout day (races by race_date; dividends by the day each race id jra:<YYYY-MM-DD>:<VENUE>:<NN> encodes; or one without statistics) is decoded and its holdout rows dropped inside the scan; the private report counts such row groups per table until #502 isolates them too",
    "d14_boundary": "the owner's decision (2026-11-27, on #82, for study tickets; #430's definition of done quotes it): a development read meets D14 when no holdout value reaches Python, a pyarrow dataset filter dropping holdout days or holdout race ids inside the scan; Arrow decoding, in C++ before the filter, the column chunks of a row group that mixes development and holdout days is accepted, and isolating holdout days at the file or row-group level is #502's scope. The races and dividends reads meet this boundary; skipping final_odds's spanning row groups goes beyond it"
  },
  "exclusions": ["a holdout day (D14) and any day before 2014-06-07 (D10)", "a race whose status is not 'run'", "a jump race (surface 'jump'): flat races only (owner, 2026-11-27)", "a race whose surface is not turf or dirt, whose class is not in pool_scan.CLASS_GROUPS, whose field size is not 1-18 or whose venue is not a JRA venue", "a race with no priced WIN combination (no favourite strength)", "a race and pool whose payout may be withheld (takeout.race_takeout boost_may_be_withheld, a premium boost; as in #131)", "a race and pool an age-restricted boost entry matches: the jravan conditions code cannot show the age restriction, so race_takeout(..., race_class_raw=None) raises RaceConditionsNeeded and the rate is unknown", "a race and pool with no __pool__ total (pool not sold)", "a race and pool sold on the final board with no paying dividends row: not established or a special payout (the jravan dividends builder writes no row for either, jravan/dividends.py) or not yet delivered; it is excluded and counted, never scored as if every ticket lost", "a race and pool whose paying (non-refund) dividends row names a combination that is not on the pool's priced, non-refunded final board (absent, printed without a price as 0000 or cancelled as ****, refunded, or not a combination of the board): unsettled, excluded and counted, never scored as if every ticket lost and never a data error of the pool", "the development days of a final_odds row group whose race_date range spans a holdout day, and of any other row group inside that range ('store_reader_guarantee')", "a refunded combination is staked nowhere (pool_scan.combination_returns)"],
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
    "rule": "per pool, in this order: 'all'; 'band:<label>' for each odds band; '<segment>:<level>' for each segment and level; then 'class:<group>&<other>:<level>' for other in crossed_with_class, in that order, every class group and level; a cell with no bets is a row with null figures",
    "decided_by": "Claude, under the owner's decision (reframe, 2026-11-27, on #430; its definition of done quotes it): the cross cells are among the 'Items for the spec' that Claude decides and registers, each alternative a counted variant, and the members are task 1's marginal cells plus the crosses Claude registers, so a cross this spec does not register (the full joint cross included, 'alternatives_not_run') is a choice recorded here",
    "crossed_with_class": ["band", "field", "fav", "pool_size", "surface", "venue"],
    "per_pool": "1 + 7 bands + 28 segment levels + 5 classes x (7 bands + 4 field + 4 fav + 3 pool size + 2 surface + 10 venue) = 36 + 150 = 186 cells, the same list for the ranking, the reports, the candidate file and SPA/StepM"
  },
  "shrinkage": {
    "WIN, QUINELLA, FORECAST, TRIO (and PLACE, QUINELLA_PLACE as roi78)": {
      "model": "#78, core.eval.segment_roi.fit_segment_roi: ln omega = mu[odds band] + the sum of crossed random effects, one N(0, tau_c^2) per column, the module's priors (MU_PRIOR_SD 1.0, TAU_PRIOR_SCALE 0.1)",
      "columns": ["class", "field", "fav", "pool_size", "surface", "venue", "class_x_band", "class_x_field", "class_x_fav", "class_x_pool_size", "class_x_surface", "class_x_venue"],
      "interactions": "class_x_<other> is the level '<class group>|<level>'; a cross cell's query names class, its other level and the interaction",
      "draws": 1000,
      "seed": "numpy SeedSequence([run seed, pool index, span index]).generate_state(1)",
      "grouped_bets": "pool_scan.grouped_bets by race, then per (segment levels, band, takeout, odds group): odds below 10 exact, from 10 on a log grid of ratio 1.1 (group k holds [10 x 1.1^k, 10 x 1.1^(k+1))); a group's odds are its harmonic mean, so its mean implied probability is exact (only losers' terms see the grid)"
    },
    "PLACE, QUINELLA_PLACE": {
      "decided_by": "Claude, under 'authority' (the ticket's Context lists the shrinkage for Place and Wide among the items for the spec)",
      "why": "a Place or Wide dividend is not its odds (it falls in the final low-high range and depends on the co-placers), so #78's likelihood, exact for one-winner pools, does not measure the return by itself; the ranking's shrink is on the realised return, and #78's fit runs beside it on a representative price ('roi78')",
      "model": "normal empirical Bayes per segment family over odds-band baselines. A band's baseline rho_b = sum(return) / sum(rate) of the pool's band b on the span is a fixed effect, not shrunk, with a posterior ('band_baseline'). A cell's effect e = sum(return) / sum(rate x rho_b) over its bets, rho_b at its point estimate (0 for a band that paid nothing, so e is over the cell's bets in bands that paid), with its meeting-bootstrap se; tau^2 of a family = max(0, mean over its cells with a finite e of (e - 1)^2 - se^2), each cell's meeting-bootstrap se; a cell's shrink weight B = tau^2 / (tau^2 + s^2) (0 when both are 0), s its meeting-bootstrap se, or for a cell with no hit its zero-hit se ('zero_hit_cell'); shrunk effect normal, mean 1 + B (e - 1), sd sqrt(B) s",
      "band_baseline": "each band's rho_b carries its posterior, N_DRAWS draws per band shared by the pool's cells on the span: for a band that paid, a gamma with rho_b as mean and the band cell's meeting-bootstrap sd of rho as sd (a point at rho_b when that sd is 0); for a band that paid nothing, where the bootstrap has no spread, the Jeffreys posterior of a zero-hit Poisson hit rate per combination, Gamma(1/2, rate n_b), times the band's mean odds over its mean payout rate (a paid Place or Wide combination pays at least about its low-end odds): Gamma(shape 1/2, scale sum(odds) / (n_b x sum(rate))), so a sparse band that paid nothing keeps a positive upper bound",
      "zero_return_band": "a cell whose bets all sit in bands that paid nothing has no expected return under the point baselines (e is 0 / 0): it holds no information on its effect, which is the family's prior, N(1, tau^2), and it enters no family's tau^2; its omega_full keeps its bands' baseline posterior",
      "zero_hit_cell": "a cell with an expected return under the point baselines (D = sum(rate x rho_b) > 0) and no hit has e = 0 and a meeting-bootstrap se of 0, every resample returning nothing, which would make B 1 and its omega a certain 0. Its s is instead the Jeffreys zero-hit sampling error, the construction 'band_baseline' uses for a band that paid nothing: the cell's sum of returns is its mean odds (mean_odds, low-end; a paid combination pays at least about it) times its expected hits, whose Jeffreys posterior after zero hits is Gamma(1/2, 1), sd sqrt(1/2); so s = sqrt(1/2) x mean_odds / D. B < 1 then, and the shrunk effect keeps a spread that narrows as the cell's losing bets grow. Only s changes: e stays 0, the cell enters its family's tau^2 with its meeting-bootstrap se as every cell does, and a cell that paid reads as in v3",
      "draws": "numpy default_rng(SeedSequence([run seed, pool index, span index, 1])): the band draws, then each cell's effect draws in spec order",
      "families": "class, field, fav, pool_size, surface, venue, and class&<other> for each other in crossed_with_class"
    },
    "roi78": {
      "pools": "PLACE, QUINELLA_PLACE",
      "model": "#78's fit_segment_roi with the one-winner pools' columns, draws and seed, on the pool's grouped bets: a combination's outcome is whether it paid; its price is a representative final odds o = kappa_b x its low-end odds, kappa_b = sum(return) / sum(low-end odds) over the band's paid combinations on the span (1 for a band that paid nothing): the band's realised return per unit of low-end odds when it pays. Then p = omega x rate / o, and omega x rate is the expected return per unit at the final dividend wherever a paid combination of the band returns kappa_b x its low end, which kappa_b's definition makes exact in each band's aggregate; the scaling moves only the band effects (mu[b] + ln kappa_b), never a segment effect. Like #78 on the other pools, the likelihood takes a race's combinations as independent bets",
      "variants": "'<pool>/roi78-<reading>/<span>', both readings and spans, reported beside the empirical-Bayes readings in the private report; the ranking, the rows and the candidate file keep the empirical-Bayes omega_full",
      "failure": "a fit that cannot apply (a band whose bets all paid, a mode on p = 1) is a failed record with its reason; the empirical-Bayes variants are scored apart"
    }
  },
  "readings": {
    "omega_full": "the cell's omega with its band effects, weighted by payout rate: #78's mean over the cell's bets of exp(mu[band] + every random effect of the bet's levels), each bet weighing by its race and pool payout rate (SegmentROI.omega_draws(..., rate_weighted=True)), so that a draw times the cell's payout_rate is the cell's mean expected return per unit however it mixes ordinary and boosted rates; empirical Bayes: draws of (sum over bands of w_b rho_b) x the shrunk effect, w_b the cell's share of payout rate in band b, rho_b's draws its band baseline posterior and the effect's draws normal, censored at 0 (an effect is a ratio of returns); a band cell reads its band's baseline posterior and 'all' its flat rho, normal with its bootstrap sd. The ranking, the rows and the candidate file use it (for Place and Wide, the empirical-Bayes one)",
    "segment_effect": "the segment effect alone: #78's exp(sum of the cell's named random effects); empirical Bayes: the shrunk effect, normal. Undefined for 'all' and band cells",
    "roi78": "for Place and Wide, both readings again from #78's fit on the representative price ('shrinkage', 'roi78'), recorded as their own variants"
  },
  "bounds": "omega is the posterior mean (of the draws, held within the bounds); one-sided 95% bounds each: omega_lo the 5% and omega_hi the 95% posterior quantile (of the draws; a normal reading, the empirical-Bayes segment_effect and 'all', omega -/+ 1.6449 sd); P(omega > 1/rate) is the posterior probability that omega x payout_rate exceeds 1 (the share of draws, or the normal tail): for omega_full, that the cell's mean expected return per unit exceeds 1",
  "ranking": "every cell of every pool, full span, omega_full: shrunk_edge = omega x payout_rate - 1 (expected return per unit at the final price, takeout included, minus 1), descending; cells without one last; ties in pool and spec order. Every cell is ranked and none is closed: no kill rule ('authority')",
  "bootstrap": {
    "unit": "meeting (races.meeting_id), D7",
    "n_boot": 1000,
    "level": 0.95,
    "draws": "multinomial counts of the span's meetings, shared by the pool's cells",
    "seed": "numpy default_rng(SeedSequence([run seed, pool index, span index]))"
  },
  "spa_stepm": {
    "families": "per pool, <family_id>-cells-<pool>: one record per pre-registered cell per run, its flat-box rule on the full span, scored with bets and failed without; SPA and StepM (core.eval.snooping) test exactly this run's non-failed cell records, so a narrower rerun is tested on its own cells and not on every cell an earlier run scored; reported with the SPA family's registry size (family_size) and, apart, its variants searched: that plus the same SPA family of each superseded spec",
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
    "family": "one record per pool, shrinkage, reading and span: '<pool>/<reading>/<span>' (6 x 2 x 2) and, for Place and Wide, '<pool>/roi78-<reading>/<span>' (2 x 2 x 2), 32 in all; scored when that shrinkage fitted on the span, failed otherwise; its metrics are every cell's omega with bounds, P(omega > 1/rate), flat rho and mean return",
    "family_size": "registry.family_size(family_id) after the run, on every row and in the result",
    "variants_searched": "family_size + registry.family_size(f) for each superseded family f (v1, v2, v3) after the run, on every row and in the result, reported apart from family_size: their variants ran on the same days and count too; each SPA family's size and its variants searched (the same sum over its v1, v2 and v3 SPA families) are reported beside its SPA p",
    "failures": "a pool with a data error in any year (pool_scan refuses a malformed board or dividend row) is failed on both spans, its cell rules too; a shrinkage that does not fit fails that pool and span; the private report gives each reason"
  },
  "alternatives_not_run": {
    "joint_cross": "the full joint cross odds band x class x field x favourite x pool size x surface x venue, 7 x 5 x 4 x 4 x 3 x 2 x 10 = 33600 cells per pool: most would hold a handful of races or none, no cell could show an effect at any useful power, and SPA over that many rules would test nothing. The cross cells are Claude's to register (owner's reframe, 2026-11-27, on #430; 'cells', 'decided_by'): the members are task 1's marginal cells plus the crosses registered, and the crosses run are class, the owner's axis (maidens and debuts against the big graded races), with each other dimension, 150 of the 186 cells",
    "segment_roi_78_at_the_low_end": "#78's fit on Place or Wide at the bare low end of the range: its omega would describe the probability of placing at that price, not the return, which depends on the co-placers; roi78 scales each band's price by its realised return when it pays ('shrinkage')",
    "point_band_baseline": "v2's point baseline for a Place or Wide band that paid nothing: a sparse band can pay nothing by chance, so a known zero would rule out a return the data cannot rule out ('band_baseline')",
    "bootstrap_se_for_a_zero_hit_cell": "v3's meeting-bootstrap se of 0 for a Place or Wide cell with no hit in bands that pay: a few losing bets would read as a certain total loss ('zero_hit_cell')",
    "zero_hit_se_in_tau2": "the zero-hit se in the family's tau^2 as well: it is a model sd after zero hits, not the bootstrap sampling sd the moment estimate subtracts, and it is largest for the sparsest cells, so a few sparse zero-hit cells could drive a family's tau^2 to 0 and move every cell of the family, those that paid included, to the prior with no spread",
    "zero_hit_cell_unshrunk": "a zero-hit cell's effect as its Jeffreys posterior alone, unshrunk: its mean, half of mean_odds / D, grows as the cell gets sparser, so a sparse cell with no hit could rank above cells that paid, its family's information unused",
    "skipping_races_and_dividends_row_groups": "skipping a races or dividends row group that spans a holdout day as final_odds's are skipped: their row groups do not follow the price windows (the jravan builders write races in one row group and dividends in row groups by row count), so a skipped one would drop the development days of every window it meets; they hold no price, and are filtered and counted instead until #502 ('store_reader_guarantee')",
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
    "candidate_cells": "$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json, every cell in rank order in the candidate-cell schema plus omega, omega_lo, omega_hi, family_size and variants_searched",
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
- **Cross cells, and why not the full joint cross.** The owner's decision (reframe, 2026-11-27,
  on #430, quoted in its definition of done) puts the cross cells among the items for the spec
  that Claude decides and registers, each alternative a counted variant. The members are task
  1's marginal cells plus the crosses registered here, so a cross not registered, the full joint
  cross included, is a choice recorded in the spec. The crosses run are class with each other
  dimension (odds band, field size, favourite strength, pool size, surface and venue), because
  class is the owner's axis: maidens and debuts against the big graded races. Together they make
  150 of each pool's 186 cells. The full joint cross of odds band, class, field size, favourite
  strength, pool size, surface and venue is not registered. It would be 33,600 cells per pool,
  most holding a handful of races or none. No such cell could show an effect at any useful
  power, and SPA over that many rules would test nothing. Crosses that do not involve class were
  not run.
- **Shrinkage for Place and Wide.** The ticket listed this as an item for the spec, so Claude
  chose it. The ranking uses the empirical-Bayes shrink of the realised return. #78's fit also
  runs, as variants of its own (`roi78`). A Place or Wide dividend is not its odds: it falls in
  the final low-high range and depends on the co-placers. So #78 at the bare low end of the range
  would describe the probability of placing at that price, not the return. The fit therefore
  prices each combination at its low-end odds times its band's realised return per unit of
  low-end odds when it pays, with "it paid" as the outcome. Omega times the payout rate is then
  the expected return wherever a paid combination of the band returns that multiple of its low
  end, which the multiple's definition makes exact in each band's aggregate. The scaling moves
  only the band effects, never a segment effect. A fit that cannot apply is a failed record with
  its reason, and the empirical-Bayes variants still score.
- **A band that paid nothing (Place and Wide).** A sparse band can pay nothing by chance, so its
  baseline is not a known 0. It takes the Jeffreys posterior of a zero-hit Poisson hit rate,
  Gamma(1/2, n), times its mean low-end odds over its mean payout rate: a paid combination pays
  at least about its low-end odds. A band that paid is a gamma with its flat rho as mean and its
  meeting-bootstrap sd as sd. A cell's `omega_full` draws are its rate-weighted band baseline
  draws times its effect's draws. A cell whose bets all sit in bands that paid nothing carries no
  information on its effect, so it keeps the family's prior effect, N(1, tau²), and its bands'
  baseline posterior.
- **A cell with no hit in bands that pay (Place and Wide, v4).** Its realised effect is 0, and
  its meeting-bootstrap se is 0 too, because every resample returns nothing. v3 shrank it with
  that se, so its omega and upper bound were exactly 0. v4 keeps the effect at 0 and gives its
  shrink the Jeffreys zero-hit se, the construction a band that paid nothing already uses: the
  cell's sum of returns is its mean low-end odds times its expected hits, whose Jeffreys
  posterior after zero hits is Gamma(1/2, 1). The se is then sqrt(1/2) times the cell's mean odds
  over its expected return, so the shrink weight is below 1 and the posterior keeps a spread
  that narrows as its losing bets grow. The family's tau² keeps every cell's bootstrap se, and a
  cell that paid reads as in v3. Not run: the zero-hit se in tau² as well (a model sd after zero
  hits, not the sampling sd the moment estimate subtracts; a few sparse zero-hit cells could
  drive tau² to 0 and move every cell of the family to the prior), and the zero-hit cell's
  Jeffreys posterior unshrunk (a sparse cell with no hit could rank above cells that paid).
- **A dividend off the priced board.** A race and pool whose paying dividend names a combination
  the priced, non-refunded board does not hold is unsettled: it is excluded and counted, never
  scored as if every ticket lost and never a data error of the pool.
- **Row groups that span a holdout day.** `final_odds` row groups are skipped whole, by their
  footer statistics, so none is decoded. `races` and `dividends` are filtered and counted instead
  (see above).
- **Ranking, not a kill rule.** Every cell is ranked by its shrunk expected return. The owner's
  reframe removed the kill rule and the open/closed verdict.
- **Odds in the fit.** Exact below 10, and a 1.1 log grid from 10 at the harmonic-mean price.
  Exact odds everywhere were not run, because the long tail of a Trio board makes too many
  grouped rows for the 16 GB limit.
- **Family size and variants searched.** Reported apart. The family size is this family's
  registry size. Variants searched is that plus the sizes of v1, v2 and v3, the families it
  supersedes, whose variants ran on the same days before v4 was frozen. The same holds for each
  SPA family.
