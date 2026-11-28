# Experiment report: N1, the final-board segment scan of six pools (Japan, #430)

Ticket #430 (epic #72). **Every figure is withheld (D21)**: the scan reads the JRA-VAN store, a licensed source, so this report says what ran and where its figures are, and gives no count, rate, span, sign, bound or rank derived from that data. Nothing is adopted, no bet follows from it, and the holdout is not opened.

- Registry family `japan-n1-pool-scan-v6`, spec sha256 `4e04d2d8d0d1e96645bd0bac94a0b9b552811896d3146fd7cb21f935fab5fd45` (`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it before the run read any row.
- Authority: the owner's reframe of 2026-11-27 on #430 removed the sign-off ("Claude pre-registers the cell list and metrics and freezes them in the registry before the run. No owner sign-off.") and the kill rule ("It ranks every cell by expected return per 100 yen at the final price (takeout included, CI beside it). None is closed, and models aim at the top-ranked cells first."). The frozen spec quotes it in full (`authority`).
- It supersedes v5, `japan-n1-pool-scan-v5` (spec sha256 `43b7ad67f504f923ac079b38be4e740f4e3771934b94f55a38a3d856fa9fc6b8`), v4, `japan-n1-pool-scan-v4` (spec sha256 `a166d99f7320b9596ffc31d1db4d1efdd33e0a3c84a2aad0a16c98a92139a91c`), v3, `japan-n1-pool-scan-v3` (spec sha256 `785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c`), v2, `japan-n1-pool-scan-v2` (spec sha256 `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6`), and v1, `japan-n1-pool-scan` (spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`), all frozen and run on 2026-11-27 (UTC). The seventh review of PR #496 found that v5's one-winner #78 fits capped each settled win at the board quote, so a capped quote paid above the board lost its excess from the ranked expected return and its bounds (raw rho and SPA kept it), that its store diagnostics read the dividends row groups' race-id statistics, which return a holdout race's id to Python, and that a rerun replaced the run of record's private report and candidate file. v6 fits uncapped settled wins, reads date statistics only and names each run's outputs apart, never replacing an earlier run's. The fifth review of PR #496 found that v4 skipped whole every price row group that spans a holdout day, so once an export holds a holdout day a year's development races in its annual row group would go unscored, and that its one-winner #78 fits credited a dead heat at the board price, dropping the settled dividend (raw rho was already settled). v5 reads such row groups through the scan's day filter and fits settled wins. The third review of PR #496 found that v3 read a Place or Wide cell with no hit, in an odds band that pays in other cells, as a certain total loss: its meeting-bootstrap se is 0, so its shrink weight was 1 and its omega and upper bound exactly 0. v4 gives such a cell the Jeffreys zero-hit se and changes nothing for a cell that paid. The second review found that v2 scored a race whose dividend named a combination off the priced board as if every ticket lost, read a Place or Wide band that paid nothing as a known zero, reported variants searched as the family size, decoded price row groups that span a holdout day, crossed class with three dimensions only, and gave Place and Wide no #78 fit. v1's defects are in the spec (`supersedes`). Their records stay in the registry, superseded; nothing is deleted.
- Requested range: development days from 2014-06-07 to 2026-11-27 (the run's arguments, not the store's measured span); holdout days are never read and `LockedHoldout` is never opened.
- The store reads (D14, #502): `final_odds` and `races` are read one year at a time through `read_store_table`, whose dataset filter drops holdout days inside the Parquet scan, and `dividends` is filtered in its scan to the development race ids. A row group that mixes development and holdout days (the jravan builders' annual row group once an export holds a holdout day) is decoded in the scan and its development days are scored; a row group of a single holdout day is pruned by its statistics. That is the owner's D14 boundary for study tickets (2026-11-27, on #82): a development read meets D14 when no holdout value reaches Python, and Arrow decoding a mixed row group in C++ before the filter is accepted; isolating holdout days at the file or row-group level is #502's scope, and the private report counts, for `final_odds` and `races`, the row groups whose race_date range spans a holdout day until then; no race-id statistic of `dividends` is read (`store_reader_guarantee`, `counted`, `d14_boundary`).
- Seed 20261127; code at `1ebbadea9686`; command `uv run horsetrack japan pool-scan run --from 2014-06-07 --to 2026-11-27 --seed 20261127`.
- Family size 32 (this family's distinct variants in the registry); variants searched 176 (that plus v1's to v5's, whose variants ran on the same days). Both are on every row.
- Private report: `$HORSETRACK_DATA_DIR/reports/pool-scan-japan-4e04d2d8d0d1e96645bd0bac94a0b9b552811896d3146fd7cb21f935fab5fd45-2014-06-07_2026-11-27-seed20261127-1f354ff51ed1.md`; candidate cells: `$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan-2014-06-07_2026-11-27-seed20261127-1f354ff51ed1.json`; both written through `write_report` with the source `jravan`.

## Registry records

One record per pool, shrinkage, reading and span in `japan-n1-pool-scan-v6`, in spec order:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `1adf7dfcc7e26f4ca098d2ceb1acbbebcefc0ed7c8383ae9e447fe4996f8a719` |
| `WIN/omega_full/from-2024-01-01` | `107eb61db93fa021f5f9c2167571aa6acc6a525fad3a23af38322092de63cee7` |
| `WIN/segment_effect/from-2014-06-07` | `6b96da87e3836fea6134c91c30ec4dae4b95075adce46429119462f21092a930` |
| `WIN/segment_effect/from-2024-01-01` | `2350e37a4f36ba83a3254c64e1612a7e95a318e315009c80c033eecb58c8db62` |
| `PLACE/omega_full/from-2014-06-07` | `8c80bc231d5f66150d7f28d710a6acc2c721171e4a758450dfc833ad74bac351` |
| `PLACE/omega_full/from-2024-01-01` | `231052c3622afdafb46509ae3fcb7ab1e4b4506f2672878e2f4aa085cbea447c` |
| `PLACE/segment_effect/from-2014-06-07` | `ba072c3a1791ccd8a0c90170adb7d1746ba28e4716f2d4f4ba79fc02efd6b4ab` |
| `PLACE/segment_effect/from-2024-01-01` | `d95eef0a1fd74707b7471ea316bc4b3006d5c484a4d42dcfb795a39529a73578` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `8628974e648b7d4e40942b7cdd5942b3c5ad227421754694ed5e62e8241a010a` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `dc24b52b6817b7ae8eb3c3e4023bbe86ba326ab22055191c6e281d291db29a46` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `2de1d4be63aef51ef83277fcd546d8a036eab91aba3751b1ce2a3921173f2f6d` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `d689a797f9df0bd045f3551f94780556040d639643041a8e05fc9a6bd90eb90b` |
| `QUINELLA/omega_full/from-2014-06-07` | `be842192539ba5c5208ba35d4a3252e5e2669b80564a258b6aabd9d7ec7aa459` |
| `QUINELLA/omega_full/from-2024-01-01` | `6fb7c5cb70a7cf26716bd7d1d3b0dab51a26c6492b0ab20dc5563638eb4b8eb5` |
| `QUINELLA/segment_effect/from-2014-06-07` | `422ddd32fcad52cc5e5712e1de00f01e831cd7a2214a81ccb38204ff4ae88694` |
| `QUINELLA/segment_effect/from-2024-01-01` | `221698903744dc6e5ddbae81c690ccf8dacbc76e30a5fa64cc7e4c692a13e466` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `7249fa6e6501cfe3c6e5e0084a79b9016a3a8493b67a30450288e6ea106ebd2f` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `c704f3bd35555ef6be29895790186694cc12c7a488daa61ed36d356e48f3baa6` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `a82dbe3cebbabe3ba38fee839e3ff4b21616578933da9741776c776349526423` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `f5d16023ef45077655a82b140555928182fbb3fe9ba17d65350c398224d9387a` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `11dcf5d87f02a0f0f1eccc36145dd931ac28b9caca2f60660ae5f38087f0485a` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `e68e1bede50f5c491015eb0c0bcc1e53461191b1001e98d26f96ce27bd5bd00c` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `9398fdef5e94df2f557d5fe10fa79b12d1c7c805d10b079794a3ecde13779d8b` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `78a4ba411ff12ec4cf6fe18a954db6adca9d4cab97d086de270dc276435888f7` |
| `FORECAST/omega_full/from-2014-06-07` | `440cf7aca18eea02d3739adbd5fde86cdf706ba3aa3f0cc57867af85be01fa9c` |
| `FORECAST/omega_full/from-2024-01-01` | `065a223fab47ab5dee1f502b715160b7fe45140107f8fd4e91faeba09b92702c` |
| `FORECAST/segment_effect/from-2014-06-07` | `d06e4b25f86581717d5c504596fb80bd187e019f42c1a0f2b441ce131145f227` |
| `FORECAST/segment_effect/from-2024-01-01` | `847d3e1c5f01e8f8b013f9d8fac6cc6f0a96c4bce2f619a7d36325b52ef51457` |
| `TRIO/omega_full/from-2014-06-07` | `11891ac4e366fc92cafa2598f4c1f6ce6bf86cfe24c1663521186c69270f1516` |
| `TRIO/omega_full/from-2024-01-01` | `b7d95fc2bb6385b362ae9dcb0b2387daeffd5e7541a1d7c5e463aeb23a2b9949` |
| `TRIO/segment_effect/from-2014-06-07` | `799b52b7a3ff24276bf2c1f7ee50a01268bb558fa7dd55bdef51d4f31da1f903` |
| `TRIO/segment_effect/from-2024-01-01` | `e31ec304dcfafec89d1587b10856f622cbafa3645d0fc039e371d3163f16b041` |

The cell rules SPA and StepM test are one record per pre-registered cell in each pool's family; their run ids are in the private report, and the sha256 of each family's sorted run ids is:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v6-cells-win` | `ef3d67938c3eb8fce8b4ffba7fe24e9aebd8aa3e1a6bb92c4586d8a54fbd02df` |
| `japan-n1-pool-scan-v6-cells-place` | `bc7c95309803751123647e35ceaa83b0be2932d2fc5a83de7c6b9dcfddefc309` |
| `japan-n1-pool-scan-v6-cells-quinella` | `8818d7a6c986865d0fb4bf38d9a3e58c96f12cbdfe905418cb03a8a824f6ac56` |
| `japan-n1-pool-scan-v6-cells-quinella_place` | `69017b179e4f3b10fd640ac3011e0c85a68217848ce52c846e43582175ad1791` |
| `japan-n1-pool-scan-v6-cells-forecast` | `76e82668bd4d5d1144b2d90bdc036cfdc6ebc2c2d559c2a09587d57542eb41f4` |
| `japan-n1-pool-scan-v6-cells-trio` | `3fe78f35d48fc64bcf40c16f7b9de49f4949959f0f45f0930c601bfb9091cd95` |

## Hypothesis

E8, as frozen: in some pre-registered cell of a JRA pool's final board the hierarchical omega differs from 1 once the favourite-longshot bias is taken out as odds-band fixed effects, and some cell's expected return per unit at the final price reaches break-even (omega at or above 1 / payout rate). The null, the starting point: every cell is priced like its pool.

## Variants tried

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO): two omega readings (`omega_full`, the band effects included; `segment_effect`, the segment effect alone) on two spans (from 2014-06-07; from 2024-01-01, reported apart). The cells are `all`, each odds band, each level of class, field size, favourite strength, pool size, surface and venue, and class crossed with each of the other six. The shrinkage is #78's hierarchical fit for the one-winner pools, on settled wins (each hit counts for its dividend over its board odds, uncapped: below 1 for a dead heat that splits the pool, above 1 for a capped quote paid above the board), its omega weighted by payout rate. For Place and Wide it is an empirical-Bayes shrink of the realised return over odds-band baselines that carry their posterior (a band that paid nothing keeps a Jeffreys posterior, not a known zero, and a cell with no hit in bands that pay keeps the Jeffreys zero-hit se in its shrink), which ranks them; beside it, as variants of their own (`roi78`), #78's fit on a representative price: each combination's low-end odds times its band's realised return per unit of low-end odds when it pays, the outcome being whether it paid. Every cell of every pool is ranked by its shrunk expected return (the owner's reframe: none is closed); the ranks, the omegas and their bounds are withheld (D21). Which variants failed, and their figures, are withheld (D21): the private report gives each reason, and the registry summary shows no status of a licensed-source run. v1's to v5's variants, superseded, are records of `japan-n1-pool-scan`, `japan-n1-pool-scan-v2`, `japan-n1-pool-scan-v3`, `japan-n1-pool-scan-v4` and `japan-n1-pool-scan-v5` and count toward variants searched.

## Alternatives rejected and why

- The full joint cross (odds band x class x field x favourite x pool size x surface x venue, 33600 cells per pool): most cells would hold a handful of races or none, so none could show an effect at any useful power, and SPA over that many rules would test nothing. The cross cells are Claude's to register (the owner's reframe, 2026-11-27, on #430): the cells cross class, the owner's axis (maidens and debuts against the big graded races), with each other dimension: 150 of each pool's 186 cells.
- v5's cap of each settled win at the board quote: a capped quote paid above the board lost its excess from the shrunk reading while raw rho kept it.
- v4's skip of every price row group that spans a holdout day: with one row group per year it would drop a year's development races once an export holds a holdout day, beyond what the owner's D14 boundary asks. v4's board-price hits in the one-winner fits: a dead heat paid below the board odds was credited at the quoted price. A settlement factor per odds band would spread a dead heat's shortfall over its band's other cells, and a raw per-cell factor on the shrunk omega would leave the segment effect unsettled; settled wins carry the settlement through the fit.
- v3's bootstrap se of 0 for a Place or Wide cell with no hit in bands that pay: a few losing bets would read as a certain total loss. The zero-hit se in the family's tau^2 as well: it is a model sd after zero hits, largest for the sparsest cells, so a few of them could drive tau^2 to 0 and move every cell of the family to the prior. The zero-hit cell's Jeffreys posterior unshrunk: a sparse cell with no hit could rank above cells that paid.
- #78's fit on Place and Wide at the bare low end of the range: its omega would describe the probability of placing at that price, not the return; the representative price scales each band by its realised return when it pays.
- A point baseline for a Place or Wide band that paid nothing (v2): a sparse band can pay nothing by chance, so a known zero would rule out a return it cannot rule out.
- A kill rule (close a pool with no open cell): removed by the owner's reframe; every cell is ranked and none is closed.
- Exact odds in the fit for every price: a Trio board's long tail makes the grouped rows too many for the 16 GB limit; from odds 10 a 1.1 log grid with the harmonic-mean price keeps the mean implied probability exact.
- A fixed yen basis for the pool-size tertiles: pool totals move with sales from year to year, so a fixed basis could sort the cells by year.
- Trifecta and bracket quinella: a follow-up that runs after N1 (owner, 2026-11-27).

## What this rules out

withheld (D21): the private report gives, per pool, which cells' one-sided bounds cover break-even, the SPA p and the StepM survivors. Whatever it shows, an N1 cell is a calibration target at the final price (bands, favourite and pool size are read from the final board, look-ahead for selection, #88), not a betting rule.

## Follow-ups filed

- #501: Trifecta and bracket quinella on the same scan (proposed).
- #502: date-isolated row groups in the jravan store, so no holdout day is decoded (proposed).
- #462: Family P on development folds: the top-ranked cells as decision-time rules (proposed).
- #436: N6: the A2 holdout power table for the top-ranked cells.

## Runs, memory and the registry summary

Everything above this section is `render_public_report`'s output for the run of record: v6's
second run, at commit `1ebbade`, on the jravan `dividends` rebuilt with #443's builder. This
section records the trail.

- **v6 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-28 at 19:54 JST at
  commit `54073fe`, before any scored read of `japan-n1-pool-scan-v6`. It wrote the family's only
  freeze line in the japan registry, which precedes every record of the family, both runs'.
- **Why v6 ran twice.** v6's first run, at `54073fe` (below), read the jravan `dividends` build
  of 2026-11-26, made before #443 (`32b2110`, merged into this branch at `8a1312d`). That builder
  wrote no refund row, so a combination holding a horse withdrawn after the final board was
  priced would have been scored as a staked loss, against the frozen spec's rule that a refunded
  combination is staked nowhere (`pool_scan.combination_returns`). The eighth review of PR #496
  asked for the rebuild and a rerun under the same frozen spec; the owner's restart instructions
  of 2026-11-28 confirmed it (absorbing #695).
- **jravan `dividends` rebuild.** `uv run horsetrack japan jravan dividends` (every year file of
  the export's HR records; no `--year`) ran on 2026-11-28 from 22:52 to 22:53 JST at commit
  `1ebbade`, the branch with #443's builder, and replaced
  `$HORSETRACK_DATA_DIR/processed/jravan/dividends.parquet` atomically (a temporary file and
  `os.replace`). The HR year files it read are those of the earlier build (unchanged since
  2026-11-26), so only the builder differs, and the rebuilt file carries `is_refund` rows. The
  earlier build and its build report are kept under
  `$HORSETRACK_DATA_DIR/archive/jravan-dividends-pre-443-20261126/`, so the first run's input
  stays on disk. It took 20 s of wall time and a resident set of about 0.59 GB; beside it ran
  #531's `horsetrack japan competition-study`, which reads no `dividends`. Two readers misread the refund rows until their tickets land and were not run: E25's
  `cross_pool_run._pair_rows` (#648) and `horsetrack japan jravan reconcile` (#649).
- **v6 run of record, at `1ebbade`** (the tables above). The same command as the first run,
  under `/usr/bin/time -l`:
  `uv run horsetrack japan pool-scan run --from 2014-06-07 --to 2026-11-27 --seed 20261127`,
  with `--public` to a scratch file outside the repository (the text above). `/usr/bin/time -l`
  gave a maximum resident set size of 1,513,029,632 bytes (about 1.51 GB, under the 2.5 GB cap)
  and 2,053 s of wall time on this 16 GB Mac. It completed and printed its summary, with
  `summary_error` null. It ran on 2026-11-29 (JST), a JRA race day, from 00:53 to 01:27 JST, before
  racing: it waited from 22:53 JST for two other tickets' study runs to end (#531's `horsetrack
  japan competition-study`, at about 00:13 JST, and #102's `horsetrack japan late-path-study`, at
  about 00:51 JST), and started once at least 40% of memory was free (42% at the start). Beside
  it ran the race-day services of 2026-11-29 from the main checkout (`horsetrack japan pull-vm`
  and `board-serve` and, from about 01:02 JST, `shadow-day`, each a resident set under 30 MB),
  #573's `horsetrack japan debut-prior` (another ticket's study run, about 0.2 GB, from about
  01:18 JST to before 01:25 JST) and other sessions' test runs, at load averages near 7 at the
  start.
- **What the rerun changes.** The spec is v6's as frozen: `prereg_spec()` still hashes to the
  frozen sha256, so `run_scan` ran without a new freeze. Of the three store files the run reads,
  only `dividends.parquet` differs (its `data_builds` fingerprint changed; those of
  `final_odds.parquet` and `races.parquet` did not), so every record of the rerun has a new run
  id and was appended beside the first run's, none replaced. The code is the branch after two
  merges of main: `pool_scan_run.py` is unchanged since `54073fe`; #443 changed only
  `pool_scan.py`'s docstring (the scan already read `is_refund` rows as refunds); #502's store
  reader skips single-day row groups and reads by byte range (the same rows from these files,
  built before #502); #88 added `segment_roi` readings the scan does not take (`omega_draws` and
  `effect_draws` are as they were). Checked against the first run's outputs: the candidate-cell
  file is byte-identical, and the private report differs only in run ids, the commit, the data
  fingerprint, its written time and the key order of the fit diagnostics. So the refund rows
  named no combination the scan had staked: the final board already prints a combination of a
  horse withdrawn after sales without a price (`****`, which the scan reads as refunded, or
  `----`, never staked), and a check of 2025's refund rows against that year's board found each
  one on the board unpriced. The first run's figures stand; the rerun is the run of record
  because its input is the corrected build. Its private report and candidate file are new files
  (named by its run ids); the first run's stay where they were. Neither output names a race on
  or after 2026-11-28.
- **What v6 changes.** Three things: settled wins in the one-winner fits are no longer capped at
  the board quote (`settlement`), the store diagnostics read race_date statistics only and leave
  `dividends` uncounted (`counted`), and each run's private report and candidate file are named
  by its days, seed and run ids, so no run replaces an earlier one's (`outputs`). Checked, on the
  first run, against v5's candidate-cell file: the same cells; every row's raw figures (bets,
  race days, meetings, mean return and its se, mean odds, payout rate) hold v5's, and every Place
  and Wide row holds all of v5's figures. Whether and where a one-winner row's shrunk reading
  changed is withheld (D21). v5's private report and candidate file were left as they were.
- **The leaderboard's inputs.** Both v6 candidate files stay in
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/`. Being byte-identical, the leaderboard reads
  every cell of the second as an exact copy (no second row), so each v6 cell is one row; checked
  with `leaderboard_build`'s reader, no file written.
- **The `local` test.**
  `tests/countries/japan/research/test_pool_scan_run.py::test_real_store_one_month_reads_development_days_only`
  scans the store's last month into a temporary registry and a temporary report directory. Its
  outputs were not read and informed nothing in any spec.
- **Registry summary.** Each run wrote N1's pages only (`write_committed_summary`): v6's seven
  family files under `docs/experiments/registry/japan/`, each from its family's records alone
  (after the rerun, both runs' records), and their seven links merged into `README.md`, every
  other line of the index kept (`summary_error` null both times). The merge orders the index by
  family id, which moved lines that union merges of main had left out of order: at the first run
  `japan-131-exacta-logscore-v3` and `japan-282-exacta-fav-first-avoidance-v2`, at the rerun the
  three `japan-form-bands-t2-win` and two `japan-handicap-speed-96-final-win` families. `git
  status` after the rerun showed the seven v6 files and `README.md`, and nothing else under
  `docs/`. v6's files show only each record's E id and variant, its status and values withheld
  (licensed source).

### Superseded: v6's first run, at `54073fe`

v6's first run read the pre-#443 `dividends` build (above). Its records stay in the registry
beside the rerun's, in the same families; its private report stays under
`$HORSETRACK_DATA_DIR/reports/pool-scan-japan-4e04d2d8d0d1e96645bd0bac94a0b9b552811896d3146fd7cb21f935fab5fd45-2014-06-07_2026-11-27-seed20261127-e15d38e82012.md`
and its candidate-cell file under
`$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan-2014-06-07_2026-11-27-seed20261127-e15d38e82012.json`.

- **The run, at `54073fe`.** `uv run horsetrack japan pool-scan run --from 2014-06-07 --to
  2026-11-27 --seed 20261127` under `/usr/bin/time -l` gave a maximum resident set size of
  1,287,159,808 bytes (about 1.29 GB, under the 2.5 GB cap) and 1,964 s of wall time on this
  16 GB Mac; exit 0. It ran on 2026-11-28, a JRA race day, from 19:54 to 20:27 JST, after racing:
  it waited until 16:45 JST, then for the other studies running on the machine until 19:30 JST,
  and started once at least 40% of memory was free (42% at the start). Beside it ran two other
  tickets' study runs: #459's `horsetrack japan exotic-c5` (a resident set of about 0.4 GB at the
  start; it ended during the run) and #538's `horsetrack japan lineage-study` (about 0.6 GB at
  the start; a second run of it started at about 20:19 JST), with other sessions' test runs, at
  load averages near 10 to 16. That load explains most of the longer wall time. The read path is
  v5's: each year of the store is one `read_store_table` read through its day filter, one pool's
  board at a time is in memory, and one fit runs at a time.

Its records, one per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `24db61b1180d6782a5e5888c02518219e2d4dc3ff305385857dc3f22eff72ea9` |
| `WIN/omega_full/from-2024-01-01` | `4acdc72220a21f16cd4020c8b2e2a612601ea55537bf0e4beafaae281b750099` |
| `WIN/segment_effect/from-2014-06-07` | `dc40d668987cd8f96ad9ddfe44388860ea4aa37ea39a1994158cfc6030283dd8` |
| `WIN/segment_effect/from-2024-01-01` | `2f3606a3e4f62d3cbb2d2016e68519fbdbf3ad3c2d679b7941099a29ab55f3da` |
| `PLACE/omega_full/from-2014-06-07` | `ddb68fc92d706ab09b261f6f806c5b0644a187dc8785f36e70bacd983f8172d2` |
| `PLACE/omega_full/from-2024-01-01` | `ecd61601e1c80370dc986fb9f3caf1560f625803be08f760c3b32a97940db351` |
| `PLACE/segment_effect/from-2014-06-07` | `ab480242d3a97241ac512140272bbae71de788b01a13106dd2465715cae14373` |
| `PLACE/segment_effect/from-2024-01-01` | `dcd1c07738fa0c8cc4cd308379dc029b1ff69a42012be5e6971a76d5fd65849e` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `84134638e2abc13eb04dba55a0c404a474e57f9d6c21b7a2eaac5f5e5b6ccdf7` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `983edfe281547b21abe5edcaf9df4fbcf5bf5b8447f0fba483a95d0798327046` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `4266127c22a97f43f6e99c5e99adf7ec2b3c2dc92977f63276be3d30cd6e5308` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `7d0e91fc430c1116913880b9644a25dc431c202c5c69e94e925e82dfc06892f9` |
| `QUINELLA/omega_full/from-2014-06-07` | `a633b20325bcc697a20259da288fcf257a48f0644c55a9af0acf02c7adadd26e` |
| `QUINELLA/omega_full/from-2024-01-01` | `07bea183180afcd6853f67388276fc0aafc14bc7436a30241f23d967b88837ce` |
| `QUINELLA/segment_effect/from-2014-06-07` | `807bb251623e89d98a94f8d2407a74b2f7ddc32cbbb6960ba7f7ee909bf9d303` |
| `QUINELLA/segment_effect/from-2024-01-01` | `9d46a8f342c3076192bb6c841f2a9808b42e9d946b4b44f843aa5894b1fb3b48` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `298c3861051f8f5d8042dbe0a67519308acd09f3087b881c41c217d8efaa244f` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `0ab5a3355e8fcdda99ec246dd2e85581b6151e28219281cf6b6b7f100b24c6aa` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `5f15506949332c9ec9336d0b63a98dff4031908194da25015571a27014c505e8` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `24ec0ff0ded20328dd4981dd07b075e2d34ef815e5b0bb4b03d40f8a68954d30` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `13c46d876cd83b14252c8dbf57de5d46f80e5953a0365c88739b874504d78efd` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `2f4f4bc0c11ba1407458d354c4ca5d6612b70ba967013f0b88bb8768101130b9` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `8aa69173fe4692fcf10ba2d3bf3c74d05eab9f97bab9bc0fc83c9ba03d603db2` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `220bd878e151e4e4646d8ead1e79cf0bad1cd212c59899fba4a3967d2e90ef08` |
| `FORECAST/omega_full/from-2014-06-07` | `c91e413c28bcb18ed6f0a389e8636fb445374813efdd01056c4dc27bb95739b3` |
| `FORECAST/omega_full/from-2024-01-01` | `e794db1bde6f8c385dd6bb300143c50cc25773c820cbdf6c73c2d72899bd1d0f` |
| `FORECAST/segment_effect/from-2014-06-07` | `592b8f354e748b13a45e298949e2d378bd059517359c4d7f7f4d558ac04a3a25` |
| `FORECAST/segment_effect/from-2024-01-01` | `a7a0087d0f1a0ad251846f5a695906e3c3a17857cae9a23b718279026f68f474` |
| `TRIO/omega_full/from-2014-06-07` | `f030bce4357e21ad681c886766f29a909750b6b8823974d525750f5f806672cc` |
| `TRIO/omega_full/from-2024-01-01` | `ae724ec3d05abfc55ba6a7b9ede2ee7db0e9bb388e5e4e4592367fd179320bdd` |
| `TRIO/segment_effect/from-2014-06-07` | `293d4e514393aa8ab75375e455c8aed31aaac2c85e71fa077b0ef2da0704f969` |
| `TRIO/segment_effect/from-2024-01-01` | `bc64f7b3c49451e33ba26aa11a5d81147dc17bf6a34e53943e9d9f48c6b72f4a` |

The sha256 of its sorted run ids in each SPA family:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v6-cells-win` | `d1256f8c3c4d2126abd9b2d57425c5fd2a787268d1db6360db44e94bc7ea21e8` |
| `japan-n1-pool-scan-v6-cells-place` | `995dbd3a22ee8d84c5481e697403d427dd0d457ef5543aaddd37d58ec7dc51d5` |
| `japan-n1-pool-scan-v6-cells-quinella` | `54628d56a483bf6a4327abf8fdf907db091147d2f84c5e68e7e8c19ce668bece` |
| `japan-n1-pool-scan-v6-cells-quinella_place` | `38d31f8ff5b706d218d410a3f324b4313b4a6b1a8774b2d3c5a6d433c9bf798d` |
| `japan-n1-pool-scan-v6-cells-forecast` | `e40fbed6065416ba275d58022bb4804568385be4f9465fc4e905ee23bff9255c` |
| `japan-n1-pool-scan-v6-cells-trio` | `67077bf64b059d0f6766db40d69a6704aa54b4940d6dafa2a153058649c0f770` |

### Superseded: v5, `japan-n1-pool-scan-v5`

v5 (spec sha256 `43b7ad67f504f923ac079b38be4e740f4e3771934b94f55a38a3d856fa9fc6b8`) was frozen at
commit `65687a1` on 2026-11-28 (JST), and its run of record ran there. The seventh review of PR
#496 found the three defects that v6 fixes (above): its one-winner #78 fits capped each settled
win at the board quote, so a capped quote paid above the board lost its excess from the shrunk
reading while raw rho and SPA kept it; its store diagnostics read the race-id statistics of the
`dividends` row groups, which return a holdout race's id to Python before the development
filter (whether any did in v5's run was not checked, since checking would read them); and a
rerun into the default folder would replace the run of record's private report and candidate
file. Its records stay in the registry, superseded: every variant is a record and nothing is
deleted. Its variants count toward variants searched. Its private report stays under
`$HORSETRACK_DATA_DIR/reports/pool-scan-japan-43b7ad67f504f923ac079b38be4e740f4e3771934b94f55a38a3d856fa9fc6b8.md`,
and its run of record's candidate-cell file under
`$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json`; v6's run wrote neither.

- **v5 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-28 (JST) at commit
  `65687a1`, before any scored read of `japan-n1-pool-scan-v5`. It wrote the family's only freeze
  line in the japan registry.
- **v5 run of record, at `65687a1`.** `/usr/bin/time -l` gave a maximum resident set size of
  2,054,586,368 bytes (about 2.05 GB, under the 2.5 GB cap) and 1,239 s of wall time on this
  16 GB Mac. It ran on 2026-11-28 from 08:15 to 08:36 JST, a JRA race day, and finished before
  racing. No other job of this ticket ran beside it; another ticket's study run (#219),
  CPU-bound with a resident set of about 27 MB, was running on the machine throughout. The read
  path was v4's with one change: each year of the store was one `read_store_table` read through
  its day filter, with no row group skipped. One year of the store, one pool's board at a time,
  was in memory, and one fit ran at a time.
- **What v5 changed.** Two things: the store reads (`store_reader_guarantee` in the spec) and
  settled wins in the one-winner fits (`settlement`). Checked against v4's candidate-cell file:
  the same cells; every row's raw figures (bets, race days, meetings, mean return and its se,
  mean odds, payout rate) held v4's, and every Place and Wide row held all of v4's figures.
  Whether and where a one-winner row's shrunk reading changed is withheld (D21).
- **v5's registry summary.** The v5 run wrote N1's pages only (`write_committed_summary`): v5's
  seven family files under `docs/experiments/registry/japan/`, each from its family's records
  alone, and their seven links merged into `README.md`, every other line of the index kept
  (`summary_error` was null). v5's files show only each record's E id and variant, its status
  and values withheld (licensed source).

v5's run of record, one record per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `c1bcf7b417c32bec46abc8127f5fa986f3ea79bc7fccc18616105584886bcb7a` |
| `WIN/omega_full/from-2024-01-01` | `911a79a1708f5de245ab807da8d28b4f0d6876b199c19c0c3a4714c75f48a25a` |
| `WIN/segment_effect/from-2014-06-07` | `414ccb9908696ae62a39a5803cc145ca13c4ae477b211088edfed30c0ae436d8` |
| `WIN/segment_effect/from-2024-01-01` | `b48a70c41ae6948b8674a21467f92388e4df1433df114d7a70f676386aa6074b` |
| `PLACE/omega_full/from-2014-06-07` | `f4bc7a8ad28508da89ab5e3ef42791e9ee2babb1dc40a36d1db27f8bbb73e4ef` |
| `PLACE/omega_full/from-2024-01-01` | `ed0b7f3cbc28227a48798656974ec15ad11737322e3a0844696c9b8707090c50` |
| `PLACE/segment_effect/from-2014-06-07` | `ac0a8e0975510b8c6cb9fd15afbff070eaca22ba6f53dc420b6b52bf8da08756` |
| `PLACE/segment_effect/from-2024-01-01` | `dae0a97e6bded495b692c8cc5ee751dd15db421c1921c9f52dabab023ef2e859` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `e1470195a0529bb49270ef9694695c57f27c11cb6816b70e827580be96682541` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `6e562fde958dbee5b85d323e0a88cb3efa64fbd76b9bcae327a2b6effe3bc0be` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `0f7830db21607b6e1360d9648b24ad8dcbb9c6b1be466a05c90c320bc9838768` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `b4e055d7dce24fbee982ccc2f62ed072a422fa988ee433b42936474b91589fd2` |
| `QUINELLA/omega_full/from-2014-06-07` | `2b435df28ea4f48066c26a8e69ec58b9fa3e1c81e2a46d03431a34e0ce80c105` |
| `QUINELLA/omega_full/from-2024-01-01` | `50a7cefe619d7ed6700a87c02a28f0e7a3be36bef6bccdb6116fa0e87a19a65d` |
| `QUINELLA/segment_effect/from-2014-06-07` | `f23d45505fb9a7909459e8e35135bfccd2cd0d81756f32c9bd1ac7417fae0a3c` |
| `QUINELLA/segment_effect/from-2024-01-01` | `442d36255d6cba009184085c743cf016dbdc88d7af3b3a5fb98b0eda00cd2888` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `51c2b46491d66013f605cf401219d25524b30c79bc3f461b933484ad12bc52d7` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `5c1513b59015eeb56adaf64df2424b671984b19139c6a745c343805ef7debe6b` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `eb0a049d781689f1cfb433d4c4e311d1f26038c6ba63f664f5ce8820a867502a` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `8d19394cf618351a666126211df0907af10a8857377685ec6891b0401538f281` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `979b8ef4b0c2ed628f71407458df004c4ba24c339dccd61811ead06cac0589a9` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `defcc790031cb475762a4f2ed82bd7141a1031b9244eb5f44ac2104aca392037` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `da57278bfec05d47615ebd2a9ccb2bf1edcdba5d65e9d815ab03e6a27dcf9d5c` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `6c4b2f1dc30a906f58cc91e8ee84bca663c63c8a77efe9bd8b45b4da1c06d8fd` |
| `FORECAST/omega_full/from-2014-06-07` | `baae827ef9176862fb3fa5897772caf04468717835327eb40fd4e035c6c8042c` |
| `FORECAST/omega_full/from-2024-01-01` | `83ebc0505197bc2549c2c8498bf3fe103bb9b813688fbb8d3cf1f1c8aa3a2112` |
| `FORECAST/segment_effect/from-2014-06-07` | `f48f6b36148e0857d78ff645fc623c8bde98933a72993927ea73d7245e80fdd1` |
| `FORECAST/segment_effect/from-2024-01-01` | `93d49d2af27074b2326a367f10427ced1166ae5324e9e5b88e0463d097acbb00` |
| `TRIO/omega_full/from-2014-06-07` | `7216b3d0c81fceee14ba1e84baeacd34b25a129582b90f3481bcb63993e8ee2c` |
| `TRIO/omega_full/from-2024-01-01` | `3b1dbdf6bfd33341022413894e8a2dbaf2957f3ac8a6f7feb8430258976b6588` |
| `TRIO/segment_effect/from-2014-06-07` | `ea2b164fa3b9135bc47dcf8c759f7a36b11ddc321c605249f820174d1c304b9f` |
| `TRIO/segment_effect/from-2024-01-01` | `b94fad23fbfdddf3ba7971b8173440d5471dbb0a0e4131cf48b464b741efb6fa` |

The sha256 of the sorted run ids of each of its SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v5-cells-win` | `44fd86191ee928b09feb73a0b7df41c0f160e3346397ab638a718de55285b4c5` |
| `japan-n1-pool-scan-v5-cells-place` | `61e9ac0101e7967609e6ec8be95aba99affc026d75621654b6214f5e7be32f8e` |
| `japan-n1-pool-scan-v5-cells-quinella` | `94e53f35563a8a692d71ba90930b6c976ca4df79529a3ebef80fb71d577d54eb` |
| `japan-n1-pool-scan-v5-cells-quinella_place` | `ec43f4aa9a51bbaacbd9ec0fdebb23488268da62d851d88feee4e7809ce7e7a9` |
| `japan-n1-pool-scan-v5-cells-forecast` | `b9a867ebd15248c450649f10cc22878e56d906b93b765250f048abd1164309f9` |
| `japan-n1-pool-scan-v5-cells-trio` | `346b724d27282f442a15d84b23c5e33249d6f1b914dc55783c5bbdcdcc664be8` |

### Superseded: v4, `japan-n1-pool-scan-v4`

v4 (spec sha256 `a166d99f7320b9596ffc31d1db4d1efdd33e0a3c84a2aad0a16c98a92139a91c`) was frozen at
commit `520f100` on 2026-11-28 (JST), and its run of record ran there. The fifth review of PR
#496 found the two defects that v5 fixes (above): it skipped whole every price row group whose
race_date range spans a holdout day, so once an export holds a holdout day a rerun would leave
that year's development races unscored, and its one-winner #78 fits credited a dead heat at the
board price. Its records stay in the registry, superseded: every variant is a record and nothing
is deleted. Its variants count toward variants searched. Its private report stays under
`$HORSETRACK_DATA_DIR/reports/pool-scan-japan-a166d99f7320b9596ffc31d1db4d1efdd33e0a3c84a2aad0a16c98a92139a91c.md`,
and its run of record's candidate-cell file under
`$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan-superseded-v4.json`.

- **v4 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-28 (JST) at commit
  `520f100`, before any scored read of `japan-n1-pool-scan-v4`. It wrote the family's only freeze
  line in the japan registry.
- **v4 run of record, at `520f100`.** `/usr/bin/time -l` gave a maximum resident set size of
  1,411,399,680 bytes (about 1.41 GB) and 1,513 s of wall time on this 16 GB Mac, with no other
  heavy job beside it. It ran on 2026-11-28 from 02:37 to 03:03 JST, a JRA race day, and
  finished before racing. The read path was that of v3's run of record (`826c665`): one year of
  the store, one pool's board at a time, was in memory, and one fit ran at a time.
- **What v4 changed.** Only the zero-hit cell rule of Place and Wide (`zero_hit_cell` in the
  spec). Checked against v3's candidate-cell file: every row of the one-winner pools, and every
  Place and Wide row whose cell paid, held v3's figures. Whether and where a zero-hit cell's row
  changed is withheld (D21).
- **v4's registry summary.** The v4 run's `write_summary` regenerated the summary of every family
  in the japan registry (`summary_error` was null). The committed summary took v4's seven files
  and `README.md` from a read-only view of the japan registry that held only the families it
  already held plus v4's, into a temporary directory; every other file there was as before the
  run. v4's files show only each record's E id and variant, its status and values withheld
  (licensed source).

v4's run of record, one record per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `4184d46e67190cbdfc931679befefb0a0be86e9864d91e094dfe7187df466a08` |
| `WIN/omega_full/from-2024-01-01` | `df0e27b7235c7c5fea958e2bbb06fd66b3bb9c0f0dcb357ded50d432ef0da8ac` |
| `WIN/segment_effect/from-2014-06-07` | `7818fca2b4ce8e36c640291a83b86ae266c5ab7bb5f6bce126f353246953125e` |
| `WIN/segment_effect/from-2024-01-01` | `996a4c245b0850af9708b84a3eda816204d6ea8973f76b1cbfdfee236e7696a4` |
| `PLACE/omega_full/from-2014-06-07` | `7e281865de0e3e3f808ce6dd2f4c89d91086a9bba22cfe31a7bc83bde2488444` |
| `PLACE/omega_full/from-2024-01-01` | `a458fbd5fc322c785fe453fb6b3b3a28ed2b9c0091db8dc26ab48a56297eca83` |
| `PLACE/segment_effect/from-2014-06-07` | `bf1bcbf85a58026f2cace4d869d7ec08a3cdcfbf1579344cb01b7d4eeaa9f548` |
| `PLACE/segment_effect/from-2024-01-01` | `cfc7bb50530aa043382b7935946b0d74a88fb2f83f9796f01d8bc9bc1a942435` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `1bffd236e6537474dda15d3b6f1eff535309f2b16113d6af1427f12099e0b312` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `182542bea67c8b5cd66cc9972194cfa0c451d5b6d46af63448872f17d8138ca7` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `e6a0821b7def947f7c1bbce5a425aade3b9791077b505cb8a1e677222f4fe5cd` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `6e7838adcd05da881604ac64a51a2516f8dde21aa8c09a363fc575c7211374ed` |
| `QUINELLA/omega_full/from-2014-06-07` | `f5c5a9a49c4399ac7f79dcf5892705cc278d636a2b862c752a65199fe40e548f` |
| `QUINELLA/omega_full/from-2024-01-01` | `40fabde019fcffade109b83beb90f14320b90bf5b1974a7c7c5fbd5dec5dc606` |
| `QUINELLA/segment_effect/from-2014-06-07` | `f6f79be56698fda16f8c19de4a2276c47ae9ef00a7f9f4e7cb0e65ca4853387d` |
| `QUINELLA/segment_effect/from-2024-01-01` | `33506a871a5a84cb1f4695837309047ff7e62a5eefd62731b5e2949ef3e43d41` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `f655217e4ee063ee572e224318a2afc47c77aa1e9163dd0d71caee3c862fabcf` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `8518f54250ddc57ef5d06d23b7297658c19e73f250590aec4741262c37858144` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `20421659900455970d371b76ba767ef15449e7c2efe6606b32eecbab8ab27f5f` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `9fa29d8f0a444489aa319ce58ca384d4ed34bb7b279a9c75cf2bc16afd469c29` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `f90f646711e1fdb91d95a1fbef9c3542ac70cdae331894136d787ec9e3a3633c` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `145d5d936051cb66921484851f7514ed9474e0959259a02977adb09b92e2204b` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `7688a35bd2108bf32fe632bf92a8e3ae81f3ffa875da550b8f4860e0cf96e349` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `c7197f4a56ae5f23b0347ccfbc6a4fc228364fbe1a7e95dea63435ddda158253` |
| `FORECAST/omega_full/from-2014-06-07` | `253123fe2a55d7894c7de00e47b82cb131f2c40f80d929dab73fbbf87deb17bb` |
| `FORECAST/omega_full/from-2024-01-01` | `653a884d01591dc4fbc4dbd12315b9583a890fc27ccf9b944b2f30a44f9830a0` |
| `FORECAST/segment_effect/from-2014-06-07` | `217005db55bda5c54d5b7315094efd66f7d4cb57624b84779e1183e46734ad72` |
| `FORECAST/segment_effect/from-2024-01-01` | `89f5a7d5217b6ed225ef34f7398abeab4904567ccf39374993874d6c1b2a0b0b` |
| `TRIO/omega_full/from-2014-06-07` | `f115af5fe763728aac60c903db6d7d311deabdbe3dd14c12e3f19babefb58ce9` |
| `TRIO/omega_full/from-2024-01-01` | `4384a78a17256c3a6c7e225c5d6cebb8f477e565105154aade2c2446c5b2fdd1` |
| `TRIO/segment_effect/from-2014-06-07` | `9a1481f6865ec3929c0932d54de78b579d6cf822e0a1eb30b92f4512a5946b7a` |
| `TRIO/segment_effect/from-2024-01-01` | `45e6703144aae89613c973a0624e2115e3520f2a934234400effd68f723ab2b5` |

The sha256 of the sorted run ids of each of its SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v4-cells-win` | `d2485922a85b6c2b3ceb6831c15f346273286f8f27be9096d20911c736c48df0` |
| `japan-n1-pool-scan-v4-cells-place` | `11a56ef6f44beeb96b3783b957ac1c6af8fd90c90a7f797fe69029c6f12a471d` |
| `japan-n1-pool-scan-v4-cells-quinella` | `1f361c40247f6c756afdc642f041d7b2f300fcc469ed56cf10aa8f1e37582b68` |
| `japan-n1-pool-scan-v4-cells-quinella_place` | `ee2354dbd9ac6feb96d89c712de6c57230af215180e29a71250c6b1e0ef489b6` |
| `japan-n1-pool-scan-v4-cells-forecast` | `5881b84a77fca391f3db019bdf28e8eb5bbafd5a4a0df9198087b5e88fb01a0d` |
| `japan-n1-pool-scan-v4-cells-trio` | `b22385f5dbecf5fb1b332d2abac39bad18a554a8f431e80c637e0ea10a4a69e8` |

### Superseded: v3, `japan-n1-pool-scan-v3`

v3 (spec sha256 `785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c`) was frozen at
commit `de0413f` on 2026-11-27, and its run of record ran at commit `826c665`. The third review of
PR #496 found the zero-hit cell rule that v4 fixes (above). Its records stay in the registry,
superseded: every variant is a record and nothing is deleted. Its variants count toward variants
searched. Its private report stays under
`$HORSETRACK_DATA_DIR/reports/pool-scan-japan-785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c.md`,
and its run of record's candidate-cell file under
`$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan-superseded-v3.json`.

- **v3 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-27 at commit `de0413f`,
  before any scored read of `japan-n1-pool-scan-v3`. It wrote the family's only freeze line in the
  japan registry.
- **Merge of main, at `2415969`.** It brought #525's summary-file rules into `core.eval.registry`.
  The hash of `prereg_spec()` did not change.
- **First v3 run, at `2415969`.** `/usr/bin/time -l` gave a maximum resident set size of
  1,791,098,880 bytes (about 1.79 GB) and 1,409 s of wall time. The run exposed a numerical defect
  in `core.eval.segment_roi`'s importance resampling: with log weights of large magnitude, the
  normalised weights can sum past the tolerance of numpy's multinomial, which then raises. Commit
  `4a6cd80` renormalises the weights, and the scan was rerun there. The first run's records stay
  in the registry (below). Its private outputs are kept under `$HORSETRACK_DATA_DIR/reports/`
  with the suffix `-first-run-2415969`.
- **Second v3 run, at `4a6cd80`.** `/usr/bin/time -l` gave a maximum resident set size of
  2,550,972,416 bytes (about 2.55 GB) and 1,326 s of wall time, above the 2.5 GB cap, so it is
  superseded (below) and its records stay in the registry. Its private outputs are kept under
  `$HORSETRACK_DATA_DIR/reports/` with the suffix `-superseded-4a6cd80`.
- **Memory, at `826c665`.** Profiling the read phase showed the cause. A year's read of
  `final_odds` decoded every pool's strings densely (trifecta's rows included) before the pool
  filter kept N1's, and Arrow's default allocator kept each such transient resident, so the
  resident set ratcheted up year by year. `read_store_table` now decodes its string columns as
  dictionaries inside the scan and casts the rows it keeps back to strings, the same table. The
  scan reads each year's board one pool at a time, column-scoped, and allocates from Arrow's
  system pool, which hands freed memory back. What is read and scored is unchanged: the spec hash
  is the same, and the run of record's candidate-cell file is identical to the second run's.
- **v3 run of record, at `826c665`** (the tables below). `/usr/bin/time -l` gave a maximum
  resident set size of 1,822,457,856 bytes (about 1.82 GB) and 1,288 s of wall time on this 16 GB
  Mac, with no other heavy job beside it.
- **v3's registry summary.** Main's `write_summary` (#525) maps the japan registry's legacy `/`
  family ids, so each v3 run regenerated the summary of every family in the registry
  (`summary_error` is null). The committed summary took v3's seven files and `README.md` from a
  read-only view of the japan registry that held only the families it already held plus v3's, as
  v4's do above.

v3's run of record, one record per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `f4cd2bec2497093c3e79105107838c4c686d112f9dd1e9a7a0e173a563974ad9` |
| `WIN/omega_full/from-2024-01-01` | `c1ec42c8432bf25564f090a486ba5477a860768c6dbf9e1959e1827b12e33b79` |
| `WIN/segment_effect/from-2014-06-07` | `da3fd6b60ee3532bccdfb8fc79558d29f6caf487188d8b5dc383f51ee9170282` |
| `WIN/segment_effect/from-2024-01-01` | `924a92099ddc55ef495627713d0feff6a9f972b146e1dc80f2baf5ee863a3486` |
| `PLACE/omega_full/from-2014-06-07` | `21a608a0694050425e23b2daaebb26fb74ecae9ec84a6a0dea3ffe49fa712dab` |
| `PLACE/omega_full/from-2024-01-01` | `163f6a7d549a853ebae383d9dd583f7c295745ed2f09a7dd57e1ed9e537a7acb` |
| `PLACE/segment_effect/from-2014-06-07` | `215e17a75e866e0cfde49e4f1a51eb34fdc456e1e4136d1fc9f36898b9f415fe` |
| `PLACE/segment_effect/from-2024-01-01` | `cac7194cecbabda17018c79df291a217e1e92425f0093876043d53a75c85e36b` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `2c6ae60a233c018c713e954d5a9f746d1417baf2a80b223259badd95da26a1a6` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `6cbbcbeaffe80ef2a21ca063c529cecfecee9fc39e5e5de52b2534ac37e2e669` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `f364714f493060d97b2fea269e03362d506be81f72af0a010358fa152646ef30` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `99ec3de79a008aefd006528317aa7808da7e98ac834f661d0adcbf08d23d0602` |
| `QUINELLA/omega_full/from-2014-06-07` | `9c43a165f4d271370c8405e7c1c594d8a80e4e8d5c0d3d48b83cc1a2e6005e82` |
| `QUINELLA/omega_full/from-2024-01-01` | `535a62eb0ea4a7f3bd6dddc531573bc2ec321e3c5fbb1266af39f9ed69d72945` |
| `QUINELLA/segment_effect/from-2014-06-07` | `cceeaa123cb12daa6ce3e54e82030eaef69b9127fc23137811a895b58f401231` |
| `QUINELLA/segment_effect/from-2024-01-01` | `c27282abe5e26404610b1f3ac55cf97e779e35692defae58703350e5aaefa388` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `489cca6cda45c48285eaee270b22ab7b28c3fc951d706a5e39a700d21fde7d09` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `b956ca27e77d7e1d141ccbf4003923b8583ae2709a49768f6dc24f99961b23cd` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `3cf48969feb560500b2b40a5e3c50d439a82a9bb96db3f523ff577cac4bddb69` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `ed0672a17fdc4814ff3c0fecd87bf718d23249ad928915878509f42318ff1414` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `34a5f4c8e0c64b46c954003dee201eff24b4a8d6686d8f5d0b018f12be4ca820` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `ef9658b9a61eb0d1dfeb30dc6845ce190b466283a0c962b1369a156244fea5f1` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `12e968f9d1a8aad6034faed43842822438160cd6d2e9ecdf43dda546acba3177` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `6e1984bc7a921b3352a78b0227fcf9059f2c84d7686a05f43ff0eead768e4b7f` |
| `FORECAST/omega_full/from-2014-06-07` | `7bd967b69a0c8ac90011cf938fcf80a601379ce0a93bf475f553f0837bed17d6` |
| `FORECAST/omega_full/from-2024-01-01` | `77b715f67671344dc7c83015721eaddcc6cfd7868ae909809027a5d5adb66b9b` |
| `FORECAST/segment_effect/from-2014-06-07` | `e947142347468b7d107117b68dfb2798e0d16784db024af5a885bd6819f53fc1` |
| `FORECAST/segment_effect/from-2024-01-01` | `f04e6bc5f78a9c8f1883f8ceec90d519a54446107620e5c22a558d7f37e38677` |
| `TRIO/omega_full/from-2014-06-07` | `369821981bc509933548aec6b3406f48ad35728ddf03b53a5100df14b9cedf4d` |
| `TRIO/omega_full/from-2024-01-01` | `e4b2b138bb50054048f8144148c500d10163a32b27aa1876e8e5fd065651b8d6` |
| `TRIO/segment_effect/from-2014-06-07` | `592cf508dec50a6c084fec38bd5a1e2dcd9579e3a0ca0ffab77c90615820ee2b` |
| `TRIO/segment_effect/from-2024-01-01` | `c970ee63d994a080684bd4d5a6dcca092a2eee6f645a41926bd7a56b9fa38af5` |

The sha256 of the sorted run ids of each of v3's SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v3-cells-win` | `7df0047431d064574de0d982df31ff75f48582cb019afc3310e91f0085b85b66` |
| `japan-n1-pool-scan-v3-cells-place` | `6120ddc8fb2e40e1246b9141b67d490bda5234f4005d2688e000566903081c83` |
| `japan-n1-pool-scan-v3-cells-quinella` | `3b79a848141517f318db9355fb1f0dc4a4902b616c24930743122a769b592cd5` |
| `japan-n1-pool-scan-v3-cells-quinella_place` | `4bb173ad774b1f25b0911ff3b373eebe40f89db9f46e30f889df150cc4d69c66` |
| `japan-n1-pool-scan-v3-cells-forecast` | `92a54b59d1dbdc1895506a3ccedb4b06087adba9a542f784d450b9b6d282210b` |
| `japan-n1-pool-scan-v3-cells-trio` | `3d22a34bbdb43de495c505466716983a9f326e7688b5c04cdfdadf2d4b284e47` |

#### v3's second run, at `4a6cd80` (over the memory cap)

Its records, one per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `e27049ad73ecd75c4b901ad933f62d54671f4886785b2819c774212caa5092bb` |
| `WIN/omega_full/from-2024-01-01` | `ef762d83a23503b638789a219cf0dd27f22fa6ed3b2ae0c2c1867b455424ac04` |
| `WIN/segment_effect/from-2014-06-07` | `d85fd5fab74f5198eb01cefa0a4a47afbb548457bc4935be12febe689f49a987` |
| `WIN/segment_effect/from-2024-01-01` | `78700d189c74f050399a77b8fda2a36d99019a64fee4f5d997625fbab6e100ba` |
| `PLACE/omega_full/from-2014-06-07` | `677f3e6f77020ec399e0e452e98c646e1ab4f9498de30542f2cc119ff877b1c2` |
| `PLACE/omega_full/from-2024-01-01` | `b0bbe4deebb73eff074ecf71c4cb2c60be17168277ed1caa02df28b11be09daa` |
| `PLACE/segment_effect/from-2014-06-07` | `7d71c76eab9d1b22c7e1b03bdc32a5e2932722b16e8056607ab3e50d198cb2e7` |
| `PLACE/segment_effect/from-2024-01-01` | `de2ab372b871dafdff4d0c39758b7bbb5ce4a5c1af30353a6d1f32a064d181d3` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `b59dc3bb6a1b07b9d74cefd34ff5a62c1c3c9855570f71b776f5068005b6bbd8` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `bd8294cf4ced0f0a155c878d5504f6b53b00457f8e59453c621db8d5a3c77c20` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `73ac22352b16ba9efcd81600716f674e43fcbd16af45115c38ae0b34e9f5e095` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `49ab7259ebd984bbd4102e8b7e7906dde53af011e075e63d4a646d69b32491fc` |
| `QUINELLA/omega_full/from-2014-06-07` | `7c252d6b3319813769957f0f9727c1df806d821db530bf0c2c064ea5324460a2` |
| `QUINELLA/omega_full/from-2024-01-01` | `5c85674c389cd1482469d8f6f4aeaa8661fb72cbfcb09892207d817b6ca596cb` |
| `QUINELLA/segment_effect/from-2014-06-07` | `cbaca8584951918727889b7721b36130a2861f5668e461fb128a9d26b81dc86c` |
| `QUINELLA/segment_effect/from-2024-01-01` | `e68d54466a217d0bc1c9cd7527bfccec6b061b68a98250f0f92d6f67d80faf33` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `db4e4da26b3fc7f3251dface560878172d3f8c1fffebcc7138666fca0d87cfcf` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `064f95375068d31cff0c1b9042217a1736f1db7ad06f69533f9752d31068b15f` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `4a5e11a2d02cc0982a7e2348cda1886766c1b473679de7ebbbc38ec71a165082` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `8cf15fb890efa735bf3d22433f0c72174c431d910ea7c3daf6c57fcf20fe90a6` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `3f3de50a59c07fac5d5d608e3e1e03c7736f6e86292bee828b266c625e39eb77` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `d47c3461a2bcce3cfced17e6ca1bc269bc7dde4906fae8500838e2e43180ba3a` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `4e231defb9b0a8a2c0dd53f13a1ae8a7a585c6b457b7c33fb371a5415b1f58b6` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `520a026cfb951bea395868b72263f1f4df06c5a5e4af9c4afb92e7f65b9c6c23` |
| `FORECAST/omega_full/from-2014-06-07` | `8959a515be69fae993ff3cc86c6e75471625b4b3af252bde88bb7ed90257c46f` |
| `FORECAST/omega_full/from-2024-01-01` | `7195c64fad21514bf3f7ecb92dc04086e3807700174013888d7b7d4647df1493` |
| `FORECAST/segment_effect/from-2014-06-07` | `f6916ad97eed6d467acddd2b3b9b9646c444323daa34559c3ca6d5151821fc35` |
| `FORECAST/segment_effect/from-2024-01-01` | `6c4c288bfc4c881f7968ff6853f26d4591f414b3cfbcf3531a8786237fc363d6` |
| `TRIO/omega_full/from-2014-06-07` | `d264b26b4bcb2b085dace6ba7004b4125d9edf3ba5d320de80e0e19342777c2e` |
| `TRIO/omega_full/from-2024-01-01` | `358b628b1e2983a0e5220444281ab0d9bf677b780ac5cb0270fd80dc993ac6d3` |
| `TRIO/segment_effect/from-2014-06-07` | `dd92ba5ac9c142a96f4f1ffb12a4200fe6c834aa5b85ab5a8da81522a604b0d8` |
| `TRIO/segment_effect/from-2024-01-01` | `1dd649f645c735efd2a89bde187c9d158d576be416b1231bbec50a13ffc88bac` |

The sha256 of the sorted run ids of each of its SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v3-cells-win` | `2d3bb1ef54947f21c5348466ce0d928d7422f5efc115a5b01bcfc99872ee4760` |
| `japan-n1-pool-scan-v3-cells-place` | `87392ea997bc99a6c4ae82e75d3146a20aea86dbc33269ed2e1b668b3af90b2b` |
| `japan-n1-pool-scan-v3-cells-quinella` | `8cf1360e97914632d7cbe5554365628ee2ebbb071d304a8f1ddc0b4b36c92023` |
| `japan-n1-pool-scan-v3-cells-quinella_place` | `3926df7190616b93e2a723b75db198216885a9e0e50f1e884ad9cb6568df0c26` |
| `japan-n1-pool-scan-v3-cells-forecast` | `26f748dba07abee04c4130f5afdd7f58d287f9fe284550555582c3bd2c0f5692` |
| `japan-n1-pool-scan-v3-cells-trio` | `1c10f6f02c90311ed6264c1ccdc350311f1690da99da1fb7f7523faf8c98c229` |

#### v3's first run, at `2415969`

Its records, one per pool, shrinkage, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `76f7fe3bd44abbca7afe7fe12d1024b1e42362a03f1c57bed51b79f44ae67193` |
| `WIN/omega_full/from-2024-01-01` | `ca2f7d0f7af6f62e3b33f927e6068791ae718b6e1f576941031ec7c38b47e1ff` |
| `WIN/segment_effect/from-2014-06-07` | `f545b575d5bdfb63ab2ac8ee6a751ec9ccfee37971f8d062b8383a89741705f0` |
| `WIN/segment_effect/from-2024-01-01` | `5562ae4d1e397fcd7cd3cb33cdea66c25b57d0aa9dc540c62185dcfd127b23e0` |
| `PLACE/omega_full/from-2014-06-07` | `246a5c194b385547a84dcf3073d7ef80931bdacb15d9553159e85b7cef59968d` |
| `PLACE/omega_full/from-2024-01-01` | `45c88362958fdb4f0f3464d657940faee8a21c5c9d916f84a2d0ac2ebe8136a0` |
| `PLACE/segment_effect/from-2014-06-07` | `995ec94399199804774caac5a0f0e676f17ede4c70fb7fa77f7ac2076ea40443` |
| `PLACE/segment_effect/from-2024-01-01` | `19e86afae5ddb4145029b4ab114cd6c4b2a1a5431211ebaa6299584235912702` |
| `PLACE/roi78-omega_full/from-2014-06-07` | `5e96cf26ea8c01379d88b646e641525723fbd0cde74a06bced2c80b079f4e632` |
| `PLACE/roi78-omega_full/from-2024-01-01` | `e0e93deda695435306083013cc43e7e15383bfcd9010baeca219774ec374ec54` |
| `PLACE/roi78-segment_effect/from-2014-06-07` | `fd6d4ec0aed2fb1841c5ba618e24d91f317cdd41fdbbd1d0f73b92f148ae6597` |
| `PLACE/roi78-segment_effect/from-2024-01-01` | `4e384bc8098e7ea5a24a5d85b4f21026feb1c966cb5495517fa9cc57419c20a8` |
| `QUINELLA/omega_full/from-2014-06-07` | `7d5f0b38b809e99337a96ba176a766a19e0074a59f7f2d6917e077aa4a24a1fa` |
| `QUINELLA/omega_full/from-2024-01-01` | `33faee11af571fe6f301e29a36269150cf2f78ad7f2d03705b7bd4da33c3dfeb` |
| `QUINELLA/segment_effect/from-2014-06-07` | `ac1609a7fa7f01b055f6e6638421ef3dc2a860930f68302582bb0e990af604aa` |
| `QUINELLA/segment_effect/from-2024-01-01` | `95b1b7af4811978b4694f5dd9dfafc55cb17d95cca41c2cbe629b352ba831c66` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `9e358a1e09905ebab14bb7955d33785d1b141022c6459a2fdf621f53fcdaff7f` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `6e2758a1de34d14a7e7035effb6c376c86727e76f417e61416f5ed99b83bc774` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `1e680f8cb35338f6840746948a2c38169b6db14b042f7e1e0e96880f898b4c30` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `d3b3b2e1dd75d50880c365193de89060ddc296b364cc820b36fd14c6df1adeb8` |
| `QUINELLA_PLACE/roi78-omega_full/from-2014-06-07` | `9c389eb89d78445f216f45b0be8ec77761ebe674bfd7aef5a3e746b0010a2f5f` |
| `QUINELLA_PLACE/roi78-omega_full/from-2024-01-01` | `419f0d7c175e06e5e64e277fd094052e7708fa3fc4967e9e582dd087ce167bfd` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2014-06-07` | `4bccf165d5cae5f1a02b204a0071108da8e6620b1fcbdc9ef3a26266c62e97ff` |
| `QUINELLA_PLACE/roi78-segment_effect/from-2024-01-01` | `ed8770b0564024708a84390f8b75fa626fff22c3348255c089419a54ef344a73` |
| `FORECAST/omega_full/from-2014-06-07` | `47adc3c00899d87d779dd79301efc12ad47bde25b2ccd5506fda076972069c57` |
| `FORECAST/omega_full/from-2024-01-01` | `85de16e4b425f1197d7e3c222d37333655e29d023e710134ee5bc0091dbd5460` |
| `FORECAST/segment_effect/from-2014-06-07` | `49ecf32279af61a69663ec8b743edd84b5ebdb5da46053e53014710fbd8c94cb` |
| `FORECAST/segment_effect/from-2024-01-01` | `535db641d2f9422bd1c9a9fc384e7532458f014c4b3115838a3d06d8b3431cda` |
| `TRIO/omega_full/from-2014-06-07` | `5d3a7613555197c6a50bd975606be0b4a16339e0f7dcae40b7b02173162d1338` |
| `TRIO/omega_full/from-2024-01-01` | `3860b274e131ba5048429e9457989e850c5157072ed8db88b634bbc958da3756` |
| `TRIO/segment_effect/from-2014-06-07` | `aefd3e2bc16b1a36818aac789480c8ffaf81818df275d1ef1ff834a32850e159` |
| `TRIO/segment_effect/from-2024-01-01` | `2bb7cfb10526fe649658c19c7c65c9993f7a378fceb33a3465c3fa45f9e52854` |

The sha256 of the sorted run ids of each of its SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v3-cells-win` | `df1baa74576f0f614b8abe17a881d53b9ec7748053e23461c34bd4b6ce72e293` |
| `japan-n1-pool-scan-v3-cells-place` | `7a3b2f8fd4b985350e9090d7d66a1b408d1fbb41e8ee3b44cd1cbcb81c63ea1e` |
| `japan-n1-pool-scan-v3-cells-quinella` | `fc8281016437171433e5c80bf85d54499aa629e96b63609ab51ee679bb5a5f00` |
| `japan-n1-pool-scan-v3-cells-quinella_place` | `f3cf4239e993b5010316e938339e19da96b2f24c3e78f2125d21c39b0774cddf` |
| `japan-n1-pool-scan-v3-cells-forecast` | `60c1f33a0e11570cc19d643296cc0ecdf3ab02f397b95a2910689ab7245fabcb` |
| `japan-n1-pool-scan-v3-cells-trio` | `18f5485af8a2f32096c5e5059324978515ce900a73584c1ac85a7b4c898b7c9b` |

### Superseded: v2, `japan-n1-pool-scan-v2`

v2 (spec sha256 `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6`) was frozen at
commit `0f7ed74` and run there on 2026-11-27. The second review of PR #496 found the six defects
that v3 fixed (above). Its records stay in the registry, superseded: every variant is a record and
nothing is deleted. Its variants count toward variants searched, and its private report stays
under `$HORSETRACK_DATA_DIR/reports/pool-scan-japan-ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6.md`.

- **v2 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-27 at commit `0f7ed74`,
  before any scored read of `japan-n1-pool-scan-v2`. It wrote the family's only freeze line in the
  japan registry.
- **v2 run of record, at `0f7ed74`**. `/usr/bin/time -l` gave a maximum
  resident set size of 1,643,855,872 bytes (about 1.64 GB) and 511 s of wall time on this 16 GB
  Mac. One year of the store was in memory at a time, one fit ran at a time, and no other heavy
  job ran beside it.
- **v2's registry summary.** From commit `2d637c6`, `write_summary` shows only the E id and variant of
  a licensed-source run, with its status and values withheld. A status can be read off the data:
  v1 recorded a cell rule with no bets as failed. The run recorded `summary_error`, because main's
  `write_summary` refuses the japan registry: it holds #131's and #282's families, whose ids
  contain `/` (the fix is in PR #280). The committed summary under
  `docs/experiments/registry/japan/` was then regenerated by `write_summary` from the registry
  records of the families it already holds plus the seven v2 families. The seven v1 files keep
  their rows in the new format, the seven v2 files are new, and `README.md` gains their seven
  lines. Every other family's file is untouched. The other licensed families (the c1 nowcast and
  c4 meta files) take the new format on their next regeneration, by their own tickets, so this PR
  rewrites no other ticket's file. The families of other open tickets wait for their own PRs.

v2's run of record, one record per pool, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `f337ce93c18b4b7881c87076a6a280d59db877ff3682b48af798c6bb9cc86cf1` |
| `WIN/omega_full/from-2024-01-01` | `44bd24b271b34c8cce27ca91315b70a49d29f0e588455e796c89d0428967fcd2` |
| `WIN/segment_effect/from-2014-06-07` | `3249315027d552fc4deb3b41c3db4fc3665d0b0640cfd832c85321d8ef150747` |
| `WIN/segment_effect/from-2024-01-01` | `81290b6d708800d21fd896925fe6df9461818fecb70fc50b4a24766baef1e77b` |
| `PLACE/omega_full/from-2014-06-07` | `c9cdfbb19c1c0013c3399b080b52d6bc925718d610c85a606e404f3ae33ef871` |
| `PLACE/omega_full/from-2024-01-01` | `cf7726e9dbc0c21c4e75fb612d1a3aea7c8038e10b95fc91d423d40700c07a0e` |
| `PLACE/segment_effect/from-2014-06-07` | `a0ee685a491c07406512cf725d20df0b75b1e89e3edd50e2ffe33399e15a5c56` |
| `PLACE/segment_effect/from-2024-01-01` | `784675ac86a786d35829cc6c3ee5a39dd3723f8f5d943f47bfa6f42844620d2b` |
| `QUINELLA/omega_full/from-2014-06-07` | `6e6189105479e00f8c5f34a85f8dd0eabf395d51abf0566629f76d8332e71cca` |
| `QUINELLA/omega_full/from-2024-01-01` | `bd56e748aa02dd234b2b8a8c09b46b228da502cd5aaaea4b1cd5de88683a08c2` |
| `QUINELLA/segment_effect/from-2014-06-07` | `d2adc4875a92bf5a4f307951d3ad7b6c79a8785f45d0dd5c75e3231337a82c5b` |
| `QUINELLA/segment_effect/from-2024-01-01` | `984de4afc719ba4080b3eb281a0b8e4e0aa82a08c86fe406e7987231fa7bea5c` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `d3ba8e27a49673439d24f1ef7d30fafc0ab3f3a092a3ba9d952e2032e30272da` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `c92f57e3f28cf8c05ec0f810b12f816524e63c0a0c860ffd40ea165df821e2fb` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `4a56f92b3fca259442be78a8a71c00737bb53862ee810c18022d0841b93e2972` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `e0d8f0aa0305ee3249407c4e3e000f3e0f8c3df2db8c69de26951f92d46a2c87` |
| `FORECAST/omega_full/from-2014-06-07` | `3701ae772066ca71682d7e99720e01683e347b3c4e26237ad8e71d79e78db901` |
| `FORECAST/omega_full/from-2024-01-01` | `c77ca01dd19b3e1e16058c4e4cdb5a725fc920da59f3c72f943279ba2f36c0d2` |
| `FORECAST/segment_effect/from-2014-06-07` | `1ff113155d437b6ab56861c336f38caf0c6fc33983823d29a760677fe38bd767` |
| `FORECAST/segment_effect/from-2024-01-01` | `f4dc0eae1020f2d08b3e1365ef5036a56c796bb75bfc40c1b7588788f147fab8` |
| `TRIO/omega_full/from-2014-06-07` | `b0438fe67cad8f0091c5264b020586d11e9a8d951f982147d72d1085f9dc1125` |
| `TRIO/omega_full/from-2024-01-01` | `024544cdf5609ca6282e6def179abde87e4fc43a06b92eed1579971e09616aed` |
| `TRIO/segment_effect/from-2014-06-07` | `d9e7f40ff0f734d8ef16d9ae55b480804bdc129165ddf83a55776a8b91e6f281` |
| `TRIO/segment_effect/from-2024-01-01` | `8b960b6f1cbff30ee748073d34c8462226d1401ae3a23991f685be7ebd41cf5e` |

The sha256 of the sorted run ids of each of v2's SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v2-cells-win` | `a804ef476bccb9e5bbc0533273767c7dbbaa37c78df6fef41ae988287f7df721` |
| `japan-n1-pool-scan-v2-cells-place` | `db8d1479372460781eb67bd97fbb2f67b111a07c7e54838acac9f7f0f9863637` |
| `japan-n1-pool-scan-v2-cells-quinella` | `a36f797b79e00a18ce0a041ae244668ab0d678c9b2adbb230e3a8f1001f4884f` |
| `japan-n1-pool-scan-v2-cells-quinella_place` | `657c5b862421afa8c02e75a68f946c9d7d542ede4529b8c93506832245044c3e` |
| `japan-n1-pool-scan-v2-cells-forecast` | `fe88a951ecc7796a78ecfe2225621048a74d5b8d6687b5bc63f8999da00b0602` |
| `japan-n1-pool-scan-v2-cells-trio` | `6a0034c271dedea5f6ef9f3615b42fff8dc4a5995b1aa0e50d085b5234fa3c84` |

### Superseded: v1, `japan-n1-pool-scan`

v1 (spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`) was frozen at
commit `d5f9ed6` and run on 2026-11-27. Review of PR #496 found the four scoring defects that v2
fixed (the spec's `supersedes`). Its records stay in the registry, superseded: every variant is a record and nothing
is deleted. Its variants count toward variants searched, and its private report stays under
`$HORSETRACK_DATA_DIR/reports/pool-scan-japan-74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9.md`.

- **First full run, at `d5f9ed6`.** It completed scoring, appended its records and wrote both
  private outputs. Its peak resident set was above the 2.5 GB cap, and it then stopped at
  `write_summary`. Its records are identical in variant, params, status and metrics to those of
  v1's run of record.
- **Aborted attempt, at `d0e0182`.** It was stopped during its reads, once its resident set had
  passed the cap, before it appended anything. It holds no record and no output.
- **Memory.** The peak came from the store reader, not from the scan. `read_store_table` served
  its dataset from one buffer over the whole memory-mapped file, and Arrow asks the kernel to page
  in every buffer it maps, so a one-year `final_odds` read paged in the whole file. Commit
  `0278c31` reads the snapshot by byte range instead. Commit `d0e0182` builds the fit's input from
  categoricals, which gives identical draws.
- **Run of record, at `0278c31`.** `/usr/bin/time -l` gave a maximum resident set size of
  1,916,321,792 bytes (about 1.92 GB) and 531 s of wall time.

v1's run of record, one record per pool, reading and span:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `d2b44ec8f81735f515fa49c043d369351e679197139fe31bc7e706ec2886c5d9` |
| `WIN/omega_full/from-2024-01-01` | `e4cc64f16c7e737a5b070a7fffa7a272438362037f49405fde0dba316bf3250b` |
| `WIN/segment_effect/from-2014-06-07` | `93a8a4fafc5432783c0b3ff09ee49a5e250c1cbaf644017fb7bcc6c93fa343c8` |
| `WIN/segment_effect/from-2024-01-01` | `478bac7f4cef9985e401a132cbb56e3c7bcc90372413cffdeb14710713c183cd` |
| `PLACE/omega_full/from-2014-06-07` | `0ab62f7068118874aff4499bba12d657dd77f9c69c8cacb69f8b737a843ff39b` |
| `PLACE/omega_full/from-2024-01-01` | `5a1bd76bdb8e455a235c18eda242560a23008efae381e2ea7a2adf9d2ede05de` |
| `PLACE/segment_effect/from-2014-06-07` | `88b33247b902906da476e176c165ea8bb2337e386b075753e689f8e5340825aa` |
| `PLACE/segment_effect/from-2024-01-01` | `099676f3990a177487ec579141ae20688fc161e6343225e716f80232f18abc2c` |
| `QUINELLA/omega_full/from-2014-06-07` | `0265dafc384b5760b403637c88000771a154c4589f69c2ce0ff04f9ba27e5ac1` |
| `QUINELLA/omega_full/from-2024-01-01` | `753d5a52123529f2d454e9572877a2f2c3b0dd7d55a87cc15b27086438591c60` |
| `QUINELLA/segment_effect/from-2014-06-07` | `d21bcfb8d15e4a9e1d56c96f807a4f99d4e3c78db4c453572147d8c1dc109fcb` |
| `QUINELLA/segment_effect/from-2024-01-01` | `98fc7b614b6dffadd1457307ab0b6532334e8200141d5bac1a62cf5a9b457484` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `df3a998a68e988f0ade8c5a961ad7bb67a7236d9ae3fef7e8f81480eea10f60b` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `44704151a80aede09df0ffebd503c942a74ef815cc60c47466320775b221d3f0` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `6382b97745f890ff36da6b76d3c4d0d3b0e141763585dc98eaf938159f6a9bda` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `2f0619840c8e7b42e1d97fb7cea70b63998757332b6c6367fbf335f281d812ec` |
| `FORECAST/omega_full/from-2014-06-07` | `c3e942088a890a1dec783c2b75ab4c60545f7947d2e310d2f2563197c1467046` |
| `FORECAST/omega_full/from-2024-01-01` | `3336f7a012720a9c16ca96eb2339966247ebe9171aac424673f9dc7e343343f8` |
| `FORECAST/segment_effect/from-2014-06-07` | `dca77c2a4e0f396cd44a9bda3048432e34e82c63a4a6fa2fe7a0241b376288bb` |
| `FORECAST/segment_effect/from-2024-01-01` | `d2ceae5a944e95767027a32fa8a443b616b406de7013522954e96c6f18fc2de2` |
| `TRIO/omega_full/from-2014-06-07` | `c1bb570473d34b05ae18cb91ae5add159f1bf9d79fc1f205790e84d9006d8226` |
| `TRIO/omega_full/from-2024-01-01` | `303daaf3e3c27eff811e7b71dcb58f99e94dadbf514b74dc0efd2b75d70e074b` |
| `TRIO/segment_effect/from-2014-06-07` | `2577be4eec63491dfe403392e0fae8e5d9b728fc2d0bb8cf899f17fc2ce08b8a` |
| `TRIO/segment_effect/from-2024-01-01` | `346f11c06c25bff294c7d748e9d2e759bc9df9585497c0d16c5a21cbecd32c8c` |

The sha256 of the sorted run ids of each of v1's SPA families:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-cells-win` | `74b68e679c4a022893b56af26267416e535b0a0825bbc0065b93dcb7dc231bc1` |
| `japan-n1-pool-scan-cells-place` | `2a85840284892dbed26799729455db69edd08e7d1eae3f375a98496176f0af9a` |
| `japan-n1-pool-scan-cells-quinella` | `d1f4ff2b6b10d84c2aabb0c2658ac8ca6f508c4e7136381a9da83326e2198570` |
| `japan-n1-pool-scan-cells-quinella_place` | `279f519ebff3996753bd189b4a70d38d942dae92ad82e49be6d3b72eb7ac148c` |
| `japan-n1-pool-scan-cells-forecast` | `d6aa30d1026e74e4cfd47cc6b3bb675436b6ddbcb0b2ed0ed6f84cff321ce598` |
| `japan-n1-pool-scan-cells-trio` | `b335198a70380d9166d4885f7575051953052c618c25766a94c196a9b5cb01df` |

v1's first full run, at `d5f9ed6`:

| variant | run id |
| --- | --- |
| `WIN/omega_full/from-2014-06-07` | `940e71ccfef99a15f8abd6aff42b7b80f3338231f30938123412fce67aba2974` |
| `WIN/omega_full/from-2024-01-01` | `7c3c50c9db07c892e82b0eed39d1ebcb72b4fe01f16a6e0d367c1709c352e4c8` |
| `WIN/segment_effect/from-2014-06-07` | `eb3f52abf7bc2f84ea118b861647251330a7bfc7648ac6d5b8ed92ea1a8d5d81` |
| `WIN/segment_effect/from-2024-01-01` | `c742210f41c7cac3427caa950a4e3eed7aa26b84608e2575fd03a822fb85eb53` |
| `PLACE/omega_full/from-2014-06-07` | `c4e254939d7f4e6261ff9e527c08293fe55be1162c20f1adc930410fdd272a3d` |
| `PLACE/omega_full/from-2024-01-01` | `baf195f5e6d72ecf14fda242400523c91f1216e0c27c291b784c6c3796132002` |
| `PLACE/segment_effect/from-2014-06-07` | `1d4ef8aa56d24c4847cdaeb33290d8859f560e107252fc29fa8c3097bd105fe5` |
| `PLACE/segment_effect/from-2024-01-01` | `85208e57e5faefd4d0aa699208389679f27ce40748fc32828b33e3fc3b5d39b2` |
| `QUINELLA/omega_full/from-2014-06-07` | `aad7185ae2de53c2f29a708e7968079493b00802bdc7399dd83ce7cd2f3af77c` |
| `QUINELLA/omega_full/from-2024-01-01` | `4c8579fd223daadd9753c01daeec16de84748cf0bd0e52f97f42ad1bc9bbe631` |
| `QUINELLA/segment_effect/from-2014-06-07` | `4a8312c2e8b790824be9bccb22128e9118d011f4fe583dc12faceb180b319656` |
| `QUINELLA/segment_effect/from-2024-01-01` | `2b5a30e2ad62cfd0eb2e08959b95eb7cefb12d6212bd79d9a12089f2014e5201` |
| `QUINELLA_PLACE/omega_full/from-2014-06-07` | `8086ab28c062f8d82f29a0089435f0226ceb2df9fb0a6978d60d94ca1bfb79ff` |
| `QUINELLA_PLACE/omega_full/from-2024-01-01` | `fb5861bb603eb31744e4ec596efb8a731a71989e03a48c4c6fb3f4b7bcee7ac9` |
| `QUINELLA_PLACE/segment_effect/from-2014-06-07` | `80c1b1e3b35eabcb135477fb32447bb1065a01af42d724b2e116b45edf7f7b31` |
| `QUINELLA_PLACE/segment_effect/from-2024-01-01` | `bdc5c8d1532496eb19a8087fca1cf7a4c6069d7a58e12caecc748c61a16a87df` |
| `FORECAST/omega_full/from-2014-06-07` | `3973fc9ed9f9ee99aa6f0db24fec29403c58f524a0d63b6cecc351928060e349` |
| `FORECAST/omega_full/from-2024-01-01` | `2bba18aab72c7e2709595055b1e1b84800c8e1e1200c84de2206f791ab4317ea` |
| `FORECAST/segment_effect/from-2014-06-07` | `5d83a37b4f70410052993d949ea2e4fe048b86e8ce0df84c5dc0263b6e11acb5` |
| `FORECAST/segment_effect/from-2024-01-01` | `f38e0cb5259ebb5dc6950c8423ea4f702c6c2dc23e928794fdf3eb0e5e1ec04c` |
| `TRIO/omega_full/from-2014-06-07` | `d40d19e2712008705a480eddb8bfc18f694193865311bdca45cd50714f67df38` |
| `TRIO/omega_full/from-2024-01-01` | `57c8b4390fbf15c74f5c7015e6b684419fa6d940283083337011bc537a3c342d` |
| `TRIO/segment_effect/from-2014-06-07` | `433a92733b756972f79f081d07da7086d9fc49c1e368bfed6c17c5b58320f562` |
| `TRIO/segment_effect/from-2024-01-01` | `e95059f7f72a927ca9560a65c10b35ec09f7eadd9e42b052021f8c11c201be50` |
