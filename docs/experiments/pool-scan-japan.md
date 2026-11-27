# Experiment report: N1, the final-board segment scan of six pools (Japan, #430)

Ticket #430 (epic #72). **Every figure is withheld (D21)**: the scan reads the JRA-VAN store, a licensed source, so this report says what ran and where its figures are, and gives no count, rate, span, sign, bound or rank derived from that data. Nothing is adopted, no bet follows from it, and the holdout is not opened.

- Registry family `japan-n1-pool-scan-v3`, spec sha256 `785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c` (`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it before the run read any row.
- Authority: the owner's reframe of 2026-11-27 on #430 removed the sign-off ("Claude pre-registers the cell list and metrics and freezes them in the registry before the run. No owner sign-off.") and the kill rule ("It ranks every cell by expected return per 100 yen at the final price (takeout included, CI beside it). None is closed, and models aim at the top-ranked cells first."). The frozen spec quotes it in full (`authority`).
- It supersedes v2, `japan-n1-pool-scan-v2` (spec sha256 `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6`), and v1, `japan-n1-pool-scan` (spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`), both run the same day. The second review of PR #496 found that v2 scored a race whose dividend named a combination off the priced board as if every ticket lost, read a Place or Wide band that paid nothing as a known zero, reported variants searched as the family size, decoded price row groups that span a holdout day, crossed class with three dimensions only, and gave Place and Wide no #78 fit. v1's defects are in the spec (`supersedes`). Their records stay in the registry, superseded; nothing is deleted.
- Requested range: development days from 2014-06-07 to 2026-11-27 (the run's arguments, not the store's measured span); holdout days are never read and `LockedHoldout` is never opened.
- The price rows (D14, #502): from the footer statistics of `final_odds`, a row group whose race_date range spans a holdout day is never decoded. Prices are read only on the day windows no such row group touches, each through `read_store_table`, whose scan prunes by the same statistics every row group outside the window and drops holdout days. The jravan builders write one row group per year, so until #502 isolates row groups by day a year whose row group spans a holdout day is skipped whole, its development days unscored; the private report counts the row groups skipped. `races` (no price or result) is read on the same windows and `dividends` filtered in its scan to the windows' development race ids: their row groups are filtered, not skipped, so one spanning a holdout day would still be decoded, its holdout rows dropped inside the scan, until #502 isolates them too; the private report counts them.
- Seed 20261127; code at `826c66526b1e`; command `uv run horsetrack japan pool-scan run --from 2014-06-07 --to 2026-11-27 --seed 20261127`.
- Family size 32 (this family's distinct variants in the registry); variants searched 80 (that plus v1's and v2's, whose variants ran on the same days). Both are on every row.
- Private report: `$HORSETRACK_DATA_DIR/reports/pool-scan-japan-785a45700efca6286c8cd85c4f5b0cd17409b4f5cbcde1abb27368a6f19a356c.md`; candidate cells: `$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json`; both written through `write_report` with the source `jravan`.

## Registry records

One record per pool, shrinkage, reading and span in `japan-n1-pool-scan-v3`, in spec order:

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

The cell rules SPA and StepM test are one record per pre-registered cell in each pool's family; their run ids are in the private report, and the sha256 of each family's sorted run ids is:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v3-cells-win` | `7df0047431d064574de0d982df31ff75f48582cb019afc3310e91f0085b85b66` |
| `japan-n1-pool-scan-v3-cells-place` | `6120ddc8fb2e40e1246b9141b67d490bda5234f4005d2688e000566903081c83` |
| `japan-n1-pool-scan-v3-cells-quinella` | `3b79a848141517f318db9355fb1f0dc4a4902b616c24930743122a769b592cd5` |
| `japan-n1-pool-scan-v3-cells-quinella_place` | `4bb173ad774b1f25b0911ff3b373eebe40f89db9f46e30f889df150cc4d69c66` |
| `japan-n1-pool-scan-v3-cells-forecast` | `92a54b59d1dbdc1895506a3ccedb4b06087adba9a542f784d450b9b6d282210b` |
| `japan-n1-pool-scan-v3-cells-trio` | `3d22a34bbdb43de495c505466716983a9f326e7688b5c04cdfdadf2d4b284e47` |

## Hypothesis

E8, as frozen: in some pre-registered cell of a JRA pool's final board the hierarchical omega differs from 1 once the favourite-longshot bias is taken out as odds-band fixed effects, and some cell's expected return per unit at the final price reaches break-even (omega at or above 1 / payout rate). The null, the starting point: every cell is priced like its pool.

## Variants tried

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO): two omega readings (`omega_full`, the band effects included; `segment_effect`, the segment effect alone) on two spans (from 2014-06-07; from 2024-01-01, reported apart). The cells are `all`, each odds band, each level of class, field size, favourite strength, pool size, surface and venue, and class crossed with each of the other six. The shrinkage is #78's hierarchical fit for the one-winner pools, its omega weighted by payout rate. For Place and Wide it is an empirical-Bayes shrink of the realised return over odds-band baselines that carry their posterior (a band that paid nothing keeps a Jeffreys posterior, not a known zero), which ranks them; beside it, as variants of their own (`roi78`), #78's fit on a representative price: each combination's low-end odds times its band's realised return per unit of low-end odds when it pays, the outcome being whether it paid. Every cell of every pool is ranked by its shrunk expected return (the owner's reframe: none is closed); the ranks, the omegas and their bounds are withheld (D21). Which variants failed, and their figures, are withheld (D21): the private report gives each reason, and the registry summary shows no status of a licensed-source run. v1's and v2's variants, superseded, are records of `japan-n1-pool-scan` and `japan-n1-pool-scan-v2` and count toward variants searched.

## Alternatives rejected and why

- The full joint cross (odds band x class x field x favourite x pool size x surface x venue, 33600 cells per pool): most cells would hold a handful of races or none, so none could show an effect at any useful power, and SPA over that many rules would test nothing. The cells cross class, the owner's axis (maidens and debuts against the big graded races), with each other dimension: 150 of each pool's 186 cells.
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

Everything above this section is `render_public_report`'s output for the run of record: v3, run
at commit `826c665`. This section records the trail.

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
  in the registry (below). Its private outputs are kept beside the run of record's, under
  `$HORSETRACK_DATA_DIR/reports/` with the suffix `-first-run-2415969`.
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
- **v3 run of record, at `826c665`** (the tables above). `/usr/bin/time -l` gave a maximum
  resident set size of 1,822,457,856 bytes (about 1.82 GB) and 1,288 s of wall time on this 16 GB
  Mac, with no other heavy job beside it. One year of the store, one pool's board at a time, was
  in memory, and one fit ran at a time.
- **The `local` test.**
  `tests/countries/japan/research/test_pool_scan_run.py::test_real_store_one_month_reads_development_days_only`
  scans the store's last month into a temporary registry and a temporary report directory. Its
  outputs were not read and informed nothing in either spec.
- **Registry summary.** Main's `write_summary` (#525) now maps the japan registry's legacy `/`
  family ids, so each v3 run regenerated the summary of every family in the registry
  (`summary_error` is null). As in round 1, the committed summary under
  `docs/experiments/registry/japan/` holds only the families it already held plus v3's seven.
  `write_summary` was run on a read-only view of the japan registry that holds only those
  families' records, into a temporary directory; v3's seven files and `README.md` were taken from
  it, and every other file there is as on main. The families of other open tickets wait for their
  own PRs. v3's files show only each record's E id and variant, its status and values withheld
  (licensed source), for every v3 run.

### Superseded: v3's second run, at `4a6cd80` (over the memory cap)

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

### Superseded: v3's first run, at `2415969`

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
that v3 fixes (above). Its records stay in the registry, superseded: every variant is a record and
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
