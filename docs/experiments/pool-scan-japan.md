# Experiment report: N1, the final-board segment scan of six pools (Japan, #430)

Ticket #430 (epic #72). **Every figure is withheld (D21)**: the scan reads the JRA-VAN store, a licensed source, so this report says what ran and where its figures are, and gives no count, rate, span, sign, bound or rank derived from that data. Nothing is adopted, no bet follows from it, and the holdout is not opened.

- Registry family `japan-n1-pool-scan-v2`, spec sha256 `ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6` (`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it before the run read any row.
- Authority: the owner's reframe of 2026-11-27 on #430 removed the sign-off ("Claude pre-registers the cell list and metrics and freezes them in the registry before the run. No owner sign-off.") and the kill rule ("It ranks every cell by expected return per 100 yen at the final price (takeout included, CI beside it). None is closed, and models aim at the top-ranked cells first."). The frozen spec quotes it in full (`authority`).
- It supersedes `japan-n1-pool-scan` (spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`), run the same day: review found that v1 scored a sold board with no dividend row as if every ticket lost, took a cell's expected return as mean omega times mean payout rate where boosted and ordinary rates mix, checked SPA's columns against every cell an earlier run scored, and gave no Place or Wide reading for a cell whose odds band returned nothing. v1's records stay in the registry, superseded; nothing is deleted.
- Requested range: development days from 2014-06-07 to 2026-11-27 (the run's arguments, not the store's measured span); holdout days are never read and `LockedHoldout` is never opened.
- The store reader's guarantee: `read_store_table` pushes the holdout exclusion into the Parquet scan as its filter, so no holdout row is materialized, and `dividends` is filtered inside its scan to the development race ids. Parquet prunes row groups by their min/max statistics only and the jravan builders write one row group per year, so a year whose row group holds holdout days is decoded from disk and its holdout rows are dropped inside the scan. Row groups are not yet isolated by day (proposed #502).
- Seed 20261127; code at `0f7ed7431ecf`; command `uv run horsetrack japan pool-scan run --from 2014-06-07 --to 2026-11-27 --seed 20261127`.
- Variants searched (on every row; this family's size plus v1's, whose variants ran on the same days): 48.
- Private report: `$HORSETRACK_DATA_DIR/reports/pool-scan-japan-ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6.md`; candidate cells: `$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json`; both written through `write_report` with the source `jravan`.

## Registry records

One record per pool, reading and span in `japan-n1-pool-scan-v2`, in spec order:

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

The cell rules SPA and StepM test are one record per pre-registered cell in each pool's family; their run ids are in the private report, and the sha256 of each family's sorted run ids is:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-v2-cells-win` | `a804ef476bccb9e5bbc0533273767c7dbbaa37c78df6fef41ae988287f7df721` |
| `japan-n1-pool-scan-v2-cells-place` | `db8d1479372460781eb67bd97fbb2f67b111a07c7e54838acac9f7f0f9863637` |
| `japan-n1-pool-scan-v2-cells-quinella` | `a36f797b79e00a18ce0a041ae244668ab0d678c9b2adbb230e3a8f1001f4884f` |
| `japan-n1-pool-scan-v2-cells-quinella_place` | `657c5b862421afa8c02e75a68f946c9d7d542ede4529b8c93506832245044c3e` |
| `japan-n1-pool-scan-v2-cells-forecast` | `fe88a951ecc7796a78ecfe2225621048a74d5b8d6687b5bc63f8999da00b0602` |
| `japan-n1-pool-scan-v2-cells-trio` | `6a0034c271dedea5f6ef9f3615b42fff8dc4a5995b1aa0e50d085b5234fa3c84` |

## Hypothesis

E8, as frozen: in some pre-registered cell of a JRA pool's final board the hierarchical omega differs from 1 once the favourite-longshot bias is taken out as odds-band fixed effects, and some cell's expected return per unit at the final price reaches break-even (omega at or above 1 / payout rate). The null, the starting point: every cell is priced like its pool.

## Variants tried

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO): two omega readings (`omega_full`, the band effects included; `segment_effect`, the segment effect alone) on two spans (from 2014-06-07; from 2024-01-01, reported apart). The shrinkage is #78's hierarchical fit for the one-winner pools, its omega weighted by payout rate, and a normal empirical-Bayes shrink of the realised return for Place and Wide. Every cell of every pool is ranked by its shrunk expected return (the owner's reframe: none is closed); the ranks, the omegas and their bounds are withheld (D21). Which variants failed, and their figures, are withheld (D21): the private report gives each reason, and the registry summary shows no status of a licensed-source run. v1's variants, superseded, are records of `japan-n1-pool-scan` and count toward variants searched.

## Alternatives rejected and why

- The full joint cross (odds band x class x field x favourite x pool size x surface x venue, 33600 cells per pool): most cells would hold a handful of races or none, so none could show an effect at any useful power, and SPA over that many rules would test nothing. The outline registers task 1's marginal cells plus stated crosses: class crossed with pool size, with favourite strength and with odds band, the owner's axes (maidens and debuts against the big graded races, around the favourite, by price), 70 of each pool's 106 cells.
- #78's fit on Place and Wide at the low end of the range: a Place or Wide dividend is not its odds, so its omega would describe the placing probability, not the return. The ticket left the shrinkage for these pools to the spec; Claude chose the empirical-Bayes shrink of the realised return.
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

Everything above this section is `render_public_report`'s output for the run of record, v2. This
section records the trail.

- **v2 freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-27 at commit `0f7ed74`,
  before any scored read of `japan-n1-pool-scan-v2`. It wrote the family's only freeze line in the
  japan registry.
- **v2 run of record, at `0f7ed74`** (the tables above). `/usr/bin/time -l` gave a maximum
  resident set size of 1,643,855,872 bytes (about 1.64 GB) and 511 s of wall time on this 16 GB
  Mac. One year of the store was in memory at a time, one fit ran at a time, and no other heavy
  job ran beside it.
- **The `local` test.**
  `tests/countries/japan/research/test_pool_scan_run.py::test_real_store_one_month_reads_development_days_only`
  scans the store's last month into a temporary registry and a temporary report directory. Its
  outputs were not read and informed nothing in either spec.
- **Registry summary.** From commit `2d637c6`, `write_summary` shows only the E id and variant of
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

### Superseded: v1, `japan-n1-pool-scan`

v1 (spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9`) was frozen at
commit `d5f9ed6` and run on 2026-11-27. Review of PR #496 found the four scoring defects that v2
fixes (above). Its records stay in the registry, superseded: every variant is a record and nothing
is deleted. Its variants count toward v2's variants searched, and its private report stays under
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
