# Experiment report: N1, the final-board segment scan of six pools (Japan, #430)

Ticket #430 (epic #72). **Every figure is withheld (D21)**: the scan reads the JRA-VAN store, a licensed source, so this report says what ran and where its figures are, and gives no count, rate, span, sign, bound or rank derived from that data. Nothing is adopted, no bet follows from it, and the holdout is not opened.

- Registry family `japan-n1-pool-scan`, spec sha256 `74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9` (`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it under the owner's reframe of 2026-11-27 before the run read any row.
- Requested range: development days from 2014-06-07 to 2026-11-27 (the run's arguments, not the store's measured span); holdout days are never read and `LockedHoldout` is never opened.
- Seed 20261127; code at `0278c3145a0c`; command `uv run horsetrack japan pool-scan run`.
- Variants searched (the family size on every row): 24.
- Private report: `$HORSETRACK_DATA_DIR/reports/pool-scan-japan-74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9.md`; candidate cells: `$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json`; both written through `write_report` with the source `jravan`.

## Registry records

One record per pool, reading and span in `japan-n1-pool-scan`, in spec order:

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

The cell rules SPA and StepM test are one record per pre-registered cell in each pool's family; their run ids are in the private report, and the sha256 of each family's sorted run ids is:

| SPA family | run ids sha256 |
| --- | --- |
| `japan-n1-pool-scan-cells-win` | `74b68e679c4a022893b56af26267416e535b0a0825bbc0065b93dcb7dc231bc1` |
| `japan-n1-pool-scan-cells-place` | `2a85840284892dbed26799729455db69edd08e7d1eae3f375a98496176f0af9a` |
| `japan-n1-pool-scan-cells-quinella` | `d1f4ff2b6b10d84c2aabb0c2658ac8ca6f508c4e7136381a9da83326e2198570` |
| `japan-n1-pool-scan-cells-quinella_place` | `279f519ebff3996753bd189b4a70d38d942dae92ad82e49be6d3b72eb7ac148c` |
| `japan-n1-pool-scan-cells-forecast` | `d6aa30d1026e74e4cfd47cc6b3bb675436b6ddbcb0b2ed0ed6f84cff321ce598` |
| `japan-n1-pool-scan-cells-trio` | `b335198a70380d9166d4885f7575051953052c618c25766a94c196a9b5cb01df` |

## Hypothesis

E8, as frozen: in some pre-registered cell of a JRA pool's final board the hierarchical omega differs from 1 once the favourite-longshot bias is taken out as odds-band fixed effects, and some cell's expected return per unit at the final price reaches break-even (omega at or above 1 / payout rate). The null, the starting point: every cell is priced like its pool.

## Variants tried

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO): two omega readings (`omega_full`, the band effects included; `segment_effect`, the segment effect alone) on two spans (from 2014-06-07; from 2024-01-01, reported apart). The shrinkage is #78's hierarchical fit for the one-winner pools and a normal empirical-Bayes shrink of the realised return for Place and Wide. Every cell of every pool is ranked by its shrunk expected return; the ranks, the omegas and their bounds are withheld (D21). Which variants failed is in the registry summary; their figures are withheld (D21).

## Alternatives rejected and why

- #78's fit on Place and Wide at the low end of the range: a Place or Wide dividend is not its odds, so its omega would describe the placing probability, not the return.
- A kill rule (close a pool with no open cell): replaced by the owner's reframe; every cell is ranked and none is closed.
- Exact odds in the fit for every price: a Trio board's long tail makes the grouped rows too many for the 16 GB limit; from odds 10 a 1.1 log grid with the harmonic-mean price keeps the mean implied probability exact.
- Trifecta and bracket quinella: a follow-up that runs after N1 (owner, 2026-11-27).

## What this rules out

withheld (D21): the private report gives, per pool, which cells' one-sided bounds cover break-even, the SPA p and the StepM survivors. Whatever it shows, an N1 cell is a calibration target at the final price (bands, favourite and pool size are read from the final board, look-ahead for selection, #88), not a betting rule.

## Follow-ups

- Trifecta and bracket quinella on the same scan (owner, 2026-11-27).
- The top-ranked cells as decision-time rules (Family P) and in the power table (N6).

## Runs, memory and the registry summary

Everything above this section is `render_public_report`'s output for the run of record. This
section records the trail.

- **Freeze.** `uv run horsetrack japan pool-scan freeze` ran on 2026-11-27 at commit `d5f9ed6`,
  before any scored read of the family. It wrote the family's only freeze line in the japan
  registry.
- **First full run, at `d5f9ed6`.** It completed scoring, appended its records and wrote both
  private outputs. Its peak resident set was above the 2.5 GB cap, and it then stopped at
  `write_summary` (below). Its records stay in the registry (append-only), and they are identical
  in variant, params, status and metrics to the run of record's, so the family size is unchanged.
  Its run ids:

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

- **Memory.** The peak came from the store reader, not from the scan. `read_store_table` served
  its dataset from one buffer over the whole memory-mapped file, and Arrow asks the kernel to page
  in every buffer it maps, so a one-year `final_odds` read paged in the whole file. Commit
  `0278c31` reads the snapshot by byte range instead. Commit `d0e0182` builds the fit's input from
  categoricals, which gives identical draws.
- **Aborted attempt, at `d0e0182`.** It was stopped during its reads, once its resident set had
  passed the cap, before it appended anything. It holds no record and no output.
- **Run of record, at `0278c31`** (the table above). `/usr/bin/time -l` gave a maximum resident
  set size of 1,916,321,792 bytes (about 1.92 GB) and 531 s of wall time on this 16 GB Mac, with
  one year of the store in memory at a time and one fit at a time.
- **The `local` test.** `tests/countries/japan/research/test_pool_scan_run.py::test_real_store_one_month_reads_development_days_only`
  scans the store's last month into a temporary registry and a temporary report directory. It
  ran during the test runs before the freeze. Its outputs were not read and informed nothing in
  the spec.
- **Registry summary.** Main's `write_summary` refuses the japan registry: it holds #131's and
  #282's families, whose ids contain `/` (the fix is in PR #280). The run therefore recorded
  `summary_error` and went on. The committed summary under `docs/experiments/registry/japan/` was
  then regenerated by `write_summary` over the families it already holds plus the seven N1
  families. Every existing file came out byte-identical, the seven N1 files are new, and
  `README.md` gains their seven lines. The families of other open tickets, `/` ids included, wait
  for their own PRs and for PR #280.
