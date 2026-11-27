# Experiment report: N8, the popularity-rank spread scan (Japan, #503)

Ticket #503 (epic #72). **Every figure is withheld (D21)**: the scan reads the JRA-VAN store, a
licensed source, so this report says what ran and where its figures are, and gives no count, rate,
span, sign, bound or rank derived from that data. Nothing is adopted, no bet follows from it, and
the holdout is not opened.

- Registry families (one per pool), each frozen with `Registry.freeze_family` on 2026-11-27 before
  the run read any row (`uv run horsetrack japan spread-scan freeze`; the spec is quoted in
  `docs/specs/spread-scan-prereg.md`):

  | family | spec sha256 |
  | --- | --- |
  | `japan-n8-spread-scan-exacta` | `a380b3aaca6d675cbe9393b780970b8b72a81b0057143a59ffcdfc099b38499b` |
  | `japan-n8-spread-scan-quinella` | `f30589a50bf50545eced734496c8fd87e9113ebbeaa280c65db23162421d65aa` |
  | `japan-n8-spread-scan-wide` | `e33234f83cf4a7ff6fb43f299e96594072ce2b100e18ba527d634cd5001fab52` |
  | `japan-n8-spread-scan-trio` | `fc1bb54bac4fc1752ce7f3f793ac2992b326dc827efa22d52898dc806ecd5611` |
  | `japan-n8-spread-scan-trifecta` | `27bc2561847e5437136002010745f343a27dd4d42c1a6de00655f2d2b26126f8` |
  | `japan-n8-spread-scan-bracket` | `5b509a0e1d97d78c03e32bb9ab685b748bce98b342813d3475c55eb851c1597c` |

  The pre-registration's hash (all six specs together, `spread_scan.spec_hash()`):
  `6f041ec1399564dc0505f7408f205c7d34b6146a13c9b5442e97f8009b08be00`.
- Requested range: development days from 2014-06-07 to 2026-11-27 (the run's arguments, not the
  store's measured span); holdout days are never read and `LockedHoldout` is never opened.
- Seed 20261127, 1,000 meeting-bootstrap replicates; run of record at code `65fcbfc` (PR #524's
  review fixes); command
  `uv run horsetrack japan spread-scan run --from 2014-06-07 --to 2026-11-27`. The first run, at
  `c59a484`, is superseded (see "Runs, memory and the registry summary").
- Variants searched: 27 spreads, 2 variants per family (`t2`, `lookahead`), 12 registry records
  per run; by design at most 78 cell labels per spread, so at most 2,106 ranked rows per variant.
- Private report of the run of record:
  `$HORSETRACK_DATA_DIR/reports/spread-scan-japan-9291d2e5b95029e4.md` (named per run by a digest
  of the spec hash, range, seed, replicates, commit and store fingerprint, and named in each of
  that run's records as `params.report`; later code names a run's report in `data_builds.report`,
  see "Family size" below); candidate cells:
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/spread-scan.json`; both written through
  `write_report` with the source `jravan`.

## Registry records

Run of record, at `65fcbfc`: one record per (family, variant), in spec order:

| family / variant | run id |
| --- | --- |
| `japan-n8-spread-scan-exacta/t2` | `5f8dc63395d20a80f1ff38a8c8c64575cceeac5dd010fdc15c274ec2283eca53` |
| `japan-n8-spread-scan-exacta/lookahead` | `0318f87edfaee94d956d44025dc348fd50357aab6c67c3c6874ea36ca8394641` |
| `japan-n8-spread-scan-quinella/t2` | `6f29fe50a6f6a6bdb8845f7b97d853a8a8b5fe59b04a788046a45d94e5bea7a7` |
| `japan-n8-spread-scan-quinella/lookahead` | `7974e5971b4548577ae24c3178d5e287165e4352ef09ea81a46db544df20d6e7` |
| `japan-n8-spread-scan-wide/t2` | `bb475e11aa113307aae6a4c1fcb5e8a652a9dea221af3898733b8ab06384c2e5` |
| `japan-n8-spread-scan-wide/lookahead` | `ce533a5081455e6166c55a3e3a155f943b85275590976bc47514ce52cf6e544d` |
| `japan-n8-spread-scan-trio/t2` | `7189e522dbff80db95f0dd1ac0f0272b6cebac5d7a8ead1446624a6255c170ec` |
| `japan-n8-spread-scan-trio/lookahead` | `beba22bee7e9504b4acc3b7d0f3c682d4282a6bc4bd823de4dbeeb97830f0687` |
| `japan-n8-spread-scan-trifecta/t2` | `46505b8b645c063d516abde8ccd5eb96f03fc5a3680cc1396afcd4162633f38c` |
| `japan-n8-spread-scan-trifecta/lookahead` | `27397d795c50404bf569219ae566778e99248dbd9b19c297cbc08281af780976` |
| `japan-n8-spread-scan-bracket/t2` | `b00553e4f34e4ebf15067fb36b63a44799391198fd7550f6b41b519227df36a9` |
| `japan-n8-spread-scan-bracket/lookahead` | `331fe612b91a219355a8b6264089ffd8d052e3b1629f179a91c23c39d189b8ad` |

The first run, at `c59a484`, superseded by the review fixes; its records stay in the registry
(append-only):

| family / variant | run id |
| --- | --- |
| `japan-n8-spread-scan-exacta/t2` | `766f5030441eaa8d4c9190be9e2f7615a1e530a3bfd9ce23cb442387d558ad41` |
| `japan-n8-spread-scan-exacta/lookahead` | `6118b378944d2db49b3d2ead4e2069d673527b73eacaacbee3f2180299ac157c` |
| `japan-n8-spread-scan-quinella/t2` | `e8055f6d4f996b9549f40dda6c3cdd1d30e462ab2b40b35eea0788c8a386bdd6` |
| `japan-n8-spread-scan-quinella/lookahead` | `1d3f2d947b5af39d2b24705000794375b9a34be7abac7338faa7aa7a59d24d17` |
| `japan-n8-spread-scan-wide/t2` | `69e4b1507b577c014b73336376aae04a368691aabaf44fb01e951c61f1abbe5d` |
| `japan-n8-spread-scan-wide/lookahead` | `48ab792ed1a67654bbbeac0cc5f92594ab03a6a1dce2b71ac1018f32050db903` |
| `japan-n8-spread-scan-trio/t2` | `36d8a8919ffb3be5ca6d5602d342eeac3e69c619ff4316c706bb1c1e3eb1d2eb` |
| `japan-n8-spread-scan-trio/lookahead` | `cc5b2954edc9a7a6636454ecc4d77c6286e052a7447757b8fb2559786828a0e1` |
| `japan-n8-spread-scan-trifecta/t2` | `d072a53c97f3fee21c6026071d8c1742d43fefc16ba60dc8869cc3179ad93a88` |
| `japan-n8-spread-scan-trifecta/lookahead` | `317b1b5d990866dbaa327b08374b580cd9865e0c808d7b4378be7a84788ecde3` |
| `japan-n8-spread-scan-bracket/t2` | `7cdc83dabb6e47e11d2987baf69193f1e3dd26816139e1f6bf51bd9463f49659` |
| `japan-n8-spread-scan-bracket/lookahead` | `4a6f7799428bb6b4e4218b331c10906ac286d43207a1ded2b13f65c62448caa9` |

## Hypothesis

P503, as frozen: some pre-registered spread, bought on the T-2 Win popularity ranks, returns more
than its stake at the paid dividends in some cell of race class, field size, favourite strength,
surface, pool-size tertile, class x favourite or class x field (shrunk edge above 0). The null, the
starting point: no spread beats the takeout in any cell. The look-ahead variant ranks on the final
Win board and is a ceiling, not a strategy.

## Variants tried

Per family and variant one record: the exacta family (eight spreads: the favourite over ranks
2..k, ranks 2..k over the favourite, boxes 1-3 and 1-4), quinella and Wide (the favourite with
ranks 2..k, box 1-4, box 2-5), trio (the favourite as axis with 2-5 and 2-6, boxes 1-4 and 1-5,
formation 1-2 / 1-4 / 2-6), trifecta (1 / 2-4 / 2-4, 1 / 2-4 / 2-7, 2-4 / 1 / 2-7) and bracket
quinella (the favourite's bracket with those of ranks 2 and 3), each on the T-2 ranks (`t2`) and
on the final ranks (`lookahead`). Every cell with a bet is a ranked row with its rho, its
one-sided meeting-bootstrap bounds, its shrunk edge and P(return > 100 yen); the ranks, the rhos,
their bounds and the shrunk edges are withheld (D21). No variant failed in either run (the
registry summary lists every run's status).

## Alternatives rejected and why

- #78's hierarchical fit (`core.eval.segment_roi`): its likelihood needs one combination's odds per
  bet, and a spread is several tickets priced at the paid dividends, not at a board price; the
  scan uses N1's normal empirical-Bayes shrink of the realised return (its Place and Wide method),
  with a pooled-dispersion floor on each cell's se so a cell of one or two meetings is not taken
  as exact.
- Ranks from each exotic pool's own board at T-2: the jravan time series holds Win, Place, bracket
  quinella and quinella boards only, so every spread ranks on the Win board.
- Buying the ranks that exist in a field smaller than a spread's highest rank: it would change the
  spread's definition race to race; those races are skipped for that spread and counted.
- A pass/fail gate or kill rule: replaced by the owner's reframe (2026-11-27); every cell is a row.
- SPA and StepM over the spreads: not in this ticket's outline; listed as a follow-up.

## What this rules out

withheld (D21): the private report gives, per variant, each spread's return on all races and the
top-ranked cells with their bounds. Whatever it shows, a `t2` cell is a candidate rule for the
leaderboard (#447) and a `lookahead` cell is a ceiling only; nothing here is adopted.

## Follow-ups

- The top-ranked `t2` cells as decision-time rules on later development days and in the holdout
  ranking check (owner reframe, 2026-11-27), through #447's leaderboard and #451's backfill.
- SPA and StepM over the spread cells, as N1 runs per pool.
- Spreads over other rank sources (model probabilities, #139's optimiser) once they exist.

## Runs, memory and the registry summary

- **Freeze.** `uv run horsetrack japan spread-scan freeze` ran on 2026-11-27 at commit
  `c59a484`, before any scored read of the six families. It wrote each family's only freeze line in
  the japan registry. The review fixes change no family's spec (the hashes above are unchanged),
  so that freeze stands for both runs.
- **First run, at `c59a484`, superseded.** Its 12 records (the second table above) and its
  private report, `$HORSETRACK_DATA_DIR/reports/spread-scan-japan-6f041ec1399564dc0505f7408f205c7d34b6146a13c9b5442e97f8009b08be00.md`,
  stay. PR #524's review found that its code could let the two variants bet different races
  where the two boards rank fields of different lengths, and cut the pool-size tertiles over races
  holding no bet, both against the frozen spec; the fixed code follows the spec, and the scan was
  run again on it.
- **Run of record, at `65fcbfc`** (the first table above), same range, seed and replicates.
- **Memory and wall time.** Both runs ran under `/usr/bin/time -l` on this 16 GB Mac; their wall
  time and peak memory are withheld (D21), as `docs/specs/jravan.md` withholds the store's, in the
  private run note `$HORSETRACK_DATA_DIR/reports/spread-scan-japan-runs.md`. The scan holds one
  calendar year of the store in memory at a time and reads named columns only; `final_odds` is
  read for its WIN rows only, by byte range (`read_development` on an `OSFile`), not through the
  memory-mapped reader, and each development day's `odds_timeseries` file for its WIN rows.
- **Registry summary.** Main's `write_summary` refuses the full japan registry (#131's and #282's
  family ids contain `/`, PR #280). The first run's command regenerated the committed summary
  under `docs/experiments/registry/japan/` with `write_summary` over the families it already held
  plus the six N8 families (every existing file byte-identical, six new N8 files, six new
  `README.md` lines). Since the review fixes the command writes the six N8 files from their own
  records only and merges their lines into `README.md`, keeping every family it lists
  (`studies/ability.py`'s way); the run of record adds its 12 rows to the six N8 files and leaves
  every other file, `README.md` included, as it was. Since PR #524's second review the command
  also never drops a published row: when the local registry would not write again every row an
  existing N8 file lists (a fresh data directory), it keeps that file as it was, names it and
  exits 1, the run itself recorded.
- **Family size.** `Registry.family_size` reads 4 for each N8 family, for 2 distinct rules (`t2`,
  `lookahead`). It compares every `params` key, and the run of record's records at `65fcbfc`
  carry the report's file name in `params.report`, which the first run's records at `c59a484` do
  not, so the same two rules count twice; with that key set aside, the two runs' records are the
  same 2 variants per family (checked on the local registry). The registry is append-only, so
  those records stay as written. PR #524's second review moved the name to `data_builds.report`,
  after its `@`, which `family_size` reads as a refresh rather than an identity: from that code
  on, a rerun of the same spec, range, seed and replicates at another commit or on a refreshed
  store adds runs, not variants (a first run at that code still counts once against the earlier
  records, whose builds hold no `report` entry). That change touches provenance only, no scored
  output, so the scan was not run again.
