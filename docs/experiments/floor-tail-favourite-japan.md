# Experiment report: floor, tail and favourite avoidance rules L2, L3 and L7 on the same races (Japan, #82)

Ticket #82 (epic #72; 08 §8 A4, price part; §5.1 L2, L3, L7; §4 E3, E7, E12, E14, R3; 04 §1.3). Family L, #81's
paired design: each rule is compared with its stated comparator on the same races, by each side's mean return per
unit staked (payout / stake) at final dividends and the paired per-race difference between them, each with
meeting-clustered one-sided bounds. Under the owner's reframe of 2026-11-27, every option is a ranked row: the rule's
tickets and its comparator's are both rows, an avoid rule drops nothing from the data, and there is no pass/fail.
**Nothing here is adopted.** Every figure below is an aggregate over the JRA results store (netkeiba race results,
final Win odds and dividends), our own logged boards and the shipped boost calendar. The 100-vs-110 prediction, its
reconstruction diagnostic and its counts of predictions read the JRA-VAN H1 votes; their figures and their result are
withheld (D21) and are in a private report only. There is no race or runner row here.

- Registry family `japan-family-l-floor-tail-favourite-v5`, spec sha256
  `dd4d0926b6bc8d557739653a8612eb0d648085a57f4b733fa6b74ccfd4c32183`. `run_study` froze the spec before it read any
  row; the freeze line is the record of the owner's inputs (there is no sign-off key). The spec holds the rules, the
  tail odds and ranks, the entropy cut points, the comparators, the selections, the decision time (T-2) and its
  post-time source, who counts as a runner at t (a scratch at t included) and how a later scratch is settled, L2's
  no-bet race, the board fields, the bootstrap, the Place-votes reading, the Place rule (a premium PLACE pool has no
  prediction), the reconstruction diagnostic and its tolerance, the counts of predictions, the walk-forward scheme
  (`config/folds.yaml`'s, under `evaluation`), and the 98 variant names (`family_size` 98: 50 descriptive records
  and a walk-forward record for each of the 48 paired comparisons). Its `supersedes` entry names the four earlier
  families of this ticket and why each was replaced (below).
- Code at `64ccd07` (`countries/japan/studies/avoidance_price.py`, built on #81's `studies/avoidance.py`). Owner
  inputs are in `countries/japan/config/avoidance_price.yaml`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.avoidance_price --run`. The family is frozen, so a rerun with
  this spec adds runs of the same variants. Another spec is refused (`FamilyAlreadyFrozen`).
- Store: `processed/jra/races.parquet@9cce511831074b64`, `processed/jra/runners.parquet@6f3cec95347e0132`,
  `processed/jra/dividends.parquet@b3d635352bb007df`, `processed/jra/odds_timeseries (2 days)@7bd8e900bd0a1e6e`
  (the same builds as #81's run). Post times: every one of the 24 logged races took t from
  `board.snapshot.post_times_for`, which found no `<VENUE>.races.json` sidecar for either day and read HKTrack's
  stored card (`processed/jra/upcoming/jra_<day>_KYO/races.parquet`); no logged race lacks a post time. The at-t
  records carry that input as one build, `jra_post_times` = `board.snapshot.post_times_for (the <VENUE>.races.json
  sidecar, else the card)@af0475f243f53b16`, the part after `@` a digest of the race -> scheduled post mapping; the
  coverage counts are in the full report. The JRA-VAN build read by the 100-vs-110 check is named in the private
  report.
- The full report is under the data directory, at `reports/floor-tail-favourite-japan-dd4d0926b6bc.md`, with
  `-dropped.csv` beside it. The 100-vs-110 check is in `reports/floor-tail-favourite-japan-dd4d0926b6bc-plus10-jravan.md`
  and its `.csv` (private, JRA-VAN-derived). Every record is in
  `docs/experiments/registry/japan/japan-family-l-floor-tail-favourite-v5.md`, each walk-forward record with its
  per-window differences (28 windows for each of the 23 final-price comparisons, one window, 2026-10-01..2026-12-31,
  for each of the 25 board-at-t comparisons); the two 100-vs-110 records show neither their status nor any value there (licensed
  source). The command's printed summary names them only as withheld.
- One run, 2026-11-27: 191 s, 1.87 GB peak resident memory (v4: 156 s, 1.09 GB). Most of the rise is the votes read,
  which now filters inside pyarrow's dataset scan so that no holdout row reaches Python (C below); that scan decodes
  each selected row group of `pool_votes` at once.

## Runs of this ticket

Every run is in the registry; the four earlier families keep their records and are superseded, not deleted. Their
final-price rows are the same as v4's to the last digit (the changes below touch only the board at t and the
100-vs-110 check), v3's board-at-t rows are the same as v4's, and each of v5's 48 descriptive comparison records
equals v4's (status, every metric, n and bound).

| Family | Spec sha256 | Git | Records | Status |
| --- | --- | --- | ---: | --- |
| `japan-family-l-floor-tail-favourite-v5` | `dd4d0926b6bc...` | `64ccd07` | 98 | Current. The 48 descriptive paired comparisons and their 48 walk-forward records (23 + 23 at the final price, 25 + 25 at t) are scored; the two 100-vs-110 records are withheld (D21). |
| `japan-family-l-floor-tail-favourite-v4` | `1e8f6cc0d7d7...` | `acc0e2b` | 50 | Superseded by fix round 4 (below): a premium PLACE pool was rebuilt with its premium taken as paid, one reconstruction miss rejected the whole 100-vs-110 table, and no record scored the walk-forward test windows alone. Its 48 comparison records equal v5's descriptive ones. |
| `japan-family-l-floor-tail-favourite-v3` | `c726de4222a4...` | `6b75cac` | 50 | Superseded by the second review (below): its 100-vs-110 check rebuilt Place dividends with an equal split of the profit, not JRA's published rule. |
| `japan-family-l-floor-tail-favourite-v2` | `d987d60b9b3b...` | `2321118` | 50 | Superseded: at t it counted a horse scratched before t as a runner (below). |
| `japan-family-l-floor-tail-favourite` | `a3bfd6a65fa8...` | `88c40aa` | 50 | Superseded by the review of 2026-11-27 (below). 24 scored at the final price; all 26 board-at-t variants data pending (no post time on the logged days). |

A first attempt at `f0d90a1` was stopped before it scored or appended anything; its only trace is the first family's
freeze line, which that family's run reused.

## What changed in this round and why

Fix round 4 follows the owner-requested restart plan (GPT-6 Astra's review of `1774fb3`, posted on #82 on
2026-11-27). Its diagnosis: the churn came from turning a paired comparison into a general check of JRA's settlement.
The 100-vs-110 table is in scope; certifying every placed horse's dividend, handling every premium exception and
rejecting the whole table on one discrepancy are not. So observations, predictions and reconstruction diagnostics are
now kept apart, and every realised result and every attempted prediction stays, each with its status. Each change is
pinned by a test in `tests/countries/japan/test_avoidance_price.py` or `tests/countries/japan/models/test_v0_store.py`,
and the changed pre-registration is a new family, v5, frozen before any read:

- **A. An unresolved premium.** v4 added the 5% Place premium to every race the boost calendar marks premium or ultra
  premium, but JRA may withhold it when betting is very concentrated (`takeout`'s `boost_may_be_withheld`) and the
  store holds no signal of whether it was paid. Such a race now has its prediction unavailable, with a reason naming
  the unresolved premium; the premium is never inferred from the realised dividends. Its realised dividend stays and
  no other race moves. Verified premium settlement is #481.
- **B. No whole-table rejection.** In v4 one placed horse off its realised dividend by more than the tolerance failed
  the record: the registry kept no metric and the rebuilt predictions were cleared. The reconstruction check is now a
  diagnostic: a race with such a miss has its row labelled discrepant and keeps its prediction, the predicted x
  realised table counts every available prediction, and the record stays scored with its metrics. The counts come
  from the rows' final statuses: attempted (the floor-bound placed favourites), available (with a prediction),
  unavailable (without one, each with its reason) and discrepant (available, labelled by the diagnostic). They agree
  across the registry record, the private report and its CSV; the public report gives the attempted count only (its
  realised table), the others being built on the votes (D21).
- **C. The holdout read boundary (D14).** `models/v0/store.read_store_table`, which #81's development read uses,
  materialised the asked columns for the whole date range and only then dropped holdout days, so holdout prices and
  outcomes entered memory outside `LockedHoldout.open`. It now reads the date column first and scopes the scan of the
  other columns to the development days, so a row group of holdout days alone is skipped on its statistics and no
  other column of a holdout day reaches Python; what it returns for development days is unchanged for every caller.
  The votes reader decoded each selected row group's batches into Python before filtering to the wanted races; it now
  filters inside the dataset scan. The tests trace the read itself (no Python frame of the package or of pyarrow ever
  holds a holdout value), not only its output. In fix round 5 the merge with main replaced this branch's
  `read_store_table` change with main's #415 (`core.eval.holdout.file_development_filter` over one memory-mapped
  snapshot), the same reader-side boundary, which the owner accepted for #82 on 2026-11-27: a development read meets
  D14 when no holdout value reaches Python. Arrow still decodes a row group that mixes development and holdout days
  before its filter drops the holdout rows; row-group isolation is #502.
- **D. A votes file that cannot be opened.** A `pool_votes.parquet` that exists but cannot be read for its digest
  aborted the run after the freeze. The votes' data build now records the failure, scoring goes on, the realised
  counts stand and the prediction is unavailable. After fix round 5 the probe of the file sits under the same handler
  (a `jravan` directory the process cannot traverse no longer aborts the run either), and the build keeps the path as
  its identity whatever the file's availability, the failure going in the refresh
  (`processed/jravan/pool_votes.parquet@unreadable-PermissionError`, `@missing`, or the file's digest), so a rerun
  of the same family with the file restored refreshes the two 100-vs-110 records instead of adding two variants.
- **E. Walk-forward rows.** Every development race was scored once (`paired_same_races:development`), the first
  training span included. Each paired comparison now has, beside that descriptive record (#81's scheme and its name,
  unchanged), a walk-forward record named `<variant>:walk_forward`: the same per-race returns over the races in the
  test windows of `core.eval.folds.walk_forward` under Japan's scheme (`config/folds.yaml`: quarters, expanding,
  min_train 22, a 7-day embargo; the first test window is 2020-Q1). Its pooled figures carry D7's bounds, and each
  window's paired difference and n are in the registry record, labelled with the window. Nothing is fitted, so the
  scheme only decides which races are scored; #77's entropy cut points come from 2014-06-07 to 2019-12-31, the first
  window's training span, so the walk-forward rows are out of their sample. The 100-vs-110 check has no walk-forward
  record.
- **F. The command's printed summary (D21).** `run_study`'s returned summary, which `main` prints, named a failed
  100-vs-110 record among `failed_or_pending_variants`, telling the withheld check's result outside `write_report`.
  It now lists only the jra-store variants there and names the two votes-built records under `withheld_variants`.
- **G. A new family.** The changed semantics are frozen as `japan-family-l-floor-tail-favourite-v5`, superseding v4;
  the earlier families keep their records.

Every jra-store figure of v5's descriptive records equals v4's to the last digit; at t, each walk-forward record equals
its descriptive one (the two logged days are in the 2026-Q4 test window).

### Round 2 (v3 to v4)

The review of v3 (2026-11-27, round 2) found one defect in the 100-vs-110 check's Place rule and five smaller ones.
Each is now pinned by a test in `tests/countries/japan/test_avoidance_price.py` or
`tests/core/eval/test_registry.py`, and the changed pre-registration is a new family, v4, frozen before any read:

1. **The Place rule.** v1 to v3 rebuilt each placed horse's Place dividend by an equal-split rule: the net pool first
   returned the placed horses' stakes and the profit left was split equally over them. That is not how JRA pays
   Place. It is replaced by JRA's published stake-back-share rule (払戻金の計算方法,
   `core.pricing.stake_back_share_dividend`): each placed horse's payout pool is its own tickets plus an equal share
   of the tickets on the horses that did not place, times the Place rate without any premium (0.80), shared over its
   own tickets, rounded down to 10 yen and never under 100. Plus-10 is then tested on that base payout alone. On a
   premium or ultra-premium PLACE pool (the boost calendar), 5% of sales is added after Plus-10, split equally over
   the placed horses and paid per ticket, rounded down to 10 yen; v1 to v3 instead rebuilt such a race at its boosted
   rate.
   The premium is taken as paid; JRA may withhold it, and a race where it did would fail the reconstruction check.
   The prediction, the reconstruction check and the cross-tab stay JRA-VAN-derived and private (D21).
2. **A prediction that cannot be made.** When floor-bound placed favourites exist but none of their races can be
   rebuilt (the votes missing or unreadable), the record keeps the realised counts from the jra store and holds no
   reconstruction check and no predicted cell: the prediction is unavailable, not a scored zero. A covered selection
   with no eligible favourite is still a scored zero, every cell 0.
3. **A scratch at t.** A runner whose latest WIN or PLACE row at t is SCR now has no price in any pool at t, Wide
   pairs included, even when an older row of the other pool still prices it (`docs/specs/board-at-t.md`, the board
   module's rule). v3 let such a runner stay eligible for L2's Place and Wide tickets and dropped its race from L3's
   rank rule. The board-at-t rows are unchanged from v3's, so the rule changed no logged race.
4. **A failed check's rows.** When the reconstruction check fails, each rebuilt row of the private 100-vs-110 CSV has
   its prediction cleared and is labelled rejected; its realised dividend stays.
5. **The post-time build.** The at-t records carried the scheduled-post coverage count inside the build's name, so a
   backfilled post time would have counted every at-t variant again in the family size. They now carry one build
   whose identity names the source and whose refresh value is a digest of the race -> scheduled post mapping; the
   coverage counts are in the full report.
6. **The registry summary.** `core.eval.registry.write_summary` showed the status of a licensed-source run, and a
   status can tell the result of a withheld check. A licensed-source run now shows only its E id, its variant and
   `withheld (licensed source)`. The summaries of every licensed-source family were regenerated with it: v1 to v3 of
   this family, the three c1 nowcast families and the c4 meta family; no other line changed.

Every figure from the jra store in v4, at the final price and at t, equals v3's to the last digit.

### Round 1 (v1 to v2 and v3)

The review of the first run (2026-11-27) found six defects. Each is pinned by a test in
`tests/countries/japan/test_avoidance_price.py`, and the changed pre-registration was a new family, frozen before any
read:

1. **t on the logged days.** The logged captures carry no `post_utc`, so the first run had no board at t. When a
   race's captures carry none, t is now the scheduled post - 2 min from `board.snapshot.post_times_for` (the
   `<VENUE>.races.json` sidecar, else HKTrack's stored card), as #86's board-quality study reads it. A capture's own
   `post_utc` wins over the schedule; with neither, the race stays data pending.
2. **Selection at t from the board at t alone.** The first run chose the favourite, the top-3 box and every set among
   the starters of the final results, so a horse scratched after t dropped out of the choice and the next one was
   promoted. A runner at t is now a horse priced on the board at t, whatever the results say later, and a ticket
   holding a runner the results mark scratched is refunded (stake back): the store holds no refund row for a scratch,
   so the refund comes from `runners.is_scratched`. Look-ahead variants keep the final starters.
3. **The reconstruction check.** Before its predicted x realised table, the 100-vs-110 check rebuilds the Place
   dividend of every placed horse of its races (the floor favourite and the others) from the votes and compares it
   with the realised one. The tolerance, 10 yen per 100 (one 10-yen rounding step; a floor horse's 100 or 110 is the
   check's own question), is frozen in the spec. A miss above it fails the record with its reason, and no table is
   scored. The check is JRA-VAN-derived: it and the cross-tab are in the private report (withheld, D21).
4. **Coverage at t.** A race counts for the at-t 100-vs-110 check only once its favourite at t has a PLACE price at t;
   with no such race the check is data pending, not a zero.
5. **L2's no-bet race.** A race whose every priced ticket is at the floor stays paired: the rule bets nothing and its
   side returns its stake (`rules.L2.no_bet_race: stake_back`, as L7's skipped race). Only a race without board data
   drops.
6. **Votes that cannot be read.** A missing or unreadable `pool_votes.parquet` keeps the realised counts from the jra
   store; only the prediction is empty.

v2 ran these six changes and its at-t rows showed one more defect: it counted as a runner at t every horse of any
priced row on the board, and the logged Wide board still prices the pairs of a horse scratched before t (two of the
logged races). L2's Wide rule then bet pairs no one could bet, and L3's rank rule dropped those two races. v3 counts
a runner at t from the WIN and PLACE boards only, which mark a scratched runner unpriced; nothing else changed.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D2 | The board at t is read at post - 2 min (`config/decision.yaml` `primary_min`). | `docs/problem-statement.md` §1, #14 |
| D3 | Every JRA pool is in research scope, exotics at tier 2. | §1, #14; amended 2026-11-24 (#127) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided (`config/scoring.yaml`). | §3, #14 |
| D10 | Only races from 2014-06-07. | §4, #14 |
| D11, D14, D15 | Holdout A2 from 2026-11-28 is fully locked for outcomes and prices. It is not read or opened here. | §5, #14; `config/holdout.yaml` |
| D17 | The family size counts every variant, each cut point, pool and evaluation included: 98. Every record carries it as `variants_searched`. | §6, #14 |
| D9 | The walk-forward test windows: Japan's scheme in `config/folds.yaml` (quarters anchored on January, expanding, min_train 22, embargo 7 days), read through `core.eval.folds.load_country_scheme` and frozen in the spec. | §4, #14; #70 |
| D21 | No figure from a licensed source is in the repository. The 100-vs-110 prediction, its reconstruction diagnostic and its counts of predictions are JRA-VAN-derived and withheld, their result and status included. | §8, #14; owner 2026-11-26 (strict) |
| Floor | A board price at or below 1.1 is at the floor. | owner, 2026-11-24 |
| #82 answer | `l3_tail_odds: [51.0, 101.0]` (fractional 50/1 and 100/1, #77's convention); `l7_entropy_cut_points`: #77's quintiles (0.6952092433071905, 0.7467255002267639, 0.7851488760086348, 0.8215443360884407, from `config/calibration.yaml`), applied to final and board-at-t entropy; `sets: every_set`; `favourite_ties: lower_horse_no`; `l7_favourite_boxes: top3_box`; `l7_skipped_race: stake_back`; `plus10_inputs: jravan_pool_votes`. Confirmed: L3's ranks and L7's box read only the WIN board at t (BRACKET_QUINELLA excluded); L2 reads the PLACE and Wide boards at t only. | #82, owner comment of 2026-11-26 |
| Reframe | Every rule x variant is a ranked row against its comparator with n, CI and the variants searched; no D18 pass/fail; no sign-off key; an avoid rule drops nothing from the data. | #82, owner comment of 2026-11-27 |

Fixed from the request without a choice: the exotic tail ranks 9th or worse and 13th or worse by WIN odds (E12); a
handicap is `races.weight_type == "handicap"` (#77).

## Hypothesis

The frozen spec's `hypotheses`, in full:

- **L2 (E14, R3).** "The return of the base strategy less its tickets at the floor against the same tickets; R3: the
  100-vs-110 check of floor-bound placed favourites."
- **L3 (E12).** "The return of the strategy less its tail tickets against the strategy with the tail."
- **L7 (E3, E7).** "The return with no favourite Place in the flattest-entropy quintile, and no favourite Win, Place
  or box in handicaps, against the same tickets in all races."

The null is the starting point: no difference between a rule and its comparator. A positive paired difference is the
rule returning more per unit staked than its comparator on the same races.

## Variants tried

The family has 98 variants (`Registry.family_size`) and 98 records, all from one run at git `64ccd07`: the 48
descriptive paired comparisons and their 48 walk-forward records, all scored (23 + 23 at the final price, 25 + 25 at
t), and the two 100-vs-110 records (one per selection, descriptive only), which are withheld (D21). The tables below
up to the ranked options are the descriptive records, unchanged from v4; the walk-forward rows follow them.

Data and split:
- Development days from 2014-06-07 to 2026-11-23: 43,130 run races on 3,597 meetings, 601,265 starters, 2,454
  handicaps. The dividend rows read are WIN, PLACE, QUINELLA_PLACE, QUINELLA, FORECAST, TRIO and TIERCE: 474,675.
- No day in the store is a holdout day (the store ends 2026-11-23). The readers drop holdout days anyway; the votes
  reader refuses a holdout race; `LockedHoldout` was not opened.
- Nothing is fitted. A descriptive record uses every development race once (`paired_same_races:development`); a
  walk-forward record uses the races of the 28 quarterly test windows from 2020-Q1 to 2026-Q4 alone (23,871 races on
  1,991 meetings; the training span 2014-06-07 to 2019-12-31 is scored by none of them).

Selections:
- `final (look-ahead)`: the choice reads the final Win odds (`runners.win_odds`), which are not known at a decision
  time, among the final starters.
- `board at t (T-2)`: odds_timeseries through `read_asof` at the race's post time - 2 min, the post from HKTrack's
  card for all 24 logged races (2026-11-21 and 2026-11-22, Kyoto: two meetings). Three races of 2026-11-21 have no
  capture fetched by t, so 21 races on 2 meetings are paired at t. A runner is a horse priced in the WIN or PLACE pool
  of the board at t; one of the 21 races holds a runner priced at t whom the results mark scratched, and its tickets
  are refunded.

Each side's return is its mean over the paired races of payout / stake; the paired difference is the mean of rule
minus comparator. Lower and upper are the 5% and 95% quantiles of the bootstrap over meetings (D7; 10,000 replicates,
seed 82), each a one-sided 95% bound, per variant, with no family-wise correction. Under `stake_back`, an L7 race the
rule skips returns its stake (1.0), and so does an L2 race whose every ticket is at the floor.

### Paired comparisons at the final price (look-ahead, descriptive)

The base strategy is one unit on every starter (WIN, PLACE) or every combination of the starters (exotics), flat.
L2's comparator is the same tickets with those at the floor; L3's, the same tickets with the tail; L7's, the same
favourite tickets in every race. "Races changed" counts the paired races where the rule drops a ticket or skips the
race. No race was dropped from any scored variant.

| Rule | Pool | Cut | Races | Rule return [lower, upper] | Comparator return [lower, upper] | Paired difference | Lower (5%) | Upper (95%) | Races changed |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| L2 | WIN | floor | 43,130 | 0.7245 [0.7138, 0.7353] | 0.7247 [0.7140, 0.7355] | -0.00018 | -0.00030 | -0.00004 | 195 |
| L3 | WIN | above 51 | 43,130 | 0.7941 [0.7877, 0.8005] | 0.7247 [0.7140, 0.7355] | +0.06939 | +0.05864 | +0.08002 | 42,208 |
| L3 | PLACE | above 51 | 43,130 | 0.7968 [0.7936, 0.8000] | 0.7316 [0.7269, 0.7364] | +0.06520 | +0.06040 | +0.06992 | 42,208 |
| L3 | WIN | above 101 | 43,130 | 0.7860 [0.7778, 0.7941] | 0.7247 [0.7140, 0.7355] | +0.06127 | +0.05240 | +0.06986 | 38,404 |
| L3 | PLACE | above 101 | 43,130 | 0.7862 [0.7825, 0.7901] | 0.7316 [0.7269, 0.7364] | +0.05464 | +0.05070 | +0.05852 | 38,404 |
| L3 | QUINELLA_PLACE | rank 9 or worse | 43,130 | 0.7613 [0.7561, 0.7666] | 0.6189 [0.6126, 0.6255] | +0.14238 | +0.13375 | +0.15059 | 41,120 |
| L3 | QUINELLA | rank 9 or worse | 43,130 | 0.7342 [0.7258, 0.7427] | 0.6018 [0.5910, 0.6129] | +0.13244 | +0.11902 | +0.14535 | 41,120 |
| L3 | FORECAST | rank 9 or worse | 43,130 | 0.7273 [0.7184, 0.7362] | 0.5989 [0.5870, 0.6113] | +0.12837 | +0.11382 | +0.14240 | 41,120 |
| L3 | TRIO | rank 9 or worse | 43,130 | 0.7031 [0.6930, 0.7135] | 0.5350 [0.5231, 0.5473] | +0.16813 | +0.15248 | +0.18334 | 41,120 |
| L3 | TIERCE | rank 9 or worse | 43,130 | 0.7060 [0.6943, 0.7179] | 0.5664 [0.5513, 0.5825] | +0.13958 | +0.12054 | +0.15776 | 41,120 |
| L3 | QUINELLA_PLACE | rank 13 or worse | 43,130 | 0.6934 [0.6875, 0.6994] | 0.6189 [0.6126, 0.6255] | +0.07447 | +0.06860 | +0.08016 | 30,701 |
| L3 | QUINELLA | rank 13 or worse | 43,130 | 0.6825 [0.6718, 0.6933] | 0.6018 [0.5910, 0.6129] | +0.08068 | +0.07147 | +0.08975 | 30,701 |
| L3 | FORECAST | rank 13 or worse | 43,130 | 0.6809 [0.6693, 0.6929] | 0.5989 [0.5870, 0.6113] | +0.08198 | +0.07216 | +0.09167 | 30,701 |
| L3 | TRIO | rank 13 or worse | 43,130 | 0.6231 [0.6123, 0.6345] | 0.5350 [0.5231, 0.5473] | +0.08810 | +0.07612 | +0.10017 | 30,701 |
| L3 | TIERCE | rank 13 or worse | 43,130 | 0.6512 [0.6377, 0.6654] | 0.5664 [0.5513, 0.5825] | +0.08481 | +0.06922 | +0.10007 | 30,701 |
| L7 | PLACE | flattest quintile | 43,130 | 0.8863 [0.8820, 0.8906] | 0.8410 [0.8360, 0.8460] | +0.04525 | +0.04240 | +0.04818 | 8,613 |
| L7 | WIN | handicap | 43,130 | 0.8009 [0.7916, 0.8101] | 0.7864 [0.7767, 0.7958] | +0.01452 | +0.01198 | +0.01707 | 2,454 |
| L7 | PLACE | handicap | 43,130 | 0.8533 [0.8485, 0.8582] | 0.8410 [0.8360, 0.8460] | +0.01229 | +0.01086 | +0.01372 | 2,454 |
| L7 | QUINELLA_PLACE | handicap, top-3 box | 43,130 | 0.8167 [0.8092, 0.8243] | 0.8032 [0.7953, 0.8110] | +0.01358 | +0.01140 | +0.01574 | 2,454 |
| L7 | QUINELLA | handicap, top-3 box | 43,130 | 0.8057 [0.7948, 0.8164] | 0.7887 [0.7774, 0.7997] | +0.01705 | +0.01413 | +0.01995 | 2,454 |
| L7 | FORECAST | handicap, top-3 box | 43,130 | 0.7797 [0.7687, 0.7903] | 0.7611 [0.7498, 0.7721] | +0.01859 | +0.01573 | +0.02147 | 2,454 |
| L7 | TRIO | handicap, top-3 box | 43,130 | 0.7845 [0.7624, 0.8073] | 0.7711 [0.7478, 0.7956] | +0.01339 | +0.00634 | +0.02017 | 2,454 |
| L7 | TIERCE | handicap, top-3 box | 43,130 | 0.6982 [0.6777, 0.7195] | 0.6767 [0.6552, 0.6988] | +0.02151 | +0.01553 | +0.02730 | 2,454 |

### Paired comparisons at the board at t (T-2, descriptive)

The same rules and comparators, chosen on the board at t: 21 races on 2 meetings (Kyoto, 2026-11-21 and 2026-11-22),
the same 21 races in every row; the 3 races without a board at t are dropped from each. **Single-run and small.** With
two meetings the bootstrap resamples two clusters, so each lower and upper only spans the two meetings' own means: the
bounds are not a sampling interval here, and none of these rows is evidence on its own. L2's WIN rule changes no race
(no WIN price is at the floor at t).

| Rule | Pool | Cut | Races | Rule return [lower, upper] | Comparator return [lower, upper] | Paired difference | Lower (5%) | Upper (95%) | Races changed |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| L2 | WIN | floor | 21 | 0.4154 [0.2536, 0.5367] | 0.4154 [0.2536, 0.5367] | +0.00000 | +0.00000 | +0.00000 | 0 |
| L2 | PLACE | floor | 21 | 0.4258 [0.3945, 0.4674] | 0.4637 [0.4241, 0.5165] | -0.03797 | -0.04910 | -0.02962 | 9 |
| L2 | QUINELLA_PLACE | floor | 21 | 0.2795 [0.2301, 0.3454] | 0.2807 [0.2301, 0.3481] | -0.00115 | -0.00267 | +0.00000 | 1 |
| L3 | WIN | above 51 | 21 | 0.4288 [0.3831, 0.4631] | 0.4154 [0.2536, 0.5367] | +0.01344 | -0.07363 | +0.12953 | 20 |
| L3 | PLACE | above 51 | 21 | 0.6593 [0.5670, 0.7823] | 0.4637 [0.4241, 0.5165] | +0.19554 | +0.14289 | +0.26575 | 20 |
| L3 | WIN | above 101 | 21 | 0.5070 [0.3244, 0.6440] | 0.4154 [0.2536, 0.5367] | +0.09163 | +0.07075 | +0.10729 | 17 |
| L3 | PLACE | above 101 | 21 | 0.5640 [0.4997, 0.6496] | 0.4637 [0.4241, 0.5165] | +0.10022 | +0.07559 | +0.13306 | 17 |
| L3 | QUINELLA_PLACE | rank 9 or worse | 21 | 0.8183 [0.6548, 1.0362] | 0.2807 [0.2301, 0.3481] | +0.53756 | +0.42463 | +0.68813 | 19 |
| L3 | QUINELLA | rank 9 or worse | 21 | 0.4692 [0.3943, 0.5253] | 0.1705 [0.1580, 0.1800] | +0.29861 | +0.23635 | +0.34531 | 19 |
| L3 | FORECAST | rank 9 or worse | 21 | 0.4351 [0.3159, 0.5246] | 0.1616 [0.1374, 0.1798] | +0.27347 | +0.17847 | +0.34473 | 19 |
| L3 | TRIO | rank 9 or worse | 21 | 0.6016 [0.4067, 0.8615] | 0.1142 [0.0731, 0.1690] | +0.48741 | +0.33363 | +0.69245 | 19 |
| L3 | TIERCE | rank 9 or worse | 21 | 0.4534 [0.4226, 0.4945] | 0.0955 [0.0760, 0.1214] | +0.35796 | +0.34663 | +0.37307 | 19 |
| L3 | QUINELLA_PLACE | rank 13 or worse | 21 | 0.3824 [0.3005, 0.4916] | 0.2807 [0.2301, 0.3481] | +0.10166 | +0.07033 | +0.14344 | 13 |
| L3 | QUINELLA | rank 13 or worse | 21 | 0.2204 [0.2040, 0.2327] | 0.1705 [0.1580, 0.1800] | +0.04988 | +0.04603 | +0.05277 | 13 |
| L3 | FORECAST | rank 13 or worse | 21 | 0.2050 [0.1704, 0.2310] | 0.1616 [0.1374, 0.1798] | +0.04339 | +0.03305 | +0.05114 | 13 |
| L3 | TRIO | rank 13 or worse | 21 | 0.1797 [0.1137, 0.2678] | 0.1142 [0.0731, 0.1690] | +0.06552 | +0.04061 | +0.09875 | 13 |
| L3 | TIERCE | rank 13 or worse | 21 | 0.1382 [0.1131, 0.1717] | 0.0955 [0.0760, 0.1214] | +0.04274 | +0.03708 | +0.05029 | 13 |
| L7 | PLACE | flattest quintile | 21 | 1.1476 [1.1167, 1.1889] | 1.1286 [0.9667, 1.2500] | +0.01905 | -0.13333 | +0.22222 | 5 |
| L7 | WIN | handicap | 21 | 0.8952 [0.6778, 1.0583] | 0.8476 [0.6778, 0.9750] | +0.04762 | +0.00000 | +0.08333 | 1 |
| L7 | PLACE | handicap | 21 | 1.1190 [0.9667, 1.2333] | 1.1286 [0.9667, 1.2500] | -0.00952 | -0.01667 | +0.00000 | 1 |
| L7 | QUINELLA_PLACE | handicap, top-3 box | 21 | 1.2556 [1.0296, 1.4250] | 1.2635 [1.0296, 1.4389] | -0.00794 | -0.01389 | +0.00000 | 1 |
| L7 | QUINELLA | handicap, top-3 box | 21 | 1.3921 [1.0778, 1.8111] | 1.3444 [0.9944, 1.8111] | +0.04762 | +0.00000 | +0.08333 | 1 |
| L7 | FORECAST | handicap, top-3 box | 21 | 1.2825 [1.0069, 1.6500] | 1.2349 [0.9236, 1.6500] | +0.04762 | +0.00000 | +0.08333 | 1 |
| L7 | TRIO | handicap, top-3 box | 21 | 0.9381 [0.5556, 1.2250] | 0.8905 [0.5556, 1.1417] | +0.04762 | +0.00000 | +0.08333 | 1 |
| L7 | TIERCE | handicap, top-3 box | 21 | 0.6571 [0.4648, 0.8014] | 0.6095 [0.4648, 0.7181] | +0.04762 | +0.00000 | +0.08333 | 1 |

L7's handicap rows at t rest on the one handicap of the two days, and its flattest-quintile row on 5 races. The ranked
options at t, from the same records, are in the full report.

### Walk-forward rows (the test windows of `config/folds.yaml`, v5)

Each descriptive comparison above has one walk-forward record, `<variant>:walk_forward` (fold scheme
`walk_forward:quarter x1 step 1, expanding, min_train 22, embargo 7d, anchor month 1`): the same per-race returns over
the races in the 28 quarterly test windows from 2020-01-01 to 2026-12-31 alone, the first training span (2014-06-07
to 2019-12-31) scored by none. These are the evaluation rows; the descriptive rows above are the historical
description. Bounds are D7's meeting-clustered one-sided bounds over the windows' races (1,991 meetings at the final
price); each window's own paired difference and n are in the registry summary, and "Windows +/-" counts the windows
whose difference is above or below 0 (a window at exactly 0 is in neither). At the final price (look-ahead):

| Rule | Pool | Cut | Races | Rule return [lower, upper] | Comparator return [lower, upper] | Paired difference | Lower (5%) | Upper (95%) | Races changed | Windows +/- |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| L2 | WIN | floor | 23,871 | 0.7206 [0.7063, 0.7346] | 0.7209 [0.7066, 0.7348] | -0.00028 | -0.00038 | -0.00018 | 119 | 6 / 21 |
| L3 | WIN | above 51 | 23,871 | 0.7909 [0.7822, 0.7995] | 0.7209 [0.7066, 0.7348] | +0.07001 | +0.05568 | +0.08448 | 23,309 | 28 / 0 |
| L3 | PLACE | above 51 | 23,871 | 0.7964 [0.7921, 0.8007] | 0.7301 [0.7239, 0.7364] | +0.06629 | +0.05988 | +0.07269 | 23,309 | 28 / 0 |
| L3 | WIN | above 101 | 23,871 | 0.7784 [0.7676, 0.7890] | 0.7209 [0.7066, 0.7348] | +0.05749 | +0.04539 | +0.06907 | 21,119 | 25 / 3 |
| L3 | PLACE | above 101 | 23,871 | 0.7832 [0.7783, 0.7884] | 0.7301 [0.7239, 0.7364] | +0.05311 | +0.04773 | +0.05840 | 21,119 | 28 / 0 |
| L3 | QUINELLA_PLACE | rank 9 or worse | 23,871 | 0.7571 [0.7503, 0.7642] | 0.6201 [0.6115, 0.6289] | +0.13704 | +0.12594 | +0.14834 | 22,529 | 28 / 0 |
| L3 | QUINELLA | rank 9 or worse | 23,871 | 0.7253 [0.7144, 0.7362] | 0.5989 [0.5848, 0.6135] | +0.12637 | +0.10913 | +0.14364 | 22,529 | 28 / 0 |
| L3 | FORECAST | rank 9 or worse | 23,871 | 0.7181 [0.7067, 0.7296] | 0.5987 [0.5829, 0.6147] | +0.11943 | +0.10040 | +0.13835 | 22,529 | 28 / 0 |
| L3 | TRIO | rank 9 or worse | 23,871 | 0.6958 [0.6830, 0.7090] | 0.5256 [0.5110, 0.5409] | +0.17021 | +0.15079 | +0.18952 | 22,529 | 28 / 0 |
| L3 | TIERCE | rank 9 or worse | 23,871 | 0.6963 [0.6819, 0.7112] | 0.5592 [0.5387, 0.5807] | +0.13711 | +0.11179 | +0.16140 | 22,529 | 28 / 0 |
| L3 | QUINELLA_PLACE | rank 13 or worse | 23,871 | 0.6834 [0.6756, 0.6914] | 0.6201 [0.6115, 0.6289] | +0.06337 | +0.05539 | +0.07131 | 16,342 | 28 / 0 |
| L3 | QUINELLA | rank 13 or worse | 23,871 | 0.6665 [0.6531, 0.6803] | 0.5989 [0.5848, 0.6135] | +0.06761 | +0.05517 | +0.07967 | 16,342 | 27 / 1 |
| L3 | FORECAST | rank 13 or worse | 23,871 | 0.6655 [0.6502, 0.6810] | 0.5987 [0.5829, 0.6147] | +0.06678 | +0.05313 | +0.08003 | 16,342 | 25 / 3 |
| L3 | TRIO | rank 13 or worse | 23,871 | 0.6007 [0.5866, 0.6151] | 0.5256 [0.5110, 0.5409] | +0.07509 | +0.06035 | +0.08980 | 16,342 | 28 / 0 |
| L3 | TIERCE | rank 13 or worse | 23,871 | 0.6265 [0.6088, 0.6444] | 0.5592 [0.5387, 0.5807] | +0.06729 | +0.04640 | +0.08719 | 16,342 | 26 / 2 |
| L7 | PLACE | flattest quintile | 23,871 | 0.8883 [0.8826, 0.8940] | 0.8460 [0.8391, 0.8528] | +0.04237 | +0.03837 | +0.04638 | 4,761 | 28 / 0 |
| L7 | WIN | handicap | 23,871 | 0.8081 [0.7961, 0.8199] | 0.7943 [0.7820, 0.8066] | +0.01388 | +0.01051 | +0.01723 | 1,326 | 25 / 3 |
| L7 | PLACE | handicap | 23,871 | 0.8575 [0.8508, 0.8641] | 0.8460 [0.8391, 0.8528] | +0.01150 | +0.00954 | +0.01345 | 1,326 | 27 / 1 |
| L7 | QUINELLA_PLACE | handicap, top-3 box | 23,871 | 0.8195 [0.8094, 0.8297] | 0.8073 [0.7969, 0.8181] | +0.01221 | +0.00919 | +0.01518 | 1,326 | 24 / 4 |
| L7 | QUINELLA | handicap, top-3 box | 23,871 | 0.8008 [0.7864, 0.8153] | 0.7852 [0.7704, 0.8004] | +0.01561 | +0.01163 | +0.01943 | 1,326 | 25 / 3 |
| L7 | FORECAST | handicap, top-3 box | 23,871 | 0.7759 [0.7615, 0.7903] | 0.7588 [0.7441, 0.7740] | +0.01704 | +0.01308 | +0.02077 | 1,326 | 24 / 4 |
| L7 | TRIO | handicap, top-3 box | 23,871 | 0.7890 [0.7584, 0.8203] | 0.7800 [0.7478, 0.8130] | +0.00902 | -0.00160 | +0.01894 | 1,326 | 17 / 11 |
| L7 | TIERCE | handicap, top-3 box | 23,871 | 0.6962 [0.6680, 0.7256] | 0.6771 [0.6475, 0.7075] | +0.01916 | +0.01052 | +0.02726 | 1,326 | 22 / 6 |

No race in a window was dropped from any of these. At t (T-2), both logged days fall in the 2026-Q4 window, so each of
the 25 walk-forward records at t scores the same 21 races as its descriptive record and equals it figure for figure
(the table above at t); it is one window, not a walk-forward history.

### L7: what the skipped favourites returned (derived)

Under `stake_back` the rule's side counts each skipped race at 1.0, so an L7 rule row ranks above its comparator
whenever the skipped favourites returned under 1; the paired difference is (k / N) x (1 - r), with N the paired races,
k the skipped ones and r the skipped favourites' own mean return. The table solves for r from the recorded figures,
and for the rest of the races from the comparator's mean. This is arithmetic on the records, not a registry metric:
its bounds come from the difference's bounds with k held fixed and are approximate (the bootstrap redraws k).

| Variant | Races skipped (k) | Skipped favourites' return r [approx. lower, upper] | The other races | All races |
| --- | ---: | --- | ---: | ---: |
| Favourite PLACE, flattest quintile | 8,613 | 0.7734 [0.7588, 0.7877] | 0.8579 | 0.8410 |
| Favourite WIN, handicaps | 2,454 | 0.7448 [0.7000, 0.7894] | 0.7889 | 0.7864 |
| Favourite PLACE, handicaps | 2,454 | 0.7840 [0.7588, 0.8092] | 0.8445 | 0.8410 |
| Top-3 QUINELLA_PLACE box, handicaps | 2,454 | 0.7614 [0.7233, 0.7996] | 0.8057 | 0.8032 |
| Top-3 QUINELLA box, handicaps | 2,454 | 0.7003 [0.6494, 0.7517] | 0.7940 | 0.7887 |
| Top-3 FORECAST box, handicaps | 2,454 | 0.6733 [0.6227, 0.7236] | 0.7664 | 0.7611 |
| Top-3 TRIO box, handicaps | 2,454 | 0.7648 [0.6456, 0.8886] | 0.7715 | 0.7711 |
| Top-3 TIERCE box, handicaps | 2,454 | 0.6220 [0.5202, 0.7270] | 0.6800 | 0.6767 |

### Ranked options (mean return per ¥100 staked, final price, look-ahead)

Each scored comparison gives two options; a comparator shared by several variants (every WIN ticket serves L2 and both
L3 WIN cut-offs) is listed once. L7 rule rows count a skipped race as ¥100 back, so their place in this list reflects
stake-back as much as betting (see the derived table above). The rank orders the options and is not a test. No option
returns above its stake.

| Rank | Option | Return per ¥100 | Lower (5%) | Upper (95%) |
| ---: | --- | ---: | ---: | ---: |
| 1 | `L7:flattest_quintile:PLACE:final_look_ahead` | 88.63 | 88.20 | 89.06 |
| 2 | `L7:handicap:PLACE:final_look_ahead` | 85.33 | 84.85 | 85.82 |
| 3 | favourite PLACE, every race (comparator) | 84.10 | 83.60 | 84.60 |
| 4 | `L7:handicap:QUINELLA_PLACE_top3_box:final_look_ahead` | 81.67 | 80.92 | 82.43 |
| 5 | `L7:handicap:QUINELLA_top3_box:final_look_ahead` | 80.57 | 79.48 | 81.64 |
| 6 | favourite top-3 QUINELLA_PLACE box, every race (comparator) | 80.32 | 79.53 | 81.10 |
| 7 | `L7:handicap:WIN:final_look_ahead` | 80.09 | 79.16 | 81.01 |
| 8 | `L3:PLACE:above_51:final_look_ahead` | 79.68 | 79.36 | 80.00 |
| 9 | `L3:WIN:above_51:final_look_ahead` | 79.41 | 78.77 | 80.05 |
| 10 | favourite top-3 QUINELLA box, every race (comparator) | 78.87 | 77.74 | 79.97 |
| 11 | favourite WIN, every race (comparator) | 78.64 | 77.67 | 79.58 |
| 12 | `L3:PLACE:above_101:final_look_ahead` | 78.62 | 78.25 | 79.01 |
| 13 | `L3:WIN:above_101:final_look_ahead` | 78.60 | 77.78 | 79.41 |
| 14 | `L7:handicap:TRIO_top3_box:final_look_ahead` | 78.45 | 76.24 | 80.73 |
| 15 | `L7:handicap:FORECAST_top3_box:final_look_ahead` | 77.97 | 76.87 | 79.03 |
| 16 | favourite top-3 TRIO box, every race (comparator) | 77.11 | 74.78 | 79.56 |
| 17 | `L3:QUINELLA_PLACE:rank_9_or_worse:final_look_ahead` | 76.13 | 75.61 | 76.66 |
| 18 | favourite top-3 FORECAST box, every race (comparator) | 76.11 | 74.98 | 77.21 |
| 19 | `L3:QUINELLA:rank_9_or_worse:final_look_ahead` | 73.42 | 72.58 | 74.27 |
| 20 | every PLACE ticket (comparator) | 73.16 | 72.69 | 73.64 |
| 21 | `L3:FORECAST:rank_9_or_worse:final_look_ahead` | 72.73 | 71.84 | 73.62 |
| 22 | every WIN ticket (comparator) | 72.47 | 71.40 | 73.55 |
| 23 | `L2:WIN:final_look_ahead` | 72.45 | 71.38 | 73.53 |
| 24 | `L3:TIERCE:rank_9_or_worse:final_look_ahead` | 70.60 | 69.43 | 71.79 |
| 25 | `L3:TRIO:rank_9_or_worse:final_look_ahead` | 70.31 | 69.30 | 71.35 |
| 26 | `L7:handicap:TIERCE_top3_box:final_look_ahead` | 69.82 | 67.77 | 71.95 |
| 27 | `L3:QUINELLA_PLACE:rank_13_or_worse:final_look_ahead` | 69.34 | 68.75 | 69.94 |
| 28 | `L3:QUINELLA:rank_13_or_worse:final_look_ahead` | 68.25 | 67.18 | 69.33 |
| 29 | `L3:FORECAST:rank_13_or_worse:final_look_ahead` | 68.09 | 66.93 | 69.29 |
| 30 | favourite top-3 TIERCE box, every race (comparator) | 67.67 | 65.52 | 69.88 |
| 31 | `L3:TIERCE:rank_13_or_worse:final_look_ahead` | 65.12 | 63.77 | 66.54 |
| 32 | `L3:TRIO:rank_13_or_worse:final_look_ahead` | 62.31 | 61.23 | 63.45 |
| 33 | every QUINELLA_PLACE combination (comparator) | 61.89 | 61.26 | 62.55 |
| 34 | every QUINELLA combination (comparator) | 60.18 | 59.10 | 61.29 |
| 35 | every FORECAST combination (comparator) | 59.89 | 58.70 | 61.13 |
| 36 | every TIERCE combination (comparator) | 56.64 | 55.13 | 58.25 |
| 37 | every TRIO combination (comparator) | 53.50 | 52.31 | 54.73 |

### L2: the 100-vs-110 check (Plus-10, 04 §1.3)

Floor-bound placed favourites at the final price: the favourite on the final Win odds (ties to the lower horse
number) whose Place paid at most 110. At t: the favourite on the Win board at t whose PLACE low figure at t is at or
below 1.1 (a race counts only when that favourite has a PLACE price at t), and that placed. From the jra store and our
own logged boards alone:

| Selection | Races | Floor-bound placed favourites | Paid 100 (Plus-10 withheld) | Paid 110 | Paid above 110 |
| --- | ---: | ---: | ---: | ---: | ---: |
| final (look-ahead) | 43,130 | 10,236 | 607 | 9,629 | 0 |
| board at t (T-2) | 21 with a board at t | 9 | 2 | 6 | 1 |

At t a floor price on the board need not end at the floor: one of the nine paid above 110.

The prediction of each of these dividends (100 or 110) from its race's settled PLACE pool reads the JRA-VAN H1 votes
(the jravan `pool_votes` table, #368): sales from the pool row net of refunds, the placed horses from the jra
dividends, and the frozen Place rule, JRA's published stake-back share (`core.pricing.stake_back_share_dividend` at the
Place rate, 0.80), then `predicted_floor_dividend` on the base payout. A race whose PLACE pool the boost calendar marks
premium or ultra premium has no prediction (its premium settlement is unresolved; #481). The frozen reconstruction
diagnostic compares every placed horse's rebuilt Place dividend with the realised one (tolerance 10 yen per 100) and
labels a race with a larger miss discrepant, keeping its prediction in the predicted x realised table. The diagnostic,
the cross-tab, the counts of predictions attempted, available, unavailable and discrepant, and the two records'
status are in the private report (withheld, D21): `floor-tail-favourite-japan-dd4d0926b6bc-plus10-jravan.md` under the
data directory. The attempted count is the realised table above.

## Alternatives rejected and why

- **A decision-time selection on history from the JRA-VAN time series.** The jravan `odds_timeseries` holds WIN,
  PLACE, BRACKET_QUINELLA and QUINELLA boards on history, which could give L2, L3 and L7 a board at t before the
  logger's days. It is another ticket's ground (#404 covers similar work) and would make those rows JRA-VAN-derived
  (withheld). Not built here; filed as #464.
- **Waiting for #352's post-time backfill before scoring at t** (the first run's choice). Rejected in review:
  `board.snapshot.post_times_for` already gives the scheduled post from HKTrack's card, and #86 reads it the same way.
  A capture's own `post_utc` still wins when it has one.
- **Counting a runner at t from every priced row of the board** (v2). The logged Wide board still prices the pairs of
  a horse scratched before t; the WIN and PLACE boards mark it unpriced.
- **Settling a scratch from the dividends table.** The store holds no refund row for a scratched runner, so a ticket
  on a runner chosen at t and scratched later would lose instead of being refunded; the refund comes from
  `runners.is_scratched`.
- **A reconstruction tolerance of 0.** JRA rounds Place dividends down to 10 yen, and a floor horse's 100 or 110 is
  the check's own question, so one rounding step is the frozen tolerance.
- **Reading the Place votes through `read_label`.** The view reads a label-only table whole; `pool_votes` is about
  1.2 GB. The reader prunes row groups by race_id statistics and reads five columns, PLACE rows of the wanted
  development races only, filtered inside the scan.
- **Filtering the votes in Python, batch by batch (v1 to v4).** It held about 0.6 GB at its peak against about 2.4 GB
  for the scan filter when the votes are read alone, but each batch of a row group mixing development and holdout
  races reached Python whole, holdout votes included (D14). The boundary wins; smaller row groups in `pool_votes`
  would bring the memory back down.
- **Rejecting the whole 100-vs-110 table on one reconstruction miss (v1 to v4).** One placed horse off its realised
  dividend erased every prediction and the record's metrics. The miss is now a diagnostic on its race's row, and the
  rest of the evidence stays (B).
- **Taking a premium as paid (v4), or inferring it from the realised dividends.** JRA may withhold the premium and the
  store holds no settlement signal; either choice would decide the prediction from an assumption or from the answer.
  The prediction is unavailable instead (A); verified premium settlement is #481.
- **Re-scoring each walk-forward window from the races up.** Nothing is fitted, so a race's paired returns are the
  same in either evaluation; the walk-forward record restricts the descriptive record's per-race returns to the test
  windows' races. The pooled figures are bootstrapped over those races' meetings; the per-window differences carry no
  bounds of their own (a point and n per window: 28 windows for each of the 23 final-price comparisons, one for each
  of the 25 board-at-t comparisons).
- **The equal split of the Place profit (v1 to v3).** It is not JRA's published Place rule. JRA pays each placed
  horse (its own stakes plus an equal share of the losing stakes) times the payout rate, shared over its own tickets
  (the stake-back share). `core.pricing.stake_back_share_dividend` already implements it, so the study reuses it
  rather than a second copy.
- **Rebuilding a premium race at its boosted rate (v1 to v3).** The premium is 5% of sales split equally over the
  placed horses and paid after Plus-10, not a higher rate inside the stake-back share. v4 added it after Plus-10 at
  the base rate 0.80; v5 makes no prediction for such a race (above).
- **A scored zero when the votes cannot be read (v2, v3).** Eligible favourites with no prediction are not zero
  favourites; the record now holds no check and no predicted cell, and the realised counts stand.
- **`drop_race` for tied Win prices and no L7 boxes.** The owner chose `lower_horse_no` and `top3_box`.
- **A family-wise correction over the 98 variants.** The reframe ranks options; the leaderboard (#449) shrinks within
  each pool. The bounds here are per variant, and the family size is on every record.

## What this rules out

The history rows are look-ahead, development days 2014-06-07 to 2026-11-23; the board-at-t rows are 21 races on two
meetings. Nothing below adopts a rule, and no option at the final price returns above its stake.

- **The walk-forward rows (v5) against the descriptive ones.** Over the 28 test windows from 2020-Q1 (23,871 races)
  every rule keeps the sign and roughly the size of its descriptive difference. L3's tail rules return more than
  their comparators by +0.053 to +0.070 (WIN and PLACE) and +0.063 to +0.170 (exotics), every lower bound above
  +0.045, and in 25 to 28 of the 28 windows. L2's WIN floor rule stays just below its comparator (-0.00028, bounds
  -0.00038 and -0.00018; 21 windows below 0, 6 above). L7's rules return more than their comparators under
  stake-back (+0.009 to +0.042); the handicap TRIO box is the one whose lower bound is below 0 (-0.0016; 17 windows
  above, 11 below). At t the walk-forward rows are the descriptive ones: one window, two meetings.

- **L3, the tail, at the final price.** Dropping WIN or PLACE tickets on runners whose final WIN odds are above 51 or
  101 raises the per-race mean return by +0.055 to +0.069 per unit staked, every lower bound above +0.050. Dropping
  exotic combinations holding a runner ranked 9th or worse raises it by +0.128 to +0.168 (lower bounds +0.114 or
  more), and 13th or worse by +0.074 to +0.088. This is the favourite-longshot bias of the final price that the
  ledger's tail-hinge row (#294's run) measures. At t, on the 21 logged races, every L3 rule row also returns more
  than its comparator (+0.013 to +0.538), but two meetings cannot say whether the bias survives a choice at t, when
  late money has not yet moved the tail.
- **L2, the floor, in the WIN pool at the final price.** Dropping the WIN tickets priced at or below 1.1 (195 races)
  changes the race's mean return by -0.00018 (bounds -0.00030 and -0.00004). Floor-priced Win tickets returned
  slightly more than the race's other tickets, so the rule ranks just below its comparator; the effect is under 0.1%
  of stake. L2 on the PLACE and Wide boards has no history row (no final board of those pools on history). At t, the
  PLACE rule changed 9 of the 21 races and returned 0.426 against 0.464 for the same tickets with the floor; the Wide
  rule changed one race; no WIN price was at the floor at t.
- **L7 at the final price.** Every L7 rule row returns more than its comparator, as stake-back implies whenever the
  skipped favourites lose. The derived table is the informative read: the favourite's Place in the flattest-entropy
  quintile returned about 0.773 against about 0.858 in the other races, and in handicaps the favourite's Win about
  0.745 against 0.789, its Place about 0.784 against 0.845, and the top-3 QUINELLA and FORECAST boxes about 0.70 and
  0.67 against 0.79 and 0.77. The TRIO box's handicap return is not told apart from the rest (0.765, approximate
  bounds 0.646 to 0.889). At t, the two logged days hold one handicap and 5 flattest-quintile races: too few to read.
- **The 100-vs-110 check, realised part.** Of 10,236 floor-bound placed favourites, 607 (5.9%) were paid 100, so
  Plus-10 was withheld in about one floor-bound case in seventeen at the final price. At t, 9 favourites priced at the
  floor on the PLACE board placed: 2 paid 100, 6 paid 110 and 1 above 110. The prediction of which ones pay 100, the
  reconstruction diagnostic and the counts of predictions are in the private report (withheld, D21).
- **A decision time, first look only.** Every rule now has its board-at-t rows, on 21 races over 2 meetings; more
  logged days are needed before any of them can rank a rule.

## Follow-ups

Filed under epic #72 as `status:proposed`:

- #464: L2, L3 and L7 at t on history from the jravan `odds_timeseries` (WIN and PLACE boards), labelled
  JRA-VAN-derived and withheld; check the overlap with #404 first.
- #465: the Place payout rule for the 100-vs-110 check, built to reproduce JRA's Place dividends from the H1 votes,
  then the check re-run as a new family. This round covers its scope (v4, JRA's stake-back share); the figures stay
  private (D21).
- #466: L7's skipped and kept favourites as their own leaderboard rows per ¥100 actually staked, after #449, so
  stake-back does not lift them.
- #352: the post-time sidecar backfill. This study now takes the scheduled post from `post_times_for`, but #81's board
  at t still reads the captures' `post_utc` only.
- #344: `write_summary` refuses a family id with a '/', which other tickets' families in the registry hold, so this
  summary was written over the families already indexed plus v4, and for v5 over those plus v5 (only v5's file and the
  index changed).
- #481: the Place premium's settlement on premium days from a verified signal, so a premium race can have a
  prediction again (v5 leaves it unavailable).
- The votes read's memory: smaller row groups in `pool_votes` (or a race-scoped layout) would bring the scan filter's
  peak back near the old batch loop's without letting a holdout row reach Python. Not filed yet.
- This frozen family re-run as more days are logged, so its board-at-t rows rest on more than two meetings.
