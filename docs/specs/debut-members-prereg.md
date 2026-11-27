# Pre-registration: newcomer and maiden members at the final Win price (#438, N7)

Status: **frozen** in the Japan run registry, without a sign-off stop, per the owner's reframe on
#438 (2026-11-27) under the problem statement's D15-D19 as amended: Claude pre-registers the spec
and freezes it before the run, and the run follows with no result gate. Claude froze it before the
run read any store row. The frozen JSON is `debut_members.build_spec(seed=438)` in
`src/horsetrack/countries/japan/studies/debut_members.py`, registered as family
`japan-debut-maiden-members-v2` with spec sha256
`55bf539436caf585b84eccccc49be9b1c1f9b66aea1cbcc81e727a5bc85e44d3`. This page describes that JSON.
If the two ever differ, the JSON is the record. The registry refuses a run whose spec differs by
one byte, so a change of method needs a new family.

This study makes its modeling choices here, not in `Ruling:` lines.

## Question

Does any member that could carry signal for a debut or maiden runner add to the final Win price in
newcomer (DEBUT) and maiden (MAIDEN) races? This tests the owner's hypothesis: "Probably in
maidens and debuts versus the big G1 races."

- **E34**, as #438 reads 08 §4: members carry information the final Win price lacks for runners
  with thin history, and this is concentrated in debut and maiden races. The members tested are
  the pooled ability, the workouts, body weight and the late Win path, one at a time and
  together.
- **E37**: a pedigree prior (sire × surface × distance band, from earlier runs only) adds to the
  final Win price in debut and maiden races.
- **Null**, the starting point: every member weight is 0, so the final Win price already holds
  what the member says about a debut or maiden runner (ΔLL 0, encompassing weight 0).

Nothing is adopted or rejected (D16-D19). Each variant is a ranked row, with its encompassing
weight beside it.

## Races and decision time

- The JRA store's run races of `race_class` DEBUT or MAIDEN (the store's codes; `research/pool_scan`
  calls them the newcomer and maiden groups). They are taken on development days from 2014-06-07
  (D10), from races with a priced winner, with every priced starter. Holdout days are never read,
  and the holdout is not opened (D11-D15).
- The target is the race's winner in the Win pool. A dead heat's winners count at 1/k each (D6).
- The decision time is t = post_utc − 2 min (D2's primary, `config/decision.yaml`). post_utc is
  `races.post_time_local` on the JST wall clock of the race date (#227).

## Members

| Member | Ticket | Status | Features (`<member>__<name>`) | Known at t when |
| --- | --- | --- | --- | --- |
| `ability` | #92 | built | `u`, `se_u` | the card is published, 00:00 JST of race day |
| `pedigree_prior` | #438 (moved from #95) | built | `prior` | the card is published, 00:00 JST |
| `workouts_jrdb` | #137 | built (private, D21) | `ff_best_vs_par_s`, `ff_trend_s`, `days_since_hard`, `course_kinds_since_last_race` | strictly after (race day − 1) 20:00 JST |
| `workouts_jravan` | #229 | built | `ff_best_vs_par_s`, `ff_trend_s` | the newest workout used is dated before race day and published strictly before t |
| `jvdata_mining` | #406 | **missing** | none | listed and left out; it is added as a counted variant when #406 lands |
| `body_weight` | #227 | built | `horse_wt_kg`, `horse_wt_change_kg` | post_utc − 60 min (#227, `config/availability.yaml`) |
| `late_win` | #438 (0B41, #369) | built | `ln_share_move_t20_t2`, `ln_share_move_t5_t2` | post_utc − 2 min |

- **ability**: #92's per-runner u and its Laplace σ (`se_u`), out of fold, from the export that
  the latest scored record of `japan-c2-ability-final-win-v2` names. The model is pooled jockey,
  trainer and sire on #91's rating. It is out of fold from 2015-Q1 and missing before that.
- **pedigree_prior**: the runs are every not-scratched starter of the JRA store's run races from
  2010-01-01 (results only, no price). Each run's sire is the JRA-VAN `horse_pedigree.sire_id`
  (#92's source), and the field is the race's starters. A run scores the share of its field it
  beat, (field − finish) / (field − 1). A run without a placing, a field under 2 and a jump race
  are not counted. A cell is sire × surface (turf, dirt) × distance band: sprint up to 1400 m,
  mile up to 1800 m, intermediate up to 2200 m, long beyond. The prior is
  logit((sum + 10 × 0.5) / (runs + 10)) over the cell's runs strictly before the race day, which
  is 0 when there are none. A runner without a sire, a distance or a turf or dirt surface has none.
- **workouts_jrdb**: #137's `workout_features` at race morning. These are identical at t, because
  #137's CHA and CYB records count from (race day − 1) 20:00 JST.
- **workouts_jravan**: #229's `ff_best_vs_par_s` and `ff_trend_s`. The par is the median final
  furlong per centre and course over every workout dated before race day, with at least 30 of
  them. The features use the newest 3 workouts with a par, from the horse's workouts dated in
  [race day − 365, race day − 1]. They are computed in bulk per race day with #229's own best and
  trend code.
- **body_weight**: the stored race-day weight and its change.
- **late_win**: ln s(T−2) − ln s(T−m) for m = 20 and m = 5. s is the Win board's normalised 1/odds
  at T−x, taken from the day's JRA-VAN `odds_timeseries` file through #434's `boards_at`. The
  board at T−x is each runner's latest WIN row available (`fetched_utc`, else `snapshot_utc`) at
  or before the study's own post_utc − x, where post_utc is the JRA store's (the decision time's
  post above). The file's own `post_utc` and `minutes_to_post` are JRA-VAN's post and never set a
  cutoff: with JRA at 12:00 and JRA-VAN at 12:02, the 11:59 snapshot is not on the T−2 board. The
  value is missing when either board lacks the runner or the race has no post time.

## The fit

- **Offset**: ln q, where q is the final Win odds' 1/O normalised within the race, with
  coefficient 1 (D1). All weights 0 gives the market exactly.
- **Design**, per training sample: each feature is standardised by the mean and sd of its known
  values. A missing value sits at 0, beside a missing flag whenever a training row lacks it (a
  flag that repeats another is dropped). A column that varies within no training race is dropped.
- **Model**: the conditional logit u = ln q + X w (`models.condlogit`), with ridge 1e-3 on w. It is
  fitted per class (DEBUT, MAIDEN) and per D9 fold, on the fold's training rows of that class,
  for each built member alone and for every built member together (`combined`).
- **Folds**: D9 (`config/folds.yaml`): quarters, expanding, a 1-week embargo and 22 quarters
  minimum, walked forward over the study's race days. Every fold's test quarter is scored.

## Variants

There are 14, one registry record each, with failures included: `member:<m>:<class>` for each of
the six built members and `combined:<class>`, for class DEBUT and MAIDEN.

## Metrics

- **Gain**: the per-race winner ΔLL, ll(q) − ll(model), out of fold over the scored races, with
  D7's meeting bootstrap at 95% one-sided (10,000 replicates, seed 438).
- **Weight**: each design column's weight, fitted on every scored row of the class (in sample),
  with the meeting-clustered sandwich se and one-sided 95% bounds.
- **Encompassing weight**: the Fair-Shiller weight a of the variant's out-of-fold ln p in
  u = a ln p + b ln q over the scored rows (`models.meta.encompassing`), meeting-clustered, with
  one-sided bounds. It is a column beside each ranked row.
- **N5** (`core.eval.tail.tail_curve`): the claim is p (out of fold) × the final Win odds for each
  scored runner, at flat stakes. The quantiles are 0.5, 0.8, 0.9, 0.95 and 0.99, and the curve is
  settled at Win. Place waits for #410.
- **Settlement** (`settle_win`): a winner returns its own WIN dividend row / 100 (a refund row is
  no dividend), each winner of a dead heat its own, and a loser 0. A race any of whose winners (a
  dead heat's too) has no WIN dividend row is excluded from the N5 curve: it is not settled at the
  final odds, nor by another pool's row, and it is not a loss. It still scores ΔLL, and its
  ranked row counts such scored races as `n5_excluded_races`, in the private report only.
- **Ranking**: the ranked rows order the variants by their ΔLL. Each row carries its bounds,
  encompassing weight and top N5 band.

## Outputs

- Private, under `$HORSETRACK_DATA_DIR/reports/` through `write_report` with the sources `jra`,
  `jravan` and `jrdb`: the report `debut-members-japan-<spec hash>-<run digest>.md`, its JSON
  beside it, and the ranked rows as `candidate-cells/<the same stem>.json`. The run digest is a
  hash of the run's date range and input builds, so a later run never overwrites an earlier one.
- Registry: one record per planned variant, failures included (a failed read fails every variant
  and is recorded too), each naming the sources `jra`, `jravan` and `jrdb`, so the committed
  summary withholds every value. The command exits nonzero when no variant scored.
- Public: `docs/experiments/debut-members-japan.md` (`render_public_report`), which shows no
  figure (D21).

## D21: where the figures go

Every figure of this study is built on JRA-VAN or JRDB data, or comes from the run that reads
them, so every figure is withheld. A numeric result reaches a file only two ways, both under the
data directory and outside the tree:

- through `write_report`, which refuses the repository: the private report, its JSON and the
  candidate-cell file;
- through the run registry, `default_registry("japan")` at
  `$HORSETRACK_DATA_DIR/registry/japan.jsonl`. It is the private record of every run that
  CLAUDE.md requires, failures included, and each record carries its variant's metrics, as every
  study's records do (`studies/ability.py`, `meta.py` and `calibration.py` among them) and as the
  planned leaderboard reads them. Every record names a licensed source, so `write_summary` shows
  each one as withheld in the committed summary.

The public report, the ledger row and this page carry no count, rate, span, sign or bound.

## v1, kept and superseded

Family `japan-debut-maiden-members-v1` (its frozen spec, its records and its committed summary)
stays in the record, superseded by v2. v1 cut the late Win boards by the day file's
`minutes_to_post`, which is measured to the JRA-VAN post rather than the study's JRA post, so a
snapshot after the decision time could enter the T−2 board when the two posts differ. v1 also
settled a race whose winner had no WIN dividend row at the final odds in the N5 curve. v2 fixes
both, as above; nothing else in the method changed.

## Owner decisions applied

D1 (the final price as conditioner), D2 (T−2), D6, D7, D9, D10, D11-D15 (holdout A2 from
2026-11-28, fully locked, not opened), D16-D19 (ranked rows, nothing adopted), D21 (every figure
withheld), and the owner's reframe of #438 (2026-11-27): no result gate and no sign-off stop; a
member whose ticket has not landed is listed and then added as a counted variant; each JRDB
feature is a ranked variant, with private results.
