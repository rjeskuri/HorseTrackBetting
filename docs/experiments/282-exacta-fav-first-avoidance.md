# Experiment report: fav_first exacta loss avoidance, family L (#282)

Family `japan/282-exacta-fav-first-avoidance-v1`, superseded by v2 (last section) (D18's loss-avoidance rule; spec
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md), family L). The owner
approved the family as drafted on #282 on 2026-11-26 and reframed it on 2026-11-27 (every cut is a
ranked row, both the kept and the skipped set are reported, no 100-meeting gate). Frozen in the
Japan run registry on 2026-11-27, before any scored run, with hash
`8ae769a7a9767f962d100d959b6567f09b8f8e154c282978d19fa018ece0434f`. Registry hypothesis id `E43`;
every record is in
[registry/japan/japan--282-exacta-fav-first-avoidance-v1.md](registry/japan/japan--282-exacta-fav-first-avoidance-v1.md).
Code: `countries/japan/research/exacta_loss_avoidance.py`
(`python -m horsetrack.countries.japan.research.exacta_loss_avoidance {freeze|score|summary}`),
which reuses #131's `exacta_study` and `exacta_consistency`. Stacked on #280. Nothing is adopted
and no rule is staked; each member becomes a leaderboard row once the leaderboard writer (#449)
lands.

## Hypothesis

Stated before scoring (spec section 5, family L): at T−2, skip a fav_first exacta c when the
pool's implied share is more than k times the Harville probability from the Win board,
`q_c(t) / p_c(t) > k`, with k in {1.1, 1.2, 1.3, 1.5}, in all races or in the owner's headline cell
(favourite at 3.0 or more at T−2 in a less popular race). If the pool overloads the favourite's
combinations, the kept set returns more per ¥ than buying every fav_first combination at ¥100, and
the skipped set returns less. The comparator is that all-fav_first purchase on the same races; the
metric is return per ¥ at the final dividend, kept minus comparator, with a meeting-cluster
bootstrap (95% one-sided bounds in each direction) and a max-statistic over the 8 members.

The null, the starting point: the rule changes nothing, the kept set's difference from the
comparator is 0. What would show the rule wrong: no member's kept-minus-all lower bound, widened by
the max-statistic over the 8 members, clears 0. Note the limit of the question: removing
combinations priced above the Harville line can only help if the skipped ones return less than the
kept ones; both are reported, so a rule that "avoids losses" and one that merely shrinks the stake
can be told apart. The kept and the skipped set are each compared with the comparator.

## Variants tried

All 8 members (family size 8), each on the same logged development races, run 2026-11-27 at git
`01f3ca1`:

| Cut k | Scope | Races | Meetings | Kept: per ¥ (diff, 95% one-sided) | Skipped: per ¥ (diff, 95% one-sided) |
| --- | --- | --- | --- | --- | --- |
| 1.1 | all | 0 | 0 | - | - |
| 1.1 | headline | 0 | 0 | - | - |
| 1.2 | all | 0 | 0 | - | - |
| 1.2 | headline | 0 | 0 | - | - |
| 1.3 | all | 0 | 0 | - | - |
| 1.3 | headline | 0 | 0 | - | - |
| 1.5 | all | 0 | 0 | - | - |
| 1.5 | headline | 0 | 0 | - | - |

**Logged development meetings with `FORECAST` (exacta) boards: 0.** The run read 2 development day
folders (2026-11-21 and 2026-11-22, Kyoto), 0 holdout folders (the A2 holdout starts 2026-11-28;
`LockedHoldout` not opened). Their logs hold `WIN`, `PLACE`, `QUINELLA` and `QUINELLA_PLACE` only:
both days predate #130's exacta logging, as #281 found for E2. So n = 0 races and 0 meetings,
nothing excluded, and no interval exists. Each member is a registry record (status `scored`,
metric `n_races` = 0, no difference metric), the data-pending result. No variant failed and none
was dropped.

Nothing here is measured, so the ledger has no measured row for family L. It has one assumption: ¥100
per combination has negligible own impact (spec section 7), against which logged exacta pool totals
can be checked once they exist.

A rerun is the same 8 variants on more meetings, not new variants: `score` re-scores everything
logged each time, and a rerun on the same inputs appends nothing.

## Alternatives rejected and why

- Waiting for 100 logged meetings before scoring (the draft's gate): superseded by the owner's
  reframe of 2026-11-27 (rank everything, keep every result, no gates).
- A favourite and categories from final odds: look-ahead (spec section 5, "selection uses only the
  board at t"). Selection reads only the Win and exacta boards at T−2, a row available at its
  `fetched_utc`, else its `snapshot_utc`.
- The exacta pool total at the final board for the "less popular" tertile: it would let the scope
  read money bet after the selection time. The total at T−2 is used (a reading of spec section 6,
  which does not name the instant); #281 uses the final total for E2's proxy check.
- Own impact for the kept set: a 100-yen stake on each of a handful of combinations moves the
  dividend little; it is a ledger assumption, applied identically to kept, skipped and comparator.
- Fitting the cut: the four cuts are the approved grid. Any cut added later is a new, counted
  variant, not a replacement.

## What this rules out

Nothing. With 0 logged `FORECAST` meetings no member is measured, so family L is neither ruled in
nor ruled out. #131's E1 result (no favourite overload visible in the winner's price on history)
stands alone. When logged days exist, a kept-minus-all bound that does not clear 0 after the
max-statistic over 8 members rules out an edge from this selection no larger than the interval's
width, and says nothing about other cuts.

## Follow-ups filed

- None filed by this run. Re-run `python -m horsetrack.countries.japan.research.exacta_loss_avoidance
  score` once development days with #130 `FORECAST` boards are logged (holdout days from 2026-11-28
  are never read), then regenerate the registry summary.
- The loader's logged-board reader duplicates E2's (#281, `exacta_rules.py`); merge them once both
  have landed.

## v2: the race-card headline cut (2026-11-28)

Family `japan-282-exacta-fav-first-avoidance-v2`, frozen 2026-11-28 with hash
`9309ef5eb7b105e125b2c9d6513a5badab83b122a629b0d8cfe30b2e3919c89a`, before its run. It supersedes
`japan/282-exacta-fav-first-avoidance-v1` (hash `8ae769a7...0434f`, above), which stays in the
registry as history: frozen 2026-11-27 and scored once on 0 logged exacta meetings, so its 8
records are n = 0 and nothing else is lost. Records:
[registry/japan/japan-282-exacta-fav-first-avoidance-v2.md](registry/japan/japan-282-exacta-fav-first-avoidance-v2.md)
(v1's stay in the file linked at the top). The spec carries the decision (section 5, family L);
the project's governance lets a study's choices be pre-registered in the frozen spec and variants
be added and counted without sign-off.

**Why v2.** v1's headline cell took "less popular" from the lower tertile of each logged day's
exacta pool totals at T−2. A whole-day tertile ranks an early race by later races' totals, which
is look-ahead for a decision at T−2; #131's review found the same in E2's rule-v3 and E2 moved to
the race card in rule-v4 and v5. Main's registry also refuses a `/` in a new family id. v2 changes
the headline cut and the id, and nothing else: the 4 cuts, 2 scopes, comparator, metric, bootstrap
(meeting clusters, 95% one-sided, max-statistic over the 8 members, seed 282), λ2 and exclusions
are v1's.

**The v2 headline.** The T−2 favourite's Win odds ≥ 3.0 and a less popular race by the race card:
`race_no` 1–4, or a DEBUT or MAIDEN race (`race_class` from the store's races table, as #131's
E2). Race number and class are known before the day and the favourite's odds are from the T−2
board, so no pool total enters a selection. A race with no class in the store counts as less
popular by its race number alone. The lower tertile of pool totals is reported, per run, as the
proxy's agreement diagnostic only (`less_popular_agreement`); a race with no logged pool total is
scored all the same.

**Result.** Run 2026-11-28 at git `ee33fb4`, all 8 members, on the logged development days:

| Cut k | Scope | Races | Meetings | Kept: per ¥ (diff, 95% one-sided) | Skipped: per ¥ (diff, 95% one-sided) |
| --- | --- | --- | --- | --- | --- |
| 1.1, 1.2, 1.3, 1.5 | all | 0 | 0 | - | - |
| 1.1, 1.2, 1.3, 1.5 | headline | 0 | 0 | - | - |

**Logged development meetings with `FORECAST` (exacta) boards: 0.** The run read 2 development
day folders (2026-11-21 and 2026-11-22, Kyoto), 0 holdout folders (`is_holdout_day` is asked per
day before a folder is opened; `LockedHoldout` not opened). Both days predate #130's exacta
logging, so n = 0 races and 0 meetings, nothing excluded, and the agreement diagnostic has no
race (0 with a pool total). Each member is a registry record (status `scored`, metric `n_races` =
0, no difference metric): the data-pending result, as v1's. 9 records were appended for v2, the
freeze and the 8 members; none failed. Nothing is measured, so the ledger has no measured row for
family L.

**What this rules out.** Nothing, as for v1. The null stays the starting point.

**Follow-ups.** Re-run `score` once development days with #130 `FORECAST` boards are logged
(race days from 2026-11-28 on can be holdout days and are never read); a rerun on the same
inputs appends nothing, a rerun on more meetings re-scores the same 8 variants. Regenerate the
registry summary after each run.
