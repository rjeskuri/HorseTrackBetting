# Experiment report: #131's E2 exacta ratio-cut rules on logged boards (#281)

Scores the 16 frozen members of `japan/131-exacta-rule-v2` (E2; spec
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md) sections 5-7, frozen
hash `2d6678f59291476bf3d3028b2818bc00e631dea516bc0f01276ac4e0af8cdb75`, the hash of #280's final
head) on the #130-logged development meetings that carry exacta (`FORECAST`) boards. Registry
hypothesis id `E43`; every record is in
[registry/japan/japan--131-exacta-rule-v2.md](registry/japan/japan--131-exacta-rule-v2.md).
Code: `countries/japan/research/exacta_rules.py`, command `uv run horsetrack japan exacta-rules`,
which re-scores everything logged so far on each run (another run of the same 16 variants, not
new variants). Nothing is adopted; no rule is staked. Leaderboard rows wait for #449.

**Small n is not refused.** The frozen spec says E2 is scored "only once there are min_meetings"
(`min_meetings: 100`) and is "not yet scorable" until then. The owner's reframe of 2026-11-27
(rank everything, keep every result, no gates) supersedes that line: E2 is scored on whatever is
logged, and zero meetings give an n = 0 result with its records.

## Hypothesis

Stated before scoring (spec section 5): at T−2, buying exacta c of category X (favourite second,
or without the favourite) when the pool's implied share is below k times the Harville probability
from the Win board, `q_c(t) / p_c(t) < k` with k in {0.6, 0.7, 0.8, 0.9}, returns more than it
stakes at the final dividend after own impact, in all races or in the owner's headline cell
(favourite at 3.0 or more at t in a less popular race). The null, the starting point: no member
beats no bet (loss 0); SPA over the 16 members does not reject and StepM finds none.

## Run of 2026-11-27 (git `7ae109e`)

- Inputs read: 2 development day folders (2026-11-21 and 2026-11-22, Kyoto), 2 odds logs, 0
  holdout folders opened (the A2 holdout starts 2026-11-28; `LockedHoldout` not opened).
- **Logged development meetings with FORECAST boards: 0.** Neither log holds a single `FORECAST`
  row (they hold WIN, PLACE, QUINELLA and QUINELLA_PLACE only): both days predate #130's exacta
  logging, as #283's own-impact run found (ledger). So n = 0 races, 0 meetings, 0 meetings with
  every row (Win and exacta board at t, final boards, pool total, dividend), 0 races excluded.
- Registry: 16 records under `japan/131-exacta-rule-v2`, status `scored`, each with
  `spec_hash` equal to the frozen hash, metric `n_bets` = 0 with n = 0 and no `roi` metric;
  family size 16. Run ids (first 12 characters, in member order): `1b511e6a65c1`,
  `84856745d971`, `178e47c46007`, `ec1e7737ef25`, `0e29555cdccc`, `c0e67537ad70`,
  `da25f54e1f2f`, `185408d41457`, `2c7fa73e3252`, `4c928d620314`, `6f8a89196fed`,
  `fdde3c5310a4`, `d2c18811226f`, `b6fc9ccb06f0`, `5279b4eaea3a`, `252d21af2697`. Ordering
  component `2b822d643cf7` (#115's full-development fit, λ2 = 0.8104).
- The max test (SPA + StepM, block 5 meetings, meetings as clusters, 10,000 replicates, seed 131)
  did not run: it runs when any race is scored, and none was. No `/max-test` record.

## Variants tried

All 16 members, each on the same 0 races (development days only):

| Category | Cut k | Scope | Bets | Races | Meetings | ROI | One-sided 95% bounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fav_second | 0.6 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.6 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.7 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.7 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.8 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.8 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.9 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.9 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.6 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.6 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.7 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.7 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.8 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.8 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.9 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.9 | headline | 0 | 0 | 0 | - | - |

No member failed; none bet. ROI is total payout over total stake; its interval, once there are
bets, is a meeting bootstrap with a one-sided 95% bound in each direction (as #131's E1).

- **D, the full-board ratio** `Σ_S q / Σ_S p` on the final boards: 0 races in each of
  fav_first, fav_second and no_fav; no mean or median.
- **"Less popular" proxy against the pool tertile:** 0 races on 0 days; no agreement figure.
- **Shrinkage (`core/eval/segment_roi.py`, spec section 7): not run.** With no bet there is
  nothing to shrink; the command does not run it yet (follow-up below).

## Readings of the spec, for the owner

Modelling items the frozen text leaves open, implemented as read here; none is a ruling.

- `p_c(t)`: the normalised 1/odds of the Win board at t, through discounted Harville at #115's
  full-development λ2 (the rule's text, "Harville on the Win board at t"). #114's calibrated
  `p_cal` (the spec's common `win_probabilities`) does not exist for logged days.
- Own impact: the exact payout identity `100 (1 − t)(P + n s) / (W + s)`, `W = (1 − t) P / O`, with
  P the race's final logged FORECAST pool total, n the member's own number of bets in the race and
  t = 1 − the race's payout rate (0.25 on a race without a boost). No breakage or ¥110 minimum is
  re-applied after the impact: neither the spec nor #283 defines one.
- Each member is a standalone rule, settled with its own bet count (a 0.9 cut buys more
  combinations than 0.6, so its own impact is larger).
- "Less popular" on logged days: the races at or below the 1/3 quantile (linear interpolation) of
  the final FORECAST pool totals of each race day, all venues of the day together.
- Headline scope: favourite's Win odds at t ≥ 3.0 and the frozen proxy (race 1-4, or a newcomer
  or maiden race); the tertile is reported beside it, not used in the scope.
- A board price of 1.1 is ¥110 (float noise snapped), so the ¥110 floor leaves its share
  unidentified and the combination is never bought.
- SPA and StepM run over all 16 members whenever a race is scored, members that bet nothing
  included (snooping requires the full family); with fewer than 3 meetings arch cannot run and
  the failed max test is recorded under `japan/131-exacta-rule-v2/max-test`.

## Alternatives rejected and why

- Waiting for 100 meetings before scoring (the frozen `min_meetings`): superseded by the owner's
  reframe of 2026-11-27.
- Selecting on the final boards: look-ahead (spec section 5); selection uses only the boards at t,
  with a row available at its `fetched_utc`, else its `snapshot_utc` (docs/specs/board-at-t.md).
- One shared bet count across members for own impact: it would charge a narrow rule for a wide
  rule's stakes.

## What this rules out

Nothing. With 0 logged FORECAST meetings no member is measured, so E2 is neither ruled in nor
ruled out, and #131's E1 result (no favourite overload on history) stands alone.

## Follow-ups filed

- None filed by this run. Re-run `uv run horsetrack japan exacta-rules` once development days
  with #130 FORECAST boards are logged (holdout days from 2026-11-28 are never read); the
  `segment_roi` shrinkage of E2's bets (spec section 7) is left for that run's ticket.
