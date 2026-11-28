# Experiment report: exacta favourite overload (#131)

The pre-registered study of whether the JRA exacta pool (`FORECAST`) overloads the favourite's
combinations relative to the Win pool and an ordering model. Spec:
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md), approved by the owner
on 2026-11-25 and frozen in the Japan run registry before any scored run. Registry hypothesis id
`E43`. E1's members are ranked by their shrunk return below; every E1, E1′ and E1″ member's value
is in [131-exacta-e1-members.md](131-exacta-e1-members.md) and every registry record in
[registry/japan/](registry/japan/README.md), one file per family, its "/" written as "--":
[japan--131-exacta-category-v2.md](registry/japan/japan--131-exacta-category-v2.md),
[japan--131-exacta-category-b1-v2.md](registry/japan/japan--131-exacta-category-b1-v2.md),
[japan--131-exacta-category-b3-v2.md](registry/japan/japan--131-exacta-category-b3-v2.md),
[japan--131-exacta-category-v2--max-test.md](registry/japan/japan--131-exacta-category-v2--max-test.md)
and [japan--131-exacta-logscore-v2.md](registry/japan/japan--131-exacta-logscore-v2.md), and the
v1 files of the same names with `v1`; the post-hoc breakage diagnostic is
[japan--131-exacta-breakage.md](registry/japan/japan--131-exacta-breakage.md). The rule families
are frozen, not scored, so they have no file. Nothing is adopted; no rule is staked.

The results name the v2 families (`japan/131-exacta-*-v2`, frozen 2026-11-27). v2 supersedes the
v1 families of 2026-11-25 only for the λ2 label: the owner, 2026-11-26 on #131, "Relabel λ2
honestly: it is #115's single full-development fit (partly in-sample), not 'in_fold'". The v2
re-run gave v1's values in every record (Variants tried).

Family ids (#525, owner 2026-11-27): the registry now refuses a `/` in a family id, so the live
families are `japan-131-exacta-category-v2`, `-category-b1-v2`, `-category-b3-v2`,
`japan-131-exacta-logscore-v2`, `japan-131-exacta-rule-v4`, the max test
`japan-131-exacta-category-v2-max-test` and the breakage diagnostic `japan-131-exacta-breakage`,
with the same specs and hashes. The results below were recorded before the rename and stay under
the legacy `japan/131-…` ids named with them; a re-run writes to the new ids.

Re-run under the new ids (2026-11-28, git `ec331f7`, Sol review round 5 of 2c73ca0): E1, E1′,
E1″, C, the max test and breakage were scored again into
[japan-131-exacta-category-v2.md](registry/japan/japan-131-exacta-category-v2.md),
[japan-131-exacta-category-b1-v2.md](registry/japan/japan-131-exacta-category-b1-v2.md),
[japan-131-exacta-category-b3-v2.md](registry/japan/japan-131-exacta-category-b3-v2.md),
[japan-131-exacta-logscore-v2.md](registry/japan/japan-131-exacta-logscore-v2.md),
[japan-131-exacta-category-v2-max-test.md](registry/japan/japan-131-exacta-category-v2-max-test.md)
and [japan-131-exacta-breakage.md](registry/japan/japan-131-exacta-breakage.md). Every ρ, bound,
shrunk ρ, C value, SPA p and breakage value equals the legacy records' to the printed digit; the
re-run adds the return per ¥100 of each B2 and B1 member (the ranked table). E2's live family is
now `japan-131-exacta-rule-v5` (frozen 2026-11-28, superseding `japan-131-exacta-rule-v4`, which
is history: E2 below).

Leaderboard: every E1 and E2 member becomes a leaderboard row (option, family, pool, segment, n,
expected return, CI, variants searched) once the leaderboard writer (#449) lands. This report
does not build one; the ranked table below is E1's part of it.

## Hypothesis

Stated before scoring (spec sections 1, 4 and 5):

- The exacta pool overloads favourite-first combinations: the Harville-weighted box of the
  favourite-first exactas returns less than the race's payout rate, ρ_fav_first < 1.
- So favourite-second and no-favourite combinations are underloaded: ρ > 1 for each, most of all
  in the owner's headline cell (favourite at 3.0 or more in a less popular race).
- The null, the starting point: the pool prices every category correctly, ρ_S = 1 for every S.
- What would show it wrong, as pre-registered: SPA over E1's 84 members fails to reject at 5%,
  and StepM finds no member. A betting edge would further need ρ_S > 1/rate (about 1.33).
- Reframed by the owner on 2026-11-27: every E1 member is ranked by its shrunk return with its n
  and CI, and SPA, StepM and the family size of 84 are reported beside the ranking as the
  variants-searched context, not as a pass/fail test.

## Data

- Development race days only: every day passed `is_holdout_day("japan", d)` before any outcome,
  odds or dividend was read, the readers took only the remaining race ids, and `score_races`
  drops a holdout day again before touching its exacta rows. No race day on or after 2026-11-28
  was read; `LockedHoldout` was not opened.
- Public JRA data only (registry `sources: ["jra"]`): the store's races, runners (final Win odds)
  and `FORECAST` dividends, and #114's out-of-fold Win probabilities (fit `aac501f927b0`), which
  are built on the final Win odds. No JRDB or JRA-VAN input.
- Precondition (row counts, 2026-11-25): 43,058 development races from 2014-06-07, each with a
  `FORECAST` row (43,206 rows). #114's out-of-fold file covers 23,760 of them, its walk-forward
  test quarters from 2020-Q1, so the scored span is 2020-01-05 to 2026-11-15.
- Ordering: discounted Harville at #115's full-development fit (λ2 = 0.8104, partly in-sample;
  fit `2b822d643cf7`; registry label `full_development`), and plain Harville (λ2 = 1). #115's λ2
  is one fit over the development out-of-fold races, the scored races among them, so it has seen
  their first two places. It is not the "in-fold fitted" λ2 the owner signed (spec answer 8): a
  deviation the owner accepted on 2026-11-26, asking for the `full_development` label (spec,
  "Deviations from the signed spec"). v1 called the same fit `in_fold`.
- What the deviation touches: B3 (E1″) and C at λ2 = 0.8104 are partly in-sample. B2's null
  (ρ = 1 under π = q, before rounding) does not depend on p, so E1's ranking and its SPA and StepM
  context stand at any λ2.
- Scored span: 2020-01-05 onward (23,629 of the 43,058 development races), not the signed
  2014-06-07 onward, forced by #114's out-of-fold span; owner-accepted on 2026-11-26 ("the scored
  span from 2020-01-05 stands as documented").
- Payout rate per race from `takeout.race_takeout(..., "FORECAST")`.
- Exclusions: 84 races with a possibly withheld Premium boost, 46 dead heats for first or second,
  1 race with fewer than 5 starters, 0 refunds. Scored: 23,629 races, 1,980 meetings.
- Run (v2): git `5d68473`, 2026-11-27, members, C and the max test in one run; 86 s and 0.55 GB
  peak. The store's parquet files had been rebuilt since v1 (new digests in the records'
  `data_builds`); the scored values did not change. v1: git `4271171` (members, C) and `9910a7c`
  (the max test), 2026-11-25.

## Results

### E1 ranked by shrunk return

E1 (B2), the primary estimand ρ_S: the Harville-weighted box's total return over the races' total
payout rate, null 1. A ¥1 box returns about ρ × 0.75 at the standard exacta payout rate, so a
profit needs ρ above about 1.33. All 84 members are ranked by the empirical-Bayes shrunk ρ (shrunk
across the 14 cells of its category and λ2; ties broken by the raw ρ). The bounds are one-sided
95% meeting-bootstrap bounds (10,000 replicates, seed 131); together they form a two-sided 90%
interval. They are not adjusted for the 84 members searched: that is the context below.

The last three columns (added 2026-11-28, git `ec331f7`) give the governance metric beside ρ:
the return per ¥100 staked at the final dividend, `100 · Σ_r R_r / n`, the gross yen the box
returns per ¥100 staked on it in each of the cell's n races (one box unit per race, takeout and
breakage in), with its one-sided 95% meeting-bootstrap bounds (10,000 replicates, seed 131). It
is a registry metric (`return_per_100`), not a param, so the family stays at 84. ρ stays the
pre-registered estimand; the ranking is unchanged. A profit needs more than ¥100.

> **Final-Win-odds look-ahead.** Every row of this table, like every E1, E1′ and E1″ result in
> this report, is a historical selection made on the final Win board: B2's weights (#114's
> `p_cal`, built from final Win odds), the favourite (hence the category) and the
> favourite-odds bands are all taken from it, and none of them is known when an exacta is
> bought. These are not returns of a bet placeable before the off; the conclusions below are
> limited to that historical selection (spec section 4). The decision-time test is E2
> (`japan-131-exacta-rule-v5`), whose inputs all come from the boards at T−2.

| Rank | λ2 | Category | Cell | Registered direction | n races | Meetings | Shrunk ρ (EB) | ρ | 95% lower | 95% upper | Return per ¥100 | ¥100 lower | ¥100 upper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 (plain) | fav_first | fav_odds:2.0-2.9 | ρ < 1 | 10,375 | 1,980 | 1.0297 | 1.044 | 1.019 | 1.068 | 78.4 | 76.5 | 80.2 |
| 2 | 1 (plain) | fav_first | class:win1-3 | ρ < 1 | 10,978 | 1,979 | 1.0296 | 1.043 | 1.017 | 1.068 | 78.3 | 76.4 | 80.2 |
| 3 | 1 (plain) | fav_first | field:11-14 | ρ < 1 | 7,939 | 1,930 | 1.0296 | 1.046 | 1.016 | 1.076 | 78.6 | 76.3 | 80.8 |
| 4 | 1 (plain) | fav_first | popularity:less | ρ < 1 | 11,070 | 1,980 | 1.0291 | 1.033 | 1.009 | 1.057 | 77.6 | 75.8 | 79.3 |
| 5 | 1 (plain) | fav_first | field:5-10 | ρ < 1 | 3,820 | 1,615 | 1.0290 | 1.037 | 0.999 | 1.074 | 77.9 | 75.0 | 80.7 |
| 6 | 1 (plain) | fav_first | headline | ρ < 1 | 2,923 | 1,572 | 1.0290 | 1.047 | 0.990 | 1.107 | 78.6 | 74.3 | 83.1 |
| 7 | 1 (plain) | fav_first | class:maiden | ρ < 1 | 10,559 | 1,980 | 1.0289 | 1.031 | 1.007 | 1.055 | 77.4 | 75.6 | 79.2 |
| 8 | 1 (plain) | fav_first | fav_odds:le1.9 | ρ < 1 | 5,449 | 1,860 | 1.0287 | 1.028 | 1.003 | 1.053 | 77.2 | 75.3 | 79.1 |
| 9 | 1 (plain) | fav_first | fav_odds:ge5.0 | ρ < 1 | 189 | 177 | 1.0287 | 0.844 | 0.582 | 1.122 | 63.3 | 43.7 | 84.2 |
| 10 | 1 (plain) | fav_first | all | ρ < 1 | 23,629 | 1,980 | 1.0286 | 1.028 | 1.011 | 1.044 | 77.2 | 75.9 | 78.4 |
| 11 | 1 (plain) | fav_first | popularity:rest | ρ < 1 | 12,559 | 1,979 | 1.0284 | 1.022 | 0.998 | 1.047 | 76.8 | 74.9 | 78.6 |
| 12 | 1 (plain) | fav_first | fav_odds:3.0-4.9 | ρ < 1 | 7,616 | 1,943 | 1.0282 | 1.010 | 0.974 | 1.045 | 75.8 | 73.2 | 78.5 |
| 13 | 1 (plain) | fav_first | field:15-18 | ρ < 1 | 11,870 | 1,965 | 1.0277 | 1.012 | 0.986 | 1.037 | 76.0 | 74.1 | 77.9 |
| 14 | 1 (plain) | fav_first | class:open | ρ < 1 | 2,092 | 1,655 | 1.0277 | 0.930 | 0.870 | 0.991 | 69.8 | 65.3 | 74.4 |
| 15 | `full_development` (0.8104) | fav_first | field:11-14 | ρ < 1 | 7,939 | 1,930 | 1.0255 | 1.044 | 1.014 | 1.073 | 78.4 | 76.2 | 80.6 |
| 16 | `full_development` (0.8104) | fav_first | class:win1-3 | ρ < 1 | 10,978 | 1,979 | 1.0255 | 1.042 | 1.017 | 1.067 | 78.2 | 76.3 | 80.1 |
| 17 | `full_development` (0.8104) | fav_first | fav_odds:2.0-2.9 | ρ < 1 | 10,375 | 1,980 | 1.0255 | 1.040 | 1.016 | 1.065 | 78.1 | 76.3 | 80.0 |
| 18 | `full_development` (0.8104) | fav_first | headline | ρ < 1 | 2,923 | 1,572 | 1.0255 | 1.037 | 0.980 | 1.096 | 77.9 | 73.6 | 82.3 |
| 19 | `full_development` (0.8104) | fav_first | field:5-10 | ρ < 1 | 3,820 | 1,615 | 1.0255 | 1.034 | 0.996 | 1.070 | 77.6 | 74.8 | 80.4 |
| 20 | `full_development` (0.8104) | fav_first | popularity:less | ρ < 1 | 11,070 | 1,980 | 1.0255 | 1.026 | 1.002 | 1.049 | 77.0 | 75.3 | 78.8 |
| 21 | `full_development` (0.8104) | fav_first | all | ρ < 1 | 23,629 | 1,980 | 1.0255 | 1.024 | 1.008 | 1.041 | 76.9 | 75.7 | 78.2 |
| 22 | `full_development` (0.8104) | fav_first | class:maiden | ρ < 1 | 10,559 | 1,980 | 1.0255 | 1.024 | 1.000 | 1.048 | 76.9 | 75.1 | 78.7 |
| 23 | `full_development` (0.8104) | fav_first | popularity:rest | ρ < 1 | 12,559 | 1,979 | 1.0255 | 1.023 | 0.999 | 1.047 | 76.8 | 75.0 | 78.7 |
| 24 | `full_development` (0.8104) | fav_first | fav_odds:le1.9 | ρ < 1 | 5,449 | 1,860 | 1.0255 | 1.022 | 0.997 | 1.047 | 76.7 | 74.9 | 78.6 |
| 25 | `full_development` (0.8104) | fav_first | field:15-18 | ρ < 1 | 11,870 | 1,965 | 1.0255 | 1.009 | 0.983 | 1.034 | 75.7 | 73.8 | 77.6 |
| 26 | `full_development` (0.8104) | fav_first | fav_odds:3.0-4.9 | ρ < 1 | 7,616 | 1,943 | 1.0255 | 1.008 | 0.973 | 1.043 | 75.7 | 73.1 | 78.3 |
| 27 | `full_development` (0.8104) | fav_first | class:open | ρ < 1 | 2,092 | 1,655 | 1.0255 | 0.935 | 0.875 | 0.997 | 70.2 | 65.7 | 74.8 |
| 28 | `full_development` (0.8104) | fav_first | fav_odds:ge5.0 | ρ < 1 | 189 | 177 | 1.0255 | 0.886 | 0.611 | 1.179 | 66.5 | 45.8 | 88.4 |
| 29 | 1 (plain) | fav_second | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0078 | 1.036 | 1.003 | 1.071 | 77.8 | 75.3 | 80.4 |
| 30 | 1 (plain) | fav_second | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0076 | 1.034 | 1.001 | 1.067 | 77.6 | 75.2 | 80.2 |
| 31 | `full_development` (0.8104) | fav_second | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0071 | 1.036 | 1.002 | 1.071 | 77.8 | 75.2 | 80.4 |
| 32 | `full_development` (0.8104) | fav_second | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0069 | 1.033 | 1.000 | 1.067 | 77.6 | 75.1 | 80.1 |
| 33 | 1 (plain) | fav_second | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0066 | 1.025 | 0.992 | 1.059 | 77.0 | 74.5 | 79.6 |
| 34 | `full_development` (0.8104) | fav_second | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0059 | 1.024 | 0.991 | 1.058 | 76.9 | 74.4 | 79.5 |
| 35 | 1 (plain) | fav_second | headline | ρ > 1 | 2,923 | 1,572 | 1.0058 | 1.056 | 0.984 | 1.128 | 79.3 | 73.9 | 84.7 |
| 36 | `full_development` (0.8104) | fav_second | headline | ρ > 1 | 2,923 | 1,572 | 1.0050 | 1.055 | 0.983 | 1.127 | 79.3 | 73.9 | 84.6 |
| 37 | 1 (plain) | fav_second | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0048 | 1.010 | 0.967 | 1.053 | 75.8 | 72.6 | 79.1 |
| 38 | 1 (plain) | fav_second | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0047 | 1.195 | 0.837 | 1.593 | 89.7 | 62.8 | 119.6 |
| 39 | 1 (plain) | fav_second | class:open | ρ > 1 | 2,092 | 1,655 | 1.0044 | 1.002 | 0.921 | 1.085 | 75.3 | 69.1 | 81.5 |
| 40 | 1 (plain) | fav_second | all | ρ > 1 | 23,629 | 1,980 | 1.0044 | 1.004 | 0.981 | 1.027 | 75.4 | 73.6 | 77.1 |
| 41 | `full_development` (0.8104) | fav_second | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0041 | 1.009 | 0.967 | 1.053 | 75.8 | 72.6 | 79.1 |
| 42 | 1 (plain) | fav_second | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0041 | 1.001 | 0.967 | 1.036 | 75.1 | 72.6 | 77.8 |
| 43 | `full_development` (0.8104) | fav_second | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0039 | 1.199 | 0.840 | 1.599 | 90.0 | 63.0 | 120.0 |
| 44 | 1 (plain) | fav_second | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0039 | 0.995 | 0.953 | 1.040 | 74.7 | 71.5 | 78.1 |
| 45 | `full_development` (0.8104) | fav_second | class:open | ρ > 1 | 2,092 | 1,655 | 1.0037 | 1.001 | 0.919 | 1.084 | 75.2 | 69.0 | 81.4 |
| 46 | `full_development` (0.8104) | fav_second | all | ρ > 1 | 23,629 | 1,980 | 1.0036 | 1.003 | 0.980 | 1.026 | 75.3 | 73.6 | 77.1 |
| 47 | 1 (plain) | no_fav | class:open | ρ > 1 | 2,092 | 1,655 | 1.0036 | 1.063 | 1.021 | 1.106 | 79.8 | 76.6 | 83.0 |
| 48 | 1 (plain) | no_fav | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0036 | 1.022 | 0.938 | 1.103 | 76.7 | 70.4 | 82.7 |
| 49 | 1 (plain) | no_fav | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0036 | 1.019 | 0.995 | 1.043 | 76.5 | 74.7 | 78.3 |
| 50 | 1 (plain) | no_fav | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0036 | 1.010 | 0.993 | 1.026 | 75.8 | 74.6 | 77.1 |
| 51 | 1 (plain) | no_fav | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0036 | 1.005 | 0.988 | 1.022 | 75.5 | 74.2 | 76.7 |
| 52 | 1 (plain) | no_fav | all | ρ > 1 | 23,629 | 1,980 | 1.0036 | 1.004 | 0.991 | 1.017 | 75.4 | 74.4 | 76.3 |
| 53 | 1 (plain) | no_fav | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0036 | 1.004 | 0.986 | 1.021 | 75.4 | 74.0 | 76.7 |
| 54 | 1 (plain) | no_fav | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0036 | 1.002 | 0.982 | 1.023 | 75.2 | 73.7 | 76.8 |
| 55 | 1 (plain) | no_fav | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0036 | 1.001 | 0.983 | 1.020 | 75.2 | 73.8 | 76.6 |
| 56 | 1 (plain) | no_fav | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0036 | 1.000 | 0.979 | 1.022 | 75.1 | 73.5 | 76.7 |
| 57 | 1 (plain) | no_fav | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0036 | 0.998 | 0.959 | 1.038 | 75.0 | 72.0 | 77.9 |
| 58 | 1 (plain) | no_fav | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0036 | 0.996 | 0.978 | 1.014 | 74.8 | 73.4 | 76.1 |
| 59 | 1 (plain) | no_fav | headline | ρ > 1 | 2,923 | 1,572 | 1.0036 | 0.986 | 0.958 | 1.013 | 74.0 | 71.9 | 76.1 |
| 60 | 1 (plain) | no_fav | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0036 | 0.972 | 0.935 | 1.010 | 73.0 | 70.2 | 75.9 |
| 61 | 1 (plain) | fav_second | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0035 | 0.992 | 0.953 | 1.031 | 74.5 | 71.6 | 77.5 |
| 62 | `full_development` (0.8104) | fav_second | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0033 | 1.000 | 0.966 | 1.034 | 75.1 | 72.6 | 77.7 |
| 63 | `full_development` (0.8104) | fav_second | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0031 | 0.995 | 0.952 | 1.039 | 74.7 | 71.5 | 78.0 |
| 64 | `full_development` (0.8104) | fav_second | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0027 | 0.991 | 0.952 | 1.031 | 74.5 | 71.5 | 77.4 |
| 65 | 1 (plain) | fav_second | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0025 | 0.962 | 0.911 | 1.014 | 72.2 | 68.4 | 76.1 |
| 66 | `full_development` (0.8104) | fav_second | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0017 | 0.961 | 0.910 | 1.014 | 72.2 | 68.4 | 76.1 |
| 67 | 1 (plain) | fav_second | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0014 | 0.977 | 0.945 | 1.010 | 73.4 | 71.0 | 75.8 |
| 68 | 1 (plain) | fav_second | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0013 | 0.973 | 0.938 | 1.008 | 73.0 | 70.5 | 75.7 |
| 69 | `full_development` (0.8104) | fav_second | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0006 | 0.977 | 0.945 | 1.009 | 73.3 | 70.9 | 75.8 |
| 70 | `full_development` (0.8104) | fav_second | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0005 | 0.972 | 0.938 | 1.007 | 73.0 | 70.4 | 75.6 |
| 71 | `full_development` (0.8104) | no_fav | class:open | ρ > 1 | 2,092 | 1,655 | 0.9971 | 1.062 | 1.022 | 1.104 | 79.8 | 76.7 | 82.9 |
| 72 | `full_development` (0.8104) | no_fav | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 0.9968 | 1.004 | 0.988 | 1.020 | 75.4 | 74.2 | 76.6 |
| 73 | `full_development` (0.8104) | no_fav | field:11-14 | ρ > 1 | 7,939 | 1,930 | 0.9967 | 1.011 | 0.988 | 1.034 | 75.9 | 74.2 | 77.7 |
| 74 | `full_development` (0.8104) | no_fav | popularity:rest | ρ > 1 | 12,559 | 1,979 | 0.9962 | 0.998 | 0.982 | 1.015 | 75.0 | 73.7 | 76.2 |
| 75 | `full_development` (0.8104) | no_fav | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 0.9961 | 1.035 | 0.952 | 1.116 | 77.7 | 71.5 | 83.8 |
| 76 | `full_development` (0.8104) | no_fav | field:15-18 | ρ > 1 | 11,870 | 1,965 | 0.9959 | 0.995 | 0.978 | 1.013 | 74.7 | 73.5 | 76.0 |
| 77 | `full_development` (0.8104) | no_fav | all | ρ > 1 | 23,629 | 1,980 | 0.9959 | 0.995 | 0.983 | 1.008 | 74.8 | 73.8 | 75.7 |
| 78 | `full_development` (0.8104) | no_fav | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 0.9957 | 0.986 | 0.948 | 1.024 | 74.0 | 71.2 | 76.9 |
| 79 | `full_development` (0.8104) | no_fav | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 0.9957 | 0.993 | 0.975 | 1.011 | 74.6 | 73.2 | 75.9 |
| 80 | `full_development` (0.8104) | no_fav | popularity:less | ρ > 1 | 11,070 | 1,980 | 0.9957 | 0.992 | 0.973 | 1.012 | 74.5 | 73.0 | 76.0 |
| 81 | `full_development` (0.8104) | no_fav | class:maiden | ρ > 1 | 10,559 | 1,980 | 0.9956 | 0.990 | 0.970 | 1.011 | 74.4 | 72.9 | 75.9 |
| 82 | `full_development` (0.8104) | no_fav | headline | ρ > 1 | 2,923 | 1,572 | 0.9953 | 0.980 | 0.953 | 1.007 | 73.6 | 71.6 | 75.6 |
| 83 | `full_development` (0.8104) | no_fav | field:5-10 | ρ > 1 | 3,820 | 1,615 | 0.9953 | 0.963 | 0.927 | 1.001 | 72.3 | 69.6 | 75.2 |
| 84 | `full_development` (0.8104) | no_fav | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 0.9952 | 0.987 | 0.970 | 1.005 | 74.2 | 72.8 | 75.5 |

Reading the ranked rows (the null result, as rows rather than a closed verdict):

- Every shrunk ρ lies between 0.995 and 1.030: every member returns within about 3% of the
  payout rate, so a ¥1 box returns about ¥0.75 to ¥0.77. The best-ranked member is about 0.30
  short of the 1.33 a profit needs, and no member's 95% upper bound reaches 1.33 except the
  favourite-at-5.0-or-more cell (189 races), whose bounds are too wide to say.
- The top 28 rows are all fav_first, at a shrunk ρ of 1.026 to 1.030: the favourite-first box
  returns slightly more than the pool-fair rate, the opposite of the registered direction
  (ρ < 1). fav_second follows at 1.000 to 1.008 and no_fav at 0.995 to 1.004, both close to 1
  rather than above it.
- Shrinkage pulls every cell to its category's mean: the between-cell variance is small next to
  the cells' bootstrap SEs. The raw ρ of a thin cell (fav_second, favourite at 5.0 or more:
  1.195 on 189 races) shrinks to about 1.004.
- Nominal one-sided bounds that exclude 1 in the registered direction: fav_first class:open
  (ρ 0.935, upper 0.997); fav_second class:maiden (1.036, lower 1.002); no_fav class:open (1.062,
  lower 1.022), the same three cells at both λ2, plus fav_second popularity:less at its edge
  (lower 1.000 at λ2 = 0.8104, 1.001 plain). Shrunk, they rank with their category: fav_first
  class:open at 1.026 to 1.028 (ranks 14 and 27, now above 1), fav_second class:maiden and
  popularity:less at 1.007 to 1.008 (ranks 29 to 32), no_fav class:open at 0.997 to 1.004 (ranks
  47 and 71).
- Per ¥100 at the final dividend (2026-11-28): every member returns between ¥63.3 and ¥90.0 per
  ¥100 staked. The "all" cells: ¥76.9 [75.7, 78.2] fav_first, ¥75.3 [73.6, 77.1] fav_second and
  ¥74.8 [73.8, 75.7] no_fav at λ2 0.8104 (¥77.2, ¥75.4, ¥75.4 plain). Every cell of more than
  2,000 races has its upper bound below ¥85 (the highest, fav_second headline, ¥79.3 with upper
  bound ¥84.7); only the favourite-at-5.0-or-more cell (189 races) reaches past ¥100 (fav_second,
  upper bound ¥119.6 plain and ¥120.0 at λ2 0.8104), too thin to say. Every box, as selected
  historically on the final Win board, loses about a quarter of its stake.

### What B2 measures, and what it does not

- B2 measures the historical paid return of a category's Harville-weighted box against the
  payout rate. Its expectation is `rate · Σ_S π_c p_c/q_c ÷ Σ_S p_c` (spec section 4), a
  p-weighted average of `π_c/q_c` over the category: a ρ near 1 can hide over- and underpriced
  combinations that offset, so it does not show that each combination is priced fairly. When
  outcomes follow the exacta pool's unrounded shares (π = q), the expectation is `rate`, so
  expected ρ = 1, however much more mass the pool puts on favourite-first combinations than Win
  plus Harville does; at the paid, rounded-down dividend it is slightly below 1 (Breakage below).
- So ρ ≈ 1 does not show that the pool agrees with Win plus Harville. The direct cross-pool
  ratio (the pool's `Σ_S q / Σ_S p` against Win-implied Harville, the ticket's quantity) is not
  measured by B2. History cannot compute it, since only the paying exacta's dividend is stored;
  it needs full exacta boards (D, on #130-logged days, #281).
- Two further results are reported beside B2; both are partly in-sample at λ2 = 0.8104 (#115's
  full-development fit; #284 refits it per fold):
  - B3 (E1″) is an outcome-calibration diagnostic of Harville: the equal-weighted mean of `π/p`
    over the category's combinations. It gives neither the direction nor the size of the
    cross-pool ratio `Σ_S q / Σ_S p`, which weights by p: the equal-weighted mean can sit on the
    other side of 1 from it, even when outcomes follow the exacta pool.
  - C measures the pool's log-score gain over Harville on the paying exacta. A positive gain
    means the two differ and the pool forecasts better; it gives no sign or size by category.

### Breakage (descriptive, post-hoc)

E1 centres its loss on the nominal payout rate, while B2 is paid at dividends rounded down to
¥10. If the pool is fair before rounding, the paid ρ is about `1 − b_S`, where `b_S` is the
share of the category's unrounded payout that rounding takes. Short dividends lose more to
rounding, so `b_S` is largest for fav_first. Computed from the same 23,629 development races
(`exacta_run breakage`, git `27af886`, registry family `japan/131-exacta-breakage`, one record
per λ2), with the unrounded dividend taken at the midpoint `D + 5` of its ¥10 step, as
`implied_share` does. Every `b_S` and unrounded ρ below is that midpoint estimate, not a bound:
the actual rounding loss of a dividend lies anywhere from ¥0 to just under ¥10. The frozen E1
loss and the paid-return results above are unchanged.

| Category | b_S at D + 5, cell "all" (λ2 0.8104 / 1) | b_S at D + 5 over the 14 cells | ρ paid, "all" (λ2 0.8104) | ρ at D + 5, "all" [95% lower, upper] |
|---|---|---|---|---|
| fav_first | 0.36% / 0.39% | 0.07% to 0.68% | 1.0245 | 1.0282 [1.0116, 1.0450] |
| fav_second | 0.28% / 0.28% | 0.07% to 0.44% | 1.0031 | 1.0059 [0.9828, 1.0293] |
| no_fav | 0.09% / 0.10% | 0.05% to 0.16% | 0.9955 | 0.9964 [0.9840, 1.0090] |

- At the midpoint, rounding takes an estimated 0.05% to 0.68% of the payout across the cells
  (most in fav_first with the favourite at 1.9 or less: 0.63% and 0.68%). It pushes the paid ρ
  down, most for fav_first, so it can make a fair pool look like a favourite-first shortfall. The
  midpoint estimate is not the largest such effect: a larger actual rounding loss gives a larger
  one.
- Overall, fav_first's paid ρ is above 1 (1.024, lower bound 1.008), the opposite of a
  shortfall; adding any rounding loss back can only raise it (1.028, lower bound 1.012, at the
  midpoint).
- The one nominal fav_first exclusion in the registered direction, class:open (paid ρ 0.935,
  upper bound 0.997 at λ2 0.8104), is 0.938 at the midpoint with upper bound 0.9998 (0.933,
  upper bound 0.994 plain). That upper bound is 0.0002 below 1, so whether the nominal exclusion
  holds once breakage is added back depends on the within-step assumption; this report draws no
  conclusion either way.
- The other nominal exclusions (fav_second class:maiden and popularity:less, no_fav class:open)
  are above 1, and adding a rounding loss back only raises them.
- Limits: the midpoint assumption, and the ¥100 minimum payout (dividends at the floor are not
  rounded down from a uniform value). The bounds are E1's one-sided 95% meeting-bootstrap
  bounds, unadjusted for the 84 cells.
- Floor dividends (Sol review round 5 of 2c73ca0, 2026-11-28): a dividend at or below ¥110 may
  have been raised by JRA's minimum payout, so `D + 5` is not its unrounded value. Since git
  `ec331f7` the diagnostic counts such races per cell (`n_floor`, and an unavailable
  `floor_correction` marker in the registry), leaves them out of the unrounded ρ and `b_S`
  (unavailable when every race of a cell is at the floor) and keeps them in the paid ρ. The
  re-run (`japan-131-exacta-breakage`, two records) finds no scored race at or below ¥110:
  `n_floor` is 0 in every cell and every value above is unchanged.

### Variants searched (context, not a pass/fail test)

- E1 searched 84 members (λ2 {`full_development` 0.8104, 1} × 3 categories × 14 cells; family
  size 84 in the registry, no failed member). With 84 members about four nominal one-sided
  exclusions are expected by chance, and the members are strongly correlated (each race is in
  "all" and in one level of every marginal).
- Hansen SPA over the 84: p = 0.175 (consistent; lower 0.116, upper 0.187), 1,980 meeting
  clusters, stationary bootstrap with mean block 5 meetings, 10,000 replicates.
- Romano-Wolf StepM at a 5% family-wise error: no member.

### Reported variants

- C, the log-score total (pool's winning-exacta share against the model, nats per race, one-sided
  95% bounds): #115's full-development fit (`full_development`, λ2 = 0.8104, partly in-sample)
  +0.0133 [0.0108, 0.0158]; plain Harville +0.0306 [0.0270, 0.0341]. The exacta pool forecasts
  the paying exacta better than Win plus either ordering model.
- E1′ (B1, flat box), ρ for "all": fav_first 0.958 [0.928, 0.988], fav_second 0.855
  [0.813, 0.899], no_fav 0.777 [0.752, 0.803]. Equal stakes on every combination lose more than
  the takeout, most in no_fav: longshot exactas are overbet (a favourite-longshot bias inside the
  exacta pool), the opposite of what the owner's idea needs. Per ¥100 at the final dividend
  (2026-11-28): ¥71.9 [69.7, 74.2], ¥64.2 [61.0, 67.5] and ¥58.4 [56.5, 60.3]; every B1 member's
  value is in its registry file.
- C without a race above ¥110, or of one meeting (Sol review round 5 of 2c73ca0): since git
  `ec331f7` a C sample with no scored dividend above ¥110 keeps both λ2 records with
  `log_score_gain` unavailable (n 0) instead of failing after E1's records were written, and a
  one-meeting sample keeps its mean without bounds. Neither case arises in the scored races; C's
  values above are unchanged in the re-run.
- E1″ (B3, Harville's calibration, mean return, null 1), "all": #115's full-development fit
  (`full_development`, λ2 = 0.8104, partly in-sample) fav_first 0.931 [0.902, 0.962], fav_second
  0.864 [0.808, 0.924], no_fav 1.053 [0.974, 1.148]; plain Harville 1.191 [1.136, 1.247], 0.736
  [0.688, 0.786], 1.501 [1.351, 1.686]. As an outcome-calibration diagnostic: Harville's
  equal-weighted mean of outcome over model probability departs from 1 by category, plain
  Harville more so.
- Read together: B2's weighted paid return, "all" at λ2 = 0.8104, is 1.024 [1.008, 1.041]
  favourite-first, 1.003 [0.980, 1.026] favourite-second and 0.995 [0.983, 1.008] without the
  favourite, and the shrunk ρ of every member lies between 0.995 and 1.030. These are weighted
  averages over each category: they do not show that every combination is fairly priced, since
  over- and underpriced combinations can offset. B3 says nothing about the pool. Neither
  measures how far the pool's prices sit from Win plus Harville (section "What B2 measures"),
  and B3 and C at λ2 = 0.8104 are partly in-sample. Spec section 4's reading of B2 and B3
  together ("B2 < rate on fav_first with B3 ≈ 1" as a cross-pool overload) is withdrawn: B3's
  equal weighting can point the other way from the cross-pool ratio.
- E2 (ratio-cut rules, `japan-131-exacta-rule-v4`): **not scored.** rule-v3's headline scope
  ranked "less popular" by the tertile of pool totals within each logged day, so an early race
  could be classified with later races' totals, information not available when it is selected.
  rule-v4 (owner, 2026-11-27) takes "less popular" from the race card instead, the same
  definition as E1's headline cell (race_no 1–4 or a DEBUT/MAIDEN race), reports the tertile only
  as the proxy's agreement diagnostic, and has no meeting gate: it scores the #130-logged
  development days available at each run, with n and the CI (#281). The v3 freeze stays in the
  registry as history, not executed. No logged exacta board exists yet (two logged Kyoto days,
  2026-11-21 and 22, with WIN, PLACE, QUINELLA and QUINELLA_PLACE only).
  - rule-v5 (`japan-131-exacta-rule-v5`, frozen 2026-11-28, Sol review round 5 of 2c73ca0)
    supersedes rule-v4, which took `p_c(t)` from #114's out-of-fold `p_cal` (built from final
    Win odds) and required a #114 row: look-ahead for a T−2 rule. v5 takes `p_c(t)` from the
    normalised implied Win shares on the T−2 Win board, fed to the discounted Harville at the
    registered λ2, and scores every #130-logged development day with T−2 Win and exacta boards;
    nothing else changes. v4's freeze stays (under both ids) as history, not executed. #281
    scores E2 under v5.
- D (full-board Σq/Σp descriptive): not run, for the same reason: no logged exacta board exists.

## Variants tried

All on the development races above, meeting bootstrap, 10,000 replicates, seed 131. The v2
families (frozen 2026-11-27, run at git `5d68473`):

- E1 `japan/131-exacta-category-v2`: 84 members (B2 × λ2 {`full_development` 0.8104, 1} × 3 ×
  14), all scored, each with its one-sided bounds and its EB value; ranked above, raw table in
  [131-exacta-e1-members.md](131-exacta-e1-members.md).
- E1′ `japan/131-exacta-category-b1-v2`: 42 members (B1 has no λ2), all scored.
- E1″ `japan/131-exacta-category-b3-v2`: 84 members, all scored.
- C `japan/131-exacta-logscore-v2`: 2 members, scored (above).
- Max test `japan/131-exacta-category-v2/max-test`: one record (SPA p 0.175, StepM at a 5%
  family-wise error: no member).
- E2 `japan-131-exacta-rule-v4`: frozen 2026-11-27 (first under the legacy id
  `japan/131-exacta-rule-v4`, the same spec), 16 members, not scored (above). It
  supersedes `japan/131-exacta-rule-v3` (frozen the same day, not scored and not executed, its
  freeze kept), whose daily-tertile "less popular" cutoff used later races' pool totals; v4 uses
  the race-card proxy and drops the meeting gate. v3 in turn superseded
  `japan/131-exacta-rule-v2` (frozen the same day, not scored, its freeze kept), whose favourite
  came from the final Win odds; v3 and v4 select on the Win board at T-2, as spec sections 5 and
  6 require.
- Breakage `japan/131-exacta-breakage`: post-hoc and descriptive, not a test and not in any
  family's max test; two records (one per λ2), each with the breakage share and the unrounded ρ
  of all 42 category × cell members (git `27af886`, 30 s, 0.44 GB peak).

The 2026-11-28 re-run under the #525 ids (git `ec331f7`, Sol review round 5 of 2c73ca0), on the
same 23,629 races and 1,980 meetings, every value equal to the legacy records':

- `japan-131-exacta-category-v2` (84), `-category-b1-v2` (42), `-category-b3-v2` (84),
  `japan-131-exacta-logscore-v2` (2) and `japan-131-exacta-category-v2-max-test` (1): 213
  records, no failure, 7 min; B2 and B1 records carry `return_per_100` with its bounds.
- `japan-131-exacta-breakage`: two records, `n_floor` 0 in every cell, 38 s.
- E2 `japan-131-exacta-rule-v5`: frozen 2026-11-28, 16 members, not scored. It supersedes
  `japan-131-exacta-rule-v4`, frozen again that day under its #525 id with its 2026-11-27 spec
  (same hash as `japan/131-exacta-rule-v4`), so v5's `supersedes` names a freeze; v4 is not
  executed.

The superseded v1 families (frozen and scored 2026-11-25, label `in_fold` for the same fit),
whose records stay in the registry:

- `japan/131-exacta-category-v1` (84), `-b1-v1` (42), `-b3-v1` (84), `japan/131-exacta-logscore-v1`
  (2): every metric of every v2 record equals its v1 record's (value, bounds and n), checked
  record by record in the registry on 2026-11-27. v1 is superseded only for the label (owner,
  2026-11-26).
- Max test `japan/131-exacta-category-v1/max-test`: two records.
  - **Failed (void):** the first run (git `4271171`) passed the spec's 95% confidence level as
    `snooping.stepm`'s `level`, which is the test size, so StepM ran at a family-wise error of
    0.95 and "rejected" 13 members. That record stays in the registry; its StepM list means
    nothing. In family `japan/131-exacta-category-v1/max-test`
    ([japan--131-exacta-category-v1--max-test.md](registry/japan/japan--131-exacta-category-v1--max-test.md)), it is the first `spa+stepm` record (the first three
    rows, `stepm_superior` 13); the valid re-run is the next three rows (`stepm_superior` 0). SPA
    does not take a level and is identical in both records (p 0.175).
  - The re-run (git `9910a7c`, FWER 0.05): SPA p = 0.175, StepM no member; the v2 record equals it.
- `japan/131-exacta-rule-v1`: frozen, not scored.

## Alternatives rejected and why

- A, the mean of q_w/p_w over races whose winner is in S: it selects on the winner, so it reports
  an overload when the pool is exactly right (spec section 4). Not run.
- C split by category: not a proper score per category (a forecast that overloads S gains on S).
  Only the total is reported.
- The ticket's per-race Σq/Σp ratio (D) as the historical estimand: history holds only the
  winning exacta's dividend, so it cannot be computed; it waits for logged boards.
- `core/eval/segment_roi.py` for E1's shrinkage: it needs every bet's odds and outcome, and losing
  exactas' dividends are unknown on history. Normal-normal EB on the bootstrap SEs instead.
- Two-sided tests: the owner signed one-sided directions (answer 2).
- Crossed segment cells: 14 marginal cells plus one pre-stated cross keep the family at 84.
- Historical rule tests selected on final odds: look-ahead; E2 selects its favourite from the
  board at T-2, and rule-v4 its "less popular" from the race card (rule-v3's daily pool-total
  tertile used later races' totals; it is now an agreement diagnostic only). rule-v5 also takes
  `p_c(t)` from the T−2 Win board rather than #114's final-odds calibration.

## What this rules out

All of it is about the historical paid-return proxy: the Harville-weighted paid return of each
category's box against the payout rate, an average that can hide offsetting over- and
underpriced combinations. B2 does not measure the direct cross-pool ratio (the pool against Win-implied
Harville), so nothing here rules out the pool putting more mass on favourite-first combinations
than Harville does ("What B2 measures"; B3 and C bear on it only indirectly and are partly
in-sample).

- A favourite-first shortfall in the paid return, overall: ρ_fav_first is 1.024 with a
  one-sided 95% lower bound of 1.008, so a paid return below the payout rate (ρ < 1) is
  excluded at that level; the favourite-first box returns slightly more than the pool-fair rate,
  not less, and more again with breakage added back (1.028, lower bound 1.012, at the D + ¥5
  midpoint; any rounding loss added back only raises it). This direction
  was not the registered alternative, so it is an observation, not a tested claim.
- A paid return of favourite-second or no-favourite boxes above the payout rate by more than
  about 2.6% and 0.8% (upper bounds 1.026 and 1.008 overall, λ2 = 0.8104), and, in the owner's
  headline cell, by more than about 12.7% for favourite-second (upper bound 1.127).
- A profit from boxing a category at final dividends: it needs ρ > 1/rate ≈ 1.33, and every cell
  with more than 2,000 races has its upper bound below 1.13. Only the favourite-at-5.0-or-more
  cell (189 races) is too thin to rule it out. Per ¥100 (2026-11-28): every such cell's upper
  bound is below ¥85 against the ¥100 a profit needs.
- All of the above is limited to the historical selection on the final Win board (look-ahead:
  the note under the ranked table). It rules out no rule decided before the off; E2 tests one.
- What it does not rule out: a rule that selects single combinations by their board ratio at
  T-2 (E2). A box averages over a category; a selective rule can still find mispriced
  combinations inside it, and only logged boards can test that.
- Nor does it rule out a cross-pool gap: the exacta pool pricing favourite-first combinations
  higher or lower than Win plus Harville. That needs the full-board ratio D on logged days
  (#281).

## Follow-ups filed

All `status:proposed` under epic #72:

- #281 Score #131's E2 exacta ratio-cut rules on logged boards (with the D descriptive on the same
  days; the owner's 2026-11-27 reframe drops its meeting gate). It scores
  `japan-131-exacta-rule-v4`, frozen 2026-11-27, whose race-card popularity cutoff uses only
  information available when each race is selected. Since 2026-11-28 it scores
  `japan-131-exacta-rule-v5`, which also takes `p_c(t)` and eligibility from the T−2 boards.
- #282 Pre-register a fav_first exacta loss-avoidance rule (family L); its cuts become ranked rows.
- #283 Check the ¥100 exacta own-impact assumption against #130 pool totals.
- #284 Refit λ2 per walk-forward fold for #131's B3 and C, so both are strictly out-of-sample.
- #449, the leaderboard writer (not this ticket's): once it lands, every E1 and E2 member is a
  leaderboard row.

## #284: λ2 refitted per walk-forward fold (B3 and C out of sample)

Registry families `japan-131-exacta-category-b3-v3` and `japan-131-exacta-logscore-v3` (frozen
2026-11-28, scored at git `25995af`), files
[japan-131-exacta-category-b3-v3.md](registry/japan/japan-131-exacta-category-b3-v3.md) and
[japan-131-exacta-logscore-v3.md](registry/japan/japan-131-exacta-logscore-v3.md). They sit beside
the v2 families, which are unchanged: both λ2 versions are kept as ranked rows,
`full_development` (#115's single fit, partly in-sample, the rows above) and `per_fold` (below).
Public JRA data only (exacta dividends, final Win odds, results); development days only, nothing
touches A2 holdout days. Exacta takeout 25%.

**Hypothesis (stated before scoring).** B3 and C at #115's λ2 = 0.8104 were partly in-sample: that
fit saw the races they were scored on. If the in-sample part matters, B3 and C move when λ2 is
fitted on each fold's training races alone. The null, the starting point: they do not, beyond what
the sample change below alone does.

**Method.** For each fold of `folds.yaml` (quarterly, expanding, one-week embargo) over the
development calendar, λ2 is `ordering.fit_ordering`'s stage-1 maximum-likelihood fit
(discounted Harville) on the development out-of-fold (#114) races dated on or before the fold's
`train_end`: no race of the test window, the embargo or a later fold reaches it. Each race is then
scored at the λ2 of the fold holding its date, with the v2 member design, estimand, bootstrap
(meeting clusters, 10,000 replicates, seed 131) and exclusions unchanged. The spec pins the fold
scheme it was frozen under, so a changed `folds.yaml` is a new family.

**Excluded: 2020-Q1.** The out-of-fold Win probabilities start with 2020-Q1, so that quarter has
no training race and no λ2 (n_train_races 0). Its races are left out and counted
(`no_training_races` 910 of the development races; 907 of them scored in v2). There is no
minimum training size: 2020-Q2 trains on 869 races. The per-fold rows therefore rest on 22,722
races (1,903 meetings), 2020-04-04 to 2026-11-15, against v2's 23,629 (1,980 meetings): the
comparison below changes the λ2 fit and drops that quarter at once, and the two are not
separated. Other exclusions are v2's (dead heat 43, under 5 starters 1, boost may be withheld 84).

| Test window | Training ends | Training races | λ2 |
| --- | --- | --- | --- |
| 2020-01-01 to 2020-03-31 | 2019-12-29 | 0 | none (left out) |
| 2020-04-01 to 2020-06-30 | 2020-03-28 | 869 | 0.8205 |
| 2020-07-01 to 2020-09-30 | 2020-06-27 | 1,728 | 0.7883 |
| 2020-10-01 to 2020-12-31 | 2020-09-26 | 2,548 | 0.8099 |
| 2021-01-01 to 2021-03-31 | 2020-12-29 | 3,426 | 0.8197 |
| 2021-04-01 to 2021-06-30 | 2021-03-27 | 4,272 | 0.8184 |
| 2021-07-01 to 2021-09-30 | 2021-06-26 | 5,131 | 0.8229 |
| 2021-10-01 to 2021-12-31 | 2021-09-25 | 5,953 | 0.8217 |
| 2022-01-01 to 2022-03-31 | 2021-12-29 | 6,861 | 0.8230 |
| 2022-04-01 to 2022-06-30 | 2022-03-26 | 7,687 | 0.8211 |
| 2022-07-01 to 2022-09-30 | 2022-06-25 | 8,548 | 0.8124 |
| 2022-10-01 to 2022-12-31 | 2022-09-24 | 9,395 | 0.8153 |
| 2023-01-01 to 2023-03-31 | 2022-12-29 | 10,305 | 0.8131 |
| 2023-04-01 to 2023-06-30 | 2023-03-25 | 11,131 | 0.8166 |
| 2023-07-01 to 2023-09-30 | 2023-06-24 | 11,993 | 0.8111 |
| 2023-10-01 to 2023-12-31 | 2023-09-24 | 12,865 | 0.8080 |
| 2024-01-01 to 2024-03-31 | 2023-12-30 | 13,747 | 0.8056 |
| 2024-04-01 to 2024-06-30 | 2024-03-30 | 14,627 | 0.8075 |
| 2024-07-01 to 2024-09-30 | 2024-06-29 | 15,495 | 0.8102 |
| 2024-10-01 to 2024-12-31 | 2024-09-28 | 16,320 | 0.8092 |
| 2025-01-01 to 2025-03-31 | 2024-12-29 | 17,179 | 0.8079 |
| 2025-04-01 to 2025-06-30 | 2025-03-29 | 18,002 | 0.8078 |
| 2025-07-01 to 2025-09-30 | 2025-06-28 | 18,860 | 0.8068 |
| 2025-10-01 to 2025-12-31 | 2025-09-27 | 19,757 | 0.8063 |
| 2026-01-01 to 2026-03-31 | 2025-12-28 | 20,606 | 0.8088 |
| 2026-04-01 to 2026-06-30 | 2026-03-28 | 21,425 | 0.8109 |
| 2026-07-01 to 2026-09-30 | 2026-06-27 | 22,282 | 0.8116 |
| 2026-10-01 to 2026-12-31 | 2026-09-26 | 23,178 | 0.8100 |

The per-fold λ2 ranges from 0.7883 (2020-Q3, 1,728 training races) to 0.8230 (2022-Q1) and settles
within 0.8056 to 0.8116 from 2023-Q4 on; #115's single fit of 0.8104 sits inside the later folds'
range. A discount below 1 holds in every fold.

**Results.** B3 (mean return, Harville's calibration on the category, null 1) for every member,
`per_fold` beside `full_development` (the v2 record, partly in-sample, 23,629 races), one-sided 95%
lower and upper bounds of the meeting bootstrap in brackets (not adjusted for multiplicity):

| Category | Cell | n per_fold | B3 per_fold | n full_development | B3 full_development |
| --- | --- | --- | --- | --- | --- |
| fav_first | all | 22,722 | 0.935 [0.904, 0.966] | 23,629 | 0.931 [0.902, 0.962] |
| fav_first | fav_odds:le1.9 | 5,245 | 0.901 [0.857, 0.947] | 5,449 | 0.890 [0.848, 0.934] |
| fav_first | fav_odds:2.0-2.9 | 9,982 | 0.938 [0.894, 0.986] | 10,375 | 0.938 [0.895, 0.984] |
| fav_first | fav_odds:3.0-4.9 | 7,314 | 0.955 [0.890, 1.022] | 7,616 | 0.952 [0.890, 1.019] |
| fav_first | fav_odds:ge5.0 | 181 | 0.871 [0.535, 1.253] | 189 | 0.917 [0.577, 1.286] |
| fav_first | field:5-10 | 3,687 | 0.963 [0.899, 1.030] | 3,820 | 0.962 [0.899, 1.027] |
| fav_first | field:11-14 | 7,633 | 0.981 [0.927, 1.038] | 7,939 | 0.975 [0.924, 1.029] |
| fav_first | field:15-18 | 11,402 | 0.894 [0.848, 0.942] | 11,870 | 0.893 [0.848, 0.939] |
| fav_first | class:maiden | 10,173 | 0.889 [0.843, 0.937] | 10,559 | 0.884 [0.839, 0.930] |
| fav_first | class:win1-3 | 10,535 | 0.995 [0.950, 1.043] | 10,978 | 0.990 [0.946, 1.036] |
| fav_first | class:open | 2,014 | 0.848 [0.761, 0.938] | 2,092 | 0.862 [0.776, 0.955] |
| fav_first | popularity:less | 10,662 | 0.888 [0.844, 0.934] | 11,070 | 0.884 [0.841, 0.928] |
| fav_first | popularity:rest | 12,060 | 0.976 [0.934, 1.020] | 12,559 | 0.973 [0.932, 1.015] |
| fav_first | headline | 2,809 | 0.896 [0.793, 1.007] | 2,923 | 0.895 [0.796, 1.003] |
| fav_second | all | 22,722 | 0.855 [0.801, 0.914] | 23,629 | 0.864 [0.808, 0.924] |
| fav_second | fav_odds:le1.9 | 5,245 | 0.775 [0.692, 0.865] | 5,449 | 0.777 [0.696, 0.866] |
| fav_second | fav_odds:2.0-2.9 | 9,982 | 0.902 [0.802, 1.011] | 10,375 | 0.898 [0.801, 1.006] |
| fav_second | fav_odds:3.0-4.9 | 7,314 | 0.824 [0.753, 0.901] | 7,616 | 0.856 [0.770, 0.951] |
| fav_second | fav_odds:ge5.0 | 181 | 1.870 [0.843, 3.279] | 189 | 1.793 [0.818, 3.185] |
| fav_second | field:5-10 | 3,687 | 0.838 [0.738, 0.950] | 3,820 | 0.840 [0.738, 0.949] |
| fav_second | field:11-14 | 7,633 | 0.811 [0.731, 0.898] | 7,939 | 0.809 [0.732, 0.896] |
| fav_second | field:15-18 | 11,402 | 0.891 [0.806, 0.983] | 11,870 | 0.908 [0.819, 1.007] |
| fav_second | class:maiden | 10,173 | 0.873 [0.784, 0.969] | 10,559 | 0.896 [0.800, 0.999] |
| fav_second | class:win1-3 | 10,535 | 0.845 [0.771, 0.924] | 10,978 | 0.843 [0.770, 0.922] |
| fav_second | class:open | 2,014 | 0.821 [0.671, 0.993] | 2,092 | 0.812 [0.665, 0.977] |
| fav_second | popularity:less | 10,662 | 0.872 [0.786, 0.964] | 11,070 | 0.892 [0.800, 0.990] |
| fav_second | popularity:rest | 12,060 | 0.841 [0.772, 0.915] | 12,559 | 0.839 [0.770, 0.913] |
| fav_second | headline | 2,809 | 0.929 [0.784, 1.090] | 2,923 | 1.007 [0.811, 1.229] |
| no_fav | all | 22,722 | 1.061 [0.977, 1.156] | 23,629 | 1.053 [0.974, 1.148] |
| no_fav | fav_odds:le1.9 | 5,245 | 0.989 [0.833, 1.173] | 5,449 | 0.976 [0.823, 1.148] |
| no_fav | fav_odds:2.0-2.9 | 9,982 | 1.055 [0.972, 1.144] | 10,375 | 1.050 [0.970, 1.136] |
| no_fav | fav_odds:3.0-4.9 | 7,314 | 1.116 [0.941, 1.369] | 7,616 | 1.110 [0.941, 1.362] |
| no_fav | fav_odds:ge5.0 | 181 | 1.234 [0.948, 1.556] | 189 | 1.200 [0.924, 1.521] |
| no_fav | field:5-10 | 3,687 | 0.908 [0.813, 1.014] | 3,820 | 0.903 [0.811, 1.005] |
| no_fav | field:11-14 | 7,633 | 0.987 [0.908, 1.073] | 7,939 | 0.982 [0.906, 1.067] |
| no_fav | field:15-18 | 11,402 | 1.160 [1.011, 1.342] | 11,870 | 1.149 [1.004, 1.326] |
| no_fav | class:maiden | 10,173 | 1.078 [0.914, 1.279] | 10,559 | 1.062 [0.903, 1.258] |
| no_fav | class:win1-3 | 10,535 | 1.050 [0.979, 1.125] | 10,978 | 1.046 [0.978, 1.119] |
| no_fav | class:open | 2,014 | 1.029 [0.921, 1.143] | 2,092 | 1.049 [0.939, 1.163] |
| no_fav | popularity:less | 10,662 | 1.096 [0.938, 1.288] | 11,070 | 1.078 [0.924, 1.265] |
| no_fav | popularity:rest | 12,060 | 1.030 [0.967, 1.096] | 12,559 | 1.031 [0.971, 1.097] |
| no_fav | headline | 2,809 | 1.279 [0.863, 1.931] | 2,923 | 1.257 [0.854, 1.906] |

C (the log-score total: mean of log q_w − log p_w, nats per race, one-sided 95% bounds): `per_fold`
+0.0135 [0.0109, 0.0161] over 22,722 races in 1,903 meetings, against `full_development` (partly
in-sample) +0.0133 [0.0108, 0.0158] over 23,629 races. Both λ2 versions are kept as rows.

**What it shows.** Out of sample, B3 and C land where the in-sample fit put them. Across the 42
B3 members the per-fold value differs from the `full_development` value by 0.0055 at the median
and by at most 0.078 (fav_second headline, 0.929 against 1.007, and the 181-race fav_second
fav_odds:ge5.0 cell, 1.870 against 1.793, both with CIs far wider than the difference); in "all",
fav_first 0.935 against 0.931, fav_second 0.855 against 0.864 and no_fav 1.061 against 1.053,
each well inside the other's bound. The pool still forecasts the paying exacta better than Win
plus Harville by about 0.013 nats per race, with the lower bound above 0 (0.0109).

**What it rules out.** The partly in-sample fit did not manufacture B3's departures from 1 or C's
positive gain: with λ2 fitted on training races alone they are the same size and sign, so the
#131 reading of B3 (an outcome-calibration diagnostic: favourite-first 0.935 and favourite-second
0.855, one-sided upper bounds 0.966 and 0.914, below 1; no_fav 1.061, bounds 0.977 to 1.156, not
distinguishable from 1) and of C does not depend on the in-sample fit. It rules out nothing about B2, whose null does not depend on p, and it does
not change #131's conclusions: B3 still gives no direction for the cross-pool ratio, and nothing
is adopted. It does not separate the λ2 refit from the drop of 2020-Q1; a comparison on the same
22,722 races would need the `full_development` rows re-scored on them, which this ticket did not
do (the v2 rows are kept as recorded).

**Variants tried.** B3: 42 members (3 categories × 14 cells, `per_fold`); C: 1; no other λ2
(plain Harville is not re-scored: it has no fit), no other fold scheme, one run at `25995af`.
Failures: none. Reports with the full tables:
`$HORSETRACK_DATA_DIR/reports/japan/131-exacta/per-fold-25995afe232c-39cb0745.{json,md}` (outside
the repository; aggregates of public JRA data).

Follow-up: none filed. #284 closes the "partly in-sample" caveat on B3 and C; the v2 B3 and C rows
stay labelled partly in-sample.
