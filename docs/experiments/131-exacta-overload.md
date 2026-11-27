# Experiment report: exacta favourite overload (#131)

The pre-registered study of whether the JRA exacta pool (`FORECAST`) overloads the favourite's
combinations relative to the Win pool and an ordering model. Spec:
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md), approved by the owner
on 2026-11-25 and frozen in the Japan run registry before any scored run. Registry hypothesis id
`E43`. E1's members are ranked by their shrunk return below; every E1, E1′ and E1″ member's value
is in [131-exacta-e1-members.md](131-exacta-e1-members.md) and every registry record in
[registry/japan/](registry/japan/README.md), one file per family (the file names of the
`japan/131-…` families are TBD). Nothing is adopted; no rule is staked.

The results name the v2 families (`japan/131-exacta-*-v2`, frozen 2026-11-27). v2 supersedes the
v1 families of 2026-11-25 only for the λ2 label: the owner, 2026-11-26 on #131, "Relabel λ2
honestly: it is #115's single full-development fit (partly in-sample), not 'in_fold'". The v2
re-run gave v1's values in every record (Variants tried).

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
  (ρ = 1 under π = q) does not depend on p, so E1's ranking and its SPA and StepM context stand at
  any λ2.
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

| Rank | λ2 | Category | Cell | Registered direction | n races | Meetings | Shrunk ρ (EB) | ρ | 95% lower | 95% upper |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 (plain) | fav_first | fav_odds:2.0-2.9 | ρ < 1 | 10,375 | 1,980 | 1.0297 | 1.044 | 1.019 | 1.068 |
| 2 | 1 (plain) | fav_first | class:win1-3 | ρ < 1 | 10,978 | 1,979 | 1.0296 | 1.043 | 1.017 | 1.068 |
| 3 | 1 (plain) | fav_first | field:11-14 | ρ < 1 | 7,939 | 1,930 | 1.0296 | 1.046 | 1.016 | 1.076 |
| 4 | 1 (plain) | fav_first | popularity:less | ρ < 1 | 11,070 | 1,980 | 1.0291 | 1.033 | 1.009 | 1.057 |
| 5 | 1 (plain) | fav_first | field:5-10 | ρ < 1 | 3,820 | 1,615 | 1.0290 | 1.037 | 0.999 | 1.074 |
| 6 | 1 (plain) | fav_first | headline | ρ < 1 | 2,923 | 1,572 | 1.0290 | 1.047 | 0.990 | 1.107 |
| 7 | 1 (plain) | fav_first | class:maiden | ρ < 1 | 10,559 | 1,980 | 1.0289 | 1.031 | 1.007 | 1.055 |
| 8 | 1 (plain) | fav_first | fav_odds:le1.9 | ρ < 1 | 5,449 | 1,860 | 1.0287 | 1.028 | 1.003 | 1.053 |
| 9 | 1 (plain) | fav_first | fav_odds:ge5.0 | ρ < 1 | 189 | 177 | 1.0287 | 0.844 | 0.582 | 1.122 |
| 10 | 1 (plain) | fav_first | all | ρ < 1 | 23,629 | 1,980 | 1.0286 | 1.028 | 1.011 | 1.044 |
| 11 | 1 (plain) | fav_first | popularity:rest | ρ < 1 | 12,559 | 1,979 | 1.0284 | 1.022 | 0.998 | 1.047 |
| 12 | 1 (plain) | fav_first | fav_odds:3.0-4.9 | ρ < 1 | 7,616 | 1,943 | 1.0282 | 1.010 | 0.974 | 1.045 |
| 13 | 1 (plain) | fav_first | field:15-18 | ρ < 1 | 11,870 | 1,965 | 1.0277 | 1.012 | 0.986 | 1.037 |
| 14 | 1 (plain) | fav_first | class:open | ρ < 1 | 2,092 | 1,655 | 1.0277 | 0.930 | 0.870 | 0.991 |
| 15 | `full_development` (0.8104) | fav_first | field:11-14 | ρ < 1 | 7,939 | 1,930 | 1.0255 | 1.044 | 1.014 | 1.073 |
| 16 | `full_development` (0.8104) | fav_first | class:win1-3 | ρ < 1 | 10,978 | 1,979 | 1.0255 | 1.042 | 1.017 | 1.067 |
| 17 | `full_development` (0.8104) | fav_first | fav_odds:2.0-2.9 | ρ < 1 | 10,375 | 1,980 | 1.0255 | 1.040 | 1.016 | 1.065 |
| 18 | `full_development` (0.8104) | fav_first | headline | ρ < 1 | 2,923 | 1,572 | 1.0255 | 1.037 | 0.980 | 1.096 |
| 19 | `full_development` (0.8104) | fav_first | field:5-10 | ρ < 1 | 3,820 | 1,615 | 1.0255 | 1.034 | 0.996 | 1.070 |
| 20 | `full_development` (0.8104) | fav_first | popularity:less | ρ < 1 | 11,070 | 1,980 | 1.0255 | 1.026 | 1.002 | 1.049 |
| 21 | `full_development` (0.8104) | fav_first | all | ρ < 1 | 23,629 | 1,980 | 1.0255 | 1.024 | 1.008 | 1.041 |
| 22 | `full_development` (0.8104) | fav_first | class:maiden | ρ < 1 | 10,559 | 1,980 | 1.0255 | 1.024 | 1.000 | 1.048 |
| 23 | `full_development` (0.8104) | fav_first | popularity:rest | ρ < 1 | 12,559 | 1,979 | 1.0255 | 1.023 | 0.999 | 1.047 |
| 24 | `full_development` (0.8104) | fav_first | fav_odds:le1.9 | ρ < 1 | 5,449 | 1,860 | 1.0255 | 1.022 | 0.997 | 1.047 |
| 25 | `full_development` (0.8104) | fav_first | field:15-18 | ρ < 1 | 11,870 | 1,965 | 1.0255 | 1.009 | 0.983 | 1.034 |
| 26 | `full_development` (0.8104) | fav_first | fav_odds:3.0-4.9 | ρ < 1 | 7,616 | 1,943 | 1.0255 | 1.008 | 0.973 | 1.043 |
| 27 | `full_development` (0.8104) | fav_first | class:open | ρ < 1 | 2,092 | 1,655 | 1.0255 | 0.935 | 0.875 | 0.997 |
| 28 | `full_development` (0.8104) | fav_first | fav_odds:ge5.0 | ρ < 1 | 189 | 177 | 1.0255 | 0.886 | 0.611 | 1.179 |
| 29 | 1 (plain) | fav_second | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0078 | 1.036 | 1.003 | 1.071 |
| 30 | 1 (plain) | fav_second | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0076 | 1.034 | 1.001 | 1.067 |
| 31 | `full_development` (0.8104) | fav_second | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0071 | 1.036 | 1.002 | 1.071 |
| 32 | `full_development` (0.8104) | fav_second | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0069 | 1.033 | 1.000 | 1.067 |
| 33 | 1 (plain) | fav_second | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0066 | 1.025 | 0.992 | 1.059 |
| 34 | `full_development` (0.8104) | fav_second | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0059 | 1.024 | 0.991 | 1.058 |
| 35 | 1 (plain) | fav_second | headline | ρ > 1 | 2,923 | 1,572 | 1.0058 | 1.056 | 0.984 | 1.128 |
| 36 | `full_development` (0.8104) | fav_second | headline | ρ > 1 | 2,923 | 1,572 | 1.0050 | 1.055 | 0.983 | 1.127 |
| 37 | 1 (plain) | fav_second | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0048 | 1.010 | 0.967 | 1.053 |
| 38 | 1 (plain) | fav_second | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0047 | 1.195 | 0.837 | 1.593 |
| 39 | 1 (plain) | fav_second | class:open | ρ > 1 | 2,092 | 1,655 | 1.0044 | 1.002 | 0.921 | 1.085 |
| 40 | 1 (plain) | fav_second | all | ρ > 1 | 23,629 | 1,980 | 1.0044 | 1.004 | 0.981 | 1.027 |
| 41 | `full_development` (0.8104) | fav_second | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0041 | 1.009 | 0.967 | 1.053 |
| 42 | 1 (plain) | fav_second | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0041 | 1.001 | 0.967 | 1.036 |
| 43 | `full_development` (0.8104) | fav_second | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0039 | 1.199 | 0.840 | 1.599 |
| 44 | 1 (plain) | fav_second | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0039 | 0.995 | 0.953 | 1.040 |
| 45 | `full_development` (0.8104) | fav_second | class:open | ρ > 1 | 2,092 | 1,655 | 1.0037 | 1.001 | 0.919 | 1.084 |
| 46 | `full_development` (0.8104) | fav_second | all | ρ > 1 | 23,629 | 1,980 | 1.0036 | 1.003 | 0.980 | 1.026 |
| 47 | 1 (plain) | no_fav | class:open | ρ > 1 | 2,092 | 1,655 | 1.0036 | 1.063 | 1.021 | 1.106 |
| 48 | 1 (plain) | no_fav | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 1.0036 | 1.022 | 0.938 | 1.103 |
| 49 | 1 (plain) | no_fav | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0036 | 1.019 | 0.995 | 1.043 |
| 50 | 1 (plain) | no_fav | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 1.0036 | 1.010 | 0.993 | 1.026 |
| 51 | 1 (plain) | no_fav | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0036 | 1.005 | 0.988 | 1.022 |
| 52 | 1 (plain) | no_fav | all | ρ > 1 | 23,629 | 1,980 | 1.0036 | 1.004 | 0.991 | 1.017 |
| 53 | 1 (plain) | no_fav | field:15-18 | ρ > 1 | 11,870 | 1,965 | 1.0036 | 1.004 | 0.986 | 1.021 |
| 54 | 1 (plain) | no_fav | popularity:less | ρ > 1 | 11,070 | 1,980 | 1.0036 | 1.002 | 0.982 | 1.023 |
| 55 | 1 (plain) | no_fav | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0036 | 1.001 | 0.983 | 1.020 |
| 56 | 1 (plain) | no_fav | class:maiden | ρ > 1 | 10,559 | 1,980 | 1.0036 | 1.000 | 0.979 | 1.022 |
| 57 | 1 (plain) | no_fav | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0036 | 0.998 | 0.959 | 1.038 |
| 58 | 1 (plain) | no_fav | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0036 | 0.996 | 0.978 | 1.014 |
| 59 | 1 (plain) | no_fav | headline | ρ > 1 | 2,923 | 1,572 | 1.0036 | 0.986 | 0.958 | 1.013 |
| 60 | 1 (plain) | no_fav | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0036 | 0.972 | 0.935 | 1.010 |
| 61 | 1 (plain) | fav_second | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0035 | 0.992 | 0.953 | 1.031 |
| 62 | `full_development` (0.8104) | fav_second | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 1.0033 | 1.000 | 0.966 | 1.034 |
| 63 | `full_development` (0.8104) | fav_second | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 1.0031 | 0.995 | 0.952 | 1.039 |
| 64 | `full_development` (0.8104) | fav_second | field:11-14 | ρ > 1 | 7,939 | 1,930 | 1.0027 | 0.991 | 0.952 | 1.031 |
| 65 | 1 (plain) | fav_second | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0025 | 0.962 | 0.911 | 1.014 |
| 66 | `full_development` (0.8104) | fav_second | field:5-10 | ρ > 1 | 3,820 | 1,615 | 1.0017 | 0.961 | 0.910 | 1.014 |
| 67 | 1 (plain) | fav_second | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0014 | 0.977 | 0.945 | 1.010 |
| 68 | 1 (plain) | fav_second | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0013 | 0.973 | 0.938 | 1.008 |
| 69 | `full_development` (0.8104) | fav_second | popularity:rest | ρ > 1 | 12,559 | 1,979 | 1.0006 | 0.977 | 0.945 | 1.009 |
| 70 | `full_development` (0.8104) | fav_second | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 1.0005 | 0.972 | 0.938 | 1.007 |
| 71 | `full_development` (0.8104) | no_fav | class:open | ρ > 1 | 2,092 | 1,655 | 0.9971 | 1.062 | 1.022 | 1.104 |
| 72 | `full_development` (0.8104) | no_fav | fav_odds:3.0-4.9 | ρ > 1 | 7,616 | 1,943 | 0.9968 | 1.004 | 0.988 | 1.020 |
| 73 | `full_development` (0.8104) | no_fav | field:11-14 | ρ > 1 | 7,939 | 1,930 | 0.9967 | 1.011 | 0.988 | 1.034 |
| 74 | `full_development` (0.8104) | no_fav | popularity:rest | ρ > 1 | 12,559 | 1,979 | 0.9962 | 0.998 | 0.982 | 1.015 |
| 75 | `full_development` (0.8104) | no_fav | fav_odds:ge5.0 | ρ > 1 | 189 | 177 | 0.9961 | 1.035 | 0.952 | 1.116 |
| 76 | `full_development` (0.8104) | no_fav | field:15-18 | ρ > 1 | 11,870 | 1,965 | 0.9959 | 0.995 | 0.978 | 1.013 |
| 77 | `full_development` (0.8104) | no_fav | all | ρ > 1 | 23,629 | 1,980 | 0.9959 | 0.995 | 0.983 | 1.008 |
| 78 | `full_development` (0.8104) | no_fav | fav_odds:le1.9 | ρ > 1 | 5,449 | 1,860 | 0.9957 | 0.986 | 0.948 | 1.024 |
| 79 | `full_development` (0.8104) | no_fav | fav_odds:2.0-2.9 | ρ > 1 | 10,375 | 1,980 | 0.9957 | 0.993 | 0.975 | 1.011 |
| 80 | `full_development` (0.8104) | no_fav | popularity:less | ρ > 1 | 11,070 | 1,980 | 0.9957 | 0.992 | 0.973 | 1.012 |
| 81 | `full_development` (0.8104) | no_fav | class:maiden | ρ > 1 | 10,559 | 1,980 | 0.9956 | 0.990 | 0.970 | 1.011 |
| 82 | `full_development` (0.8104) | no_fav | headline | ρ > 1 | 2,923 | 1,572 | 0.9953 | 0.980 | 0.953 | 1.007 |
| 83 | `full_development` (0.8104) | no_fav | field:5-10 | ρ > 1 | 3,820 | 1,615 | 0.9953 | 0.963 | 0.927 | 1.001 |
| 84 | `full_development` (0.8104) | no_fav | class:win1-3 | ρ > 1 | 10,978 | 1,979 | 0.9952 | 0.987 | 0.970 | 1.005 |

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
  exacta pool), the opposite of what the owner's idea needs.
- E1″ (B3, Harville's calibration, mean return, null 1), "all": #115's full-development fit
  (`full_development`, λ2 = 0.8104, partly in-sample) fav_first 0.931 [0.902, 0.962], fav_second
  0.864 [0.808, 0.924], no_fav 1.053 [0.974, 1.148]; plain Harville 1.191 [1.136, 1.247], 0.736
  [0.688, 0.786], 1.501 [1.351, 1.686]. The ordering model is miscalibrated by category, plain
  Harville badly so; the exacta pool is not.
- Read together (spec section 4): B2 ≈ 1 in every category while B3 departs from 1, so the gap
  between the exacta pool and Win plus Harville sits in the ordering model, not in the exacta
  pool.
- E2 (ratio-cut rules): **not yet scorable.** The frozen family needs at least 100 #130-logged
  development meetings with exacta boards; the data directory holds 0 (two logged Kyoto days,
  2026-11-21 and 22, with WIN, PLACE, QUINELLA and QUINELLA_PLACE only). The owner's 2026-11-27
  reframe drops the 100-meeting gate in favour of scoring the logged days available at each run;
  that change belongs to #281, and the frozen E2 spec keeps `min_meetings` 100 here.
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
- E2 `japan/131-exacta-rule-v2`: frozen, 16 members, not scored (above).

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
    nothing. In [registry/japan/](registry/japan/README.md), family
    `japan/131-exacta-category-v1/max-test`, it is the first `spa+stepm` record (the first three
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
- Historical rule tests selected on final odds: look-ahead; E2 selects only from the board at T-2.

## What this rules out

- A favourite-first overload in the JRA exacta pool, overall: ρ_fav_first is 1.024 with a
  one-sided 95% lower bound of 1.008, so any overload (ρ < 1) is excluded at that level; the
  favourite-first box returns slightly more than the pool-fair rate, not less. This direction was
  not the registered alternative, so it is an observation, not a tested claim.
- An underload of favourite-second or no-favourite combinations of more than about 2.6% and 0.8%
  of the payout (upper bounds 1.026 and 1.008 overall, λ2 = 0.8104), and, in the owner's headline
  cell, of more than about 12.7% for favourite-second (upper bound 1.127).
- A profit from boxing a category at final dividends: it needs ρ > 1/rate ≈ 1.33, and every cell
  with more than 2,000 races has its upper bound below 1.13. Only the favourite-at-5.0-or-more
  cell (189 races) is too thin to rule it out.
- What it does not rule out: a rule that selects single combinations by their board ratio at
  T-2 (E2). A box averages over a category; a selective rule can still find mispriced
  combinations inside it, and only logged boards can test that.

## Follow-ups filed

All `status:proposed` under epic #72:

- #281 Score #131's E2 exacta ratio-cut rules on logged boards (with the D descriptive on the same
  days; the owner's 2026-11-27 reframe drops its 100-meeting gate).
- #282 Pre-register a fav_first exacta loss-avoidance rule (family L); its cuts become ranked rows.
- #283 Check the ¥100 exacta own-impact assumption against #130 pool totals.
- #284 Refit λ2 per walk-forward fold for #131's B3 and C, so both are strictly out-of-sample.
- #449, the leaderboard writer (not this ticket's): once it lands, every E1 and E2 member is a
  leaderboard row.
