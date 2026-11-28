# Experiment report: the C4 meld of the out-of-fold members at the final Win price (Japan, #98)

Ticket #98 (epic #73; 08 §3.2 C4 and cross-cutting, §4 E41, E42). The run melds the out-of-fold members built so far
into #87's C4 meta-model at the final Win price: #91's dynamic rating, #92's ability model and #93's six interaction
members. Each member is measured standalone and in the meld, the meld's marginal table (all against all minus each
member) shares one set of meeting draws, every member has an encompassing coefficient, and the run tests E41
(precision-weighted α against a constant α) and E42 (gated weights in debut races and handicaps), with the members'
correlations and the Heskes ambiguity. **Nothing is adopted** (D19: every variant is a ranked row), and the holdout
is not opened.

**Licensed figures are withheld (D21).** #92's ability model reads the JRA-VAN sire, so every record that uses it
(the full meld, its ablations but one, every marginal gain, E41, E42, the correlations and the ambiguity) carries the
source `jravan`: the committed summary shows no value for it, and this report names it by option id only. The rating
and the interaction members read the JRA results store alone, so their standalone and encompassing records and the
meld without the ability member are public, and their figures are below. The populations' race sets do not depend on
JRA-VAN data: the ability and interaction exports hold the same 572,772 (race, horse, oof_fold) rows.

- Registry family `japan-c4-meld-final-win`, spec sha256
  `86c86e1970c123fc1b0724ac9559280d05d719d01352d236de8a221b66e4c3dc`, frozen by `run_study` before it read any row.
  The spec holds the members and their files' sha256, the populations, the folds and the scored-quarter rule, the
  fit (stage 2, ridge 1 on member terms), every variant (the κ grid, the gates), the Clark-West statistic, the
  tail and leaderboard rule, D7's bootstrap (meetings, 95% one-sided, 10,000 draws, seed 98) and the decisions.
- **No trial run on real outcomes.** Before the freeze the study ran end to end once on the real member and price
  inputs with simulated winners (each race's winner drawn from the normalised final price; the store's placings
  overwritten before any use, no dividend read, nothing written), to check every code path, the run time and the
  memory. Its input path was also run under CPython 3.14 with pandas 3 (CI's), giving the same counts.
- Code: `countries/japan/models/meld.py` (tested in `tests/countries/japan/test_meld.py`) and
  `countries/japan/studies/meld.py`, at `a0544b2`. To reproduce: `uv run python -m
  horsetrack.countries.japan.studies.meld --run`. On this 16 GB Mac the run took 560 s with a peak resident memory
  of about 1.3 GB. A rerun appends only records the family does not hold.
- Data builds: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`,
  `processed/jra/dividends.parquet` (Win rows of the scored races, read by `studies.interactions.win_paid`) and the
  member exports `features/japan/rating-7ec8931213c6.parquet@7544cd84b8021f55` (#91),
  `features/japan/ability-japan-12b1eed01095-9a3f58d49fb3.parquet@aa7c6c991ea516bf` (#92, family
  `japan-c2-ability-final-win-v2`) and `features/japan/interactions-japan-2dde55836222-28bea4f0bfa3.parquet@1278d57c65703be8`
  (#93, family `japan-c2-interactions-final-win-v2`).
- The full report is under the data directory at
  `$HORSETRACK_DATA_DIR/reports/meld-japan-86c86e1970c1-e13d304b7ffe.md`, with its tables as JSON beside it; both
  were written through `write_report` with the sources `jra` and `jravan`. Every record names the stem in
  `data_builds.meld_artifacts`. The records are in `docs/experiments/registry/japan/japan-c4-meld-final-win.md`.
  The run's log is `$HORSETRACK_DATA_DIR/logs/meld-98-run-a0544b2.log`.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Shrinkage target | α = 0, the calibrated market alone: the ridge falls on every member term and never on the market's. | #98, owner comment 2026-11-24 |
| Gates and segment weights | Pre-registered by Claude as variants (E42: debut races, handicaps, both). | #98, owner reframe 2026-11-27 |
| Standalone, marginal, encompassing | Columns beside each ranked row, not adoption thresholds. | #98, owner reframe 2026-11-27 |
| JRDB and JRA-VAN as inputs | Allowed in private models; every such result stays out of the repository. | #98, 2026-11-24; D21 |
| D9 | Quarterly expanding walk-forward, 1-week embargo, 5 seasons of level-2 training on every scored fold. | Problem statement; config/folds.yaml |
| D7 | Meetings (venue × day), 95% one-sided bounds. | Problem statement; config/scoring.yaml |

## Design

- **Level 2.** `studies.meta.store_frame`'s priced starters of run races with a winner on development days (holdout
  days are dropped by `read_store_table` and `split_off`; the holdout is never opened), q the normalised final Win
  price. Each row is labelled with the D9 quarter that tests it. `models.meld.member_frame` aligns each member by race
  and horse and refuses a row whose `oof_fold` is not that quarter; `models.stacking.join_members` and `check_level2`
  then check every member's provenance and each scored quarter's level-2 training window.
- **Populations.** `from2020`: every member, from 2020-Q1 (the rating's first out-of-fold quarter). `from2015`: every
  member but the rating, from 2015-Q1. The quarters before the first one with 5 seasons of level-2 training form one
  seed block that only trains. **Scored:** `from2020` on 7 quarters, 2025-Q2 to 2026-Q4 (to 2026-11-23), 5,729 races
  in 478 meetings; `from2015` on 27 quarters, 2020-Q2 to 2026-Q4, 22,959 races in 1,914 meetings. Each scored quarter
  is predicted by fits on every earlier row of its population at least 7 days before its first race day.
- **Fit.** `models.meld.fit_meld` in the stage-2 form, u = Σ_k α_k(z) w_ik m_ik + β ln q, ridge 1 on every member
  term (a N(0, 1) prior: decisive only where the races cannot inform a term, such as the rating's weight in debut
  races, where every runner has the prior rating), none on β. The comparator is the market-only stage-2 meta
  (u = β ln q) fitted the same way on the same rows.
- **Scores.** ΔLL is the comparator's winner log loss minus the variant's, per scored race, with D7's paired meeting
  bootstrap (one seed and the same races for every variant, so the marginal table shares its draws). Each model's
  return is N5's top claim band at the final price: a flat Win bet on every scored runner whose out-of-fold p times
  the final odds is at or above the 99th percentile of the variant's claims, settled at the recorded Win dividend.

## Hypothesis

From the frozen spec. E40 (the C4 meld): a member, or the meld, beats the market-only meta out of fold; the null is
that none does, which is what Bates-Granger predicts for members whose errors correlate with the market's (05 §5.2):
α = 0 is the signature of a member with nothing beyond the price. E41: precision weighting α_i = α_0 / (1 + κσ_i²)
beats a constant α out of fold; the null is a ΔLL and a Clark-West statistic whose bounds cover 0. E42: the members'
weights differ between debut races, handicaps and other races; the null is a gated meld no better than the pooled
one, in any cell.

## Variants tried

Every variant is one record, failures included: 87 in all (53 withheld). Bounds are D7's one-sided 95% pair (the 5%
and 95% bootstrap quantiles); ΔLL in nats per race against the market-only meta; return per ¥100 in the top claim
band (its bets in brackets).

Public, `from2015` (27 quarters from 2020-Q2):

| Variant | ΔLL vs market only | Return per ¥100, top band |
| --- | --- | --- |
| market_only | (the comparator) | 75.71 (72.20 to 79.21; 3,156 bets) |
| standalone:ix_base | +0.000319 (+0.000138 to +0.000498) | 81.29 (76.36 to 86.53) |
| standalone:ix_horse_x_jockey | +0.000319 (+0.000138 to +0.000498) | 81.29 (76.36 to 86.53) |
| standalone:ix_trainer_x_track | +0.000308 (+0.000130 to +0.000485) | 82.42 (77.35 to 87.55) |
| standalone:ix_horse_aptitude | +0.000314 (+0.000136 to +0.000489) | 81.80 (76.71 to 86.97) |
| standalone:ix_trainer_debut | +0.000299 (+0.000129 to +0.000466) | 83.58 (78.27 to 88.92) |
| standalone:ix_transfer_step | +0.000316 (+0.000137 to +0.000494) | 81.05 (75.99 to 86.18) |
| meld:all minus ability (the six interaction members) | +0.000195 (−0.000044 to +0.000434) | 80.91 (75.26 to 86.73) |

Encompassing (`from2015`, in-sample on the scored rows, meeting-clustered): every interaction member's coefficient
beside ln q is negative, −0.0523 to −0.0534, with upper bounds −0.0316 to −0.0327.

Public, `from2020` (7 quarters from 2025-Q2):

| Variant | ΔLL vs market only | Return per ¥100, top band |
| --- | --- | --- |
| market_only | (the comparator) | 83.02 (74.83 to 91.21; 784 bets) |
| standalone:rating | +0.000596 (+0.000125 to +0.001065) | 88.42 (72.80 to 105.04) |
| standalone:ix_base | +0.000808 (+0.000314 to +0.001297) | 79.08 (67.41 to 91.00) |
| standalone:ix_horse_x_jockey | +0.000808 (+0.000314 to +0.001297) | 79.08 (67.41 to 91.00) |
| standalone:ix_trainer_x_track | +0.000802 (+0.000323 to +0.001275) | 79.01 (67.51 to 90.97) |
| standalone:ix_horse_aptitude | +0.000798 (+0.000311 to +0.001282) | 79.08 (67.54 to 91.02) |
| standalone:ix_trainer_debut | +0.000802 (+0.000323 to +0.001280) | 80.41 (68.85 to 92.74) |
| standalone:ix_transfer_step | +0.000812 (+0.000326 to +0.001296) | 79.35 (67.96 to 91.60) |
| meld:all minus ability (the rating and the six interaction members) | +0.000422 (−0.000183 to +0.001015) | 79.01 (66.71 to 91.84) |

Encompassing (`from2020`): the rating −0.0677 (−0.1048 to −0.0307); the interaction members −0.0873 to −0.0886, upper
bounds −0.0481 to −0.0494.

Withheld (licensed source; option ids under Leaderboard rows): in each population, the ability member standalone and
its encompassing coefficient; the meld of every member and each ablation that keeps the ability member; every
member's marginal gain (relative to the full meld); E41, the meld with the rating's or the ability member's α
precision-weighted at κ = c / median(σ²), c ∈ {0.25, 1, 4}, against the constant-α meld (paired ΔLL and Clark-West);
E42, the meld gated on debut races, on handicaps and on both, against the pooled meld overall and per cell, with each
cell's in-sample α; the members' within-race correlations and the ambiguity, each with meeting-bootstrap bounds.

Not run (failed records, each with its reason): `not_run:competition` (#94's terms), `not_run:lineage` (#95's
members), `not_run:p_speed` (#97) and `e41:reff` (E41's R_eff weighting): none of these has an out-of-fold export per
runner for the meld.

## Alternatives rejected and why

- **One population.** The rating is out of fold only from 2020-Q1, so a meld with every member scores only from
  2025-Q2 under D9's 5 seasons. The second population (without the rating, from 2015-Q1) scores four times the races.
- **D4's calibrated market as the comparator.** #92's review showed it cannot isolate a member (its market term
  differs); the market-only stage-2 meta, fitted on the same rows, can.
- **Ridge 0.** A gated cell in which a member is constant within every race (the rating in debut races) has no
  information, and the unpenalised fit is singular; ridge 1 sets such a term to α = 0, the owner's target, and is
  negligible on a term the races inform.
- **Segment slopes for E42.** With race-level 0/1 gates and no race in both cells, segment slopes and gate cells span
  the same design; the gates are run.
- **A choice of κ.** Each κ is its own ranked variant (D19); none is selected.
- **Neural or tree metas.** Not excluded (the owner's reframe); a proposed follow-up, expected to rank low.

## What this rules out

At the final Win price, on 22,959 races from 2020-Q2, each jra-only interaction member's stage-2 meta beats the
market-only meta by about +0.0003 nats per race (every lower bound at or above +0.00013) with a negative weight beside
ln q, as #93 found for its base; the six melded together do no better (+0.00019, bounds −0.00004 and +0.00043): the
members are near copies of one another, so the meld spends parameters for no new information. On the 5,729 races
from 2025-Q2 the rating and each interaction member also beat the market-only meta, again with negative weights. No
variant's top claim band returns above 100 per ¥100: every public upper bound is below 100 except the rating's
(105.04, on 784 bets), and the market-only band itself returns 75.71 (from 2020-Q2) and 83.02 (from 2025-Q2). The
gains come with negative member weights beside ln q, so they read as a correction of the price's calibration through
the member, not a view that pays against the price. Nothing is adopted; each row ranks on the leaderboard.

## Leaderboard rows

87 option ids, `japan-c4-meld-final-win/<variant>`, each with 87 variants searched. Public and measured (rank among
the public board's 442 ranked rows when it is rendered with `--public`; the WIN pool's shrinkage pulls every one to
the same value, so they order by their raw return): `from2020:standalone:rating` 170,
`from2015:standalone:ix_trainer_debut` 174, `from2020:market_only` 175, `from2015:standalone:ix_trainer_x_track` 176,
`from2015:standalone:ix_horse_aptitude` 177, `from2015:standalone:ix_base` 178,
`from2015:standalone:ix_horse_x_jockey` 179, `from2015:standalone:ix_transfer_step` 180, `from2015:meld:all minus
ability` 182, `from2020:standalone:ix_trainer_debut` 184, `from2020:standalone:ix_transfer_step` 188,
`from2020:standalone:ix_base` 189, `from2020:standalone:ix_horse_aptitude` 190, `from2020:standalone:ix_horse_x_jockey`
191, `from2020:meld:all minus ability` 192, `from2020:standalone:ix_trainer_x_track` 193, `from2015:market_only` 195.
Public and unmeasured (no return): the 13 encompassing rows of the rating and the interaction members, and the four
not-run records. Withheld (D21), by option id only: in `from2020`, `standalone:ability`, `meld:all`, `meld:all minus
rating` and `meld:all minus ix_<block>` for each of the six blocks, `e41:rating:c[0.25]`, `e41:rating:c[1]`,
`e41:rating:c[4]`, `e41:ability:c[0.25]`, `e41:ability:c[1]`, `e41:ability:c[4]`, `e42:gates[debut]`,
`e42:gates[handicap]`, `e42:gates[debut,handicap]`, `marginal:<member>` for the eight members,
`encompassing:ability`, `correlations` and `ambiguity`; in `from2015`, the same without the rating's variants.

## Follow-ups filed

None filed by this PR; proposed for the owner to file (status:proposed under epic #73): out-of-fold member exports
for #94's competition terms, #95's lineage members and #97's p_speed, so a rerun can meld them; E41's R_eff weighting
from #94's race comparability; a `MetaAdapter` for a `MeldFit`, so C6 can read the meld; and a neural or tree meta
as a ranked variant.
