# Experiment report: exotic pool-choice rules L1, L4 and L5 on the same races (Japan, #81)

Ticket #81 (epic #72; 08 §8 A4, exotic part; §5.1 L1, L4, L5; §4 E11, E13, E16). Family L (D18): each rule is
compared with its stated comparator on the same races, by the paired per-race difference in realised return
(payout / stake) at final dividends, with a one-sided bound. A rule only has to lose less. Nothing here claims a
profit. **No rule is adopted.** Adoption and the Family L evidence standard are the owner's (08 §9.10-11), and
neither is decided here. Every figure is an aggregate over the JRA results store (netkeiba race results and
dividends) and the shipped boost calendar (`config/takeout_boosts.yaml`). There is no JRDB or JRA-VAN input and no
race or runner row.

- Registry family `japan-family-l-exotic-pool-choice`, spec sha256
  `211b05e12eedf3b163b2f99d7a04a6cec8fa23ccf20e6dd13404c63ded2569a1`. `run_study` froze the spec before it read any
  row. The spec holds the rules, both threshold grids, the comparators, the stakes, the selections, the decision
  time (T-2), the board fields, the bootstrap, the owner decisions and the 18 variant names (`family_size` 18).
- Code at `d05026c` (`countries/japan/studies/avoidance.py`). Owner inputs are in
  `countries/japan/config/avoidance.yaml`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.avoidance --run`. The family is frozen, so a rerun with
  this spec adds runs of the same variants. Another spec is refused (`FamilyAlreadyFrozen`).
- Store: `processed/jra/races.parquet@9cce511831074b64`, `processed/jra/runners.parquet@6f3cec95347e0132`,
  `processed/jra/dividends.parquet@b3d635352bb007df`, `processed/jra/odds_timeseries (2 days)@7bd8e900bd0a1e6e`.
- The full report is under the data directory, at `reports/exotic-pool-choice-japan-211b05e12eed.md`. Each
  dropped race and its reason is beside it, in `-dropped.csv`. Every record is in
  `docs/experiments/registry/japan/japan-family-l-exotic-pool-choice.md`.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D2 | The board at t is read at post - 2 min (`config/decision.yaml` `primary_min`). | `docs/problem-statement.md` §1, #14 |
| D3 | Every JRA pool is in research scope, with exotics at tier 2. | §1, #14; amended 2026-11-24 (#127) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided (`config/scoring.yaml`). | §3, #14 |
| D10 | Only races from 2014-06-07: every comparison reads a price or the per-race takeout. | §4, #14 |
| D11, D14, D15 | Holdout A2 from 2026-11-28 is fully locked for outcomes and prices, and only the owner opens it. It is not read or opened here. | §5, #14; the #71 answer (`config/holdout.yaml`) |
| D17 | The family size counts every variant, each L1 grid member included: 18. | §6, #14; the #81 answer |
| D18 | Family L is a paired difference against a stated comparator on the same races, one-sided. A rule only has to lose less. | §6, #14 |
| D21 | No figure from a licensed source is in the repository. None is used. | §8, #14 |
| #81 answer | `l1_threshold_trio` [20, 50, 100] and `l1_threshold_quinella` [5, 10, 20] (decimal odds), as grids of variants; `sets: every_set`; `l4_market_order_ties: lower_horse_no`; `l4_trifecta_comparators: [trio, box]`; `l5_mode: report`. | #81, owner comment of 2026-11-25 |
| #81 answer | L4's sign: rule = the do-not-bet ticket, comparator = the equivalent ticket, so a do-not-bet ticket that loses more shows negative. The same-bracket variant is labelled 'card only (no price read)'. | #81, 2026-11-25 |

Not decided, and not needed for this run: the Family L evidence standard and the adoption of any L rule (08 §9.10-11).

## Hypothesis

The frozen spec's `hypotheses`, in full:

- **L1 (E11).** "A set in the direct pool when short and in the ordered synthetic when long loses less than always
  the direct pool, on the same sets." For a set priced under X, the rule bets the TRIO or QUINELLA. At X or above,
  it bets the six TIERCE or two FORECAST orders, with the stake split evenly. The comparator always bets the direct
  pool.
- **L4 (E13).** "The same-bracket BRACKET_QUINELLA and the market-order trifecta return less than the equivalent
  ticket in another pool."
- **L5 (E16's arithmetic).** "The payout rate of each exotic pool on boosted races." This is a report, not a test
  (`l5_mode: report`).

The direction was frozen with the spec: one-sided (D18), L1 positive, and L4 negative by the owner's sign. The reading
of the bounds is D18's: L1 holds for a variant if its lower bound is above 0, and L4 holds if its upper bound is below
0. The null is the starting point: no difference between a rule and its comparator.

## Variants tried

The family has 18 variants (`Registry.family_size`) and 18 records, all from one run at git `d05026c`. 10 scored
(9 comparisons and the L5 report). 8 are recorded as failed, with data pending.

Data and split:
- The rows are development days from 2014-06-07 to 2026-11-23: 1,353 race days, 3,597 meetings and 43,130 races.
  All are run races. There are 601,265 starters (2,156 scratched runners are out).
- The study read the dividend rows of BRACKET_QUINELLA, QUINELLA, FORECAST, TRIO and TIERCE for those races: 214,376
  rows, none a refund.
- No day in the store is a holdout day (the store ends 2026-11-23, and A2 starts 2026-11-28). The readers drop
  holdout days anyway, and `LockedHoldout` was not opened.
- Nothing is fitted. Every development race is used once (`paired_same_races:development`).
- On history, only the winning combination's final price exists. A losing set stakes the same and pays nothing on
  both sides, so each race's difference comes from its paying rows and the number of sets bet. The sets are every
  combination of the race's starters, flat: 600 yen per TRIO set and 200 yen per QUINELLA pair.

Selections:
- `final (look-ahead)`: the choice reads the final price. For L1, that is the winning set's TRIO or QUINELLA dividend
  / 100. For the market-order trifecta, it is the final Win odds. The final price is not known at a decision time.
- `board at t (T-2)`: odds_timeseries through `read_asof` at the race's `post_utc` - 2 min.
- `card only (no price read)`: the card's brackets and scratches.

CIs: the paired difference is the mean over paired races of rule return minus comparator return, per unit staked.
Lower and upper are the 5% and 95% quantiles of the paired bootstrap over meetings (D7; 10,000 replicates, seed 81).
Each is a one-sided 95% bound. The bounds are per variant, with no family-wise correction.

### L1: the price rule against always the direct pool

A positive difference means the price rule loses less.

| Pool | X | Selection | Races | Dropped | Rule return | Comparator return | Paired difference | Lower (5%) | Upper (95%) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRIO | 20 | final (look-ahead) | 43,130 | 0 | 0.56784 | 0.53499 | +0.03285 | +0.02576 | +0.04042 |
| TRIO | 50 | final (look-ahead) | 43,130 | 0 | 0.56945 | 0.53499 | +0.03445 | +0.02737 | +0.04204 |
| TRIO | 100 | final (look-ahead) | 43,130 | 0 | 0.56954 | 0.53499 | +0.03454 | +0.02749 | +0.04209 |
| QUINELLA | 5 | final (look-ahead) | 43,130 | 0 | 0.59925 | 0.60181 | -0.00256 | -0.00642 | +0.00136 |
| QUINELLA | 10 | final (look-ahead) | 43,130 | 0 | 0.60008 | 0.60181 | -0.00173 | -0.00559 | +0.00218 |
| QUINELLA | 20 | final (look-ahead) | 43,130 | 0 | 0.60154 | 0.60181 | -0.00027 | -0.00409 | +0.00364 |
| TRIO | 20, 50, 100 | board at t (T-2) | 0 | 0 | | | data pending | | |
| QUINELLA | 5, 10, 20 | board at t (T-2) | 0 | 24 | | | data pending | | |

Each data-pending row stands for three variants, one per X, and each is its own registry record.
- TRIO at t: no development day has a logged TRIO board. The logged days (2026-11-21 and 2026-11-22, Kyoto)
  predate tier-2 logging (#130).
- QUINELLA at t: 24 races have a logged QUINELLA board, and all 24 were dropped as "the rule does not cover it". The
  frozen board at t is `post_utc` - 2 min, with `post_utc` taken from the race's latest capture. On both days
  `post_utc`, `fetched_utc` and `minutes_to_post` are null on every row: the logs have no `KYO.races.json`
  post-time sidecar (#22), so there is no t and the board at t is empty. The report labels these variants "no
  QUINELLA board logged at t". The board is logged, but it has no post time. #352 backfills the sidecar and re-runs
  the same family.

### L4: the do-not-bet tickets against the equivalent ticket

The difference is the do-not-bet ticket's return minus the equivalent ticket's. A negative difference means the
do-not-bet ticket loses more.

| Rule | Comparator | Selection | Races | Dropped | Rule return | Comparator return | Paired difference | Lower (5%) | Upper (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-bracket BRACKET_QUINELLA, every same-bracket pair | The QUINELLAs within the bracket, stake split evenly | card only (no price read) | 41,168 | 1,962 | 0.51420 | 0.56708 | -0.05288 | -0.07081 | -0.03659 |
| Market-order TIERCE (three shortest Win prices, in order; ties to the lower horse number) | TRIO on the same three horses | final (look-ahead) | 43,130 | 0 | 0.65579 | 0.77109 | -0.11529 | -0.15262 | -0.07650 |
| Market-order TIERCE | The six TIERCE orders (box), stake split evenly | final (look-ahead) | 43,130 | 0 | 0.65579 | 0.67666 | -0.02087 | -0.06137 | +0.02152 |
| Market-order TIERCE | TRIO | board at t (T-2) | 0 | 24 | | | data pending | | |
| Market-order TIERCE | box | board at t (T-2) | 0 | 24 | | | data pending | | |

- Same-bracket drops: 1,951 races have no BRACKET_QUINELLA dividend row, and every one has 8 or fewer starters (the
  pool is not sold in small fields). 11 races have no bracket holding two starters: each had 9 declared and 8
  starters.
- Market-order at t: 24 races have a logged WIN board, and all 24 were dropped for the cause given under L1 (no
  `post_utc`, so no board at t). The report labels them "no WIN board". #352 covers them.

### L5: the takeout calendar (report only, no test)

| Pool | Ordinary races | Super premium | Ultra premium | Premium | Base rate | Paid rate on boosted races |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BRACKET_QUINELLA | 42,507 | 539 | 84 | | 0.775 | 0.80 (super), 0.85 (ultra) |
| QUINELLA | 41,606 | 539 | 84 | 901 | 0.775 | 0.80 (super), 0.85 (ultra), 0.825 (premium) |
| FORECAST | 42,507 | 539 | 84 | | 0.75 | 0.80, 0.85 |
| TRIO | 42,507 | 539 | 84 | | 0.75 | 0.80, 0.85 |
| TIERCE | 42,507 | 539 | 84 | | 0.725 | 0.80, 0.85 |

The calendar flags all 84 ultra-premium races and all 901 QUINELLA premium races as "boost may be withheld". Every
development race got a rate: none needed a conditions label it lacked. The ticket expected about 240 boosted races.
The shipped calendar (#76, #197) gives 623 super- and ultra-premium development races, plus the 901 QUINELLA premium
races.

The takeout-neutral ratio is the ordered pool's payout rate over the direct pool's. It is the payout ratio of the
synthetic to the direct pool if both priced the event alike.

| Ordered / direct | Boosts | Races | Neutral ratio |
| --- | --- | ---: | ---: |
| TIERCE / TRIO | ordinary | 42,507 | 0.96667 |
| TIERCE / TRIO | super premium, ultra premium | 539, 84 | 1 |
| FORECAST / QUINELLA | ordinary | 41,606 | 0.96774 |
| FORECAST / QUINELLA | QUINELLA premium only | 901 | 0.90909 |
| FORECAST / QUINELLA | super premium, ultra premium | 539, 84 | 1 |

## Alternatives rejected and why

- **One L1 threshold per pool.** The owner registered grids, each member counted in the family size (#81 answer). A
  single X picked after the run would have been a hidden fourth or fifth variant.
- **A decision-time choice on history.** No historical exotic board exists. The history rows read the final price
  and are labelled look-ahead. The board-at-t variants are frozen in the same family and score when boards exist
  (#352, #353).
- **Taking t from the card's post time (`races.post_time_local`) when the capture has none.** The frozen spec
  defines t from the capture's `post_utc`. Changing that source would change the frozen definition, so a second
  family would be needed. #352 backfills the logger's post-time sidecar instead, and the spec stands.
- **A selection of sets** (model-chosen or price-banded). `sets: every_set` is the only option built and the one the
  owner chose. A selection at t needs its own ticket.
- **L5's same-race comparison against a base-rate dividend.** The owner chose `report`. It would also be
  approximate: rebuilding a dividend at the base rate ignores the pool's breakage.
- **`drop_race` for tied Win prices.** The owner chose `lower_horse_no`.
- **A family-wise test over the 18 variants.** D18 signs Family L as paired and one-sided. Whether and how Family L
  corrects for the grid is the owner's evidence standard (08 §9.10), which is open. The bounds here are per variant.

## What this rules out

All of this is on history, development days 2014-06-07 to 2026-11-23. Nothing below adopts a rule.

- **"Always the TRIO" as the better way to hold every set, at the final price.** For each X, the price rule loses
  less, by +0.033 to +0.035 per unit staked. The one-sided lower bounds are +0.026, +0.027 and +0.027 for X = 20, 50
  and 100. The upper bounds rule out a gain above about +0.042. X = 100 keeps the whole point estimate, so the gain
  comes from sets priced 100 or more. Sets priced 20 to 50 count slightly against it. No CI was computed for the
  difference between thresholds. This is
  look-ahead: the choice reads the winning set's final TRIO price. If winning sets shorten late, a choice at T-2
  sends more of them to the synthetic than this run did, and the gain at t may differ. Only the TRIO board at t can
  measure that (#353).
- **A sizeable QUINELLA-or-exactas effect, at the final price.** For X = 5, 10 and 20, the difference is -0.0026,
  -0.0017 and -0.0003, and each CI covers 0. A gain above +0.0014, +0.0022 or +0.0036 is ruled out, and a loss below
  -0.0064, -0.0056 or -0.0041. Holding a pair as two exactas instead of a quinella changes the return by well under
  1% of stake either way. L1 for QUINELLA is not supported.
- **Parity between the same-bracket bracket quinella and the quinellas inside the bracket.** The same-bracket ticket
  returns 0.053 less per unit staked, with bounds -0.071 and -0.037 (41,168 races). This variant reads no price, so
  its choice is available before the race. The one exception is the accepted leak of race-day scratches, which come
  from the results page (owner decision of 2026-11-25, `config/availability.yaml`).
- **Parity between the market-order trifecta and the TRIO on the same three horses**, at the final Win order. It
  returns 0.115 less per unit, with bounds -0.153 and -0.077. Against the box of six trifectas, the difference is
  -0.021 and its CI covers 0, so that variant is not supported. A loss beyond -0.061 against the box is ruled out,
  and so is a gain above +0.022.
- **A multiplicity check (descriptive, not the Family L standard).** Read as normal approximations, the three L1
  TRIO bounds and the two supported L4 bounds sit about 5 to 8 bootstrap standard errors from 0. A Bonferroni bound over
  the 18-variant family is about 2.8. The QUINELLA rows and the box row sit within 1.1.
- **Nothing at a decision time.** All 8 board-at-t variants are data pending: 3 with no TRIO board logged, and 5
  with boards logged but no post time.
- **L5 is arithmetic, not evidence.** On ordinary races the ordered pools pay about 3.3% (TIERCE) and 3.2% (FORECAST)
  less, by takeout alone. On super- and ultra-premium races the pools pay alike, and the 901 QUINELLA premium races
  tilt toward the QUINELLA. E16's pricing test on boosted days is #99's.

## Follow-ups filed

Both are filed under epic #72 as `status:proposed`.

- #352: backfill the post-time sidecar for the logged development days, recompact them, and re-run this frozen
  family so the QUINELLA and market-order board-at-t variants score. It also makes the pending reason name a
  missing post time.
- #353: re-run this frozen family once TRIO boards are logged on development days (tier 2, #130), so the L1 TRIO
  board-at-t variants score next to the look-ahead rows.

## On the leaderboard (2026-11-27)

Under the reframe (epic #447; D16 to D18 as amended 2026-11-27 in #448), the results above are measured results, not
adoption verdicts. Nothing is adopted or rejected: every option, model and member will be a row on the planned
`horsetrack japan leaderboard` (#452; this revision does not have the command yet), ranked by its shrunk return per
¥100 at the final price, and a row with no measured return will be listed after the ranking, never dropped. Where
this report says "not adopted", "rules out" or "rejected", read a measured position. This section reopens and
re-runs nothing.

- Registry family: `japan-family-l-exotic-pool-choice`.
- As task 4 (#451, planned) specifies, each L1 and L4 rule and its comparator will be a measured row (comparators
  deduplicated). The board-at-t variants, data pending above, will be listed with their failed status and no return.
  L5, the takeout calendar, will be listed with no return.
- Each row's selection label comes from that row's own selection inputs, not from the variant it was scored in:
  - L1: the threshold rule reads the winning set's final TRIO or QUINELLA price, so its rows are look-ahead
    (decision time for the board-at-t variants). Its historical comparator, always the TRIO or always the QUINELLA
    on every combination of the starters, reads no price and is selectable from the card, so it carries no
    look-ahead label: one card-only row per pool, the same for every X. The board-at-t comparator bets only the
    combinations priced on the board at t (`_priced_sets`), so a partial board gives it other tickets than the
    historical comparator and an empty board leaves it unmeasured: it is a decision-time row of its own, one per
    pool, the same for every X, listed with no return in this revision (the board-at-t variants are data pending).
  - L4, same-bracket: the rule and its comparator, the QUINELLAs within the bracket, read only the card's brackets
    and scratches, so both rows are card only.
  - L4, market-order TIERCE: the rule takes its three horses from the final Win order, and both comparators, the
    TRIO and the box, bet those same three horses, so all three rows are look-ahead (decision time at t).
  - Each paired difference keeps its rule's label: look-ahead for the L1 final-price rows and both market-order
    comparisons, decision time for the L1 board-at-t rows (no return in this revision), card only for the
    same-bracket comparison.
  - #451 as written labels every row from its variant's suffix, comparators included; the historical L1 comparator
    rows need the card-only label instead, and the board-at-t L1 comparator rows keep the decision-time label, one
    row per pool rather than one per X.
- "No rule is adopted" above now reads as a measured position: under D18 as amended, a loss-avoidance rule and its
  stated comparator will each be a row, with the paired difference beside them, and no Family L standard is needed
  before either ranks. The re-runs filed as #352 and #353 stay follow-ups.
