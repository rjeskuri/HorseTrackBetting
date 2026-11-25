# Experiment report: co-placer Place behind heavy favourites (Japan, #83, E15)

Ticket #83 (epic #72; 08 §8 A5, §5.4 E15, R11). Two research reports measured Place on the runners behind a heavy
favourite and disagreed: one found co-placers beating their odds-matched expectation, the other found flat Place on
all other runners returning the same as anywhere else (R11). This study runs 08 §5.4's three steps:
- step 1 puts both definitions on one period and one filter set;
- step 2 freezes a decision-time test (ranks 2–3 selected on the board at T-2) and runs it on the logged days;
- step 3 checks the stake-back-share mechanism against logged Place money.

**Nothing is adopted, and nothing enters Family P from this ticket.** Entry into Family P is the owner's decision
(08 §5.4 Decision). No family config was changed.

Every figure here is an aggregate over the JRA results store (netkeiba race results: final Win odds, finishing
positions, dead heats and PLACE dividends) and the JRA odds logs. No JRDB, no JRA-VAN, and no runner or race row
appears. No figure from the research reports is reproduced here.

- Registry family `japan-coplacer-place-e15`, spec sha256
  `4a54f12ea0d6a56ce0f92110d8a84d4d817765204c6e4800938caf84cf920604`. The spec was frozen before any outcome was
  read. It holds:
  - the period, the common filter, the bands and the pooled rows;
  - both metrics and the odds-matching rule;
  - places paid and the bootstrap;
  - the step-2 rules (decision time, board, qualifying race, selection, primary and secondary metrics, when it is
    scored) and the step-3 rules;
  - the owner decisions, and the Family P statement.
- Code at `fb8b189` (`countries/japan/studies/coplacer.py`). To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.coplacer --run`.
- Store: `processed/jra/runners.parquet@b3226f2a22bfce26`, `processed/jra/races.parquet@4e6a394a57e3101a`,
  `processed/jra/dividends.parquet@dc8b0d185078498e`.
- Logs: `odds_snapshots/jra/2026-11-21/KYO.jsonl.gz@e36a723de2948d4a`,
  `odds_snapshots/jra/2026-11-22/KYO.jsonl.gz@c07e4cfe887e48af`.
- The full report is under the data directory, at `reports/coplacer-japan-4a54f12ea0d6.md`. Every record is in
  `docs/experiments/registry-japan.md`.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D2 | The decision time is T-2, the primary point of the grid T-60/30/16/5/2/1 (`config/decision.yaml`). Step 2 selects at T-2. | `docs/problem-statement.md` §1, #14 |
| D5 | Place's baseline is its own final prices. The odds-matched expectation here is 06's comparator (the final Win price's ventile by places paid). It is measured, not adopted. | §2, #14 |
| D7 | Bounds are meeting-clustered (venue x day), 95% one-sided (`config/scoring.yaml`; 10,000 replicates, seed 83). | §3, #14 |
| D10 | Races before 2014-06-07 are excluded wherever the price enters, so the one period is the price era. | §4, #14 |
| D11–D15 | Holdout A2 from 2026-11-28 is fully locked for outcomes and prices. No holdout row or log was read, and the holdout was not opened. | §5, #14; #71 |
| D21 | No figure derived from a licensed source appears in the repository. None is used here. | #14 |
| 08 §5.4 Decision | Entry into Family P is the owner's. Nothing enters it from this family. | 08 §5.4 |

## Hypothesis

Each hypothesis was written into the frozen spec before any outcome was read. The step-1 failure rules were written
after the run, with this report, and are marked so. The null is the starting point: Place on a co-runner returns
what its final Win price predicts, whatever the favourite's price.

- **Step 1** (08 §5.4): with both definitions in one table, "the gain sits in ranks 2–3 in the odds-matched metric
  and vanishes in an all-others raw average".
  - Failure rule (written after the run): it fails if ranks 2–3 behind a favourite at or under 1.5 show no odds-matched gain (one-sided lower bound at or
    under 0).
  - It also fails if the all-others raw average behind a heavy favourite clearly differs from the one behind a
    favourite above 2.0.
- **Step 2** (08 §4 E15, P2): ranks 2–3 selected on the board at T-2, behind a favourite at or under 1.5 at T-2,
  return above their odds-matched expectation at the final price. The claim rests on the one-sided lower bound being
  above 0. The secondary metric is the absolute Place return.
- **Step 3**: under the stake-back-share rule, a co-placer's dividend rises when a heavily backed favourite misses.
  The open question is whether that alone predicts the odds-matched gap.

## Variants tried

The family size is 6 variants (`Registry.family_size`), with 6 records. Step 1 scored all four of its variants.
Steps 2 and 3 are registry records with status `failed` and the reason "data pending". Their counts are kept as
metrics, so a later run with data scores the same variants.

Data and split:
- The rows are development days from 2014-06-07 to 2026-11-15: 43,058 races, 600,231 priced starters and 3,591
  meetings. 2,150 scratched runners are out. No race had an unpriced starter, and no race was void.
- The common filter keeps races where:
  - Place was sold (1 race dropped);
  - the favourite is unique on the final Win odds (822 dropped);
  - there is no dead heat among finishers 1–3 (258 dropped);
  - no Place ticket was refunded (none in the store).

  That leaves 41,979 races and 585,135 starters with all fields, and 41,220 races and 580,142 starters with 8+
  runners.
- Rank is 1 plus the number of starters at strictly lower final Win odds. The favourite's band uses its final odds.
  That is look-ahead, so step 1 is descriptive.
- Place return is the PLACE dividend over the 100-yen unit, and 0 when the runner did not place.
- The odds-matched metric is Place return minus the mean return of the runner's (final-odds ventile, places paid)
  cell. The cut points are the frame's own Win-odds quantiles at 0.05 to 0.95, computed with favourites included.
- CIs: [lo, hi] are the 5% and 95% quantiles of the bootstrap over meetings. Each is a one-sided 95% bound (D7), so
  the pair is not a two-sided 95% interval. No family-wise correction is applied across cells.

### Step 1: the reconciliation table, all fields (06's slice)

The raw table shows ROI in percent (return - 1). The odds-matched table shows return minus expectation, in points
per 100 staked. Each cell reads value [lo, hi] (n co-runners). "≤ 1.5" pools the first two bands. "2–3" and
"all others" pool ranks.

Raw ROI:

| Co-runner rank | Fav ≤ 1.3 | Fav 1.4–1.5 | Fav ≤ 1.5 | Fav 1.6–2.0 | Fav > 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | -6.2 [-13.0, +0.7] (991) | -4.2 [-8.4, +0.1] (2,032) | -4.9 [-8.4, -1.3] (3,023) | -14.1 [-15.7, -12.5] (8,676) | -19.0 [-19.7, -18.2] (31,124) |
| 3 | -7.8 [-16.2, +0.7] (985) | -11.3 [-16.7, -5.9] (2,003) | -10.1 [-14.7, -5.6] (2,988) | -16.3 [-18.3, -14.3] (8,576) | -21.6 [-22.5, -20.6] (30,252) |
| 2–3 | -7.0 [-13.0, -1.0] (1,976) | -7.7 [-11.3, -3.9] (4,035) | -7.5 [-10.6, -4.4] (6,011) | -15.2 [-16.4, -13.9] (17,252) | -20.2 [-20.8, -19.7] (61,376) |
| 4–5 | -25.0 [-31.7, -18.2] (1,972) | -10.4 [-15.3, -5.4] (4,026) | -15.2 [-19.2, -11.2] (5,998) | -21.1 [-22.9, -19.3] (17,123) | -23.8 [-24.6, -23.0] (60,627) |
| 6–8 | -25.6 [-34.2, -16.5] (2,809) | -14.1 [-20.5, -7.4] (5,845) | -17.8 [-23.1, -12.6] (8,654) | -22.8 [-25.2, -20.5] (25,297) | -24.5 [-25.4, -23.6] (90,649) |
| 9+ | -40.2 [-51.9, -27.3] (4,067) | -35.3 [-42.8, -27.4] (9,334) | -36.8 [-43.3, -29.9] (13,401) | -37.7 [-40.5, -34.9] (45,088) | -34.4 [-35.7, -33.1] (191,680) |
| all others | -27.5 [-33.3, -21.3] (10,824) | -20.9 [-24.7, -16.9] (23,240) | -23.0 [-26.3, -19.7] (34,064) | -27.7 [-29.0, -26.4] (104,760) | -28.4 [-29.0, -27.8] (404,332) |

Return minus odds-matched expectation:

| Co-runner rank | Fav ≤ 1.3 | Fav 1.4–1.5 | Fav ≤ 1.5 | Fav 1.6–2.0 | Fav > 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | +14.0 [+7.2, +20.8] (991) | +15.2 [+11.0, +19.5] (2,032) | +14.8 [+11.2, +18.4] (3,023) | +4.3 [+2.7, +5.9] (8,676) | -1.0 [-1.8, -0.2] (31,124) |
| 3 | +13.4 [+5.1, +21.8] (985) | +9.9 [+4.5, +15.3] (2,003) | +11.0 [+6.5, +15.6] (2,988) | +4.6 [+2.6, +6.6] (8,576) | -1.3 [-2.2, -0.4] (30,252) |
| 2–3 | +13.7 [+7.7, +19.7] (1,976) | +12.6 [+9.0, +16.4] (4,035) | +12.9 [+9.8, +16.1] (6,011) | +4.5 [+3.2, +5.7] (17,252) | -1.1 [-1.7, -0.6] (61,376) |
| 4–5 | -2.3 [-9.1, +4.4] (1,972) | +10.9 [+6.0, +15.9] (4,026) | +6.5 [+2.6, +10.5] (5,998) | -0.1 [-2.0, +1.7] (17,123) | -2.8 [-3.5, -2.0] (60,627) |
| 6–8 | +1.7 [-6.9, +10.8] (2,809) | +10.7 [+4.3, +17.4] (5,845) | +7.8 [+2.6, +13.0] (8,654) | +0.7 [-1.6, +3.0] (25,297) | -3.0 [-3.9, -2.1] (90,649) |
| 9+ | +4.9 [-6.8, +17.8] (4,067) | +7.0 [-0.5, +14.9] (9,334) | +6.3 [-0.1, +13.2] (13,401) | +2.3 [-0.4, +5.1] (45,088) | +0.5 [-0.8, +1.8] (191,680) |
| all others | +4.4 [-1.3, +10.5] (10,824) | +9.5 [+5.7, +13.6] (23,240) | +7.9 [+4.7, +11.2] (34,064) | +1.9 [+0.5, +3.2] (104,760) | -1.0 [-1.6, -0.4] (404,332) |

### Step 1: the reconciliation table, 8+ runners (04's slice)

Raw ROI:

| Co-runner rank | Fav ≤ 1.3 | Fav 1.4–1.5 | Fav ≤ 1.5 | Fav 1.6–2.0 | Fav > 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | -7.8 [-14.5, -0.9] (900) | -5.2 [-9.5, -0.8] (1,918) | -6.0 [-9.7, -2.4] (2,818) | -14.0 [-15.7, -12.4] (8,384) | -19.0 [-19.8, -18.2] (30,852) |
| 3 | -7.0 [-15.5, +1.9] (896) | -9.6 [-15.2, -4.0] (1,885) | -8.8 [-13.4, -4.1] (2,781) | -15.9 [-18.0, -13.9] (8,284) | -21.6 [-22.6, -20.7] (29,991) |
| 2–3 | -7.4 [-13.2, -1.3] (1,796) | -7.4 [-11.1, -3.6] (3,803) | -7.4 [-10.5, -4.3] (5,599) | -15.0 [-16.3, -13.7] (16,668) | -20.3 [-20.8, -19.8] (60,843) |
| 4–5 | -25.4 [-31.9, -18.3] (1,792) | -9.2 [-14.2, -4.2] (3,802) | -14.4 [-18.5, -10.3] (5,594) | -20.6 [-22.4, -18.7] (16,547) | -23.8 [-24.6, -23.0] (60,099) |
| 6–8 | -22.9 [-31.8, -13.6] (2,685) | -12.7 [-19.1, -5.9] (5,684) | -16.0 [-21.3, -10.5] (8,369) | -22.2 [-24.7, -19.9] (24,826) | -24.3 [-25.3, -23.4] (90,208) |
| 9+ | -40.2 [-51.9, -27.3] (4,067) | -35.3 [-42.8, -27.4] (9,334) | -36.8 [-43.3, -29.9] (13,401) | -37.7 [-40.5, -34.9] (45,088) | -34.4 [-35.7, -33.1] (191,680) |
| all others | -27.4 [-33.3, -21.1] (10,340) | -20.6 [-24.5, -16.5] (22,623) | -22.7 [-26.1, -19.3] (32,963) | -27.6 [-28.9, -26.2] (103,129) | -28.4 [-29.0, -27.8] (402,830) |

Return minus odds-matched expectation:

| Co-runner rank | Fav ≤ 1.3 | Fav 1.4–1.5 | Fav ≤ 1.5 | Fav 1.6–2.0 | Fav > 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | +12.7 [+6.1, +19.7] (900) | +14.5 [+10.3, +19.0] (1,918) | +14.0 [+10.4, +17.6] (2,818) | +4.5 [+2.9, +6.1] (8,384) | -1.0 [-1.8, -0.2] (30,852) |
| 3 | +13.7 [+5.2, +22.6] (896) | +11.5 [+6.0, +17.2] (1,885) | +12.2 [+7.6, +16.9] (2,781) | +5.1 [+3.1, +7.2] (8,284) | -1.3 [-2.2, -0.3] (29,991) |
| 2–3 | +13.2 [+7.4, +19.4] (1,796) | +13.0 [+9.3, +16.8] (3,803) | +13.1 [+10.0, +16.2] (5,599) | +4.8 [+3.5, +6.1] (16,668) | -1.1 [-1.7, -0.6] (60,843) |
| 4–5 | -4.2 [-10.7, +2.8] (1,792) | +11.4 [+6.4, +16.4] (3,802) | +6.4 [+2.3, +10.5] (5,594) | +0.1 [-1.7, +1.9] (16,547) | -2.7 [-3.5, -2.0] (60,099) |
| 6–8 | +2.4 [-6.5, +11.7] (2,685) | +11.0 [+4.6, +17.7] (5,684) | +8.2 [+2.9, +13.7] (8,369) | +0.5 [-1.9, +2.9] (24,826) | -3.0 [-3.9, -2.1] (90,208) |
| 9+ | +4.9 [-6.7, +17.8] (4,067) | +6.9 [-0.6, +14.8] (9,334) | +6.3 [-0.1, +13.2] (13,401) | +2.3 [-0.4, +5.1] (45,088) | +0.5 [-0.8, +1.8] (191,680) |
| all others | +4.1 [-1.7, +10.4] (10,340) | +9.7 [+5.8, +13.8] (22,623) | +8.0 [+4.6, +11.3] (32,963) | +1.9 [+0.6, +3.3] (103,129) | -1.0 [-1.6, -0.4] (402,830) |

### Step 1: which part of R11 each definition explains

Both reports' patterns appear in one table, on one period and one filter set. The two definitions measure different
things.

- **The metric explains the largest part.** 04's definition is raw ROI on all others, 8+ runners and a favourite
  at or under 1.3. It returns -27.4 [-33.3, -21.1], against -28.4 [-29.0, -27.8] behind a favourite above 2.0:
  flat, as 04 reported. On the same runners, the odds-matched metric is +4.1 [-1.7, +10.4] against -1.0. The raw
  comparison across bands does not hold the runners' prices fixed. Behind a heavy favourite the other runners are
  longer-priced, and longer prices return less at every favourite band (the favourite-longshot bias). That hides
  most of the gap the odds-matched metric shows.
- **Composition by co-runner rank explains the rest of the raw view.** The gain sits in ranks 2–3:
  - odds-matched, +12.9 [+9.8, +16.1] behind a favourite at or under 1.5 (all fields), against -1.1 behind one above
    2.0;
  - raw, -7.5 against -20.2, so the gap shows even without matching.

  Rank 9+ runners lose 34 to 40% raw in every band. They are 39% of the others behind a favourite at or under 1.3
  (8+ runners) and contribute about -16 of that slice's -27.4. An all-others average is therefore mostly a 9+
  average. Their odds-matched return is near 0 in every band (+0.5 to +7.0, every lower bound at or under 0). They
  lose what their price predicts, no more.
- **The favourite cutoff explains part.** 04's band (at or under 1.3) is where an all-others average gains least:
  - behind a favourite at or under 1.3, ranks 4–8 show no odds-matched gain (-4.2 [-10.7, +2.8] and +2.4 [-6.5,
    +11.7] in 8+ fields);
  - behind one at 1.4–1.5, they show +11.4 and +11.0;
  - ranks 2–3 gain in both bands (+13.2 and +13.0).

  06's cutoff (at or under 1.5) takes in the 1.4–1.5 band, so its all-others figure comes out positive: +7.9 [+4.7,
  +11.2] here. The cutoff matters for an all-others average, not for ranks 2–3. The cells have no family-wise
  correction.
- **Field size explains nothing.** All fields and 8+ runners agree within 2.7 points in every cell (1.9 on the
  odds-matched metric), well inside each cell's bounds. Only 759
  filtered races have 5–7 runners.
- **The period is not tested.** D10 limits the study to 2014-06-07 onward, so 06's earlier years are out. One period
  reproduces both reports' patterns, so the period difference is not needed to explain R11.
- **Both definitions agree nothing is positive.** Ranks 2–3 behind a favourite at or under 1.5 lose -7.5 [-10.6,
  -4.4] raw (all fields). The upper bound rules out a flat-stake return above about -4.4%. The gain is relative to
  the odds curve. It is a "lose-less" choice (08 P2), not an edge, and it was selected on the final price.

### Step 2: the decision-time test (frozen; data pending)

- Spec:
  - selection: on the board at T-2 (`board.snapshot.board_at`; rows available at or before t; post times from the
    sidecar, else HKTrack's card), WIN ranks 2 and 3 in races with five or more priced starters and a unique
    favourite at or under 1.5 at t;
  - primary metric: Place return at the final dividend minus the odds-matched expectation of the runner's final
    Win odds and field size (reference: the all-fields step-1 frame), with D7 bounds, one-sided;
  - secondary metric: absolute Place return;
  - no race is dropped for its outcome;
  - scored once the selections with a result span two or more meetings. The evidence standard is the owner's
    (D16–D18).
- Run on 2 logged days (2026-11-21 and 2026-11-22, KYO, 24 races): 3 races had no board by T-2, 20 had a favourite
  above 1.5 at T-2, and 1 qualified, giving 2 selections.
- Result: **data pending**. The store ends at 2026-11-15, so neither selection has a result yet: 0 selections with a
  result, over 0 meetings. The registry record holds these counts.

### Step 3: the mechanism (data pending)

`place_payout_scenarios` prices a co-placer's dividend through `core.pricing.stake_back_share_dividend`, with the
favourite among its co-placers ('fav_places') or unplaced ('fav_misses').

- The logs carry the PLACE pool total for all 24 logged races, and per-runner Place money for none. The odds log
  holds money on pool-total rows only (`docs/specs/odds-log.md`).
- The Place odds range cannot stand in for that money. A heavy favourite's Place price sits at the 1.1 floor, which
  is exactly where its share is not identified.
- Result: **data pending: Place pool totals (#4)**. The totals are logged; the per-runner money is not.

## Alternatives rejected and why

- **06's period (from 2010).** D10 excludes pre-reform prices. One period was required, and it reproduces both
  patterns.
- **An external odds-matched reference** (all development runners, or a fixed earlier window). The pinned
  definition matches each frame against itself, favourites included. Each field set is therefore matched against
  its own odds curve, and the two field sets agree within 1.9 points on that metric.
- **Matching on the Place price instead of the Win price.** 06's comparator is the Win odds ventile by places paid,
  and Place prices are floor-bound at exactly the heavy favourites in question.
- **Selecting the favourite band at t in step 1.** History has no board at t outside the licensed 10am proxy (#84).
  Step 1 is descriptive at the final price, and step 2 is the decision-time test.
- **Dropping races for dead heats or refunds in step 2.** That conditions on the outcome after t, so step 2 drops
  nothing for its outcome.
- **Deriving per-runner Place money from the displayed Place range (step 3).** It is not identified at the floor
  (see step 3).
- **A family-wise test over the cells.** That belongs to A6's pre-registered Family P study with the holdout, and
  entry into it is the owner's.

## What this rules out

- **R11 as a contradiction of measurement.** On one period and one filter set, both reports' patterns hold at once.
  The raw all-others average behind a favourite at or under 1.3 is flat against other bands (-27.4 vs -28.4), and
  ranks 2–3 behind a favourite at or under 1.5 beat their odds-matched expectation (+12.9 [+9.8, +16.1]). The
  disagreement is one of definition: the metric, the rank composition and the favourite cutoff.
- **A raw edge in ranks 2–3 at the final price.** Raw ROI is -7.5 [-10.6, -4.4] behind a favourite at or under 1.5,
  and every rank and favourite band has a raw upper bound at or under +1.9.
- **Rank 9+ as anything but price.** Their large raw loss (34–40%) is what their odds predict: the odds-matched
  return is near 0 in every band.
- **Not ruled out.** Whether the gain survives selection at T-2 (step 2, data pending) and whether the payout rule
  explains it (step 3, data pending). This study is in-sample and descriptive. It overlaps the years the proposing
  research used, and the bands use the final price.

## Follow-ups filed

None filed by this run. Proposed, for the orchestrator to file under epic #72 as `status:proposed`:

- Re-run step 2 once logged days have results in the store: a store rebuild after each race day, or same-day
  results once JRA publishes them. The frozen spec scores the same variant.
- Find a public source of per-runner Place money, or show that none exists, so that step 3 can run. Until then
  step 3 stays data pending on #4.
- E15 on the JRDB 10am board is #84 (existing; it reuses this module's selection).
