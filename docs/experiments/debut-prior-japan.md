# Experiment report: #93's trainer debut prior by segment at the final Win price (Japan, #573)

Ticket #573. #93's `trainer_debut` member (a debut indicator and a (trainer, debut) variance component on top of the
base) is cut by race segment, `<race_class>/debut` when any starter debuts and `<race_class>/seasoned` otherwise, and
scored at the final Win price from #93's out-of-fold level-1 export. **Nothing is adopted** (D16-D19): every segment is
a ranked row. The holdout is not opened. Every figure is an aggregate over the JRA results store (netkeiba race results,
final Win odds and Win dividends; no JRDB, no JRA-VAN), so the figures are public. No runner or race row appears.

- Registry family `japan-debut-prior-segments-final-win`, spec sha256
  `dd5879a0485442c9aa30ea783b48307e331093761fc925b6600532c82a6f16de`, frozen by the run before it read any store row.
  The pre-registration is `docs/specs/debut-prior-prereg.md`.
- Code: `countries/japan/studies/debut_prior.py`. To reproduce, run `uv run horsetrack japan debut-prior` (it reads
  #93's v2 export named by the latest scored record of `japan-c2-interactions-final-win-v2`).
- Private copy of the table: `$HORSETRACK_DATA_DIR/reports/debut-prior-japan-dd5879a04854.md` and its JSON.
- Records: `docs/experiments/registry/japan/japan-debut-prior-segments-final-win.md` (19 scored, one per segment).
- Scored races: 23,871 over 28 folds (D9, 2020-Q1 to 2026-Q4), the same as #93's.

## Hypothesis

- The trainer's debut prior, which #93's own run showed adds nothing over all races at the final Win price (every
  paired ΔLL's upper bound at or below +0.000014), may add where debut runners are common: debut and maiden races
  (the owner's hypothesis on #438, "maidens and debuts versus the big G1 races").
- **Null**, the starting point: ΔLL 0 in every segment, and the member's top-claim bets return no more than the
  market's.
- It would be shown wrong, for this segment, by a lower ΔLL bound at or below 0 (D7, one-sided 95%).

## Variants tried

The 19 segments with a scored race, one registry record each, none failed. ΔLL is the member's stage-2 meta against the
base's stage-2 meta, nats per race, with D7's one-sided 95% meeting bounds (10,000 resamples, seed 573). The tail is the
top 1% of the segment's claims p x O_final, flat unit stakes at the recorded Win dividends, return per ¥100 with the
same bounds.

| rank | segment | races | ΔLL | ΔLL bounds | tail bets | tail return per ¥100 | tail bounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | OP/debut | 50 | +0.00061 | -0.00032, +0.00167 | 7 | 0 | 0, 0 |
| 2 | G1/debut | 20 | +0.00040 | -0.00047, +0.00133 | 4 | 275 | 0, 550 |
| 3 | OP/seasoned | 666 | +0.00012 | -0.000003, +0.00025 | 89 | 80.3 | 47.9, 118.4 |
| 4 | C2/debut | 68 | +0.00011 | -0.00030, +0.00051 | 10 | 61.0 | 0, 114.4 |
| 5 | G3/seasoned | 486 | +0.00007 | -0.00010, +0.00024 | 72 | 63.2 | 27.8, 107.5 |
| 6 | G2/seasoned | 273 | +0.00005 | -0.00017, +0.00027 | 38 | 49.7 | 19.0, 82.7 |
| 7 | C3/seasoned | 1,441 | +0.00002 | -0.00006, +0.00011 | 207 | 76.9 | 53.0, 105.0 |
| 8 | MAIDEN/seasoned | 5,815 | -0.00001 | -0.00005, +0.00003 | 798 | 89.5 | 77.5, 102.4 |
| 9 | C1/debut | 266 | -0.00003 | -0.00030, +0.00026 | 37 | 64.6 | 32.9, 98.2 |
| 10 | MAIDEN/debut | 2,755 | -0.00003 | -0.00014, +0.00008 | 428 | 75.5 | 58.3, 94.3 |
| 11 | G1/seasoned | 155 | -0.00003 | -0.00033, +0.00026 | 26 | 188.1 | 94.8, 303.8 |
| 12 | DEBUT/debut | 2,074 | -0.00004 | -0.00036, +0.00027 | 274 | 80.8 | 65.8, 96.5 |
| 13 | C1/seasoned | 6,158 | -0.00004 | -0.00008, -0.000005 | 816 | 86.2 | 74.7, 98.5 |
| 14 | C2/seasoned | 3,163 | -0.00005 | -0.00011, +0.000003 | 418 | 81.0 | 68.1, 94.9 |
| 15 | L/seasoned | 422 | -0.00007 | -0.00023, +0.00009 | 58 | 71.6 | 35.5, 112.5 |
| 16 | C3/debut | 20 | -0.00010 | -0.00082, +0.00061 | 3 | 80.0 | 80, 80 |
| 17 | G2/debut | 8 | -0.00030 | -0.00121, +0.00058 | 2 | 0 | 0, 0 |
| 18 | L/debut | 15 | -0.00052 | -0.00113, +0.00006 | 2 | 0 | 0, 0 |
| 19 | G3/debut | 16 | -0.00088 | -0.00186, +0.000008 | 3 | 0 | 0, 0 |

The bounds are one-sided mirror bounds, not a two-sided interval ([lo, hi] is the 5% and 95% quantiles of the
bootstrap mean). A tail of a few bets (the debut cells of OP, G1, G2, G3, L, C3) has bounds that come from a handful
of meetings and carry no information: read them by their bet counts.

## Alternatives rejected and why

- **Refit #93's level 1 on each segment.** It would change the member, not measure it; the out-of-fold predictions are
  #93's own, so the segments are cut from one set of fits and no segment is fitted on its own outcome.
- **A debut label from the price frame's runs.** The price era starts in 2014, so a horse whose earlier runs predate it
  would read as a debut. The label is taken on the whole level-1 history from 2010-01-01 and the window is cut after.
- **A segment cut on the horse's debut rather than the race's.** The runner-level label is in the code
  (`debut_labels`), but the unit of a ΔLL is the race, and a maiden or newcomers' race mixes debut and seasoned
  runners; the race's segment is one string.
- **Tail quantiles 0.5 and 0.9 as well.** The frozen cut has the top 1% only, #93's top-band row; the wider bands are
  #93's own table.

## What this rules out

- No segment shows a gain: in all 19 the lower bound is below 0 (OP/seasoned's is -0.000003), and in the segments the
  owner's hypothesis names the point is slightly negative: MAIDEN/debut (2,755 races, -0.00003, upper bound +0.00008)
  and DEBUT/debut (2,074 races, -0.00004, upper bound +0.00027). A gain above +0.0001 nats per race is excluded in
  MAIDEN/debut and above +0.0003 in DEBUT/debut. The null stands in debut and maiden races as it does over all of
  #93's races. C1/seasoned is below 0 on both bounds (-0.00004; -0.00008, -0.000005), the only segment where the
  member is measurably worse than the base.
- The small debut cells (G1, G2, G3, L, C3: 8 to 20 races each) exclude nothing under about +0.0006 to +0.002 nats per
  race; they are ranked, not read.
- The top 1% of claims returns 75.5 per ¥100 in MAIDEN/debut (bounds 58.3, 94.3) and 80.8 in DEBUT/debut (65.8, 96.5):
  both upper bounds are below 100. G1/seasoned's 188 (94.8, 303.8; 26 bets in 18 meetings) is a long-shot tail with a
  lower bound below 100. No tail has a lower bound above 100.
- Nineteen segments were looked at, so a bound that excluded 0 in one of them would be expected by chance about once;
  none does in the gain's favour.

## Leaderboard rows

One option per segment, `japan-debut-prior-segments-final-win/segment:<segment>`, the tail return as its return per
¥100 (kind member, pool WIN, selection look_ahead because the final price is not known at decision time; the
convention block is in each record's `params.leaderboard`). The 19 are ranked by the leaderboard's own shrunk return,
not by ΔLL (the ΔLL rank is in the table above); `horsetrack japan leaderboard` adds them at its next build. The
spec's ΔLL rank is in each record's `params.rank`.

## Follow-ups filed

None. The debut and maiden races hold no gain for this prior at the final price; the members that carry new
information for them are #438's.
