# Pre-registration: #93's trainer debut prior by segment at the final Win price (#573)

Status: **frozen** in the Japan run registry before the run read any store row, without a sign-off
stop (D15-D19 as amended). The frozen JSON is `debut_prior.build_spec(bootstrap)` in
`src/horsetrack/countries/japan/studies/debut_prior.py`, registered as family
`japan-debut-prior-segments-final-win`; its spec sha256 is printed in the run's report and on
every registry record (`params.spec_sha256`). This page describes that JSON. If the two differ,
the JSON is the record. The registry refuses a run whose spec differs by one byte, so a change of
method needs a new family. The study's modeling choices are here, not in `Ruling:` lines.

## Question

#93's `trainer_debut` member (a debut indicator and a variance component per (trainer, debut)
cell, on top of the base member) is already a ranked row over every scored race. Where does it
earn its place: in debut and maiden races, as the owner's hypothesis for #438 says, or nowhere?

- **Null**, the starting point: the member's stage-2 meta is no better than the base's stage-2 meta
  at the final Win price in any segment (ΔLL 0), and its top-claim bets return no more than the
  segment's other bets.

Nothing is adopted or rejected (D16-D19): every segment is one ranked row.

## The cut

The unit is the race. Its segment is `<race_class>/<debut|seasoned>`:

- `race_class` is the store's code (DEBUT, MAIDEN, and the classes above them); a missing class
  reads `unknown`. The axes are `debut`, `maiden` (class DEBUT or MAIDEN) and `race_class`; the
  cut is their product, which is the string above (`maiden` is a function of the class).
- A runner **debuts** when the horse has no run at a strictly earlier post time in the level-1
  history (the JRA store's run races from 2010-01-01, every not-scratched starter). The race's own
  run, another run at the same post time (`cell_sizes`' tie rule) and a later run never count, and
  no outcome is read. The label is computed on the whole history, then the scored window is cut,
  so a run before the window still counts.
- A race is `debut` when any starter debuts, else `seasoned`.

No segment is dropped; a segment of one race keeps its row.

## Score, per segment

Over the scored races (D9's level-2 folds from 2020-Q1, development days only), from #93's
out-of-fold level-1 export (the latest scored record of `japan-c2-interactions-final-win-v2`
names it) with #87's stage-2 meta at the final Win price rebuilt from it exactly as
`studies/interactions.py` does (D4's calibrated market out of fold; no level-1 model is fitted):

- **ΔLL** = the mean over the segment's races of ln p_member(winner) − ln p_base(winner), where
  p_member is the `trainer_debut` member's stage-2 probability and p_base the base's, each
  normalised within the race; a dead heat's k winners weigh 1/k each (D6). Bounds: D7's one-sided
  95% paired bootstrap over meetings (`core.eval.bootstrap.paired_bootstrap`, `scoring.yaml`'s
  `n_boot` and level, seed 573).
- **Tail return** = the return per ¥100 of the segment's bets whose claim p_member × O_final is at
  or above the 0.99 quantile of the segment's claims (`np.quantile`, linear), flat unit stakes,
  settled at the recorded Win dividend of the horse number (`interactions.win_paid`: a dead heat
  pays each winner its dividend; a race with no WIN row is unsettled and fails the run). Bounds:
  the same bootstrap over meetings.
- **Rank**: by ΔLL, highest first.

## Records, leaderboard, publication

- One registry record per segment (`segment:<segment>`), a failed run as a `run:aborted` record;
  each carries ΔLL with its bounds (`delta_ll`) and the tail return with its bounds
  (`return_per_100`, `n_meetings`) under `params.leaderboard`, so the segment's tail is a row on
  the leaderboard (kind member, pool WIN, selection `look_ahead`: the final price).
- The JRA results store is the only source: no JRDB, no JRA-VAN, so the figures are public. The
  per-segment table is still written through `write_report` (sources `jra`).
- The holdout is not opened.
