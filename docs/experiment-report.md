# Experiment report

Every modeling pull request carries one. Copy the headings below into the PR description, or link a file that follows them, and fill each one in.

The report is public: it goes into a pull request on a public repository. In every section below, a result built on a licensed source (JRDB, JRA-VAN) is cited by its registry record or its private report (written through `core.eval.reports.write_report`) and its option id, never by a figure: no return, interval, log-loss gain, count or sign (D21).

## Hypothesis

State what you expect to be true and what result would show you wrong, before you look at the outcome.

## Variants tried

List every variant run, including the ones that failed, each with its confidence interval and the split it was evaluated on.

## Alternatives rejected and why

Name the approaches you considered but did not run or did not keep, and give the reason for each.

## What this rules out

State the measured result: the return per ¥100 at the final price and its confidence interval for each variant, and which values those intervals exclude. Report it as a measurement, not a verdict: nothing is adopted or rejected, and a weak result ranks low on the leaderboard rather than being dropped.

## Leaderboard rows

List the option ids this study adds to the country's leaderboard and where each ranks, failed variants included; a variant with no measured return is listed unranked (D16). A row built on a licensed source (JRDB, JRA-VAN) appears here by option id only, with no rank or figure (D21).

## Follow-ups filed

List the follow-up tickets this work leads to, by ticket number.
