# Experiment report: C5 against the Place and Wide dividends, per-ticket de-vig (Japan, #459)

Ticket #459's second family (epic #73; 08 §4 E43, E16; the review of PR #668 under the owner's reframe on #99 of 2026-11-27: each candidate Place and Wide de-vig is a registered variant). The first family, `japan-c5-exotic-dividends` (`docs/experiments/exotic-c5-japan.md`), scored Place and Wide against one baseline, the chance a paid ticket pays taken as P × its statute money share. This family scores the same C5 grid on Place and Wide against a second baseline computable from the paid dividends alone: each paid ticket's own implied share, min(1, R·100/(D_c + 5)) (R the race's payout rate, boosts in; D_c its dividend per 100 yen), the money on it against what it pays, as in a pool with one winner (P × #99 question 3(b)'s equal split). Same races, folds, inputs, orderings, blends and floor treatments, by the log loss at 1/k per paid combination; only the baseline differs. So a no-blend variant's ΔLL minus its statute twin's (the `statute` columns, from the first family's registry records) is the statute baseline's log loss minus this one's, and the leaderboard bet, which never reads the baseline, returns what its twin returns. **Nothing is adopted** (D16-D19): every variant is a ranked row. The holdout is not opened and no holdout day is read. **stage2:q:form is JRDB-fed (D21)**: its variants ran and are withheld; this report gives no figure, count or sign built on it. The figures below are the publishable pair's, normalised 1/odds and D4's calibrated market, from the store's results and dividends alone.

- Registry family `japan-c5-exotic-dividends-per-ticket`, spec sha256 `b559e99c14f73a307e9f075dcb405a67b713feec2640d2d31ea0be4e908c65d1`, frozen before the run read any store row; the pre-registration is `docs/specs/exotic-c5-prereg.md`.
- Code: `countries/japan/studies/exotic_c5.py` at `3ae0b543806a`. Reproduce: `uv run horsetrack japan exotic-c5 --family japan-c5-exotic-dividends-per-ticket --members <HKTrack>/outputs/oof/m-v1-20261115-6102e309.parquet`. #418's level-2 guard (`studies.meta.check_level2`) came after this family's first run, whose stage2:q:form records stand: under it, this command refuses this member file before any fit, because D9's five seasons are met at level 2 on none of the scored folds (#87's documented shortfall), and records every stage2:q:form variant failed.
- Run window: 2014-06-07 to 2026-11-27, development days only.
- Records: `docs/experiments/registry/japan/japan-c5-exotic-dividends-per-ticket.md` (the stage2:q:form records withheld). The private report, with every table, is under `$HORSETRACK_DATA_DIR/reports/` (`exotic-c5-japan-<spec hash>-<run digest>.md`).

## Hypothesis

- **E43.** C5 (an input's Win probabilities through an ordering, optionally blended with the pool) beats the per-ticket dividend baseline in log loss on the paid combinations of Place and Wide, per pool: mean ΔLL (dividends minus model) above 0
- **E16.** the E43 table on the race-pools that paid the super_premium rate (80%)
- **Null**, the starting point: the per-ticket dividends are at least as good as C5 in Place and in Wide (ΔLL <= 0): C5 adds nothing to the pools' own prices.

## Data

| count | n |
| --- | --- |
| races_read | 43130 |
| runners_read | 603421 |
| races_not_run | 0 |
| runners_without_a_run_race | 0 |
| scratched_runners | 2156 |
| unpriced_starters | 0 |
| starters_without_a_horse_number | 0 |
| races_with_a_horse_number_twice | 0 |
| races_under_3_priced_starters | 0 |
| races | 43130 |
| starters | 601265 |
| race_pools_paid | 301910 |
| race_pools_left_out_refund | 0 |
| race_pools_left_out_special_payout | 0 |
| race_pools_left_out_unpriced | 0 |
| race_pools_left_out_not_in_field | 0 |
| race_pools_scoreable | 301910 |
| folds | 28 |
| scored_races | 23871 |

## Variants tried

One registry record each, failures included. ΔLL is the dividends' log loss minus the model's, in nats per race-pool (positive: C5 better); lower and upper are D7's one-sided 95% meeting-bootstrap bounds. The return is a flat ¥100 on the model's most probable combination per race-pool at the paid dividends (look-ahead: the inputs are the final Win prices), with the same bounds. The statute columns are the statute twin's ΔLL and bounds, the first family's record of the same variant (empty where it holds none).

### E43: per pool, against the dividends (every scored race-pool)

| variant | races | meetings | model LL | dividend LL | ΔLL | lower | upper | statute ΔLL | statute lower | statute upper | alpha (median of folds) | return per ¥100 | return lower | return upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| place:norm:harv:none:tick:drop | 18087 | 1991 | 1.23355 | 1.24188 | 0.00833606 | 0.00636555 | 0.0102976 | 0.0274557 | 0.0255373 | 0.0293532 |  | 76.5683 | 75.7084 | 77.4389 |
| place:norm:harv:lin:tick:drop | 18087 | 1991 | 1.2264 | 1.24188 | 0.0154841 | 0.0142757 | 0.0166873 | 0.0274557 | 0.0255373 | 0.0293532 | 0.426344 |  |  |  |
| place:norm:harv:none:tick:bound | 23871 | 1991 | 1.12822 | 1.12238 | -0.00583645 | -0.00747859 | -0.00423804 | 0.0257462 | 0.0241993 | 0.0272435 |  | 84.441 | 83.7674 | 85.115 |
| place:norm:harv:lin:tick:bound | 23871 | 1991 | 1.11376 | 1.12238 | 0.00862199 | 0.0079045 | 0.00932981 | 0.0257462 | 0.0241993 | 0.0272435 | 0.593254 |  |  |  |
| wide:norm:harv:none:tick:drop | 23709 | 1991 | 2.51312 | 2.48056 | -0.0325572 | -0.0353351 | -0.0298073 | -0.0278307 | -0.0306269 | -0.0250654 |  | 80.0173 | 78.6569 | 81.3775 |
| wide:norm:harv:lin:tick:drop | 23709 | 1991 | 2.47599 | 2.48056 | 0.00456404 | 0.00393613 | 0.00519396 | 0.00440321 | 0.00373589 | 0.00507569 | 0.747639 |  |  |  |
| wide:norm:harv:none:tick:bound | 23871 | 1991 | 2.50127 | 2.46931 | -0.0319559 | -0.0347293 | -0.0292168 | -0.0268398 | -0.0296311 | -0.0240798 |  | 80.2212 | 78.8684 | 81.5762 |
| wide:norm:harv:lin:tick:bound | 23871 | 1991 | 2.46464 | 2.46931 | 0.00467484 | 0.00405121 | 0.00529717 | 0.00466576 | 0.00399188 | 0.00533852 | 0.748243 |  |  |  |
| place:cal:harv:none:tick:drop | 18087 | 1991 | 1.23668 | 1.24188 | 0.00520459 | 0.0030691 | 0.00732357 | 0.0243242 | 0.022258 | 0.0263745 |  | 76.743 | 75.8809 | 77.6049 |
| place:cal:harv:lin:tick:drop | 18087 | 1991 | 1.22644 | 1.24188 | 0.0154455 | 0.0142233 | 0.0166397 | 0.0243164 | 0.0222798 | 0.0263308 | 0.443546 |  |  |  |
| place:cal:harv:none:tick:bound | 23871 | 1991 | 1.13068 | 1.12238 | -0.00829742 | -0.0100857 | -0.00654451 | 0.0232852 | 0.0216205 | 0.0249201 |  | 84.5708 | 83.9024 | 85.2419 |
| place:cal:harv:lin:tick:bound | 23871 | 1991 | 1.11327 | 1.12238 | 0.00911302 | 0.00836365 | 0.00985325 | 0.0232875 | 0.0216455 | 0.0248997 | 0.587817 |  |  |  |
| wide:cal:harv:none:tick:drop | 23709 | 1991 | 2.51815 | 2.48056 | -0.0375952 | -0.0407638 | -0.0344525 | -0.0328687 | -0.0360524 | -0.029714 |  | 79.9966 | 78.6334 | 81.3529 |
| wide:cal:harv:lin:tick:drop | 23709 | 1991 | 2.47518 | 2.48056 | 0.00537866 | 0.00465238 | 0.00610095 | 0.00512081 | 0.00435207 | 0.00588741 | 0.727535 |  |  |  |
| wide:cal:harv:none:tick:bound | 23871 | 1991 | 2.50618 | 2.46931 | -0.036864 | -0.0400099 | -0.0337098 | -0.0317479 | -0.0349112 | -0.0285951 |  | 80.2007 | 78.8437 | 81.5524 |
| wide:cal:harv:lin:tick:bound | 23871 | 1991 | 2.46379 | 2.46931 | 0.00552472 | 0.00480256 | 0.00624428 | 0.00542833 | 0.0046529 | 0.00619901 | 0.72789 |  |  |  |
| place:norm:hen:none:tick:drop | 18087 | 1991 | 1.21626 | 1.24188 | 0.0256269 | 0.0235213 | 0.0277146 | 0.0447465 | 0.042674 | 0.0467906 |  | 76.7452 | 75.8842 | 77.6147 |
| place:norm:hen:lin:tick:drop | 18087 | 1991 | 1.21626 | 1.24188 | 0.0256192 | 0.0235141 | 0.0277072 | 0.0447465 | 0.042674 | 0.0467906 | 0 |  |  |  |
| place:norm:hen:none:tick:bound | 23871 | 1991 | 1.12139 | 1.12238 | 0.000995808 | -0.000685977 | 0.00264902 | 0.0325784 | 0.0309574 | 0.0341734 |  | 84.5662 | 83.8955 | 85.2393 |
| place:norm:hen:lin:tick:bound | 23871 | 1991 | 1.11615 | 1.12238 | 0.00623161 | 0.00537355 | 0.00707938 | 0.0325784 | 0.0309574 | 0.0341734 | 0.55618 |  |  |  |
| wide:norm:hen:none:tick:drop | 23709 | 1991 | 2.49432 | 2.48056 | -0.0137594 | -0.0154405 | -0.0120807 | -0.00903296 | -0.0107048 | -0.00734943 |  | 80.0548 | 78.6778 | 81.4054 |
| wide:norm:hen:lin:tick:drop | 23709 | 1991 | 2.47907 | 2.48056 | 0.00148543 | 0.00111361 | 0.00185488 | 0.00349038 | 0.00293517 | 0.00404598 | 0.781404 |  |  |  |
| wide:norm:hen:none:tick:bound | 23871 | 1991 | 2.4836 | 2.46931 | -0.0142887 | -0.0159574 | -0.012613 | -0.00917265 | -0.0108439 | -0.00749792 |  | 80.2585 | 78.89 | 81.6022 |
| wide:norm:hen:lin:tick:bound | 23871 | 1991 | 2.46796 | 2.46931 | 0.00135492 | 0.00101098 | 0.00169871 | 0.00345803 | 0.00291847 | 0.00400027 | 0.796132 |  |  |  |
| place:cal:hen:none:tick:drop | 18087 | 1991 | 1.21433 | 1.24188 | 0.0275506 | 0.0254206 | 0.0296349 | 0.0466702 | 0.0445885 | 0.0487133 |  | 76.7596 | 75.8979 | 77.623 |
| place:cal:hen:lin:tick:drop | 18087 | 1991 | 1.21433 | 1.24188 | 0.0275506 | 0.0254206 | 0.0296349 | 0.0466702 | 0.0445885 | 0.0487133 | 0 |  |  |  |
| place:cal:hen:none:tick:bound | 23871 | 1991 | 1.11852 | 1.12238 | 0.0038631 | 0.00215391 | 0.00553978 | 0.0354457 | 0.0337973 | 0.0370526 |  | 84.5872 | 83.9132 | 85.2581 |
| place:cal:hen:lin:tick:bound | 23871 | 1991 | 1.11434 | 1.12238 | 0.00804372 | 0.0070342 | 0.00903236 | 0.0354457 | 0.0337973 | 0.0370526 | 0.460816 |  |  |  |
| wide:cal:hen:none:tick:drop | 23709 | 1991 | 2.48852 | 2.48056 | -0.00796261 | -0.00975087 | -0.00621313 | -0.00323614 | -0.00502466 | -0.00148161 |  | 79.9679 | 78.6085 | 81.3129 |
| wide:cal:hen:lin:tick:drop | 23709 | 1991 | 2.47703 | 2.48056 | 0.00352475 | 0.00287997 | 0.00415659 | 0.00604852 | 0.0052439 | 0.00683641 | 0.625912 |  |  |  |
| wide:cal:hen:none:tick:bound | 23871 | 1991 | 2.4777 | 2.46931 | -0.00838773 | -0.0101799 | -0.00665772 | -0.00327164 | -0.00505376 | -0.00153573 |  | 80.1722 | 78.823 | 81.5083 |
| wide:cal:hen:lin:tick:bound | 23871 | 1991 | 2.46595 | 2.46931 | 0.00336309 | 0.00274531 | 0.00396712 | 0.00605904 | 0.00527281 | 0.00683119 | 0.639789 |  |  |  |
| place:norm:qc:none:tick:drop | 18087 | 1991 | 1.21534 | 1.24188 | 0.0265414 | 0.0244317 | 0.0286356 | 0.045661 | 0.0435961 | 0.0477059 |  | 76.822 | 75.9608 | 77.6901 |
| place:norm:qc:lin:tick:drop | 18087 | 1991 | 1.21534 | 1.24188 | 0.0265414 | 0.0244317 | 0.0286356 | 0.045661 | 0.0435961 | 0.0477059 | 0 |  |  |  |
| place:norm:qc:none:tick:bound | 23871 | 1991 | 1.11817 | 1.12238 | 0.00421178 | 0.00250162 | 0.00590283 | 0.0357944 | 0.0341568 | 0.037385 |  | 84.6399 | 83.9705 | 85.3163 |
| place:norm:qc:lin:tick:bound | 23871 | 1991 | 1.11367 | 1.12238 | 0.00871139 | 0.00769259 | 0.00972031 | 0.0357944 | 0.0341568 | 0.037385 | 0.450469 |  |  |  |
| wide:norm:qc:none:tick:drop | 23709 | 1991 | 2.48837 | 2.48056 | -0.00781118 | -0.00968182 | -0.00597547 | -0.00308471 | -0.00495819 | -0.00126007 |  | 80.1126 | 78.7538 | 81.4602 |
| wide:norm:qc:lin:tick:drop | 23709 | 1991 | 2.47628 | 2.48056 | 0.00427813 | 0.0035705 | 0.00496885 | 0.00651264 | 0.0056464 | 0.00735112 | 0.607956 |  |  |  |
| wide:norm:qc:none:tick:bound | 23871 | 1991 | 2.47702 | 2.46931 | -0.0077044 | -0.00957419 | -0.00588467 | -0.00258831 | -0.00446504 | -0.000770461 |  | 80.3159 | 78.9594 | 81.6596 |
| wide:norm:qc:lin:tick:bound | 23871 | 1991 | 2.465 | 2.46931 | 0.00431567 | 0.00362079 | 0.0049903 | 0.00677737 | 0.00591485 | 0.00760832 | 0.613045 |  |  |  |
| place:cal:qc:none:tick:drop | 18087 | 1991 | 1.21533 | 1.24188 | 0.0265569 | 0.0244223 | 0.0286704 | 0.0456765 | 0.0435744 | 0.0477204 |  | 76.6158 | 75.7534 | 77.4795 |
| place:cal:qc:lin:tick:drop | 18087 | 1991 | 1.21533 | 1.24188 | 0.0265569 | 0.0244223 | 0.0286704 | 0.0456765 | 0.0435744 | 0.0477204 | 0 |  |  |  |
| place:cal:qc:none:tick:bound | 23871 | 1991 | 1.11789 | 1.12238 | 0.00448668 | 0.00275905 | 0.00620035 | 0.0360693 | 0.0343964 | 0.0376856 |  | 84.4678 | 83.7959 | 85.1439 |
| place:cal:qc:lin:tick:bound | 23871 | 1991 | 1.11326 | 1.12238 | 0.00912428 | 0.00807107 | 0.0101735 | 0.0360693 | 0.0343964 | 0.0376856 | 0.435661 |  |  |  |
| wide:cal:qc:none:tick:drop | 23709 | 1991 | 2.48776 | 2.48056 | -0.00719762 | -0.00914165 | -0.00530111 | -0.00247115 | -0.00441212 | -0.000576718 |  | 79.989 | 78.6229 | 81.3316 |
| wide:cal:qc:lin:tick:drop | 23709 | 1991 | 2.47576 | 2.48056 | 0.00480094 | 0.00402283 | 0.00556267 | 0.0070629 | 0.00613373 | 0.00797134 | 0.575993 |  |  |  |
| wide:cal:qc:none:tick:bound | 23871 | 1991 | 2.47643 | 2.46931 | -0.00712017 | -0.00905754 | -0.00523164 | -0.00200408 | -0.00394673 | -0.000119645 |  | 80.1931 | 78.8361 | 81.5282 |
| wide:cal:qc:lin:tick:bound | 23871 | 1991 | 2.46448 | 2.46931 | 0.00482876 | 0.00406753 | 0.00556955 | 0.00732168 | 0.00639479 | 0.00822325 | 0.58351 |  |  |  |
| place:norm:dh:none:tick:drop | 18087 | 1991 | 1.21897 | 1.24188 | 0.0229092 | 0.0208906 | 0.0249057 | 0.0420288 | 0.0400681 | 0.0439876 |  | 76.7408 | 75.8694 | 77.6141 |
| place:norm:dh:lin:tick:drop | 18087 | 1991 | 1.21903 | 1.24188 | 0.022855 | 0.0209591 | 0.0247264 | 0.0420288 | 0.0400681 | 0.0439876 | 0.0748606 |  |  |  |
| place:norm:dh:none:tick:bound | 23871 | 1991 | 1.12091 | 1.12238 | 0.00146666 | -0.000143817 | 0.00306329 | 0.0330493 | 0.0315024 | 0.0345692 |  | 84.5738 | 83.8995 | 85.2497 |
| place:norm:dh:lin:tick:bound | 23871 | 1991 | 1.11504 | 1.12238 | 0.00733967 | 0.00649593 | 0.0081771 | 0.0330493 | 0.0315024 | 0.0345692 | 0.536439 |  |  |  |
| wide:norm:dh:none:tick:drop | 23709 | 1991 | 2.49411 | 2.48056 | -0.0135555 | -0.0153648 | -0.0117875 | -0.00882898 | -0.0106398 | -0.00704983 |  | 79.9798 | 78.6208 | 81.3278 |
| wide:norm:dh:lin:tick:drop | 23709 | 1991 | 2.47798 | 2.48056 | 0.00257978 | 0.00212075 | 0.00303253 | 0.00414785 | 0.00354547 | 0.00474173 | 0.738837 |  |  |  |
| wide:norm:dh:none:tick:bound | 23871 | 1991 | 2.48298 | 2.46931 | -0.013672 | -0.0154772 | -0.0119015 | -0.00855595 | -0.0103533 | -0.00677492 |  | 80.1839 | 78.8261 | 81.5269 |
| wide:norm:dh:lin:tick:bound | 23871 | 1991 | 2.46679 | 2.46931 | 0.00252343 | 0.00208311 | 0.00296083 | 0.00424643 | 0.00364955 | 0.00483392 | 0.747225 |  |  |  |
| place:cal:dh:none:tick:drop | 18087 | 1991 | 1.21634 | 1.24188 | 0.0255425 | 0.0234747 | 0.0275647 | 0.0446621 | 0.0426326 | 0.0466432 |  | 76.6468 | 75.7921 | 77.5146 |
| place:cal:dh:lin:tick:drop | 18087 | 1991 | 1.21635 | 1.24188 | 0.0255319 | 0.0234673 | 0.027552 | 0.0446621 | 0.0426326 | 0.0466432 | 0 |  |  |  |
| place:cal:dh:none:tick:bound | 23871 | 1991 | 1.11851 | 1.12238 | 0.00387294 | 0.00221285 | 0.00550096 | 0.0354556 | 0.0338635 | 0.0370012 |  | 84.4879 | 83.8243 | 85.1599 |
| place:cal:dh:lin:tick:bound | 23871 | 1991 | 1.11384 | 1.12238 | 0.00854303 | 0.00757547 | 0.00949555 | 0.0354556 | 0.0338635 | 0.0370012 | 0.466579 |  |  |  |
| wide:cal:dh:none:tick:drop | 23709 | 1991 | 2.48902 | 2.48056 | -0.00846408 | -0.0102729 | -0.00668515 | -0.00373761 | -0.00553937 | -0.00194721 |  | 80.1561 | 78.7965 | 81.5149 |
| wide:cal:dh:lin:tick:drop | 23709 | 1991 | 2.47649 | 2.48056 | 0.00406597 | 0.00342709 | 0.00469829 | 0.00614707 | 0.00536258 | 0.00692259 | 0.632267 |  |  |  |
| wide:cal:dh:none:tick:bound | 23871 | 1991 | 2.47789 | 2.46931 | -0.00857855 | -0.010388 | -0.00680593 | -0.00346246 | -0.00527173 | -0.00167569 |  | 80.359 | 79.0055 | 81.7083 |
| wide:cal:dh:lin:tick:bound | 23871 | 1991 | 2.46531 | 2.46931 | 0.00400624 | 0.00338388 | 0.00461797 | 0.00628444 | 0.00550397 | 0.00705143 | 0.64325 |  |  |  |

### E16: the E43 table on super_premium race-pools

| variant | races | meetings | model LL | dividend LL | ΔLL | lower | upper | statute ΔLL | statute lower | statute upper | return per ¥100 | return lower | return upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp:place:norm:harv:none:tick:drop | 342 | 37 | 1.20808 | 1.21063 | 0.00254755 | -0.00832118 | 0.014142 | 0.0222796 | 0.0113739 | 0.0340932 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:norm:harv:lin:tick:drop | 342 | 37 | 1.19906 | 1.21063 | 0.0115684 | 0.00511233 | 0.018474 | 0.0222796 | 0.0113739 | 0.0340932 |  |  |  |
| sp:place:norm:harv:none:tick:bound | 443 | 37 | 1.11393 | 1.10401 | -0.00991624 | -0.0190589 | -0.000462656 | 0.0206473 | 0.0119496 | 0.0298421 | 91.1287 | 87.6524 | 94.5721 |
| sp:place:norm:harv:lin:tick:bound | 443 | 37 | 1.09749 | 1.10401 | 0.00651863 | 0.00246471 | 0.0107661 | 0.0206473 | 0.0119496 | 0.0298421 |  |  |  |
| sp:wide:norm:harv:none:tick:drop | 392 | 33 | 2.4348 | 2.42756 | -0.00724086 | -0.0228486 | 0.00986379 | -0.00282298 | -0.0185913 | 0.0143659 | 90.8163 | 78.402 | 103.147 |
| sp:wide:norm:harv:lin:tick:drop | 392 | 33 | 2.41914 | 2.42756 | 0.00841451 | 0.0047919 | 0.0123121 | 0.00842865 | 0.00454442 | 0.0125963 |  |  |  |
| sp:wide:norm:harv:none:tick:bound | 395 | 33 | 2.42121 | 2.4145 | -0.00671727 | -0.0220473 | 0.0101017 | -0.00179375 | -0.0170607 | 0.0148949 | 90.962 | 78.6571 | 103.207 |
| sp:wide:norm:harv:lin:tick:bound | 395 | 33 | 2.40596 | 2.4145 | 0.0085375 | 0.00500063 | 0.0123896 | 0.0087513 | 0.00495114 | 0.0128684 |  |  |  |
| sp:place:cal:harv:none:tick:drop | 342 | 37 | 1.20997 | 1.21063 | 0.000656618 | -0.010409 | 0.0125252 | 0.0203887 | 0.00926655 | 0.0322746 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:cal:harv:lin:tick:drop | 342 | 37 | 1.19872 | 1.21063 | 0.0119041 | 0.0057211 | 0.0185648 | 0.0201373 | 0.00921244 | 0.0317605 |  |  |  |
| sp:place:cal:harv:none:tick:bound | 443 | 37 | 1.11574 | 1.10401 | -0.0117301 | -0.0214434 | -0.00181759 | 0.0188334 | 0.00981203 | 0.0283204 | 91.1287 | 87.6524 | 94.5721 |
| sp:place:cal:harv:lin:tick:bound | 443 | 37 | 1.09706 | 1.10401 | 0.00695017 | 0.00284436 | 0.0112221 | 0.0187332 | 0.00984051 | 0.0280283 |  |  |  |
| sp:wide:cal:harv:none:tick:drop | 392 | 33 | 2.43717 | 2.42756 | -0.00960965 | -0.0263124 | 0.00854578 | -0.00519177 | -0.0220973 | 0.0128973 | 91.8112 | 79.4176 | 103.919 |
| sp:wide:cal:harv:lin:tick:drop | 392 | 33 | 2.41795 | 2.42756 | 0.00961255 | 0.00553884 | 0.0141171 | 0.00957515 | 0.00523439 | 0.0143285 |  |  |  |
| sp:wide:cal:harv:none:tick:bound | 395 | 33 | 2.42352 | 2.4145 | -0.00902552 | -0.0254594 | 0.00876402 | -0.00410201 | -0.0205562 | 0.0137296 | 91.9494 | 79.6212 | 103.975 |
| sp:wide:cal:harv:lin:tick:bound | 395 | 33 | 2.40474 | 2.4145 | 0.00975428 | 0.00576255 | 0.0141954 | 0.00991855 | 0.00560974 | 0.014657 |  |  |  |
| sp:place:norm:hen:none:tick:drop | 342 | 37 | 1.19733 | 1.21063 | 0.0132993 | 0.000691451 | 0.0268094 | 0.0330314 | 0.020134 | 0.0467622 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:norm:hen:lin:tick:drop | 342 | 37 | 1.19733 | 1.21063 | 0.0132993 | 0.000691451 | 0.0268094 | 0.0330314 | 0.020134 | 0.0467622 |  |  |  |
| sp:place:norm:hen:none:tick:bound | 443 | 37 | 1.11172 | 1.10401 | -0.00770788 | -0.0176626 | 0.00252183 | 0.0228556 | 0.0129005 | 0.0330774 | 91.1287 | 87.6524 | 94.5721 |
| sp:place:norm:hen:lin:tick:bound | 443 | 37 | 1.1021 | 1.10401 | 0.00190654 | -0.00301354 | 0.00705214 | 0.0228556 | 0.0129005 | 0.0330774 |  |  |  |
| sp:wide:norm:hen:none:tick:drop | 392 | 33 | 2.43427 | 2.42756 | -0.00671179 | -0.019991 | 0.00702142 | -0.00229391 | -0.0154779 | 0.011369 | 91.8112 | 79.4176 | 103.919 |
| sp:wide:norm:hen:lin:tick:drop | 392 | 33 | 2.42571 | 2.42756 | 0.00185196 | -0.00115107 | 0.00494779 | 0.00429713 | -0.000146872 | 0.00885684 |  |  |  |
| sp:wide:norm:hen:none:tick:bound | 395 | 33 | 2.42132 | 2.4145 | -0.00682101 | -0.0199993 | 0.00694454 | -0.00189749 | -0.0149111 | 0.0115475 | 91.9494 | 79.6212 | 103.975 |
| sp:wide:norm:hen:lin:tick:bound | 395 | 33 | 2.41274 | 2.4145 | 0.00175272 | -0.00104201 | 0.00464339 | 0.00438276 | 0.000110956 | 0.00876535 |  |  |  |
| sp:place:cal:hen:none:tick:drop | 342 | 37 | 1.19479 | 1.21063 | 0.0158377 | 0.00331249 | 0.0291811 | 0.0355698 | 0.0229286 | 0.0490953 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:cal:hen:lin:tick:drop | 342 | 37 | 1.19479 | 1.21063 | 0.0158377 | 0.00331249 | 0.0291811 | 0.0355698 | 0.0229286 | 0.0490953 |  |  |  |
| sp:place:cal:hen:none:tick:bound | 443 | 37 | 1.10884 | 1.10401 | -0.00483123 | -0.0148849 | 0.00531056 | 0.0257323 | 0.0157912 | 0.0359885 | 91.1287 | 87.6524 | 94.5721 |
| sp:place:cal:hen:lin:tick:bound | 443 | 37 | 1.1011 | 1.10401 | 0.00291198 | -0.0029731 | 0.00893351 | 0.0257323 | 0.0157912 | 0.0359885 |  |  |  |
| sp:wide:cal:hen:none:tick:drop | 392 | 33 | 2.42779 | 2.42756 | -0.000235983 | -0.0136841 | 0.013945 | 0.0041819 | -0.00922024 | 0.0182548 | 91.8112 | 79.4176 | 103.919 |
| sp:wide:cal:hen:lin:tick:drop | 392 | 33 | 2.42296 | 2.42756 | 0.00459796 | -0.000398571 | 0.00985813 | 0.00767217 | 0.00151227 | 0.0141324 |  |  |  |
| sp:wide:cal:hen:none:tick:bound | 395 | 33 | 2.41485 | 2.4145 | -0.000349714 | -0.0137234 | 0.0138404 | 0.0045738 | -0.00853756 | 0.018471 | 91.9494 | 79.6212 | 103.975 |
| sp:wide:cal:hen:lin:tick:bound | 395 | 33 | 2.41003 | 2.4145 | 0.00446364 | -0.000320712 | 0.00949899 | 0.00779897 | 0.00183803 | 0.0140807 |  |  |  |
| sp:place:norm:qc:none:tick:drop | 342 | 37 | 1.19564 | 1.21063 | 0.0149842 | 0.00288366 | 0.0280183 | 0.0347163 | 0.0224695 | 0.0479091 | 85.5848 | 80.9091 | 90.2671 |
| sp:place:norm:qc:lin:tick:drop | 342 | 37 | 1.19564 | 1.21063 | 0.0149842 | 0.00288366 | 0.0280183 | 0.0347163 | 0.0224695 | 0.0479091 |  |  |  |
| sp:place:norm:qc:none:tick:bound | 443 | 37 | 1.1081 | 1.10401 | -0.00409471 | -0.0138176 | 0.00580562 | 0.0264688 | 0.0170292 | 0.0362659 | 90.8804 | 87.4374 | 94.3018 |
| sp:place:norm:qc:lin:tick:bound | 443 | 37 | 1.10043 | 1.10401 | 0.00358319 | -0.00223789 | 0.00953627 | 0.0264688 | 0.0170292 | 0.0362659 |  |  |  |
| sp:wide:norm:qc:none:tick:drop | 392 | 33 | 2.42648 | 2.42756 | 0.0010757 | -0.011317 | 0.0143545 | 0.00549357 | -0.00687912 | 0.0187836 | 90.8163 | 78.402 | 103.147 |
| sp:wide:norm:qc:lin:tick:drop | 392 | 33 | 2.42171 | 2.42756 | 0.00584958 | 0.000975855 | 0.0110583 | 0.00870889 | 0.00283628 | 0.014995 |  |  |  |
| sp:wide:norm:qc:none:tick:bound | 395 | 33 | 2.41308 | 2.4145 | 0.00142189 | -0.0108423 | 0.0146506 | 0.00634541 | -0.00568486 | 0.0193765 | 90.962 | 78.6571 | 103.207 |
| sp:wide:norm:qc:lin:tick:bound | 395 | 33 | 2.40853 | 2.4145 | 0.00596554 | 0.00127407 | 0.0110655 | 0.00912855 | 0.00342116 | 0.0152815 |  |  |  |
| sp:place:cal:qc:none:tick:drop | 342 | 37 | 1.19502 | 1.21063 | 0.0156051 | 0.0034367 | 0.0288091 | 0.0353371 | 0.0229382 | 0.0487453 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:cal:qc:lin:tick:drop | 342 | 37 | 1.19502 | 1.21063 | 0.0156051 | 0.0034367 | 0.0288091 | 0.0353371 | 0.0229382 | 0.0487453 |  |  |  |
| sp:place:cal:qc:none:tick:bound | 443 | 37 | 1.10749 | 1.10401 | -0.00347975 | -0.0134718 | 0.00661801 | 0.0270837 | 0.017426 | 0.0371784 | 91.1287 | 87.6524 | 94.5721 |
| sp:place:cal:qc:lin:tick:bound | 443 | 37 | 1.10005 | 1.10401 | 0.00396384 | -0.0021029 | 0.0101689 | 0.0270837 | 0.017426 | 0.0371784 |  |  |  |
| sp:wide:cal:qc:none:tick:drop | 392 | 33 | 2.42482 | 2.42756 | 0.00273662 | -0.010182 | 0.0166125 | 0.0071545 | -0.00576311 | 0.0210205 | 90.8163 | 78.402 | 103.147 |
| sp:wide:cal:qc:lin:tick:drop | 392 | 33 | 2.42073 | 2.42756 | 0.00683213 | 0.00135767 | 0.0126852 | 0.00977539 | 0.00336752 | 0.0166651 |  |  |  |
| sp:wide:cal:qc:none:tick:bound | 395 | 33 | 2.41152 | 2.4145 | 0.00297918 | -0.0097962 | 0.0167985 | 0.00790269 | -0.00473676 | 0.021554 | 90.962 | 78.6571 | 103.207 |
| sp:wide:cal:qc:lin:tick:bound | 395 | 33 | 2.4076 | 2.4145 | 0.00689767 | 0.00155477 | 0.0126084 | 0.0101435 | 0.00388452 | 0.0168825 |  |  |  |
| sp:place:norm:dh:none:tick:drop | 342 | 37 | 1.1984 | 1.21063 | 0.0122284 | 0.000373286 | 0.0249583 | 0.0319605 | 0.0198147 | 0.0449296 | 86.0819 | 81.4899 | 90.6287 |
| sp:place:norm:dh:lin:tick:drop | 342 | 37 | 1.19779 | 1.21063 | 0.0128382 | 0.00176347 | 0.0247203 | 0.0319605 | 0.0198147 | 0.0449296 |  |  |  |
| sp:place:norm:dh:none:tick:bound | 443 | 37 | 1.11001 | 1.10401 | -0.00599965 | -0.0154709 | 0.00373488 | 0.0245639 | 0.0151224 | 0.0342729 | 91.2415 | 87.8959 | 94.5929 |
| sp:place:norm:dh:lin:tick:bound | 443 | 37 | 1.10058 | 1.10401 | 0.00342596 | -0.00144457 | 0.00850142 | 0.0245639 | 0.0151224 | 0.0342729 |  |  |  |
| sp:wide:norm:dh:none:tick:drop | 392 | 33 | 2.42896 | 2.42756 | -0.00139739 | -0.0143081 | 0.0121688 | 0.00302048 | -0.00992582 | 0.016642 | 91.8112 | 79.4176 | 103.919 |
| sp:wide:norm:dh:lin:tick:drop | 392 | 33 | 2.42333 | 2.42756 | 0.00422552 | 0.000815489 | 0.00779284 | 0.00645138 | 0.00204254 | 0.0110806 |  |  |  |
| sp:wide:norm:dh:none:tick:bound | 395 | 33 | 2.41574 | 2.4145 | -0.00124481 | -0.0140631 | 0.0122481 | 0.00367871 | -0.00898786 | 0.0169973 | 91.9494 | 79.6212 | 103.975 |
| sp:wide:norm:dh:lin:tick:bound | 395 | 33 | 2.41031 | 2.4145 | 0.00418464 | 0.00090078 | 0.00761904 | 0.00664101 | 0.00234607 | 0.0111445 |  |  |  |
| sp:place:cal:dh:none:tick:drop | 342 | 37 | 1.19581 | 1.21063 | 0.0148226 | 0.00276407 | 0.0277366 | 0.0345547 | 0.0222692 | 0.0477746 | 85.9357 | 81.1975 | 90.6181 |
| sp:place:cal:dh:lin:tick:drop | 342 | 37 | 1.19583 | 1.21063 | 0.0147981 | 0.00275473 | 0.0276819 | 0.0345547 | 0.0222692 | 0.0477746 |  |  |  |
| sp:place:cal:dh:none:tick:bound | 443 | 37 | 1.10796 | 1.10401 | -0.00394979 | -0.0136849 | 0.00597744 | 0.0266137 | 0.0170272 | 0.0366083 | 91.1512 | 87.6749 | 94.6055 |
| sp:place:cal:dh:lin:tick:bound | 443 | 37 | 1.1001 | 1.10401 | 0.00390885 | -0.00170317 | 0.00971526 | 0.0266137 | 0.0170272 | 0.0366083 |  |  |  |
| sp:wide:cal:dh:none:tick:drop | 392 | 33 | 2.42471 | 2.42756 | 0.00284651 | -0.0102877 | 0.016729 | 0.00726439 | -0.00571612 | 0.0211028 | 91.8112 | 79.4176 | 103.919 |
| sp:wide:cal:dh:lin:tick:drop | 392 | 33 | 2.42122 | 2.42756 | 0.00634039 | 0.00143104 | 0.0114887 | 0.0091308 | 0.00320462 | 0.0153788 |  |  |  |
| sp:wide:cal:dh:none:tick:bound | 395 | 33 | 2.41155 | 2.4145 | 0.00294293 | -0.0101277 | 0.0168019 | 0.00786645 | -0.00496725 | 0.0214778 | 91.9494 | 79.6212 | 103.975 |
| sp:wide:cal:dh:lin:tick:bound | 395 | 33 | 2.40821 | 2.4145 | 0.0062907 | 0.00153242 | 0.0112867 | 0.00935658 | 0.00361467 | 0.0154586 |  |  |  |

### stage2:q:form: ran, withheld (D21)

72 variants read the member; each is a registry record whose figures are withheld. By name and status:

| E | variant | status |
| --- | --- | --- |
| E43 | `place:form:harv:none:tick:drop` | scored |
| E16 | `sp:place:form:harv:none:tick:drop` | scored |
| E43 | `place:form:harv:lin:tick:drop` | scored |
| E16 | `sp:place:form:harv:lin:tick:drop` | scored |
| E43 | `place:form:harv:none:tick:bound` | scored |
| E16 | `sp:place:form:harv:none:tick:bound` | scored |
| E43 | `place:form:harv:lin:tick:bound` | scored |
| E16 | `sp:place:form:harv:lin:tick:bound` | scored |
| E43 | `wide:form:harv:none:tick:drop` | scored |
| E16 | `sp:wide:form:harv:none:tick:drop` | scored |
| E43 | `wide:form:harv:lin:tick:drop` | scored |
| E16 | `sp:wide:form:harv:lin:tick:drop` | scored |
| E43 | `wide:form:harv:none:tick:bound` | scored |
| E16 | `sp:wide:form:harv:none:tick:bound` | scored |
| E43 | `wide:form:harv:lin:tick:bound` | scored |
| E16 | `sp:wide:form:harv:lin:tick:bound` | scored |
| E43 | `place:form:hen:none:tick:drop` | scored |
| E16 | `sp:place:form:hen:none:tick:drop` | scored |
| E43 | `place:form:hen:lin:tick:drop` | scored |
| E16 | `sp:place:form:hen:lin:tick:drop` | scored |
| E43 | `place:form:hen:none:tick:bound` | scored |
| E16 | `sp:place:form:hen:none:tick:bound` | scored |
| E43 | `place:form:hen:lin:tick:bound` | scored |
| E16 | `sp:place:form:hen:lin:tick:bound` | scored |
| E43 | `wide:form:hen:none:tick:drop` | scored |
| E16 | `sp:wide:form:hen:none:tick:drop` | scored |
| E43 | `wide:form:hen:lin:tick:drop` | scored |
| E16 | `sp:wide:form:hen:lin:tick:drop` | scored |
| E43 | `wide:form:hen:none:tick:bound` | scored |
| E16 | `sp:wide:form:hen:none:tick:bound` | scored |
| E43 | `wide:form:hen:lin:tick:bound` | scored |
| E16 | `sp:wide:form:hen:lin:tick:bound` | scored |
| E43 | `place:form:qc:none:tick:drop` | scored |
| E16 | `sp:place:form:qc:none:tick:drop` | scored |
| E43 | `place:form:qc:lin:tick:drop` | scored |
| E16 | `sp:place:form:qc:lin:tick:drop` | scored |
| E43 | `place:form:qc:none:tick:bound` | scored |
| E16 | `sp:place:form:qc:none:tick:bound` | scored |
| E43 | `place:form:qc:lin:tick:bound` | scored |
| E16 | `sp:place:form:qc:lin:tick:bound` | scored |
| E43 | `wide:form:qc:none:tick:drop` | scored |
| E16 | `sp:wide:form:qc:none:tick:drop` | scored |
| E43 | `wide:form:qc:lin:tick:drop` | scored |
| E16 | `sp:wide:form:qc:lin:tick:drop` | scored |
| E43 | `wide:form:qc:none:tick:bound` | scored |
| E16 | `sp:wide:form:qc:none:tick:bound` | scored |
| E43 | `wide:form:qc:lin:tick:bound` | scored |
| E16 | `sp:wide:form:qc:lin:tick:bound` | scored |
| E43 | `place:form:dh:none:tick:drop` | scored |
| E16 | `sp:place:form:dh:none:tick:drop` | scored |
| E43 | `place:form:dh:lin:tick:drop` | scored |
| E16 | `sp:place:form:dh:lin:tick:drop` | scored |
| E43 | `place:form:dh:none:tick:bound` | scored |
| E16 | `sp:place:form:dh:none:tick:bound` | scored |
| E43 | `place:form:dh:lin:tick:bound` | scored |
| E16 | `sp:place:form:dh:lin:tick:bound` | scored |
| E43 | `wide:form:dh:none:tick:drop` | scored |
| E16 | `sp:wide:form:dh:none:tick:drop` | scored |
| E43 | `wide:form:dh:lin:tick:drop` | scored |
| E16 | `sp:wide:form:dh:lin:tick:drop` | scored |
| E43 | `wide:form:dh:none:tick:bound` | scored |
| E16 | `sp:wide:form:dh:none:tick:bound` | scored |
| E43 | `wide:form:dh:lin:tick:bound` | scored |
| E16 | `sp:wide:form:dh:lin:tick:bound` | scored |
| E43 | `place:form:harv:loglin` | failed |
| E43 | `place:form:hen:loglin` | failed |
| E43 | `place:form:qc:loglin` | failed |
| E43 | `place:form:dh:loglin` | failed |
| E43 | `wide:form:harv:loglin` | failed |
| E43 | `wide:form:hen:loglin` | failed |
| E43 | `wide:form:qc:loglin` | failed |
| E43 | `wide:form:dh:loglin` | failed |

### Failed variants (publishable inputs)

| E | variant | error |
| --- | --- | --- |
| E43 | `place:norm:harv:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:norm:hen:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:norm:qc:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:norm:dh:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:norm:harv:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:norm:hen:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:norm:qc:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:norm:dh:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:cal:harv:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:cal:hen:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:cal:qc:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `place:cal:dh:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:cal:harv:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:cal:hen:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:cal:qc:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |
| E43 | `wide:cal:dh:loglin` | not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid dividends alone (owner answer 6 on #99) |

## Alternatives rejected and why

- Adding the per-ticket de-vig to japan-c5-exotic-dividends's grid: that spec was frozen before its run read a row, and a changed spec is a new family, so the second de-vig is a family of its own and the first family's spec and records stand.
- Running Win, Quinella, Exacta, Trio, Trifecta and E44 here too: a pool that pays one combination per result has one share, R*100/(D+5), under either de-vig, so their records would repeat the first family's.
- A de-vig read from the Place and Wide pools' own final board or votes (the chance the money gives each ticket): not computable from the paid dividends, which are all the store holds; it needs #4/#21's logs of the full final boards.
- A per-pool scale on either de-vig fitted on the training races: a parameter fitted on outcomes makes the baseline a model; both de-vigs here are fixed formulas of the dividends.

## What this rules out

A mechanical reading of each publishable no-blend E43 variant's bounds; a measurement, not a verdict (nothing is adopted or dropped).

- `place:norm:harv:none:tick:drop`: ΔLL 0.008336 [0.006366, 0.0103]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.02746 [0.02554, 0.02935]; return per ¥100 76.57 [75.71, 77.44]
- `place:norm:harv:none:tick:bound`: ΔLL -0.005836 [-0.007479, -0.004238]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL 0.02575 [0.0242, 0.02724]; return per ¥100 84.44 [83.77, 85.12]
- `wide:norm:harv:none:tick:drop`: ΔLL -0.03256 [-0.03534, -0.02981]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.02783 [-0.03063, -0.02507]; return per ¥100 80.02 [78.66, 81.38]
- `wide:norm:harv:none:tick:bound`: ΔLL -0.03196 [-0.03473, -0.02922]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.02684 [-0.02963, -0.02408]; return per ¥100 80.22 [78.87, 81.58]
- `place:cal:harv:none:tick:drop`: ΔLL 0.005205 [0.003069, 0.007324]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.02432 [0.02226, 0.02637]; return per ¥100 76.74 [75.88, 77.6]
- `place:cal:harv:none:tick:bound`: ΔLL -0.008297 [-0.01009, -0.006545]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL 0.02329 [0.02162, 0.02492]; return per ¥100 84.57 [83.9, 85.24]
- `wide:cal:harv:none:tick:drop`: ΔLL -0.0376 [-0.04076, -0.03445]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.03287 [-0.03605, -0.02971]; return per ¥100 80 [78.63, 81.35]
- `wide:cal:harv:none:tick:bound`: ΔLL -0.03686 [-0.04001, -0.03371]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.03175 [-0.03491, -0.0286]; return per ¥100 80.2 [78.84, 81.55]
- `place:norm:hen:none:tick:drop`: ΔLL 0.02563 [0.02352, 0.02771]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04475 [0.04267, 0.04679]; return per ¥100 76.75 [75.88, 77.61]
- `place:norm:hen:none:tick:bound`: ΔLL 0.0009958 [-0.000686, 0.002649]: no difference excluded (the bounds straddle 0); its statute twin's ΔLL 0.03258 [0.03096, 0.03417]; return per ¥100 84.57 [83.9, 85.24]
- `wide:norm:hen:none:tick:drop`: ΔLL -0.01376 [-0.01544, -0.01208]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.009033 [-0.0107, -0.007349]; return per ¥100 80.05 [78.68, 81.41]
- `wide:norm:hen:none:tick:bound`: ΔLL -0.01429 [-0.01596, -0.01261]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.009173 [-0.01084, -0.007498]; return per ¥100 80.26 [78.89, 81.6]
- `place:cal:hen:none:tick:drop`: ΔLL 0.02755 [0.02542, 0.02963]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04667 [0.04459, 0.04871]; return per ¥100 76.76 [75.9, 77.62]
- `place:cal:hen:none:tick:bound`: ΔLL 0.003863 [0.002154, 0.00554]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.03545 [0.0338, 0.03705]; return per ¥100 84.59 [83.91, 85.26]
- `wide:cal:hen:none:tick:drop`: ΔLL -0.007963 [-0.009751, -0.006213]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.003236 [-0.005025, -0.001482]; return per ¥100 79.97 [78.61, 81.31]
- `wide:cal:hen:none:tick:bound`: ΔLL -0.008388 [-0.01018, -0.006658]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.003272 [-0.005054, -0.001536]; return per ¥100 80.17 [78.82, 81.51]
- `place:norm:qc:none:tick:drop`: ΔLL 0.02654 [0.02443, 0.02864]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04566 [0.0436, 0.04771]; return per ¥100 76.82 [75.96, 77.69]
- `place:norm:qc:none:tick:bound`: ΔLL 0.004212 [0.002502, 0.005903]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.03579 [0.03416, 0.03738]; return per ¥100 84.64 [83.97, 85.32]
- `wide:norm:qc:none:tick:drop`: ΔLL -0.007811 [-0.009682, -0.005975]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.003085 [-0.004958, -0.00126]; return per ¥100 80.11 [78.75, 81.46]
- `wide:norm:qc:none:tick:bound`: ΔLL -0.007704 [-0.009574, -0.005885]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.002588 [-0.004465, -0.0007705]; return per ¥100 80.32 [78.96, 81.66]
- `place:cal:qc:none:tick:drop`: ΔLL 0.02656 [0.02442, 0.02867]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04568 [0.04357, 0.04772]; return per ¥100 76.62 [75.75, 77.48]
- `place:cal:qc:none:tick:bound`: ΔLL 0.004487 [0.002759, 0.0062]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.03607 [0.0344, 0.03769]; return per ¥100 84.47 [83.8, 85.14]
- `wide:cal:qc:none:tick:drop`: ΔLL -0.007198 [-0.009142, -0.005301]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.002471 [-0.004412, -0.0005767]; return per ¥100 79.99 [78.62, 81.33]
- `wide:cal:qc:none:tick:bound`: ΔLL -0.00712 [-0.009058, -0.005232]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.002004 [-0.003947, -0.0001196]; return per ¥100 80.19 [78.84, 81.53]
- `place:norm:dh:none:tick:drop`: ΔLL 0.02291 [0.02089, 0.02491]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04203 [0.04007, 0.04399]; return per ¥100 76.74 [75.87, 77.61]
- `place:norm:dh:none:tick:bound`: ΔLL 0.001467 [-0.0001438, 0.003063]: no difference excluded (the bounds straddle 0); its statute twin's ΔLL 0.03305 [0.0315, 0.03457]; return per ¥100 84.57 [83.9, 85.25]
- `wide:norm:dh:none:tick:drop`: ΔLL -0.01356 [-0.01536, -0.01179]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.008829 [-0.01064, -0.00705]; return per ¥100 79.98 [78.62, 81.33]
- `wide:norm:dh:none:tick:bound`: ΔLL -0.01367 [-0.01548, -0.0119]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.008556 [-0.01035, -0.006775]; return per ¥100 80.18 [78.83, 81.53]
- `place:cal:dh:none:tick:drop`: ΔLL 0.02554 [0.02347, 0.02756]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.04466 [0.04263, 0.04664]; return per ¥100 76.65 [75.79, 77.51]
- `place:cal:dh:none:tick:bound`: ΔLL 0.003873 [0.002213, 0.005501]: C5 beats the dividends (lower bound above 0); its statute twin's ΔLL 0.03546 [0.03386, 0.037]; return per ¥100 84.49 [83.82, 85.16]
- `wide:cal:dh:none:tick:drop`: ΔLL -0.008464 [-0.01027, -0.006685]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.003738 [-0.005539, -0.001947]; return per ¥100 80.16 [78.8, 81.51]
- `wide:cal:dh:none:tick:bound`: ΔLL -0.008579 [-0.01039, -0.006806]: the dividends beat C5 (upper bound below 0); its statute twin's ΔLL -0.003462 [-0.005272, -0.001676]; return per ¥100 80.36 [79.01, 81.71]

## Leaderboard rows

Every record is a row of `horsetrack japan leaderboard` under `japan-c5-exotic-dividends-per-ticket/<variant>`: each no-blend E43 variant a measured row (its `return_per_100`), each no-blend E16 variant a `[super_premium]` segment row, every other variant (the linear and log-linear blends, E44, failures) unmeasured and unranked. The stage2:q:form rows are withheld: listed by option id only. Ranks are the leaderboard's (regenerated by the command).

- `japan-c5-exotic-dividends-per-ticket/place:norm:harv:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:harv:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:harv:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:harv:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:harv:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:harv:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:harv:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:harv:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:harv:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:harv:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:harv:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:harv:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:harv:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:harv:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:harv:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:harv:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:harv:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:harv:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:harv:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:harv:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:harv:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:harv:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:harv:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:harv:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:harv:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:harv:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:harv:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:harv:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:harv:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:harv:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:harv:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:harv:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:harv:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:harv:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:hen:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:hen:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:hen:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:hen:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:hen:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:hen:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:hen:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:hen:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:hen:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:hen:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:hen:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:hen:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:hen:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:hen:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:hen:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:hen:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:hen:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:hen:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:hen:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:hen:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:hen:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:hen:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:hen:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:hen:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:hen:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:hen:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:hen:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:hen:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:hen:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:hen:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:hen:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:hen:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:hen:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:hen:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:qc:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:qc:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:qc:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:qc:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:qc:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:qc:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:qc:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:qc:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:qc:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:qc:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:qc:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:qc:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:qc:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:qc:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:qc:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:qc:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:qc:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:qc:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:qc:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:qc:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:qc:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:qc:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:qc:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:qc:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:qc:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:qc:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:qc:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:qc:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:qc:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:qc:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:qc:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:qc:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:qc:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:qc:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:dh:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:dh:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:dh:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:dh:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:norm:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:norm:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:dh:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:dh:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:dh:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:dh:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:norm:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:norm:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:dh:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:dh:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:dh:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:dh:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/place:cal:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:cal:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:dh:none:tick:drop`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:dh:none:tick:drop`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:dh:lin:tick:drop` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:dh:none:tick:bound`
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:dh:none:tick:bound`[super_premium]
- `japan-c5-exotic-dividends-per-ticket/wide:cal:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:cal:dh:lin:tick:bound` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:dh:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:dh:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:dh:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:dh:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:dh:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:dh:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/place:form:dh:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:place:form:dh:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:dh:none:tick:drop` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:dh:none:tick:drop`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:dh:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:dh:lin:tick:drop` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:dh:none:tick:bound` (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:dh:none:tick:bound`[super_premium] (withheld, D21)
- `japan-c5-exotic-dividends-per-ticket/wide:form:dh:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/sp:wide:form:dh:lin:tick:bound` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:harv:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:hen:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:qc:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:norm:dh:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:harv:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:hen:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:qc:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:norm:dh:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:harv:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:hen:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:qc:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:cal:dh:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:harv:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:hen:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:qc:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:cal:dh:loglin` (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:harv:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:hen:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:qc:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/place:form:dh:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:harv:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:hen:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:qc:loglin` (withheld, D21) (unmeasured)
- `japan-c5-exotic-dividends-per-ticket/wide:form:dh:loglin` (withheld, D21) (unmeasured)

## Follow-ups filed

Proposed (`status:proposed` under epic #73; not linked as its sub-issues, since #73 is at GitHub's cap of 100):

- #671: run the log-linear pool (`exotic.pool_blend`) once #4/#21's odds logs hold full final boards of the exotic pools, and settle the leaderboard bet at the full board's expected value rather than the most probable combination.
- #672: score C5 at the decision time (D2: T-2) on logged boards instead of the final Win price, so its leaderboard rows are decision-time selections, not look-ahead.
- #673: feed C5 the D9-compliant meta-model once #91's and #98's members land (stage2:q:form is #87's documented D9 shortfall).
- #674: append a run's registry records in one batch (`Registry.append` re-verifies the hash chain on every record, so a run of hundreds of variants spends most of its wall time appending).
