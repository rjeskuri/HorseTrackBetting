# Experiment registry summary

Written by `horsetrack.core.eval.registry.write_summary` from the run registry under
the data directory; do not edit by hand. Every run is listed, failures included. A run
built on a licensed source shows no values here.

## japan-c4-meta-final-win

| E id | Variant | Status | Fold | Metric | Value | CI | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E40 | market:normalised | scored | 2024-01-01 | delta_ll_vs_calibrated_market | -0.00226525 | - | 912 |
| E40 | market:normalised | scored | 2024-04-01 | delta_ll_vs_calibrated_market | 0.00152545 | - | 888 |
| E40 | market:normalised | scored | 2024-07-01 | delta_ll_vs_calibrated_market | -0.000653317 | - | 815 |
| E40 | market:normalised | scored | 2024-10-01 | delta_ll_vs_calibrated_market | -0.000705408 | - | 839 |
| E40 | market:normalised | scored | 2025-01-01 | delta_ll_vs_calibrated_market | 6.32356e-05 | - | 864 |
| E40 | market:normalised | scored | 2025-04-01 | delta_ll_vs_calibrated_market | -0.00189331 | - | 863 |
| E40 | market:normalised | scored | 2025-07-01 | delta_ll_vs_calibrated_market | -0.00301432 | - | 888 |
| E40 | market:normalised | scored | 2025-10-01 | delta_ll_vs_calibrated_market | -0.000707431 | - | 840 |
| E40 | market:normalised | scored | 2026-01-01 | delta_ll_vs_calibrated_market | 0.000473552 | - | 858 |
| E40 | market:normalised | scored | 2026-04-01 | delta_ll_vs_calibrated_market | -0.00157815 | - | 864 |
| E40 | market:normalised | scored | 2026-07-01 | delta_ll_vs_calibrated_market | -0.00045196 | - | 888 |
| E40 | market:normalised | scored | 2026-10-01 | delta_ll_vs_calibrated_market | -0.00264131 | - | 528 |
| E40 | market:normalised | scored | all_folds | mean_log_loss | 1.91131 | - | 10047 |
| E40 | market:normalised | scored | all_folds | baseline_mean_log_loss | 1.91037 | - | 10047 |
| E40 | market:normalised | scored | all_folds | delta_ll_vs_calibrated_market | -0.000939486 | [-0.00169459, -0.000164998] | 10047 |
| E40 | market:normalised | scored | all_folds | delta_pseudo_r2_vs_calibrated_market | -0.000362493 | - | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | q_mass_above[0.000000] | 0.774681 | [0.769833, 0.77952] | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | q_mass_above[0.095310] | 0 | [0, 0] | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | q_mass_above[0.223144] | 0 | [0, 0] | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | q_mass_above[0.405465] | 0 | [0, 0] | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | q_mass_above[0.693147] | 0 | [0, 0] | 10047 |
| E9 | ceiling:calibrated_market | scored | all_folds | rms_ln_pi_over_q | 0.0582411 | [0.0577251, 0.0587395] | 10047 |
| E40 | member_alone:form | scored | withheld (licensed source) | | | | |
| E40 | stage2:q:form | scored | withheld (licensed source) | | | | |
| E9 | ceiling:stage2:q:form | scored | withheld (licensed source) | | | | |
| E40 | offset:q:form | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:q:form | scored | withheld (licensed source) | | | | |
| E40 | offset:cal:form | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:cal:form | scored | withheld (licensed source) | | | | |
| E40 | encompassing:form | scored | withheld (licensed source) | | | | |
| E40 | member_alone:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | stage2:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:stage2:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | offset:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | offset:cal:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:cal:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | encompassing:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | member_alone:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | stage2:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:stage2:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | offset:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | offset:cal:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E9 | ceiling:offset:cal:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | encompassing:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:stage2:q:form | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:q:form | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:cal:form | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:encompassing:form | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:stage2:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:q:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:cal:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:encompassing:lgbm_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:stage2:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:q:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:offset:cal:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
| E40 | meeting_clustered:encompassing:clogit_no_jrdb | scored | withheld (licensed source) | | | | |
