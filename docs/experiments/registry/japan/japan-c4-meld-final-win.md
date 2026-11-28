# Experiment registry summary

Written by `horsetrack.core.eval.registry.write_summary` from the run registry under
the data directory; do not edit by hand. Every run is listed, failures included. A run
built on a licensed source shows no values here.

## japan-c4-meld-final-win

| E id | Variant | Status | Fold | Metric | Value | CI | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E40 | not_run:competition | failed | - | - | - | - | - |
| E40 | not_run:lineage | failed | - | - | - | - | - |
| E40 | not_run:p_speed | failed | - | - | - | - | - |
| E41 | e41:reff | failed | - | - | - | - | - |
| E40 | from2020:market_only | scored | all_folds | mean_log_loss | 1.9235 | - | 5729 |
| E40 | from2020:market_only | scored | development | return_per_100 | 83.023 | [74.8278, 91.2091] | 784 |
| E40 | from2020:market_only | scored | development | n_meetings | 330 | - | 784 |
| E40 | from2020:market_only | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 63.5603 | [56.9292, 70.5391] | 39165 |
| E40 | from2020:market_only | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 81.9498 | [79.7242, 84.2885] | 31332 |
| E40 | from2020:market_only | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 77.8511 | [75.5907, 80.0966] | 7050 |
| E40 | from2020:market_only | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 83.023 | [74.8278, 91.2091] | 784 |
| E40 | from2020:standalone:rating | scored | all_folds | mean_log_loss | 1.9229 | - | 5729 |
| E40 | from2020:standalone:rating | scored | all_folds | delta_ll_vs_market_only | 0.000595893 | [0.000124924, 0.00106501] | 5729 |
| E40 | from2020:standalone:rating | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000932199 | - | 863 |
| E40 | from2020:standalone:rating | scored | 2025-07-01 | delta_ll_vs_market_only | -0.000141883 | - | 888 |
| E40 | from2020:standalone:rating | scored | 2025-10-01 | delta_ll_vs_market_only | 0.00110037 | - | 840 |
| E40 | from2020:standalone:rating | scored | 2026-01-01 | delta_ll_vs_market_only | 1.06816e-05 | - | 858 |
| E40 | from2020:standalone:rating | scored | 2026-04-01 | delta_ll_vs_market_only | 0.000346236 | - | 864 |
| E40 | from2020:standalone:rating | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00103936 | - | 888 |
| E40 | from2020:standalone:rating | scored | 2026-10-01 | delta_ll_vs_market_only | 0.00109811 | - | 528 |
| E40 | from2020:standalone:rating | scored | development | return_per_100 | 88.4184 | [72.8019, 105.039] | 784 |
| E40 | from2020:standalone:rating | scored | development | n_meetings | 361 | - | 784 |
| E40 | from2020:standalone:rating | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.61 | [58.2335, 71.303] | 39165 |
| E40 | from2020:standalone:rating | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.5289 | [76.88, 82.2681] | 31332 |
| E40 | from2020:standalone:rating | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 82.1787 | [77.9585, 86.6395] | 7050 |
| E40 | from2020:standalone:rating | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 88.4184 | [72.8019, 105.039] | 784 |
| E40 | from2020:standalone:ix_base | scored | all_folds | mean_log_loss | 1.92269 | - | 5729 |
| E40 | from2020:standalone:ix_base | scored | all_folds | delta_ll_vs_market_only | 0.000807556 | [0.000314226, 0.00129698] | 5729 |
| E40 | from2020:standalone:ix_base | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000699464 | - | 863 |
| E40 | from2020:standalone:ix_base | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000474118 | - | 888 |
| E40 | from2020:standalone:ix_base | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000887062 | - | 840 |
| E40 | from2020:standalone:ix_base | scored | 2026-01-01 | delta_ll_vs_market_only | 8.26224e-05 | - | 858 |
| E40 | from2020:standalone:ix_base | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00141164 | - | 864 |
| E40 | from2020:standalone:ix_base | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00120585 | - | 888 |
| E40 | from2020:standalone:ix_base | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000938178 | - | 528 |
| E40 | from2020:standalone:ix_base | scored | development | return_per_100 | 79.0816 | [67.4073, 91.0018] | 784 |
| E40 | from2020:standalone:ix_base | scored | development | n_meetings | 359 | - | 784 |
| E40 | from2020:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.8237 | [58.3168, 71.4476] | 39165 |
| E40 | from2020:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.1999 | [76.5935, 81.9153] | 31332 |
| E40 | from2020:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.4922 | [78.8001, 88.4765] | 7050 |
| E40 | from2020:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.0816 | [67.4073, 91.0018] | 784 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | mean_log_loss | 1.92269 | - | 5729 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | delta_ll_vs_market_only | 0.000807556 | [0.000314226, 0.00129698] | 5729 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000699465 | - | 863 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000474117 | - | 888 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000887063 | - | 840 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2026-01-01 | delta_ll_vs_market_only | 8.26233e-05 | - | 858 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00141164 | - | 864 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00120585 | - | 888 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000938179 | - | 528 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | development | return_per_100 | 79.0816 | [67.4073, 91.0018] | 784 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | development | n_meetings | 359 | - | 784 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.8237 | [58.3168, 71.4476] | 39165 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.1999 | [76.5935, 81.9153] | 31332 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.4922 | [78.8001, 88.4765] | 7050 |
| E40 | from2020:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.0816 | [67.4073, 91.0018] | 784 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | mean_log_loss | 1.9227 | - | 5729 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | delta_ll_vs_market_only | 0.000801539 | [0.000322941, 0.00127517] | 5729 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000759471 | - | 863 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000501647 | - | 888 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000790942 | - | 840 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2026-01-01 | delta_ll_vs_market_only | 0.000155141 | - | 858 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00135465 | - | 864 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00114313 | - | 888 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000962326 | - | 528 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | development | return_per_100 | 79.0051 | [67.5149, 90.9706] | 784 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | development | n_meetings | 358 | - | 784 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.3151 | [57.8617, 70.9547] | 39165 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.7696 | [77.1992, 82.4575] | 31332 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.7943 | [79.1614, 88.7704] | 7050 |
| E40 | from2020:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.0051 | [67.5149, 90.9706] | 784 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | mean_log_loss | 1.9227 | - | 5729 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | delta_ll_vs_market_only | 0.000798053 | [0.00031064, 0.00128214] | 5729 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000649255 | - | 863 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000459271 | - | 888 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000885517 | - | 840 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2026-01-01 | delta_ll_vs_market_only | 0.000106714 | - | 858 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00139851 | - | 864 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00119903 | - | 888 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000938363 | - | 528 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | development | return_per_100 | 79.0816 | [67.5369, 91.022] | 784 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | development | n_meetings | 360 | - | 784 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.7834 | [58.3073, 71.3941] | 39165 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.331 | [76.707, 82.0541] | 31332 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.1333 | [78.6428, 88.0164] | 7050 |
| E40 | from2020:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.0816 | [67.5369, 91.022] | 784 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | mean_log_loss | 1.9227 | - | 5729 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | delta_ll_vs_market_only | 0.000801719 | [0.00032282, 0.00127967] | 5729 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000507868 | - | 863 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2025-07-01 | delta_ll_vs_market_only | 0.0005084 | - | 888 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000899595 | - | 840 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2026-01-01 | delta_ll_vs_market_only | 6.15479e-05 | - | 858 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00144467 | - | 864 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00135868 | - | 888 |
| E40 | from2020:standalone:ix_trainer_debut | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000833572 | - | 528 |
| E40 | from2020:standalone:ix_trainer_debut | scored | development | return_per_100 | 80.4082 | [68.8507, 92.7396] | 784 |
| E40 | from2020:standalone:ix_trainer_debut | scored | development | n_meetings | 362 | - | 784 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 63.9436 | [57.4182, 70.6991] | 39165 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 80.2451 | [77.5973, 83.0082] | 31332 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.5887 | [79.0562, 88.3659] | 7050 |
| E40 | from2020:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 80.4082 | [68.8507, 92.7396] | 784 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | mean_log_loss | 1.92269 | - | 5729 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | delta_ll_vs_market_only | 0.000812295 | [0.000325633, 0.00129577] | 5729 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000708271 | - | 863 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000465849 | - | 888 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000886533 | - | 840 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2026-01-01 | delta_ll_vs_market_only | 0.00012575 | - | 858 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00137872 | - | 864 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00123746 | - | 888 |
| E40 | from2020:standalone:ix_transfer_step | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000920589 | - | 528 |
| E40 | from2020:standalone:ix_transfer_step | scored | development | return_per_100 | 79.3495 | [67.9643, 91.5963] | 784 |
| E40 | from2020:standalone:ix_transfer_step | scored | development | n_meetings | 360 | - | 784 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.7331 | [58.2515, 71.374] | 39165 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 79.2544 | [76.6482, 81.9534] | 31332 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 83.7234 | [79.2457, 88.5091] | 7050 |
| E40 | from2020:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.3495 | [67.9643, 91.5963] | 784 |
| E40 | from2020:meld:all minus ability | scored | all_folds | mean_log_loss | 1.92308 | - | 5729 |
| E40 | from2020:meld:all minus ability | scored | all_folds | delta_ll_vs_market_only | 0.000421754 | [-0.000183119, 0.00101461] | 5729 |
| E40 | from2020:meld:all minus ability | scored | 2025-04-01 | delta_ll_vs_market_only | -0.000195114 | - | 863 |
| E40 | from2020:meld:all minus ability | scored | 2025-07-01 | delta_ll_vs_market_only | -4.8943e-05 | - | 888 |
| E40 | from2020:meld:all minus ability | scored | 2025-10-01 | delta_ll_vs_market_only | 0.00120883 | - | 840 |
| E40 | from2020:meld:all minus ability | scored | 2026-01-01 | delta_ll_vs_market_only | -0.00098909 | - | 858 |
| E40 | from2020:meld:all minus ability | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00130178 | - | 864 |
| E40 | from2020:meld:all minus ability | scored | 2026-07-01 | delta_ll_vs_market_only | 0.0010546 | - | 888 |
| E40 | from2020:meld:all minus ability | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000757711 | - | 528 |
| E40 | from2020:meld:all minus ability | scored | development | return_per_100 | 79.0051 | [66.7087, 91.8403] | 784 |
| E40 | from2020:meld:all minus ability | scored | development | n_meetings | 364 | - | 784 |
| E40 | from2020:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 66.4164 | [60.0979, 72.9967] | 39165 |
| E40 | from2020:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.7572 | [75.0691, 80.5405] | 31332 |
| E40 | from2020:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 81.0638 | [76.5149, 85.9193] | 7050 |
| E40 | from2020:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 79.0051 | [66.7087, 91.8403] | 784 |
| E40 | from2020:encompassing:rating | scored | in_sample | encompassing_coef | -0.0677422 | [-0.104795, -0.030689] | 5729 |
| E40 | from2020:encompassing:ix_base | scored | in_sample | encompassing_coef | -0.0877892 | [-0.127088, -0.0484901] | 5729 |
| E40 | from2020:encompassing:ix_horse_x_jockey | scored | in_sample | encompassing_coef | -0.0877892 | [-0.127088, -0.0484901] | 5729 |
| E40 | from2020:encompassing:ix_trainer_x_track | scored | in_sample | encompassing_coef | -0.0883458 | [-0.127447, -0.0492446] | 5729 |
| E40 | from2020:encompassing:ix_horse_aptitude | scored | in_sample | encompassing_coef | -0.0873315 | [-0.126578, -0.0480854] | 5729 |
| E40 | from2020:encompassing:ix_trainer_debut | scored | in_sample | encompassing_coef | -0.0882714 | [-0.127721, -0.0488216] | 5729 |
| E40 | from2020:encompassing:ix_transfer_step | scored | in_sample | encompassing_coef | -0.0886125 | [-0.127828, -0.0493972] | 5729 |
| E40 | from2015:market_only | scored | all_folds | mean_log_loss | 1.92566 | - | 22959 |
| E40 | from2015:market_only | scored | development | return_per_100 | 75.7066 | [72.2011, 79.2145] | 3156 |
| E40 | from2015:market_only | scored | development | n_meetings | 625 | - | 3156 |
| E40 | from2015:market_only | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.835 | [61.944, 67.8025] | 157763 |
| E40 | from2015:market_only | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.824 | [76.6199, 79.0588] | 126211 |
| E40 | from2015:market_only | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 84.1938 | [80.5066, 88.2656] | 28397 |
| E40 | from2015:market_only | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 75.7066 | [72.2011, 79.2145] | 3156 |
| E40 | from2015:standalone:ix_base | scored | all_folds | mean_log_loss | 1.92534 | - | 22959 |
| E40 | from2015:standalone:ix_base | scored | all_folds | delta_ll_vs_market_only | 0.000318926 | [0.000138169, 0.000497669] | 22959 |
| E40 | from2015:standalone:ix_base | scored | 2020-04-01 | delta_ll_vs_market_only | -0.000606822 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2020-07-01 | delta_ll_vs_market_only | -0.000260558 | - | 816 |
| E40 | from2015:standalone:ix_base | scored | 2020-10-01 | delta_ll_vs_market_only | 0.000407448 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000171592 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2021-04-01 | delta_ll_vs_market_only | 0.00044009 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000586362 | - | 816 |
| E40 | from2015:standalone:ix_base | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00108712 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2022-01-01 | delta_ll_vs_market_only | 0.00143399 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000972255 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2022-07-01 | delta_ll_vs_market_only | -0.00015221 | - | 840 |
| E40 | from2015:standalone:ix_base | scored | 2022-10-01 | delta_ll_vs_market_only | 3.64164e-05 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000301678 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000666713 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000745802 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000278063 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000865323 | - | 912 |
| E40 | from2015:standalone:ix_base | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000659076 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2024-07-01 | delta_ll_vs_market_only | 0.00086732 | - | 815 |
| E40 | from2015:standalone:ix_base | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000114061 | - | 839 |
| E40 | from2015:standalone:ix_base | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000239816 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2025-04-01 | delta_ll_vs_market_only | 0.00058672 | - | 863 |
| E40 | from2015:standalone:ix_base | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000431905 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000743985 | - | 840 |
| E40 | from2015:standalone:ix_base | scored | 2026-01-01 | delta_ll_vs_market_only | 8.78466e-05 | - | 858 |
| E40 | from2015:standalone:ix_base | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00116489 | - | 864 |
| E40 | from2015:standalone:ix_base | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00102798 | - | 888 |
| E40 | from2015:standalone:ix_base | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000816483 | - | 528 |
| E40 | from2015:standalone:ix_base | scored | development | return_per_100 | 81.2928 | [76.3561, 86.526] | 3156 |
| E40 | from2015:standalone:ix_base | scored | development | n_meetings | 863 | - | 3156 |
| E40 | from2015:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.1558 | [62.2576, 68.0738] | 157763 |
| E40 | from2015:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.1159 | [75.5778, 78.7013] | 126211 |
| E40 | from2015:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 84.9382 | [81.5149, 88.7122] | 28397 |
| E40 | from2015:standalone:ix_base | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 81.2928 | [76.3561, 86.526] | 3156 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | mean_log_loss | 1.92534 | - | 22959 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | delta_ll_vs_market_only | 0.000318926 | [0.00013817, 0.000497669] | 22959 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2020-04-01 | delta_ll_vs_market_only | -0.000606822 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2020-07-01 | delta_ll_vs_market_only | -0.000260558 | - | 816 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2020-10-01 | delta_ll_vs_market_only | 0.000407448 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000171591 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2021-04-01 | delta_ll_vs_market_only | 0.000440089 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000586363 | - | 816 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00108712 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2022-01-01 | delta_ll_vs_market_only | 0.00143399 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000972256 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2022-07-01 | delta_ll_vs_market_only | -0.00015221 | - | 840 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2022-10-01 | delta_ll_vs_market_only | 3.64176e-05 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000301678 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000666714 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000745802 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000278064 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000865324 | - | 912 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000659077 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2024-07-01 | delta_ll_vs_market_only | 0.00086732 | - | 815 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000114061 | - | 839 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000239816 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000586721 | - | 863 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000431904 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000743986 | - | 840 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2026-01-01 | delta_ll_vs_market_only | 8.78474e-05 | - | 858 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00116489 | - | 864 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00102798 | - | 888 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000816483 | - | 528 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | development | return_per_100 | 81.2928 | [76.3561, 86.526] | 3156 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | development | n_meetings | 863 | - | 3156 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.1558 | [62.2576, 68.0738] | 157763 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.1159 | [75.5778, 78.7013] | 126211 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 84.9382 | [81.5149, 88.7122] | 28397 |
| E40 | from2015:standalone:ix_horse_x_jockey | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 81.2928 | [76.3561, 86.526] | 3156 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | mean_log_loss | 1.92535 | - | 22959 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | delta_ll_vs_market_only | 0.000307981 | [0.000129916, 0.000484986] | 22959 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2020-04-01 | delta_ll_vs_market_only | -0.000592852 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2020-07-01 | delta_ll_vs_market_only | -0.000267511 | - | 816 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2020-10-01 | delta_ll_vs_market_only | 0.000428135 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000184867 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2021-04-01 | delta_ll_vs_market_only | 0.000431386 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000543073 | - | 816 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00105919 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2022-01-01 | delta_ll_vs_market_only | 0.00139976 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000973386 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2022-07-01 | delta_ll_vs_market_only | -0.000114041 | - | 840 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2022-10-01 | delta_ll_vs_market_only | 1.01829e-05 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000289847 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2023-04-01 | delta_ll_vs_market_only | 0.00061799 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000739673 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000231275 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000877317 | - | 912 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000645931 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2024-07-01 | delta_ll_vs_market_only | 0.000826717 | - | 815 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000164838 | - | 839 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000236196 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2025-04-01 | delta_ll_vs_market_only | 0.00064035 | - | 863 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000455143 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000668884 | - | 840 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2026-01-01 | delta_ll_vs_market_only | 0.000143197 | - | 858 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00112491 | - | 864 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2026-07-01 | delta_ll_vs_market_only | 0.000981467 | - | 888 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000836218 | - | 528 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | development | return_per_100 | 82.424 | [77.3527, 87.5545] | 3156 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | development | n_meetings | 853 | - | 3156 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.3109 | [62.3976, 68.2436] | 157763 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 76.8186 | [75.2892, 78.396] | 126211 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 85.2724 | [81.8875, 88.9324] | 28397 |
| E40 | from2015:standalone:ix_trainer_x_track | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 82.424 | [77.3527, 87.5545] | 3156 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | mean_log_loss | 1.92534 | - | 22959 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | delta_ll_vs_market_only | 0.000313736 | [0.000136124, 0.00048878] | 22959 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2020-04-01 | delta_ll_vs_market_only | -0.000603535 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2020-07-01 | delta_ll_vs_market_only | -0.000246969 | - | 816 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2020-10-01 | delta_ll_vs_market_only | 0.000396314 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000177602 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2021-04-01 | delta_ll_vs_market_only | 0.000437932 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000579625 | - | 816 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00103637 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2022-01-01 | delta_ll_vs_market_only | 0.0014233 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000934977 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2022-07-01 | delta_ll_vs_market_only | -0.000158988 | - | 840 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2022-10-01 | delta_ll_vs_market_only | 9.46933e-05 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000291695 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000671556 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000749503 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2023-10-01 | delta_ll_vs_market_only | 0.00028926 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000867718 | - | 912 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000634142 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2024-07-01 | delta_ll_vs_market_only | 0.000829739 | - | 815 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000104419 | - | 839 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000218009 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000543407 | - | 863 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000417637 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2025-10-01 | delta_ll_vs_market_only | 0.00073996 | - | 840 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2026-01-01 | delta_ll_vs_market_only | 0.000106699 | - | 858 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00114936 | - | 864 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00101778 | - | 888 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000812976 | - | 528 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | development | return_per_100 | 81.7966 | [76.7138, 86.9711] | 3156 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | development | n_meetings | 856 | - | 3156 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.0767 | [62.1719, 67.9935] | 157763 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.0458 | [75.5098, 78.6267] | 126211 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 85.633 | [82.1781, 89.3879] | 28397 |
| E40 | from2015:standalone:ix_horse_aptitude | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 81.7966 | [76.7138, 86.9711] | 3156 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | mean_log_loss | 1.92536 | - | 22959 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | delta_ll_vs_market_only | 0.000298908 | [0.000129134, 0.000465678] | 22959 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2020-04-01 | delta_ll_vs_market_only | -0.000576099 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2020-07-01 | delta_ll_vs_market_only | -0.00013791 | - | 816 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2020-10-01 | delta_ll_vs_market_only | 0.000372226 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000251412 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2021-04-01 | delta_ll_vs_market_only | 0.00029022 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000562253 | - | 816 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2021-10-01 | delta_ll_vs_market_only | 0.000952602 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2022-01-01 | delta_ll_vs_market_only | 0.00133406 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000722075 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2022-07-01 | delta_ll_vs_market_only | 0.000113153 | - | 840 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2022-10-01 | delta_ll_vs_market_only | 8.65144e-05 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000227559 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000576278 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000833534 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000256617 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000662897 | - | 912 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000735578 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2024-07-01 | delta_ll_vs_market_only | 0.000987626 | - | 815 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000247348 | - | 839 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000461008 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000421507 | - | 863 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000446229 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000736358 | - | 840 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2026-01-01 | delta_ll_vs_market_only | 7.29633e-05 | - | 858 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00116255 | - | 864 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00112055 | - | 888 |
| E40 | from2015:standalone:ix_trainer_debut | scored | 2026-10-01 | delta_ll_vs_market_only | 0.00071882 | - | 528 |
| E40 | from2015:standalone:ix_trainer_debut | scored | development | return_per_100 | 83.5773 | [78.2653, 88.9161] | 3156 |
| E40 | from2015:standalone:ix_trainer_debut | scored | development | n_meetings | 839 | - | 3156 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 64.652 | [61.7432, 67.606] | 157763 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.6258 | [76.1298, 79.1354] | 126211 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 85.2167 | [81.7286, 88.992] | 28397 |
| E40 | from2015:standalone:ix_trainer_debut | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 83.5773 | [78.2653, 88.9161] | 3156 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | mean_log_loss | 1.92534 | - | 22959 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | delta_ll_vs_market_only | 0.000315803 | [0.000137188, 0.000493786] | 22959 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2020-04-01 | delta_ll_vs_market_only | -0.00059686 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2020-07-01 | delta_ll_vs_market_only | -0.000278979 | - | 816 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2020-10-01 | delta_ll_vs_market_only | 0.00039607 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2021-01-01 | delta_ll_vs_market_only | -0.000173266 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2021-04-01 | delta_ll_vs_market_only | 0.000432535 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2021-07-01 | delta_ll_vs_market_only | 0.000575225 | - | 816 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00107465 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2022-01-01 | delta_ll_vs_market_only | 0.00143843 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2022-04-01 | delta_ll_vs_market_only | 0.000948166 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2022-07-01 | delta_ll_vs_market_only | -0.000147118 | - | 840 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2022-10-01 | delta_ll_vs_market_only | 6.22717e-05 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000327835 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000670586 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000740218 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000281998 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2024-01-01 | delta_ll_vs_market_only | -0.000884761 | - | 912 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2024-04-01 | delta_ll_vs_market_only | 0.0006569 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2024-07-01 | delta_ll_vs_market_only | 0.000833777 | - | 815 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2024-10-01 | delta_ll_vs_market_only | -0.000133918 | - | 839 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000233435 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000596758 | - | 863 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2025-07-01 | delta_ll_vs_market_only | 0.000425424 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000746746 | - | 840 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2026-01-01 | delta_ll_vs_market_only | 0.000121059 | - | 858 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00114242 | - | 864 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2026-07-01 | delta_ll_vs_market_only | 0.00105521 | - | 888 |
| E40 | from2015:standalone:ix_transfer_step | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000803501 | - | 528 |
| E40 | from2015:standalone:ix_transfer_step | scored | development | return_per_100 | 81.0456 | [75.9922, 86.18] | 3156 |
| E40 | from2015:standalone:ix_transfer_step | scored | development | n_meetings | 862 | - | 3156 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.1106 | [62.2108, 68.0233] | 157763 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 77.174 | [75.6416, 78.7731] | 126211 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 84.9583 | [81.5337, 88.7122] | 28397 |
| E40 | from2015:standalone:ix_transfer_step | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 81.0456 | [75.9922, 86.18] | 3156 |
| E40 | from2015:meld:all minus ability | scored | all_folds | mean_log_loss | 1.92546 | - | 22959 |
| E40 | from2015:meld:all minus ability | scored | all_folds | delta_ll_vs_market_only | 0.000194505 | [-4.42116e-05, 0.000433801] | 22959 |
| E40 | from2015:meld:all minus ability | scored | 2020-04-01 | delta_ll_vs_market_only | 0.000300968 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2020-07-01 | delta_ll_vs_market_only | -0.00126879 | - | 816 |
| E40 | from2015:meld:all minus ability | scored | 2020-10-01 | delta_ll_vs_market_only | -0.000292828 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2021-01-01 | delta_ll_vs_market_only | 0.000597283 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2021-04-01 | delta_ll_vs_market_only | -3.14578e-05 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2021-07-01 | delta_ll_vs_market_only | -0.00040027 | - | 816 |
| E40 | from2015:meld:all minus ability | scored | 2021-10-01 | delta_ll_vs_market_only | 0.00150302 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2022-01-01 | delta_ll_vs_market_only | 0.000579464 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2022-04-01 | delta_ll_vs_market_only | 0.00162947 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2022-07-01 | delta_ll_vs_market_only | -0.000701449 | - | 840 |
| E40 | from2015:meld:all minus ability | scored | 2022-10-01 | delta_ll_vs_market_only | -0.00128687 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2023-01-01 | delta_ll_vs_market_only | -0.000347022 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2023-04-01 | delta_ll_vs_market_only | 0.000600319 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2023-07-01 | delta_ll_vs_market_only | -0.000437459 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2023-10-01 | delta_ll_vs_market_only | 0.000195357 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2024-01-01 | delta_ll_vs_market_only | -0.00077662 | - | 912 |
| E40 | from2015:meld:all minus ability | scored | 2024-04-01 | delta_ll_vs_market_only | 0.000633871 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2024-07-01 | delta_ll_vs_market_only | 0.00135287 | - | 815 |
| E40 | from2015:meld:all minus ability | scored | 2024-10-01 | delta_ll_vs_market_only | 0.000325571 | - | 839 |
| E40 | from2015:meld:all minus ability | scored | 2025-01-01 | delta_ll_vs_market_only | -0.000313961 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2025-04-01 | delta_ll_vs_market_only | 0.000549539 | - | 863 |
| E40 | from2015:meld:all minus ability | scored | 2025-07-01 | delta_ll_vs_market_only | 0.00035627 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2025-10-01 | delta_ll_vs_market_only | 0.000671882 | - | 840 |
| E40 | from2015:meld:all minus ability | scored | 2026-01-01 | delta_ll_vs_market_only | -0.000687455 | - | 858 |
| E40 | from2015:meld:all minus ability | scored | 2026-04-01 | delta_ll_vs_market_only | 0.00121671 | - | 864 |
| E40 | from2015:meld:all minus ability | scored | 2026-07-01 | delta_ll_vs_market_only | 0.000742034 | - | 888 |
| E40 | from2015:meld:all minus ability | scored | 2026-10-01 | delta_ll_vs_market_only | 0.000685655 | - | 528 |
| E40 | from2015:meld:all minus ability | scored | development | return_per_100 | 80.9094 | [75.2586, 86.727] | 3156 |
| E40 | from2015:meld:all minus ability | scored | development | n_meetings | 914 | - | 3156 |
| E40 | from2015:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0,0.5]] | 65.844 | [62.9746, 68.7509] | 157763 |
| E40 | from2015:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.5,0.9]] | 76.2827 | [74.6638, 77.9843] | 126211 |
| E40 | from2015:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.9,0.99]] | 84.8604 | [81.3503, 88.6498] | 28397 |
| E40 | from2015:meld:all minus ability | scored | all_folds | tail_return_per_100[claim_q[0.99,1]] | 80.9094 | [75.2586, 86.727] | 3156 |
| E40 | from2015:encompassing:ix_base | scored | in_sample | encompassing_coef | -0.0534344 | [-0.0742085, -0.0326603] | 22959 |
| E40 | from2015:encompassing:ix_horse_x_jockey | scored | in_sample | encompassing_coef | -0.0534344 | [-0.0742085, -0.0326603] | 22959 |
| E40 | from2015:encompassing:ix_trainer_x_track | scored | in_sample | encompassing_coef | -0.0523012 | [-0.0730307, -0.0315716] | 22959 |
| E40 | from2015:encompassing:ix_horse_aptitude | scored | in_sample | encompassing_coef | -0.0532393 | [-0.0740236, -0.0324549] | 22959 |
| E40 | from2015:encompassing:ix_trainer_debut | scored | in_sample | encompassing_coef | -0.0529057 | [-0.0735947, -0.0322166] | 22959 |
| E40 | from2015:encompassing:ix_transfer_step | scored | in_sample | encompassing_coef | -0.0531583 | [-0.0739187, -0.032398] | 22959 |
| E40 | from2015:ambiguity | scored | withheld (licensed source) | | | | |
| E40 | from2015:correlations | scored | withheld (licensed source) | | | | |
| E41 | from2015:e41:ability:c[0.25] | scored | withheld (licensed source) | | | | |
| E41 | from2015:e41:ability:c[1] | scored | withheld (licensed source) | | | | |
| E41 | from2015:e41:ability:c[4] | scored | withheld (licensed source) | | | | |
| E42 | from2015:e42:gates[debut,handicap] | scored | withheld (licensed source) | | | | |
| E42 | from2015:e42:gates[debut] | scored | withheld (licensed source) | | | | |
| E42 | from2015:e42:gates[handicap] | scored | withheld (licensed source) | | | | |
| E40 | from2015:encompassing:ability | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ability | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_base | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_horse_aptitude | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_horse_x_jockey | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_trainer_debut | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_trainer_x_track | scored | withheld (licensed source) | | | | |
| E40 | from2015:marginal:ix_transfer_step | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_base | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_horse_aptitude | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_horse_x_jockey | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_trainer_debut | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_trainer_x_track | scored | withheld (licensed source) | | | | |
| E40 | from2015:meld:all minus ix_transfer_step | scored | withheld (licensed source) | | | | |
| E40 | from2015:standalone:ability | scored | withheld (licensed source) | | | | |
| E40 | from2020:ambiguity | scored | withheld (licensed source) | | | | |
| E40 | from2020:correlations | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:ability:c[0.25] | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:ability:c[1] | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:ability:c[4] | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:rating:c[0.25] | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:rating:c[1] | scored | withheld (licensed source) | | | | |
| E41 | from2020:e41:rating:c[4] | scored | withheld (licensed source) | | | | |
| E42 | from2020:e42:gates[debut,handicap] | scored | withheld (licensed source) | | | | |
| E42 | from2020:e42:gates[debut] | scored | withheld (licensed source) | | | | |
| E42 | from2020:e42:gates[handicap] | scored | withheld (licensed source) | | | | |
| E40 | from2020:encompassing:ability | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ability | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_base | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_horse_aptitude | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_horse_x_jockey | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_trainer_debut | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_trainer_x_track | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:ix_transfer_step | scored | withheld (licensed source) | | | | |
| E40 | from2020:marginal:rating | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_base | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_horse_aptitude | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_horse_x_jockey | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_trainer_debut | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_trainer_x_track | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus ix_transfer_step | scored | withheld (licensed source) | | | | |
| E40 | from2020:meld:all minus rating | scored | withheld (licensed source) | | | | |
| E40 | from2020:standalone:ability | scored | withheld (licensed source) | | | | |
