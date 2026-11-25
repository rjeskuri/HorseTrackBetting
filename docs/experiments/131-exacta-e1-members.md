# #131 exacta study: E1 results (42711715fa0c)

Every E1 (B2), E1′ (B1) and E1″ (B3) member of the 2026-11-25 run (git `42711715fa0c`), as written
by `exacta_run.write_outputs` under the data directory and copied here; aggregates only. The
report is [131-exacta-overload.md](131-exacta-overload.md). Development races 2020-01-05 to
2026-11-15 with #114 out-of-fold Win probabilities: 23,629 races, 1,980 meetings.

- estimate: ρ_S (B1, B2: total return over total payout rate, null 1) or the mean return per race
  (B3: Harville's calibration on S, null 1).
- 95% lo and 95% hi: the one-sided 95% lower and upper bounds of a meeting bootstrap (10,000
  replicates, seed 131), each a one-sided bound, not a two-sided interval. Not adjusted for
  multiplicity: the family-wise result is SPA and StepM in the report.
- lambda2: `in_fold` is the registry label frozen in the spec, kept as written; it means #115's
  full-development fit (λ2 = 0.8104, partly in-sample), one fit over the development out-of-fold
  races that saw these races' first two places, not a per-fold fit. `1.0` is plain Harville. B1
  does not use p, so its `in_fold` label carries no λ2. B3 at 0.8104 is partly in-sample; B2's
  null (ρ = 1 under π = q) does not depend on p. The deviation awaits the owner's confirmation
  (spec, "Deviations from the signed spec").
- EB: B2 only, the empirical-Bayes estimate shrunk across the 14 cells of its category and λ2.
- n: races in the cell (each race is scored in all three categories).

| estimator | lambda2 | category | cell | n | estimate | 95% lo | 95% hi | EB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2 | in_fold | fav_first | all | 23629 | 1.024 | 1.008 | 1.041 | 1.025 |
| B2 | in_fold | fav_first | fav_odds:le1.9 | 5449 | 1.022 | 0.997 | 1.047 | 1.025 |
| B2 | in_fold | fav_first | fav_odds:2.0-2.9 | 10375 | 1.040 | 1.016 | 1.065 | 1.025 |
| B2 | in_fold | fav_first | fav_odds:3.0-4.9 | 7616 | 1.008 | 0.973 | 1.043 | 1.025 |
| B2 | in_fold | fav_first | fav_odds:ge5.0 | 189 | 0.886 | 0.611 | 1.179 | 1.025 |
| B2 | in_fold | fav_first | field:5-10 | 3820 | 1.034 | 0.996 | 1.070 | 1.025 |
| B2 | in_fold | fav_first | field:11-14 | 7939 | 1.044 | 1.014 | 1.073 | 1.025 |
| B2 | in_fold | fav_first | field:15-18 | 11870 | 1.009 | 0.983 | 1.034 | 1.025 |
| B2 | in_fold | fav_first | class:maiden | 10559 | 1.024 | 1.000 | 1.048 | 1.025 |
| B2 | in_fold | fav_first | class:win1-3 | 10978 | 1.042 | 1.017 | 1.067 | 1.025 |
| B2 | in_fold | fav_first | class:open | 2092 | 0.935 | 0.875 | 0.997 | 1.025 |
| B2 | in_fold | fav_first | popularity:less | 11070 | 1.026 | 1.002 | 1.049 | 1.025 |
| B2 | in_fold | fav_first | popularity:rest | 12559 | 1.023 | 0.999 | 1.047 | 1.025 |
| B2 | in_fold | fav_first | headline | 2923 | 1.037 | 0.980 | 1.096 | 1.025 |
| B2 | in_fold | fav_second | all | 23629 | 1.003 | 0.980 | 1.026 | 1.004 |
| B2 | in_fold | fav_second | fav_odds:le1.9 | 5449 | 0.995 | 0.952 | 1.039 | 1.003 |
| B2 | in_fold | fav_second | fav_odds:2.0-2.9 | 10375 | 1.000 | 0.966 | 1.034 | 1.003 |
| B2 | in_fold | fav_second | fav_odds:3.0-4.9 | 7616 | 1.009 | 0.967 | 1.053 | 1.004 |
| B2 | in_fold | fav_second | fav_odds:ge5.0 | 189 | 1.199 | 0.840 | 1.599 | 1.004 |
| B2 | in_fold | fav_second | field:5-10 | 3820 | 0.961 | 0.910 | 1.014 | 1.002 |
| B2 | in_fold | fav_second | field:11-14 | 7939 | 0.991 | 0.952 | 1.031 | 1.003 |
| B2 | in_fold | fav_second | field:15-18 | 11870 | 1.024 | 0.991 | 1.058 | 1.006 |
| B2 | in_fold | fav_second | class:maiden | 10559 | 1.036 | 1.002 | 1.071 | 1.007 |
| B2 | in_fold | fav_second | class:win1-3 | 10978 | 0.972 | 0.938 | 1.007 | 1.001 |
| B2 | in_fold | fav_second | class:open | 2092 | 1.001 | 0.919 | 1.084 | 1.004 |
| B2 | in_fold | fav_second | popularity:less | 11070 | 1.033 | 1.000 | 1.067 | 1.007 |
| B2 | in_fold | fav_second | popularity:rest | 12559 | 0.977 | 0.945 | 1.009 | 1.001 |
| B2 | in_fold | fav_second | headline | 2923 | 1.055 | 0.983 | 1.127 | 1.005 |
| B2 | in_fold | no_fav | all | 23629 | 0.995 | 0.983 | 1.008 | 0.996 |
| B2 | in_fold | no_fav | fav_odds:le1.9 | 5449 | 0.986 | 0.948 | 1.024 | 0.996 |
| B2 | in_fold | no_fav | fav_odds:2.0-2.9 | 10375 | 0.993 | 0.975 | 1.011 | 0.996 |
| B2 | in_fold | no_fav | fav_odds:3.0-4.9 | 7616 | 1.004 | 0.988 | 1.020 | 0.997 |
| B2 | in_fold | no_fav | fav_odds:ge5.0 | 189 | 1.035 | 0.952 | 1.116 | 0.996 |
| B2 | in_fold | no_fav | field:5-10 | 3820 | 0.963 | 0.927 | 1.001 | 0.995 |
| B2 | in_fold | no_fav | field:11-14 | 7939 | 1.011 | 0.988 | 1.034 | 0.997 |
| B2 | in_fold | no_fav | field:15-18 | 11870 | 0.995 | 0.978 | 1.013 | 0.996 |
| B2 | in_fold | no_fav | class:maiden | 10559 | 0.990 | 0.970 | 1.011 | 0.996 |
| B2 | in_fold | no_fav | class:win1-3 | 10978 | 0.987 | 0.970 | 1.005 | 0.995 |
| B2 | in_fold | no_fav | class:open | 2092 | 1.062 | 1.022 | 1.104 | 0.997 |
| B2 | in_fold | no_fav | popularity:less | 11070 | 0.992 | 0.973 | 1.012 | 0.996 |
| B2 | in_fold | no_fav | popularity:rest | 12559 | 0.998 | 0.982 | 1.015 | 0.996 |
| B2 | in_fold | no_fav | headline | 2923 | 0.980 | 0.953 | 1.007 | 0.995 |
| B2 | 1.0 | fav_first | all | 23629 | 1.028 | 1.011 | 1.044 | 1.029 |
| B2 | 1.0 | fav_first | fav_odds:le1.9 | 5449 | 1.028 | 1.003 | 1.053 | 1.029 |
| B2 | 1.0 | fav_first | fav_odds:2.0-2.9 | 10375 | 1.044 | 1.019 | 1.068 | 1.030 |
| B2 | 1.0 | fav_first | fav_odds:3.0-4.9 | 7616 | 1.010 | 0.974 | 1.045 | 1.028 |
| B2 | 1.0 | fav_first | fav_odds:ge5.0 | 189 | 0.844 | 0.582 | 1.122 | 1.029 |
| B2 | 1.0 | fav_first | field:5-10 | 3820 | 1.037 | 0.999 | 1.074 | 1.029 |
| B2 | 1.0 | fav_first | field:11-14 | 7939 | 1.046 | 1.016 | 1.076 | 1.030 |
| B2 | 1.0 | fav_first | field:15-18 | 11870 | 1.012 | 0.986 | 1.037 | 1.028 |
| B2 | 1.0 | fav_first | class:maiden | 10559 | 1.031 | 1.007 | 1.055 | 1.029 |
| B2 | 1.0 | fav_first | class:win1-3 | 10978 | 1.043 | 1.017 | 1.068 | 1.030 |
| B2 | 1.0 | fav_first | class:open | 2092 | 0.930 | 0.870 | 0.991 | 1.028 |
| B2 | 1.0 | fav_first | popularity:less | 11070 | 1.033 | 1.009 | 1.057 | 1.029 |
| B2 | 1.0 | fav_first | popularity:rest | 12559 | 1.022 | 0.998 | 1.047 | 1.028 |
| B2 | 1.0 | fav_first | headline | 2923 | 1.047 | 0.990 | 1.107 | 1.029 |
| B2 | 1.0 | fav_second | all | 23629 | 1.004 | 0.981 | 1.027 | 1.004 |
| B2 | 1.0 | fav_second | fav_odds:le1.9 | 5449 | 0.995 | 0.953 | 1.040 | 1.004 |
| B2 | 1.0 | fav_second | fav_odds:2.0-2.9 | 10375 | 1.001 | 0.967 | 1.036 | 1.004 |
| B2 | 1.0 | fav_second | fav_odds:3.0-4.9 | 7616 | 1.010 | 0.967 | 1.053 | 1.005 |
| B2 | 1.0 | fav_second | fav_odds:ge5.0 | 189 | 1.195 | 0.837 | 1.593 | 1.005 |
| B2 | 1.0 | fav_second | field:5-10 | 3820 | 0.962 | 0.911 | 1.014 | 1.002 |
| B2 | 1.0 | fav_second | field:11-14 | 7939 | 0.992 | 0.953 | 1.031 | 1.003 |
| B2 | 1.0 | fav_second | field:15-18 | 11870 | 1.025 | 0.992 | 1.059 | 1.007 |
| B2 | 1.0 | fav_second | class:maiden | 10559 | 1.036 | 1.003 | 1.071 | 1.008 |
| B2 | 1.0 | fav_second | class:win1-3 | 10978 | 0.973 | 0.938 | 1.008 | 1.001 |
| B2 | 1.0 | fav_second | class:open | 2092 | 1.002 | 0.921 | 1.085 | 1.004 |
| B2 | 1.0 | fav_second | popularity:less | 11070 | 1.034 | 1.001 | 1.067 | 1.008 |
| B2 | 1.0 | fav_second | popularity:rest | 12559 | 0.977 | 0.945 | 1.010 | 1.001 |
| B2 | 1.0 | fav_second | headline | 2923 | 1.056 | 0.984 | 1.128 | 1.006 |
| B2 | 1.0 | no_fav | all | 23629 | 1.004 | 0.991 | 1.017 | 1.004 |
| B2 | 1.0 | no_fav | fav_odds:le1.9 | 5449 | 0.998 | 0.959 | 1.038 | 1.004 |
| B2 | 1.0 | no_fav | fav_odds:2.0-2.9 | 10375 | 1.001 | 0.983 | 1.020 | 1.004 |
| B2 | 1.0 | no_fav | fav_odds:3.0-4.9 | 7616 | 1.010 | 0.993 | 1.026 | 1.004 |
| B2 | 1.0 | no_fav | fav_odds:ge5.0 | 189 | 1.022 | 0.938 | 1.103 | 1.004 |
| B2 | 1.0 | no_fav | field:5-10 | 3820 | 0.972 | 0.935 | 1.010 | 1.004 |
| B2 | 1.0 | no_fav | field:11-14 | 7939 | 1.019 | 0.995 | 1.043 | 1.004 |
| B2 | 1.0 | no_fav | field:15-18 | 11870 | 1.004 | 0.986 | 1.021 | 1.004 |
| B2 | 1.0 | no_fav | class:maiden | 10559 | 1.000 | 0.979 | 1.022 | 1.004 |
| B2 | 1.0 | no_fav | class:win1-3 | 10978 | 0.996 | 0.978 | 1.014 | 1.004 |
| B2 | 1.0 | no_fav | class:open | 2092 | 1.063 | 1.021 | 1.106 | 1.004 |
| B2 | 1.0 | no_fav | popularity:less | 11070 | 1.002 | 0.982 | 1.023 | 1.004 |
| B2 | 1.0 | no_fav | popularity:rest | 12559 | 1.005 | 0.988 | 1.022 | 1.004 |
| B2 | 1.0 | no_fav | headline | 2923 | 0.986 | 0.958 | 1.013 | 1.004 |
| B1 | in_fold | fav_first | all | 23629 | 0.958 | 0.928 | 0.988 |  |
| B1 | in_fold | fav_first | fav_odds:le1.9 | 5449 | 0.921 | 0.875 | 0.967 |  |
| B1 | in_fold | fav_first | fav_odds:2.0-2.9 | 10375 | 0.966 | 0.922 | 1.011 |  |
| B1 | in_fold | fav_first | fav_odds:3.0-4.9 | 7616 | 0.972 | 0.912 | 1.035 |  |
| B1 | in_fold | fav_first | fav_odds:ge5.0 | 189 | 1.000 | 0.621 | 1.411 |  |
| B1 | in_fold | fav_first | field:5-10 | 3820 | 0.989 | 0.926 | 1.053 |  |
| B1 | in_fold | fav_first | field:11-14 | 7939 | 0.998 | 0.948 | 1.050 |  |
| B1 | in_fold | fav_first | field:15-18 | 11870 | 0.921 | 0.876 | 0.967 |  |
| B1 | in_fold | fav_first | class:maiden | 10559 | 0.914 | 0.870 | 0.958 |  |
| B1 | in_fold | fav_first | class:win1-3 | 10978 | 1.003 | 0.959 | 1.047 |  |
| B1 | in_fold | fav_first | class:open | 2092 | 0.942 | 0.835 | 1.059 |  |
| B1 | in_fold | fav_first | popularity:less | 11070 | 0.915 | 0.873 | 0.958 |  |
| B1 | in_fold | fav_first | popularity:rest | 12559 | 0.995 | 0.953 | 1.037 |  |
| B1 | in_fold | fav_first | headline | 2923 | 0.916 | 0.823 | 1.018 |  |
| B1 | in_fold | fav_second | all | 23629 | 0.855 | 0.813 | 0.899 |  |
| B1 | in_fold | fav_second | fav_odds:le1.9 | 5449 | 0.788 | 0.719 | 0.862 |  |
| B1 | in_fold | fav_second | fav_odds:2.0-2.9 | 10375 | 0.857 | 0.790 | 0.928 |  |
| B1 | in_fold | fav_second | fav_odds:3.0-4.9 | 7616 | 0.866 | 0.797 | 0.940 |  |
| B1 | in_fold | fav_second | fav_odds:ge5.0 | 189 | 2.229 | 0.936 | 4.202 |  |
| B1 | in_fold | fav_second | field:5-10 | 3820 | 0.880 | 0.782 | 0.983 |  |
| B1 | in_fold | fav_second | field:11-14 | 7939 | 0.824 | 0.762 | 0.890 |  |
| B1 | in_fold | fav_second | field:15-18 | 11870 | 0.868 | 0.804 | 0.937 |  |
| B1 | in_fold | fav_second | class:maiden | 10559 | 0.839 | 0.776 | 0.906 |  |
| B1 | in_fold | fav_second | class:win1-3 | 10978 | 0.872 | 0.807 | 0.940 |  |
| B1 | in_fold | fav_second | class:open | 2092 | 0.847 | 0.716 | 0.983 |  |
| B1 | in_fold | fav_second | popularity:less | 11070 | 0.839 | 0.778 | 0.902 |  |
| B1 | in_fold | fav_second | popularity:rest | 12559 | 0.869 | 0.808 | 0.933 |  |
| B1 | in_fold | fav_second | headline | 2923 | 0.936 | 0.802 | 1.077 |  |
| B1 | in_fold | no_fav | all | 23629 | 0.777 | 0.752 | 0.803 |  |
| B1 | in_fold | no_fav | fav_odds:le1.9 | 5449 | 0.704 | 0.640 | 0.772 |  |
| B1 | in_fold | no_fav | fav_odds:2.0-2.9 | 10375 | 0.768 | 0.732 | 0.806 |  |
| B1 | in_fold | no_fav | fav_odds:3.0-4.9 | 7616 | 0.832 | 0.793 | 0.875 |  |
| B1 | in_fold | no_fav | fav_odds:ge5.0 | 189 | 1.117 | 0.888 | 1.371 |  |
| B1 | in_fold | no_fav | field:5-10 | 3820 | 0.742 | 0.684 | 0.801 |  |
| B1 | in_fold | no_fav | field:11-14 | 7939 | 0.786 | 0.743 | 0.834 |  |
| B1 | in_fold | no_fav | field:15-18 | 11870 | 0.782 | 0.744 | 0.822 |  |
| B1 | in_fold | no_fav | class:maiden | 10559 | 0.714 | 0.674 | 0.756 |  |
| B1 | in_fold | no_fav | class:win1-3 | 10978 | 0.817 | 0.781 | 0.855 |  |
| B1 | in_fold | no_fav | class:open | 2092 | 0.889 | 0.816 | 0.964 |  |
| B1 | in_fold | no_fav | popularity:less | 11070 | 0.725 | 0.686 | 0.767 |  |
| B1 | in_fold | no_fav | popularity:rest | 12559 | 0.823 | 0.790 | 0.858 |  |
| B1 | in_fold | no_fav | headline | 2923 | 0.785 | 0.710 | 0.867 |  |
| B3 | in_fold | fav_first | all | 23629 | 0.931 | 0.902 | 0.962 |  |
| B3 | in_fold | fav_first | fav_odds:le1.9 | 5449 | 0.890 | 0.848 | 0.934 |  |
| B3 | in_fold | fav_first | fav_odds:2.0-2.9 | 10375 | 0.938 | 0.895 | 0.984 |  |
| B3 | in_fold | fav_first | fav_odds:3.0-4.9 | 7616 | 0.952 | 0.890 | 1.019 |  |
| B3 | in_fold | fav_first | fav_odds:ge5.0 | 189 | 0.917 | 0.577 | 1.286 |  |
| B3 | in_fold | fav_first | field:5-10 | 3820 | 0.962 | 0.899 | 1.027 |  |
| B3 | in_fold | fav_first | field:11-14 | 7939 | 0.975 | 0.924 | 1.029 |  |
| B3 | in_fold | fav_first | field:15-18 | 11870 | 0.893 | 0.848 | 0.939 |  |
| B3 | in_fold | fav_first | class:maiden | 10559 | 0.884 | 0.839 | 0.930 |  |
| B3 | in_fold | fav_first | class:win1-3 | 10978 | 0.990 | 0.946 | 1.036 |  |
| B3 | in_fold | fav_first | class:open | 2092 | 0.862 | 0.776 | 0.955 |  |
| B3 | in_fold | fav_first | popularity:less | 11070 | 0.884 | 0.841 | 0.928 |  |
| B3 | in_fold | fav_first | popularity:rest | 12559 | 0.973 | 0.932 | 1.015 |  |
| B3 | in_fold | fav_first | headline | 2923 | 0.895 | 0.796 | 1.003 |  |
| B3 | in_fold | fav_second | all | 23629 | 0.864 | 0.808 | 0.924 |  |
| B3 | in_fold | fav_second | fav_odds:le1.9 | 5449 | 0.777 | 0.696 | 0.866 |  |
| B3 | in_fold | fav_second | fav_odds:2.0-2.9 | 10375 | 0.898 | 0.801 | 1.006 |  |
| B3 | in_fold | fav_second | fav_odds:3.0-4.9 | 7616 | 0.856 | 0.770 | 0.951 |  |
| B3 | in_fold | fav_second | fav_odds:ge5.0 | 189 | 1.793 | 0.818 | 3.185 |  |
| B3 | in_fold | fav_second | field:5-10 | 3820 | 0.840 | 0.738 | 0.949 |  |
| B3 | in_fold | fav_second | field:11-14 | 7939 | 0.809 | 0.732 | 0.896 |  |
| B3 | in_fold | fav_second | field:15-18 | 11870 | 0.908 | 0.819 | 1.007 |  |
| B3 | in_fold | fav_second | class:maiden | 10559 | 0.896 | 0.800 | 0.999 |  |
| B3 | in_fold | fav_second | class:win1-3 | 10978 | 0.843 | 0.770 | 0.922 |  |
| B3 | in_fold | fav_second | class:open | 2092 | 0.812 | 0.665 | 0.977 |  |
| B3 | in_fold | fav_second | popularity:less | 11070 | 0.892 | 0.800 | 0.990 |  |
| B3 | in_fold | fav_second | popularity:rest | 12559 | 0.839 | 0.770 | 0.913 |  |
| B3 | in_fold | fav_second | headline | 2923 | 1.007 | 0.811 | 1.229 |  |
| B3 | in_fold | no_fav | all | 23629 | 1.053 | 0.974 | 1.148 |  |
| B3 | in_fold | no_fav | fav_odds:le1.9 | 5449 | 0.976 | 0.823 | 1.148 |  |
| B3 | in_fold | no_fav | fav_odds:2.0-2.9 | 10375 | 1.050 | 0.970 | 1.136 |  |
| B3 | in_fold | no_fav | fav_odds:3.0-4.9 | 7616 | 1.110 | 0.941 | 1.362 |  |
| B3 | in_fold | no_fav | fav_odds:ge5.0 | 189 | 1.200 | 0.924 | 1.521 |  |
| B3 | in_fold | no_fav | field:5-10 | 3820 | 0.903 | 0.811 | 1.005 |  |
| B3 | in_fold | no_fav | field:11-14 | 7939 | 0.982 | 0.906 | 1.067 |  |
| B3 | in_fold | no_fav | field:15-18 | 11870 | 1.149 | 1.004 | 1.326 |  |
| B3 | in_fold | no_fav | class:maiden | 10559 | 1.062 | 0.903 | 1.258 |  |
| B3 | in_fold | no_fav | class:win1-3 | 10978 | 1.046 | 0.978 | 1.119 |  |
| B3 | in_fold | no_fav | class:open | 2092 | 1.049 | 0.939 | 1.163 |  |
| B3 | in_fold | no_fav | popularity:less | 11070 | 1.078 | 0.924 | 1.265 |  |
| B3 | in_fold | no_fav | popularity:rest | 12559 | 1.031 | 0.971 | 1.097 |  |
| B3 | in_fold | no_fav | headline | 2923 | 1.257 | 0.854 | 1.906 |  |
| B3 | 1.0 | fav_first | all | 23629 | 1.191 | 1.136 | 1.247 |  |
| B3 | 1.0 | fav_first | fav_odds:le1.9 | 5449 | 1.119 | 1.044 | 1.196 |  |
| B3 | 1.0 | fav_first | fav_odds:2.0-2.9 | 10375 | 1.214 | 1.133 | 1.299 |  |
| B3 | 1.0 | fav_first | fav_odds:3.0-4.9 | 7616 | 1.214 | 1.104 | 1.335 |  |
| B3 | 1.0 | fav_first | fav_odds:ge5.0 | 189 | 1.061 | 0.634 | 1.540 |  |
| B3 | 1.0 | fav_first | field:5-10 | 3820 | 1.175 | 1.066 | 1.291 |  |
| B3 | 1.0 | fav_first | field:11-14 | 7939 | 1.240 | 1.153 | 1.336 |  |
| B3 | 1.0 | fav_first | field:15-18 | 11870 | 1.163 | 1.081 | 1.249 |  |
| B3 | 1.0 | fav_first | class:maiden | 10559 | 1.163 | 1.078 | 1.252 |  |
| B3 | 1.0 | fav_first | class:win1-3 | 10978 | 1.243 | 1.166 | 1.323 |  |
| B3 | 1.0 | fav_first | class:open | 2092 | 1.056 | 0.919 | 1.208 |  |
| B3 | 1.0 | fav_first | popularity:less | 11070 | 1.158 | 1.077 | 1.243 |  |
| B3 | 1.0 | fav_first | popularity:rest | 12559 | 1.219 | 1.148 | 1.293 |  |
| B3 | 1.0 | fav_first | headline | 2923 | 1.159 | 0.974 | 1.368 |  |
| B3 | 1.0 | fav_second | all | 23629 | 0.736 | 0.688 | 0.786 |  |
| B3 | 1.0 | fav_second | fav_odds:le1.9 | 5449 | 0.645 | 0.578 | 0.718 |  |
| B3 | 1.0 | fav_second | fav_odds:2.0-2.9 | 10375 | 0.761 | 0.679 | 0.853 |  |
| B3 | 1.0 | fav_second | fav_odds:3.0-4.9 | 7616 | 0.744 | 0.670 | 0.826 |  |
| B3 | 1.0 | fav_second | fav_odds:ge5.0 | 189 | 1.602 | 0.731 | 2.840 |  |
| B3 | 1.0 | fav_second | field:5-10 | 3820 | 0.735 | 0.647 | 0.830 |  |
| B3 | 1.0 | fav_second | field:11-14 | 7939 | 0.691 | 0.625 | 0.765 |  |
| B3 | 1.0 | fav_second | field:15-18 | 11870 | 0.766 | 0.690 | 0.850 |  |
| B3 | 1.0 | fav_second | class:maiden | 10559 | 0.762 | 0.681 | 0.849 |  |
| B3 | 1.0 | fav_second | class:win1-3 | 10978 | 0.719 | 0.657 | 0.787 |  |
| B3 | 1.0 | fav_second | class:open | 2092 | 0.687 | 0.563 | 0.826 |  |
| B3 | 1.0 | fav_second | popularity:less | 11070 | 0.759 | 0.681 | 0.842 |  |
| B3 | 1.0 | fav_second | popularity:rest | 12559 | 0.715 | 0.656 | 0.778 |  |
| B3 | 1.0 | fav_second | headline | 2923 | 0.876 | 0.708 | 1.069 |  |
| B3 | 1.0 | no_fav | all | 23629 | 1.501 | 1.351 | 1.686 |  |
| B3 | 1.0 | no_fav | fav_odds:le1.9 | 5449 | 1.516 | 1.232 | 1.831 |  |
| B3 | 1.0 | no_fav | fav_odds:2.0-2.9 | 10375 | 1.476 | 1.334 | 1.634 |  |
| B3 | 1.0 | no_fav | fav_odds:3.0-4.9 | 7616 | 1.527 | 1.202 | 2.034 |  |
| B3 | 1.0 | no_fav | fav_odds:ge5.0 | 189 | 1.399 | 1.064 | 1.779 |  |
| B3 | 1.0 | no_fav | field:5-10 | 3820 | 1.186 | 1.051 | 1.330 |  |
| B3 | 1.0 | no_fav | field:11-14 | 7939 | 1.362 | 1.227 | 1.514 |  |
| B3 | 1.0 | no_fav | field:15-18 | 11870 | 1.696 | 1.413 | 2.047 |  |
| B3 | 1.0 | no_fav | class:maiden | 10559 | 1.606 | 1.296 | 1.993 |  |
| B3 | 1.0 | no_fav | class:win1-3 | 10978 | 1.427 | 1.310 | 1.554 |  |
| B3 | 1.0 | no_fav | class:open | 2092 | 1.357 | 1.199 | 1.519 |  |
| B3 | 1.0 | no_fav | popularity:less | 11070 | 1.634 | 1.336 | 2.009 |  |
| B3 | 1.0 | no_fav | popularity:rest | 12559 | 1.384 | 1.283 | 1.493 |  |
| B3 | 1.0 | no_fav | headline | 2923 | 1.886 | 1.098 | 3.190 |  |
