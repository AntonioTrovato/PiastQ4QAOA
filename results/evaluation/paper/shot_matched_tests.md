# PIAST-Q technique vs. noiseless reference with the same shots

Configurations: S200, S600 (noiseless simulator) and Raw, MEM, M3, TREx (PIAST-Q), 10 repetitions each. HV, IGD and ND are computed, per repetition, against the non-dominated union of the six fronts. Best: number of clusters in which the configuration selects the lower-energy bitstring between Raw and S200 (ties count for both). Effect sizes: |d| <0.2 N, <0.5 S, <0.8 M, else L; |A12-0.5| <0.06 N, <0.14 S, <0.21 M, else L.

Shapiro-Wilk (alpha=0.05) on each group; both normal: Welch's t-test + Cohen's d; otherwise Mann-Whitney U + Vargha-Delaney A12 (two-sided).


## Flex

| Comparison | Metric | SW technique | SW reference | Test | p | Effect size |
|---|---|---|---|---|---|---|
| Raw vs S200 | Energy | 0.046 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.473 | A12=0.40 (S) |
| Raw vs S200 | HV | 0.626 (normal) | 0.060 (normal) | Welch t | 0.050 | d=-0.97 (L) |
| Raw vs S200 | IGD | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.089 | A12=0.27 (L) |
| Raw vs S200 | ND | 0.001 (non-normal) | 0.005 (non-normal) | Mann-Whitney U | 0.272 | A12=0.35 (M) |
| Raw vs S200 | Best | 0.886 (normal) | 0.119 (normal) | Welch t | 0.459 | d=-0.34 (S) |
| MEM vs S200 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.791 | A12=0.54 (N) |
| MEM vs S200 | HV | 0.577 (normal) | 0.060 (normal) | Welch t | 0.033 | d=-1.05 (L) |
| MEM vs S200 | IGD | 0.001 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.076 | A12=0.26 (L) |
| MEM vs S200 | ND | 0.009 (non-normal) | 0.005 (non-normal) | Mann-Whitney U | 0.058 | A12=0.24 (L) |
| M3 vs S200 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.910 | A12=0.48 (N) |
| M3 vs S200 | HV | 0.490 (normal) | 0.060 (normal) | Welch t | 0.029 | d=-1.10 (L) |
| M3 vs S200 | IGD | 0.001 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.076 | A12=0.26 (L) |
| M3 vs S200 | ND | 0.041 (non-normal) | 0.005 (non-normal) | Mann-Whitney U | 0.140 | A12=0.30 (M) |
| TREx vs S600 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.678 | A12=0.44 (S) |
| TREx vs S600 | HV | 0.333 (normal) | 0.023 (non-normal) | Mann-Whitney U | 0.734 | A12=0.45 (N) |
| TREx vs S600 | IGD | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.427 | A12=0.39 (S) |
| TREx vs S600 | ND | 0.000 (non-normal) | 0.043 (non-normal) | Mann-Whitney U | 0.449 | A12=0.60 (S) |

## Gzip

| Comparison | Metric | SW technique | SW reference | Test | p | Effect size |
|---|---|---|---|---|---|---|
| Raw vs S200 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.089 | A12=0.73 (L) |
| Raw vs S200 | HV | 0.016 (non-normal) | 0.002 (non-normal) | Mann-Whitney U | 0.473 | A12=0.60 (S) |
| Raw vs S200 | IGD | 0.306 (normal) | 0.260 (normal) | Welch t | 0.008 | d=-1.39 (L) |
| Raw vs S200 | ND | 0.000 (non-normal) | 0.010 (non-normal) | Mann-Whitney U | 0.384 | A12=0.62 (S) |
| Raw vs S200 | Best | 0.073 (normal) | 0.140 (normal) | Welch t | 0.600 | d=-0.24 (S) |
| MEM vs S200 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.045 | A12=0.77 (L) |
| MEM vs S200 | HV | 0.018 (non-normal) | 0.002 (non-normal) | Mann-Whitney U | 0.473 | A12=0.60 (S) |
| MEM vs S200 | IGD | 0.338 (normal) | 0.260 (normal) | Welch t | 0.008 | d=-1.37 (L) |
| MEM vs S200 | ND | 0.007 (non-normal) | 0.010 (non-normal) | Mann-Whitney U | 0.880 | A12=0.53 (N) |
| M3 vs S200 | Energy | 0.000 (non-normal) | 0.000 (non-normal) | Mann-Whitney U | 0.045 | A12=0.77 (L) |
| M3 vs S200 | HV | 0.020 (non-normal) | 0.002 (non-normal) | Mann-Whitney U | 0.473 | A12=0.60 (S) |
| M3 vs S200 | IGD | 0.279 (normal) | 0.260 (normal) | Welch t | 0.008 | d=-1.37 (L) |
| M3 vs S200 | ND | 0.027 (non-normal) | 0.010 (non-normal) | Mann-Whitney U | 0.520 | A12=0.41 (S) |
| TREx vs S600 | Energy | 0.000 (non-normal) | 0.753 (normal) | Mann-Whitney U | 0.734 | A12=0.45 (N) |
| TREx vs S600 | HV | 0.019 (non-normal) | 0.007 (non-normal) | Mann-Whitney U | 1.000 | A12=0.50 (N) |
| TREx vs S600 | IGD | 0.129 (normal) | 0.722 (normal) | Welch t | 0.165 | d=-0.66 (M) |
| TREx vs S600 | ND | 0.028 (non-normal) | 0.003 (non-normal) | Mann-Whitney U | 0.325 | A12=0.64 (M) |
