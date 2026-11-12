# Raw vs MEM vs M3 vs TREx on PIAST-Q

Configurations: S200, S600 (noiseless simulator) and Raw, MEM, M3, TREx (PIAST-Q), 10 repetitions each. HV, IGD and ND are computed, per repetition, against the non-dominated union of the six fronts. Best: number of clusters in which the configuration selects the lower-energy bitstring between Raw and S200 (ties count for both). Effect sizes: |d| <0.2 N, <0.5 S, <0.8 M, else L; |A12-0.5| <0.06 N, <0.14 S, <0.21 M, else L.

Shapiro-Wilk (alpha=0.05) on each group; all normal: one-way ANOVA + Tukey HSD + Cohen's d; otherwise Kruskal-Wallis + Dunn's test (Bonferroni) + Vargha-Delaney A12.


## Flex


### Energy

Shapiro-Wilk p: Raw 0.046 (non-normal), MEM 0.000 (non-normal), M3 0.000 (non-normal), TREx 0.000 (non-normal).  
Kruskal-Wallis: p = 0.148; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.37 (S) |
| Raw vs M3 | 1.000 | A12=0.41 (S) |
| Raw vs TREx | 1.000 | A12=0.69 (M) |
| MEM vs M3 | 1.000 | A12=0.55 (N) |
| MEM vs TREx | 0.197 | A12=0.76 (L) |
| M3 vs TREx | 0.397 | A12=0.73 (L) |

### HV

Shapiro-Wilk p: Raw 0.626 (normal), MEM 0.577 (normal), M3 0.490 (normal), TREx 0.333 (normal).  
ANOVA: p = 0.950; post-hoc: Tukey HSD.

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 0.967 | d=0.20 (S) |
| Raw vs M3 | 0.959 | d=0.24 (S) |
| Raw vs TREx | 0.963 | d=0.23 (S) |
| MEM vs M3 | 1.000 | d=0.02 (N) |
| MEM vs TREx | 1.000 | d=0.01 (N) |
| M3 vs TREx | 1.000 | d=-0.01 (N) |

### IGD

Shapiro-Wilk p: Raw 0.000 (non-normal), MEM 0.001 (non-normal), M3 0.001 (non-normal), TREx 0.000 (non-normal).  
Kruskal-Wallis: p = 0.595; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.50 (N) |
| Raw vs M3 | 1.000 | A12=0.49 (N) |
| Raw vs TREx | 1.000 | A12=0.34 (M) |
| MEM vs M3 | 1.000 | A12=0.49 (N) |
| MEM vs TREx | 1.000 | A12=0.36 (M) |
| M3 vs TREx | 1.000 | A12=0.36 (M) |

### ND

Shapiro-Wilk p: Raw 0.001 (non-normal), MEM 0.009 (non-normal), M3 0.041 (non-normal), TREx 0.000 (non-normal).  
Kruskal-Wallis: p = 0.781; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.56 (S) |
| Raw vs M3 | 1.000 | A12=0.56 (S) |
| Raw vs TREx | 1.000 | A12=0.45 (N) |
| MEM vs M3 | 1.000 | A12=0.47 (N) |
| MEM vs TREx | 1.000 | A12=0.38 (S) |
| M3 vs TREx | 1.000 | A12=0.39 (S) |

## Gzip


### Energy

Shapiro-Wilk p: Raw 0.000 (non-normal), MEM 0.000 (non-normal), M3 0.000 (non-normal), TREx 0.000 (non-normal).  
Kruskal-Wallis: p = 0.023; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.45 (N) |
| Raw vs M3 | 1.000 | A12=0.45 (N) |
| Raw vs TREx | 0.130 | A12=0.82 (L) |
| MEM vs M3 | 1.000 | A12=0.52 (N) |
| MEM vs TREx | 0.051 | A12=0.83 (L) |
| M3 vs TREx | 0.060 | A12=0.83 (L) |

### HV

Shapiro-Wilk p: Raw 0.016 (non-normal), MEM 0.018 (non-normal), M3 0.020 (non-normal), TREx 0.019 (non-normal).  
Kruskal-Wallis: p = 0.618; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.46 (N) |
| Raw vs M3 | 1.000 | A12=0.49 (N) |
| Raw vs TREx | 1.000 | A12=0.64 (M) |
| MEM vs M3 | 1.000 | A12=0.54 (N) |
| MEM vs TREx | 1.000 | A12=0.64 (M) |
| M3 vs TREx | 1.000 | A12=0.64 (M) |

### IGD

Shapiro-Wilk p: Raw 0.306 (normal), MEM 0.338 (normal), M3 0.279 (normal), TREx 0.129 (normal).  
ANOVA: p = 0.003; post-hoc: Tukey HSD.

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | d=-0.03 (N) |
| Raw vs M3 | 1.000 | d=-0.03 (N) |
| Raw vs TREx | 0.010 | d=-1.23 (L) |
| MEM vs M3 | 1.000 | d=-0.01 (N) |
| MEM vs TREx | 0.011 | d=-1.21 (L) |
| M3 vs TREx | 0.012 | d=-1.21 (L) |

### ND

Shapiro-Wilk p: Raw 0.000 (non-normal), MEM 0.007 (non-normal), M3 0.027 (non-normal), TREx 0.028 (non-normal).  
Kruskal-Wallis: p = 0.273; post-hoc: Dunn (Bonferroni).

| Pair | p (adjusted) | Effect size |
|---|---|---|
| Raw vs MEM | 1.000 | A12=0.64 (M) |
| Raw vs M3 | 0.305 | A12=0.74 (L) |
| Raw vs TREx | 1.000 | A12=0.60 (S) |
| MEM vs M3 | 1.000 | A12=0.63 (S) |
| MEM vs TREx | 1.000 | A12=0.47 (N) |
| M3 vs TREx | 1.000 | A12=0.34 (M) |
