# Archive: single-objective run with the OLD alphas (v1)

Data of the first PIAST-Q execution (30 Oct - 6 Nov 2026) for the single-objective
subjects `gsdtsr` and `paintcontrol`, trained and evaluated with the OLD QUBO weights
(`gsdtsr` alpha = 0, `paintcontrol` alpha = 0.45). These alphas were found to be wrong on
9 Nov 2026 (commit edbf6a6d, Optuna values from SelectQAOA) and the two subjects were
re-trained and re-executed. Kept for traceability only; the multi-objective subjects
(`flex`, `gzip`) are unaffected and their data stays in `results/qaoa_tcs/`.

Contents: frozen circuits + QUBOs (`trained_qaoa_circuits/`), raw/TREx counts and suites
(`results/qaoa_tcs/`), the full evaluation and comparison report of the 4-subject v1 run
(`results/evaluation/`), and the 4 Nov drift check (`calibration/drift/`).
