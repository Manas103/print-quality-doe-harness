"""Monte Carlo power estimation.

statsmodels' analytic power tools (TTestPower, FTestPower) assume a known
noise model and a single, clean effect size; this project's actual noise
comes from three stacked sources (process noise in the hidden score,
halftone dot-placement randomness, and simulated sensor noise on the
rendered image), which is exactly the case the design brief calls out as
legitimate to handle with your own Monte Carlo estimate instead. This
module repeatedly re-simulates the whole pipeline (fresh design, fresh
ground-truth noise, fresh image render, fresh OLS fit) and reports, for
every model term, the empirical fraction of repetitions in which that term
came out significant under `analysis.SIGNIFICANCE_CRITERION`.

For a term with a true nonzero effect this fraction estimates statistical
power. For a term with a true zero effect (purge_interval, and every null_*
column) this fraction estimates the empirical false-positive rate, and is
reported as such, not relabeled as power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from print_doe.analysis import fit_ols, effect_table
from print_doe.design import full_factorial_design, FACTOR_NAMES
from print_doe.simulate import simulate_responses


def monte_carlo_power(n_reps: int = 200, base_seed: int = 555000, replicates: int = 3) -> pd.DataFrame:
    """Run `n_reps` independent full simulate+fit cycles on freshly generated
    96-run designs and return, per model term, the fraction of reps flagged
    significant (empirical power for real effects, empirical false-positive
    rate for the true-null purge_interval term)."""
    hit_counts = None
    term_order = None

    for rep in range(n_reps):
        design = full_factorial_design(replicates=replicates, seed=base_seed + rep)
        simulated = simulate_responses(design, seed=base_seed + 1_000_000 + rep)
        result, _X = fit_ols(simulated, response_col="banding_score")
        table = effect_table(result)

        if hit_counts is None:
            term_order = table["term"].tolist()
            hit_counts = {t: 0 for t in term_order}
        for t, sig in zip(table["term"], table["significant"]):
            hit_counts[t] += int(bool(sig))

    rows = [{"term": t, "n_reps": n_reps, "hits": hit_counts[t], "empirical_rate": hit_counts[t] / n_reps}
            for t in term_order]
    return pd.DataFrame(rows)
