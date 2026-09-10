"""Monte Carlo power estimation for the filtration-bench extension.

Two distinct uses, mirroring `power.py`'s repeated-resimulation pattern:

1. `monte_carlo_power_filtration`: per-model-term empirical power / false-
   positive rate on the 4-factor factorial design, the direct filtration
   analogue of `power.py.monte_carlo_power`, reusing `analysis.py`'s
   response-agnostic `fit_ols`/`effect_table` machinery.
2. `power_curve_for_defect_reduction`: a new kind of analysis this repo did
   not previously have. For a two-condition comparison (baseline vs a
   treatment with a specified true fractional reduction in expected defect
   count), sweep replicate counts per condition and report empirical power
   to detect the reduction at each replicate count, via repeated
   resimulation through the real noisy render-then-count pipeline (never a
   closed-form power formula plugged in directly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from print_doe.analysis import fit_ols, effect_table
from print_doe.filtration_design import FACTOR_NAMES, full_factorial_design
from print_doe.filtration_simulate import simulate_filtration_responses
from print_doe.filtration_response import (
    expected_particle_rate, score_for_expected_rate, measured_response_for_filtration_run,
)


def monte_carlo_power_filtration(n_reps: int = 200, base_seed: int = 665000,
                                  replicates: int = None) -> pd.DataFrame:
    """Run `n_reps` independent full simulate+fit cycles on freshly generated
    filtration designs and return, per model term, the fraction of reps
    flagged significant (empirical power for the 3 seeded main effects and
    the 1 seeded interaction; empirical false-positive rate for
    `purge_volume` and every unseeded 2-way interaction)."""
    from print_doe.filtration_design import REPLICATES as _DEFAULT_REPLICATES
    if replicates is None:
        replicates = _DEFAULT_REPLICATES

    hit_counts = None
    term_order = None

    for rep in range(n_reps):
        design = full_factorial_design(replicates=replicates, seed=base_seed + rep)
        simulated = simulate_filtration_responses(design, seed=base_seed + 1_000_000 + rep)
        result, _X = fit_ols(simulated, response_col="defect_count", factor_names=FACTOR_NAMES)
        table = effect_table(result)

        if hit_counts is None:
            term_order = table["term"].tolist()
            hit_counts = {t: 0 for t in term_order}
        for t, sig in zip(table["term"], table["significant"]):
            hit_counts[t] += int(bool(sig))

    rows = [{"term": t, "n_reps": n_reps, "hits": hit_counts[t], "empirical_rate": hit_counts[t] / n_reps}
            for t in term_order]
    return pd.DataFrame(rows)


def _simulate_condition_sample(score: float, n: int, seed_tuple) -> np.ndarray:
    """Draw `n` independent measured defect counts at a fixed hidden score,
    each through its own fresh render-then-count pipeline draw (independent
    particle placement, independent sensor noise), exactly like `n`
    independent physical bench runs at the same setting."""
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = np.random.default_rng((*seed_tuple, i))
        measured, _n_particles = measured_response_for_filtration_run(score, rng)
        out[i] = measured
    return out


def power_curve_for_defect_reduction(replicate_counts, baseline_score: float,
                                      reduction_frac: float = 0.20, n_reps: int = 300,
                                      base_seed: int = 771000, alpha: float = 0.05) -> pd.DataFrame:
    """Monte Carlo power curve for detecting a true `reduction_frac`
    reduction in expected defect count, baseline vs treatment, at each
    replicate count `n` in `replicate_counts`.

    The treatment score is constructed so its expected particle count
    (`filtration_response.expected_particle_rate`) is exactly
    `(1 - reduction_frac)` times the baseline's expected particle count;
    this is a true reduction in the underlying process, not a reduction
    applied to the measured output. Both baseline and treatment samples are
    then drawn through the identical noisy render-then-count pipeline used
    everywhere else in this repo.

    For each replicate count, `n_reps` independent Monte Carlo repetitions
    each redraw a fresh baseline sample and a fresh treatment sample of that
    size and run a two-sided Welch's t-test (unequal variance, the
    appropriate test since baseline and treatment need not share the same
    noise variance); empirical power is the fraction of repetitions where
    p < alpha.
    """
    baseline_rate = expected_particle_rate(baseline_score)
    treatment_rate = baseline_rate * (1.0 - reduction_frac)
    treatment_score = score_for_expected_rate(treatment_rate)

    rows = []
    for n in replicate_counts:
        hits = 0
        baseline_means = []
        treatment_means = []
        for rep in range(n_reps):
            baseline_sample = _simulate_condition_sample(
                baseline_score, n, (base_seed, 0, n, rep))
            treatment_sample = _simulate_condition_sample(
                treatment_score, n, (base_seed, 1, n, rep))
            baseline_means.append(baseline_sample.mean())
            treatment_means.append(treatment_sample.mean())
            _t_stat, p_value = stats.ttest_ind(baseline_sample, treatment_sample, equal_var=False)
            if p_value < alpha:
                hits += 1
        rows.append({
            "replicates_per_condition": n,
            "n_reps": n_reps,
            "hits": hits,
            "empirical_power": hits / n_reps,
            "mean_baseline_defect_count": float(np.mean(baseline_means)),
            "mean_treatment_defect_count": float(np.mean(treatment_means)),
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
        })
    return pd.DataFrame(rows)
