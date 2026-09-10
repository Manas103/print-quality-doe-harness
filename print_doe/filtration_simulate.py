"""Glue module: run a filtration-bench design matrix through the hidden
ground-truth score generator and the OpenCV render/measure pipeline,
producing the measured defect-count response column that every filtration
analysis in this repo fits.

Mirrors `simulate.py` exactly: this is the one place that imports both
`filtration_ground_truth` and `filtration_response`. It is deliberately
thin: `simulate_filtration_responses` does not do any statistics, and
`analysis.py` never imports `filtration_ground_truth`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from print_doe import filtration_ground_truth
from print_doe.filtration_response import measured_response_for_filtration_run


def simulate_filtration_responses(design: pd.DataFrame, seed: int,
                                   response_col: str = "defect_count") -> pd.DataFrame:
    """Given a filtration design matrix (must contain the four
    `filtration_design.FACTOR_NAMES` columns), simulate a measured defect
    count for every row.

    Each run gets its own independent RNG stream, seeded from `seed` and the
    run's `design_run_id`, so replicate runs of the same factor combination
    get independent process noise and independent particle-placement/sensor
    noise, exactly as independent physical bench runs would.
    """
    out = design.copy()
    responses = np.empty(len(out), dtype=np.float64)
    true_scores = np.empty(len(out), dtype=np.float64)
    true_particle_counts = np.empty(len(out), dtype=np.float64)

    for i, row in enumerate(out.itertuples(index=False)):
        row_dict = row._asdict()
        run_id = row_dict["design_run_id"]
        rng = np.random.default_rng((seed, int(run_id)))
        score = filtration_ground_truth.true_score(row_dict, rng)
        true_scores[i] = score
        measured, n_particles = measured_response_for_filtration_run(score, rng)
        responses[i] = measured
        true_particle_counts[i] = n_particles

    out[response_col] = responses
    out["_true_score_debug_only"] = true_scores
    out["_true_particle_count_debug_only"] = true_particle_counts
    return out
