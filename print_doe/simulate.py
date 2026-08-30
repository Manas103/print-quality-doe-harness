"""Glue module: run a design matrix through the hidden ground-truth score
generator and the OpenCV render/measure pipeline, producing the measured
response column that every analysis in this repo fits.

This is the one place that imports both `ground_truth` and `render`. It is
deliberately thin: `simulate_responses` does not do any statistics, and
`analysis.py` never imports `ground_truth`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from print_doe import ground_truth
from print_doe.render import measured_response_for_run


def simulate_responses(design: pd.DataFrame, seed: int, response_col: str = "banding_score") -> pd.DataFrame:
    """Given a design matrix (must contain the five FACTOR_NAMES columns; may
    also contain null_* columns, which are ignored here exactly as they are
    ignored inside `ground_truth.true_score`), simulate a measured response
    for every row.

    Each run gets its own independent RNG stream, seeded from `seed` and the
    run's `design_run_id`, so replicate runs of the same factor combination
    get independent process noise and independent halftone/sensor noise,
    exactly as independent physical press runs would.
    """
    out = design.copy()
    responses = np.empty(len(out), dtype=np.float64)
    true_scores = np.empty(len(out), dtype=np.float64)

    for i, row in enumerate(out.itertuples(index=False)):
        row_dict = row._asdict()
        run_id = row_dict["design_run_id"]
        rng = np.random.default_rng((seed, int(run_id)))
        score = ground_truth.true_score(row_dict, rng)
        true_scores[i] = score
        responses[i] = measured_response_for_run(score, rng)

    out[response_col] = responses
    out["_true_score_debug_only"] = true_scores
    return out
