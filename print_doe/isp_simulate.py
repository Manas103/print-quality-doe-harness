"""Glue: runs a design matrix through the ISP chain and the fixed
detector, and returns the measured accuracy column. Never does statistics
itself; that is analysis.py's job, called from the scripts.
"""
from __future__ import annotations

import numpy as np

from print_doe.isp_design import FACTOR_NAMES
from print_doe.isp_detector import DetectorCNN, accuracy, train_detector
from print_doe.isp_pipeline import apply_isp_chain
from print_doe.isp_scene import generate_eval_set

EVAL_SET_SIZE = 150
EVAL_SEED_BASE = 30260101
TRAIN_SEED = 30250101
TRAIN_SET_SIZE = 2000


def train_fixed_detector() -> DetectorCNN:
    """Trained once on a baseline (untouched, all factors low) eval set."""
    frames, labels = generate_eval_set(TRAIN_SET_SIZE, seed=TRAIN_SEED)
    return train_detector(frames, labels, epochs=150, seed=0)


def run_design(design, detector: DetectorCNN):
    """design: a pandas DataFrame from isp_design.full_factorial_design or
    ofat_design, with a `design_run_id` column and the 5 factor columns.
    Returns the same DataFrame with an added `accuracy` column.
    """
    accuracies = []
    for _, row in design.iterrows():
        run_id = int(row["design_run_id"])
        frames, labels = generate_eval_set(EVAL_SET_SIZE, seed=EVAL_SEED_BASE + run_id)
        settings = {name: int(row[name]) for name in FACTOR_NAMES}
        processed = np.stack([apply_isp_chain(f, settings) for f in frames])
        accuracies.append(accuracy(detector, processed, labels))
    out = design.copy()
    out["accuracy"] = accuracies
    return out
