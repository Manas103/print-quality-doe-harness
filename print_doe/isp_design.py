"""Design-matrix construction for the ISP/detection extension: the full
replicated 2^5 factorial over 5 image-signal-processing parameters, and
the one-factor-at-a-time (OFAT) comparator on the identical run budget.

Mirrors `design.py` and `filtration_design.py` exactly (same orthogonal
coding, randomized run-order, reproducible-seed pattern) but is a separate
factor space over ISP settings rather than press or filtration settings.
Nothing in this file knows about the detector or the rendering pipeline in
`isp_pipeline.py`; it only produces coded (-1 / +1) factor columns.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# The five ISP parameters this experiment varies, in the order they appear
# on every design matrix in this module. Order here is for readability
# only; `isp_pipeline.apply_isp_chain` applies them in a fixed pipeline
# order (demosaic, denoise, white balance, tone curve, sharpening)
# regardless of the order they are listed here.
FACTOR_NAMES = [
    "denoise",
    "demosaic",
    "tone_curve",
    "sharpening",
    "white_balance",
]

N_FACTORS = len(FACTOR_NAMES)          # 5
RUNS_PER_REPLICATE = 2 ** N_FACTORS    # 32 unique combinations
REPLICATES = 3
TOTAL_RUNS = REPLICATES * RUNS_PER_REPLICATE  # 96

N_OFAT_CONDITIONS = N_FACTORS + 1
OFAT_REPLICATES_PER_CONDITION = TOTAL_RUNS // N_OFAT_CONDITIONS  # 16
assert OFAT_REPLICATES_PER_CONDITION * N_OFAT_CONDITIONS == TOTAL_RUNS


def full_factorial_design(replicates: int = REPLICATES, seed: int = 20261101) -> pd.DataFrame:
    """Build the replicated full 2^5 two-level factorial design over the 5
    ISP parameters. 32 unique coded combinations x `replicates` = 96 runs.
    """
    combos = list(itertools.product([-1, 1], repeat=N_FACTORS))
    assert len(combos) == RUNS_PER_REPLICATE

    rows = []
    for rep in range(1, replicates + 1):
        for combo in combos:
            rows.append({"replicate": rep, **dict(zip(FACTOR_NAMES, combo))})

    df = pd.DataFrame(rows)
    df.insert(0, "design_run_id", np.arange(1, len(df) + 1))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "run_order", np.arange(1, len(df) + 1))
    return df


def ofat_design(replicates_per_condition: int = OFAT_REPLICATES_PER_CONDITION,
                 baseline: int = -1, seed: int = 20261115) -> pd.DataFrame:
    """One-factor-at-a-time comparator on the identical 96-run budget: the
    all-baseline condition plus one condition per factor flipped high,
    everything else held at baseline."""
    conditions = [dict.fromkeys(FACTOR_NAMES, baseline)]
    for factor in FACTOR_NAMES:
        cond = dict.fromkeys(FACTOR_NAMES, baseline)
        cond[factor] = -baseline
        conditions.append(cond)

    rows = []
    for rep in range(1, replicates_per_condition + 1):
        for cond_idx, cond in enumerate(conditions):
            rows.append({"replicate": rep, "condition": cond_idx, **cond})

    df = pd.DataFrame(rows)
    df.insert(0, "design_run_id", np.arange(1, len(df) + 1))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "run_order", np.arange(1, len(df) + 1))
    return df
