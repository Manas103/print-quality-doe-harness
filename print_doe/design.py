"""Design-matrix construction: the full replicated 2^5 factorial, the null-factor
columns, and the one-factor-at-a-time (OFAT) comparator design.

Nothing in this file knows about the response-generating ground truth in
`ground_truth.py`. It only produces coded (-1 / +1) factor columns.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# The five controllable press settings, in the order they appear on every
# design matrix in this repo. Order matters only for readability; the design
# is orthogonal regardless of column order.
FACTOR_NAMES = [
    "ink_pressure",
    "dwell_time",
    "substrate_speed",
    "nozzle_drive_voltage",
    "purge_interval",
]

N_FACTORS = len(FACTOR_NAMES)          # 5
N_NULL_FACTORS = 20
REPLICATES = 3
RUNS_PER_REPLICATE = 2 ** N_FACTORS    # 32
TOTAL_RUNS = REPLICATES * RUNS_PER_REPLICATE  # 96


def full_factorial_design(replicates: int = REPLICATES, seed: int = 20260501) -> pd.DataFrame:
    """Build the replicated full 2^5 two-level factorial design.

    32 unique coded combinations x `replicates` = 96 runs by default. Run
    order is randomized (a real press run would be randomized to avoid
    confounding a factor with drift over time) using a seed so the design is
    reproducible, but the *design* itself (which combination appears, how
    many times) is a deterministic, fully crossed factorial, not randomized.
    """
    combos = list(itertools.product([-1, 1], repeat=N_FACTORS))
    assert len(combos) == RUNS_PER_REPLICATE

    rows = []
    for rep in range(1, replicates + 1):
        for combo in combos:
            rows.append({"replicate": rep, **dict(zip(FACTOR_NAMES, combo))})

    df = pd.DataFrame(rows)
    df.insert(0, "design_run_id", np.arange(1, len(df) + 1))

    # Randomize physical run order (does not change the design, only the
    # order runs would be executed on a press to avoid time-trend
    # confounding). Kept reproducible with a fixed seed.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "run_order", np.arange(1, len(df) + 1))
    return df


def add_null_factors(design: pd.DataFrame, n_null: int = N_NULL_FACTORS, seed: int = 990101) -> pd.DataFrame:
    """Append `n_null` coded (-1/+1) factor columns that are independently
    randomized and never read by the response generator.

    The RNG stream used here (`seed`) is disjoint from the RNG streams used
    anywhere in `ground_truth.py` or `render.py`, so these columns cannot
    accidentally correlate with the response through a shared random state.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(design)
    out = design.copy()
    for i in range(1, n_null + 1):
        out[f"null_{i}"] = rng.choice([-1, 1], size=n_rows)
    return out


def ofat_design(replicates_per_condition: int = 16, baseline: int = -1, seed: int = 20260515) -> pd.DataFrame:
    """Build the one-factor-at-a-time comparator design on the same 96-run
    budget as the factorial design.

    Six distinct conditions: the all-baseline run, and one run per factor
    with only that factor flipped to its high level, everything else held at
    baseline. `replicates_per_condition=16` x 6 conditions = 96 runs, exactly
    matching `design.TOTAL_RUNS`.
    """
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
