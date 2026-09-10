"""Design-matrix construction for the filtration-bench extension: the full
replicated 2^4 factorial over 4 bench settings, and the one-factor-at-a-time
(OFAT) comparator on the identical run budget.

Mirrors `design.py` exactly (same orthogonal-coding, randomized-run-order,
reproducible-seed pattern) but is a separate factor space: this module knows
nothing about the press-settings design, and nothing in this file knows about
the response-generating ground truth in `filtration_ground_truth.py`. It only
produces coded (-1 / +1) factor columns.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# The four controllable filtration-bench settings, in the order they appear
# on every design matrix in this module. Order matters only for readability;
# the design is orthogonal regardless of column order.
FACTOR_NAMES = [
    "pore_size",
    "dispense_volume",
    "purge_volume",
    "flow_rate",
]

N_FACTORS = len(FACTOR_NAMES)          # 4
RUNS_PER_REPLICATE = 2 ** N_FACTORS    # 16 unique combinations
# 5 replicates x 16 = 80 runs. Chosen, not defaulted from the press design,
# so that the OFAT comparator (5 conditions: baseline + one flip per factor)
# divides the same run budget evenly at 16 replicates per condition, exactly
# mirroring the press design's 16-replicates-per-OFAT-condition choice
# without forcing an unrelated total run count to match the 5-factor design.
REPLICATES = 5
TOTAL_RUNS = REPLICATES * RUNS_PER_REPLICATE  # 80

# OFAT: baseline + one condition per factor = 5 conditions.
N_OFAT_CONDITIONS = N_FACTORS + 1
OFAT_REPLICATES_PER_CONDITION = TOTAL_RUNS // N_OFAT_CONDITIONS  # 16
assert OFAT_REPLICATES_PER_CONDITION * N_OFAT_CONDITIONS == TOTAL_RUNS


def full_factorial_design(replicates: int = REPLICATES, seed: int = 20260601) -> pd.DataFrame:
    """Build the replicated full 2^4 two-level factorial design over the 4
    filtration-bench settings.

    16 unique coded combinations x `replicates` = 80 runs by default. Run
    order is randomized (a real bench run would be randomized to avoid
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

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "run_order", np.arange(1, len(df) + 1))
    return df


def ofat_design(replicates_per_condition: int = OFAT_REPLICATES_PER_CONDITION,
                 baseline: int = -1, seed: int = 20260615) -> pd.DataFrame:
    """Build the one-factor-at-a-time comparator design on the same 80-run
    budget as `full_factorial_design`.

    Five distinct conditions: the all-baseline run, and one run per factor
    with only that factor flipped to its high level, everything else held at
    baseline. `replicates_per_condition=16` x 5 conditions = 80 runs, exactly
    matching `TOTAL_RUNS`.
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
