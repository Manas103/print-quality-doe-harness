"""The hidden ground truth response generator.

This is the ONLY place in the repo that knows which effects are real. The
analysis code (`analysis.py`, `power.py`, and every script under `scripts/`)
never imports `TRUE_MAIN_EFFECTS` or `TRUE_INTERACTION` from here; it only
ever sees the measured response column that `render.py` produces from the
score this module computes. The only code that is allowed to compare an
analysis result against these constants is the validation/reporting step
that checks "did we recover what we seeded", which is exactly the honesty
check this project claims to run.

Seeded effects (5 controllable factors: 4 main effects, 1 two-factor
interaction; `purge_interval` is a real controllable factor with zero true
effect, main or interaction, included so the analysis also has to correctly
report a null result on a factor that was actually varied, not just on the
20 synthetic null columns added in `design.add_null_factors`):

    main effect  ink_pressure            +8.0
    main effect  dwell_time              -6.0
    main effect  substrate_speed         +5.0
    main effect  nozzle_drive_voltage    +7.0
    interaction  ink_pressure x nozzle_drive_voltage   +4.0
    (no effect)  purge_interval           0.0

Regression-coefficient convention: with -1/+1 coded factors, the listed
number is the *coefficient* on the coded column, i.e. half the difference
between the high-level mean and the low-level mean. This matches what
statsmodels OLS estimates directly and what the manual contrast oracle in
analysis.py computes.
"""
from __future__ import annotations

from print_doe.design import FACTOR_NAMES

INTERCEPT = 50.0

TRUE_MAIN_EFFECTS = {
    "ink_pressure": 8.0,
    "dwell_time": -6.0,
    "substrate_speed": 5.0,
    "nozzle_drive_voltage": 7.0,
    "purge_interval": 0.0,
}

TRUE_INTERACTION_FACTORS = ("ink_pressure", "nozzle_drive_voltage")
TRUE_INTERACTION_COEF = 4.0

PROCESS_NOISE_SD = 2.0

assert set(TRUE_MAIN_EFFECTS) == set(FACTOR_NAMES)
assert TRUE_INTERACTION_FACTORS[0] in FACTOR_NAMES and TRUE_INTERACTION_FACTORS[1] in FACTOR_NAMES


def true_score(factor_levels: dict, rng) -> float:
    """Compute the hidden, noiseless-except-for-process-noise quality score
    for one run.

    `factor_levels` may be an entire run row (a pandas Series or dict that
    also contains null_1..null_20, run_order, replicate, etc.); this
    function only ever reads the five keys named in FACTOR_NAMES. It is
    structurally incapable of reading a null factor: no key containing
    "null" is ever looked up (see test_ground_truth_isolation.py, which
    both fuzzes the null columns and inspects this function's source to
    confirm the substring "null" never appears in it).
    """
    score = INTERCEPT
    for factor, coef in TRUE_MAIN_EFFECTS.items():
        score += coef * factor_levels[factor]
    a, b = TRUE_INTERACTION_FACTORS
    score += TRUE_INTERACTION_COEF * factor_levels[a] * factor_levels[b]
    score += rng.normal(0.0, PROCESS_NOISE_SD)
    return float(score)
