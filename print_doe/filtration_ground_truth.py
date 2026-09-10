"""The hidden ground truth response generator for the filtration-bench
extension.

This is the ONLY place in the filtration extension that knows which effects
are real. The design and analysis code (`filtration_design.py`,
`filtration_response.py`, `filtration_simulate.py`, `analysis.py`,
`filtration_power.py`, and every script under `scripts/`) never imports
`TRUE_MAIN_EFFECTS` or `TRUE_INTERACTION_COEF` from here; it only ever sees
the measured defect-count response that `filtration_response.py` produces
from the score this module computes. The only code allowed to compare an
analysis result against these constants is the validation/reporting step
that checks "did we recover what we seeded" (`scripts/run_filtration_*.py`),
exactly the same precedent as `ground_truth.py` being imported only by
`scripts/run_factorial_analysis.py` for the same reason.

Seeded effects (4 controllable bench settings: 3 main effects, 1 two-factor
interaction; `purge_volume` is a real controllable factor with zero true
effect, main or interaction, included so the analysis also has to correctly
report a null result on a factor that was actually varied, exactly
mirroring `purge_interval` in `ground_truth.py`):

    main effect  pore_size          +6.0
    main effect  dispense_volume    +4.0
    main effect  flow_rate          +5.0
    interaction  pore_size x flow_rate   +3.0
    (no effect)  purge_volume        0.0

Regression-coefficient convention: with -1/+1 coded factors, the listed
number is the *coefficient* on the coded column, i.e. half the difference
between the high-level mean and the low-level mean. This matches what
statsmodels OLS estimates directly and what the manual contrast oracle in
analysis.py computes.

Interpretation: the hidden score models a membrane's true contamination
load (larger pore size lets more particles through, higher dispense volume
and flow rate push more material across the membrane per run); higher score
means more true particles are actually deposited on the membrane for that
run. `filtration_response.py` never sees this score directly; it only sees
it as the input to a noisy Poisson particle-placement and imaging-count
pipeline.
"""
from __future__ import annotations

from print_doe.filtration_design import FACTOR_NAMES

INTERCEPT = 40.0

TRUE_MAIN_EFFECTS = {
    "pore_size": 6.0,
    "dispense_volume": 4.0,
    "purge_volume": 0.0,
    "flow_rate": 5.0,
}

TRUE_INTERACTION_FACTORS = ("pore_size", "flow_rate")
TRUE_INTERACTION_COEF = 3.0

PROCESS_NOISE_SD = 2.5

assert set(TRUE_MAIN_EFFECTS) == set(FACTOR_NAMES)
assert TRUE_INTERACTION_FACTORS[0] in FACTOR_NAMES and TRUE_INTERACTION_FACTORS[1] in FACTOR_NAMES


def true_score(factor_levels: dict, rng) -> float:
    """Compute the hidden, noiseless-except-for-process-noise contamination
    score for one filtration-bench run.

    `factor_levels` may be an entire run row (a pandas Series or dict); this
    function only ever reads the four keys named in FACTOR_NAMES.
    """
    score = INTERCEPT
    for factor, coef in TRUE_MAIN_EFFECTS.items():
        score += coef * factor_levels[factor]
    a, b = TRUE_INTERACTION_FACTORS
    score += TRUE_INTERACTION_COEF * factor_levels[a] * factor_levels[b]
    score += rng.normal(0.0, PROCESS_NOISE_SD)
    return float(score)
