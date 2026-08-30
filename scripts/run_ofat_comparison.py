"""Run the one-factor-at-a-time (OFAT) comparator on the same 96-run budget
as the factorial design, through the identical simulated response
generator, and show honestly that OFAT cannot detect the seeded
ink_pressure x nozzle_drive_voltage interaction.

Run with:
    .venv\\Scripts\\python.exe scripts\\run_ofat_comparison.py

Writes docs/ofat_summary.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from print_doe.analysis import build_design_matrix, effect_table, SIGNIFICANCE_CRITERION
from print_doe.design import ofat_design, FACTOR_NAMES, TOTAL_RUNS
from print_doe.simulate import simulate_responses

DOCS = Path(__file__).resolve().parents[1] / "docs"
DESIGN_SEED = 20260515
SIM_SEED = 8181

INTERACTION_A, INTERACTION_B = "ink_pressure", "nozzle_drive_voltage"


def main():
    design = ofat_design(seed=DESIGN_SEED)
    assert len(design) == TOTAL_RUNS

    simulated = simulate_responses(design, seed=SIM_SEED)

    lines = []
    lines.append("=== OFAT design shape ===")
    lines.append(f"total runs: {len(simulated)} (6 conditions x 16 replicates), same budget as the factorial design")
    lines.append("conditions: all-baseline, plus one run per factor with only that factor flipped high")
    lines.append("")

    # --- Main-effects-only OLS: this part OFAT CAN do ---
    X_main = build_design_matrix(simulated, factor_names=FACTOR_NAMES, include_interactions=False)
    X_main = sm.add_constant(X_main, has_constant="add")
    y = simulated["banding_score"].astype(float)
    main_result = sm.OLS(y, X_main).fit()
    main_table = effect_table(main_result)
    lines.append("=== OFAT main-effects-only model (the model OFAT's own design supports) ===")
    lines.append(main_table.to_string(index=False, formatters={
        "estimate": "{:.4f}".format, "std_err": "{:.4f}".format,
        "ci_lower": "{:.4f}".format, "ci_upper": "{:.4f}".format, "p_value": "{:.6f}".format,
    }))
    lines.append("")

    # --- Evidence 1: the interaction column is an exact linear function of
    # the two main-effect columns in the OFAT design (perfect confounding). ---
    a = simulated[INTERACTION_A].astype(float).values
    b = simulated[INTERACTION_B].astype(float).values
    interaction_col = a * b
    design_for_fit = np.column_stack([np.ones_like(a), a, b])
    coefs, residuals, rank, sv = np.linalg.lstsq(design_for_fit, interaction_col, rcond=None)
    predicted = design_for_fit @ coefs
    ss_res = float(np.sum((interaction_col - predicted) ** 2))
    ss_tot = float(np.sum((interaction_col - interaction_col.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    lines.append("=== Evidence 1: interaction column is a linear function of the two main columns ===")
    lines.append(f"regressing ({INTERACTION_A} x {INTERACTION_B}) on [1, {INTERACTION_A}, {INTERACTION_B}] "
                 f"in the OFAT design: R^2 = {r_squared:.10f}")
    lines.append("R^2 = 1.0 means the interaction column carries zero information beyond what the two main")
    lines.append("columns already carry in this design; it is exactly, not approximately, confounded.")
    lines.append("")

    # --- Evidence 2: attempt to fit the interaction term directly; the
    # design matrix including it is rank-deficient. ---
    X_with_interaction = build_design_matrix(simulated, factor_names=FACTOR_NAMES, include_interactions=False)
    X_with_interaction[f"{INTERACTION_A}:{INTERACTION_B}"] = interaction_col
    X_with_interaction = sm.add_constant(X_with_interaction, has_constant="add")
    rank_full = np.linalg.matrix_rank(X_with_interaction.values)
    n_cols = X_with_interaction.shape[1]
    lines.append("=== Evidence 2: design matrix rank when the interaction term is forced into the model ===")
    lines.append(f"columns in [const, 5 main effects, {INTERACTION_A}:{INTERACTION_B}]: {n_cols}")
    lines.append(f"numerical rank of that design matrix: {rank_full}")
    lines.append(f"rank deficient: {rank_full < n_cols}")

    interaction_result = sm.OLS(y, X_with_interaction).fit()
    interaction_row = interaction_result.params.get(f"{INTERACTION_A}:{INTERACTION_B}", float("nan"))
    interaction_se = interaction_result.bse.get(f"{INTERACTION_A}:{INTERACTION_B}", float("nan"))
    lines.append(f"statsmodels OLS still returns a coefficient via pseudo-inverse when asked "
                 f"(estimate={interaction_row:.4f}, std_err={interaction_se}), but with a rank-deficient design "
                 f"this coefficient is not identifiable: it is the minimum-norm solution among infinitely many "
                 f"that fit the OFAT data equally well, not a unique estimate of the interaction, and its "
                 f"standard error is undefined (statsmodels reports it via a generalized inverse, not a real "
                 f"sampling distribution). The true seeded interaction, +4.0, cannot be told apart from 0 or any")
    lines.append(f"other value using only this design; the honest conclusion is 'not estimable', not 'estimated as {interaction_row:.2f}'.")
    lines.append("")
    lines.append("=== Conclusion ===")
    lines.append("The factorial design's own effect table (docs/factorial_summary.txt) flags")
    lines.append(f"{INTERACTION_A}:{INTERACTION_B} significant on the identical 96-run budget. The OFAT design, which")
    lines.append("never varies both factors away from baseline in the same run, cannot estimate that interaction")
    lines.append("at all: it is exactly confounded with the two main effects (R^2 = 1.0 above), not merely")
    lines.append("underpowered to detect it.")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "ofat_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {DOCS / 'ofat_summary.txt'}")


if __name__ == "__main__":
    main()
