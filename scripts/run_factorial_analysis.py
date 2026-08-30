"""Build the replicated 2^5 factorial design (96 runs) plus 20 null factor
columns, simulate the press through the OpenCV render/measure pipeline,
fit the full factorial OLS model, and report effect recovery, false
positives, and Monte Carlo power/CI for every effect.

Run with:
    .venv\\Scripts\\python.exe scripts\\run_factorial_analysis.py

Writes docs/factorial_effect_table.txt, docs/factorial_summary.txt, and two
sample rendered frames to docs/sample_run_low.png / sample_run_high.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from print_doe import ground_truth
from print_doe.analysis import (
    fit_ols, effect_table, manual_contrast_effect, SIGNIFICANCE_CRITERION,
    bonferroni_significant, ALPHA,
)
from print_doe.design import add_null_factors, full_factorial_design, FACTOR_NAMES, N_NULL_FACTORS, TOTAL_RUNS
from print_doe.power import monte_carlo_power
from print_doe.render import render_run_image
from print_doe.simulate import simulate_responses

DOCS = Path(__file__).resolve().parents[1] / "docs"
DESIGN_SEED = 20260501
SIM_SEED = 4242


def main():
    design = full_factorial_design(seed=DESIGN_SEED)
    design = add_null_factors(design)
    assert len(design) == TOTAL_RUNS

    simulated = simulate_responses(design, seed=SIM_SEED)
    null_cols = [f"null_{i}" for i in range(1, N_NULL_FACTORS + 1)]

    result, X = fit_ols(simulated, response_col="banding_score", null_cols=null_cols)
    table = effect_table(result)

    lines = []
    lines.append("=== Design shape ===")
    lines.append(f"total runs: {len(simulated)} (32 unique combinations x 3 replicates)")
    lines.append(f"factors: {FACTOR_NAMES}")
    lines.append(f"null factor columns: {N_NULL_FACTORS}")
    lines.append("")
    lines.append(f"Significance criterion: {SIGNIFICANCE_CRITERION}")
    lines.append("")
    lines.append("=== Full effect table (main effects, 2-way interactions, null mains) ===")
    lines.append(table.to_string(index=False, formatters={
        "estimate": "{:.4f}".format, "std_err": "{:.4f}".format,
        "ci_lower": "{:.4f}".format, "ci_upper": "{:.4f}".format, "p_value": "{:.6f}".format,
    }))
    lines.append("")

    # --- Reference oracle cross-check on the 5 seeded-effect terms ---
    lines.append("=== Manual contrast oracle cross-check (no statsmodels) ===")
    check_terms = {
        "ink_pressure": "ink_pressure",
        "dwell_time": "dwell_time",
        "substrate_speed": "substrate_speed",
        "nozzle_drive_voltage": "nozzle_drive_voltage",
        "ink_pressure:nozzle_drive_voltage": ("ink_pressure", "nozzle_drive_voltage"),
    }
    for term, cols in check_terms.items():
        ols_est = table.loc[table["term"] == term, "estimate"].iloc[0]
        manual_est = manual_contrast_effect(simulated, "banding_score", cols)
        lines.append(f"{term}: OLS={ols_est:.4f}  manual_contrast={manual_est:.4f}  diff={abs(ols_est - manual_est):.6f}")
    lines.append("")

    # --- Effect recovery: the 4 seeded main effects + 1 seeded interaction ---
    seeded_terms = [
        "ink_pressure", "dwell_time", "substrate_speed", "nozzle_drive_voltage",
        "ink_pressure:nozzle_drive_voltage",
    ]
    recovered = table[table["term"].isin(seeded_terms) & table["significant"]]
    n_recovered = len(recovered)
    lines.append(f"=== Effect recovery: {n_recovered} of {len(seeded_terms)} seeded effects flagged significant ===")
    for t in seeded_terms:
        row = table[table["term"] == t].iloc[0]
        lines.append(f"  {t}: significant={bool(row['significant'])} (p={row['p_value']:.6f})")
    lines.append("")

    # --- purge_interval: real factor, true zero effect, no interaction seeded on it ---
    purge_row = table[table["term"] == "purge_interval"].iloc[0]
    lines.append(f"purge_interval (true zero main effect, not counted in seeded-effect recovery): "
                 f"significant={bool(purge_row['significant'])} (p={purge_row['p_value']:.6f})")
    lines.append("")

    # --- False positives over 20 null factors ---
    null_table = table[table["term"].isin(null_cols)].copy()
    n_false_positive_raw = int(null_table["significant"].sum())
    lines.append(f"=== False positives over {N_NULL_FACTORS} null factors, nominal alpha=0.05 per test: "
                 f"{n_false_positive_raw} of {N_NULL_FACTORS} ===")
    if n_false_positive_raw:
        lines.append(null_table[null_table["significant"]].to_string(index=False))
    lines.append("(20 independent true-null tests at alpha=0.05 have an expected value of 1.0 false")
    lines.append(" positives by construction; this is the nominal per-test rate applied with no")
    lines.append(" correction for testing 20 factors at once.)")
    lines.append("")

    # --- Same claim, Bonferroni-corrected for the 20-test screening family.
    # This is the methodologically standard treatment of a null-factor
    # screening claim (control the family-wise error rate across the 20
    # simultaneous null tests), applied only here, not to the 5 seeded
    # effects above (which are pre-specified confirmatory tests, still
    # judged at the nominal 95% CI). See print_doe/analysis.py
    # bonferroni_significant docstring. ---
    null_table["bonferroni_significant"] = bonferroni_significant(null_table, n_tests=N_NULL_FACTORS)
    n_false_positive_bonferroni = int(null_table["bonferroni_significant"].sum())
    lines.append(f"=== Same claim, Bonferroni-corrected for the {N_NULL_FACTORS}-test screening family "
                 f"(adjusted alpha={ALPHA / N_NULL_FACTORS:.4f}): {n_false_positive_bonferroni} of {N_NULL_FACTORS} ===")
    if n_false_positive_bonferroni:
        lines.append(null_table[null_table["bonferroni_significant"]].to_string(index=False))
    lines.append("")

    # --- Monte Carlo power / empirical detection rate on every effect ---
    lines.append("=== Monte Carlo power (200 independent re-simulations, fresh noise each time) ===")
    mc = monte_carlo_power(n_reps=200)
    lines.append(mc.to_string(index=False))
    lines.append("")
    lines.append("For ink_pressure, dwell_time, substrate_speed, nozzle_drive_voltage, and")
    lines.append("ink_pressure:nozzle_drive_voltage the empirical_rate above is statistical power")
    lines.append("(true nonzero effect). For purge_interval and any 2-way interaction not seeded,")
    lines.append("empirical_rate is the empirical false-positive rate (true zero effect), which")
    lines.append("should sit near the nominal alpha = 0.05.")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "factorial_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(DOCS / "factorial_effect_table.csv", index=False)

    # --- Two sample rendered frames for the README, low score vs high score ---
    rng = np.random.default_rng(777)
    low_image = render_run_image(score=25.0, rng=rng)
    high_image = render_run_image(score=75.0, rng=rng)
    cv2.imwrite(str(DOCS / "sample_run_low.png"), low_image)
    cv2.imwrite(str(DOCS / "sample_run_high.png"), high_image)

    print("\n".join(lines))
    print(f"\nWrote {DOCS / 'factorial_summary.txt'}")
    print(f"Wrote {DOCS / 'factorial_effect_table.csv'}")


if __name__ == "__main__":
    main()
