"""Build the replicated 2^4 filtration-bench factorial design (80 runs),
simulate the bench through the OpenCV render/measure defect-counting
pipeline, fit the full factorial OLS model (reusing analysis.py's
response-agnostic machinery), and report effect recovery and Monte Carlo
power for every term.

Run with:
    .venv\\Scripts\\python.exe scripts\\run_filtration_factorial_analysis.py

Writes docs/filtration_factorial_summary.txt,
docs/filtration_effect_table.csv, and two sample rendered frames to
docs/filtration_sample_run_low.png / filtration_sample_run_high.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from print_doe import filtration_ground_truth
from print_doe.analysis import fit_ols, effect_table, manual_contrast_effect, SIGNIFICANCE_CRITERION
from print_doe.filtration_design import FACTOR_NAMES, TOTAL_RUNS, REPLICATES, full_factorial_design
from print_doe.filtration_power import monte_carlo_power_filtration
from print_doe.filtration_response import render_filtration_frame
from print_doe.filtration_simulate import simulate_filtration_responses

DOCS = Path(__file__).resolve().parents[1] / "docs"
DESIGN_SEED = 20260601
SIM_SEED = 5252


def main():
    design = full_factorial_design(seed=DESIGN_SEED)
    assert len(design) == TOTAL_RUNS

    simulated = simulate_filtration_responses(design, seed=SIM_SEED)

    result, X = fit_ols(simulated, response_col="defect_count", factor_names=FACTOR_NAMES)
    table = effect_table(result)

    lines = []
    lines.append("=== Filtration-bench design shape ===")
    lines.append(f"total runs: {len(simulated)} (16 unique combinations x {REPLICATES} replicates)")
    lines.append(f"factors: {FACTOR_NAMES}")
    lines.append("")
    lines.append(f"Significance criterion: {SIGNIFICANCE_CRITERION}")
    lines.append("")
    lines.append("=== Full effect table (4 main effects, 6 two-way interactions) ===")
    lines.append(table.to_string(index=False, formatters={
        "estimate": "{:.4f}".format, "std_err": "{:.4f}".format,
        "ci_lower": "{:.4f}".format, "ci_upper": "{:.4f}".format, "p_value": "{:.6f}".format,
    }))
    lines.append("")

    # --- Reference oracle cross-check on the 4 seeded-effect terms ---
    lines.append("=== Manual contrast oracle cross-check (no statsmodels) ===")
    check_terms = {
        "pore_size": "pore_size",
        "dispense_volume": "dispense_volume",
        "flow_rate": "flow_rate",
        "pore_size:flow_rate": ("pore_size", "flow_rate"),
    }
    for term, cols in check_terms.items():
        ols_est = table.loc[table["term"] == term, "estimate"].iloc[0]
        manual_est = manual_contrast_effect(simulated, "defect_count", cols)
        lines.append(f"{term}: OLS={ols_est:.4f}  manual_contrast={manual_est:.4f}  diff={abs(ols_est - manual_est):.6f}")
    lines.append("")

    # --- Effect recovery: the 3 seeded main effects + 1 seeded interaction ---
    seeded_terms = ["pore_size", "dispense_volume", "flow_rate", "pore_size:flow_rate"]
    recovered = table[table["term"].isin(seeded_terms) & table["significant"]]
    n_recovered = len(recovered)
    lines.append(f"=== Effect recovery: {n_recovered} of {len(seeded_terms)} seeded effects flagged significant ===")
    for t in seeded_terms:
        row = table[table["term"] == t].iloc[0]
        lines.append(f"  {t}: significant={bool(row['significant'])} (p={row['p_value']:.6f})")
    lines.append("")

    # --- purge_volume: real factor, true zero effect, no interaction seeded on it ---
    purge_row = table[table["term"] == "purge_volume"].iloc[0]
    lines.append(f"purge_volume (true zero main effect, not counted in seeded-effect recovery): "
                 f"significant={bool(purge_row['significant'])} (p={purge_row['p_value']:.6f})")
    lines.append("")

    # --- Monte Carlo power / empirical detection rate on every effect ---
    lines.append("=== Monte Carlo power (200 independent re-simulations, fresh noise each time) ===")
    mc = monte_carlo_power_filtration(n_reps=200)
    lines.append(mc.to_string(index=False))
    lines.append("")
    lines.append("For pore_size, dispense_volume, flow_rate, and pore_size:flow_rate the")
    lines.append("empirical_rate above is statistical power (true nonzero effect). For")
    lines.append("purge_volume and any 2-way interaction not seeded, empirical_rate is the")
    lines.append("empirical false-positive rate (true zero effect), which should sit near the")
    lines.append("nominal alpha = 0.05.")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "filtration_factorial_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(DOCS / "filtration_effect_table.csv", index=False)

    # --- Two sample rendered frames for the README, low score vs high score ---
    rng = np.random.default_rng(878)
    low_image, _ = render_filtration_frame(score=22.0, rng=rng)
    high_image, _ = render_filtration_frame(score=58.0, rng=rng)
    cv2.imwrite(str(DOCS / "filtration_sample_run_low.png"), low_image)
    cv2.imwrite(str(DOCS / "filtration_sample_run_high.png"), high_image)

    print("\n".join(lines))
    print(f"\nWrote {DOCS / 'filtration_factorial_summary.txt'}")
    print(f"Wrote {DOCS / 'filtration_effect_table.csv'}")


if __name__ == "__main__":
    main()
