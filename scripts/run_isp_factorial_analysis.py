"""Builds the 96-run ISP factorial design, trains the fixed detector once,
measures accuracy under every run's ISP chain, fits the OLS effect model
(reusing print_doe.analysis unchanged), and writes docs/isp_factorial_*.

Run: .venv\\Scripts\\python scripts\\run_isp_factorial_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import itertools

from print_doe import analysis
from print_doe.isp_design import FACTOR_NAMES, full_factorial_design
from print_doe.isp_simulate import TRAIN_SET_SIZE, run_design, train_fixed_detector


def main():
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    print("Training fixed detector on baseline (untouched) scenes...")
    detector = train_fixed_detector()

    design = full_factorial_design()
    print(f"Running {len(design)} ISP-factorial design points...")
    results = run_design(design, detector)
    results.to_csv(docs / "isp_factorial_runs.csv", index=False)

    ols_result, X = analysis.fit_ols(results, response_col="accuracy", factor_names=FACTOR_NAMES)
    table = analysis.effect_table(ols_result)
    table.to_csv(docs / "isp_effect_table.csv", index=False)

    main_effect_terms = FACTOR_NAMES
    interaction_term = "tone_curve:denoise"
    if interaction_term not in table["term"].values:
        interaction_term = "denoise:tone_curve"

    lines = []
    lines.append(f"ISP factorial design: {len(design)} runs ({2**len(FACTOR_NAMES)} unique combos x "
                  f"{len(design)//2**len(FACTOR_NAMES)} replicates)")
    lines.append(f"Fixed detector trained once on {TRAIN_SET_SIZE} baseline scenes")
    lines.append("")
    lines.append("Effect table (OLS, 95% CI):")
    lines.append(table.to_string(index=False))
    lines.append("")

    interaction_row = table[table["term"] == interaction_term].iloc[0]
    interaction_points = interaction_row["estimate"] * 2 * 100  # full low->high swing, percentage points
    lines.append(f"Interaction term: {interaction_term}")
    lines.append(f"Interaction estimate (coded coefficient): {interaction_row['estimate']:.4f}")
    lines.append(f"Interaction full swing (low->high, percentage points): {interaction_points:.2f}")
    lines.append(f"Interaction p-value: {interaction_row['p_value']:.4g}, significant: {interaction_row['significant']}")
    lines.append("")

    main_a, main_b = interaction_term.split(":")
    main_a_row = table[table["term"] == main_a].iloc[0]
    main_b_row = table[table["term"] == main_b].iloc[0]
    main_a_points = abs(main_a_row["estimate"] * 2 * 100)
    main_b_points = abs(main_b_row["estimate"] * 2 * 100)
    lines.append(f"Main effect |{main_a}| (percentage points): {main_a_points:.2f}")
    lines.append(f"Main effect |{main_b}| (percentage points): {main_b_points:.2f}")
    larger_than_both = abs(interaction_points) > main_a_points and abs(interaction_points) > main_b_points
    lines.append(f"Interaction larger in magnitude than both main effects alone: {larger_than_both}")
    lines.append("")

    manual = analysis.manual_contrast_effect(results, "accuracy", (main_a, main_b))
    lines.append(f"Manual contrast oracle cross-check (no statsmodels) for {interaction_term}: "
                 f"{manual * 2 * 100:.2f} percentage points "
                 f"(OLS: {interaction_points:.2f}, abs diff: {abs(manual * 2 * 100 - interaction_points):.3f})")

    summary = "\n".join(lines)
    print(summary)
    (docs / "isp_factorial_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
