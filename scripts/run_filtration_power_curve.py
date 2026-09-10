"""Power-curve search: for a baseline-vs-treatment comparison where the
treatment produces a true 20% reduction in expected defect count, sweep
replicate counts per condition and find the minimum replicate count at
which empirical Monte Carlo power to detect the reduction reaches 95%.

This is a new kind of analysis relative to `run_filtration_factorial_analysis.py`:
it is not about recovering seeded factorial effects, it is about how many
replicates a baseline-vs-treatment bench comparison needs.

Baseline is the filtration-bench process running at its nominal setting
(`filtration_ground_truth.INTERCEPT`, i.e. the score with every factor at
its design center); this script is the validation/reporting step, so it is
allowed to import `filtration_ground_truth` for this reference value, the
same precedent as `run_factorial_analysis.py` importing `ground_truth`.

Run with:
    .venv\\Scripts\\python.exe scripts\\run_filtration_power_curve.py

Writes docs/filtration_power_curve.csv and docs/filtration_power_curve_summary.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from print_doe import filtration_ground_truth
from print_doe.filtration_power import power_curve_for_defect_reduction

DOCS = Path(__file__).resolve().parents[1] / "docs"

REPLICATE_COUNTS = [2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40]
REDUCTION_FRAC = 0.20
N_REPS = 400
TARGET_POWER = 0.95


def main():
    baseline_score = filtration_ground_truth.INTERCEPT

    curve = power_curve_for_defect_reduction(
        replicate_counts=REPLICATE_COUNTS,
        baseline_score=baseline_score,
        reduction_frac=REDUCTION_FRAC,
        n_reps=N_REPS,
    )

    DOCS.mkdir(exist_ok=True)
    curve.to_csv(DOCS / "filtration_power_curve.csv", index=False)

    at_or_above_target = curve[curve["empirical_power"] >= TARGET_POWER]
    if len(at_or_above_target):
        min_replicates = int(at_or_above_target["replicates_per_condition"].min())
        found = True
    else:
        min_replicates = None
        found = False

    lines = []
    lines.append("=== Power curve: baseline vs 20% defect-count reduction ===")
    lines.append(f"baseline hidden score: {baseline_score} (filtration_ground_truth.INTERCEPT, nominal bench setting)")
    lines.append(f"reduction_frac: {REDUCTION_FRAC} (true reduction applied to the expected particle rate, "
                 f"not to the measured output)")
    lines.append(f"n_reps per replicate count: {N_REPS}")
    lines.append(f"target power: {TARGET_POWER}")
    lines.append("")
    lines.append(curve.to_string(index=False, formatters={
        "empirical_power": "{:.4f}".format,
        "mean_baseline_defect_count": "{:.3f}".format,
        "mean_treatment_defect_count": "{:.3f}".format,
        "baseline_rate": "{:.3f}".format,
        "treatment_rate": "{:.3f}".format,
    }))
    lines.append("")
    if found:
        lines.append(f"=== Minimum replicates per condition at which empirical power >= {TARGET_POWER}: "
                     f"{min_replicates} ===")
    else:
        max_tested = max(REPLICATE_COUNTS)
        max_power = float(curve["empirical_power"].max())
        lines.append(f"=== Empirical power did not reach {TARGET_POWER} within the tested range "
                     f"(up to {max_tested} replicates per condition; max observed power {max_power:.4f}) ===")

    (DOCS / "filtration_power_curve_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {DOCS / 'filtration_power_curve.csv'}")
    print(f"Wrote {DOCS / 'filtration_power_curve_summary.txt'}")


if __name__ == "__main__":
    main()
