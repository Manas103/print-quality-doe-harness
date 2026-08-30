"""pytest suite for print_doe. Run with:
    .venv\\Scripts\\python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from print_doe import ground_truth
from print_doe.analysis import (
    ALPHA, bonferroni_significant, build_design_matrix, effect_table, fit_ols, manual_contrast_effect,
)
from print_doe.design import (
    FACTOR_NAMES, N_NULL_FACTORS, REPLICATES, RUNS_PER_REPLICATE, TOTAL_RUNS,
    add_null_factors, full_factorial_design, ofat_design,
)
from print_doe.render import measure_banding_score, measured_response_for_run, render_run_image
from print_doe.simulate import simulate_responses


# --- design.py -----------------------------------------------------------

def test_full_factorial_design_shape_and_balance():
    d = full_factorial_design(seed=1)
    assert len(d) == TOTAL_RUNS == 96
    # every one of the 32 unique combinations appears exactly REPLICATES times
    combo_counts = d.groupby(FACTOR_NAMES).size()
    assert len(combo_counts) == RUNS_PER_REPLICATE == 32
    assert (combo_counts == REPLICATES).all()


def test_full_factorial_design_is_orthogonal():
    """A real 2-level full factorial's coded columns (main effects and any
    2-way products) are exactly orthogonal: every pairwise dot product is 0."""
    d = full_factorial_design(seed=1)
    unique = d.drop_duplicates(subset=FACTOR_NAMES)[FACTOR_NAMES]
    assert len(unique) == 32
    X = unique.to_numpy(dtype=float)
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            assert np.dot(X[:, i], X[:, j]) == 0, f"columns {i},{j} not orthogonal"


def test_full_factorial_design_reproducible_given_same_seed():
    d1 = full_factorial_design(seed=42)
    d2 = full_factorial_design(seed=42)
    pd.testing.assert_frame_equal(d1, d2)


def test_add_null_factors_are_plus_minus_one_and_disjoint_from_design_columns():
    d = full_factorial_design(seed=1)
    d = add_null_factors(d, n_null=N_NULL_FACTORS)
    null_cols = [f"null_{i}" for i in range(1, N_NULL_FACTORS + 1)]
    assert len(null_cols) == 20
    for c in null_cols:
        assert set(d[c].unique()) <= {-1, 1}
    # null columns must not be literal copies of a real factor or of each other
    for c in null_cols:
        for f in FACTOR_NAMES:
            assert not d[c].equals(d[f]), f"{c} is identical to {f}"


def test_ofat_design_shape_matches_factorial_budget():
    d = ofat_design(seed=5)
    assert len(d) == TOTAL_RUNS  # same 96-run budget as the factorial design
    assert d["condition"].nunique() == len(FACTOR_NAMES) + 1  # baseline + one per factor
    # every non-baseline condition flips exactly one factor from baseline
    baseline_row = d[d["condition"] == 0].iloc[0]
    for cond_idx, factor in enumerate(FACTOR_NAMES, start=1):
        cond_rows = d[d["condition"] == cond_idx]
        flipped = [f for f in FACTOR_NAMES if cond_rows.iloc[0][f] != baseline_row[f]]
        assert flipped == [factor], f"condition {cond_idx} should flip only {factor}, flipped {flipped}"


# --- ground_truth.py -------------------------------------------------------

def test_ground_truth_true_score_never_reads_null_columns_by_source_inspection():
    """Structural guarantee: the substring 'null' never appears in
    true_score's executable body (docstring excluded, since the docstring
    itself explains this guarantee and legitimately contains the word), so
    it is not merely untested that null columns are ignored, it is
    impossible for this function to read one."""
    import ast

    src = inspect.getsource(ground_truth.true_score)
    tree = ast.parse(src)
    func = tree.body[0]
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # drop the docstring node
    body_src = "\n".join(ast.unparse(node) for node in body)
    assert "null" not in body_src.lower(), f"executable body mentions 'null': {body_src}"


def test_ground_truth_true_score_ignores_null_columns_when_fuzzed():
    """Behavioral confirmation of the structural guarantee above: changing
    every null_* value must not change the computed score at all."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    row = {f: 1 for f in FACTOR_NAMES}
    row_with_nulls_a = {**row, **{f"null_{i}": -1 for i in range(1, 21)}}
    row_with_nulls_b = {**row, **{f"null_{i}": 1 for i in range(1, 21)}}
    score_a = ground_truth.true_score(row_with_nulls_a, rng1)
    score_b = ground_truth.true_score(row_with_nulls_b, rng2)
    assert score_a == score_b


def test_ground_truth_matches_hand_computed_value_at_all_high():
    """At all factors +1 with process noise disabled (rng that always
    returns 0), the score must equal intercept + sum(main effects) +
    interaction coefficient, computed by hand from the documented values."""

    class ZeroRng:
        def normal(self, loc, scale):
            return 0.0

    row = {f: 1 for f in FACTOR_NAMES}
    score = ground_truth.true_score(row, ZeroRng())
    expected = 50.0 + 8.0 - 6.0 + 5.0 + 7.0 + 0.0 + 4.0 * 1 * 1
    assert score == pytest.approx(expected)


# --- render.py --------------------------------------------------------------

def test_render_measured_response_is_reproducible_given_same_rng_state():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    r1 = measured_response_for_run(60.0, rng1)
    r2 = measured_response_for_run(60.0, rng2)
    assert r1 == r2


def test_render_higher_score_produces_more_banding_on_average():
    """Not a single-sample comparison (rendering is stochastic): averages
    several independent draws per score level, since this is what the
    entire repo's effect-recovery claim depends on being true."""
    rng_low = np.random.default_rng(1)
    rng_high = np.random.default_rng(2)
    low_scores = [measured_response_for_run(30.0, rng_low) for _ in range(20)]
    high_scores = [measured_response_for_run(80.0, rng_high) for _ in range(20)]
    assert np.mean(high_scores) > np.mean(low_scores)


def test_measure_banding_score_zero_for_uniform_image():
    uniform = np.full((96, 96), 128, dtype=np.uint8)
    assert measure_banding_score(uniform) == pytest.approx(0.0, abs=1e-9)


def test_dot_probability_never_saturates_for_realistic_score_range():
    """Guards the affine (not clipped) score-to-response mapping documented
    in render.py: for the full range of scores this repo's ground truth can
    plausibly produce, the safety clip at [0.03, 0.97] must never fire."""
    from print_doe.render import DOT_PROB_CENTER, DOT_PROB_GAIN, BANDING_GAIN, SCORE_CENTER
    plausible_scores = np.linspace(0.0, 100.0, 50)  # generous bound around the true 20-90ish range
    for score in plausible_scores:
        baseline_shift = DOT_PROB_GAIN * (score - SCORE_CENTER)
        amplitude = BANDING_GAIN * score
        lo = DOT_PROB_CENTER + baseline_shift - amplitude
        hi = DOT_PROB_CENTER + baseline_shift + amplitude
        assert lo > 0.03 and hi < 0.97, f"score={score} would saturate the clip: [{lo},{hi}]"


# --- analysis.py --------------------------------------------------------------

def test_manual_contrast_oracle_matches_ols_on_a_noiseless_synthetic_dataset():
    """Constructs a tiny noiseless dataset with a known main effect and
    interaction, fits OLS, and checks the from-scratch contrast oracle
    reproduces the OLS coefficient exactly (up to floating point)."""
    df = pd.DataFrame({
        "a": [-1, -1, 1, 1],
        "b": [-1, 1, -1, 1],
        "y": [10.0, 10.0, 20.0, 20.0],  # pure main effect on 'a', coefficient 5
    })
    result, _ = fit_ols(df, response_col="y", factor_names=["a", "b"], include_interactions=True)
    table = effect_table(result)
    ols_a = table.loc[table["term"] == "a", "estimate"].iloc[0]
    manual_a = manual_contrast_effect(df, "y", "a")
    assert ols_a == pytest.approx(manual_a, abs=1e-9)
    assert manual_a == pytest.approx(5.0, abs=1e-9)


def test_bonferroni_significant_matches_manual_threshold():
    table = pd.DataFrame({"term": ["x1", "x2", "x3"], "p_value": [0.001, 0.03, 0.2]})
    n_tests = 20
    flags = bonferroni_significant(table, n_tests=n_tests, family_alpha=ALPHA)
    adjusted = ALPHA / n_tests
    expected = table["p_value"] < adjusted
    assert list(flags) == list(expected)
    # 0.001 < 0.0025 (significant), 0.03 and 0.2 are not
    assert list(flags) == [True, False, False]


def test_build_design_matrix_includes_all_ten_two_way_interactions_for_five_factors():
    d = full_factorial_design(seed=1)
    X = build_design_matrix(d, factor_names=FACTOR_NAMES, include_interactions=True)
    interaction_cols = [c for c in X.columns if ":" in c]
    from math import comb
    assert len(interaction_cols) == comb(5, 2) == 10


# --- simulate.py glue -----------------------------------------------------

def test_simulate_responses_reproducible_given_same_seed():
    d = full_factorial_design(seed=1).head(10)
    r1 = simulate_responses(d, seed=999)
    r2 = simulate_responses(d, seed=999)
    pd.testing.assert_series_equal(r1["banding_score"], r2["banding_score"])


def test_simulate_responses_replicates_of_same_combo_get_independent_noise():
    """Two rows with identical factor levels but different design_run_id
    (i.e. different replicates) must not produce identical responses (their
    RNG streams are seeded from different run_ids)."""
    d = full_factorial_design(seed=1)
    same_combo = d[d[FACTOR_NAMES].apply(tuple, axis=1) == tuple(d.iloc[0][FACTOR_NAMES])]
    assert len(same_combo) == REPLICATES
    sim = simulate_responses(same_combo, seed=1)
    assert sim["banding_score"].nunique() == REPLICATES
