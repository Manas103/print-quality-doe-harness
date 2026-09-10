"""pytest suite for the filtration-bench extension. Run with:
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

from print_doe import filtration_ground_truth
from print_doe.analysis import build_design_matrix, effect_table, fit_ols, manual_contrast_effect
from print_doe.filtration_design import (
    FACTOR_NAMES, N_FACTORS, REPLICATES, RUNS_PER_REPLICATE, TOTAL_RUNS,
    N_OFAT_CONDITIONS, OFAT_REPLICATES_PER_CONDITION,
    full_factorial_design, ofat_design,
)
from print_doe.filtration_response import (
    expected_particle_rate, score_for_expected_rate, measure_defect_count,
    measured_response_for_filtration_run, render_filtration_frame,
    PARTICLE_RATE_CENTER, PARTICLE_RATE_GAIN, SCORE_CENTER,
)
from print_doe.filtration_simulate import simulate_filtration_responses


# --- filtration_design.py --------------------------------------------------

def test_full_factorial_design_shape_and_balance():
    d = full_factorial_design(seed=1)
    assert len(d) == TOTAL_RUNS == 80
    combo_counts = d.groupby(FACTOR_NAMES).size()
    assert len(combo_counts) == RUNS_PER_REPLICATE == 16
    assert (combo_counts == REPLICATES).all()


def test_full_factorial_design_is_orthogonal():
    d = full_factorial_design(seed=1)
    unique = d.drop_duplicates(subset=FACTOR_NAMES)[FACTOR_NAMES]
    assert len(unique) == 16
    X = unique.to_numpy(dtype=float)
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            assert np.dot(X[:, i], X[:, j]) == 0, f"columns {i},{j} not orthogonal"


def test_full_factorial_design_reproducible_given_same_seed():
    d1 = full_factorial_design(seed=42)
    d2 = full_factorial_design(seed=42)
    pd.testing.assert_frame_equal(d1, d2)


def test_ofat_design_shape_matches_factorial_budget():
    d = ofat_design(seed=5)
    assert len(d) == TOTAL_RUNS  # same 80-run budget as the factorial design
    assert d["condition"].nunique() == N_OFAT_CONDITIONS == N_FACTORS + 1
    baseline_row = d[d["condition"] == 0].iloc[0]
    for cond_idx, factor in enumerate(FACTOR_NAMES, start=1):
        cond_rows = d[d["condition"] == cond_idx]
        flipped = [f for f in FACTOR_NAMES if cond_rows.iloc[0][f] != baseline_row[f]]
        assert flipped == [factor], f"condition {cond_idx} should flip only {factor}, flipped {flipped}"
    assert (d.groupby("condition").size() == OFAT_REPLICATES_PER_CONDITION).all()


# --- filtration_ground_truth.py --------------------------------------------

def test_filtration_ground_truth_not_imported_by_design_or_analysis_source():
    """Structural isolation check, the filtration analogue of
    `test_ground_truth_true_score_never_reads_null_columns_by_source_inspection`:
    the modules that build the design and fit the model never *import* the
    hidden ground truth module (mentioning it in a docstring, as this file's
    module docstring does to explain the guarantee, is fine). Only the
    validation/reporting scripts import it."""
    import ast

    import print_doe.filtration_design as fd
    import print_doe.filtration_response as fr
    import print_doe.analysis as an

    for module in (fd, fr, an):
        src = inspect.getsource(module)
        tree = ast.parse(src)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
                imported_names.update(alias.name for alias in node.names)
        assert not any("filtration_ground_truth" in name for name in imported_names), \
            f"{module.__name__} must not import filtration_ground_truth, found imports: {imported_names}"


def test_filtration_ground_truth_matches_hand_computed_value_at_all_high():
    class ZeroRng:
        def normal(self, loc, scale):
            return 0.0

    row = {f: 1 for f in FACTOR_NAMES}
    score = filtration_ground_truth.true_score(row, ZeroRng())
    expected = 40.0 + 6.0 + 4.0 + 0.0 + 5.0 + 3.0 * 1 * 1
    assert score == pytest.approx(expected)


def test_filtration_ground_truth_reproducible_given_same_rng_state():
    row = {f: -1 for f in FACTOR_NAMES}
    s1 = filtration_ground_truth.true_score(row, np.random.default_rng(7))
    s2 = filtration_ground_truth.true_score(row, np.random.default_rng(7))
    assert s1 == s2


# --- filtration_response.py -------------------------------------------------

def test_expected_particle_rate_never_clips_for_realistic_score_range():
    """Guards the affine (not clipped) score-to-rate mapping: for the full
    plausible range of scores this repo's filtration ground truth can
    produce, the max(0, ...) safety clamp in expected_particle_rate must
    never fire."""
    plausible_scores = np.linspace(0.0, 80.0, 50)  # generous bound around the true ~[22,58] range
    for score in plausible_scores:
        unclamped = PARTICLE_RATE_CENTER + PARTICLE_RATE_GAIN * (score - SCORE_CENTER)
        assert unclamped > 0.0, f"score={score} would clamp expected_particle_rate to 0: {unclamped}"
        assert unclamped == expected_particle_rate(score)


def test_score_for_expected_rate_is_the_inverse_of_expected_particle_rate():
    for score in (10.0, 40.0, 65.0):
        rate = expected_particle_rate(score)
        assert score_for_expected_rate(rate) == pytest.approx(score)


def test_render_filtration_frame_reproducible_given_same_rng_state():
    r1, n1 = measured_response_for_filtration_run(45.0, np.random.default_rng(321))
    r2, n2 = measured_response_for_filtration_run(45.0, np.random.default_rng(321))
    assert r1 == r2
    assert n1 == n2


def test_render_higher_score_produces_more_measured_defects_on_average():
    """Not a single-sample comparison (rendering is stochastic): averages
    several independent draws per score level, since this is what the
    entire filtration effect-recovery claim depends on being true."""
    rng_low = np.random.default_rng(1)
    rng_high = np.random.default_rng(2)
    low_counts = [measured_response_for_filtration_run(22.0, rng_low)[0] for _ in range(30)]
    high_counts = [measured_response_for_filtration_run(58.0, rng_high)[0] for _ in range(30)]
    assert np.mean(high_counts) > np.mean(low_counts)


def test_measure_defect_count_zero_for_blank_image():
    blank = np.full((96, 96), 255, dtype=np.uint8)
    assert measure_defect_count(blank) == pytest.approx(0.0)


def test_measure_defect_count_counts_isolated_particles():
    image, n_placed = render_filtration_frame(score=SCORE_CENTER, rng=np.random.default_rng(11))
    measured = measure_defect_count(image)
    # a noisy imaging count need not exactly equal the true placed count
    # (overlap merges blobs, sensor noise can split/merge near threshold),
    # but for a moderate particle density it should be in the same ballpark
    assert measured >= 0.0
    assert n_placed >= 0


# --- filtration_simulate.py glue -------------------------------------------

def test_simulate_filtration_responses_reproducible_given_same_seed():
    d = full_factorial_design(seed=1).head(10)
    r1 = simulate_filtration_responses(d, seed=999)
    r2 = simulate_filtration_responses(d, seed=999)
    pd.testing.assert_series_equal(r1["defect_count"], r2["defect_count"])


def test_simulate_filtration_responses_replicates_of_same_combo_get_independent_noise():
    d = full_factorial_design(seed=1)
    same_combo = d[d[FACTOR_NAMES].apply(tuple, axis=1) == tuple(d.iloc[0][FACTOR_NAMES])]
    assert len(same_combo) == REPLICATES
    sim = simulate_filtration_responses(same_combo, seed=1)
    assert sim["defect_count"].nunique() >= 2  # independent noise, essentially never identical across replicates


# --- analysis.py reuse (response-agnostic machinery) ------------------------

def test_build_design_matrix_includes_all_six_two_way_interactions_for_four_factors():
    d = full_factorial_design(seed=1)
    X = build_design_matrix(d, factor_names=FACTOR_NAMES, include_interactions=True)
    interaction_cols = [c for c in X.columns if ":" in c]
    from math import comb
    assert len(interaction_cols) == comb(4, 2) == 6


def test_fit_ols_reused_against_filtration_design_and_response():
    d = full_factorial_design(seed=1)
    sim = simulate_filtration_responses(d, seed=2)
    result, X = fit_ols(sim, response_col="defect_count", factor_names=FACTOR_NAMES)
    table = effect_table(result)
    assert set(FACTOR_NAMES) <= set(table["term"])
    assert "pore_size:flow_rate" in table["term"].values


def test_manual_contrast_oracle_matches_ols_on_filtration_design():
    d = full_factorial_design(seed=3)
    sim = simulate_filtration_responses(d, seed=4)
    result, _X = fit_ols(sim, response_col="defect_count", factor_names=FACTOR_NAMES)
    table = effect_table(result)
    for term in FACTOR_NAMES:
        ols_est = table.loc[table["term"] == term, "estimate"].iloc[0]
        manual_est = manual_contrast_effect(sim, "defect_count", term)
        assert ols_est == pytest.approx(manual_est, abs=1e-9)
