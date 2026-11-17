from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from print_doe import analysis
from print_doe.isp_design import FACTOR_NAMES, N_FACTORS, TOTAL_RUNS, full_factorial_design, ofat_design
from print_doe.isp_detector import DetectorCNN, accuracy, train_detector
from print_doe.isp_pipeline import apply_isp_chain
from print_doe.isp_scene import FRAME_SIZE, generate_eval_set, generate_scene


def test_factorial_design_shape_and_balance():
    design = full_factorial_design()
    assert len(design) == TOTAL_RUNS == 96
    for factor in FACTOR_NAMES:
        assert set(design[factor].unique()) == {-1, 1}
        assert design[factor].sum() == 0  # balanced: equal -1s and +1s


def test_factorial_design_orthogonal():
    design = full_factorial_design()
    X = design[FACTOR_NAMES].to_numpy(dtype=float)
    gram = X.T @ X
    off_diag = gram - np.diag(np.diag(gram))
    assert np.allclose(off_diag, 0.0)


def test_factorial_design_reproducible():
    a = full_factorial_design(seed=1)
    b = full_factorial_design(seed=1)
    pd_equal = (a[FACTOR_NAMES].to_numpy() == b[FACTOR_NAMES].to_numpy()).all()
    assert pd_equal


def test_ofat_design_matches_run_budget():
    design = ofat_design()
    assert len(design) == TOTAL_RUNS
    assert design["condition"].nunique() == N_FACTORS + 1


def test_scene_generator_label_matches_rendered_intensity():
    rng = np.random.default_rng(0)
    frame, has_object = generate_scene(rng)
    assert frame.shape == (FRAME_SIZE, FRAME_SIZE)
    if has_object:
        assert frame.max() >= 150  # the object is a bright blob by construction


def test_scene_generator_reproducible():
    frames_a, labels_a = generate_eval_set(50, seed=7)
    frames_b, labels_b = generate_eval_set(50, seed=7)
    assert np.array_equal(frames_a, frames_b)
    assert np.array_equal(labels_a, labels_b)


@pytest.mark.parametrize("factor,level", [
    ("denoise", -1), ("denoise", 1),
    ("demosaic", -1), ("demosaic", 1),
    ("tone_curve", -1), ("tone_curve", 1),
    ("sharpening", -1), ("sharpening", 1),
    ("white_balance", -1), ("white_balance", 1),
])
def test_isp_step_preserves_shape_and_dtype(factor, level):
    rng = np.random.default_rng(1)
    frame, _ = generate_scene(rng)
    settings = {name: -1 for name in FACTOR_NAMES}
    settings[factor] = level
    out = apply_isp_chain(frame, settings)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_isp_chain_identity_at_all_low_is_not_a_no_op_copy_bug():
    """All factors at -1 should be the literal untouched frame: no ISP step
    should silently mutate the input array in place."""
    rng = np.random.default_rng(2)
    frame, _ = generate_scene(rng)
    original = frame.copy()
    settings = {name: -1 for name in FACTOR_NAMES}
    out = apply_isp_chain(frame, settings)
    assert np.array_equal(frame, original)  # input untouched
    assert np.array_equal(out, original)    # all-low really is identity


def test_denoise_blur_reduces_high_frequency_variance():
    rng = np.random.default_rng(3)
    frame, _ = generate_scene(rng)
    settings = {name: -1 for name in FACTOR_NAMES}
    clean = apply_isp_chain(frame, settings)
    settings["denoise"] = 1
    denoised = apply_isp_chain(frame, settings)
    assert float(np.var(denoised.astype(np.float32))) <= float(np.var(clean.astype(np.float32)))


def test_detector_trains_above_chance_on_clean_scenes():
    frames, labels = generate_eval_set(600, seed=42)
    train_frames, train_labels = frames[:400], labels[:400]
    test_frames, test_labels = frames[400:], labels[400:]
    model = train_detector(train_frames, train_labels, epochs=150, seed=0)
    acc = accuracy(model, test_frames, test_labels)
    assert acc > 0.85  # the object is a large, high-contrast blob; this should be an easy task


def test_detector_accuracy_deterministic_given_fixed_model_and_frames():
    model = DetectorCNN()
    frames, labels = generate_eval_set(20, seed=5)
    acc_a = accuracy(model, frames, labels)
    acc_b = accuracy(model, frames, labels)
    assert acc_a == acc_b


def test_manual_contrast_oracle_matches_ols_on_isp_design():
    """Reference-oracle cross-check using a synthetic (non-detector) response
    column, so this test does not depend on CNN training time: builds a
    design and a response with a known seeded coefficient structure, then
    checks print_doe.analysis's OLS estimate against the manual contrast
    oracle, exactly as the press-settings and filtration suites do."""
    design = full_factorial_design(replicates=2)
    rng = np.random.default_rng(99)
    # A synthetic response with a known main effect and interaction, plus noise.
    response = (
        3.0 * design["denoise"]
        + 2.0 * design["tone_curve"]
        + 1.5 * design["denoise"] * design["tone_curve"]
        + rng.normal(0, 0.2, size=len(design))
    )
    df = design.copy()
    df["accuracy"] = response
    result, _ = analysis.fit_ols(df, response_col="accuracy", factor_names=FACTOR_NAMES)
    table = analysis.effect_table(result)

    ols_interaction = table.loc[table["term"] == "denoise:tone_curve", "estimate"].iloc[0]
    manual_interaction = analysis.manual_contrast_effect(df, "accuracy", ("denoise", "tone_curve"))
    assert abs(ols_interaction - manual_interaction) < 0.05
