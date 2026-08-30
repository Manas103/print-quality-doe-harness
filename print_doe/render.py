"""Render a synthetic halftone/banding defect image from a hidden quality
score, then measure a banding score back out of the rendered pixels with
OpenCV.

This is the honest-OpenCV-use path the project claims: `render_run_image`
never hands its caller a formula, it hands them an 8-bit grayscale image.
`measure_banding_score` never sees the score that produced the image; it
only sees pixel data, exactly like a real line-scan camera would. The
measured value that the design fits, and that the whole rest of this repo
calls "the response", is the OUTPUT of `measure_banding_score`, not the
`score` argument used to render it.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BAND_PERIOD_ROWS = 12           # rows per banding cycle
SCORE_CENTER = 50.0             # matches ground_truth.INTERCEPT; keeps the probability swing centered
DOT_PROB_CENTER = 0.5
DOT_PROB_GAIN = 0.006            # score units (relative to SCORE_CENTER) -> baseline dot-probability shift
BANDING_GAIN = 0.0015            # score units -> row-banding dot-probability amplitude
INK_GRAY_LOW = 25
INK_GRAY_HIGH = 55
SENSOR_NOISE_SD = 4.0            # 8-bit gray levels
# The two gains above are picked, before any effect-recovery number is
# measured, so that dot_prob_per_row stays inside roughly [0.15, 0.85] for
# every score value this repo's ground truth can plausibly produce (see
# test_render_measurement.py::test_dot_probability_never_saturates), which
# keeps the render step an affine, not a clipped/saturating, function of
# score. A saturating (clipped) mapping was tried first and is exactly the
# bug described in the README "Findings" section: clipping bent the
# score-to-response relationship, and OLS soaked up that bend as spurious
# 2-way interactions among factors that share no true interaction.


def render_run_image(score: float, rng: np.random.Generator,
                      height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """Synthesize one halftone defect-simulation frame for a press run whose
    hidden quality score is `score`.

    Higher score means more ink laid down and more row-to-row banding
    (worse print quality). Both the baseline dot probability and the
    banding amplitude are affine functions of `score`; the mapping from
    score to pixels is fixed before any effect-recovery numbers are
    measured (see README "Validation"), and it is not adjusted afterward.
    """
    rows = np.arange(height).reshape(-1, 1)
    baseline_shift = DOT_PROB_GAIN * (score - SCORE_CENTER)
    band_modulation = np.sin(2.0 * np.pi * rows / BAND_PERIOD_ROWS) * (BANDING_GAIN * score)
    dot_prob_per_row = DOT_PROB_CENTER + baseline_shift + band_modulation
    # Safety clip only; for the score range this repo's ground truth
    # actually produces, this never fires (see the test referenced above).
    dot_prob_per_row = np.clip(dot_prob_per_row, 0.03, 0.97)

    draws = rng.random((height, width))
    dot_mask = draws < dot_prob_per_row  # broadcasts (height,1) against (height,width)

    canvas = np.full((height, width), 255.0, dtype=np.float64)
    ink_gray = rng.uniform(INK_GRAY_LOW, INK_GRAY_HIGH, size=(height, width))
    canvas[dot_mask] = ink_gray[dot_mask]

    canvas += rng.normal(0.0, SENSOR_NOISE_SD, size=(height, width))
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    return image


def measure_banding_score(image: np.ndarray) -> float:
    """Measure a banding score from a rendered frame using OpenCV.

    Converts to grayscale (a no-op if already single channel, but keeps the
    function correct if given a 3-channel frame), computes the mean
    intensity of every row with `cv2.reduce`, and returns the standard
    deviation of those row means. Row-to-row banding shows up as periodic
    variation in row-mean intensity; a perfectly uniform halftone has
    row-mean std near 0, and stronger banding drives it up.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    row_means = cv2.reduce(gray, 1, cv2.REDUCE_AVG, dtype=cv2.CV_64F).flatten()
    return float(np.std(row_means))


def measure_density_score(image: np.ndarray) -> float:
    """Measure mean ink density (0 = all white, 255 = all black-inverted)
    from a rendered frame, using OpenCV's mean intensity reduction. Reported
    alongside the banding score as a secondary, independently interpretable
    quality metric; the analysis in this repo fits the banding score."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(255.0 - cv2.mean(gray)[0])


def measured_response_for_run(score: float, rng: np.random.Generator) -> float:
    """Full render-then-measure pipeline used everywhere in this repo: turn
    a hidden score into an image, then measure the banding score back out of
    that image with OpenCV. Returns the banding score alone (see
    `measure_banding_score`); `measure_density_score` is available for
    diagnostic use but is not the fitted response."""
    image = render_run_image(score, rng)
    return measure_banding_score(image)
