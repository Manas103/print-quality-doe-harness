"""Render a synthetic membrane-inspection frame from a hidden contamination
score, then measure a defect count back out of the rendered pixels with
OpenCV.

This is the filtration extension's honest-OpenCV-use path, parallel to
`render.py` but deliberately a different measurement shape: instead of a
banding std-dev read out of a halftone frame, this module places a Poisson
number of particle blobs on a simulated membrane image and counts them back
out with `cv2.connectedComponentsWithStats`, the way a real automated
particle-inspection camera would (threshold, then count connected regions,
not divide by a known ground-truth count).

`render_filtration_frame` never hands its caller a formula, it hands them an
8-bit grayscale image. `measure_defect_count` never sees the score or the
true particle count that produced the image; it only sees pixel data. The
measured value that the filtration analysis fits is the OUTPUT of
`measure_defect_count`, not the `score` argument used to render it, and it
is a genuinely noisy count: overlapping particles can merge into a single
connected component, so the measured count is a real, imperfect readout of
the true count, not the true count itself.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

SCORE_CENTER = 40.0              # matches filtration_ground_truth.INTERCEPT; keeps the rate mapping centered
PARTICLE_RATE_CENTER = 40.0      # expected particle count at SCORE_CENTER
PARTICLE_RATE_GAIN = 0.5         # score units (relative to SCORE_CENTER) -> expected-particle-count shift
# The center and gain above are picked, before any effect-recovery number is
# measured, so that the expected particle count stays comfortably positive
# (never clipped to 0) for every score value this repo's filtration ground
# truth can plausibly produce (see
# test_filtration_response.py::test_expected_particle_rate_never_clips_for_realistic_score_range).
# render.py's Findings section documents a saturating (clipped) mapping as a
# real bug that bent an earlier score-to-response relationship and produced
# spurious interactions; this module keeps the same affine, wide-margin
# discipline for exactly that reason.

PARTICLE_RADIUS_LOW = 1
PARTICLE_RADIUS_HIGH = 2
PARTICLE_GRAY_LOW = 15
PARTICLE_GRAY_HIGH = 55
BACKGROUND_GRAY = 255
SENSOR_NOISE_SD = 5.0             # 8-bit gray levels
BINARY_THRESHOLD = 210            # pixels darker than this are read as "particle" by the inspection step


def expected_particle_rate(score: float) -> float:
    """Affine, non-negative-clamped map from hidden score to the expected
    (Poisson mean) number of particles deposited on the membrane for one
    run. Clamped at 0 only as a mathematical safety net; for the score range
    this repo's ground truth actually produces, the clamp never fires (see
    the test referenced above)."""
    return max(0.0, PARTICLE_RATE_CENTER + PARTICLE_RATE_GAIN * (score - SCORE_CENTER))


def score_for_expected_rate(rate: float) -> float:
    """Inverse of `expected_particle_rate`: the score that produces a given
    expected particle count, for the affine (non-clamped) region of the
    mapping. Used by the power-curve analysis (claim 4) to construct a
    treatment condition with a specified true fractional change in expected
    particle count relative to a baseline score, without ever computing the
    measured response as algebra: the returned score is still run through
    the full noisy render-then-count pipeline like every other run."""
    return SCORE_CENTER + (rate - PARTICLE_RATE_CENTER) / PARTICLE_RATE_GAIN


def render_filtration_frame(score: float, rng: np.random.Generator,
                             height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH):
    """Synthesize one membrane-inspection frame for a filtration-bench run
    whose hidden contamination score is `score`.

    A Poisson-distributed number of particles (mean = `expected_particle_rate(score)`)
    is drawn, then each particle is placed at a uniformly random position with
    a randomized radius and randomized ink gray level, on an otherwise blank
    background, with sensor noise added over the whole frame. Higher score
    means more particles deposited on average; particles that happen to
    overlap merge visually, exactly the way real inspected particulate does.

    Returns `(image, n_particles_placed)`; `n_particles_placed` is the true
    number of particles drawn (debug-only, never fed to the analysis code,
    exactly like `simulate.py`'s `_true_score_debug_only` column).
    """
    rate = expected_particle_rate(score)
    n_particles = int(rng.poisson(rate))

    canvas = np.full((height, width), float(BACKGROUND_GRAY), dtype=np.float64)

    if n_particles > 0:
        xs = rng.integers(0, width, size=n_particles)
        ys = rng.integers(0, height, size=n_particles)
        radii = rng.integers(PARTICLE_RADIUS_LOW, PARTICLE_RADIUS_HIGH + 1, size=n_particles)
        grays = rng.uniform(PARTICLE_GRAY_LOW, PARTICLE_GRAY_HIGH, size=n_particles)
        for x, y, r, g in zip(xs, ys, radii, grays):
            cv2.circle(canvas, (int(x), int(y)), int(r), float(g), thickness=-1)

    canvas += rng.normal(0.0, SENSOR_NOISE_SD, size=(height, width))
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    return image, n_particles


def measure_defect_count(image: np.ndarray) -> float:
    """Measure a defect (particle) count from a rendered frame using OpenCV.

    Thresholds the frame (pixels darker than `BINARY_THRESHOLD` are "ink",
    the rest is background), then counts 8-connected components with
    `cv2.connectedComponentsWithStats`, the standard OpenCV blob-counting
    primitive. The background label is excluded. This is a real counting
    readout, not the true particle count: two particles whose circles
    overlap are counted as one connected component, and sensor noise can
    occasionally split or merge blobs near the threshold, exactly the kind
    of imperfection a real automated inspection camera has.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    num_labels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return float(num_labels - 1)  # exclude the background label


def measured_response_for_filtration_run(score: float, rng: np.random.Generator):
    """Full render-then-measure pipeline used everywhere in the filtration
    extension: turn a hidden score into a membrane-inspection image, then
    measure a defect count back out of that image with OpenCV.

    Returns `(measured_defect_count, n_particles_placed)`; the second value
    is debug-only (see `render_filtration_frame`)."""
    image, n_particles = render_filtration_frame(score, rng)
    return measure_defect_count(image), n_particles
