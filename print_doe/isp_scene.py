"""Synthetic scene generator for the ISP/detection extension.

A scene is a 64x64 8-bit grayscale frame: textured background noise, plus
a bright circular object on a fraction of frames. This is a small,
honestly synthetic stand-in for a real sensor capture; no physical camera
or ISP hardware was involved. The object flag is read only by
`isp_detector.py`'s training and evaluation code, never by the ISP
transform chain in `isp_pipeline.py`.
"""
from __future__ import annotations

import numpy as np

FRAME_SIZE = 64
OBJECT_RATE = 0.5  # balanced classes; this is a classification eval set, not a prevalence study


def generate_scene(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    base = rng.integers(70, 100, size=(FRAME_SIZE, FRAME_SIZE), dtype=np.int16)
    noise = rng.normal(0, 8, size=(FRAME_SIZE, FRAME_SIZE))
    frame = np.clip(base + noise, 0, 255).astype(np.float32)

    has_object = rng.random() < OBJECT_RATE
    if has_object:
        yy, xx = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]
        cx = rng.integers(14, FRAME_SIZE - 14)
        cy = rng.integers(14, FRAME_SIZE - 14)
        radius = rng.integers(6, 10)
        intensity = rng.integers(190, 220)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist2 <= radius ** 2
        frame[mask] = intensity

    return np.clip(frame, 0, 255).astype(np.uint8), bool(has_object)


def generate_eval_set(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = np.empty((n, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    labels = np.empty(n, dtype=bool)
    for i in range(n):
        frames[i], labels[i] = generate_scene(rng)
    return frames, labels
