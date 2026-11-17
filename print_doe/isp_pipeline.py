"""Five real OpenCV image-signal-processing steps, each toggled between a
"low" (-1) and "high" (+1) setting by the factorial design in
`isp_design.py`, applied in a fixed pipeline order regardless of which
factors are listed where in `FACTOR_NAMES`: demosaic, denoise, white
balance, tone curve, sharpening. This mirrors a real ISP's ordering
(demosaic early, tone mapping and sharpening late) and is fixed before any
accuracy number in this extension is measured.

Every step is pixel operations on the actual rendered frame (OpenCV calls
or direct numpy arithmetic on the pixel array), never a formula applied to
a hidden ground-truth score; the detector only ever sees processed pixels.
"""
from __future__ import annotations

import cv2
import numpy as np


def _demosaic(image: np.ndarray, level: int) -> np.ndarray:
    """Low: untouched. High: a Bayer-mosaic-style artifact, simulated by
    downsampling 2x with nearest-neighbor then upsampling back 2x, which
    reproduces the blocky aliasing a poor demosaic interpolator leaves
    behind."""
    if level < 0:
        return image
    h, w = image.shape
    small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _denoise(image: np.ndarray, level: int) -> np.ndarray:
    """Low: untouched (sensor noise kept). High: a strong Gaussian blur, the
    standard cheap denoise, which also blurs the object's edges."""
    if level < 0:
        return image
    return cv2.GaussianBlur(image, (7, 7), sigmaX=2.0)


def _white_balance(image: np.ndarray, level: int) -> np.ndarray:
    """Low: untouched. High: a global gain shift (a simplified grayscale
    stand-in for a channel-gain white-balance correction)."""
    if level < 0:
        return image
    out = image.astype(np.float32) * 0.82
    return np.clip(out, 0, 255).astype(np.uint8)


def _tone_curve(image: np.ndarray, level: int) -> np.ndarray:
    """Low: untouched (linear). High: a gamma curve (gamma=1.8) that
    compresses the highlight range, reducing the contrast between a bright
    object and a bright background."""
    if level < 0:
        return image
    normalized = image.astype(np.float32) / 255.0
    curved = np.power(normalized, 1.8)
    return np.clip(curved * 255.0, 0, 255).astype(np.uint8)


def _sharpening(image: np.ndarray, level: int) -> np.ndarray:
    """Low: untouched. High: unsharp masking (image + 1.0 * (image - blur))."""
    if level < 0:
        return image
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)
    sharpened = image.astype(np.float32) + 1.0 * (image.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_isp_chain(image: np.ndarray, settings: dict[str, int]) -> np.ndarray:
    """settings maps factor name -> coded level (-1 or +1). Applies in the
    fixed real-ISP-like order: demosaic, denoise, white_balance,
    tone_curve, sharpening."""
    out = image
    out = _demosaic(out, settings["demosaic"])
    out = _denoise(out, settings["denoise"])
    out = _white_balance(out, settings["white_balance"])
    out = _tone_curve(out, settings["tone_curve"])
    out = _sharpening(out, settings["sharpening"])
    return out
