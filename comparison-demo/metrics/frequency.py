"""Frequency-domain metrics (LaTeX document, section 7)."""

from __future__ import annotations

import numpy as np

from .base import ComparisonMetric, Direction, Image, MetricResult, to_gray


class PhaseCorrelation(ComparisonMetric):
    key = "phase_correlation"
    name = "Phase correlation"
    family = "Frequency"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "kuglin1975"
    requires_alignment = False
    description = (
        "Peak of the normalised cross-power spectrum (Hann-windowed luma), "
        "relative to the peak for identical images. Robust to translation; "
        "the peak position estimates the shift."
    )

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        a, b = to_gray(ref).astype(np.float64), to_gray(test).astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        h, w = a.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        fa, fb = np.fft.fft2(a * window), np.fft.fft2(b * window)
        cross = fa * np.conj(fb)
        mag = np.abs(cross)

        # Only frequencies where the *reference* has signal take part (smooth
        # images have almost no high-frequency content, whose phase would be
        # numerical noise). The peak is divided by the value an identical
        # image would reach, so that identical images score exactly 1.
        power = np.abs(fa) ** 2
        warnings = []
        if power.max() < 1e-12:
            warnings.append("reference has no spatial structure (constant image): peak is 0")
            keep = np.zeros_like(power, dtype=bool)
        else:
            keep = power > 1e-12 * power.max()
        unit = np.divide(cross, mag, out=np.zeros_like(cross), where=keep & (mag > 0))
        corr = np.fft.ifft2(unit).real
        ideal = keep.sum() / keep.size
        peak = float(corr.max() / ideal) if ideal > 0 else 0.0
        iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
        # corr peaks at -s when the test image is the reference shifted by s.
        dy = -(iy if iy <= h // 2 else iy - h)
        dx = -(ix if ix <= w // 2 else ix - w)
        return MetricResult(
            value=peak,
            details={"estimated_shift_yx": (int(dy), int(dx))},
            warnings=warnings,
        )

    def _to_similarity(self, value: float) -> float:
        return value
