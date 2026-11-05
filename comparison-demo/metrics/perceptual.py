"""Hand-crafted perceptual metrics: FSIM, HaarPSI, VIFp (LaTeX document, section 3).

These are re-implementations from the published descriptions, written in numpy;
numbers can differ slightly from the authors' reference code.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve1d, correlate, uniform_filter

from ._filterbank import phase_congruency
from .base import ComparisonMetric, Direction, Image, MetricResult


def _yiq255(image: Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YIQ channels of an RGB image, each on a 0-255 scale."""
    rgb = image.astype(np.float64) * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (
        0.299 * r + 0.587 * g + 0.114 * b,
        0.596 * r - 0.274 * g - 0.322 * b,
        0.211 * r - 0.523 * g + 0.312 * b,
    )


# ------------------------------------------------------------------- FSIM


class _FSIMBase(ComparisonMetric):
    """Shared computation; subclasses choose whether chroma is used."""

    family = "Feature"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "zhang2011fsim"

    _CHROMA = False
    _T1, _T2, _T3, _T4, _LAMBDA = 0.85, 160.0, 200.0, 200.0, 0.03
    _SCHARR = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64) / 16.0

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        if min(ref.shape[:2]) < 16:
            raise ValueError("image too small for FSIM (needs at least 16px per side)")
        y1, i1, q1 = _yiq255(ref)
        y2, i2, q2 = _yiq255(test)

        f = max(1, int(round(min(y1.shape) / 256)))  # as in the reference code
        if f > 1:
            y1, i1, q1, y2, i2, q2 = (
                uniform_filter(c, f, mode="reflect")[::f, ::f] for c in (y1, i1, q1, y2, i2, q2)
            )

        pc1, pc2 = phase_congruency(y1), phase_congruency(y2)
        g1, g2 = self._gradient(y1), self._gradient(y2)

        s_pc = (2 * pc1 * pc2 + self._T1) / (pc1**2 + pc2**2 + self._T1)
        s_g = (2 * g1 * g2 + self._T2) / (g1**2 + g2**2 + self._T2)
        s_l = s_pc * s_g
        if self._CHROMA:
            s_i = (2 * i1 * i2 + self._T3) / (i1**2 + i2**2 + self._T3)
            s_q = (2 * q1 * q2 + self._T4) / (q1**2 + q2**2 + self._T4)
            s_c = (s_i * s_q).astype(complex) ** self._LAMBDA
            s_l = s_l * s_c.real

        pcm = np.maximum(pc1, pc2)
        if pcm.sum() == 0:
            return MetricResult(
                value=math.nan,
                warnings=["no phase congruency (flat images): FSIM is undefined"],
            )
        return MetricResult(
            value=float((s_l * pcm).sum() / pcm.sum()),
            map=np.clip(1.0 - s_l, 0.0, 1.0).astype(np.float32),
            map_label="1 − local feature similarity" + (" (downsampled)" if f > 1 else ""),
        )

    def _gradient(self, a: np.ndarray) -> np.ndarray:
        gx = correlate(a, self._SCHARR, mode="nearest")
        gy = correlate(a, self._SCHARR.T, mode="nearest")
        return np.hypot(gx, gy)

    def _to_similarity(self, value: float) -> float:
        return value


class FSIM(_FSIMBase):
    key = "fsim"
    name = "FSIM"
    description = (
        "Feature similarity: phase congruency and gradient magnitude on luma, "
        "pooled with phase congruency as weight."
    )


class FSIMc(_FSIMBase):
    key = "fsimc"
    name = "FSIMc"
    description = "FSIM plus a chrominance similarity term computed in YIQ."
    _CHROMA = True


# ---------------------------------------------------------------- HaarPSI


def _haar_kernel(scale: int, horizontal: bool) -> np.ndarray:
    """2-D Haar filter at ``scale`` (1..3); entries are ±2^-scale."""
    length = 2**scale
    g = np.concatenate([-np.ones(length // 2), np.ones(length // 2)]) * 2 ** (-scale / 2)
    h = np.ones(length) * 2 ** (-scale / 2)
    return np.outer(g, h) if horizontal else np.outer(h, g)


class HaarPSI(ComparisonMetric):
    key = "haarpsi"
    name = "HaarPSI"
    family = "Wavelet"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "reisenhofer2018haarpsi"
    description = (
        "Haar-wavelet similarity at two scales (horizontal/vertical) plus "
        "chroma, weighted by the coarser-scale response; colour variant "
        "(HaarPSI_C), on 2×-downsampled images."
    )

    _C = 30.0
    _ALPHA = 4.2

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        h, w = (ref.shape[0] // 2) * 2, (ref.shape[1] // 2) * 2
        if min(h, w) < 16:
            raise ValueError("image too small for HaarPSI (needs at least 16px per side)")

        def prep(img: Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            half = lambda a: 0.25 * (a[0:h:2, 0:w:2] + a[1:h:2, 0:w:2] + a[0:h:2, 1:w:2] + a[1:h:2, 1:w:2])
            return tuple(half(c) for c in _yiq255(img))

        y1, i1, q1 = prep(ref)
        y2, i2, q2 = prep(test)

        def sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (2 * a * b + self._C) / (a * a + b * b + self._C)

        def logistic(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-self._ALPHA * x))

        local, weights = [], []
        for horizontal in (True, False):
            def resp(img: np.ndarray, scale: int) -> np.ndarray:
                return np.abs(correlate(img, _haar_kernel(scale, horizontal), mode="reflect"))

            s = sum(sim(resp(y1, j), resp(y2, j)) for j in (1, 2))
            local.append(logistic(0.5 * s))
            weights.append(np.maximum(resp(y1, 3), resp(y2, 3)))

        mean2 = lambda a: uniform_filter(a, 2, mode="reflect")
        chroma = 0.5 * (sim(np.abs(mean2(i1)), np.abs(mean2(i2))) + sim(np.abs(mean2(q1)), np.abs(mean2(q2))))
        local.append(logistic(chroma))
        weights.append(0.5 * (weights[0] + weights[1]))

        den = float(sum(w_.sum() for w_ in weights))
        if den < 1e-6:  # numerically zero wavelet response: flat images
            return MetricResult(
                value=math.nan, warnings=["no luminance structure (flat images): HaarPSI is undefined"]
            )
        score = sum((l * w_).sum() for l, w_ in zip(local, weights)) / den
        value = float(np.clip(self._inverse_logistic(score), 0.0, 1.0) ** 2)

        mean_local = np.clip(self._inverse_logistic(np.mean(local, axis=0)), 0.0, 1.0) ** 2
        return MetricResult(
            value=value,
            map=np.clip(1.0 - mean_local, 0.0, 1.0).astype(np.float32),
            map_label="1 − mean local HaarPSI similarity (half resolution)",
        )

    def _inverse_logistic(self, y):
        return -np.log(1.0 / np.asarray(y) - 1.0) / self._ALPHA

    def _to_similarity(self, value: float) -> float:
        return value


# ------------------------------------------------------------------- VIFp


def _gauss1d(n: int, sigma: float) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2.0
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _valid_filter(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable filtering, keeping only fully-covered ('valid') pixels."""
    r = len(k) // 2
    out = convolve1d(convolve1d(a, k, axis=0, mode="constant"), k, axis=1, mode="constant")
    return out[r:-r, r:-r]


class PixelDomainVIF(ComparisonMetric):
    key = "vifp"
    name = "VIFp"
    family = "Information"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, math.inf)
    citation = "sheikh2006vif"
    description = (
        "Pixel-domain visual information fidelity: ratio of the information "
        "the test image conveys about the reference to that of the reference "
        "itself, over four scales (luma). Can exceed 1 for contrast enhancement."
    )

    _SIGMA_NSQ = 2.0

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        a = (ref @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).astype(np.float64) * 255.0
        b = (test @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).astype(np.float64) * 255.0
        num = den = 0.0
        for scale in range(1, 5):
            n = 2 ** (4 - scale + 1) + 1
            win = _gauss1d(n, n / 5.0)
            if scale > 1:
                a = _valid_filter(a, win)[::2, ::2]
                b = _valid_filter(b, win)[::2, ::2]
            if min(a.shape) < n + 1:
                raise ValueError("image too small for VIFp (needs roughly 64px per side)")
            mu1, mu2 = _valid_filter(a, win), _valid_filter(b, win)
            s1 = np.maximum(_valid_filter(a * a, win) - mu1 * mu1, 0.0)
            s2 = np.maximum(_valid_filter(b * b, win) - mu2 * mu2, 0.0)
            s12 = _valid_filter(a * b, win) - mu1 * mu2

            g = s12 / (s1 + 1e-10)
            sv = s2 - g * s12
            m = s1 < 1e-10
            g[m], sv[m], s1[m] = 0.0, s2[m], 0.0
            m = s2 < 1e-10
            g[m], sv[m] = 0.0, 0.0
            m = g < 0
            sv[m], g[m] = s2[m], 0.0
            sv[sv <= 1e-10] = 1e-10

            num += float(np.log10(1.0 + g * g * s1 / (sv + self._SIGMA_NSQ)).sum())
            den += float(np.log10(1.0 + s1 / self._SIGMA_NSQ).sum())
        if den == 0:
            return MetricResult(
                value=math.nan, warnings=["reference has no variance (flat image): VIFp is undefined"]
            )
        return MetricResult(value=num / den)

    def _to_similarity(self, value: float) -> float:
        return value
