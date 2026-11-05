"""Structural and gradient-based metrics (LaTeX document, section 3)."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate, gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity

from ._filterbank import filter_responses
from .base import ComparisonMetric, Direction, Image, MetricResult, to_gray

_K1, _K2 = 0.01, 0.03
_SIGMA = 1.5
_TRUNCATE = 3.5
_RADIUS = int(_TRUNCATE * _SIGMA + 0.5)  # 5 -> 11x11 Gaussian window
_WIN = 2 * _RADIUS + 1


class StructuralSimilarity(ComparisonMetric):
    key = "ssim"
    name = "SSIM"
    family = "Structural"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (-1.0, 1.0)
    citation = "wang2004ssim"
    description = (
        "Mean local similarity of luminance, contrast and structure "
        "(11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03), averaged over colour channels."
    )

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        mssim, ssim_map = structural_similarity(
            ref,
            test,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=_SIGMA,
            use_sample_covariance=False,
            win_size=_WIN,
            K1=_K1,
            K2=_K2,
            full=True,
        )
        diff_map = np.clip(1.0 - ssim_map.mean(axis=2), 0.0, 1.0).astype(np.float32)
        return MetricResult(
            value=float(mssim), map=diff_map, map_label="1 − local SSIM"
        )

    def _to_similarity(self, value: float) -> float:
        return value


def _ssim_terms(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean luminance and contrast-structure terms over the valid region."""
    c1, c2 = _K1**2, _K2**2

    def blur(a: np.ndarray) -> np.ndarray:
        return gaussian_filter(a, _SIGMA, truncate=_TRUNCATE, mode="reflect")

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    lum = (2 * mx * my + c1) / (mx * mx + my * my + c1)
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    r = _RADIUS
    return float(lum[r:-r, r:-r].mean()), float(cs[r:-r, r:-r].mean())


class MultiScaleSSIM(ComparisonMetric):
    key = "ms_ssim"
    name = "MS-SSIM"
    family = "Structural"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "wang2003msssim"
    description = (
        "SSIM evaluated over up to five dyadic scales on luma; fewer scales "
        "are used (with renormalised weights) when the image is small."
    )

    _WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        x = ref @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        y = test @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        x, y = x.astype(np.float64), y.astype(np.float64)

        min_side = min(x.shape)
        if min_side < _WIN:
            raise ValueError(f"image too small: needs at least {_WIN}px per side")
        scales = min(len(self._WEIGHTS), int(math.log2(min_side / _WIN)) + 1)
        weights = self._WEIGHTS[:scales] / self._WEIGHTS[:scales].sum()
        warnings = []
        if scales < len(self._WEIGHTS):
            warnings.append(f"small image: only {scales} of 5 scales used")

        value = 1.0
        for j in range(scales):
            lum, cs = _ssim_terms(x, y)
            cs = max(cs, 0.0)
            if j == scales - 1:
                value *= (max(lum, 0.0) * cs) ** weights[j]
            else:
                value *= cs ** weights[j]
                h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
                x = _downsample(x[:h, :w])
                y = _downsample(y[:h, :w])
        return MetricResult(value=float(value), warnings=warnings)

    def _to_similarity(self, value: float) -> float:
        return value


def _downsample(a: np.ndarray) -> np.ndarray:
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


class GradientMagnitudeSimilarityDeviation(ComparisonMetric):
    key = "gmsd"
    name = "GMSD"
    family = "Structural"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, math.inf)
    citation = "xue2014gmsd"
    description = (
        "Standard deviation of the gradient magnitude similarity map "
        "(Prewitt gradients on 2×-downsampled luma, c=170 on a 0-255 scale)."
    )

    _C = 170.0
    _HX = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        x = to_gray(ref).astype(np.float64) * 255.0
        y = to_gray(test).astype(np.float64) * 255.0
        h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
        if h < 4 or w < 4:
            raise ValueError("image too small for GMSD (needs at least 8x8 px)")
        x, y = _downsample(x[:h, :w]), _downsample(y[:h, :w])

        def grad_mag(a: np.ndarray) -> np.ndarray:
            gx = correlate(a, self._HX, mode="nearest")
            gy = correlate(a, self._HX.T, mode="nearest")
            return np.hypot(gx, gy)

        mr, md = grad_mag(x), grad_mag(y)
        gms = (2 * mr * md + self._C) / (mr**2 + md**2 + self._C)
        return MetricResult(
            value=float(gms.std()),
            map=np.clip(1.0 - gms, 0.0, 1.0).astype(np.float32),
            map_label="1 − gradient magnitude similarity (half resolution)",
        )

    def _to_similarity(self, value: float) -> float:
        # Deviations above ~0.35 are already very poor on standard databases.
        return 1.0 - value / 0.35


# ------------------------------------------------------------------ UQI


class UniversalQualityIndex(ComparisonMetric):
    key = "uqi"
    name = "UQI"
    family = "Structural"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (-1.0, 1.0)
    citation = "wang2002uqi"
    description = (
        "Correlation × luminance × contrast similarity in 8×8 sliding windows "
        "on luma (SSIM without stabilising constants)."
    )

    _WIN = 8

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        x = to_gray(ref).astype(np.float64)
        y = to_gray(test).astype(np.float64)
        if min(x.shape) < self._WIN:
            raise ValueError(f"image too small: needs at least {self._WIN}px per side")

        def mean(a: np.ndarray) -> np.ndarray:
            return uniform_filter(a, self._WIN, mode="reflect")

        mx, my = mean(x), mean(y)
        vx = np.maximum(mean(x * x) - mx * mx, 0.0)
        vy = np.maximum(mean(y * y) - my * my, 0.0)
        cxy = mean(x * y) - mx * my

        d_var = vx + vy
        d_mean = mx * mx + my * my
        var_ok, mean_ok = d_var > 1e-10, d_mean > 1e-10
        q = np.ones_like(x)
        both = var_ok & mean_ok
        q[both] = 4 * cxy[both] * mx[both] * my[both] / (d_var[both] * d_mean[both])
        only_mean = ~var_ok & mean_ok
        q[only_mean] = 2 * mx[only_mean] * my[only_mean] / d_mean[only_mean]

        r = self._WIN // 2
        return MetricResult(
            value=float(q[r:-r, r:-r].mean()),
            map=np.clip(1.0 - q, 0.0, 1.0).astype(np.float32),
            map_label="1 − local UQI",
        )

    def _to_similarity(self, value: float) -> float:
        return value


# -------------------------------------------------------------- CW-SSIM


class ComplexWaveletSSIM(ComparisonMetric):
    key = "cw_ssim"
    name = "CW-SSIM"
    family = "Structural"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "sampat2009cwssim"
    requires_alignment = False
    description = (
        "SSIM-like index on complex band-pass coefficients (4 scales × 4 "
        "orientations of a log-Gabor bank, 7×7 windows): tolerant to small "
        "shifts. Uses a log-Gabor bank rather than the steerable pyramid of "
        "the original paper, so values differ from reference implementations."
    )

    _WIN = 7
    _K = 0.01

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        x = to_gray(ref).astype(np.float64) * 255.0
        y = to_gray(test).astype(np.float64) * 255.0
        if min(x.shape) < 16:
            raise ValueError("image too small for CW-SSIM (needs at least 16px per side)")
        cx, cy = filter_responses(x), filter_responses(y)

        n = self._WIN**2

        def wsum(a: np.ndarray) -> np.ndarray:
            return uniform_filter(a, size=(1, 1, self._WIN, self._WIN), mode="reflect") * n

        ax, ay = np.abs(cx), np.abs(cy)
        cross = cx * np.conj(cy)
        s_mag = wsum(ax * ay)
        s_cross = np.abs(wsum(cross.real) + 1j * wsum(cross.imag))
        k = self._K
        mag_term = (2 * s_mag + k) / (wsum(ax**2) + wsum(ay**2) + k)
        phase_term = (2 * s_cross + k) / (2 * s_mag + k)
        q = (mag_term * phase_term).reshape(-1, *x.shape)  # bands × H × W

        margin = 8 if min(x.shape) > 48 else 0
        inner = q[:, margin : x.shape[0] - margin, margin : x.shape[1] - margin]
        return MetricResult(
            value=float(inner.mean()),
            map=np.clip(1.0 - q.mean(axis=0), 0.0, 1.0).astype(np.float32),
            map_label="1 − mean local CW-SSIM over bands",
        )

    def _to_similarity(self, value: float) -> float:
        return value
