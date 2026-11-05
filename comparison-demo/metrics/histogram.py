"""Histogram / distribution metrics (LaTeX document, section 5).

Histograms are computed per RGB channel (64 bins over ``[0, 1]``) and the
channel scores are averaged. These metrics ignore *where* values occur, so a
shuffled or shifted copy of an image scores as a perfect match.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np

from .base import ComparisonMetric, Direction, Image, MetricResult


class _HistogramMetric(ComparisonMetric):
    """Intermediate base: no ``key``, so it is not registered."""

    family = "Histogram"
    requires_alignment = False
    bins = 64

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        scores = []
        for c in range(3):
            p = self._histogram(ref[..., c])
            q = self._histogram(test[..., c])
            scores.append(self._divergence(p, q))
        return MetricResult(value=float(np.mean(scores)))

    def _histogram(self, channel: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(channel, bins=self.bins, range=(0.0, 1.0))
        return counts / counts.sum()

    @abstractmethod
    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compare two normalised histograms."""


class HistogramIntersection(_HistogramMetric):
    key = "hist_intersection"
    name = "Hist. intersection"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "swain1991"
    description = "Sum of the bin-wise minima of the normalised histograms."

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.minimum(p, q).sum())

    def _to_similarity(self, value: float) -> float:
        return value


class Bhattacharyya(_HistogramMetric):
    key = "bhattacharyya"
    name = "Bhattacharyya"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "bhattacharyya1943"
    description = "Hellinger-type distance sqrt(1 − Σ√(p·q)) between histograms."

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        bc = float(np.sqrt(p * q).sum())
        return math.sqrt(max(0.0, 1.0 - bc))

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class ChiSquare(_HistogramMetric):
    key = "chi_square"
    name = "Chi-square"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    description = "Symmetric chi-square distance ½·Σ(p−q)²/(p+q) between histograms."

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        s = p + q
        mask = s > 0
        return float(0.5 * (((p - q) ** 2)[mask] / s[mask]).sum())

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class EarthMoversDistance(_HistogramMetric):
    key = "emd"
    name = "EMD"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "rubner2000"
    description = (
        "1-D Wasserstein distance between histograms, normalised by the "
        "intensity range; accounts for how far mass has to move."
    )

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        cdf_diff = np.abs(np.cumsum(p) - np.cumsum(q)).sum()
        return float(cdf_diff / (self.bins - 1))

    def _to_similarity(self, value: float) -> float:
        # Typical differences are small; amplify so the display is readable.
        return 1.0 - min(1.0, 4.0 * value)


class JensenShannon(_HistogramMetric):
    key = "js_divergence"
    name = "Jensen–Shannon"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    unit = "bits"
    citation = "lin1991js"
    description = "Symmetric, bounded divergence between histograms (base-2 logarithm)."

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)

        def kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class KullbackLeibler(_HistogramMetric):
    key = "kl_divergence"
    name = "KL divergence"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, math.inf)
    unit = "bits"
    citation = "kullback1951"
    description = (
        "KL(reference ‖ test) between histograms, with ε-smoothing because it "
        "is infinite for empty bins. Asymmetric."
    )

    _EPS = 1e-6

    def _divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = (p + self._EPS) / (p + self._EPS).sum()
        q = (q + self._EPS) / (q + self._EPS).sum()
        return float((p * np.log2(p / q)).sum())

    def _to_similarity(self, value: float) -> float:
        return 2.0 ** (-value)
