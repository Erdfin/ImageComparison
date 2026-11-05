"""Shape / set-overlap metrics computed on edge maps (LaTeX document, section 8).

Both images are reduced to binary edge maps (Canny on luma, σ=1, hysteresis
thresholds at the 80th/95th percentile of each image's own gradient
magnitude) and the maps are compared. Percentile thresholds make the edge
detector invariant to gain/offset changes and ensure that smooth images still
produce edges; completely flat images produce none.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny

from .base import ComparisonMetric, Direction, Image, MetricResult, to_gray


def edge_map(image: Image) -> np.ndarray:
    """Binary (bool) Canny edge map of an RGB image."""
    gray = to_gray(image)
    if float(gray.max() - gray.min()) < 2.0 / 255.0:  # flat: no meaningful edges
        return np.zeros(gray.shape, dtype=bool)
    return canny(gray, sigma=1.0, low_threshold=0.8, high_threshold=0.95, use_quantiles=True)


class _EdgeMetric(ComparisonMetric):
    """Intermediate base: no ``key``, so it is not registered."""

    family = "Shape"
    requires_alignment = True

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        a, b = edge_map(ref), edge_map(test)
        result = self._compare_edges(a, b)
        result.map = (a ^ b).astype(np.float32)
        result.map_label = "edge pixels present in only one image"
        result.details.update(edge_pixels_reference=int(a.sum()), edge_pixels_test=int(b.sum()))
        return result

    @abstractmethod
    def _compare_edges(self, a: np.ndarray, b: np.ndarray) -> MetricResult:
        """Compare two boolean edge maps."""


class _OverlapMetric(_EdgeMetric):
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, 1.0)

    def _compare_edges(self, a: np.ndarray, b: np.ndarray) -> MetricResult:
        inter = int((a & b).sum())
        total_a, total_b = int(a.sum()), int(b.sum())
        if total_a == 0 and total_b == 0:
            return MetricResult(value=1.0, warnings=["no edges in either image: score defaults to 1"])
        return MetricResult(value=self._score(inter, total_a, total_b))

    @staticmethod
    @abstractmethod
    def _score(inter: int, total_a: int, total_b: int) -> float:
        """Overlap score from intersection and set sizes (not both sizes zero)."""

    def _to_similarity(self, value: float) -> float:
        return value


class EdgeJaccard(_OverlapMetric):
    key = "edge_jaccard"
    name = "Edge Jaccard (IoU)"
    citation = "jaccard1912"
    description = "|A∩B| / |A∪B| of the Canny edge maps; drops sharply if edges are merely 1 px apart."

    @staticmethod
    def _score(inter: int, total_a: int, total_b: int) -> float:
        return inter / (total_a + total_b - inter)


class EdgeDice(_OverlapMetric):
    key = "edge_dice"
    name = "Edge Dice"
    citation = "dice1945"
    description = "2|A∩B| / (|A|+|B|) of the Canny edge maps."

    @staticmethod
    def _score(inter: int, total_a: int, total_b: int) -> float:
        return 2 * inter / (total_a + total_b)


class _DistanceMetric(_EdgeMetric):
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, math.inf)
    unit = "px"
    #: fraction of the image diagonal at which the display similarity reaches 0
    _SIM_SCALE = 0.25

    def _compare_edges(self, a: np.ndarray, b: np.ndarray) -> MetricResult:
        diagonal = math.hypot(*a.shape)
        if not a.any() and not b.any():
            return MetricResult(
                value=0.0, similarity=1.0, warnings=["no edges in either image: distance is 0"]
            )
        if not a.any() or not b.any():
            return MetricResult(
                value=math.inf, similarity=0.0, warnings=["edges present in only one image: distance is infinite"]
            )
        # Distance from every pixel to the nearest edge pixel of the other map.
        to_b = distance_transform_edt(~b)
        to_a = distance_transform_edt(~a)
        value = self._distance(to_b[a], to_a[b])
        similarity = 1.0 - min(1.0, value / (self._SIM_SCALE * diagonal))
        return MetricResult(value=float(value), similarity=similarity)

    @staticmethod
    @abstractmethod
    def _distance(a_to_b: np.ndarray, b_to_a: np.ndarray) -> float:
        """Combine nearest-edge distances of A's pixels to B and B's to A."""


class EdgeHausdorff(_DistanceMetric):
    key = "edge_hausdorff"
    name = "Edge Hausdorff"
    citation = "huttenlocher1993"
    description = "Largest distance from an edge pixel of one image to the nearest edge pixel of the other."

    @staticmethod
    def _distance(a_to_b: np.ndarray, b_to_a: np.ndarray) -> float:
        return max(a_to_b.max(), b_to_a.max())


class EdgeChamfer(_DistanceMetric):
    key = "edge_chamfer"
    name = "Edge Chamfer"
    citation = "barrow1977"
    description = "Mean (symmetrised) distance from edge pixels to the nearest edge pixel of the other image."
    _SIM_SCALE = 0.05

    @staticmethod
    def _distance(a_to_b: np.ndarray, b_to_a: np.ndarray) -> float:
        return 0.5 * (a_to_b.mean() + b_to_a.mean())
