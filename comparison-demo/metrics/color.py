"""Colour-difference metrics (LaTeX document, section 4)."""

from __future__ import annotations

import math

import numpy as np
from skimage.color import deltaE_cie76, deltaE_ciede2000

from .base import ComparisonMetric, Direction, Image, MetricResult, to_lab


class _DeltaE(ComparisonMetric):
    """Shared code: mean per-pixel colour difference in CIELAB (D65, sRGB input)."""

    family = "Colour"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, math.inf)
    unit = "ΔE"
    _SCALE = 100.0

    @staticmethod
    def _delta(lab_ref: np.ndarray, lab_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        de = self._delta(to_lab(ref), to_lab(test))
        return MetricResult(
            value=float(de.mean()),
            map=np.clip(de / self._SCALE, 0.0, 1.0).astype(np.float32),
            map_label=f"{self.name} per pixel (100 = white), clipped to [0, 1]",
        )

    def _to_similarity(self, value: float) -> float:
        # Black vs. white is ~100 on both scales.
        return 1.0 - value / self._SCALE


class DeltaE76(_DeltaE):
    key = "delta_e_76"
    name = "ΔE76"
    description = "Mean Euclidean distance in CIELAB (CIE 1976)."

    @staticmethod
    def _delta(lab_ref: np.ndarray, lab_test: np.ndarray) -> np.ndarray:
        return deltaE_cie76(lab_ref, lab_test)


class DeltaE2000(_DeltaE):
    key = "delta_e_2000"
    name = "ΔE00"
    citation = "sharma2005"
    description = "Mean CIEDE2000 colour difference; around 1 is near the threshold of noticeability."

    @staticmethod
    def _delta(lab_ref: np.ndarray, lab_test: np.ndarray) -> np.ndarray:
        return deltaE_ciede2000(lab_ref, lab_test)
