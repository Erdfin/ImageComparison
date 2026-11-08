"""FLIP (NVIDIA) via the ``flip-evaluator`` package (LaTeX document, section 3).

If the package is not installed this module defines nothing, so the metric
simply does not appear in the UI.
"""

from __future__ import annotations

import numpy as np

from .base import ComparisonMetric, Direction, Image, MetricResult

try:
    import flip_evaluator as _flip
except ImportError:  # optional dependency
    _flip = None

if _flip is not None:

    class FLIP(ComparisonMetric):
        key = "flip"
        name = "FLIP"
        family = "Perceptual model"
        direction = Direction.LOWER_IS_BETTER
        value_range = (0.0, 1.0)
        citation = "andersson2020flip"
        description = (
            "NVIDIA FLIP (LDR): contrast-sensitivity filtering in an opponent "
            "colour space plus colour and feature differences; mean of the "
            f"error map, assuming {67.0:g} pixels per degree of visual angle."
        )

        #: Viewing condition assumed by the metric (pixels per degree).
        ppd = 67.0

        def _compute(self, ref: Image, test: Image) -> MetricResult:
            error_map, mean_error, _ = _flip.evaluate(
                np.ascontiguousarray(ref),
                np.ascontiguousarray(test),
                "LDR",
                inputsRGB=True,
                applyMagma=False,
                parameters={"ppd": self.ppd},
            )
            return MetricResult(
                value=float(mean_error),
                map=np.clip(np.squeeze(error_map, axis=-1), 0.0, 1.0).astype(np.float32),
                map_label="FLIP error per pixel",
            )

        def _to_similarity(self, value: float) -> float:
            return 1.0 - value
