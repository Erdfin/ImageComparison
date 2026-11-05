"""Information-theoretic metrics (LaTeX document, section 6)."""

from __future__ import annotations

import math

import numpy as np

from .base import ComparisonMetric, Direction, Image, MetricResult, to_gray

_BINS = 64


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _entropies(ref: Image, test: Image) -> tuple[float, float, float]:
    """H(X), H(Y), H(X, Y) in bits from the joint grey-level histogram."""
    joint, _, _ = np.histogram2d(
        to_gray(ref).ravel(),
        to_gray(test).ravel(),
        bins=_BINS,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    pxy = joint / joint.sum()
    return _entropy(pxy.sum(axis=1)), _entropy(pxy.sum(axis=0)), _entropy(pxy.ravel())


class MutualInformation(ComparisonMetric):
    key = "mi"
    name = "MI"
    family = "Information"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, math.log2(_BINS))
    unit = "bits"
    citation = "viola1997"
    description = (
        "Mutual information of co-located grey levels (64 bins). Needs no "
        "functional relation between the images, so even an inverted image scores high."
    )

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        hx, hy, hxy = _entropies(ref, test)
        mi = hx + hy - hxy
        warnings = []
        similarity = None
        if max(hx, hy) > 0:
            similarity = mi / max(hx, hy)
        else:
            warnings.append("both images are constant: mutual information is 0")
        return MetricResult(
            value=max(0.0, mi),
            similarity=similarity,
            details={"H_ref": hx, "H_test": hy, "H_joint": hxy},
            warnings=warnings,
        )


class NormalizedMutualInformation(ComparisonMetric):
    key = "nmi"
    name = "NMI"
    family = "Information"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (1.0, 2.0)
    citation = "studholme1999"
    description = "(H(X)+H(Y)) / H(X,Y): 2 when one image fully predicts the other, 1 when independent."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        hx, hy, hxy = _entropies(ref, test)
        if hxy == 0:
            return MetricResult(
                value=math.nan, warnings=["both images are constant: NMI is undefined"]
            )
        return MetricResult(value=(hx + hy) / hxy)

    def _to_similarity(self, value: float) -> float:
        return value - 1.0
