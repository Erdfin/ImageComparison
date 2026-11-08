"""Pixel-wise and correlation metrics (LaTeX document, section 2)."""

from __future__ import annotations

import math

import numpy as np

from .base import ComparisonMetric, Direction, Image, MetricResult


class MeanAbsoluteError(ComparisonMetric):
    key = "mae"
    name = "MAE"
    family = "Pixel-wise"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    description = "Mean absolute difference between co-located pixel values (images scaled to [0, 1])."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        diff = np.abs(ref - test)
        return MetricResult(
            value=float(diff.mean()),
            map=diff.mean(axis=2),
            map_label="|reference − test|",
        )

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class MeanSquaredError(ComparisonMetric):
    key = "mse"
    name = "MSE"
    family = "Pixel-wise"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "wang2009mse"
    description = "Mean squared difference between co-located pixel values (images scaled to [0, 1])."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        sq = (ref - test) ** 2
        return MetricResult(
            value=float(sq.mean()),
            map=sq.mean(axis=2),
            map_label="(reference − test)²",
        )

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class RootMeanSquaredError(ComparisonMetric):
    key = "rmse"
    name = "RMSE"
    family = "Pixel-wise"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)
    citation = "wang2009mse"
    description = "Square root of the MSE, in the same units as the pixel values."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        return MetricResult(value=float(math.sqrt(((ref - test) ** 2).mean())))

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value


class PeakSignalToNoiseRatio(ComparisonMetric):
    key = "psnr"
    name = "PSNR"
    family = "Pixel-wise"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (0.0, math.inf)
    unit = "dB"
    citation = "huynh2008"
    description = "10·log10(1/MSE) in decibels; infinite for identical images."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        mse = float(((ref - test) ** 2).mean())
        if mse == 0.0:
            return MetricResult(
                value=math.inf, warnings=["identical images: PSNR is infinite"]
            )
        return MetricResult(value=10.0 * math.log10(1.0 / mse))

    def _to_similarity(self, value: float) -> float:
        # 0 dB -> 0, 50 dB or more -> 1 (heuristic display scale).
        return value / 50.0


class ZeroNormalizedCrossCorrelation(ComparisonMetric):
    key = "zncc"
    name = "ZNCC"
    family = "Pixel-wise"
    direction = Direction.HIGHER_IS_BETTER
    value_range = (-1.0, 1.0)
    citation = "lewis1995"
    description = "Pearson correlation of all pixel values; invariant to gain and offset changes."

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        a = ref.ravel().astype(np.float64)
        b = test.ravel().astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if denom == 0.0:
            return MetricResult(
                value=math.nan,
                warnings=["constant image: correlation is undefined"],
            )
        return MetricResult(value=float((a * b).sum() / denom))

    def _to_similarity(self, value: float) -> float:
        return (value + 1.0) / 2.0
