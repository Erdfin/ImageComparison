"""Perceptual-hash metrics (LaTeX document, section 9).

The score is the Hamming distance between 64-bit hashes (via ``imagehash``).
"""

from __future__ import annotations

from abc import abstractmethod

import imagehash
from PIL import Image as PILImage

from .base import ComparisonMetric, Direction, Image, MetricResult, to_uint8


class _HashMetric(ComparisonMetric):
    """Intermediate base: no ``key``, so it is not registered."""

    family = "Hash"
    direction = Direction.LOWER_IS_BETTER
    requires_alignment = False
    citation = "zauner2010"
    unit = "bits"
    hash_size = 8

    def __init_subclass__(cls, **kwargs):
        # Must be set before the base class validates the concrete metric.
        if "key" in cls.__dict__:
            cls.value_range = (0.0, float(cls.hash_size**2))
        super().__init_subclass__(**kwargs)

    def _compute(self, ref: Image, test: Image) -> MetricResult:
        h1 = self._hash(PILImage.fromarray(to_uint8(ref)))
        h2 = self._hash(PILImage.fromarray(to_uint8(test)))
        distance = int(h1 - h2)
        return MetricResult(
            value=float(distance),
            details={"hash_reference": str(h1), "hash_test": str(h2)},
        )

    @abstractmethod
    def _hash(self, image: PILImage.Image) -> imagehash.ImageHash:
        """Hash a PIL image."""

    def _to_similarity(self, value: float) -> float:
        return 1.0 - value / self.hash_size**2


class AverageHash(_HashMetric):
    key = "ahash"
    name = "aHash"
    description = "8×8 grey thumbnail thresholded at its mean; Hamming distance of the 64-bit hashes."

    def _hash(self, image):
        return imagehash.average_hash(image, hash_size=self.hash_size)


class DifferenceHash(_HashMetric):
    key = "dhash"
    name = "dHash"
    description = "Sign of horizontal gradients of a 9×8 grey thumbnail; Hamming distance of the 64-bit hashes."

    def _hash(self, image):
        return imagehash.dhash(image, hash_size=self.hash_size)


class PerceptualHash(_HashMetric):
    key = "phash"
    name = "pHash"
    description = "Low-frequency DCT coefficients of a 32×32 grey thumbnail thresholded at the median."

    def _hash(self, image):
        return imagehash.phash(image, hash_size=self.hash_size)


class WaveletHash(_HashMetric):
    key = "whash"
    name = "wHash"
    description = "Like pHash but using the low-frequency band of a Haar wavelet transform."

    def _hash(self, image):
        return imagehash.whash(image, hash_size=self.hash_size)
