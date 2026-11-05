"""Abstract base class, result type and shared helpers for comparison metrics.

To add a metric, subclass :class:`ComparisonMetric` in any module inside the
``metrics`` package, set the class attributes (``key``, ``name``, ``family``,
``direction``, ``value_range``) and implement :meth:`ComparisonMetric._compute`.
The class registers itself automatically; no other file needs to change.

Image convention
----------------
Every metric receives ``float32`` arrays of shape ``(H, W, 3)``: RGB, sRGB
encoded, values in ``[0, 1]``. :meth:`ComparisonMetric.compare` converts and
validates whatever the caller passes (``uint8`` or float, grayscale or RGB), so
``_compute`` can rely on that convention.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.color import rgb2lab

Image = NDArray[np.float32]

# Rec. 709 luma weights, applied to the (gamma-encoded) RGB values.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class Direction(Enum):
    """Whether a larger raw value means the images are more similar."""

    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"

    @property
    def arrow(self) -> str:
        return "↑" if self is Direction.HIGHER_IS_BETTER else "↓"


@dataclass(slots=True)
class MetricResult:
    """Outcome of comparing one image pair with one metric.

    Attributes:
        value: Raw metric value in the metric's own units (may be ``nan`` or
            ``inf`` for degenerate inputs).
        similarity: Heuristic score in ``[0, 1]`` (1 = identical) used only to
            display different metrics side by side. ``None`` when undefined.
            It is *not* a calibrated quantity; see each metric's
            ``_to_similarity``.
        map: Optional per-pixel *difference* map, float in ``[0, 1]`` with
            0 = identical. May have a lower resolution than the inputs.
        map_label: Short description of ``map``.
        details: Extra metric-specific values (e.g. estimated shift).
        warnings: Human-readable notes about degenerate inputs or failures.
        elapsed_ms: Wall-clock time of the computation.
    """

    value: float
    similarity: float | None = None
    map: NDArray[np.float32] | None = None
    map_label: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


_REGISTRY: dict[str, type[ComparisonMetric]] = {}


class ComparisonMetric(ABC):
    """Base class for all full-reference image comparison metrics."""

    #: Unique machine-readable identifier, e.g. ``"ssim"``.
    key: ClassVar[str] = ""
    #: Display name, e.g. ``"SSIM"``.
    name: ClassVar[str] = ""
    #: Group used in the UI, e.g. ``"Structural"``.
    family: ClassVar[str] = ""
    direction: ClassVar[Direction]
    #: Nominal range of the raw value; ``inf`` is allowed for open ends.
    value_range: ClassVar[tuple[float, float]]
    #: Unit shown next to the raw value (``"dB"``, ``"bits"``, ...).
    unit: ClassVar[str] = ""
    #: One-sentence explanation shown in the UI.
    description: ClassVar[str] = ""
    #: Key of the entry in ``research/src/references.bib`` (if any).
    citation: ClassVar[str] = ""
    #: True when the metric compares co-located pixels, so a mere shift of the
    #: content changes the score.
    requires_alignment: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only classes that define their own ``key`` are concrete metrics;
        # intermediate abstract helpers are left out of the registry.
        if "key" not in cls.__dict__:
            return
        missing = [
            attr
            for attr in ("key", "name", "family", "direction", "value_range")
            if not getattr(cls, attr, None)
        ]
        if missing:
            raise TypeError(f"{cls.__name__} must define: {', '.join(missing)}")
        if cls.key in _REGISTRY:
            raise ValueError(f"duplicate metric key {cls.key!r}")
        _REGISTRY[cls.key] = cls

    # ------------------------------------------------------------------ API

    def compare(self, reference: ArrayLike, test: ArrayLike) -> MetricResult:
        """Compare ``test`` against ``reference`` and return a result.

        Raises:
            ValueError: if the images are invalid or have different shapes.
        Errors raised inside the metric itself are caught and reported as a
        ``nan`` result with an explanatory warning.
        """
        ref = as_float_rgb(reference)
        tst = as_float_rgb(test)
        if ref.shape != tst.shape:
            raise ValueError(f"image shapes differ: {ref.shape} vs {tst.shape}")

        start = time.perf_counter()
        try:
            with np.errstate(all="ignore"):
                result = self._compute(ref, tst)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            result = MetricResult(
                value=math.nan, warnings=[f"{type(exc).__name__}: {exc}"]
            )
        result.elapsed_ms = (time.perf_counter() - start) * 1000.0

        if result.similarity is None and not math.isnan(result.value):
            sim = self._to_similarity(result.value)
            if sim is not None and not math.isnan(sim):
                result.similarity = float(min(1.0, max(0.0, sim)))
        if math.isnan(result.value) and not result.warnings:
            result.warnings.append("metric is undefined for these images")
        return result

    @abstractmethod
    def _compute(self, ref: Image, test: Image) -> MetricResult:
        """Compute the raw value (and optional map) for validated images."""

    def _to_similarity(self, value: float) -> float | None:
        """Map the raw value to a heuristic ``[0, 1]`` similarity.

        Return ``None`` if no sensible mapping exists. Values are clipped to
        ``[0, 1]`` by the caller.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------- registry


def registered_metrics() -> dict[str, type[ComparisonMetric]]:
    """Return a copy of the key -> class registry."""
    return dict(_REGISTRY)


# ----------------------------------------------------------------- helpers


def as_float_rgb(image: ArrayLike) -> Image:
    """Validate ``image`` and convert it to float32 RGB in ``[0, 1]``.

    Accepts ``uint8`` (scaled by 1/255) or floating point arrays (assumed to be
    in ``[0, 1]`` and clipped), shaped ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``
    or ``(H, W, 4)`` (alpha is dropped).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {arr.shape}")
    channels = arr.shape[2]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif channels == 4:
        arr = arr[..., :3]
    elif channels != 3:
        raise ValueError(f"unsupported number of channels: {channels}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("empty image")

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        raise TypeError(f"expected uint8 or float images, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError("image contains NaN or infinite values")
    return np.clip(arr, 0.0, 1.0)


def to_gray(image: Image) -> NDArray[np.float32]:
    """Rec. 709 luma of an RGB image, shape ``(H, W)``."""
    return (image @ _LUMA).astype(np.float32)


def to_lab(image: Image) -> NDArray[np.float64]:
    """CIELAB (D65) of an sRGB image, shape ``(H, W, 3)``."""
    return rgb2lab(image)


def to_uint8(image: Image) -> NDArray[np.uint8]:
    """Quantise a float image in ``[0, 1]`` to 8 bits."""
    return np.rint(image * 255.0).astype(np.uint8)
