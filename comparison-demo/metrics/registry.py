"""Lookup and discovery of registered metrics."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import ComparisonMetric, registered_metrics

_SKIP = {"base", "registry"}


def load_builtin_metrics() -> None:
    """Import every module of the ``metrics`` package so metrics register."""
    package_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name not in _SKIP and not info.name.startswith("_"):
            importlib.import_module(f"{__package__}.{info.name}")


def all_metrics() -> list[type[ComparisonMetric]]:
    """All registered metric classes, sorted by family then name."""
    return sorted(registered_metrics().values(), key=lambda c: (c.family, c.name))


def get_metric(key: str) -> type[ComparisonMetric]:
    """Return the metric class registered under ``key``."""
    try:
        return registered_metrics()[key]
    except KeyError:
        known = ", ".join(sorted(registered_metrics()))
        raise KeyError(f"unknown metric {key!r}; known metrics: {known}") from None
