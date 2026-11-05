"""Image-pair sources. Importing this package registers every source module."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import ImagePair, ImagePairSource, ParamSpec, registered_sources


def load_builtin_sources() -> None:
    """Import every module of the ``sources`` package so sources register."""
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if info.name != "base" and not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")


def all_sources() -> list[type[ImagePairSource]]:
    """All registered source classes, sorted by display name."""
    return sorted(registered_sources().values(), key=lambda c: c.name)


def get_source(key: str) -> type[ImagePairSource]:
    try:
        return registered_sources()[key]
    except KeyError:
        known = ", ".join(sorted(registered_sources()))
        raise KeyError(f"unknown source {key!r}; known sources: {known}") from None


load_builtin_sources()

__all__ = [
    "ImagePair",
    "ImagePairSource",
    "ParamSpec",
    "all_sources",
    "get_source",
]
