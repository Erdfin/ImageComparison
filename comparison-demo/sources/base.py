"""Interface for image-pair sources.

An *image pair source* produces the two images to compare. The UI never knows
about concrete sources: it asks the registry for the available ones, builds its
controls from :meth:`ImagePairSource.params` and calls
:meth:`ImagePairSource.generate`. To add or replace a source, create a module in
the ``sources`` package with a subclass that defines ``key``/``name`` and
implements ``params`` and ``generate``.

Contract for ``generate``
-------------------------
* Returns an :class:`ImagePair` whose images are ``float32`` arrays of shape
  ``(H, W, 3)``, RGB, values in ``[0, 1]``, and with **identical shapes**.
* Takes only keyword arguments named in :meth:`params`; missing ones fall back
  to the declared defaults.
* Must be deterministic for given parameters (use an explicit ``seed`` if the
  source is random), so that sweeps over one parameter are comparable.
* Raises ``ValueError`` with a user-readable message for invalid input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

ParamKind = Literal["int", "float", "choice", "bool", "image", "text"]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Declarative description of one source parameter (drives the UI).

    ``kind="image"`` values are anything :func:`PIL.Image.open` accepts (a path
    or a binary file object).
    """

    name: str
    kind: ParamKind
    default: Any = None
    label: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    @property
    def title(self) -> str:
        return self.label or self.name.replace("_", " ").capitalize()

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("int", "float")

    def validate(self, value: Any) -> Any:
        """Coerce and check ``value``; raises ``ValueError`` if invalid."""
        match self.kind:
            case "int" | "float":
                if value is None:
                    raise ValueError(f"{self.name}: a number is required")
                number = int(round(float(value))) if self.kind == "int" else float(value)
                if self.min is not None and number < self.min:
                    raise ValueError(f"{self.name}: {number} is below the minimum {self.min}")
                if self.max is not None and number > self.max:
                    raise ValueError(f"{self.name}: {number} is above the maximum {self.max}")
                return number
            case "choice":
                if value not in self.choices:
                    raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
                return value
            case "bool":
                return bool(value)
            case "text":
                return "" if value is None else str(value)
            case _:  # "image"
                return value


@dataclass(slots=True)
class ImagePair:
    """Two images of identical shape plus free-form metadata."""

    reference: NDArray[np.float32]
    test: NDArray[np.float32]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference.shape != self.test.shape:
            raise ValueError(
                f"reference {self.reference.shape} and test {self.test.shape} differ in shape"
            )


_REGISTRY: dict[str, type[ImagePairSource]] = {}


class ImagePairSource(ABC):
    """Base class for image-pair sources (see the module docstring)."""

    key: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "key" not in cls.__dict__:
            return
        if not cls.key or not cls.name:
            raise TypeError(f"{cls.__name__} must define key and name")
        if cls.key in _REGISTRY:
            raise ValueError(f"duplicate source key {cls.key!r}")
        _REGISTRY[cls.key] = cls

    @classmethod
    @abstractmethod
    def params(cls) -> list[ParamSpec]:
        """Declare the parameters accepted by :meth:`generate`."""

    @abstractmethod
    def generate(self, **params: Any) -> ImagePair:
        """Produce an image pair; see the module docstring for the contract."""

    # ---------------------------------------------------------- conveniences

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {spec.name: spec.default for spec in cls.params()}

    def resolve_params(self, given: dict[str, Any]) -> dict[str, Any]:
        """Merge ``given`` with defaults and validate every value."""
        specs = {spec.name: spec for spec in self.params()}
        unknown = set(given) - set(specs)
        if unknown:
            raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        return {
            name: spec.validate(given.get(name, spec.default))
            for name, spec in specs.items()
        }


def registered_sources() -> dict[str, type[ImagePairSource]]:
    """Return a copy of the key -> class registry."""
    return dict(_REGISTRY)
