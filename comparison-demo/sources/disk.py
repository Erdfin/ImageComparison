"""Image pairs read from disk (or uploaded file objects)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .base import ImagePair, ImagePairSource, ParamSpec

_SIZE_POLICIES = ("resize_test", "center_crop", "error")


def _load(src: Any, label: str) -> Image.Image:
    """Open an image from a path or binary file object as RGB."""
    if src is None or (isinstance(src, str) and not src.strip()):
        raise ValueError(f"{label} image is missing")
    if hasattr(src, "seek"):
        src.seek(0)
    elif isinstance(src, (str, Path)):
        path = Path(str(src).strip().strip('"'))
        if not path.is_file():
            raise ValueError(f"{label} image not found: {path}")
        src = path
    try:
        img = Image.open(src)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{label} image could not be read: {exc}") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def _label(src: Any) -> str:
    if isinstance(src, (str, Path)):
        return Path(str(src).strip().strip('"')).name
    return str(getattr(src, "name", "<uploaded file>"))


class DiskSource(ImagePairSource):
    key = "disk"
    name = "Images from disk"
    description = "Compare two image files. Differently sized images are matched by the chosen policy."

    @classmethod
    def params(cls) -> list[ParamSpec]:
        return [
            ParamSpec("reference", "image", label="Reference image"),
            ParamSpec("test", "image", label="Test image"),
            ParamSpec(
                "size_policy",
                "choice",
                default="resize_test",
                choices=_SIZE_POLICIES,
                label="If sizes differ",
                help=(
                    "resize_test: resize the test image to the reference size; "
                    "center_crop: crop both to their common centre; "
                    "error: refuse to compare."
                ),
            ),
            ParamSpec(
                "max_side",
                "int",
                default=1024,
                min=0,
                max=8192,
                step=64,
                label="Max side (px, 0 = off)",
                help="Downscale large images (both by the same factor) to keep metrics fast.",
            ),
        ]

    def generate(self, **params: Any) -> ImagePair:
        p = self.resolve_params(params)
        ref = _load(p["reference"], "reference")
        test = _load(p["test"], "test")
        original_sizes = {"reference": ref.size, "test": test.size}

        if ref.size != test.size:
            match p["size_policy"]:
                case "resize_test":
                    test = test.resize(ref.size, Image.Resampling.LANCZOS)
                case "center_crop":
                    ref, test = _center_crop_pair(ref, test)
                case _:
                    raise ValueError(
                        f"image sizes differ: reference {ref.size}, test {test.size}"
                    )

        max_side = p["max_side"]
        if max_side and max(ref.size) > max_side:
            scale = max_side / max(ref.size)
            new_size = (max(1, round(ref.width * scale)), max(1, round(ref.height * scale)))
            ref = ref.resize(new_size, Image.Resampling.LANCZOS)
            test = test.resize(new_size, Image.Resampling.LANCZOS)

        return ImagePair(
            reference=np.asarray(ref, dtype=np.float32) / 255.0,
            test=np.asarray(test, dtype=np.float32) / 255.0,
            meta={
                "reference": _label(p["reference"]),
                "test": _label(p["test"]),
                "original_sizes": original_sizes,
                "size": ref.size,
                "size_policy": p["size_policy"],
            },
        )


def _center_crop_pair(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    w, h = min(a.width, b.width), min(a.height, b.height)

    def crop(img: Image.Image) -> Image.Image:
        left, top = (img.width - w) // 2, (img.height - h) // 2
        return img.crop((left, top, left + w, top + h))

    return crop(a), crop(b)
