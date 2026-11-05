"""Randomly generated image pairs, to probe where metrics break down.

A pair is built in three independent steps, each with its own random stream
derived from ``seed`` (so changing the distortion never changes the base
image, and sweeping ``strength`` reuses the same noise draws):

1. a *base image* of a chosen ``pattern`` (the reference);
2. a *distortion* of that image with a ``strength`` in ``[0, 1]`` (the test);
3. both are quantised to 8 bits, like real images.

``strength = 0`` always gives a test image identical to the reference.

To add a pattern or distortion, register a function in ``PATTERNS`` /
``DISTORTIONS`` below; it shows up in the UI automatically.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import hsv2rgb, rgb2hsv

from .base import ImagePair, ImagePairSource, ParamSpec

Array = np.ndarray
Pattern = Callable[[np.random.Generator, int, int], Array]
# (image, strength, rng, make_independent_image) -> distorted image
Distortion = Callable[[Array, float, np.random.Generator, Callable[[], Array]], Array]


# ------------------------------------------------------------------ patterns


def _normalise(a: Array) -> Array:
    lo, hi = a.min(axis=(0, 1), keepdims=True), a.max(axis=(0, 1), keepdims=True)
    return (a - lo) / np.maximum(hi - lo, 1e-9)


def _smooth_noise(rng: np.random.Generator, h: int, w: int, sigma: float) -> Array:
    noise = rng.random((h, w, 3))
    return _normalise(ndimage.gaussian_filter(noise, (sigma, sigma, 0), mode="wrap"))


def _white_noise(rng, h, w):
    return rng.random((h, w, 3))


def _cloudy(rng, h, w):
    return _smooth_noise(rng, h, w, max(h, w) / 32)


def _texture(rng, h, w):
    """Fractal-like noise: several octaves of smooth noise."""
    out = np.zeros((h, w, 3))
    for octave in range(5):
        out += _smooth_noise(rng, h, w, max(h, w) / (4 * 2**octave)) / 2**octave
    return _normalise(out)


def _gradient(rng, h, w):
    c0, c1 = rng.random(3), rng.random(3)
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    t = xx * np.cos(angle) + yy * np.sin(angle)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    return c0 * (1 - t[..., None]) + c1 * t[..., None]


def _checkerboard(rng, h, w):
    cell = int(rng.integers(8, 33))
    c0, c1 = rng.random(3), rng.random(3)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // cell + xx // cell) % 2).astype(bool)
    return np.where(mask[..., None], c0, c1)


def _shapes(rng, h, w):
    img = np.tile(rng.random(3), (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(int(rng.integers(8, 16))):
        colour = rng.random(3)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        size = rng.uniform(0.05, 0.25) * min(h, w)
        if rng.random() < 0.5:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= size**2
        else:
            mask = (np.abs(yy - cy) <= size) & (np.abs(xx - cx) <= size * rng.uniform(0.5, 1.5))
        img[mask] = colour
    return img


def _flat(rng, h, w):
    return np.tile(rng.random(3), (h, w, 1))


PATTERNS: dict[str, Pattern] = {
    "texture": _texture,
    "cloudy": _cloudy,
    "shapes": _shapes,
    "gradient": _gradient,
    "checkerboard": _checkerboard,
    "white_noise": _white_noise,
    "flat_colour": _flat,
}


# --------------------------------------------------------------- distortions


def _gaussian_noise(x, s, rng, _other):
    return x + rng.normal(0.0, 0.5 * s, x.shape)


def _salt_pepper(x, s, rng, _other):
    u = rng.random(x.shape[:2])
    out = x.copy()
    out[u < 0.25 * s] = 0.0
    out[u > 1.0 - 0.25 * s] = 1.0
    return out


def _blur(x, s, rng, _other):
    sigma = 8.0 * s
    return ndimage.gaussian_filter(x, (sigma, sigma, 0), mode="reflect") if sigma > 0 else x


def _brightness(x, s, rng, _other):
    return x + 0.5 * s


def _contrast(x, s, rng, _other):
    return 0.5 + (x - 0.5) * (1.0 - s)


def _gamma(x, s, rng, _other):
    return np.clip(x, 0, 1) ** (1.0 + 3.0 * s)


def _hue_shift(x, s, rng, _other):
    if s == 0:
        return x
    hsv = rgb2hsv(np.clip(x, 0, 1))
    hsv[..., 0] = (hsv[..., 0] + 0.5 * s) % 1.0
    return hsv2rgb(hsv)


def _translate(x, s, rng, _other):
    return np.roll(x, int(round(0.25 * s * x.shape[1])), axis=1)


def _rotate(x, s, rng, _other):
    if s == 0:
        return x
    return ndimage.rotate(x, 45.0 * s, axes=(0, 1), reshape=False, order=1, mode="reflect")


def _zoom(x, s, rng, _other):
    if s == 0:
        return x
    h, w = x.shape[:2]
    f = 1.0 + 0.5 * s
    big = cv2.resize(x.astype(np.float32), (round(w * f), round(h * f)), interpolation=cv2.INTER_LINEAR)
    top, left = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return big[top : top + h, left : left + w]


def _jpeg(x, s, rng, _other):
    if s == 0:
        return x
    quality = max(2, int(round(95 - 93 * s)))
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0


def _quantize(x, s, rng, _other):
    levels = int(round(256 - 254 * s))
    return np.rint(np.clip(x, 0, 1) * (levels - 1)) / (levels - 1)


def _invert(x, s, rng, _other):
    return (1.0 - s) * x + s * (1.0 - x)


def _pixel_shuffle(x, s, rng, _other):
    h, w = x.shape[:2]
    flat = x.reshape(h * w, -1).copy()
    k = int(s * h * w)
    if k >= 2:
        idx = rng.choice(h * w, size=k, replace=False)
        flat[idx] = flat[idx[rng.permutation(k)]]
    return flat.reshape(x.shape)


def _independent(x, s, rng, make_other):
    return (1.0 - s) * x + s * make_other()


DISTORTIONS: dict[str, Distortion] = {
    "gaussian_noise": _gaussian_noise,
    "salt_pepper": _salt_pepper,
    "blur": _blur,
    "brightness": _brightness,
    "contrast": _contrast,
    "gamma": _gamma,
    "hue_shift": _hue_shift,
    "translate": _translate,
    "rotate": _rotate,
    "zoom": _zoom,
    "jpeg": _jpeg,
    "quantize": _quantize,
    "invert": _invert,
    "pixel_shuffle": _pixel_shuffle,
    "independent_image": _independent,
}


def _to_8bit_grid(a: Array) -> np.ndarray:
    return (np.rint(np.clip(a, 0.0, 1.0) * 255.0) / 255.0).astype(np.float32)


# -------------------------------------------------------------------- source


class SyntheticSource(ImagePairSource):
    key = "synthetic"
    name = "Random synthetic pair"
    description = (
        "Generates a random reference image and a distorted copy. Use the "
        "strength sweep to see where each metric saturates or fails."
    )

    @classmethod
    def params(cls) -> list[ParamSpec]:
        return [
            ParamSpec("pattern", "choice", default="texture", choices=tuple(PATTERNS),
                      label="Base image pattern"),
            ParamSpec("distortion", "choice", default="gaussian_noise", choices=tuple(DISTORTIONS),
                      label="Distortion"),
            ParamSpec("strength", "float", default=0.3, min=0.0, max=1.0, step=0.01,
                      label="Distortion strength",
                      help="0 = identical images, 1 = maximum distortion."),
            ParamSpec("seed", "int", default=0, min=0, max=1_000_000, step=1, label="Seed"),
            ParamSpec("width", "int", default=256, min=16, max=2048, step=16, label="Width (px)"),
            ParamSpec("height", "int", default=256, min=16, max=2048, step=16, label="Height (px)"),
        ]

    def generate(self, **params: Any) -> ImagePair:
        p = self.resolve_params(params)
        h, w = p["height"], p["width"]
        base_ss, dist_ss, other_ss = np.random.SeedSequence(p["seed"]).spawn(3)

        pattern = PATTERNS[p["pattern"]]
        base = pattern(np.random.default_rng(base_ss), h, w)
        reference = _to_8bit_grid(base)

        def make_other() -> Array:
            return _to_8bit_grid(pattern(np.random.default_rng(other_ss), h, w)).astype(np.float64)

        distorted = DISTORTIONS[p["distortion"]](
            reference.astype(np.float64), p["strength"], np.random.default_rng(dist_ss), make_other
        )
        return ImagePair(
            reference=reference,
            test=_to_8bit_grid(distorted),
            meta={k: p[k] for k in ("pattern", "distortion", "strength", "seed", "width", "height")},
        )
