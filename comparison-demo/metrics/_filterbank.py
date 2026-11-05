"""Log-Gabor filter bank and phase congruency (shared by FSIM and CW-SSIM).

The leading underscore keeps this module out of the metric auto-discovery.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

N_SCALES = 4
N_ORIENT = 4


@lru_cache(maxsize=4)
def log_gabor_bank(
    rows: int,
    cols: int,
    n_scales: int = N_SCALES,
    n_orient: int = N_ORIENT,
    min_wavelength: float = 6.0,
    mult: float = 2.0,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.2,
) -> np.ndarray:
    """Frequency-domain log-Gabor filters, shape ``(n_orient, n_scales, rows, cols)``.

    The filters are one-sided in orientation, so filtering with them yields
    complex (quadrature) responses: real part = even, imaginary = odd.
    """
    fx = np.fft.fftfreq(cols)[None, :]
    fy = np.fft.fftfreq(rows)[:, None]
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1.0
    theta = np.arctan2(-fy, fx)
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)
    theta_sigma = np.pi / n_orient / d_theta_on_sigma

    radial = []
    for s in range(n_scales):
        fo = 1.0 / (min_wavelength * mult**s)
        lg = np.exp(-(np.log(radius / fo) ** 2) / (2 * np.log(sigma_on_f) ** 2))
        lg *= lowpass
        lg[0, 0] = 0.0
        radial.append(lg)

    bank = np.empty((n_orient, n_scales, rows, cols))
    for o in range(n_orient):
        angle = o * np.pi / n_orient
        ds = sin_t * np.cos(angle) - cos_t * np.sin(angle)
        dc = cos_t * np.cos(angle) + sin_t * np.sin(angle)
        spread = np.exp(-(np.arctan2(ds, dc) ** 2) / (2 * theta_sigma**2))
        for s in range(n_scales):
            bank[o, s] = radial[s] * spread
    bank.setflags(write=False)
    return bank


def filter_responses(image: np.ndarray) -> np.ndarray:
    """Complex responses of ``image`` (2-D) to the bank: ``(n_orient, n_scales, H, W)``."""
    bank = log_gabor_bank(*image.shape)
    spectrum = np.fft.fft2(image)
    return np.fft.ifft2(spectrum[None, None] * bank)


def phase_congruency(
    image: np.ndarray,
    k: float = 2.0,
    cut_off: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
    mult: float = 2.0,
) -> np.ndarray:
    """Phase congruency map (sum over orientations), after Kovesi.

    ``image`` is a 2-D float array (FSIM uses luma on a 0-255 scale).
    """
    responses = filter_responses(image)
    n_orient, n_scales = responses.shape[:2]
    total = np.zeros(image.shape)

    for o in range(n_orient):
        eo = responses[o]
        amp = np.abs(eo)
        sum_an = amp.sum(axis=0)
        max_an = amp.max(axis=0)
        sum_e = eo.real.sum(axis=0)
        sum_o = eo.imag.sum(axis=0)

        x_energy = np.sqrt(sum_e**2 + sum_o**2) + eps
        mean_e, mean_o = sum_e / x_energy, sum_o / x_energy
        energy = (
            eo.real * mean_e + eo.imag * mean_o - np.abs(eo.real * mean_o - eo.imag * mean_e)
        ).sum(axis=0)

        # Noise threshold estimated from the smallest-scale response.
        tau = np.median(amp[0]) / np.sqrt(np.log(4.0))
        total_tau = tau * (1 - (1 / mult) ** n_scales) / (1 - 1 / mult)
        threshold = total_tau * np.sqrt(np.pi / 2) + k * total_tau * np.sqrt((4 - np.pi) / 2)
        energy = np.maximum(energy - threshold, 0.0)

        width = (sum_an / (max_an + eps) - 1.0) / (n_scales - 1)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * gain))
        total += weight * energy / (sum_an + eps)
    return total
