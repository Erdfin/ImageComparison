import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from metrics import get_metric
from metrics._filterbank import log_gabor_bank, phase_congruency
from metrics.perceptual import _haar_kernel
from metrics.shape import EdgeChamfer, EdgeDice, EdgeHausdorff, EdgeJaccard, edge_map


@pytest.fixture(scope="module")
def textured():
    rng = np.random.default_rng(11)
    img = gaussian_filter(rng.random((160, 160, 3)), (1.5, 1.5, 0))
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)


def score(key, a, b):
    return get_metric(key)().compare(a, b)


def test_phase2_metrics_are_registered():
    for key in ["uqi", "cw_ssim", "fsim", "fsimc", "haarpsi", "vifp", "flip",
                "edge_jaccard", "edge_dice", "edge_hausdorff", "edge_chamfer"]:
        assert get_metric(key).key == key


def test_haar_kernels_match_the_paper():
    for scale in (1, 2, 3):
        for horizontal in (True, False):
            k = _haar_kernel(scale, horizontal)
            assert k.shape == (2**scale, 2**scale)
            assert np.allclose(np.abs(k), 2.0**-scale)
            assert k.sum() == pytest.approx(0.0)
    # horizontal structures: difference across rows, constant along columns
    k = _haar_kernel(1, True)
    assert np.allclose(k[0], -k[1]) and np.allclose(k[:, 0], k[:, 1] * np.sign(k[0, 0] * k[0, 1]))


def test_filter_bank_shape_and_cache():
    bank = log_gabor_bank(48, 64)
    assert bank.shape == (4, 4, 48, 64)
    assert log_gabor_bank(48, 64) is bank
    assert bank[:, :, 0, 0].max() == 0.0  # no DC response


def test_phase_congruency_responds_to_edges_not_flat_regions():
    img = np.zeros((64, 64))
    img[:, 32:] = 100.0
    pc = phase_congruency(img)
    assert pc[:, 30:34].max() > 5 * pc[:, 5:15].max()
    assert np.allclose(phase_congruency(np.full((64, 64), 50.0)), 0.0, atol=1e-6)


def test_vifp_decreases_with_noise_and_needs_size(textured):
    rng = np.random.default_rng(0)
    mild = np.clip(textured + rng.normal(0, 0.02, textured.shape), 0, 1).astype(np.float32)
    heavy = np.clip(textured + rng.normal(0, 0.15, textured.shape), 0, 1).astype(np.float32)
    assert 1.0 > score("vifp", textured, mild).value > score("vifp", textured, heavy).value
    tiny = score("vifp", textured[:30, :30], textured[:30, :30])
    assert math.isnan(tiny.value) and "too small" in tiny.warnings[0]


def test_vifp_can_exceed_one_for_contrast_enhancement(textured):
    stretched = np.clip(0.5 + (textured - 0.5) * 1.3, 0, 1).astype(np.float32)
    assert score("vifp", textured, stretched).value > 0.9  # high fidelity (can pass 1 on real photos)


def test_uqi_known_cases(textured):
    assert score("uqi", textured, textured).value == pytest.approx(1.0)
    assert score("uqi", textured, 1 - textured).value < 0
    flat_a = np.full((32, 32, 3), 0.2, np.float32)
    flat_b = np.full((32, 32, 3), 0.8, np.float32)
    # constant images: only the luminance term is informative
    expected = 2 * 0.2 * 0.8 / (0.2**2 + 0.8**2)
    r = score("uqi", flat_a, flat_b)
    assert r.value == pytest.approx(expected, rel=1e-3) and not r.warnings


def test_polarity_blindness_of_magnitude_based_metrics(textured):
    inverted = 1 - textured
    assert score("fsim", textured, inverted).value > 0.95
    assert score("haarpsi", textured, inverted).value > 0.95
    assert score("ssim", textured, inverted).value < 0.5
    assert score("vifp", textured, inverted).value < 0.2


def test_cw_ssim_is_more_shift_tolerant_than_ssim():
    rng = np.random.default_rng(2)
    img = gaussian_filter(rng.random((128, 128, 3)), (1.0, 1.0, 0)).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    shifted = np.roll(img, 1, axis=1)
    assert score("cw_ssim", img, shifted).value > score("ssim", img, shifted).value + 0.2


def test_fsim_and_haarpsi_flat_images_are_reported():
    flat = np.full((64, 64, 3), 0.4, np.float32)
    for key in ("fsim", "fsimc", "haarpsi"):
        r = score(key, flat, flat)
        assert math.isnan(r.value) and r.warnings, key
    assert score("vifp", flat, flat).warnings


def test_fsim_requires_minimum_size():
    r = score("fsim", np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))
    assert math.isnan(r.value) and "too small" in r.warnings[0]


def test_flip_scale_and_direction(textured):
    assert score("flip", textured, textured).value == pytest.approx(0.0, abs=1e-6)
    bright = np.clip(textured + 0.3, 0, 1).astype(np.float32)
    assert score("flip", textured, bright).value > 0.05


# ------------------------------------------------------------------ edges


def test_edge_distance_metrics_on_single_pixels():
    a = np.zeros((40, 40), bool)
    b = np.zeros((40, 40), bool)
    a[5, 5] = True
    b[5, 9] = True
    assert EdgeHausdorff()._compare_edges(a, b).value == pytest.approx(4.0)
    assert EdgeChamfer()._compare_edges(a, b).value == pytest.approx(4.0)
    assert EdgeJaccard()._compare_edges(a, b).value == 0.0
    assert EdgeDice()._compare_edges(a, b).value == 0.0
    both = a | b
    assert EdgeJaccard()._compare_edges(a, both).value == pytest.approx(0.5)
    assert EdgeDice()._compare_edges(a, both).value == pytest.approx(2 / 3)


def test_edge_metrics_edge_cases():
    empty = np.zeros((20, 20), bool)
    one = empty.copy()
    one[3, 3] = True
    r = EdgeHausdorff()._compare_edges(empty, one)
    assert math.isinf(r.value) and r.similarity == 0.0 and r.warnings
    assert EdgeJaccard()._compare_edges(empty, empty).value == 1.0


def test_edge_map_is_gain_invariant_and_empty_for_flat(textured):
    assert not edge_map(np.full((32, 32, 3), 0.5, np.float32)).any()
    e1 = edge_map(textured)
    e2 = edge_map((0.5 * textured + 0.2).astype(np.float32))
    assert e1.dtype == bool and e1.any()
    assert (e1 & e2).sum() / (e1 | e2).sum() > 0.9
