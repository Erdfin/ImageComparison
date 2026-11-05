import math

import numpy as np
import pytest

from metrics import ComparisonMetric, Direction, MetricResult, all_metrics, get_metric
from metrics.base import as_float_rgb

ALL = all_metrics()


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(rng.random((200, 200, 3)), (4, 4, 0))
    base = (base - base.min()) / (base.max() - base.min())
    noisy = np.clip(base + rng.normal(0, 0.1, base.shape), 0, 1)
    independent = rng.random(base.shape)
    return base.astype(np.float32), noisy.astype(np.float32), independent.astype(np.float32)


def test_registry_has_expected_metrics():
    keys = {c.key for c in ALL}
    assert len(ALL) >= 20
    assert {"mse", "psnr", "ssim", "ms_ssim", "gmsd", "delta_e_2000", "emd", "mi", "phash"} <= keys
    assert get_metric("ssim").name == "SSIM"
    with pytest.raises(KeyError):
        get_metric("nope")


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.key)
def test_metadata_is_complete(cls):
    assert cls.key and cls.name and cls.family and cls.description
    assert isinstance(cls.direction, Direction)
    lo, hi = cls.value_range
    assert lo < hi


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.key)
def test_identical_images_are_best(cls, images):
    base, _, _ = images
    r = cls().compare(base, base)
    assert not [w for w in r.warnings if "failed" in w or "Error" in w], r.warnings
    assert r.similarity == pytest.approx(1.0, abs=0.02), (cls.key, r.value)


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.key)
def test_unrelated_images_score_worse_than_identical(cls, images):
    base, _, independent = images
    metric = cls()
    same = metric.compare(base, base)
    diff = metric.compare(base, independent)
    assert diff.similarity < same.similarity
    # raw value moves in the direction declared by the metric
    if cls.direction is Direction.HIGHER_IS_BETTER:
        assert diff.value < same.value
    else:
        assert diff.value > same.value


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.key)
def test_similarity_in_unit_interval_and_maps_valid(cls, images):
    base, noisy, _ = images
    r = cls().compare(base, noisy)
    assert r.similarity is None or 0.0 <= r.similarity <= 1.0
    assert r.elapsed_ms >= 0
    if r.map is not None:
        assert r.map.ndim == 2 and r.map.min() >= 0.0 and r.map.max() <= 1.0


def test_known_values():
    a = np.full((32, 32, 3), 0.5, dtype=np.float32)
    b = a + 0.1
    assert get_metric("mse")().compare(a, b).value == pytest.approx(0.01, rel=1e-4)
    assert get_metric("mae")().compare(a, b).value == pytest.approx(0.1, rel=1e-4)
    assert get_metric("rmse")().compare(a, b).value == pytest.approx(0.1, rel=1e-4)
    assert get_metric("psnr")().compare(a, b).value == pytest.approx(20.0, abs=1e-3)
    black, white = np.zeros((32, 32, 3)), np.ones((32, 32, 3))
    assert get_metric("delta_e_2000")().compare(black, white).value == pytest.approx(100.0, abs=0.5)
    assert get_metric("delta_e_76")().compare(black, white).value == pytest.approx(100.0, abs=0.5)
    assert get_metric("hist_intersection")().compare(black, white).value == pytest.approx(0.0)


def test_uint8_and_float_inputs_agree():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    m = get_metric("ssim")()
    assert m.compare(a, b).value == pytest.approx(m.compare(a / 255.0, b / 255.0).value, abs=1e-6)


def test_grayscale_and_alpha_inputs_are_accepted():
    g = np.random.default_rng(0).random((40, 40))
    assert as_float_rgb(g).shape == (40, 40, 3)
    assert as_float_rgb(np.zeros((8, 8, 4), np.uint8)).shape == (8, 8, 3)


def test_invalid_inputs_raise():
    m = get_metric("mse")()
    with pytest.raises(ValueError):
        m.compare(np.zeros((8, 8, 3)), np.zeros((9, 8, 3)))
    with pytest.raises(ValueError):
        m.compare(np.full((8, 8, 3), np.nan), np.zeros((8, 8, 3)))
    with pytest.raises(TypeError):
        m.compare(np.zeros((8, 8, 3), dtype=np.int32), np.zeros((8, 8, 3), dtype=np.int32))


def test_constant_images_are_reported_not_crashing():
    flat = np.full((64, 64, 3), 0.3, dtype=np.float32)
    other = np.full((64, 64, 3), 0.7, dtype=np.float32)
    zncc = get_metric("zncc")().compare(flat, other)
    assert math.isnan(zncc.value) and zncc.warnings
    assert get_metric("psnr")().compare(flat, flat).warnings
    assert math.isnan(get_metric("nmi")().compare(flat, flat).value)
    # every metric must survive degenerate input without raising
    for cls in ALL:
        cls().compare(flat, other)


def test_failures_inside_a_metric_become_warnings():
    r = get_metric("ms_ssim")().compare(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))
    assert math.isnan(r.value) and "too small" in r.warnings[0]


def test_ms_ssim_uses_fewer_scales_on_small_images():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    r = get_metric("ms_ssim")().compare(a, a)
    assert r.value == pytest.approx(1.0, abs=1e-4)
    assert any("scales" in w for w in r.warnings)


def test_phase_correlation_estimates_shift():
    rng = np.random.default_rng(5)
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(rng.random((128, 128, 3)), (2, 2, 0)).astype(np.float32)
    b = np.roll(a, shift=(5, -9), axis=(0, 1))
    r = get_metric("phase_correlation")().compare(a, b)
    assert r.details["estimated_shift_yx"] == (5, -9)
    assert r.value > 0.3


def test_histogram_metrics_are_blind_to_pixel_shuffling():
    rng = np.random.default_rng(7)
    a = rng.random((64, 64, 3)).astype(np.float32)
    shuffled = a.reshape(-1, 3)[rng.permutation(64 * 64)].reshape(a.shape)
    assert get_metric("hist_intersection")().compare(a, shuffled).value == pytest.approx(1.0)
    assert get_metric("mse")().compare(a, shuffled).value > 0.05


def test_subclass_registers_itself_and_validates_attributes():
    class _Const(ComparisonMetric):
        key = "_test_const"
        name = "Const"
        family = "Test"
        direction = Direction.LOWER_IS_BETTER
        value_range = (0.0, 1.0)

        def _compute(self, ref, test):
            return MetricResult(value=0.25)

        def _to_similarity(self, value):
            return 1 - value

    assert get_metric("_test_const") is _Const
    assert _Const().compare(np.zeros((4, 4)), np.ones((4, 4))).similarity == 0.75

    with pytest.raises(TypeError):

        class _Bad(ComparisonMetric):
            key = "_test_bad"

            def _compute(self, ref, test):
                return MetricResult(value=0.0)

    with pytest.raises(ValueError):

        class _Dup(_Const):
            key = "_test_const"
