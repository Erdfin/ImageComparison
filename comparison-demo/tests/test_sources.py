import numpy as np
import pytest
from PIL import Image

from sources import ImagePair, ImagePairSource, ParamSpec, all_sources, get_source
from sources.synthetic import DISTORTIONS, PATTERNS


@pytest.fixture
def synthetic():
    return get_source("synthetic")()


def test_registry():
    keys = {c.key for c in all_sources()}
    assert {"disk", "synthetic"} <= keys
    with pytest.raises(KeyError):
        get_source("nope")


@pytest.mark.parametrize("cls", all_sources(), ids=lambda c: c.key)
def test_every_source_declares_valid_params(cls):
    specs = cls.params()
    assert len({s.name for s in specs}) == len(specs)
    for s in specs:
        if s.kind == "choice":
            assert s.default in s.choices
        if s.is_numeric:
            assert s.min <= s.default <= s.max


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("distortion", DISTORTIONS)
def test_synthetic_contract(synthetic, pattern, distortion):
    zero = synthetic.generate(pattern=pattern, distortion=distortion, strength=0.0, width=64, height=48)
    assert zero.reference.shape == zero.test.shape == (48, 64, 3)
    assert zero.reference.dtype == zero.test.dtype == np.float32
    assert 0.0 <= zero.reference.min() and zero.reference.max() <= 1.0
    assert np.array_equal(zero.reference, zero.test), "strength 0 must give identical images"

    full = synthetic.generate(pattern=pattern, distortion=distortion, strength=1.0, width=64, height=48)
    assert full.test.shape == (48, 64, 3)
    assert 0.0 <= full.test.min() and full.test.max() <= 1.0


def test_synthetic_is_deterministic_and_seed_dependent(synthetic):
    a = synthetic.generate(seed=1, strength=0.5)
    b = synthetic.generate(seed=1, strength=0.5)
    c = synthetic.generate(seed=2, strength=0.5)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.reference, b.reference)
    assert not np.array_equal(a.reference, c.reference)


def test_reference_does_not_depend_on_distortion_or_strength(synthetic):
    a = synthetic.generate(distortion="blur", strength=0.2, seed=4)
    b = synthetic.generate(distortion="jpeg", strength=0.9, seed=4)
    assert np.array_equal(a.reference, b.reference)


def test_distortion_grows_with_strength(synthetic):
    errors = [
        np.abs(synthetic.generate(distortion="gaussian_noise", strength=s).test
               - synthetic.generate(distortion="gaussian_noise", strength=s).reference).mean()
        for s in (0.0, 0.2, 0.6)
    ]
    assert errors[0] == 0 and errors[0] < errors[1] < errors[2]


def test_invalid_params(synthetic):
    with pytest.raises(ValueError):
        synthetic.generate(strength=1.5)
    with pytest.raises(ValueError):
        synthetic.generate(pattern="nope")
    with pytest.raises(TypeError):
        synthetic.generate(bogus=1)


def test_a_custom_source_plugs_in():
    class _Fixed(ImagePairSource):
        key = "_test_fixed"
        name = "Fixed"

        @classmethod
        def params(cls):
            return [ParamSpec("level", "float", default=0.5, min=0.0, max=1.0)]

        def generate(self, **params):
            p = self.resolve_params(params)
            img = np.full((8, 8, 3), p["level"], dtype=np.float32)
            return ImagePair(img, img.copy())

    assert get_source("_test_fixed") is _Fixed
    assert _Fixed().generate(level=0.2).reference[0, 0, 0] == pytest.approx(0.2)


# ------------------------------------------------------------------- disk


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    ref = Image.fromarray(rng.integers(0, 256, (60, 100, 3), dtype=np.uint8))
    same = tmp_path / "same.png"
    ref.save(tmp_path / "ref.png")
    ref.save(same)
    ref.resize((50, 30)).save(tmp_path / "small.png")
    ref.convert("L").save(tmp_path / "gray.png")
    ref.convert("RGBA").save(tmp_path / "alpha.png")
    return tmp_path


def test_disk_loads_identical_files(files):
    pair = get_source("disk")().generate(reference=str(files / "ref.png"), test=str(files / "same.png"))
    assert pair.reference.shape == (60, 100, 3)
    assert np.array_equal(pair.reference, pair.test)
    assert pair.meta["size"] == (100, 60)


def test_disk_size_policies(files):
    src = get_source("disk")()
    ref, small = str(files / "ref.png"), str(files / "small.png")
    resized = src.generate(reference=ref, test=small, size_policy="resize_test")
    assert resized.test.shape == resized.reference.shape == (60, 100, 3)
    cropped = src.generate(reference=ref, test=small, size_policy="center_crop")
    assert cropped.reference.shape == cropped.test.shape == (30, 50, 3)
    with pytest.raises(ValueError, match="sizes differ"):
        src.generate(reference=ref, test=small, size_policy="error")


def test_disk_max_side_and_modes(files):
    src = get_source("disk")()
    pair = src.generate(reference=str(files / "ref.png"), test=str(files / "gray.png"), max_side=50)
    assert pair.reference.shape == pair.test.shape == (30, 50, 3)
    alpha = src.generate(reference=str(files / "alpha.png"), test=str(files / "ref.png"))
    assert alpha.reference.shape == (60, 100, 3)


def test_disk_accepts_file_objects(files):
    with open(files / "ref.png", "rb") as f1, open(files / "ref.png", "rb") as f2:
        pair = get_source("disk")().generate(reference=f1, test=f2)
    assert np.array_equal(pair.reference, pair.test)


def test_disk_errors_are_user_readable(files):
    src = get_source("disk")()
    with pytest.raises(ValueError, match="reference image is missing"):
        src.generate()
    with pytest.raises(ValueError, match="not found"):
        src.generate(reference=str(files / "nope.png"), test=str(files / "ref.png"))
    (files / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not be read"):
        src.generate(reference=str(files / "bad.png"), test=str(files / "ref.png"))
