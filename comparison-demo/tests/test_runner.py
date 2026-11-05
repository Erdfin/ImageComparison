import numpy as np
import pytest

from runner import compare_pair, instantiate_metrics, run_sweep
from sources import get_source


def test_compare_pair_returns_one_result_per_metric():
    pair = get_source("synthetic")().generate(strength=0.3)
    metrics = instantiate_metrics(["mse", "ssim", "phash"])
    results = compare_pair(pair, metrics)
    assert set(results) == {"mse", "ssim", "phash"}


def test_sweep_table_shape_and_monotonic_noise_response():
    source = get_source("synthetic")()
    metrics = instantiate_metrics(["mse", "ssim"])
    values = np.linspace(0, 1, 5)
    df = run_sweep(source, {"distortion": "gaussian_noise", "width": 64, "height": 64},
                   "strength", values, metrics)
    assert list(df.columns) == ["x", "metric_key", "metric", "value", "similarity", "warnings"]
    assert len(df) == 5 * 2
    mse = df[df.metric_key == "mse"].sort_values("x")["value"].to_numpy()
    assert np.all(np.diff(mse) > 0)


def test_sweep_progress_and_int_params():
    calls = []
    df = run_sweep(get_source("synthetic")(), {}, "width", [32, 48], instantiate_metrics(["mae"]),
                   progress=calls.append)
    assert calls == [0.5, 1.0] and len(df) == 2


def test_sweep_rejects_non_numeric_param():
    with pytest.raises(ValueError):
        run_sweep(get_source("synthetic")(), {}, "pattern", [0, 1], instantiate_metrics(["mae"]))
