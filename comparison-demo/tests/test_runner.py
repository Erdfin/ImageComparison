import numpy as np
import pytest

from runner import compare_pair, default_workers, instantiate_metrics, run_sweep, timing_summary
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
    assert list(df.columns) == ["x", "metric_key", "metric", "value", "similarity", "warnings",
                                "elapsed_ms", "cpu_ms"]
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


def test_parallel_results_equal_sequential_results():
    source = get_source("synthetic")()
    metrics = instantiate_metrics()  # every registered metric, incl. FLIP
    params = {"distortion": "gaussian_noise", "width": 96, "height": 96}
    values = [0.0, 0.3, 0.8]
    seq = run_sweep(source, params, "strength", values, metrics, max_workers=1)
    par = run_sweep(source, params, "strength", values, metrics, max_workers=4)
    assert seq[["x", "metric_key"]].equals(par[["x", "metric_key"]])  # same order
    np.testing.assert_allclose(
        seq["value"].to_numpy(float), par["value"].to_numpy(float), equal_nan=True, rtol=1e-9
    )
    assert seq.attrs["workers"] == 1 and par.attrs["workers"] == 4


def test_compare_pair_parallel_matches_sequential():
    pair = get_source("synthetic")().generate(strength=0.4, width=96, height=96)
    metrics = instantiate_metrics()
    a = compare_pair(pair, metrics, max_workers=1)
    b = compare_pair(pair, metrics, max_workers=4)
    assert list(a) == list(b)
    for key in a:
        np.testing.assert_allclose(a[key].value, b[key].value, equal_nan=True, rtol=1e-9)


def test_timing_columns_and_summary():
    df = run_sweep(get_source("synthetic")(), {"width": 64, "height": 64}, "strength",
                   [0.0, 0.5], instantiate_metrics(["mse", "ssim", "fsim"]), max_workers=2)
    assert (df["elapsed_ms"] > 0).all() and (df["cpu_ms"] >= 0).all()
    assert df.attrs["wall_s"] > 0
    summary = timing_summary(df)
    assert set(summary["metric_key"]) == {"mse", "ssim", "fsim"}
    assert (summary["runs"] == 2).all()
    assert summary["total_cpu_ms"].is_monotonic_decreasing
    assert summary.iloc[0]["metric_key"] == "fsim"  # by far the slowest of the three


def test_default_workers_is_positive():
    assert 1 <= default_workers() <= 8
