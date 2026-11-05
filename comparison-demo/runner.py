"""UI-independent orchestration: compare a pair, or sweep a source parameter.

Metrics are independent and spend most of their time inside numpy / scipy /
OpenCV (which release the interpreter lock), so they are run in a thread pool.
Pass ``max_workers=1`` for strictly sequential execution, which gives the
cleanest per-metric wall-clock timings.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from metrics import ComparisonMetric, MetricResult, all_metrics, get_metric
from sources import ImagePair, ImagePairSource


def default_workers() -> int:
    """A sensible number of worker threads for this machine."""
    return max(1, min(8, os.cpu_count() or 1))


def instantiate_metrics(keys: Sequence[str] | None = None) -> list[ComparisonMetric]:
    """Create metric instances for ``keys`` (all registered metrics if ``None``)."""
    classes = all_metrics() if keys is None else [get_metric(k) for k in keys]
    return [cls() for cls in classes]


def _evaluate(
    pair: ImagePair, metrics: Sequence[ComparisonMetric], pool: ThreadPoolExecutor | None
) -> list[MetricResult]:
    def run(metric: ComparisonMetric) -> MetricResult:
        return metric.compare(pair.reference, pair.test)

    return list(pool.map(run, metrics)) if pool else [run(m) for m in metrics]


def compare_pair(
    pair: ImagePair, metrics: Sequence[ComparisonMetric], max_workers: int = 1
) -> dict[str, MetricResult]:
    """Evaluate every metric on ``pair``; keys are metric keys."""
    if max_workers > 1 and len(metrics) > 1:
        with ThreadPoolExecutor(max_workers) as pool:
            results = _evaluate(pair, metrics, pool)
    else:
        results = _evaluate(pair, metrics, None)
    return {m.key: r for m, r in zip(metrics, results)}


def run_sweep(
    source: ImagePairSource,
    params: Mapping[str, Any],
    sweep_param: str,
    values: Sequence[float],
    metrics: Sequence[ComparisonMetric],
    progress: Callable[[float], None] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Evaluate ``metrics`` while ``sweep_param`` takes each of ``values``.

    All other parameters stay fixed at ``params``. Returns a long-format table
    with columns ``x, metric_key, metric, value, similarity, warnings,
    elapsed_ms, cpu_ms``. ``df.attrs`` holds ``wall_s`` (total wall-clock time
    of the sweep, including image generation) and ``workers``.
    """
    specs = {s.name: s for s in source.params()}
    if sweep_param not in specs or not specs[sweep_param].is_numeric:
        raise ValueError(f"{sweep_param!r} is not a numeric parameter of this source")

    workers = max(1, int(max_workers))
    pool = ThreadPoolExecutor(workers) if workers > 1 and len(metrics) > 1 else None
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        for i, x in enumerate(values):
            # One step at a time keeps memory bounded for large images.
            pair = source.generate(**{**params, sweep_param: x})
            for metric, result in zip(metrics, _evaluate(pair, metrics, pool)):
                rows.append(
                    {
                        "x": x,
                        "metric_key": metric.key,
                        "metric": metric.name,
                        "value": result.value,
                        "similarity": result.similarity,
                        "warnings": "; ".join(result.warnings),
                        "elapsed_ms": result.elapsed_ms,
                        "cpu_ms": result.cpu_ms,
                    }
                )
            if progress:
                progress((i + 1) / len(values))
    finally:
        if pool:
            pool.shutdown()

    df = pd.DataFrame(rows)
    df.attrs["wall_s"] = time.perf_counter() - start
    df.attrs["workers"] = workers if pool else 1
    return df


def timing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric timing statistics of a sweep table, slowest first."""
    return (
        df.groupby(["metric_key", "metric"], sort=False)
        .agg(
            runs=("x", "count"),
            mean_wall_ms=("elapsed_ms", "mean"),
            max_wall_ms=("elapsed_ms", "max"),
            mean_cpu_ms=("cpu_ms", "mean"),
            total_cpu_ms=("cpu_ms", "sum"),
        )
        .reset_index()
        .sort_values("total_cpu_ms", ascending=False, ignore_index=True)
    )
