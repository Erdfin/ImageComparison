"""UI-independent orchestration: compare a pair, or sweep a source parameter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from metrics import ComparisonMetric, MetricResult, all_metrics, get_metric
from sources import ImagePair, ImagePairSource


def instantiate_metrics(keys: Sequence[str] | None = None) -> list[ComparisonMetric]:
    """Create metric instances for ``keys`` (all registered metrics if ``None``)."""
    classes = all_metrics() if keys is None else [get_metric(k) for k in keys]
    return [cls() for cls in classes]


def compare_pair(
    pair: ImagePair, metrics: Sequence[ComparisonMetric]
) -> dict[str, MetricResult]:
    """Evaluate every metric on ``pair``; keys are metric keys."""
    return {m.key: m.compare(pair.reference, pair.test) for m in metrics}


def run_sweep(
    source: ImagePairSource,
    params: Mapping[str, Any],
    sweep_param: str,
    values: Sequence[float],
    metrics: Sequence[ComparisonMetric],
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Evaluate ``metrics`` while ``sweep_param`` takes each of ``values``.

    All other parameters stay fixed at ``params``. Returns a long-format table
    with columns ``x, metric_key, metric, value, similarity, warnings``.
    """
    specs = {s.name: s for s in source.params()}
    if sweep_param not in specs or not specs[sweep_param].is_numeric:
        raise ValueError(f"{sweep_param!r} is not a numeric parameter of this source")

    rows: list[dict[str, Any]] = []
    for i, x in enumerate(values):
        pair = source.generate(**{**params, sweep_param: x})
        for metric in metrics:
            result = metric.compare(pair.reference, pair.test)
            rows.append(
                {
                    "x": x,
                    "metric_key": metric.key,
                    "metric": metric.name,
                    "value": result.value,
                    "similarity": result.similarity,
                    "warnings": "; ".join(result.warnings),
                }
            )
        if progress:
            progress((i + 1) / len(values))
    return pd.DataFrame(rows)
