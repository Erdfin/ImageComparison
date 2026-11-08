"""Image comparison metrics.

Importing this package registers every metric found in its modules.
"""

from .base import ComparisonMetric, Direction, MetricResult
from .registry import all_metrics, get_metric, load_builtin_metrics

load_builtin_metrics()

__all__ = [
    "ComparisonMetric",
    "Direction",
    "MetricResult",
    "all_metrics",
    "get_metric",
]
