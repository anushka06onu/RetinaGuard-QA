"""Evaluation metrics, cross-dataset transfer benchmarks, and synthetic corruption analysis."""

from .metrics_engine import (
    compute_multiclass_metrics,
    compute_defect_metrics,
    compute_bootstrap_ci,
    EvaluationReport
)
from .robustness_benchmark import run_robustness_sweep, compute_relative_robustness
from .cross_dataset import evaluate_cross_dataset_transfer

__all__ = [
    "compute_multiclass_metrics",
    "compute_defect_metrics",
    "compute_bootstrap_ci",
    "EvaluationReport",
    "run_robustness_sweep",
    "compute_relative_robustness",
    "evaluate_cross_dataset_transfer"
]
