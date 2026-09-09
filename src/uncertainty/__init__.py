"""Uncertainty quantification, post-hoc probability calibration, and selective prediction."""

from .calibration import TemperatureScaler, fit_temperature_scaling
from .selective_prediction import (
    compute_risk_coverage_curve,
    evaluate_selective_prediction,
    find_abstention_threshold
)
from .metrics import (
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    compute_shannon_entropy,
    compute_prediction_margin
)

__all__ = [
    "TemperatureScaler",
    "fit_temperature_scaling",
    "compute_risk_coverage_curve",
    "evaluate_selective_prediction",
    "find_abstention_threshold",
    "compute_expected_calibration_error",
    "compute_maximum_calibration_error",
    "compute_brier_score",
    "compute_shannon_entropy",
    "compute_prediction_margin"
]
