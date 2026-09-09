"""Selective prediction and Risk-Coverage analysis per Phase 12 of blueprint."""

from typing import Dict, Any, Union
import numpy as np


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """Compute selective risk (error rate) across decreasing sample coverage."""
    n = len(labels)
    order = np.argsort(-confidences)
    sorted_preds = predictions[order]
    sorted_labels = labels[order]
    sorted_confs = confidences[order]

    correct_cumulative = np.cumsum(sorted_preds == sorted_labels)
    coverages = np.arange(1, n + 1) / n
    accuracies = correct_cumulative / np.arange(1, n + 1)
    risks = 1.0 - accuracies

    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trap_fn(risks, coverages))

    return {
        "coverages": coverages,
        "accuracies": accuracies,
        "risks": risks,
        "sorted_confidences": sorted_confs,
        "aurc": round(aurc, 4)
    }


def evaluate_selective_abstention(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    min_confidence_threshold: float
) -> Dict[str, float]:
    """Evaluate accuracy on the retained subset after abstaining on low confidence."""
    accepted = confidences >= min_confidence_threshold
    coverage = float(np.mean(accepted))

    if np.sum(accepted) == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "selective_accuracy": 0.0,
            "selective_error": 0.0
        }

    acc_subset = np.mean(predictions[accepted] == labels[accepted])
    return {
        "coverage": round(coverage, 4),
        "abstention_rate": round(1.0 - coverage, 4),
        "selective_accuracy": round(float(acc_subset), 4),
        "selective_error": round(float(1.0 - acc_subset), 4)
    }
