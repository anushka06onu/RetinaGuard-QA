"""Selective prediction, abstention thresholding, and Risk-Coverage curve evaluation."""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_risk_coverage_curve(
    confidences: np.ndarray, 
    predictions: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """Compute selective accuracy and error (risk) as a function of sample coverage.

    Samples are ranked in descending order of confidence. As coverage decreases from 1.0 to 0.0,
    most uncertain samples are abstained from prediction.
    """
    n = len(labels)
    # Sort samples from most confident to least confident
    order = np.argsort(-confidences)
    sorted_preds = predictions[order]
    sorted_labels = labels[order]
    sorted_confs = confidences[order]

    correct_cumulative = np.cumsum(sorted_preds == sorted_labels)
    coverages = np.arange(1, n + 1) / n
    accuracies = correct_cumulative / np.arange(1, n + 1)
    risks = 1.0 - accuracies  # Selective risk (error rate on accepted subset)

    # Area Under Risk-Coverage Curve (AURC)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trap_fn(risks, coverages))

    return {
        "coverages": coverages,
        "accuracies": accuracies,
        "risks": risks,
        "sorted_confidences": sorted_confs,
        "aurc": aurc
    }


def find_abstention_threshold(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    target_accuracy: float = 0.95
) -> Dict[str, float]:
    """Find the minimum confidence threshold required to achieve a target selective accuracy."""
    curve = compute_risk_coverage_curve(confidences, predictions, labels)
    accs = curve["accuracies"]
    confs = curve["sorted_confidences"]
    covs = curve["coverages"]

    valid_indices = np.where(accs >= target_accuracy)[0]
    if len(valid_indices) == 0:
        # Target accuracy not reached even at highest confidence sample
        best_idx = 0
    else:
        # Largest coverage (first index meeting criteria)
        best_idx = valid_indices[-1]

    return {
        "target_accuracy": target_accuracy,
        "achieved_accuracy": float(accs[best_idx]),
        "achieved_coverage": float(covs[best_idx]),
        "abstention_rate": float(1.0 - covs[best_idx]),
        "confidence_threshold": float(confs[best_idx])
    }


def evaluate_selective_prediction(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Evaluate performance when abstaining on samples with confidence below threshold."""
    accepted_mask = confidences >= threshold
    coverage = float(np.mean(accepted_mask))

    if np.sum(accepted_mask) == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "selective_accuracy": 0.0,
            "selective_error": 0.0
        }

    acc_subset = np.mean(predictions[accepted_mask] == labels[accepted_mask])
    return {
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "selective_accuracy": float(acc_subset),
        "selective_error": float(1.0 - acc_subset)
    }
