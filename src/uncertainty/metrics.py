"""Calibration metrics: Expected Calibration Error (ECE), Brier Score, and predictive uncertainty metrics."""

from typing import Dict, Tuple, Union
import numpy as np
from scipy.special import softmax


def compute_shannon_entropy(probs: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Compute Shannon predictive entropy for probability distributions [N, C]."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    denom = np.log(base)
    entropy = -np.sum(probs * (np.log(probs) / denom), axis=-1)
    return entropy


def compute_prediction_margin(probs: np.ndarray) -> np.ndarray:
    """Compute top-1 vs top-2 probability margin."""
    sorted_probs = np.sort(probs, axis=-1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margin


def compute_expected_calibration_error(
    probs_or_logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    is_logits: bool = False
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs_or_logits: Predicted probabilities or unscaled logits [N, C].
        labels: Ground-truth target indices [N].
        num_bins: Number of equal-width confidence bins.
        is_logits: Whether input is raw logits.

    Returns:
        Dict with 'ece', 'mce', and 'avg_confidence'.
    """
    if is_logits:
        probs = softmax(probs_or_logits, axis=-1)
    else:
        probs = probs_or_logits

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            gap = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_confidence": float(np.mean(confidences)),
        "accuracy": float(np.mean(accuracies))
    }


def compute_maximum_calibration_error(
    probs_or_logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    is_logits: bool = False
) -> float:
    """Compute Maximum Calibration Error (MCE)."""
    res = compute_expected_calibration_error(probs_or_logits, labels, num_bins=num_bins, is_logits=is_logits)
    return res["mce"]


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """Compute multiclass Brier score (mean squared error of probability vectors)."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
