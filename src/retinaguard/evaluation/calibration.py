"""Probability calibration, Expected Calibration Error (ECE), and Brier score."""

from typing import Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax


class TemperatureScaler(nn.Module):
    """Post-hoc scalar temperature scaling module."""
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp


def fit_temperature_scaling(
    logits: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    max_iter: int = 100,
    lr: float = 0.01
) -> float:
    """Optimize scalar temperature T on validation logits to minimize NLL."""
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().float()

    if isinstance(labels, np.ndarray):
        labels_t = torch.from_numpy(labels).long()
    else:
        labels_t = labels.detach().long()

    scaler = TemperatureScaler(init_temp=1.0)
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    best_temp = float(torch.clamp(scaler.temperature, min=0.05).item())
    return best_temp


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

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
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "accuracy": round(float(np.mean(accuracies)), 4)
    }


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """Compute multi-class Brier score."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return round(float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))), 4)
