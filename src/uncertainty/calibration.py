"""Post-hoc probability calibration via Platt scaling and scalar/vector temperature scaling."""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize


class TemperatureScaler(nn.Module):
    """Calibrates multi-class prediction probabilities using a learned temperature parameter."""

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def get_temperature(self) -> float:
        return float(self.temperature.item())


def fit_temperature_scaling(
    logits: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray],
    max_iter: int = 100,
    lr: float = 0.01
) -> float:
    """Find optimal temperature T on validation logits to minimize Negative Log-Likelihood (NLL).

    Args:
        logits: Uncalibrated logit tensor or array of shape [N, num_classes].
        labels: Ground-truth class index tensor or array of shape [N].

    Returns:
        Optimized temperature parameter T (scalar).
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().float()

    if isinstance(labels, np.ndarray):
        labels_t = torch.from_numpy(labels).long()
    else:
        labels_t = labels.detach().long()

    temp_scaler = TemperatureScaler(init_temperature=1.0)
    optimizer = optim.LBFGS([temp_scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled = temp_scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    best_temp = float(torch.clamp(temp_scaler.temperature, min=0.05).item())
    return best_temp
