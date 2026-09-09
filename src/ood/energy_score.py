"""Energy-based Out-of-Distribution (OOD) scoring over logit outputs (Liu et al., NeurIPS 2020)."""

from typing import Union
import numpy as np
import torch
from scipy.special import logsumexp


def compute_energy_score(logits: Union[torch.Tensor, np.ndarray], temperature: float = 1.0) -> Union[torch.Tensor, np.ndarray]:
    """Compute energy scoring function S_energy(x) = T * LogSumExp(f(x) / T).

    Higher energy score indicates in-distribution (ID) with strong activations.
    Lower energy score indicates out-of-distribution (OOD) sample.
    """
    if isinstance(logits, torch.Tensor):
        return temperature * torch.logsumexp(logits / temperature, dim=-1)
    else:
        return temperature * logsumexp(logits / temperature, axis=-1)


def is_ood_energy(
    logits: Union[torch.Tensor, np.ndarray], 
    threshold: float = 1.0, 
    temperature: float = 1.0
) -> Union[torch.Tensor, np.ndarray]:
    """Flag whether an input logit vector is classified as Out-of-Distribution (energy < threshold)."""
    energy = compute_energy_score(logits, temperature=temperature)
    return energy < threshold
