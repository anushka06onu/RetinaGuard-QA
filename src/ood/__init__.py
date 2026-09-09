"""Out-of-Distribution (OOD) detection, energy scoring, and input modality validation."""

from .energy_score import compute_energy_score, is_ood_energy
from .mahalanobis import MahalanobisOODDetector
from .visual_gate import RetinalModalityGate, validate_retinal_modality

__all__ = [
    "compute_energy_score",
    "is_ood_energy",
    "MahalanobisOODDetector",
    "RetinalModalityGate",
    "validate_retinal_modality"
]
