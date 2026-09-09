"""Out-of-Distribution detection, Energy Scoring, and Modality Validation per Phase 13."""

from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from scipy.special import logsumexp


def compute_energy_score(
    logits: Union[np.ndarray, torch.Tensor], temperature: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """Compute Energy Score S_energy(x) = T * LogSumExp(f(x) / T)."""
    if isinstance(logits, torch.Tensor):
        return temperature * torch.logsumexp(logits / temperature, dim=-1)
    else:
        return temperature * logsumexp(logits / temperature, axis=-1)


def is_ood_sample(
    logits: Union[np.ndarray, torch.Tensor], threshold: float = 1.0, temperature: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """Classify as OOD if energy score is below threshold."""
    energy = compute_energy_score(logits, temperature=temperature)
    return energy < threshold


class MahalanobisOOD:
    """Class-conditional Gaussian Mahalanobis distance in latent feature space."""

    def __init__(self, num_classes: int = 3, feature_dim: int = 128):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_means: List[np.ndarray] = []
        self.precision: np.ndarray = np.eye(feature_dim, dtype=np.float32)
        self.is_fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.class_means = []
        total_cov = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float32)

        for c in range(self.num_classes):
            c_feats = features[labels == c]
            if len(c_feats) == 0:
                mean_c = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                mean_c = np.mean(c_feats, axis=0)
            self.class_means.append(mean_c)
            diff = c_feats - mean_c
            total_cov += np.dot(diff.T, diff)

        total_cov = total_cov / max(1, len(features))
        total_cov += np.eye(self.feature_dim, dtype=np.float32) * 1e-4
        self.precision = np.linalg.pinv(total_cov)
        self.is_fitted = True

    def compute_distance(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.linalg.norm(features, axis=-1)

        b = features.shape[0]
        min_dists = np.full(b, np.inf, dtype=np.float32)

        for c in range(self.num_classes):
            diff = features - self.class_means[c]
            term = np.dot(diff, self.precision)
            dists = np.sum(term * diff, axis=-1)
            min_dists = np.minimum(min_dists, dists)

        return np.sqrt(np.maximum(min_dists, 0.0))


class RetinalModalityValidator:
    """Zero-shot domain validation gate inspecting chromatic/geometric properties."""

    @staticmethod
    def validate(image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            arr = np.array(image, dtype=np.float32)

        h, w, c = arr.shape
        if h < 32 or w < 32:
            return {"is_fundus": False, "confidence": 0.0, "reason": "Resolution too small"}

        mean_r = float(np.mean(arr[:, :, 0]))
        mean_g = float(np.mean(arr[:, :, 1]))
        mean_b = float(np.mean(arr[:, :, 2]))

        if mean_r < 2.0 and mean_g < 2.0 and mean_b < 2.0:
            return {"is_fundus": False, "confidence": 0.0, "reason": "Unilluminated / Black image"}

        is_grayscale = (abs(mean_r - mean_g) < 4.0) and (abs(mean_g - mean_b) < 4.0)
        rb_ratio = mean_r / max(1.0, mean_b)

        if is_grayscale:
            return {
                "is_fundus": False,
                "confidence": 0.15,
                "reason": "Monochromatic non-fundus image (X-Ray/Microscopy)",
            }

        if rb_ratio < 1.05 and mean_b > mean_r:
            return {"is_fundus": False, "confidence": 0.25, "reason": "Inverted chromatic profile"}

        conf = min(1.0, 0.5 + 0.5 * (rb_ratio / 2.0))
        return {
            "is_fundus": True,
            "confidence": round(float(conf), 3),
            "reason": "Passed retinal modality validation",
        }
