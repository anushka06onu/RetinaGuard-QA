"""Class-conditional Gaussian Mahalanobis distance OOD detector on latent representations (Lee et al., NeurIPS 2018)."""

from typing import List, Tuple, Union
import numpy as np
import torch


class MahalanobisOODDetector:
    """Computes Mahalanobis distance to training class centroids in deep feature space."""

    def __init__(self, num_classes: int = 3, feature_dim: int = 128):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_means: List[np.ndarray] = []
        self.precision: np.ndarray = np.eye(feature_dim, dtype=np.float32)
        self.is_fitted: bool = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fit empirical class means and shared covariance matrix.

        Args:
            features: [N, D] array of extracted latent features.
            labels: [N] array of class indices.
        """
        self.class_means = []
        total_cov = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float32)

        for c in range(self.num_classes):
            c_features = features[labels == c]
            if len(c_features) == 0:
                mean_c = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                mean_c = np.mean(c_features, axis=0)
            self.class_means.append(mean_c)

            # Center and accumulate covariance
            diff = c_features - mean_c
            cov_c = np.dot(diff.T, diff)
            total_cov += cov_c

        # Tied covariance matrix with regularization
        total_cov = total_cov / max(1, len(features))
        total_cov += np.eye(self.feature_dim, dtype=np.float32) * 1e-4

        self.precision = np.linalg.pinv(total_cov)
        self.is_fitted = True

    def compute_min_mahalanobis_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance across all class clusters for each sample.

        Args:
            features: [B, D] feature embeddings.

        Returns:
            [B] array of minimum distances (higher distance = more likely OOD).
        """
        if not self.is_fitted:
            # Return Euclidean distance to zero if not fitted
            return np.linalg.norm(features, axis=-1)

        b = features.shape[0]
        min_distances = np.full(b, np.inf, dtype=np.float32)

        for c in range(self.num_classes):
            mean_c = self.class_means[c]
            diff = features - mean_c  # [B, D]
            # d^2 = sum( (diff * precision) * diff, axis=-1 )
            term = np.dot(diff, self.precision)
            dists = np.sum(term * diff, axis=-1)
            min_distances = np.minimum(min_distances, dists)

        return np.sqrt(np.maximum(min_distances, 0.0))
