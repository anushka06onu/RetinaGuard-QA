"""Patient-level and sample-level non-parametric bootstrap confidence intervals."""

from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.metrics import f1_score


def compute_patient_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    metric_fn: Callable = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Compute 95% bootstrap confidence intervals, grouping by patient ID when provided."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Observed point estimate calculated on the complete test set (Item 22)
    observed_score = float(metric_fn(y_true, y_pred)) if n > 0 else 0.0

    if patient_ids is not None:
        unique_patients = np.unique(patient_ids)
        patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}
        num_p = len(unique_patients)
    else:
        num_p = None

    scores = []
    for _ in range(n_bootstraps):
        if num_p is not None:
            # Resample patients with replacement
            sampled_p = rng.choice(unique_patients, size=num_p, replace=True)
            sampled_idx = np.concatenate([patient_to_idx[p] for p in sampled_p])
        else:
            # Resample samples with replacement
            sampled_idx = rng.choice(n, size=n, replace=True)

        if len(np.unique(y_true[sampled_idx])) < 2:
            continue

        score = metric_fn(y_true[sampled_idx], y_pred[sampled_idx])
        scores.append(score)

    if len(scores) == 0:
        return {
            "point_estimate": round(observed_score, 4),
            "observed_metric": round(observed_score, 4),
            "bootstrap_mean": round(observed_score, 4),
            "ci_lower": round(observed_score, 4),
            "ci_upper": round(observed_score, 4),
            "ci_level": ci,
            "num_bootstraps": 0,
            "num_patients": num_p if num_p is not None else n,
            "seed": seed,
        }

    scores_arr = np.array(scores)
    boot_mean = float(np.mean(scores_arr))
    lower = float(np.percentile(scores_arr, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(scores_arr, (1.0 + ci) / 2.0 * 100))

    return {
        "point_estimate": round(observed_score, 4),
        "observed_metric": round(observed_score, 4),
        "bootstrap_mean": round(boot_mean, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "ci_level": ci,
        "num_bootstraps": len(scores),
        "num_patients": num_p if num_p is not None else n,
        "seed": seed,
    }
