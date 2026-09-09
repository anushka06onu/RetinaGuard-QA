"""Comprehensive statistical metrics engine for multiclass quality grading and multi-label defect detection."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from scipy.special import softmax


@dataclass
class EvaluationReport:
    """Standardized evaluation results container."""
    dataset_name: str
    num_samples: int
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    accuracy: float
    cohen_weighted_kappa: float
    confusion_matrix: List[List[int]]
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    ovr_auroc: Optional[float] = None
    ece: Optional[float] = None
    brier_score: Optional[float] = None


def compute_multiclass_metrics(
    predictions_or_probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = ["Good", "Usable", "Reject"],
    is_probabilities: bool = False
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """Compute complete statistical evaluation suite for 3-class quality grading.

    Includes: Macro-F1, Balanced Accuracy, Weighted Kappa (quadratic weights),
    Per-class precision/recall, AUROC, and Confusion Matrix.
    """
    if is_probabilities or (predictions_or_probs.ndim == 2 and predictions_or_probs.shape[1] > 1):
        probs = predictions_or_probs
        preds = np.argmax(probs, axis=1)
    else:
        probs = None
        preds = predictions_or_probs

    # Global multi-class metrics
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    acc = float(accuracy_score(labels, preds))
    
    # Quadratic weighted Cohen's Kappa for ordinal grading
    kappa = float(cohen_kappa_score(labels, preds, weights="quadratic"))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist()

    # Per-class metrics
    p_per = precision_score(labels, preds, average=None, zero_division=0)
    r_per = recall_score(labels, preds, average=None, zero_division=0)
    f1_per = f1_score(labels, preds, average=None, zero_division=0)

    per_class_f1 = {name: float(f1_per[i]) for i, name in enumerate(class_names)}
    per_class_p = {name: float(p_per[i]) for i, name in enumerate(class_names)}
    per_class_r = {name: float(r_per[i]) for i, name in enumerate(class_names)}

    # AUROC One-vs-Rest if probabilities are supplied
    ovr_auroc = None
    if probs is not None:
        try:
            ovr_auroc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except Exception:
            ovr_auroc = None

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "cohen_weighted_kappa": kappa,
        "confusion_matrix": cm,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_p,
        "per_class_recall": per_class_r,
        "ovr_auroc": ovr_auroc
    }


def compute_defect_metrics(
    defect_probs: np.ndarray,
    defect_labels: np.ndarray,
    defect_names: List[str]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute multi-label defect classification performance (mAP, per-defect AUROC, Micro/Macro F1)."""
    preds = (defect_probs >= 0.5).astype(int)
    
    macro_f1 = float(f1_score(defect_labels, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(defect_labels, preds, average="micro", zero_division=0))

    per_defect_auroc = {}
    per_defect_ap = {}

    for i, name in enumerate(defect_names):
        y_true = defect_labels[:, i]
        y_score = defect_probs[:, i]
        if len(np.unique(y_true)) > 1:
            per_defect_auroc[name] = float(roc_auc_score(y_true, y_score))
            per_defect_ap[name] = float(average_precision_score(y_true, y_score))
        else:
            per_defect_auroc[name] = 1.0
            per_defect_ap[name] = 1.0

    mean_ap = float(np.mean(list(per_defect_ap.values())))

    return {
        "defect_macro_f1": macro_f1,
        "defect_micro_f1": micro_f1,
        "mean_average_precision": mean_ap,
        "per_defect_auroc": per_defect_auroc,
        "per_defect_ap": per_defect_ap
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=accuracy_score,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)

    scores = np.array(scores)
    point_est = float(np.mean(scores))
    lower = float(np.percentile(scores, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(scores, (1.0 + ci) / 2.0 * 100))

    return point_est, lower, upper
