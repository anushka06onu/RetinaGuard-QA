"""Comprehensive statistical metrics engine matching Phase 10 of blueprint."""

from typing import Any, Dict, List

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_quality_metrics(
    logits_or_probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = ["good", "usable", "reject"],
    is_logits: bool = True,
) -> Dict[str, Any]:
    """Compute primary 3-class quality metrics."""
    if is_logits:
        probs = softmax(logits_or_probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
    else:
        probs = logits_or_probs
        preds = np.argmax(probs, axis=-1)

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    acc = float(accuracy_score(labels, preds))

    # Quadratic weighted kappa (QWK)
    qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist()

    # Per-class metrics
    p_per = precision_score(labels, preds, average=None, zero_division=0)
    r_per = recall_score(labels, preds, average=None, zero_division=0)
    f1_per = f1_score(labels, preds, average=None, zero_division=0)

    per_class_f1 = {name: float(f1_per[i]) for i, name in enumerate(class_names)}
    per_class_p = {name: float(p_per[i]) for i, name in enumerate(class_names)}
    per_class_r = {name: float(r_per[i]) for i, name in enumerate(class_names)}

    ovr_auroc = None
    if probs is not None:
        try:
            ovr_auroc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except Exception:
            ovr_auroc = None

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "confusion_matrix": cm,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_p,
        "per_class_recall": per_class_r,
        "ovr_auroc": round(ovr_auroc, 4) if ovr_auroc is not None else None,
    }


def compute_attribute_metrics(
    preds_dict: Dict[str, np.ndarray], targets_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Compute metrics for DeepDRiD ordinal attributes (artifact, clarity, field_definition)."""
    results = {}
    for attr_name in ["artifact", "clarity", "field_definition"]:
        if attr_name in preds_dict and attr_name in targets_dict:
            y_pred = preds_dict[attr_name]
            y_true = targets_dict[attr_name]
            if len(y_true) > 0:
                f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
                mae = float(np.mean(np.abs(y_true - y_pred)))
                results[attr_name] = {
                    "macro_f1": round(f1, 4),
                    "qwk": round(qwk, 4),
                    "mae": round(mae, 4),
                }
    return results
