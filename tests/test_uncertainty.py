"""Unit tests for calibration, ECE, Brier score, and selective prediction."""

import numpy as np
from scipy.special import softmax
import pytest

from src.uncertainty.metrics import compute_expected_calibration_error, compute_brier_score, compute_shannon_entropy
from src.uncertainty.calibration import fit_temperature_scaling
from src.uncertainty.selective_prediction import compute_risk_coverage_curve, find_abstention_threshold, evaluate_selective_prediction


def test_ece_and_brier():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8]
    ])

    ece_res = compute_expected_calibration_error(probs, labels)
    assert "ece" in ece_res
    assert 0.0 <= ece_res["ece"] <= 1.0
    assert ece_res["accuracy"] == 1.0

    brier = compute_brier_score(probs, labels)
    assert brier >= 0.0


def test_temperature_scaling():
    logits = np.random.randn(30, 3)
    labels = np.random.randint(0, 3, 30)

    t = fit_temperature_scaling(logits, labels)
    assert t > 0.05


def test_selective_prediction():
    confidences = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    preds = np.array([0, 1, 2, 0, 1])
    labels = np.array([0, 1, 2, 1, 0]) # First 3 correct, last 2 wrong

    curve = compute_risk_coverage_curve(confidences, preds, labels)
    assert curve["aurc"] >= 0.0
    assert len(curve["coverages"]) == 5

    # At threshold 0.70, first 3 accepted (100% accuracy)
    eval_res = evaluate_selective_prediction(confidences, preds, labels, threshold=0.70)
    assert eval_res["selective_accuracy"] == 1.0
    assert eval_res["coverage"] == 0.6
