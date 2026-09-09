"""Experiment 04: Uncertainty Calibration, Expected Calibration Error (ECE), and Selective Prediction."""

import numpy as np
from scipy.special import softmax

from src.uncertainty.calibration import fit_temperature_scaling
from src.uncertainty.metrics import compute_expected_calibration_error, compute_brier_score
from src.uncertainty.selective_prediction import compute_risk_coverage_curve, find_abstention_threshold


def evaluate_calibration_and_selective_prediction():
    print("=== Running Experiment 04: Calibration & Selective Abstention ===")
    
    # Generate uncalibrated overconfident validation logits
    np.random.seed(42)
    n = 200
    labels = np.random.randint(0, 3, n)
    
    # Simulate overconfident uncalibrated logits
    raw_logits = np.random.randn(n, 3) * 3.5
    for i in range(n):
        if np.random.rand() > 0.3: # 70% accuracy
            raw_logits[i, labels[i]] += 4.0

    raw_probs = softmax(raw_logits, axis=-1)

    # 1. Uncalibrated Metrics
    uncal_ece = compute_expected_calibration_error(raw_probs, labels)
    uncal_brier = compute_brier_score(raw_probs, labels)
    print(f"Uncalibrated ECE:   {uncal_ece['ece']:.4f} (Accuracy: {uncal_ece['accuracy']:.4f}, Conf: {uncal_ece['avg_confidence']:.4f})")
    print(f"Uncalibrated Brier: {uncal_brier:.4f}")

    # 2. Fit Temperature Scaling
    opt_temp = fit_temperature_scaling(raw_logits, labels)
    cal_logits = raw_logits / opt_temp
    cal_probs = softmax(cal_logits, axis=-1)

    # 3. Calibrated Metrics
    cal_ece = compute_expected_calibration_error(cal_probs, labels)
    cal_brier = compute_brier_score(cal_probs, labels)
    print(f"\nFitted Optimal Temperature T: {opt_temp:.4f}")
    print(f"Calibrated ECE:     {cal_ece['ece']:.4f} (Reduction: {(uncal_ece['ece'] - cal_ece['ece']):.4f})")
    print(f"Calibrated Brier:   {cal_brier:.4f}")

    # 4. Selective Prediction & Risk-Coverage Curve
    confs = np.max(cal_probs, axis=-1)
    preds = np.argmax(cal_probs, axis=-1)
    rc_results = compute_risk_coverage_curve(confs, preds, labels)
    print(f"\nArea Under Risk-Coverage (AURC): {rc_results['aurc']:.4f}")

    # Abstention threshold for target accuracy
    target_acc = 0.90
    thresh_info = find_abstention_threshold(confs, preds, labels, target_accuracy=target_acc)
    print(f"To achieve {target_acc*100:.0f}% accuracy:")
    print(f"  Confidence Threshold: {thresh_info['confidence_threshold']:.4f}")
    print(f"  Accepted Coverage:    {thresh_info['achieved_coverage']*100:.1f}%")
    print(f"  Abstention Rate:      {thresh_info['abstention_rate']*100:.1f}% (Sent for Manual Review)")

    return {
        "uncalibrated_ece": uncal_ece,
        "calibrated_ece": cal_ece,
        "optimal_temperature": opt_temp,
        "abstention_profile": thresh_info
    }


if __name__ == "__main__":
    evaluate_calibration_and_selective_prediction()
