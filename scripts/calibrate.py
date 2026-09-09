"""Post-hoc probability calibration and selective prediction analysis runner on real validation data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from src.retinaguard.data.datasets import RetinalQualityDataset
from src.retinaguard.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
    fit_temperature_scaling,
)
from src.retinaguard.evaluation.selective import (
    compute_risk_coverage_curve,
    evaluate_selective_abstention,
)
from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate probability outputs and evaluate selective prediction on validation data."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="data/splits/eyeq_val.csv",
        help="Path to EyeQ validation split CSV",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {args.checkpoint}. Calibration requires a genuine trained model."
        )

    val_p = Path(args.val_split)
    if not val_p.is_file():
        raise FileNotFoundError(
            f"Validation split file not found at {args.val_split}. Calibration requires verified validation data."
        )

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Probability Calibration on {val_p} with {ckpt_p} ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    val_df = pd.read_csv(val_p)
    ds = RetinalQualityDataset(val_df, allow_synthetic_fallback=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["image"])
            all_logits.append(out["quality_logits"].cpu().numpy())
            all_targets.append(b["quality_target"].numpy())

    raw_logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_targets, axis=0)

    # 1. Evaluate uncalibrated predictions
    raw_probs = softmax(raw_logits, axis=-1)
    uncal_ece = compute_ece(raw_probs, labels)
    uncal_brier = compute_brier_score(raw_probs, labels)

    # 2. Fit optimal temperature parameter T on validation set
    best_temp = fit_temperature_scaling(raw_logits, labels)
    cal_logits = raw_logits / best_temp
    cal_probs = softmax(cal_logits, axis=-1)

    cal_ece = compute_ece(cal_probs, labels)
    cal_brier = compute_brier_score(cal_probs, labels)

    # 3. Selective prediction evaluation
    confs = np.max(cal_probs, axis=-1)
    preds = np.argmax(cal_probs, axis=-1)
    rc_curve = compute_risk_coverage_curve(confs, preds, labels)
    abstention_eval = evaluate_selective_abstention(
        confs, preds, labels, min_confidence_threshold=0.75
    )

    cal_results = {
        "checkpoint": str(ckpt_p),
        "validation_split": str(val_p),
        "num_validation_samples": len(labels),
        "optimal_temperature": round(float(best_temp), 4),
        "uncalibrated_ece": uncal_ece,
        "calibrated_ece": cal_ece,
        "uncalibrated_brier": uncal_brier,
        "calibrated_brier": cal_brier,
        "aurc": rc_curve["aurc"],
        "selective_abstention_at_0_75": abstention_eval,
    }

    with open(out_p / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_results, f, indent=2)

    with open(out_p / "selective_prediction.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "aurc": rc_curve["aurc"],
                "selective_abstention": abstention_eval,
                "coverages": rc_curve["coverages"].tolist()[:20],
                "accuracies": rc_curve["accuracies"].tolist()[:20],
                "risks": rc_curve["risks"].tolist()[:20],
            },
            f,
            indent=2,
        )

    # Save calibration temperature metadata
    cal_meta_p = Path("artifacts/models/calibration_metadata.json")
    cal_meta_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_meta_p, "w", encoding="utf-8") as f:
        json.dump({"temperature": round(float(best_temp), 4)}, f, indent=2)

    print(f"Optimal Temperature T: {best_temp:.4f}")
    print(f"Uncalibrated ECE:      {uncal_ece['ece']:.4f} -> Calibrated ECE: {cal_ece['ece']:.4f}")
    print(f"Area Under Risk-Coverage (AURC): {rc_curve['aurc']:.4f}")
    print("Exported results to artifacts/metrics/calibration.json and selective_prediction.json")


if __name__ == "__main__":
    main()
