"""Post-hoc probability calibration and selective prediction analysis runner on real validation data."""

import argparse
import datetime
import json
import sys
from pathlib import Path

# Ensure local package in src/ takes precedence over any installed egg
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from retinaguard.data.datasets import RetinalQualityDataset
from retinaguard.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
    fit_temperature_scaling,
)
from retinaguard.evaluation.ood import compute_energy_score
from retinaguard.evaluation.selective import (
    compute_risk_coverage_curve,
    evaluate_selective_abstention,
)
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.utils.hashing import compute_sha256


def calibrate_temperature_and_thresholds(
    checkpoint_path: str,
    val_split_path: str,
    task: str = "deepdrid_overall",
    task_head: str = None,
    trained_heads: list = None,
    output_dir: str = "artifacts/metrics",
    onnx_model_path: str = "artifacts/models/model.onnx",
) -> dict:
    """Run temperature scaling and selective prediction calibration with rigorous validation checks."""
    effective_head = task_head or (
        "quality_logits" if task == "eyeq_quality" else "overall_quality_logits"
    )

    # Untrained head rejection check
    if trained_heads is not None and effective_head not in trained_heads:
        raise ValueError(
            f"Calibration error: requested head '{effective_head}' is not among the model's trained heads: {trained_heads}."
        )

    # Cross-dataset validation split mismatch checks
    val_str = str(val_split_path).lower()
    if effective_head == "quality_logits" and "deepdrid" in val_str:
        raise ValueError(
            f"Cross-dataset validation mismatch: cannot calibrate EyeQ quality_logits on DeepDRiD split '{val_split_path}'."
        )
    if effective_head != "quality_logits" and "eyeq" in val_str:
        raise ValueError(
            f"Cross-dataset validation mismatch: cannot calibrate DeepDRiD head '{effective_head}' on EyeQ split '{val_split_path}'."
        )

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {checkpoint_path}. Calibration requires a genuine trained model."
        )

    val_p = Path(val_split_path)
    if not val_p.is_file():
        raise FileNotFoundError(
            f"Validation split file not found at {val_split_path}. Calibration requires verified validation data."
        )

    ckpt_state = (
        torch.load(ckpt_p, map_location="cpu", weights_only=False) if hasattr(torch, "load") else {}
    )
    ckpt_meta = ckpt_state.get("metadata", {}) if isinstance(ckpt_state, dict) else {}
    model_trained_heads = ckpt_meta.get("trained_heads", [])

    if model_trained_heads and effective_head not in model_trained_heads:
        raise ValueError(
            f"Calibration error: requested head '{effective_head}' is not among the model's trained heads: {model_trained_heads}."
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Running Probability Calibration [Head: {effective_head}] on {val_p} with {ckpt_p} ==="
    )
    model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_p)
    model.eval()

    val_df = pd.read_csv(val_p)
    ds = RetinalQualityDataset(val_df, allow_synthetic_fallback=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    all_logits, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["image"])
            if effective_head == "quality_logits":
                logits = out["quality_logits"].cpu().numpy()
                targets = b["quality_target"].numpy()
                masks = b["quality_mask"].numpy()
            else:
                logits = out["overall_quality_logits"].cpu().numpy()
                targets = b["overall_quality_target"].numpy()
                masks = b["overall_quality_mask"].numpy()

            all_logits.append(logits)
            all_targets.append(targets)
            all_masks.append(masks)

    raw_logits_all = np.concatenate(all_logits, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)
    masks_all = np.concatenate(all_masks, axis=0)

    # Filter strictly by valid mask (> 0.5)
    valid_idx = masks_all > 0.5
    valid_count = int(np.sum(valid_idx))
    if valid_count == 0:
        raise ValueError(
            f"Zero valid examples found for head '{effective_head}' in validation split {val_p}. "
            "Calibration cannot proceed without valid ground-truth labels."
        )

    raw_logits = raw_logits_all[valid_idx]
    labels = targets_all[valid_idx]

    num_classes = 3 if effective_head == "quality_logits" else 2
    class_order = (
        ["good", "usable", "reject"]
        if effective_head == "quality_logits"
        else ["good", "poor_or_reject"]
    )

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

    # 4. Fit decision policy thresholds on validation set
    eps = 1e-12
    p_clipped = np.clip(cal_probs, eps, 1.0)
    entropies = -np.sum(p_clipped * (np.log(p_clipped) / np.log(2.0)), axis=-1)
    energy_scores = compute_energy_score(raw_logits, temperature=best_temp)

    fitted_uncertainty_threshold = round(float(np.percentile(entropies, 95)), 4)
    fitted_ood_threshold = round(float(np.percentile(energy_scores, 5)), 4)

    ckpt_sha = compute_sha256(ckpt_p) if ckpt_p.is_file() else None
    split_sha = compute_sha256(val_p) if val_p.is_file() else None
    onnx_p = Path(onnx_model_path)
    onnx_sha = compute_sha256(onnx_p) if onnx_p.is_file() else None

    cal_results = {
        "dataset": "EyeQ" if effective_head == "quality_logits" else "DeepDRiD",
        "task": task,
        "output_head": effective_head,
        "class_order": class_order,
        "valid_sample_count": valid_count,
        "split_path": str(val_p),
        "split_sha256": split_sha,
        "source_checkpoint": str(ckpt_p),
        "source_checkpoint_sha256": ckpt_sha,
        "onnx_model_sha256": onnx_sha,
        "optimal_temperature": round(float(best_temp), 4),
        "uncertainty_definition": "predictive_entropy_base_2_bits",
        "uncertainty_threshold": fitted_uncertainty_threshold,
        "ood_energy_threshold": fitted_ood_threshold,
        "threshold_selection_method": "temperature_scaling_lbfgs_and_empirical_percentiles",
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

    # Save complete production calibration & decision metadata
    import subprocess

    def get_git_commit_sha() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"

    cal_meta_p = Path("artifacts/models/calibration_metadata.json")
    cal_meta_p.parent.mkdir(parents=True, exist_ok=True)
    calibration_metadata = {
        "schema_version": "1.0.0",
        "status": "completed",
        "eligible_as_final_result": True,
        "generated_by": "scripts/calibrate.py",
        "git_commit": get_git_commit_sha(),
        "dataset": "EyeQ" if effective_head == "quality_logits" else "DeepDRiD",
        "task": task,
        "output_head": effective_head,
        "class_order": class_order,
        "valid_sample_count": valid_count,
        "temperature": round(float(best_temp), 4),
        "uncertainty_threshold": fitted_uncertainty_threshold,
        "uncertainty_unit": "bits",
        "uncertainty_definition": "predictive_entropy_base_2_bits",
        "entropy_max": round(float(np.log2(num_classes)), 5),
        "ood_energy_threshold": fitted_ood_threshold,
        "ood_score_type": "energy",
        "ood_direction": "lower_is_ood",
        "validation_split": str(val_p),
        "validation_split_sha256": split_sha,
        "source_checkpoint": str(ckpt_p),
        "source_checkpoint_sha256": ckpt_sha,
        "model_checkpoint_sha256": ckpt_sha,
        "onnx_model_sha256": onnx_sha,
        "fitting_method": "temperature_scaling_lbfgs_and_empirical_percentiles",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(cal_meta_p, "w", encoding="utf-8") as f:
        json.dump(calibration_metadata, f, indent=2)

    print(f"Optimal Temperature T: {best_temp:.4f}")
    print(f"Uncertainty Threshold (95th percentile): {fitted_uncertainty_threshold:.4f} bits")
    print(f"OOD Energy Threshold (5th percentile): {fitted_ood_threshold:.4f}")
    print(f"Uncalibrated ECE:      {uncal_ece['ece']:.4f} -> Calibrated ECE: {cal_ece['ece']:.4f}")
    print(f"Area Under Risk-Coverage (AURC): {rc_curve['aurc']:.4f}")
    print(
        "Exported results to artifacts/metrics/calibration.json and artifacts/models/calibration_metadata.json"
    )
    return cal_results


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
        default="data/splits/deepdrid_val.csv",
        help="Path to validation split CSV",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="deepdrid_overall",
        choices=["eyeq_quality", "deepdrid_overall"],
        help="Explicit calibration task: 'eyeq_quality' (3-class) or 'deepdrid_overall' (2-class)",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="artifacts/models/model.onnx",
        help="Optional path to exported ONNX model to record onnx_model_sha256",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    args = parser.parse_args()

    calibrate_temperature_and_thresholds(
        checkpoint_path=args.checkpoint,
        val_split_path=args.val_split,
        task=args.task,
        output_dir=args.output_dir,
        onnx_model_path=args.onnx_model,
    )


if __name__ == "__main__":
    main()
