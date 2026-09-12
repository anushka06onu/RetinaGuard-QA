"""Evaluation runner script reporting internal EyeQ, held-out DeepDRiD, and zero-shot transfer metrics."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from retinaguard.data.datasets import RetinalQualityDataset
from retinaguard.data.preprocessing import get_val_transforms
from retinaguard.evaluation.bootstrap import compute_patient_bootstrap_ci
from retinaguard.evaluation.metrics import (
    compute_attribute_metrics,
    compute_quality_metrics,
)
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.utils.hashing import compute_sha256


def evaluate_dataset_partition(
    model: torch.nn.Module,
    csv_path: str,
    dataset_name: str,
    is_deepdrid: bool = False,
    is_zero_shot: bool = False,
    device: str = "cpu",
    image_size: int = 384,
    fail_on_empty: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate model on a dataset partition with strict label masking and structured return."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Split manifest file not found: {csv_path}. Evaluation requires verified split manifests."
        )

    df = pd.read_csv(p)
    if len(df) == 0:
        if fail_on_empty:
            raise ValueError(f"Split manifest {csv_path} is empty.")
        return {
            "status": "not_evaluable",
            "reason": "empty_manifest",
            "num_samples": 0,
            "metrics": None,
        }, pd.DataFrame()

    val_transform = get_val_transforms(image_size=image_size)
    ds = RetinalQualityDataset(df, transform=val_transform, allow_synthetic_fallback=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    all_logits, all_y, all_masks, all_p, all_ids = [], [], [], [], []
    attr_data = {
        "artifact": {"preds": [], "targets": [], "masks": []},
        "clarity": {"preds": [], "targets": [], "masks": []},
        "field_definition": {"preds": [], "targets": [], "masks": []},
    }

    with torch.no_grad():
        for b in loader:
            out = model(b["image"].to(device))
            if is_zero_shot:
                # In Zero-Shot transfer from EyeQ to DeepDRiD:
                # Model outputs 3-class EyeQ logits: 0=good, 1=usable, 2=reject.
                # Pre-declared binary mapping: Acceptable (Good+Usable) -> 0, Reject -> 1.
                q3_logits = out["quality_logits"].cpu().numpy()
                q3_probs = softmax(q3_logits, axis=-1)
                p_good = q3_probs[:, 0] + q3_probs[:, 1]  # Good / Usable
                p_reject = q3_probs[:, 2]  # Reject
                binary_probs = np.stack([p_good, p_reject], axis=-1)
                logits = np.log(np.clip(binary_probs, 1e-12, 1.0))
                targets = b["overall_quality_target"].numpy()
                masks = b["overall_quality_mask"].numpy()
            elif is_deepdrid:
                logits = out["overall_quality_logits"].cpu().numpy()
                targets = b["overall_quality_target"].numpy()
                masks = b["overall_quality_mask"].numpy()
                for attr_name in ["artifact", "clarity", "field_definition"]:
                    attr_logits = out[f"{attr_name}_logits"].cpu().numpy()
                    attr_target = b[f"{attr_name}_target"].numpy()
                    attr_mask = b[f"{attr_name}_mask"].numpy()
                    attr_data[attr_name]["preds"].append(np.argmax(attr_logits, axis=-1))
                    attr_data[attr_name]["targets"].append(attr_target)
                    attr_data[attr_name]["masks"].append(attr_mask)
            else:
                logits = out["quality_logits"].cpu().numpy()
                targets = b["quality_target"].numpy()
                masks = b["quality_mask"].numpy()

            all_logits.append(logits)
            all_y.append(targets)
            all_masks.append(masks)
            all_p.extend(b["patient_id"])
            all_ids.extend(b["image_id"])

    logits_arr = np.concatenate(all_logits, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)
    patients_arr = np.array(all_p)
    ids_arr = np.array(all_ids)

    # Filter strictly by valid mask (> 0.5)
    valid_idx = masks_arr > 0.5
    valid_count = int(np.sum(valid_idx))

    if valid_count == 0:
        if fail_on_empty:
            raise ValueError(
                f"No valid labeled records for partition {dataset_name} in {csv_path}."
            )
        return {
            "status": "not_evaluable",
            "reason": "no_valid_labels",
            "num_samples": 0,
            "metrics": None,
        }, pd.DataFrame()

    valid_logits = logits_arr[valid_idx]
    valid_y = y_arr[valid_idx]
    valid_patients = patients_arr[valid_idx]
    valid_ids = ids_arr[valid_idx]

    class_names = (
        ["good", "poor_reject"] if (is_deepdrid or is_zero_shot) else ["good", "usable", "reject"]
    )

    metrics = compute_quality_metrics(
        valid_logits, valid_y, class_names=class_names, is_logits=True
    )
    ci = compute_patient_bootstrap_ci(
        valid_y, np.argmax(valid_logits, axis=-1), patient_ids=valid_patients
    )
    metrics["macro_f1_95_ci"] = ci
    metrics["num_samples"] = valid_count
    metrics["num_patients"] = (
        len(np.unique(valid_patients)) if len(valid_patients) > 0 else valid_count
    )
    metrics["dataset"] = dataset_name
    metrics["total_records_in_split"] = len(y_arr)
    metrics["status"] = "evaluated"

    # Compute per-sample predictions DataFrame
    valid_probs = softmax(valid_logits, axis=-1)
    preds = np.argmax(valid_logits, axis=-1)
    confs = np.max(valid_probs, axis=-1)
    eps = 1e-12
    entropies = -np.sum(valid_probs * np.log2(np.clip(valid_probs, eps, 1.0)), axis=-1)
    energy_scores = -np.log(np.sum(np.exp(valid_logits), axis=-1))

    df_preds = pd.DataFrame(
        {
            "image_id": valid_ids,
            "patient_id": valid_patients,
            "dataset": dataset_name,
            "split": Path(csv_path).name,
            "target": valid_y,
            "prediction": preds,
            "confidence": np.round(confs, 6),
            "entropy": np.round(entropies, 6),
            "energy_score": np.round(energy_scores, 6),
        }
    )

    if is_deepdrid and not is_zero_shot:
        filtered_preds = {}
        filtered_targets = {}
        for attr_name, data in attr_data.items():
            if data["preds"] and data["targets"]:
                p_arr = np.concatenate(data["preds"], axis=0)[valid_idx]
                t_arr = np.concatenate(data["targets"], axis=0)[valid_idx]
                m_arr = np.concatenate(data["masks"], axis=0)[valid_idx]
                v_attr = m_arr > 0.5
                if np.any(v_attr):
                    filtered_preds[attr_name] = p_arr[v_attr]
                    filtered_targets[attr_name] = t_arr[v_attr]
                df_preds[f"{attr_name}_target"] = t_arr
                df_preds[f"{attr_name}_prediction"] = p_arr
                df_preds[f"{attr_name}_mask"] = m_arr

        attr_metrics = compute_attribute_metrics(filtered_preds, filtered_targets)
        metrics["attribute_metrics"] = attr_metrics

    return metrics, df_preds


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RetinaGuard model checkpoint with strict verification."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument("--eyeq-split", type=str, default="data/splits/eyeq_test.csv")
    parser.add_argument(
        "--deepdrid-split", type=str, default="data/splits/deepdrid_external_test.csv"
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run genuine zero-shot transfer evaluation on DeepDRiD using EyeQ-trained model",
    )
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {args.checkpoint}. Scientific evaluation cannot run on missing weights."
        )

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    preds_dir = out_p.parent / "predictions" if out_p.name == "metrics" else out_p / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    try:
        state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_p, map_location="cpu")
    metadata = state.get("metadata", {})
    training_datasets = metadata.get("training_datasets", [])
    eval_img_size = metadata.get("resolved_config", {}).get("data", {}).get("image_size", 384)

    model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_p)
    model.eval()

    # Zero-shot verification guard (Item 6)
    if args.zero_shot:
        if training_datasets != ["EyeQ"]:
            raise ValueError(
                f"Zero-shot evaluation requires a checkpoint trained exclusively on EyeQ. "
                f"Checkpoint metadata indicates training datasets: {training_datasets}"
            )
        if "deepdrid" in metadata.get("train_split_hashes", {}) or "deepdrid" in metadata.get(
            "val_split_hashes", {}
        ):
            raise ValueError(
                "Checkpoint contains DeepDRiD split provenance; cannot be used for genuine zero-shot evaluation."
            )

    has_eyeq = Path(args.eyeq_split).is_file()
    has_deepdrid = Path(args.deepdrid_split).is_file()

    if not (has_eyeq or has_deepdrid):
        raise FileNotFoundError(
            f"Neither EyeQ split ({args.eyeq_split}) nor DeepDRiD split ({args.deepdrid_split}) was found."
        )

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    ckpt_sha = compute_sha256(ckpt_p)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Internal EyeQ Test Evaluation
    if has_eyeq and not args.zero_shot:
        print(f"Loading EyeQ test split from {args.eyeq_split}...")
        eyeq_metrics, df_eyeq_preds = evaluate_dataset_partition(
            model,
            args.eyeq_split,
            "EyeQ (Internal Test)",
            is_deepdrid=False,
            image_size=eval_img_size,
        )
        pred_file = preds_dir / "eyeq_test_predictions.csv"
        df_eyeq_preds.to_csv(pred_file, index=False)
        pred_sha = compute_sha256(pred_file)

        eyeq_metrics.update(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/evaluate.py",
                "git_commit": git_commit,
                "checkpoint_path": str(ckpt_p),
                "checkpoint_sha256": ckpt_sha,
                "split_path": args.eyeq_split,
                "split_sha256": compute_sha256(args.eyeq_split),
                "prediction_file": str(pred_file),
                "prediction_file_sha256": pred_sha,
                "created_at_utc": created_at,
            }
        )
        with open(out_p / "eyeq_test.json", "w", encoding="utf-8") as f:
            json.dump(eyeq_metrics, f, indent=2)
        print(
            f"EyeQ Test Macro-F1: {eyeq_metrics['macro_f1']} (95% CI: [{eyeq_metrics['macro_f1_95_ci']['ci_lower']}, {eyeq_metrics['macro_f1_95_ci']['ci_upper']}])"
        )

    # 2. DeepDRiD Evaluation (Held-Out Supervised or Zero-Shot External Transfer)
    if has_deepdrid:
        is_zero_shot_eval = args.zero_shot or (
            training_datasets == ["EyeQ"] and "DeepDRiD" not in training_datasets
        )

        if is_zero_shot_eval:
            print(
                f"Running Zero-Shot External Transfer on DeepDRiD ({args.deepdrid_split}) "
                f"[Model Provenance: {training_datasets}]..."
            )
            transfer_metrics, df_zs_preds = evaluate_dataset_partition(
                model,
                args.deepdrid_split,
                "DeepDRiD (Zero-Shot Transfer)",
                is_deepdrid=True,
                is_zero_shot=True,
                image_size=eval_img_size,
            )
            pred_file = preds_dir / "zero_shot_transfer_predictions.csv"
            df_zs_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            transfer_metrics.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/evaluate.py",
                    "git_commit": git_commit,
                    "checkpoint_path": str(ckpt_p),
                    "checkpoint_sha256": ckpt_sha,
                    "split_path": args.deepdrid_split,
                    "split_sha256": compute_sha256(args.deepdrid_split),
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": pred_sha,
                    "created_at_utc": created_at,
                }
            )
            with open(out_p / "zero_shot_transfer.json", "w", encoding="utf-8") as f:
                json.dump(transfer_metrics, f, indent=2)
            print(
                f"Zero-Shot DeepDRiD Binary Macro-F1: {transfer_metrics['macro_f1']} "
                f"(95% CI: [{transfer_metrics['macro_f1_95_ci']['ci_lower']}, {transfer_metrics['macro_f1_95_ci']['ci_upper']}])"
            )
        else:
            print(f"Running Supervised Evaluation on DeepDRiD ({args.deepdrid_split})...")
            deepdrid_metrics, df_dd_preds = evaluate_dataset_partition(
                model,
                args.deepdrid_split,
                "DeepDRiD (Held-out Test)",
                is_deepdrid=True,
                is_zero_shot=False,
                image_size=eval_img_size,
            )
            pred_file = preds_dir / "deepdrid_heldout_predictions.csv"
            df_dd_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            deepdrid_metrics.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/evaluate.py",
                    "git_commit": git_commit,
                    "checkpoint_path": str(ckpt_p),
                    "checkpoint_sha256": ckpt_sha,
                    "split_path": args.deepdrid_split,
                    "split_sha256": compute_sha256(args.deepdrid_split),
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": pred_sha,
                    "created_at_utc": created_at,
                }
            )
            with open(out_p / "deepdrid_heldout.json", "w", encoding="utf-8") as f:
                json.dump(deepdrid_metrics, f, indent=2)
            print(
                f"DeepDRiD Held-Out Macro-F1: {deepdrid_metrics['macro_f1']} (95% CI: [{deepdrid_metrics['macro_f1_95_ci']['ci_lower']}, {deepdrid_metrics['macro_f1_95_ci']['ci_upper']}])"
            )


if __name__ == "__main__":
    main()
