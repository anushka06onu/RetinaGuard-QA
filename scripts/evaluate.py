"""Evaluation runner script reporting internal EyeQ and external DeepDRiD metrics with strict data integrity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.retinaguard.data.datasets import RetinalQualityDataset
from src.retinaguard.evaluation.bootstrap import compute_patient_bootstrap_ci
from src.retinaguard.evaluation.metrics import compute_quality_metrics
from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel


def evaluate_dataset_partition(
    model, csv_path: str, dataset_name: str, is_deepdrid: bool = False, device: str = "cpu"
):
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Split manifest file not found: {csv_path}. Genuine evaluation requires verified split manifests."
        )

    df = pd.read_csv(p)
    if len(df) == 0:
        raise ValueError(f"Split manifest {csv_path} is empty.")

    ds = RetinalQualityDataset(df, allow_synthetic_fallback=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    all_logits, all_y, all_masks, all_p = [], [], [], []
    attr_data = {
        "artifact": {"preds": [], "targets": [], "masks": []},
        "clarity": {"preds": [], "targets": [], "masks": []},
        "field_definition": {"preds": [], "targets": [], "masks": []},
    }

    with torch.no_grad():
        for b in loader:
            out = model(b["image"].to(device))
            if is_deepdrid:
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

    logits_arr = np.concatenate(all_logits, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)
    patients_arr = np.array(all_p)

    # Filter strictly by valid mask (> 0.5) to ignore missing-label placeholders
    valid_idx = masks_arr > 0.5
    if np.any(valid_idx):
        valid_logits = logits_arr[valid_idx]
        valid_y = y_arr[valid_idx]
        valid_patients = patients_arr[valid_idx]

        metrics = compute_quality_metrics(valid_logits, valid_y, is_logits=True)
        ci = compute_patient_bootstrap_ci(
            valid_y, np.argmax(valid_logits, axis=-1), patient_ids=valid_patients
        )
        metrics["macro_f1_95_ci"] = ci
        metrics["num_samples"] = int(np.sum(valid_idx))
    else:
        metrics = {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "macro_f1_95_ci": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "num_samples": 0,
        }

    metrics["dataset"] = dataset_name
    metrics["total_records_in_split"] = len(y_arr)

    if is_deepdrid:
        from sklearn.metrics import f1_score

        for attr_name, data in attr_data.items():
            if data["preds"] and data["targets"]:
                p_arr = np.concatenate(data["preds"], axis=0)
                t_arr = np.concatenate(data["targets"], axis=0)
                m_arr = np.concatenate(data["masks"], axis=0)
                v_attr = m_arr > 0.5
                if np.any(v_attr):
                    metrics[f"{attr_name}_macro_f1"] = round(
                        float(
                            f1_score(t_arr[v_attr], p_arr[v_attr], average="macro", zero_division=0)
                        ),
                        4,
                    )
                    metrics[f"{attr_name}_num_samples"] = int(np.sum(v_attr))
                else:
                    metrics[f"{attr_name}_macro_f1"] = 0.0
                    metrics[f"{attr_name}_num_samples"] = 0

    return metrics


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
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {args.checkpoint}. Scientific evaluation cannot run on missing weights."
        )

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"=== Evaluating RetinaGuard Checkpoint: {ckpt_p} ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    # 1. Internal EyeQ Test Evaluation
    print(f"Loading EyeQ test split from {args.eyeq_split}...")
    eyeq_metrics = evaluate_dataset_partition(model, args.eyeq_split, "EyeQ (Internal Test)")
    with open(out_p / "internal_eyeq.json", "w", encoding="utf-8") as f:
        json.dump(eyeq_metrics, f, indent=2)
    print(
        f"EyeQ Internal Macro-F1: {eyeq_metrics['macro_f1']} (95% CI: [{eyeq_metrics['macro_f1_95_ci']['ci_lower']}, {eyeq_metrics['macro_f1_95_ci']['ci_upper']}])"
    )

    # 2. DeepDRiD Held-Out Evaluation
    print(f"Loading DeepDRiD evaluation split from {args.deepdrid_split}...")
    deepdrid_metrics = evaluate_dataset_partition(
        model, args.deepdrid_split, "DeepDRiD (Held-Out Evaluation)", is_deepdrid=True
    )
    with open(out_p / "held_out_deepdrid.json", "w", encoding="utf-8") as f:
        json.dump(deepdrid_metrics, f, indent=2)
    print(
        f"DeepDRiD Held-Out Macro-F1: {deepdrid_metrics['macro_f1']} (95% CI: [{deepdrid_metrics['macro_f1_95_ci']['ci_lower']}, {deepdrid_metrics['macro_f1_95_ci']['ci_upper']}])"
    )


if __name__ == "__main__":
    main()
