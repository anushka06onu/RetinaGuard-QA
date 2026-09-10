"""Run controlled optical corruption robustness sweep (10 types x 5 severities) with clean baseline and relative degradation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from retinaguard.data.preprocessing import preprocess_image_canonical
from retinaguard.evaluation.corruptions import SyntheticCorruptionSuite
from retinaguard.evaluation.metrics import compute_quality_metrics
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic corruption robustness benchmark on real test data."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="data/splits/eyeq_test.csv",
        help="Path to test split CSV",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="eyeq_quality",
        choices=["eyeq_quality", "deepdrid_overall"],
        help="Task for evaluation: 'eyeq_quality' (3-class) or 'deepdrid_overall' (2-class)",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum test samples to evaluate per corruption (sampled reproducibly)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sample selection")
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {args.checkpoint}. Robustness benchmark requires trained model weights."
        )

    test_p = Path(args.test_split)
    if not test_p.is_file():
        raise FileNotFoundError(
            f"Test split manifest not found at {args.test_split}. Robustness benchmark requires real test image split."
        )

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Running Optical Corruption Robustness Benchmark [Task: {args.task}] on {test_p} ==="
    )
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    test_df = pd.read_csv(test_p)
    if len(test_df) == 0:
        raise ValueError(f"Test split {test_p} is empty.")

    # Mappings
    eyeq_map = {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}
    deepdrid_map = {"good": 0, "reject": 1, "poor": 1, "poor_reject": 1, 0: 0, 1: 1}

    valid_rows = []
    excluded_missing_count = 0
    excluded_missing_file_count = 0

    target_col = "quality_canonical" if args.task == "eyeq_quality" else "overall_quality_canonical"
    alt_col = "quality_raw" if args.task == "eyeq_quality" else "overall_quality_raw"
    active_map = eyeq_map if args.task == "eyeq_quality" else deepdrid_map

    for idx, row in test_df.iterrows():
        raw_val = row.get(target_col, row.get(alt_col, None))
        if pd.isna(raw_val) or raw_val not in active_map:
            excluded_missing_count += 1
            continue

        img_path = Path(str(row.get("path", "")))
        if not img_path.is_file():
            excluded_missing_file_count += 1
            continue

        valid_rows.append((img_path, active_map[raw_val]))

    if len(valid_rows) == 0:
        raise ValueError(
            f"No valid labeled records found for task '{args.task}' in {test_p}. "
            f"Excluded missing labels: {excluded_missing_count}, missing files: {excluded_missing_file_count}."
        )

    # Reproducible stratified or balanced random subset selection if max_samples < total
    rng = np.random.RandomState(args.seed)
    if len(valid_rows) > args.max_samples:
        selected_indices = rng.choice(len(valid_rows), size=args.max_samples, replace=False)
        selected_rows = [valid_rows[i] for i in selected_indices]
    else:
        selected_rows = valid_rows

    print(
        f"Selected {len(selected_rows)} test samples (Total valid: {len(valid_rows)}, "
        f"Excluded missing label: {excluded_missing_count}, Excluded missing file: {excluded_missing_file_count})"
    )

    clean_images = []
    labels = []
    for img_path, label in selected_rows:
        with Image.open(img_path) as img:
            clean_images.append(img.convert("RGB"))
        labels.append(label)

    labels_arr = np.array(labels)
    head_key = "quality_logits" if args.task == "eyeq_quality" else "overall_quality_logits"
    class_names = (
        ["good", "usable", "reject"] if args.task == "eyeq_quality" else ["good", "poor_reject"]
    )

    # 1. Clean Baseline Evaluation (Item 5 & 28)
    clean_tensors = [preprocess_image_canonical(img).squeeze(0) for img in clean_images]
    clean_batch = torch.stack(clean_tensors)
    with torch.no_grad():
        clean_logits = model(clean_batch)[head_key].cpu().numpy()
    clean_metrics = compute_quality_metrics(
        clean_logits, labels_arr, class_names=class_names, is_logits=True
    )

    clean_macro_f1 = clean_metrics["macro_f1"]
    print(
        f"Clean Baseline Macro-F1: {clean_macro_f1:.4f} | Accuracy: {clean_metrics['accuracy']:.4f}"
    )

    # 2. Corruptions Evaluation
    corruptions = SyntheticCorruptionSuite.get_all_names()
    corruption_results = {}

    for c_name in corruptions:
        corruption_results[c_name] = {}
        for sev in [1, 2, 3, 4, 5]:
            tensors = []
            for img in clean_images:
                c_img = SyntheticCorruptionSuite.apply(img, c_name, severity=sev)
                t = preprocess_image_canonical(c_img).squeeze(0)
                tensors.append(t)
            batch = torch.stack(tensors)
            with torch.no_grad():
                logits = model(batch)[head_key].cpu().numpy()
            m = compute_quality_metrics(logits, labels_arr, class_names=class_names, is_logits=True)

            degradation_macro_f1 = round(float(clean_macro_f1 - m["macro_f1"]), 4)
            corruption_results[c_name][str(sev)] = {
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "accuracy": m["accuracy"],
                "quadratic_weighted_kappa": m["quadratic_weighted_kappa"],
                "delta_macro_f1_from_clean": degradation_macro_f1,
            }

    final_payload = {
        "task": args.task,
        "test_split": str(test_p),
        "num_samples_evaluated": len(selected_rows),
        "total_valid_samples": len(valid_rows),
        "excluded_missing_labels": excluded_missing_count,
        "clean_baseline": {
            "macro_f1": clean_metrics["macro_f1"],
            "balanced_accuracy": clean_metrics["balanced_accuracy"],
            "accuracy": clean_metrics["accuracy"],
            "quadratic_weighted_kappa": clean_metrics["quadratic_weighted_kappa"],
        },
        "corruptions": corruption_results,
    }

    with open(out_p / "corruptions.json", "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(
        f"Evaluated clean baseline + {len(corruptions)} corruption types across 5 severities.\n"
        f"Saved results to {out_p / 'corruptions.json'}"
    )


if __name__ == "__main__":
    main()
