"""Baseline models evaluation suite per master checklist."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from retinaguard.evaluation.provenance import validate_baseline_result
from retinaguard.models.baselines import ClassicalQualityFeatureExtractor
from retinaguard.utils.hashing import compute_sha256


def run_majority_class_baseline(
    train_csv: str, test_csv: str, task: str = "eyeq_quality"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Majority class heuristic baseline on full dataset."""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    col = "quality_canonical" if task == "eyeq_quality" else "overall_quality_canonical"
    q_map = (
        {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}
        if task == "eyeq_quality"
        else {"good": 0, "reject": 1, "poor": 1, 0: 0, 1: 1}
    )

    valid_train = [
        (r["image_id"], q_map[r[col]])
        for _, r in df_train.iterrows()
        if pd.notna(r.get(col)) and r[col] in q_map
    ]
    valid_test = [
        (
            r["image_id"],
            r.get("patient_id", "unknown"),
            q_map[r[col]],
        )
        for _, r in df_test.iterrows()
        if pd.notna(r.get(col)) and r[col] in q_map
    ]

    train_labels = [y for _, y in valid_train]
    test_labels = [y for _, _, y in valid_test]
    test_ids = [img_id for img_id, _, _ in valid_test]
    test_patients = [pid for _, pid, _ in valid_test]

    # Find mode of train
    from collections import Counter

    counts = Counter(train_labels)
    majority_class, majority_count = counts.most_common(1)[0]
    majority_prevalence = round(float(majority_count / max(1, len(train_labels))), 4)
    preds = [majority_class] * len(test_labels)

    f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(test_labels, preds))
    bal_acc = float(balanced_accuracy_score(test_labels, preds))
    try:
        qwk = float(cohen_kappa_score(test_labels, preds, weights="quadratic"))
    except Exception:
        qwk = 0.0

    cm = confusion_matrix(test_labels, preds).tolist()

    df_preds = pd.DataFrame(
        {
            "image_id": test_ids,
            "patient_id": test_patients,
            "dataset": "EyeQ" if task == "eyeq_quality" else "DeepDRiD",
            "split": Path(test_csv).name,
            "target": test_labels,
            "prediction": preds,
            "confidence": majority_prevalence,
        }
    )

    result_dict: Dict[str, Any] = {
        "model": "Majority Class Baseline",
        "task": task,
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "confusion_matrix": cm,
        "num_train": len(train_labels),
        "num_test": len(test_labels),
        "subset_experiment": False,
    }

    return result_dict, df_preds


def run_classical_feature_baseline(
    train_csv: str,
    test_csv: str,
    model_type: str = "random_forest",
    task: str = "eyeq_quality",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    allow_missing_images: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Classical texture, contrast, and sharpness handcrafted features with RF / Logistic Regression."""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    col = "quality_canonical" if task == "eyeq_quality" else "overall_quality_canonical"
    q_map = (
        {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}
        if task == "eyeq_quality"
        else {"good": 0, "reject": 1, "poor": 1, 0: 0, 1: 1}
    )

    def extract_set(df: pd.DataFrame, max_n: Optional[int] = None):
        valid_rows = [r for _, r in df.iterrows() if pd.notna(r.get(col)) and r[col] in q_map]
        if max_n and 0 < max_n < len(valid_rows):
            # Stratified sample if possible
            labels = [q_map[r[col]] for r in valid_rows]
            valid_rows, _ = train_test_split(
                valid_rows, train_size=max_n, random_state=2026, stratify=labels
            )

        X, y, ids, pids, excluded = [], [], [], [], []
        for r in valid_rows:
            img_p = Path(str(r.get("path", "")))
            if not img_p.is_file():
                if not allow_missing_images:
                    raise FileNotFoundError(
                        f"Expected image file not found for baseline evaluation: {img_p}. "
                        "Final baseline evaluation must fail when expected images are unreadable."
                    )
                excluded.append(r["image_id"])
                continue
            try:
                with Image.open(img_p) as img:
                    feats = ClassicalQualityFeatureExtractor.extract(img.convert("RGB"))
                    X.append(feats)
                    y.append(q_map[r[col]])
                    ids.append(r["image_id"])
                    pids.append(r.get("patient_id", "unknown"))
            except Exception as e:
                if not allow_missing_images:
                    raise IOError(
                        f"Corrupted or unreadable image {img_p} encountered during baseline evaluation: {e}"
                    )
                excluded.append(r["image_id"])
        return np.array(X), np.array(y), ids, pids, excluded

    X_train, y_train, train_ids, _, excl_train = extract_set(df_train, max_n=max_train_samples)
    X_test, y_test, test_ids, test_pids, excl_test = extract_set(df_test, max_n=max_test_samples)

    is_subset = bool(
        (max_train_samples and max_train_samples > 0)
        or (max_test_samples and max_test_samples > 0)
        or len(excl_train) > 0
        or len(excl_test) > 0
    )

    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "model": f"Classical Handcrafted Features + {'Random Forest' if model_type == 'random_forest' else 'Logistic Regression'}",
            "task": task,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "confusion_matrix": [],
            "status": "not_evaluable",
            "num_train": len(X_train),
            "num_test": len(X_test),
            "subset_experiment": is_subset,
            "excluded_test_ids": excl_test,
        }, pd.DataFrame()

    # Normalize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=2026)
    else:
        clf = LogisticRegression(max_iter=500, random_state=2026)

    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)
    confs = np.max(probs, axis=-1)

    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, preds))
    bal_acc = float(balanced_accuracy_score(y_test, preds))
    try:
        qwk = float(cohen_kappa_score(y_test, preds, weights="quadratic"))
    except Exception:
        qwk = 0.0

    cm = confusion_matrix(y_test, preds).tolist()

    df_preds = pd.DataFrame(
        {
            "image_id": test_ids,
            "patient_id": test_pids,
            "dataset": "EyeQ" if task == "eyeq_quality" else "DeepDRiD",
            "split": Path(test_csv).name,
            "target": y_test,
            "prediction": preds,
            "confidence": np.round(confs, 4),
        }
    )

    model_display = f"Classical Handcrafted Features + {'Random Forest' if model_type == 'random_forest' else 'Logistic Regression'}"

    return {
        "model": model_display,
        "task": task,
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "confusion_matrix": cm,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "subset_experiment": is_subset,
        "excluded_test_ids": excl_test,
    }, df_preds


def main():
    parser = argparse.ArgumentParser(description="Run complete baseline models comparisons.")
    parser.add_argument("--eyeq-train", type=str, default="data/splits/eyeq_train.csv")
    parser.add_argument("--eyeq-test", type=str, default="data/splits/eyeq_test.csv")
    parser.add_argument(
        "--deepdrid-test", type=str, default="data/splits/deepdrid_external_test.csv"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit on training samples for fast subset benchmarking (default: full dataset)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional limit on test samples for fast subset benchmarking (default: full dataset)",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Allow skipping unreadable images in exploratory mode (marks subset_experiment=True)",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    preds_dir = out_p.parent / "predictions" if out_p.name == "metrics" else out_p / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    created_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    train_sha = compute_sha256(args.eyeq_train) if Path(args.eyeq_train).is_file() else "unknown"
    test_sha = compute_sha256(args.eyeq_test) if Path(args.eyeq_test).is_file() else "unknown"

    print("=================================================================")
    print("=== Running Baseline Models Comparison Suite                  ===")
    print("=================================================================\n")

    baseline_results = []

    # 1. Majority Class Baseline
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Majority Class Baseline...")
        maj_res, df_maj_preds = run_majority_class_baseline(args.eyeq_train, args.eyeq_test)
        if len(df_maj_preds) > 0:
            pred_file = preds_dir / "majority_baseline_predictions.csv"
            df_maj_preds.to_csv(pred_file, index=False)
            maj_res.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/train_baselines.py",
                    "git_commit": git_commit,
                    "train_split_path": args.eyeq_train,
                    "train_split_sha256": train_sha,
                    "test_split_path": args.eyeq_test,
                    "test_split_sha256": test_sha,
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": compute_sha256(pred_file),
                    "created_at_utc": created_at_utc,
                }
            )
            validate_baseline_result(maj_res)
        baseline_results.append(maj_res)
        print(
            f"Majority Class: Macro-F1 = {maj_res['macro_f1']:.4f}, Accuracy = {maj_res['accuracy']:.4f}"
        )

    # 2. Classical Handcrafted Features + Random Forest
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Classical Features + Random Forest...")
        rf_res, df_rf_preds = run_classical_feature_baseline(
            args.eyeq_train,
            args.eyeq_test,
            model_type="random_forest",
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            allow_missing_images=args.allow_missing_images or args.smoke_test,
        )
        if len(df_rf_preds) > 0:
            pred_file = preds_dir / "classical_rf_predictions.csv"
            df_rf_preds.to_csv(pred_file, index=False)
            rf_res.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/train_baselines.py",
                    "git_commit": git_commit,
                    "train_split_path": args.eyeq_train,
                    "train_split_sha256": train_sha,
                    "test_split_path": args.eyeq_test,
                    "test_split_sha256": test_sha,
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": compute_sha256(pred_file),
                    "created_at_utc": created_at_utc,
                }
            )
            validate_baseline_result(rf_res)
        baseline_results.append(rf_res)
        print(
            f"Classical + RF: Macro-F1 = {rf_res['macro_f1']:.4f}, Accuracy = {rf_res['accuracy']:.4f}"
        )

    # 3. Classical Handcrafted Features + Logistic Regression
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Classical Features + Logistic Regression...")
        lr_res, df_lr_preds = run_classical_feature_baseline(
            args.eyeq_train,
            args.eyeq_test,
            model_type="logistic_regression",
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            allow_missing_images=args.allow_missing_images or args.smoke_test,
        )
        if len(df_lr_preds) > 0:
            pred_file = preds_dir / "classical_lr_predictions.csv"
            df_lr_preds.to_csv(pred_file, index=False)
            lr_res.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/train_baselines.py",
                    "git_commit": git_commit,
                    "train_split_path": args.eyeq_train,
                    "train_split_sha256": train_sha,
                    "test_split_path": args.eyeq_test,
                    "test_split_sha256": test_sha,
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": compute_sha256(pred_file),
                    "created_at_utc": created_at_utc,
                }
            )
            validate_baseline_result(lr_res)
        baseline_results.append(lr_res)
        print(
            f"Classical + LR: Macro-F1 = {lr_res['macro_f1']:.4f}, Accuracy = {lr_res['accuracy']:.4f}"
        )

    # Save Baselines JSON and CSV
    df_baselines = pd.DataFrame(baseline_results)
    df_baselines.to_csv(out_p / "baselines_summary.csv", index=False)
    with open(out_p / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\nExported baseline comparisons to {out_p / 'baselines_summary.csv'}")
    print(df_baselines.to_string(index=False))


if __name__ == "__main__":
    main()
