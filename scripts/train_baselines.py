"""Baseline models and ablation study suite per Items 19 & 20 of master checklist."""

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

from retinaguard.models.baselines import ClassicalQualityFeatureExtractor
from retinaguard.utils.hashing import compute_sha256


def run_majority_class_baseline(
    train_csv: str, test_csv: str, task: str = "eyeq_quality", preds_dir: Optional[Path] = None
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

    majority_class = Counter(train_labels).most_common(1)[0][0]
    preds = [majority_class] * len(test_labels)

    f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(test_labels, preds))
    bal_acc = float(balanced_accuracy_score(test_labels, preds))
    try:
        qwk = float(cohen_kappa_score(test_labels, preds, weights="quadratic"))
    except Exception:
        qwk = 0.0

    df_preds = pd.DataFrame(
        {
            "image_id": test_ids,
            "patient_id": test_patients,
            "dataset": "EyeQ" if task == "eyeq_quality" else "DeepDRiD",
            "split": Path(test_csv).name,
            "target": test_labels,
            "prediction": preds,
            "confidence": 1.0,
        }
    )

    return {
        "model": "Majority Class Baseline",
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "num_train": len(train_labels),
        "num_test": len(test_labels),
        "subset_experiment": False,
    }, df_preds


def run_classical_feature_baseline(
    train_csv: str,
    test_csv: str,
    model_type: str = "random_forest",
    task: str = "eyeq_quality",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    preds_dir: Optional[Path] = None,
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

        X, y, ids, pids = [], [], [], []
        for r in valid_rows:
            img_p = Path(str(r.get("path", "")))
            if not img_p.is_file():
                continue
            try:
                with Image.open(img_p) as img:
                    feats = ClassicalQualityFeatureExtractor.extract(img.convert("RGB"))
                    X.append(feats)
                    y.append(q_map[r[col]])
                    ids.append(r["image_id"])
                    pids.append(r.get("patient_id", "unknown"))
            except Exception:
                pass
        return np.array(X), np.array(y), ids, pids

    X_train, y_train, train_ids, _ = extract_set(df_train, max_n=max_train_samples)
    X_test, y_test, test_ids, test_pids = extract_set(df_test, max_n=max_test_samples)

    is_subset = bool(
        (max_train_samples and max_train_samples > 0) or (max_test_samples and max_test_samples > 0)
    )

    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "model": f"Classical Features + {model_type}",
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "status": "not_evaluable",
            "subset_experiment": is_subset,
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

    return {
        "model": f"Classical Handcrafted Features + {'Random Forest' if model_type == 'random_forest' else 'Logistic Regression'}",
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "subset_experiment": is_subset,
    }, df_preds


def main():
    parser = argparse.ArgumentParser(
        description="Run complete baseline comparisons and ablation studies."
    )
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
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    preds_dir = out_p.parent / "predictions" if out_p.name == "metrics" else out_p / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("=== Running Baseline Models Comparison Suite (Items 19 & 20)  ===")
    print("=================================================================\n")

    baseline_results = []

    # 1. Majority Class Baseline
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Majority Class Baseline...")
        maj_res, df_maj_preds = run_majority_class_baseline(
            args.eyeq_train, args.eyeq_test, preds_dir=preds_dir
        )
        if len(df_maj_preds) > 0:
            pred_file = preds_dir / "majority_baseline_predictions.csv"
            df_maj_preds.to_csv(pred_file, index=False)
            maj_res["prediction_file"] = str(pred_file)
            maj_res["prediction_file_sha256"] = compute_sha256(pred_file)
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
            preds_dir=preds_dir,
        )
        if len(df_rf_preds) > 0:
            pred_file = preds_dir / "classical_rf_predictions.csv"
            df_rf_preds.to_csv(pred_file, index=False)
            rf_res["prediction_file"] = str(pred_file)
            rf_res["prediction_file_sha256"] = compute_sha256(pred_file)
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
            preds_dir=preds_dir,
        )
        if len(df_lr_preds) > 0:
            pred_file = preds_dir / "classical_lr_predictions.csv"
            df_lr_preds.to_csv(pred_file, index=False)
            lr_res["prediction_file"] = str(pred_file)
            lr_res["prediction_file_sha256"] = compute_sha256(pred_file)
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
