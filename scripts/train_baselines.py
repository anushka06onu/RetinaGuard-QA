"""Baseline models and ablation study suite per Items 19 & 20 of master checklist."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

from retinaguard.models.baselines import ClassicalQualityFeatureExtractor


def run_majority_class_baseline(
    train_csv: str, test_csv: str, task: str = "eyeq_quality"
) -> Dict[str, Any]:
    """Majority class heuristic baseline."""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    col = "quality_canonical" if task == "eyeq_quality" else "overall_quality_canonical"
    q_map = (
        {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}
        if task == "eyeq_quality"
        else {"good": 0, "reject": 1, "poor": 1, 0: 0, 1: 1}
    )

    train_labels = [
        q_map[r[col]] for _, r in df_train.iterrows() if pd.notna(r.get(col)) and r[col] in q_map
    ]
    test_labels = [
        q_map[r[col]] for _, r in df_test.iterrows() if pd.notna(r.get(col)) and r[col] in q_map
    ]

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

    return {
        "model": "Majority Class Baseline",
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "num_samples": len(test_labels),
    }


def run_classical_feature_baseline(
    train_csv: str, test_csv: str, model_type: str = "random_forest", task: str = "eyeq_quality"
) -> Dict[str, Any]:
    """Classical texture, contrast, and sharpness handcrafted features with RF / Logistic Regression."""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    col = "quality_canonical" if task == "eyeq_quality" else "overall_quality_canonical"
    q_map = (
        {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}
        if task == "eyeq_quality"
        else {"good": 0, "reject": 1, "poor": 1, 0: 0, 1: 1}
    )

    def extract_set(df, max_n=300):
        X, y = [], []
        for _, r in df.iterrows():
            if pd.isna(r.get(col)) or r[col] not in q_map:
                continue
            img_p = Path(str(r.get("path", "")))
            if not img_p.is_file():
                continue
            try:
                with Image.open(img_p) as img:
                    feats = ClassicalQualityFeatureExtractor.extract(img.convert("RGB"))
                    X.append(feats)
                    y.append(q_map[r[col]])
                if len(X) >= max_n:
                    break
            except Exception:
                pass
        return np.array(X), np.array(y)

    X_train, y_train = extract_set(df_train, max_n=500)
    X_test, y_test = extract_set(df_test, max_n=300)

    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "model": f"Classical Features + {model_type}",
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "status": "not_evaluable",
        }

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

    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, preds))
    bal_acc = float(balanced_accuracy_score(y_test, preds))
    try:
        qwk = float(cohen_kappa_score(y_test, preds, weights="quadratic"))
    except Exception:
        qwk = 0.0

    return {
        "model": f"Classical Handcrafted Features + {'Random Forest' if model_type == 'random_forest' else 'Logistic Regression'}",
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk if not np.isnan(qwk) else 0.0, 4),
        "num_train": len(X_train),
        "num_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run complete baseline comparisons and ablation studies."
    )
    parser.add_argument("--eyeq-train", type=str, default="data/splits/eyeq_train.csv")
    parser.add_argument("--eyeq-test", type=str, default="data/splits/eyeq_test.csv")
    parser.add_argument(
        "--deepdrid-test", type=str, default="data/splits/deepdrid_external_test.csv"
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("=== Running Baseline Models Comparison Suite (Items 19 & 20)  ===")
    print("=================================================================\n")

    baseline_results = []

    # 1. Majority Class Baseline
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Majority Class Baseline...")
        maj_res = run_majority_class_baseline(args.eyeq_train, args.eyeq_test)
        baseline_results.append(maj_res)
        print(
            f"Majority Class: Macro-F1 = {maj_res['macro_f1']:.4f}, Accuracy = {maj_res['accuracy']:.4f}"
        )

    # 2. Classical Handcrafted Features + Random Forest
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Classical Features + Random Forest...")
        rf_res = run_classical_feature_baseline(
            args.eyeq_train, args.eyeq_test, model_type="random_forest"
        )
        baseline_results.append(rf_res)
        print(
            f"Classical + RF: Macro-F1 = {rf_res['macro_f1']:.4f}, Accuracy = {rf_res['accuracy']:.4f}"
        )

    # 3. Classical Handcrafted Features + Logistic Regression
    if Path(args.eyeq_train).is_file() and Path(args.eyeq_test).is_file():
        print(">>> Evaluating Classical Features + Logistic Regression...")
        lr_res = run_classical_feature_baseline(
            args.eyeq_train, args.eyeq_test, model_type="logistic_regression"
        )
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
