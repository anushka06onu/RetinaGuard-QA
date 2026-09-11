"""Ablation study execution suite per master checklist Item 11."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from retinaguard.evaluation.provenance import validate_empirical_result
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.training.train import run_training_experiment
from retinaguard.utils.hashing import compute_sha256
from scripts.evaluate import evaluate_dataset_partition


def run_ablation_experiment(
    name: str,
    base_config_path: str,
    overrides: Dict[str, Any],
    output_dir: Path,
    eyeq_split: str,
    deepdrid_split: str,
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute a single ablation variant, evaluate on test sets, and record full provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    for section, values in overrides.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values

    cfg["experiment"]["name"] = f"ablation_{name}"
    ablation_config_p = output_dir / "ablation_config.yaml"
    with open(ablation_config_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    train_res = run_training_experiment(
        config_path=ablation_config_p,
        smoke_test=smoke_test,
        fixture_mode=False,
        override_seed=seed,
        output_dir=output_dir,
    )

    ckpt_path = Path(train_res["checkpoint_path"])
    ckpt_sha = compute_sha256(ckpt_path)
    model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_path)
    model.eval()

    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "ablation_name": name,
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "best_val_macro_f1": train_res["best_val_macro_f1"],
        "epochs_trained": train_res["epochs_trained"],
    }

    if Path(eyeq_split).is_file():
        eyeq_eval, df_eyeq = evaluate_dataset_partition(
            model, eyeq_split, f"EyeQ Test ({name})", is_deepdrid=False
        )
        pred_p = preds_dir / "eyeq_test_predictions.csv"
        df_eyeq.to_csv(pred_p, index=False)
        eyeq_eval.update(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/run_ablations.py",
                "git_commit": git_commit,
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "split_path": eyeq_split,
                "split_sha256": compute_sha256(eyeq_split),
                "prediction_file": str(pred_p),
                "prediction_file_sha256": compute_sha256(pred_p),
                "num_samples": len(df_eyeq),
                "created_at_utc": created_at,
            }
        )
        with open(metrics_dir / "eyeq_test.json", "w", encoding="utf-8") as f:
            json.dump(eyeq_eval, f, indent=2)
        validate_empirical_result(metrics_dir / "eyeq_test.json")
        record["eyeq_macro_f1"] = eyeq_eval["macro_f1"]
        record["eyeq_accuracy"] = eyeq_eval["accuracy"]

    if Path(deepdrid_split).is_file():
        is_eyeq_only = train_res.get("training_datasets") == ["EyeQ"]
        dd_eval, df_dd = evaluate_dataset_partition(
            model,
            deepdrid_split,
            f"DeepDRiD Test ({name})",
            is_deepdrid=True,
            is_zero_shot=is_eyeq_only,
        )
        pred_p = preds_dir / "deepdrid_test_predictions.csv"
        df_dd.to_csv(pred_p, index=False)
        dd_eval.update(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/run_ablations.py",
                "git_commit": git_commit,
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "split_path": deepdrid_split,
                "split_sha256": compute_sha256(deepdrid_split),
                "prediction_file": str(pred_p),
                "prediction_file_sha256": compute_sha256(pred_p),
                "num_samples": len(df_dd),
                "created_at_utc": created_at,
            }
        )
        with open(metrics_dir / "deepdrid_test.json", "w", encoding="utf-8") as f:
            json.dump(dd_eval, f, indent=2)
        validate_empirical_result(metrics_dir / "deepdrid_test.json")
        record["deepdrid_macro_f1"] = dd_eval["macro_f1"]
        record["deepdrid_accuracy"] = dd_eval["accuracy"]

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Run essential ablation studies (Single-Task vs Multi-Task, Attribute Losses)."
    )
    parser.add_argument("--base-config", type=str, default="configs/train_multitask.yaml")
    parser.add_argument("--eyeq-split", type=str, default="data/splits/eyeq_test.csv")
    parser.add_argument(
        "--deepdrid-split", type=str, default="data/splits/deepdrid_external_test.csv"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028],
        help="List of random seeds to execute per ablation variant",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/ablations")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("=== Running Essential Ablation Studies across Seeds   ===")
    print(f"=== Seeds: {args.seeds} | Output: {out_base} ===")
    print("=========================================================\n")

    ablations = [
        (
            "proposed_multitask",
            args.base_config,
            {},
        ),
        (
            "singletask_eyeq",
            "configs/train_eyeq.yaml",
            {},
        ),
        (
            "no_attribute_losses",
            args.base_config,
            {
                "model": {
                    "heads": {
                        "quality": {"num_classes": 3, "weight": 1.0},
                        "overall_quality": {"num_classes": 2, "weight": 1.0},
                        "artifact": {"num_classes": 3, "weight": 0.0},
                        "clarity": {"num_classes": 3, "weight": 0.0},
                        "field_definition": {"num_classes": 3, "weight": 0.0},
                    }
                }
            },
        ),
    ]

    all_seed_results = []
    for name, cfg_path, overrides in ablations:
        variant_dir = out_base / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            print(f"\n>>> Running Ablation Variant: {name} (Seed {seed})...")
            seed_dir = variant_dir / f"seed_{seed}"

            res = run_ablation_experiment(
                name=f"{name}_seed_{seed}",
                base_config_path=cfg_path,
                overrides=overrides,
                output_dir=seed_dir,
                eyeq_split=args.eyeq_split,
                deepdrid_split=args.deepdrid_split,
                seed=seed,
                smoke_test=args.smoke_test,
            )
            res["variant"] = name
            res["seed"] = seed
            all_seed_results.append(res)

    df_seed_ablations = pd.DataFrame(all_seed_results)
    per_seed_csv = out_base / "ablations_per_seed.csv"
    df_seed_ablations.to_csv(per_seed_csv, index=False)

    # Compute summary mean +/- sample std per variant
    summary_rows = []
    for name, _, _ in ablations:
        variant_df = df_seed_ablations[df_seed_ablations["variant"] == name]
        row: Dict[str, Any] = {"variant": name, "num_seeds": len(variant_df)}
        for metric in [
            "eyeq_macro_f1",
            "eyeq_accuracy",
            "deepdrid_macro_f1",
            "deepdrid_accuracy",
            "best_val_macro_f1",
        ]:
            if metric in variant_df.columns:
                vals = variant_df[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = round(float(vals.mean()), 4)
                    row[f"{metric}_std"] = (
                        round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
                    )
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_base / "ablations_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    with open(out_base / "ablations.json", "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    print(f"\nExported ablation per-seed to {per_seed_csv}")
    print(f"Exported ablation summary to {summary_csv}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
