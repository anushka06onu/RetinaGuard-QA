"""Three-seed training and evaluation campaign runner per Item 8 of master checklist."""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from retinaguard.training.train import run_training_experiment
from scripts.evaluate import evaluate_dataset_partition


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-seed multi-task training campaign and aggregate results."
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_multitask.yaml", help="Path to training config"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028],
        help="List of random seeds to execute",
    )
    parser.add_argument("--eyeq-split", type=str, default="data/splits/eyeq_test.csv")
    parser.add_argument(
        "--deepdrid-split", type=str, default="data/splits/deepdrid_external_test.csv"
    )
    parser.add_argument("--output-models-dir", type=str, default="artifacts/models")
    parser.add_argument("--output-metrics-dir", type=str, default="artifacts/metrics")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 2 epochs per seed for fast validation"
    )
    args = parser.parse_args()

    models_dir = Path(args.output_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("===========================================================")
    print(f"=== Starting 3-Seed Campaign across Seeds: {args.seeds} ===")
    print("===========================================================\n")

    seed_records = []

    for seed in args.seeds:
        print(f"\n>>> [SEED {seed}] Running Training Experiment...")
        train_res = run_training_experiment(
            config_path=args.config,
            smoke_test=args.smoke_test,
            fixture_mode=False,
            override_seed=seed,
            output_dir=models_dir,
        )

        ckpt_path = models_dir / f"model_seed_{seed}.ckpt"
        # Copy best.ckpt to seed-specific checkpoint
        saved_ckpt = Path(train_res["checkpoint_path"])
        if saved_ckpt.is_file() and saved_ckpt != ckpt_path:
            import shutil

            shutil.copy2(saved_ckpt, ckpt_path)

        print(
            f">>> [SEED {seed}] Checkpoint saved at {ckpt_path}. Evaluating held-out test splits..."
        )

        from retinaguard.models.multitask import RetinaGuardMultiTaskModel

        model = RetinaGuardMultiTaskModel(pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        model.eval()

        seed_entry: Dict[str, Any] = {
            "seed": seed,
            "best_val_macro_f1": train_res["best_val_macro_f1"],
            "epochs_trained": train_res["epochs_trained"],
            "checkpoint_path": str(ckpt_path),
        }

        # EyeQ test evaluation
        if Path(args.eyeq_split).is_file():
            eyeq_eval = evaluate_dataset_partition(
                model, args.eyeq_split, "EyeQ Test", is_deepdrid=False
            )
            seed_entry["eyeq_macro_f1"] = eyeq_eval["macro_f1"]
            seed_entry["eyeq_balanced_accuracy"] = eyeq_eval["balanced_accuracy"]
            seed_entry["eyeq_accuracy"] = eyeq_eval["accuracy"]
            seed_entry["eyeq_qwk"] = eyeq_eval["quadratic_weighted_kappa"]
            seed_entry["eyeq_ci_lower"] = eyeq_eval["macro_f1_95_ci"]["ci_lower"]
            seed_entry["eyeq_ci_upper"] = eyeq_eval["macro_f1_95_ci"]["ci_upper"]

        # DeepDRiD held-out evaluation
        if Path(args.deepdrid_split).is_file():
            dd_eval = evaluate_dataset_partition(
                model, args.deepdrid_split, "DeepDRiD Held-Out", is_deepdrid=True
            )
            seed_entry["deepdrid_macro_f1"] = dd_eval["macro_f1"]
            seed_entry["deepdrid_balanced_accuracy"] = dd_eval["balanced_accuracy"]
            seed_entry["deepdrid_accuracy"] = dd_eval["accuracy"]
            seed_entry["deepdrid_qwk"] = dd_eval["quadratic_weighted_kappa"]
            seed_entry["deepdrid_ci_lower"] = dd_eval["macro_f1_95_ci"]["ci_lower"]
            seed_entry["deepdrid_ci_upper"] = dd_eval["macro_f1_95_ci"]["ci_upper"]

            if "attribute_metrics" in dd_eval:
                for attr in ["artifact", "clarity", "field_definition"]:
                    if attr in dd_eval["attribute_metrics"]:
                        seed_entry[f"{attr}_qwk"] = dd_eval["attribute_metrics"][attr]["qwk"]
                        seed_entry[f"{attr}_f1"] = dd_eval["attribute_metrics"][attr]["macro_f1"]
                        seed_entry[f"{attr}_mae"] = dd_eval["attribute_metrics"][attr]["mae"]

        seed_records.append(seed_entry)

    df_seeds = pd.DataFrame(seed_records)
    per_seed_csv = metrics_dir / "per_seed_metrics.csv"
    df_seeds.to_csv(per_seed_csv, index=False)
    print(f"\nExported per-seed metrics to {per_seed_csv}")

    # Compute Aggregate Mean +/- Std
    numeric_cols = [c for c in df_seeds.columns if c not in ["seed", "checkpoint_path"]]
    summary_data = []
    for col in numeric_cols:
        vals = df_seeds[col].dropna()
        if len(vals) > 0:
            summary_data.append(
                {
                    "metric": col,
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "num_seeds": len(vals),
                }
            )

    df_summary = pd.DataFrame(summary_data)
    summary_csv = metrics_dir / "campaign_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Exported campaign summary to {summary_csv}")

    print("\n=== 3-Seed Campaign Summary ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
