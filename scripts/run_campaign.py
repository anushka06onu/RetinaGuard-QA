"""Three-seed training and evaluation campaign runner with strict per-seed isolation and auditable provenance."""

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from retinaguard.evaluation.provenance import validate_empirical_result
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.training.train import run_training_experiment
from retinaguard.utils.hashing import compute_sha256
from scripts.evaluate import evaluate_dataset_partition


def generate_seed_checksums(seed_dir: Path) -> Path:
    """Generate SHA-256 checksums file for all files in a seed directory."""
    checksum_lines = []
    for p in sorted(seed_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rel_path = p.relative_to(seed_dir)
            sha = compute_sha256(p)
            checksum_lines.append(f"{sha}  {rel_path}")
    sums_file = seed_dir / "SHA256SUMS"
    with open(sums_file, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")
    return sums_file


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-seed training and evaluation campaign with per-seed output isolation."
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
    parser.add_argument(
        "--output-runs-dir",
        type=str,
        default="artifacts/runs",
        help="Base directory for per-seed isolated runs",
    )
    parser.add_argument(
        "--output-metrics-dir",
        type=str,
        default="artifacts/metrics",
        help="Directory to save aggregated campaign metrics",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run genuine EyeQ-only zero-shot transfer campaign on DeepDRiD",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 2 epochs per seed for fast validation"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing completed seed runs",
    )
    args = parser.parse_args()

    # Automatically switch default config if zero-shot mode is specified with multitask default
    if args.zero_shot and args.config == "configs/train_multitask.yaml":
        args.config = "configs/train_eyeq.yaml"

    runs_base_dir = Path(args.output_runs_dir)
    runs_base_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "Zero-Shot Transfer Campaign" if args.zero_shot else "Supervised Multi-Task Campaign"
    print("===========================================================")
    print(f"=== Starting 3-Seed {mode_str} across Seeds: {args.seeds} ===")
    print(f"=== Config: {args.config} | Runs Dir: {runs_base_dir} ===")
    print("===========================================================\n")

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    config_sha256 = compute_sha256(args.config) if Path(args.config).is_file() else "unknown"
    seed_records = []

    for seed in args.seeds:
        seed_dir = runs_base_dir / f"seed_{seed}"
        manifest_path = seed_dir / "run_manifest.json"

        if seed_dir.exists() and manifest_path.is_file() and not args.overwrite:
            print(
                f">>> [SEED {seed}] Found existing completed run at {seed_dir}. Skipping (pass --overwrite to re-run)."
            )
            with open(manifest_path, "r", encoding="utf-8") as f:
                saved_manifest = json.load(f)
                seed_records.append(saved_manifest.get("summary_metrics", {}))
            continue

        if seed_dir.exists() and args.overwrite:
            print(f">>> [SEED {seed}] Overwriting existing run directory {seed_dir}...")
            shutil.rmtree(seed_dir)

        seed_dir.mkdir(parents=True, exist_ok=True)
        preds_dir = seed_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics_dir = seed_dir / "metrics"
        seed_metrics_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> [SEED {seed}] Running Training in Isolated Directory: {seed_dir}...")
        train_res = run_training_experiment(
            config_path=args.config,
            smoke_test=args.smoke_test,
            fixture_mode=False,
            override_seed=seed,
            output_dir=seed_dir,
        )

        ckpt_path = Path(train_res["checkpoint_path"])
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not produced at {ckpt_path}")

        ckpt_sha256 = compute_sha256(ckpt_path)
        print(f">>> [SEED {seed}] Checkpoint verified at {ckpt_path}. Evaluating test splits...")

        model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_path)
        model.eval()

        created_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        seed_entry: Dict[str, Any] = {
            "seed": seed,
            "best_val_macro_f1": train_res["best_val_macro_f1"],
            "epochs_trained": train_res["epochs_trained"],
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha256,
            "config_path": args.config,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
        }

        # EyeQ test evaluation
        if Path(args.eyeq_split).is_file():
            eyeq_eval, df_eyeq_preds = evaluate_dataset_partition(
                model, args.eyeq_split, "EyeQ Test", is_deepdrid=False
            )
            pred_file = preds_dir / "eyeq_test_predictions.csv"
            df_eyeq_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            eyeq_eval.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/run_campaign.py",
                    "git_commit": git_commit,
                    "seed": seed,
                    "config_path": args.config,
                    "config_sha256": config_sha256,
                    "dataset_task": "EyeQ_Quality_3Class",
                    "head_identity": "quality_head",
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": ckpt_sha256,
                    "split_path": args.eyeq_split,
                    "split_sha256": compute_sha256(args.eyeq_split),
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": pred_sha,
                    "num_samples": len(df_eyeq_preds),
                    "created_at_utc": created_at_utc,
                }
            )
            eyeq_json_path = seed_metrics_dir / "eyeq_test.json"
            with open(eyeq_json_path, "w", encoding="utf-8") as f:
                json.dump(eyeq_eval, f, indent=2)

            # Strict provenance verification
            validate_empirical_result(eyeq_json_path)

            seed_entry["eyeq_macro_f1"] = eyeq_eval["macro_f1"]
            seed_entry["eyeq_balanced_accuracy"] = eyeq_eval["balanced_accuracy"]
            seed_entry["eyeq_accuracy"] = eyeq_eval["accuracy"]
            seed_entry["eyeq_qwk"] = eyeq_eval["quadratic_weighted_kappa"]
            seed_entry["eyeq_ci_lower"] = eyeq_eval["macro_f1_95_ci"]["ci_lower"]
            seed_entry["eyeq_ci_upper"] = eyeq_eval["macro_f1_95_ci"]["ci_upper"]

        # DeepDRiD evaluation (Held-Out Supervised or Zero-Shot External Transfer)
        if Path(args.deepdrid_split).is_file():
            is_zs = args.zero_shot or train_res.get("training_datasets") == ["EyeQ"]
            target_json_name = "zero_shot_transfer.json" if is_zs else "deepdrid_heldout.json"
            pred_file_name = (
                "zero_shot_transfer_predictions.csv"
                if is_zs
                else "deepdrid_heldout_predictions.csv"
            )

            dd_eval, df_dd_preds = evaluate_dataset_partition(
                model,
                args.deepdrid_split,
                "DeepDRiD (Zero-Shot External Transfer)"
                if is_zs
                else "DeepDRiD (Held-Out Supervised)",
                is_deepdrid=True,
                is_zero_shot=is_zs,
            )
            pred_file = preds_dir / pred_file_name
            df_dd_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            dd_eval.update(
                {
                    "status": "completed",
                    "eligible_as_final_result": True,
                    "generated_by": "scripts/run_campaign.py",
                    "git_commit": git_commit,
                    "seed": seed,
                    "config_path": args.config,
                    "config_sha256": config_sha256,
                    "dataset_task": "DeepDRiD_ZeroShot_Binary"
                    if is_zs
                    else "DeepDRiD_Overall_Quality_Binary",
                    "head_identity": "quality_head_mapped_binary"
                    if is_zs
                    else "overall_quality_head",
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": ckpt_sha256,
                    "split_path": args.deepdrid_split,
                    "split_sha256": compute_sha256(args.deepdrid_split),
                    "prediction_file": str(pred_file),
                    "prediction_file_sha256": pred_sha,
                    "num_samples": len(df_dd_preds),
                    "created_at_utc": created_at_utc,
                }
            )
            dd_json_path = seed_metrics_dir / target_json_name
            with open(dd_json_path, "w", encoding="utf-8") as f:
                json.dump(dd_eval, f, indent=2)

            # Strict provenance verification
            validate_empirical_result(dd_json_path)

            prefix = "deepdrid_zero_shot" if is_zs else "deepdrid"
            seed_entry[f"{prefix}_macro_f1"] = dd_eval["macro_f1"]
            seed_entry[f"{prefix}_balanced_accuracy"] = dd_eval["balanced_accuracy"]
            seed_entry[f"{prefix}_accuracy"] = dd_eval["accuracy"]
            seed_entry[f"{prefix}_qwk"] = dd_eval["quadratic_weighted_kappa"]
            seed_entry[f"{prefix}_ci_lower"] = dd_eval["macro_f1_95_ci"]["ci_lower"]
            seed_entry[f"{prefix}_ci_upper"] = dd_eval["macro_f1_95_ci"]["ci_upper"]

            if not is_zs and "attribute_metrics" in dd_eval:
                for attr in ["artifact", "clarity", "field_definition"]:
                    if attr in dd_eval["attribute_metrics"]:
                        seed_entry[f"{attr}_qwk"] = dd_eval["attribute_metrics"][attr]["qwk"]
                        seed_entry[f"{attr}_f1"] = dd_eval["attribute_metrics"][attr]["macro_f1"]
                        seed_entry[f"{attr}_mae"] = dd_eval["attribute_metrics"][attr]["mae"]

        # Save Run Manifest
        run_manifest = {
            "seed": seed,
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
            "summary_metrics": seed_entry,
            "training_result": {
                "best_val_macro_f1": train_res["best_val_macro_f1"],
                "epochs_trained": train_res["epochs_trained"],
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        # Generate checksums for seed run
        generate_seed_checksums(seed_dir)
        seed_records.append(seed_entry)

    df_seeds = pd.DataFrame(seed_records)
    csv_name = "zero_shot_per_seed_metrics.csv" if args.zero_shot else "per_seed_metrics.csv"
    per_seed_csv = metrics_dir / csv_name
    df_seeds.to_csv(per_seed_csv, index=False)
    print(f"\nExported per-seed metrics to {per_seed_csv}")

    # Compute Aggregate Mean +/- Sample Std (ddof=1)
    non_numeric = [
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "git_commit",
        "created_at_utc",
    ]
    numeric_cols = [c for c in df_seeds.columns if c not in non_numeric]
    summary_data = []
    for col in numeric_cols:
        vals = df_seeds[col].dropna()
        if len(vals) > 0:
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_data.append(
                {
                    "metric": col,
                    "mean": round(float(np.mean(vals)), 4),
                    "std_sample": round(std_val, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "num_seeds": len(vals),
                }
            )

    df_summary = pd.DataFrame(summary_data)
    summary_name = "zero_shot_campaign_summary.csv" if args.zero_shot else "campaign_summary.csv"
    summary_csv = metrics_dir / summary_name
    df_summary.to_csv(summary_csv, index=False)
    print(f"Exported campaign summary to {summary_csv}")

    print(f"\n=== 3-Seed {mode_str} Summary (Sample SD ddof=1) ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
