import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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


def get_git_commit() -> str:
    """Retrieve current Git commit SHA-256."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def verify_existing_seed_run(
    seed_dir: Path,
    expected_mode: str,
    expected_config_sha: str,
    allow_cross_commit: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Verify that an existing seed run is complete, uncorrupted, and matches configuration."""
    manifest_p = seed_dir / "run_manifest.json"
    if not manifest_p.is_file():
        return False, None, "Missing run_manifest.json"

    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, None, f"Unreadable run_manifest.json: {e}"

    if manifest.get("status") != "completed":
        return False, None, f"Status is not completed: {manifest.get('status')}"
    if manifest.get("eligible_for_aggregation") is not True:
        return False, None, "eligible_for_aggregation is not True"
    if manifest.get("campaign_mode") != expected_mode:
        return False, None, f"Mode mismatch: {manifest.get('campaign_mode')} != {expected_mode}"
    if manifest.get("config_sha256") != expected_config_sha and expected_config_sha != "unknown":
        return (
            False,
            None,
            f"Config hash mismatch: {manifest.get('config_sha256')} != {expected_config_sha}",
        )

    current_commit = get_git_commit()
    if (
        not allow_cross_commit
        and manifest.get("git_commit") != current_commit
        and current_commit != "unknown"
    ):
        return (
            False,
            None,
            f"Git commit mismatch: {manifest.get('git_commit')} != {current_commit}",
        )

    sums_file = seed_dir / "SHA256SUMS"
    if not sums_file.is_file():
        return False, None, "Missing SHA256SUMS file"

    seen_rel_paths = set()
    seed_dir_resolved = seed_dir.resolve()

    with open(sums_file, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                return False, None, f"Malformed line {line_num} in SHA256SUMS: {raw_line}"
            expected_sha, rel_str = parts
            if len(expected_sha) != 64 or not all(
                c in "0123456789abcdefABCDEF" for c in expected_sha
            ):
                return False, None, f"Invalid SHA-256 hash at line {line_num}: {expected_sha}"
            if rel_str.startswith("/") or rel_str.startswith("\\"):
                return False, None, f"Absolute path forbidden in SHA256SUMS: {rel_str}"
            if ".." in rel_str.replace("\\", "/").split("/"):
                return False, None, f"Path traversal forbidden in SHA256SUMS: {rel_str}"
            if rel_str in seen_rel_paths:
                return False, None, f"Duplicate entry in SHA256SUMS: {rel_str}"
            seen_rel_paths.add(rel_str)

            target_p = (seed_dir / rel_str).resolve()
            if not target_p.is_relative_to(seed_dir_resolved):
                return False, None, f"Path points outside seed directory: {rel_str}"
            if not target_p.is_file():
                return False, None, f"Missing file listed in SHA256SUMS: {rel_str}"
            actual_sha = compute_sha256(target_p)
            if actual_sha.lower() != expected_sha.lower():
                return (
                    False,
                    None,
                    f"Checksum mismatch for {rel_str}: {actual_sha} != {expected_sha}",
                )

    # Bidirectional: verify every file on disk is listed in SHA256SUMS
    for disk_p in seed_dir.rglob("*"):
        if disk_p.is_file() and disk_p.name != "SHA256SUMS":
            rel_disk = str(disk_p.relative_to(seed_dir))
            if rel_disk not in seen_rel_paths:
                return False, None, f"Unlisted file on disk not recorded in SHA256SUMS: {rel_disk}"

    # Re-run strict validation on every metric file
    metrics_dir = seed_dir / "metrics"
    if metrics_dir.is_dir():
        for m_file in metrics_dir.glob("*.json"):
            try:
                validate_empirical_result(m_file)
            except Exception as e:
                return False, None, f"Metric validation failed for {m_file.name}: {e}"

    return True, manifest, "Verified"


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
        "--output-campaigns-dir",
        type=str,
        default="artifacts/campaigns",
        help="Directory to archive structured timestamped campaigns",
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
    parser.add_argument(
        "--allow-cross-commit-reuse",
        action="store_true",
        help="Allow reusing verified seed runs generated from previous Git commits",
    )
    args = parser.parse_args()

    # Automatically switch default config if zero-shot mode is specified with multitask default
    if args.zero_shot and args.config == "configs/train_multitask.yaml":
        args.config = "configs/train_eyeq.yaml"

    runs_base_dir = Path(args.output_runs_dir)
    runs_base_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    campaigns_dir = Path(args.output_campaigns_dir)
    campaigns_dir.mkdir(parents=True, exist_ok=True)

    mode_name = "zero_shot" if args.zero_shot else "multitask"
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

        if seed_dir.exists() and not args.overwrite:
            is_valid, saved_manifest, reason = verify_existing_seed_run(
                seed_dir,
                mode_name,
                config_sha256,
                allow_cross_commit=args.allow_cross_commit_reuse,
            )
            if is_valid and saved_manifest:
                print(
                    f">>> [SEED {seed}] Found verified completed run at {seed_dir}. Reusing results."
                )
                seed_records.append(saved_manifest.get("summary_metrics", {}))
                continue
            else:
                print(
                    f">>> [SEED {seed}] Existing run at {seed_dir} could not be verified ({reason}). Rerunning..."
                )
                shutil.rmtree(seed_dir)

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

        # Enforce zero-shot provenance (Item 5)
        if args.zero_shot:
            if train_res.get("training_datasets") != ["EyeQ"]:
                raise ValueError(
                    f"Zero-shot campaign requires training exclusively on EyeQ, got {train_res.get('training_datasets')}"
                )
            ckpt_state = torch.load(ckpt_path, map_location="cpu")
            ckpt_meta = ckpt_state.get("metadata", {})
            if (
                "deepdrid" in ckpt_meta.get("train_split_hashes", {})
                or "deepdrid" in ckpt_meta.get("val_split_hashes", {})
                or "DeepDRiD" in ckpt_meta.get("training_datasets", [])
            ):
                raise ValueError(
                    "Checkpoint metadata contains DeepDRiD split provenance; cannot be used for genuine zero-shot evaluation."
                )

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
                (
                    "DeepDRiD (Zero-Shot External Transfer)"
                    if is_zs
                    else "DeepDRiD (Held-Out Supervised)"
                ),
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
                    "dataset_task": (
                        "DeepDRiD_ZeroShot_Binary" if is_zs else "DeepDRiD_Overall_Quality_Binary"
                    ),
                    "head_identity": (
                        "quality_head_mapped_binary" if is_zs else "overall_quality_head"
                    ),
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

        # Save Run Manifest with explicit completion status (Item 6)
        run_manifest = {
            "status": "completed",
            "eligible_for_aggregation": True,
            "campaign_mode": mode_name,
            "seed": seed,
            "config_path": args.config,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
            "summary_metrics": seed_entry,
            "training_result": {
                "best_val_macro_f1": train_res["best_val_macro_f1"],
                "epochs_trained": train_res["epochs_trained"],
            },
        }
        manifest_path = seed_dir / "run_manifest.json"
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

    # Campaign-Level Archival Directory & Checksums (Self-Contained Evidence Package)
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_archive_dir = campaigns_dir / f"{mode_name}_{timestamp_str}"
    campaign_archive_dir.mkdir(parents=True, exist_ok=True)

    # Copy full per-seed directories into self-contained campaign archive
    archived_seeds = []
    for seed in args.seeds:
        src_seed = runs_base_dir / f"seed_{seed}"
        dst_seed = campaign_archive_dir / f"seed_{seed}"
        if src_seed.exists():
            if dst_seed.exists():
                shutil.rmtree(dst_seed)
            shutil.copytree(src_seed, dst_seed)
            archived_seeds.append(f"seed_{seed}")

    df_seeds.to_csv(campaign_archive_dir / csv_name, index=False)
    df_summary.to_csv(campaign_archive_dir / summary_name, index=False)

    campaign_manifest = {
        "status": "completed",
        "eligible_for_aggregation": True,
        "campaign_mode": mode_name,
        "seeds": args.seeds,
        "config_path": args.config,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archived_seed_directories": archived_seeds,
        "per_seed_csv": csv_name,
        "summary_csv": summary_name,
    }
    with open(campaign_archive_dir / "campaign_manifest.json", "w", encoding="utf-8") as f:
        json.dump(campaign_manifest, f, indent=2)

    generate_seed_checksums(campaign_archive_dir)
    print(
        f"Archived self-contained campaign evidence package and checksums to {campaign_archive_dir}"
    )

    print(f"\n=== 3-Seed {mode_str} Summary (Sample SD ddof=1) ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
