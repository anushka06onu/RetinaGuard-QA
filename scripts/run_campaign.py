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
import yaml

from retinaguard.evaluation.provenance import validate_empirical_result
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.training.train import run_training_experiment
from retinaguard.utils.hashing import compute_sha256

try:
    from scripts.evaluate import evaluate_dataset_partition
except ModuleNotFoundError:
    from evaluate import evaluate_dataset_partition


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


def verify_campaign_archive(archive_dir: Path) -> Tuple[bool, str]:
    """Independently verify a self-contained campaign archive directory with full cryptographic and structural integrity."""
    manifest_file = archive_dir / "campaign_manifest.json"
    if not manifest_file.is_file():
        return False, "Missing campaign_manifest.json"

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            c_manifest = json.load(f)
    except Exception as e:
        return False, f"Malformed campaign_manifest.json: {e}"

    if c_manifest.get("status") not in ["completed", "fixture_completed"]:
        return False, f"Campaign status not valid completed state: {c_manifest.get('status')}"

    sums_file = archive_dir / "SHA256SUMS"
    if not sums_file.is_file():
        return False, "Missing top-level SHA256SUMS"

    seen_rel_paths = set()
    archive_dir_resolved = archive_dir.resolve()

    with open(sums_file, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                return False, f"Malformed line {line_num} in campaign SHA256SUMS: {raw_line}"
            expected_sha, rel_str = parts
            if len(expected_sha) != 64 or not all(
                c in "0123456789abcdefABCDEF" for c in expected_sha
            ):
                return False, f"Invalid SHA-256 hash at line {line_num}: {expected_sha}"
            if rel_str.startswith("/") or rel_str.startswith("\\"):
                return False, f"Absolute path forbidden in SHA256SUMS: {rel_str}"
            if ".." in rel_str.replace("\\", "/").split("/"):
                return False, f"Path traversal forbidden in SHA256SUMS: {rel_str}"
            if rel_str in seen_rel_paths:
                return False, f"Duplicate entry in SHA256SUMS: {rel_str}"
            seen_rel_paths.add(rel_str)

            target_p = (archive_dir / rel_str).resolve()
            if not target_p.is_relative_to(archive_dir_resolved):
                return False, f"Path points outside campaign archive directory: {rel_str}"
            if not target_p.is_file():
                return False, f"Missing file listed in SHA256SUMS: {rel_str}"
            actual_sha = compute_sha256(target_p)
            if actual_sha.lower() != expected_sha.lower():
                return False, f"Checksum mismatch for {rel_str}: {actual_sha} != {expected_sha}"

    # Bidirectional: verify every file on disk is listed in SHA256SUMS
    for disk_p in archive_dir.rglob("*"):
        if disk_p.is_file() and disk_p.name != "SHA256SUMS":
            rel_disk = str(disk_p.relative_to(archive_dir))
            if rel_disk not in seen_rel_paths:
                return (
                    False,
                    f"Unlisted file on disk not recorded in campaign SHA256SUMS: {rel_disk}",
                )

    # Validate all metric JSON files in all archived seeds
    for seed_metrics_dir in sorted(archive_dir.glob("seed_*/metrics")):
        for m_file in sorted(seed_metrics_dir.glob("*.json")):
            try:
                res = validate_empirical_result(m_file)
                if not res.get("valid"):
                    return False, f"Invalid metric result in archive: {m_file}"
            except Exception as e:
                return False, f"Archive metric validation error for {m_file}: {e}"

    # Validate run_manifest.json in each archived seed
    for seed_dir in sorted(archive_dir.glob("seed_*")):
        if seed_dir.is_dir():
            s_manifest_p = seed_dir / "run_manifest.json"
            if not s_manifest_p.is_file():
                return False, f"Missing run_manifest.json in {seed_dir.name}"
            try:
                with open(s_manifest_p, "r", encoding="utf-8") as f:
                    s_manifest = json.load(f)
                if s_manifest.get("status") not in ["completed", "fixture_completed"]:
                    return (
                        False,
                        f"Seed run status not in completed state in {seed_dir.name}: {s_manifest.get('status')}",
                    )
            except Exception as e:
                return False, f"Malformed run_manifest.json in {seed_dir.name}: {e}"

    return True, "Archive verified successfully"


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
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Run campaign using test fixture datasets for fast verification and smoke testing",
    )
    args = parser.parse_args()

    if args.fixture_mode:
        if args.eyeq_split == "data/splits/eyeq_test.csv" and not Path(args.eyeq_split).exists():
            args.eyeq_split = "tests/fixtures/splits/eyeq_test.csv"
        if (
            args.deepdrid_split == "data/splits/deepdrid_external_test.csv"
            and not Path(args.deepdrid_split).exists()
        ):
            args.deepdrid_split = "tests/fixtures/splits/deepdrid_test.csv"

    # Check enabled datasets from config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_loaded = yaml.safe_load(f) or {}
    datasets_cfg = cfg_loaded.get("data", {}).get("datasets", {})
    eyeq_enabled = datasets_cfg.get("eyeq", {}).get("enabled", False)
    deepdrid_enabled = datasets_cfg.get("deepdrid", {}).get("enabled", True)

    is_exploratory = args.smoke_test or args.fixture_mode
    if not is_exploratory:
        if len(args.seeds) < 3:
            raise ValueError(
                f"Final campaign mode requires at least 3 unique seeds, got {len(args.seeds)}: {args.seeds}"
            )
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError(f"Duplicate seeds detected in final campaign mode: {args.seeds}")
        if eyeq_enabled and not Path(args.eyeq_split).is_file():
            raise FileNotFoundError(
                f"Final campaign protocol requires EyeQ split at {args.eyeq_split}"
            )
        if deepdrid_enabled and not Path(args.deepdrid_split).is_file():
            raise FileNotFoundError(
                f"Final campaign protocol requires DeepDRiD split at {args.deepdrid_split}"
            )

    mode_name = "zero_shot" if args.zero_shot else "multitask"
    mode_str = "Zero-Shot Transfer Campaign" if args.zero_shot else "Supervised Multi-Task Campaign"

    runs_base_dir = Path(args.output_runs_dir) / mode_name
    runs_base_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    campaigns_dir = Path(args.output_campaigns_dir)
    campaigns_dir.mkdir(parents=True, exist_ok=True)
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
            fixture_mode=args.fixture_mode,
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
            try:
                ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
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
        else:
            try:
                ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_state = torch.load(ckpt_path, map_location="cpu")
            ckpt_meta = ckpt_state.get("metadata", {})

        print(f">>> [SEED {seed}] Checkpoint verified at {ckpt_path}. Evaluating test splits...")

        ckpt_img_size = (
            ckpt_meta.get("resolved_config", {}).get("training", {}).get("image_size", 384)
        )
        model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_path)
        model.eval()

        created_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        seed_entry: Dict[str, Any] = {
            "seed": seed,
            "best_validation_objective": train_res.get(
                "best_validation_objective", train_res.get("best_val_macro_f1")
            ),
            "best_val_macro_f1": train_res.get("best_val_macro_f1"),
            "selection_metric": train_res.get("selection_metric", "primary_macro_f1"),
            "selection_mode": train_res.get("selection_mode", "max"),
            "epochs_trained": train_res["epochs_trained"],
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha256,
            "config_path": args.config,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
        }

        eyeq_completed = False
        deepdrid_completed = False

        # EyeQ test evaluation
        if Path(args.eyeq_split).is_file():
            eyeq_eval, df_eyeq_preds = evaluate_dataset_partition(
                model,
                args.eyeq_split,
                "EyeQ Test",
                is_deepdrid=False,
                image_size=ckpt_img_size,
                allow_synthetic_fallback=args.fixture_mode,
            )
            pred_file = preds_dir / "eyeq_test_predictions.csv"
            df_eyeq_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            eval_status = "fixture_completed" if args.fixture_mode else "completed"
            is_eligible = not args.fixture_mode
            origin = "constructed_fixture" if args.fixture_mode else "authorized_dataset"

            eyeq_eval.update(
                {
                    "status": eval_status,
                    "eligible_as_final_result": is_eligible,
                    "data_origin": origin,
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
            eyeq_completed = True

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
                image_size=ckpt_img_size,
                allow_synthetic_fallback=args.fixture_mode,
            )
            pred_file = preds_dir / pred_file_name
            df_dd_preds.to_csv(pred_file, index=False)
            pred_sha = compute_sha256(pred_file)

            eval_status = "fixture_completed" if args.fixture_mode else "completed"
            is_eligible = not args.fixture_mode
            origin = "constructed_fixture" if args.fixture_mode else "authorized_dataset"

            dd_eval.update(
                {
                    "status": eval_status,
                    "eligible_as_final_result": is_eligible,
                    "data_origin": origin,
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
            deepdrid_completed = True

        required_checks = []
        if Path(args.eyeq_split).is_file():
            required_checks.append(eyeq_completed)
        if Path(args.deepdrid_split).is_file():
            required_checks.append(deepdrid_completed)

        all_required_done = (
            all(required_checks)
            if not is_exploratory
            else any(required_checks)
        ) if required_checks else False

        if not is_exploratory and not all_required_done:
            raise RuntimeError(
                f"Seed {seed} failed to complete all protocol-required evaluations: "
                f"EyeQ: {eyeq_completed} (required={Path(args.eyeq_split).is_file()}), "
                f"DeepDRiD: {deepdrid_completed} (required={Path(args.deepdrid_split).is_file()})"
            )

        # Save Run Manifest with explicit completion status (Item 6 & 9)
        run_manifest = {
            "status": "fixture_completed" if args.fixture_mode else "completed",
            "eligible_for_aggregation": not args.fixture_mode,
            "data_origin": "constructed_fixture" if args.fixture_mode else "authorized_dataset",
            "campaign_mode": mode_name,
            "seed": seed,
            "eyeq_test_completed": eyeq_completed,
            "deepdrid_test_completed": deepdrid_completed,
            "all_required_evaluations_completed": all_required_done,
            "config_path": args.config,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
            "summary_metrics": seed_entry,
            "training_result": {
                "best_validation_objective": train_res.get(
                    "best_validation_objective", train_res.get("best_val_macro_f1")
                ),
                "selection_metric": train_res.get("selection_metric", "primary_macro_f1"),
                "selection_mode": train_res.get("selection_mode", "max"),
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

    # Compute Aggregate Mean +/- Sample Std (ddof=1) and 95% Confidence Intervals (Item 13)
    from scipy import stats

    numeric_cols = [
        c for c in df_seeds.select_dtypes(include=[np.number]).columns if c not in ["seed"]
    ]
    summary_data = []
    for col in numeric_cols:
        vals = pd.to_numeric(df_seeds[col], errors="coerce").dropna().values
        n = len(vals)
        if n > 0:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            if n > 1 and std_val > 1e-9:
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci_hw = float(t_crit * (std_val / np.sqrt(n)))
                ci_lower = mean_val - ci_hw
                ci_upper = mean_val + ci_hw
            else:
                ci_lower = mean_val
                ci_upper = mean_val
            summary_data.append(
                {
                    "metric": col,
                    "mean": round(mean_val, 4),
                    "std_sample": round(std_val, 4),
                    "ci_95_lower": round(ci_lower, 4),
                    "ci_95_upper": round(ci_upper, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "successful_seeds": n,
                    "expected_seeds": len(args.seeds),
                    "missing_seeds": len(args.seeds) - n,
                }
            )

    df_summary = pd.DataFrame(summary_data)
    summary_name = "zero_shot_campaign_summary.csv" if args.zero_shot else "campaign_summary.csv"
    summary_csv = metrics_dir / summary_name
    df_summary.to_csv(summary_csv, index=False)
    print(f"Exported campaign summary to {summary_csv}")

    # Select deployment checkpoint strictly using validation objective score (Item 14)
    selected_seed_entry = max(
        seed_records,
        key=lambda r: float(r.get("best_validation_objective", 0.0) or 0.0),
    )
    models_dir = campaigns_dir.parent / "models" if campaigns_dir.name == "campaigns" else campaigns_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dst = models_dir / "best.ckpt"
    
    src_ckpt_p = Path(selected_seed_entry.get("checkpoint_path", ""))
    if not args.fixture_mode and src_ckpt_p.is_file():
        shutil.copy2(src_ckpt_p, best_ckpt_dst)
        print(f"Copied selected best checkpoint ({src_ckpt_p}) to {best_ckpt_dst}")
        ckpt_rel_str = "artifacts/models/best.ckpt"
    else:
        ckpt_rel_str = str(src_ckpt_p)

    deployment_checkpoint_selection = {
        "selection_metric": selected_seed_entry.get("selection_metric", "primary_macro_f1"),
        "selection_rule": selected_seed_entry.get("selection_mode", "max"),
        "selected_seed": selected_seed_entry["seed"],
        "selected_epoch": selected_seed_entry.get("epochs_trained"),
        "validation_score": selected_seed_entry.get("best_validation_objective"),
        "checkpoint_path": ckpt_rel_str,
        "checkpoint_sha256": selected_seed_entry.get("checkpoint_sha256"),
        "selection_note": "Selected strictly by validation set objective performance across independent training seeds.",
    }
    selected_ckpt_json_p = models_dir / "selected_checkpoint.json"
    with open(selected_ckpt_json_p, "w", encoding="utf-8") as f:
        json.dump(deployment_checkpoint_selection, f, indent=2)

    # Campaign-Level Archival Directory & Checksums (Self-Contained Evidence Package)
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_archive_dir = campaigns_dir / f"{mode_name}_{timestamp_str}"
    campaign_archive_dir.mkdir(parents=True, exist_ok=True)

    # Copy resolved config into campaign archive
    archived_config_p = campaign_archive_dir / "config.yaml"
    if Path(args.config).is_file():
        shutil.copy2(args.config, archived_config_p)
    archived_config_sha256 = (
        compute_sha256(archived_config_p) if archived_config_p.is_file() else config_sha256
    )

    # Copy evaluated split manifests into campaign archive
    splits_archive_dir = campaign_archive_dir / "splits"
    splits_archive_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.eyeq_split).is_file():
        shutil.copy2(args.eyeq_split, splits_archive_dir / Path(args.eyeq_split).name)
    if Path(args.deepdrid_split).is_file():
        shutil.copy2(args.deepdrid_split, splits_archive_dir / Path(args.deepdrid_split).name)

    # Copy full per-seed directories into self-contained campaign archive and make paths relative
    archived_seeds = []
    for seed in args.seeds:
        src_seed = runs_base_dir / f"seed_{seed}"
        dst_seed = campaign_archive_dir / f"seed_{seed}"
        if src_seed.exists():
            if dst_seed.exists():
                shutil.rmtree(dst_seed)
            shutil.copytree(src_seed, dst_seed)
            archived_seeds.append(f"seed_{seed}")

            # Rewrite metric JSON files with relative paths for complete portability
            archived_metrics_dir = dst_seed / "metrics"
            if archived_metrics_dir.is_dir():
                for m_file in archived_metrics_dir.glob("*.json"):
                    with open(m_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)

                    if "checkpoint_path" in m_data:
                        m_data["checkpoint_path"] = "../" + Path(m_data["checkpoint_path"]).name
                    if "prediction_file" in m_data:
                        m_data["prediction_file"] = (
                            "../predictions/" + Path(m_data["prediction_file"]).name
                        )
                    if "split_path" in m_data:
                        m_data["split_path"] = "../../splits/" + Path(m_data["split_path"]).name

                    with open(m_file, "w", encoding="utf-8") as f:
                        json.dump(m_data, f, indent=2)

            # Rewrite run_manifest.json with relative config path
            manifest_p = dst_seed / "run_manifest.json"
            if manifest_p.is_file():
                with open(manifest_p, "r", encoding="utf-8") as f:
                    r_man = json.load(f)
                r_man["config_path"] = "../config.yaml"
                r_man["config_sha256"] = archived_config_sha256
                with open(manifest_p, "w", encoding="utf-8") as f:
                    json.dump(r_man, f, indent=2)

            # Regenerate per-seed checksums inside the archived copy
            generate_seed_checksums(dst_seed)

    df_seeds.to_csv(campaign_archive_dir / csv_name, index=False)
    df_summary.to_csv(campaign_archive_dir / summary_name, index=False)

    campaign_manifest = {
        "status": "fixture_completed" if args.fixture_mode else "completed",
        "eligible_for_aggregation": not args.fixture_mode,
        "data_origin": "constructed_fixture" if args.fixture_mode else "authorized_dataset",
        "campaign_mode": mode_name,
        "seeds": args.seeds,
        "deployment_checkpoint_selection": deployment_checkpoint_selection,
        "config_path": "config.yaml",
        "config_sha256": archived_config_sha256,
        "git_commit": git_commit,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archived_seed_directories": archived_seeds,
        "splits_directory": "splits",
        "per_seed_csv": csv_name,
        "summary_csv": summary_name,
    }
    with open(campaign_archive_dir / "campaign_manifest.json", "w", encoding="utf-8") as f:
        json.dump(campaign_manifest, f, indent=2)

    generate_seed_checksums(campaign_archive_dir)

    # Independently verify campaign archive
    is_valid_archive, arch_msg = verify_campaign_archive(campaign_archive_dir)
    if not is_valid_archive:
        raise RuntimeError(f"Campaign archive verification failed: {arch_msg}")

    print(
        f"Archived and verified self-contained campaign evidence package at {campaign_archive_dir}"
    )

    print(f"\n=== 3-Seed {mode_str} Summary (Sample SD ddof=1 & 95% CI) ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
