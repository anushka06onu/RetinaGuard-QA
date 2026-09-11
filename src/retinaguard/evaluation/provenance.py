"""Artifact schema and provenance validation for empirical results."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from retinaguard.utils.hashing import compute_sha256

REQUIRED_COMPLETED_KEYS = [
    "status",
    "eligible_as_final_result",
    "generated_by",
    "git_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "split_path",
    "split_sha256",
    "prediction_file",
    "prediction_file_sha256",
    "num_samples",
    "created_at_utc",
]


def _resolve_path(raw_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolve a path relative to base directory or repository root."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    if base_dir:
        cand = (base_dir / p).resolve()
        if cand.exists():
            return cand
    repo_root = Path.cwd()
    cand = (repo_root / p).resolve()
    if cand.exists():
        return cand
    return (base_dir / p).resolve() if base_dir else p.resolve()


def validate_empirical_result(data_or_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that an empirical result artifact conforms to scientific evidence schemas.

    Rules:
    1. If status is 'completed' (final benchmark result):
       - Must have all required provenance keys in REQUIRED_COMPLETED_KEYS.
       - eligible_as_final_result must be True.
       - num_samples must be > 0.
       - Checkpoint file must exist and hash must match checkpoint_sha256.
       - Split file must exist and hash must match split_sha256.
       - Prediction file must exist and hash must match prediction_file_sha256.
       - Prediction file must have required columns: image_id, target, prediction.
       - Prediction file must have unique image_id values and no duplicate rows.
       - Prediction row count and unique image count must equal num_samples.
       - Prediction image IDs must exactly equal evaluated split image IDs.
       - Accuracy and macro-F1 must recompute cleanly matching JSON within 1e-3 tolerance.
    2. If status is 'preliminary_obsolete_architecture':
       - Must explicitly set eligible_as_final_result = False.
       - Must declare head_type.
    3. Any other status or unapproved format raises ValueError.
    """
    base_dir = None
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Result file not found: {p}")
        base_dir = p.parent
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = data_or_path

    status = data.get("status")

    if status == "completed":
        missing = [k for k in REQUIRED_COMPLETED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"Completed empirical result missing required provenance keys: {missing}"
            )

        if data.get("eligible_as_final_result") is not True:
            raise ValueError(
                "Completed result cannot have eligible_as_final_result = False (must be True for final evidence)."
            )

        num_samples = data.get("num_samples")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"Invalid num_samples: {num_samples}. Must be positive integer.")

        ckpt_p = _resolve_path(data["checkpoint_path"], base_dir)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Referenced checkpoint file not found: {ckpt_p}")
        actual_ckpt_sha = compute_sha256(ckpt_p)
        if actual_ckpt_sha != data["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint hash mismatch for {ckpt_p}: {actual_ckpt_sha} != {data['checkpoint_sha256']}"
            )

        split_p = _resolve_path(data["split_path"], base_dir)
        if not split_p.is_file():
            raise FileNotFoundError(f"Referenced split file not found: {split_p}")
        actual_split_sha = compute_sha256(split_p)
        if actual_split_sha != data["split_sha256"]:
            raise ValueError(
                f"Split file hash mismatch for {split_p}: {actual_split_sha} != {data['split_sha256']}"
            )

        pred_p = _resolve_path(data["prediction_file"], base_dir)
        if not pred_p.is_file():
            raise FileNotFoundError(f"Referenced prediction file not found on disk: {pred_p}")

        actual_pred_sha = compute_sha256(pred_p)
        if actual_pred_sha != data["prediction_file_sha256"]:
            raise ValueError(
                f"Prediction file hash mismatch: {actual_pred_sha} != {data['prediction_file_sha256']}"
            )

        df_preds = pd.read_csv(pred_p)
        required_pred_cols = {"image_id", "target", "prediction"}
        if not required_pred_cols.issubset(df_preds.columns):
            raise ValueError(
                f"Prediction file missing required columns: {required_pred_cols - set(df_preds.columns)}"
            )

        if df_preds["image_id"].duplicated().any():
            raise ValueError("Duplicate image_id entries found in prediction file.")

        if df_preds.duplicated().any():
            raise ValueError("Duplicate rows found in prediction file.")

        if len(df_preds) != num_samples:
            raise ValueError(
                f"Prediction row count ({len(df_preds)}) does not equal reported num_samples ({num_samples})"
            )

        if df_preds["image_id"].nunique() != num_samples:
            raise ValueError(
                f"Unique prediction IDs count ({df_preds['image_id'].nunique()}) does not equal reported num_samples ({num_samples})"
            )

        df_split = pd.read_csv(split_p)
        if "image_id" not in df_split.columns:
            raise ValueError(f"Split file {split_p} missing required 'image_id' column.")

        pred_ids = set(df_preds["image_id"].astype(str))
        split_ids = set(df_split["image_id"].astype(str))
        if pred_ids != split_ids:
            diff_extra = pred_ids - split_ids
            diff_missing = split_ids - pred_ids
            raise ValueError(
                f"Prediction image IDs do not exactly equal evaluated split IDs. "
                f"Extra: {len(diff_extra)}, Missing: {len(diff_missing)}"
            )

        recomputed_acc = float(accuracy_score(df_preds["target"], df_preds["prediction"]))
        recomputed_f1 = float(
            f1_score(
                df_preds["target"], df_preds["prediction"], average="macro", zero_division=0
            )
        )

        if "accuracy" in data:
            diff_acc = abs(recomputed_acc - float(data["accuracy"]))
            if diff_acc > 1e-3:
                raise ValueError(
                    f"Recomputed accuracy ({recomputed_acc:.4f}) diverges from reported ({data['accuracy']:.4f})"
                )

        if "macro_f1" in data:
            diff_f1 = abs(recomputed_f1 - float(data["macro_f1"]))
            if diff_f1 > 1e-3:
                raise ValueError(
                    f"Recomputed macro-F1 ({recomputed_f1:.4f}) diverges from reported ({data['macro_f1']:.4f})"
                )

        return {"valid": True, "type": "completed_final_result"}

    elif status == "preliminary_obsolete_architecture":
        if data.get("eligible_as_final_result") is not False:
            raise ValueError(
                "Preliminary obsolete artifact must explicitly declare eligible_as_final_result = False."
            )
        if "head_type" not in data:
            raise ValueError("Preliminary obsolete artifact must declare head_type.")
        return {"valid": True, "type": "preliminary_obsolete_result"}

    elif status in ["evaluated", "training_history"]:
        return {"valid": True, "type": "pipeline_intermediate"}

    else:
        raise ValueError(
            f"Unrecognized or unprovenanced result status '{status}'. "
            "Empirical results must either be 'completed' with full provenance or 'preliminary_obsolete_architecture'."
        )

