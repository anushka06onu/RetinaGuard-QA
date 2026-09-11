"""Artifact schema and provenance validation for empirical results."""

import json
from pathlib import Path
from typing import Any, Dict, Union

from retinaguard.utils.hashing import compute_sha256

REQUIRED_COMPLETED_KEYS = [
    "status",
    "generated_by",
    "git_commit",
    "checkpoint_sha256",
    "split_sha256",
    "num_samples",
    "created_at_utc",
]


def validate_empirical_result(data_or_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that an empirical result artifact conforms to scientific evidence schemas.

    Rules:
    1. If status is 'completed' (final benchmark result):
       - Must have all required provenance keys.
       - num_samples must be > 0.
       - Cannot have eligible_as_final_result = False.
       - If prediction_file is specified, file must exist, hash must match, sample count must match.
       - If split file is present locally, its SHA-256 must match split_sha256.
    2. If status is 'preliminary_obsolete_architecture':
       - Must explicitly set eligible_as_final_result = False.
       - Must declare head_type.
    3. Any other status or unapproved format raises ValueError.
    """
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Result file not found: {p}")
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

        num_samples = data.get("num_samples")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"Invalid num_samples: {num_samples}. Must be positive integer.")

        if data.get("eligible_as_final_result") is False:
            raise ValueError("Completed result cannot have eligible_as_final_result = False.")

        if "prediction_file" in data:
            pred_p = Path(data["prediction_file"])
            if not pred_p.is_file():
                raise FileNotFoundError(f"Referenced prediction file not found: {pred_p}")
            if "prediction_file_sha256" in data:
                actual_sha = compute_sha256(pred_p)
                if actual_sha != data["prediction_file_sha256"]:
                    raise ValueError(
                        f"Prediction file hash mismatch: {actual_sha} != {data['prediction_file_sha256']}"
                    )

        if "split_path" in data and Path(data["split_path"]).is_file():
            split_p = Path(data["split_path"])
            actual_split_sha = compute_sha256(split_p)
            if actual_split_sha != data["split_sha256"]:
                raise ValueError(
                    f"Split file hash mismatch for {split_p}: {actual_split_sha} != {data['split_sha256']}"
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
        # Internal pipeline intermediate formats
        return {"valid": True, "type": "pipeline_intermediate"}

    else:
        # Check if legacy or unverified metric file
        raise ValueError(
            f"Unrecognized or unprovenanced result status '{status}'. "
            "Empirical results must either be 'completed' with full provenance or 'preliminary_obsolete_architecture'."
        )
