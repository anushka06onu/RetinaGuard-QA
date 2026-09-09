"""Dataset duplicate, leakage, and integrity auditing per Phase 3 of blueprint."""

import json
from pathlib import Path
from typing import Dict, List, Set, Union, Any, Optional
import pandas as pd
from PIL import Image
import imagehash

from src.retinaguard.utils.hashing import compute_sha256


def audit_dataset_integrity(manifest_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit single manifest for duplicates, class counts, missing fields, and corrupt images."""
    total_images = len(manifest_df)
    readable_count = 0
    missing_labels = manifest_df["quality_canonical"].isna().sum()

    # Exact duplicate detection
    sha_counts = manifest_df["sha256"].dropna().value_counts()
    exact_duplicates = sha_counts[sha_counts > 1].to_dict()

    # Quality class distribution
    class_dist = manifest_df["quality_canonical"].value_counts(dropna=False).to_dict()

    # Patient counts
    unique_patients = manifest_df["patient_id"].dropna().nunique()

    # Dimensions
    resolutions = []
    for _, row in manifest_df.iterrows():
        p = Path(row["path"])
        if p.exists():
            try:
                with Image.open(p) as img:
                    img.verify()
                readable_count += 1
                resolutions.append((row["width"], row["height"]))
            except Exception:
                pass

    return {
        "total_images": total_images,
        "readable_images": readable_count,
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "missing_label_count": int(missing_labels),
        "unique_patients": int(unique_patients),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "resolution_samples": resolutions[:10]
    }


def run_leakage_and_duplicate_audit(
    splits: Dict[str, pd.DataFrame],
    reports_dir: Union[str, Path] = "artifacts/reports"
) -> Dict[str, Any]:
    """Verify 0% patient and hash leakage across splits, and generate audit reports."""
    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)

    patient_sets: Dict[str, Set[str]] = {}
    sha_sets: Dict[str, Set[str]] = {}

    for split_name, df in splits.items():
        patient_sets[split_name] = set(df["patient_id"].dropna().astype(str))
        sha_sets[split_name] = set(df["sha256"].dropna().astype(str))

    patient_leaks = {}
    sha_leaks = {}
    split_names = list(splits.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            p_inter = patient_sets[s1].intersection(patient_sets[s2])
            if p_inter:
                patient_leaks[f"{s1}_vs_{s2}"] = sorted(list(p_inter))

            sha_inter = sha_sets[s1].intersection(sha_sets[s2])
            if sha_inter:
                sha_leaks[f"{s1}_vs_{s2}"] = sorted(list(sha_inter))

    isolation_passed = (len(patient_leaks) == 0) and (len(sha_leaks) == 0)

    audit_summary = {
        "isolation_passed": isolation_passed,
        "split_counts": {s: len(df) for s, df in splits.items()},
        "patient_counts": {s: len(p_ids) for s, p_ids in patient_sets.items()},
        "patient_leakages": patient_leaks,
        "sha256_leakages": sha_leaks
    }

    # Save JSON report
    with open(reports_p / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Save Markdown report
    md_content = f"""# RetinaGuard-QA Data & Leakage Audit Report

## Summary
- **Isolation Status:** {'✅ PASSED (Zero Leakage)' if isolation_passed else '❌ FAILED (Leakage Detected)'}
- **Splits Evaluated:** {', '.join(split_names)}

## Partition Counts
| Split | Total Images | Unique Patients |
|---|---|---|
"""
    for s in split_names:
        md_content += f"| {s} | {len(splits[s])} | {len(patient_sets[s])} |\n"

    md_content += f"\n## Patient Leakage Details\n"
    if patient_leaks:
        for k, v in patient_leaks.items():
            md_content += f"- **{k}:** {len(v)} leaked patient IDs: {v[:5]}...\n"
    else:
        md_content += "No patient identities cross partition boundaries (0% patient leakage).\n"

    md_content += f"\n## Hash (SHA-256) Leakage Details\n"
    if sha_leaks:
        for k, v in sha_leaks.items():
            md_content += f"- **{k}:** {len(v)} duplicate file hashes crossing splits.\n"
    else:
        md_content += "No duplicate files or identical images cross partition boundaries.\n"

    with open(reports_p / "data_audit.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    return audit_summary
