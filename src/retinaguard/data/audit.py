"""Dataset duplicate, leakage, and integrity auditing per Phase 3 of blueprint."""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import imagehash
import pandas as pd
from PIL import Image

from src.retinaguard.utils.hashing import compute_sha256


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash for a given file."""
    return compute_sha256(file_path)


def compute_perceptual_hash(
    image: Union[Image.Image, str, Path], hash_type: str = "dhash"
) -> imagehash.ImageHash:
    """Compute perceptual hash (dHash/pHash) for an image."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            img = img.convert("RGB")
            return imagehash.dhash(img) if hash_type == "dhash" else imagehash.phash(img)
    img = image.convert("RGB")
    return imagehash.dhash(img) if hash_type == "dhash" else imagehash.phash(img)


def inspect_image_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect image integrity, dimensions, color channels, and validity."""
    p = Path(file_path)
    if not p.is_file():
        return {"is_valid": False, "error": "File does not exist"}
    try:
        with Image.open(p) as img:
            w, h = img.size
            mode = img.mode
            img.verify()
        return {
            "is_valid": True,
            "width": w,
            "height": h,
            "mode": mode,
            "size_bytes": p.stat().st_size,
        }
    except Exception as exc:
        return {"is_valid": False, "error": str(exc)}


def find_duplicate_images(
    image_paths: List[Union[str, Path]], check_perceptual: bool = True, phash_threshold: int = 4
) -> Dict[str, Any]:
    """Find exact (SHA-256) and near-duplicate (dHash/pHash) image groups."""
    sha_map: Dict[str, List[str]] = {}
    phash_list: List[tuple] = []

    for p in image_paths:
        path_str = str(p)
        sha = compute_file_hash(p)
        sha_map.setdefault(sha, []).append(path_str)

        if check_perceptual:
            try:
                h = compute_perceptual_hash(p)
                phash_list.append((path_str, h))
            except Exception:
                pass

    exact_duplicates = {k: v for k, v in sha_map.items() if len(v) > 1}

    near_duplicates = []
    if check_perceptual and len(phash_list) > 1:
        for i in range(len(phash_list)):
            for j in range(i + 1, len(phash_list)):
                p1, h1 = phash_list[i]
                p2, h2 = phash_list[j]
                dist = h1 - h2
                if dist <= phash_threshold:
                    near_duplicates.append({"path_a": p1, "path_b": p2, "distance": int(dist)})

    return {
        "total_scanned": len(image_paths),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
    }


def verify_patient_split_isolation(splits: Dict[str, List[str]]) -> Dict[str, Any]:
    """Verify that filenames / patient IDs do not leak across train/val/test splits."""
    from src.retinaguard.data.adapters import extract_patient_and_eye

    split_patients: Dict[str, Set[str]] = {}
    for s_name, file_list in splits.items():
        p_set = set()
        for f in file_list:
            pid, _ = extract_patient_and_eye(f)
            if pid:
                p_set.add(pid)
        split_patients[s_name] = p_set

    leaked_patients: Dict[str, List[str]] = {}
    s_names = list(splits.keys())
    for i in range(len(s_names)):
        for j in range(i + 1, len(s_names)):
            s1, s2 = s_names[i], s_names[j]
            inter = split_patients[s1].intersection(split_patients[s2])
            if inter:
                leaked_patients[f"{s1}_vs_{s2}"] = sorted(list(inter))

    return {
        "isolation_passed": len(leaked_patients) == 0,
        "leaked_patient_ids": leaked_patients,
        "split_patient_counts": {k: len(v) for k, v in split_patients.items()},
    }


def audit_dataset_integrity(manifest_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit single manifest for duplicates, class counts, missing fields, and corrupt images."""
    total_images = len(manifest_df)
    readable_count = 0
    missing_labels = (
        manifest_df["quality_canonical"].isna().sum()
        if "quality_canonical" in manifest_df.columns
        else 0
    )

    # Exact duplicate detection
    sha_counts = manifest_df["sha256"].dropna().value_counts()
    exact_duplicates = sha_counts[sha_counts > 1].to_dict()

    # Quality class distribution
    class_dist = (
        manifest_df["quality_canonical"].value_counts(dropna=False).to_dict()
        if "quality_canonical" in manifest_df.columns
        else {}
    )

    # Patient counts
    unique_patients = (
        manifest_df["patient_id"].dropna().nunique() if "patient_id" in manifest_df.columns else 0
    )

    resolutions = []
    if "path" in manifest_df.columns:
        for _, row in manifest_df.iterrows():
            p = Path(row["path"])
            if p.is_file():
                try:
                    with Image.open(p) as img:
                        img.verify()
                    readable_count += 1
                    resolutions.append((row.get("width", 0), row.get("height", 0)))
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
        "resolution_samples": resolutions[:10],
    }


def run_leakage_and_duplicate_audit(
    splits: Dict[str, pd.DataFrame], reports_dir: Union[str, Path] = "artifacts/reports"
) -> Dict[str, Any]:
    """Verify 0% patient and hash leakage across splits, and generate audit reports."""
    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)

    patient_sets: Dict[str, Set[str]] = {}
    sha_sets: Dict[str, Set[str]] = {}

    for split_name, df in splits.items():
        patient_sets[split_name] = (
            set(df["patient_id"].dropna().astype(str)) if "patient_id" in df.columns else set()
        )
        sha_sets[split_name] = (
            set(df["sha256"].dropna().astype(str)) if "sha256" in df.columns else set()
        )

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
        "sha256_leakages": sha_leaks,
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

    md_content += "\n## Patient Leakage Details\n"
    if patient_leaks:
        for k, v in patient_leaks.items():
            md_content += f"- **{k}:** {len(v)} leaked patient IDs: {v[:5]}...\n"
    else:
        md_content += "No patient identities cross partition boundaries (0% patient leakage).\n"

    md_content += "\n## Hash (SHA-256) Leakage Details\n"
    if sha_leaks:
        for k, v in sha_leaks.items():
            md_content += f"- **{k}:** {len(v)} duplicate file hashes crossing splits.\n"
    else:
        md_content += "No duplicate files or identical images cross partition boundaries.\n"

    with open(reports_p / "data_audit.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    return audit_summary
