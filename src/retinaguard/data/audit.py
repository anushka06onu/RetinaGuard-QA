"""Dataset duplicate, leakage, provenance, and integrity auditing per Phase 3 of blueprint."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import imagehash
import pandas as pd
from PIL import Image

from retinaguard.utils.hashing import compute_sha256


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
    from retinaguard.data.adapters import extract_patient_and_eye

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
    splits: Dict[str, pd.DataFrame],
    reports_dir: Union[str, Path] = "artifacts/reports",
    split_file_paths: Optional[Dict[str, Union[str, Path]]] = None,
    mapping_config_path: Union[str, Path] = "configs/deepdrid_label_mapping.yaml",
) -> Dict[str, Any]:
    """Verify 0% patient and hash leakage across splits with complete provenance and integrity checks."""
    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)

    # Git commit provenance
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    # Script and mapping hashes
    script_p = Path("scripts/audit_dataset.py")
    script_sha = compute_sha256(script_p) if script_p.is_file() else "N/A"
    mapping_p = Path(mapping_config_path)
    mapping_sha = compute_sha256(mapping_p) if mapping_p.is_file() else "N/A"

    patient_sets: Dict[str, Set[str]] = {}
    sha_sets: Dict[str, Set[str]] = {}
    split_details: Dict[str, Any] = {}

    for split_name, df in splits.items():
        p_series = (
            df["patient_id"].dropna().astype(str)
            if "patient_id" in df.columns
            else pd.Series([], dtype=str)
        )
        sha_series = (
            df["sha256"].dropna().astype(str)
            if "sha256" in df.columns
            else pd.Series([], dtype=str)
        )

        patient_sets[split_name] = set(p_series)
        sha_sets[split_name] = set(sha_series)

        # Intra-split duplicates
        sha_counts = sha_series.value_counts()
        intra_dups = sha_counts[sha_counts > 1].to_dict()

        # Split file hash
        split_path_str = (
            str(split_file_paths.get(split_name))
            if split_file_paths and split_name in split_file_paths
            else f"data/splits/{split_name}.csv"
        )
        split_file_p = Path(split_path_str)
        split_file_sha = compute_sha256(split_file_p) if split_file_p.is_file() else "N/A"

        # Label distributions
        label_dist = {}
        for col in [
            "overall_quality_canonical",
            "artifact",
            "clarity",
            "field_definition",
            "quality_canonical",
        ]:
            if col in df.columns:
                label_dist[col] = {
                    str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()
                }

        # Image existence & recomputed SHA-256 verification (sample up to 50 or all if available)
        verified_images = 0
        missing_images = 0
        mismatched_hashes = 0
        if "path" in df.columns and "sha256" in df.columns:
            for _, row in df.iterrows():
                img_p = Path(row["path"])
                if img_p.is_file():
                    try:
                        actual_img_sha = compute_sha256(img_p)
                        if actual_img_sha == str(row["sha256"]):
                            verified_images += 1
                        else:
                            mismatched_hashes += 1
                    except Exception:
                        mismatched_hashes += 1
                else:
                    missing_images += 1

        split_details[split_name] = {
            "path": split_path_str,
            "sha256": split_file_sha,
            "records": len(df),
            "unique_patients": (
                int(df["patient_id"].dropna().nunique()) if "patient_id" in df.columns else 0
            ),
            "unique_image_hashes": (
                int(df["sha256"].dropna().nunique()) if "sha256" in df.columns else 0
            ),
            "intra_split_duplicates": intra_dups,
            "label_distributions": label_dist,
            "image_integrity": {
                "verified_images": verified_images,
                "missing_images": missing_images,
                "mismatched_hashes": mismatched_hashes,
            },
        }

    # Cross-split leakage checks
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
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "audit_script_sha256": script_sha,
        "mapping_sha256": mapping_sha,
        "splits": split_details,
        "comparisons_performed": [
            "patient_id_cross_split",
            "sha256_cross_split",
            "intra_split_duplicates",
            "image_file_existence_and_hash_verification",
            "label_distribution_verification",
        ],
        "patient_leakages": patient_leaks,
        "sha256_leakages": sha_leaks,
        "isolation_passed": isolation_passed,
        "split_counts": {s: len(df) for s, df in splits.items()},
        "patient_counts": {s: len(p_ids) for s, p_ids in patient_sets.items()},
    }

    # Save canonical reports (both cross_split_isolation_audit and data_audit for compatibility)
    for name in ["cross_split_isolation_audit.json", "data_audit.json"]:
        with open(reports_p / name, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)

    # Save Markdown report
    md_content = f"""# RetinaGuard-QA Cross-Split Isolation & Data Audit Report

## Summary
- **Schema Version:** 1.0
- **Generated At UTC:** {audit_summary['generated_at_utc']}
- **Git Commit:** `{git_commit}`
- **Audit Script SHA-256:** `{script_sha}`
- **Mapping Schema SHA-256:** `{mapping_sha}`
- **Isolation Status:** {'✅ PASSED (Zero Leakage)' if isolation_passed else '❌ FAILED (Leakage Detected)'}
- **Splits Evaluated:** {', '.join(split_names)}

## Partition Summary & Cryptographic Hashes
| Split | Split File SHA-256 | Images | Unique Patients | Unique Hashes | Intra-Split Dups |
|---|---|:---:|:---:|:---:|:---:|
"""
    for s, d in split_details.items():
        sha_short = f"`{d['sha256'][:16]}...`" if len(d["sha256"]) == 64 else d["sha256"]
        md_content += f"| **{s}** | {sha_short} | {d['records']} | {d['unique_patients']} | {d['unique_image_hashes']} | {len(d['intra_split_duplicates'])} |\n"

    md_content += "\n## Patient Isolation Verification\n"
    if patient_leaks:
        for k, v in patient_leaks.items():
            md_content += f"- ❌ **{k}:** {len(v)} leaked patient IDs: {v[:5]}...\n"
    else:
        md_content += "- ✅ **Zero Patient Leakage:** No patient identities cross partition boundaries (0% patient leakage across all splits).\n"

    md_content += "\n## Cryptographic Hash (SHA-256) Isolation Verification\n"
    if sha_leaks:
        for k, v in sha_leaks.items():
            md_content += f"- ❌ **{k}:** {len(v)} duplicate file hashes crossing splits.\n"
    else:
        md_content += "- ✅ **Zero Image Leakage:** No duplicate files or identical images cross partition boundaries.\n"

    md_content += "\n## Comparisons Performed\n"
    for c in audit_summary["comparisons_performed"]:
        md_content += f"- `{c}`\n"

    for name in ["cross_split_isolation_audit.md", "data_audit.md"]:
        with open(reports_p / name, "w", encoding="utf-8") as f:
            f.write(md_content)

    return audit_summary
