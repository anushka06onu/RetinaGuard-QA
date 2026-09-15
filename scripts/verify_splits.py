#!/usr/bin/env python3
"""Verify dataset splits against cryptographic provenance records.

Checks that local split CSV files match the exact SHA-256 digests, record
counts, patient counts, class distributions, and cross-split isolation recorded
in data/splits_provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure local src is in path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pandas as pd


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_splits_from_provenance(prov_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract flat mapping of split_key -> split_metadata from provenance data."""
    flat_splits = {}
    if "dataset_specifications" in prov_data:
        for ds_name, ds_info in prov_data["dataset_specifications"].items():
            ds_splits = ds_info.get("splits", {})
            for split_k, meta in ds_splits.items():
                split_name = (
                    f"{ds_name.lower()}_{split_k}"
                    if not split_k.startswith(ds_name.lower())
                    else split_k
                )
                flat_splits[split_name] = meta
    elif "splits" in prov_data:
        flat_splits = prov_data["splits"]
    return flat_splits


def verify_splits(
    provenance_path: Path,
    splits_dir: Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """Verify split CSVs against provenance manifest."""
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Provenance file not found: {provenance_path}")

    with open(provenance_path, "r") as f:
        prov_data = json.load(f)

    splits_manifest = extract_splits_from_provenance(prov_data)
    if not splits_manifest:
        raise ValueError(f"No splits defined in provenance file: {provenance_path}")

    report: Dict[str, Any] = {
        "provenance_file": str(provenance_path),
        "splits_audited": len(splits_manifest),
        "splits_passed": 0,
        "splits_failed": 0,
        "errors": [],
        "details": {},
    }

    patient_sets: Dict[str, set] = {}

    for split_key, expected_meta in splits_manifest.items():
        # Handle file attribute or fallback to filename / convention
        recorded_file = expected_meta.get("file")
        if recorded_file:
            split_path = Path(recorded_file)
            if not split_path.is_file() and not split_path.is_absolute():
                split_path = splits_dir / Path(recorded_file).name
        else:
            split_filename = expected_meta.get("filename", f"{split_key}.csv")
            split_path = splits_dir / split_filename

        split_report: Dict[str, Any] = {
            "path": str(split_path),
            "exists": split_path.is_file(),
            "sha256_match": False,
            "records_match": False,
            "patients_match": False,
            "distributions_match": False,
            "passed": False,
            "errors": [],
        }

        if not split_path.is_file():
            err_msg = f"Split file missing: {split_path}"
            split_report["errors"].append(err_msg)
            report["errors"].append(err_msg)
            report["splits_failed"] += 1
            report["details"][split_key] = split_report
            continue

        # 1. SHA-256 Check
        actual_sha = compute_sha256(split_path)
        expected_sha = expected_meta.get("sha256")
        split_report["actual_sha256"] = actual_sha
        split_report["expected_sha256"] = expected_sha

        if actual_sha == expected_sha:
            split_report["sha256_match"] = True
        else:
            err_msg = f"[{split_key}] SHA-256 mismatch: got {actual_sha}, expected {expected_sha}"
            split_report["errors"].append(err_msg)
            report["errors"].append(err_msg)

        # 2. Content Checks
        try:
            df = pd.read_csv(split_path)
        except Exception as exc:
            err_msg = f"[{split_key}] Failed to parse CSV: {exc}"
            split_report["errors"].append(err_msg)
            report["errors"].append(err_msg)
            report["splits_failed"] += 1
            report["details"][split_key] = split_report
            continue

        actual_records = len(df)
        expected_records = expected_meta.get("record_count", expected_meta.get("records"))
        split_report["actual_records"] = actual_records
        split_report["expected_records"] = expected_records
        if actual_records == expected_records:
            split_report["records_match"] = True
        else:
            err_msg = f"[{split_key}] Record count mismatch: got {actual_records}, expected {expected_records}"
            split_report["errors"].append(err_msg)
            report["errors"].append(err_msg)

        if "patient_id" in df.columns:
            actual_patients = int(df["patient_id"].dropna().nunique())
            p_series = df["patient_id"].dropna().astype(str)
            patient_sets[split_key] = set(p_series)
        else:
            actual_patients = 0
            patient_sets[split_key] = set()

        expected_patients = expected_meta.get("patient_count", expected_meta.get("unique_patients"))
        split_report["actual_patients"] = actual_patients
        split_report["expected_patients"] = expected_patients
        if expected_patients is None or actual_patients == expected_patients:
            split_report["patients_match"] = True
        else:
            err_msg = f"[{split_key}] Patient count mismatch: got {actual_patients}, expected {expected_patients}"
            split_report["errors"].append(err_msg)
            report["errors"].append(err_msg)

        # 3. Label Distribution Checks
        expected_dist = expected_meta.get(
            "class_distributions", expected_meta.get("label_distributions", {})
        )
        dist_match = True
        for col_name, expected_counts in expected_dist.items():
            if col_name not in df.columns:
                err_msg = f"[{split_key}] Missing expected label column '{col_name}'"
                split_report["errors"].append(err_msg)
                report["errors"].append(err_msg)
                dist_match = False
                continue
            actual_counts = {
                str(k): int(v) for k, v in df[col_name].value_counts(dropna=False).items()
            }
            for val_k, exp_v in expected_counts.items():
                act_v = actual_counts.get(str(val_k), 0)
                if act_v != exp_v:
                    err_msg = f"[{split_key}] Distribution mismatch for {col_name}[{val_k}]: got {act_v}, expected {exp_v}"
                    split_report["errors"].append(err_msg)
                    report["errors"].append(err_msg)
                    dist_match = False

        split_report["distributions_match"] = dist_match

        split_passed = (
            split_report["sha256_match"]
            and split_report["records_match"]
            and split_report["patients_match"]
            and split_report["distributions_match"]
        )
        split_report["passed"] = split_passed
        if split_passed:
            report["splits_passed"] += 1
        else:
            report["splits_failed"] += 1

        report["details"][split_key] = split_report

    # 4. Cross-split patient isolation verification (only within the same dataset)
    split_names = list(patient_sets.keys())
    leakages = {}
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            # Group by dataset prefix
            ds1 = s1.split("_")[0]
            ds2 = s2.split("_")[0]
            if ds1 == ds2:
                overlap = patient_sets[s1].intersection(patient_sets[s2])
                if overlap:
                    leakages[f"{s1}_vs_{s2}"] = sorted(list(overlap))
                    err_msg = f"Cross-split patient leakage detected between {s1} and {s2}: {len(overlap)} patients"
                    report["errors"].append(err_msg)

    report["patient_leakages"] = leakages
    report["overall_passed"] = report["splits_failed"] == 0 and len(leakages) == 0

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify dataset splits against provenance manifest."
    )
    parser.add_argument(
        "--provenance-file",
        type=Path,
        default=Path("data/splits_provenance.json"),
        help="Path to splits_provenance.json",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on any checksum or distribution mismatch",
    )
    args = parser.parse_args()

    print(f"Verifying splits in '{args.splits_dir}' against '{args.provenance_file}'...")
    try:
        report = verify_splits(args.provenance_file, args.splits_dir, strict=args.strict)
    except Exception as exc:
        print(f"FAILED: Error during verification: {exc}")
        return 1

    print(f"Splits checked: {report['splits_audited']}")
    print(f"Splits passed:  {report['splits_passed']}")
    print(f"Splits failed:  {report['splits_failed']}")

    if report["overall_passed"]:
        print(
            "SUCCESS: All split files match cryptographic provenance and distribution records exactly."
        )
        return 0
    else:
        print("FAILED: Provenance verification failed with the following errors:")
        for err in report["errors"]:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
