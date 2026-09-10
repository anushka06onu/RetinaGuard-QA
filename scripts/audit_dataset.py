"""Dataset audit runner creating cross_split_isolation_audit.json and data_audit.json from actual split manifests."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from retinaguard.data.audit import (
    run_leakage_and_duplicate_audit,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit dataset manifests and verify zero patient/hash leakage across splits."
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits", help="Directory containing split CSVs"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="artifacts/reports",
        help="Output directory for audit reports",
    )
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="configs/deepdrid_label_mapping.yaml",
        help="Path to DeepDRiD label mapping YAML config",
    )
    args = parser.parse_args()

    splits_p = Path(args.splits_dir)
    reports_p = Path(args.reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)

    print(f"=== Auditing Actual Saved Splits in {splits_p} ===")
    split_files = list(splits_p.glob("*.csv"))

    if not split_files:
        print(
            f"No split files found in {splits_p}. Please run scripts/prepare_deepdrid.py and scripts/create_splits.py after obtaining datasets."
        )
        return

    splits = {}
    split_file_paths = {}
    for f in split_files:
        df = pd.read_csv(f)
        splits[f.stem] = df
        split_file_paths[f.stem] = f

    report = run_leakage_and_duplicate_audit(
        splits,
        reports_dir=reports_p,
        split_file_paths=split_file_paths,
        mapping_config_path=args.mapping_config,
    )

    print(f"Audited {len(splits)} partitions: {list(splits.keys())}")
    print(f"Schema Version: {report.get('schema_version', '1.0')}")
    print(f"Git Commit: {report.get('git_commit', 'N/A')}")
    print(f"Audit Script SHA-256: {report.get('audit_script_sha256', 'N/A')}")
    print(f"Mapping Schema SHA-256: {report.get('mapping_sha256', 'N/A')}")
    print(
        f"  - Cross-Split Isolation: {'PASSED' if report.get('cross_split_isolation_passed') else 'FAILED'}"
    )
    print(
        f"  - Intra-Split Uniqueness: {'PASSED' if report.get('intra_split_uniqueness_passed') else 'FAILED'}"
    )
    print(
        f"  - Image Hash & Existence: {'PASSED' if report.get('image_integrity_passed') else 'FAILED'}"
    )
    print(
        f"  - Overall Audit Status: {'PASSED' if report.get('overall_audit_passed') else 'FAILED'}"
    )
    print(
        f"Saved audit reports to {reports_p / 'cross_split_isolation_audit.json'} and {reports_p / 'cross_split_isolation_audit.md'}"
    )

    if not report.get("overall_audit_passed", False):
        print("\n❌ OVERALL DATASET AUDIT FAILED! See reports for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
