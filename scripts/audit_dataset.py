"""Dataset audit runner creating data_audit.json and data_audit.md from actual split manifests."""

import argparse
from pathlib import Path

import pandas as pd

from src.retinaguard.data.audit import (
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
    args = parser.parse_args()

    splits_p = Path(args.splits_dir)
    reports_p = Path(args.reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)

    print(f"=== Auditing Actual Saved Splits in {splits_p} ===")
    split_files = list(splits_p.glob("*.csv"))

    if not split_files:
        print(
            f"No split files found in {splits_p}. Please run scripts/prepare_eyeq.py and scripts/create_splits.py after obtaining datasets."
        )
        return

    splits = {}
    for f in split_files:
        df = pd.read_csv(f)
        splits[f.stem] = df

    report = run_leakage_and_duplicate_audit(splits, reports_dir=reports_p)
    print(f"Audited {len(splits)} partitions: {list(splits.keys())}")
    print(
        f"Leakage Audit Status: {'PASSED (Zero Leakage)' if report['isolation_passed'] else 'FAILED (Leakage Detected)'}"
    )
    print(
        f"Saved audit reports to {reports_p / 'data_audit.json'} and {reports_p / 'data_audit.md'}"
    )


if __name__ == "__main__":
    main()
