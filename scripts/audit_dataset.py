"""Dataset audit runner creating data_audit.json and data_audit.md."""

import argparse
from pathlib import Path
import pandas as pd

from src.retinaguard.data.audit import audit_dataset_integrity, run_leakage_and_duplicate_audit


def main():
    print("=== Running Complete Dataset & Patient Leakage Audit ===")
    eyeq_manifest_p = Path("data/manifests/eyeq_manifest.csv")
    deepdrid_manifest_p = Path("data/manifests/deepdrid_manifest.csv")

    # If manifests don't exist, generate sample fixtures
    if not eyeq_manifest_p.exists():
        from scripts.prepare_eyeq import main as prep_eyeq
        prep_eyeq()
    if not deepdrid_manifest_p.exists():
        from scripts.prepare_deepdrid import main as prep_deepdrid
        prep_deepdrid()

    eyeq_df = pd.read_csv(eyeq_manifest_p)
    deepdrid_df = pd.read_csv(deepdrid_manifest_p)

    eyeq_audit = audit_dataset_integrity(eyeq_df)
    print(f"EyeQ Images: {eyeq_audit['total_images']} | Patients: {eyeq_audit['unique_patients']} | Duplicates: {eyeq_audit['exact_duplicate_groups']}")

    # Partition check across splits
    splits = {
        "eyeq_train": eyeq_df.iloc[:70],
        "eyeq_val": eyeq_df.iloc[70:90],
        "eyeq_test": eyeq_df.iloc[90:],
        "deepdrid_external": deepdrid_df
    }

    report = run_leakage_and_duplicate_audit(splits, reports_dir="artifacts/reports")
    print(f"Leakage Audit Status: {'PASSED (Zero Leakage)' if report['isolation_passed'] else 'FAILED'}")
    print("Saved audit reports to artifacts/reports/data_audit.json and data_audit.md")


if __name__ == "__main__":
    main()
