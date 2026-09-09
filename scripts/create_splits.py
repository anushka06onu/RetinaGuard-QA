"""Generate immutable, patient-isolated splits in data/splits/ per Phase 4."""

from pathlib import Path
import pandas as pd

from src.retinaguard.data.splits import create_patient_grouped_splits, save_split_manifests


def main():
    print("=== Generating Group-Aware Patient Splits ===")
    eyeq_manifest_p = Path("data/manifests/eyeq_manifest.csv")
    deepdrid_manifest_p = Path("data/manifests/deepdrid_manifest.csv")

    if not eyeq_manifest_p.exists():
        from scripts.prepare_eyeq import main as prep_eyeq
        prep_eyeq()
    if not deepdrid_manifest_p.exists():
        from scripts.prepare_deepdrid import main as prep_deepdrid
        prep_deepdrid()

    eyeq_df = pd.read_csv(eyeq_manifest_p)
    eyeq_splits = create_patient_grouped_splits(eyeq_df, val_ratio=0.15, test_ratio=0.15, seed=2026)
    saved_eyeq = save_split_manifests(eyeq_splits, output_dir="data/splits", prefix="eyeq")
    print("Saved EyeQ splits:")
    for k, p in saved_eyeq.items():
        print(f"  - {k}: {p} ({len(eyeq_splits[k])} samples)")

    deepdrid_df = pd.read_csv(deepdrid_manifest_p)
    deepdrid_splits = create_patient_grouped_splits(deepdrid_df, val_ratio=0.20, test_ratio=0.30, seed=2026)
    saved_deepdrid = save_split_manifests(deepdrid_splits, output_dir="data/splits", prefix="deepdrid")
    print("Saved DeepDRiD splits:")
    for k, p in saved_deepdrid.items():
        print(f"  - {k}: {p} ({len(deepdrid_splits[k])} samples)")


if __name__ == "__main__":
    main()
