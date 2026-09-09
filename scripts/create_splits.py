"""Generate immutable, patient-isolated splits in data/splits/ preserving official dataset partitions."""

import argparse
from pathlib import Path

import pandas as pd

from src.retinaguard.data.splits import (
    create_patient_grouped_splits,
    save_split_manifests,
)


def split_eyeq_preserving_official_partitions(eyeq_df: pd.DataFrame, seed: int = 2026) -> dict:
    """Preserve official EyeQ test split and partition official training into patient-grouped train/val."""
    if "source_split" in eyeq_df.columns and "test" in eyeq_df["source_split"].values:
        df_test = eyeq_df[eyeq_df["source_split"] == "test"].reset_index(drop=True)
        df_train_pool = eyeq_df[eyeq_df["source_split"] != "test"].reset_index(drop=True)

        train_val_splits = create_patient_grouped_splits(
            df_train_pool, val_ratio=0.15, test_ratio=0.0, seed=seed
        )
        return {"train": train_val_splits["train"], "val": train_val_splits["val"], "test": df_test}
    else:
        return create_patient_grouped_splits(eyeq_df, val_ratio=0.15, test_ratio=0.15, seed=seed)


def split_deepdrid_preserving_official_partitions(
    deepdrid_df: pd.DataFrame, seed: int = 2026
) -> dict:
    """Preserve official DeepDRiD published folds/partitions if present."""
    if "source_split" in deepdrid_df.columns:
        splits = {}
        for s_name in ["train", "val", "test", "external_test"]:
            sub = deepdrid_df[deepdrid_df["source_split"] == s_name].reset_index(drop=True)
            if len(sub) > 0:
                splits[s_name if s_name != "external_test" else "test"] = sub
        if "train" in splits and len(splits) >= 2:
            return splits

    return create_patient_grouped_splits(deepdrid_df, val_ratio=0.20, test_ratio=0.30, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Create immutable patient-isolated dataset splits."
    )
    parser.add_argument("--eyeq-manifest", type=str, default="data/manifests/eyeq_manifest.csv")
    parser.add_argument(
        "--deepdrid-manifest", type=str, default="data/manifests/deepdrid_manifest.csv"
    )
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    eyeq_p = Path(args.eyeq_manifest)
    deepdrid_p = Path(args.deepdrid_manifest)

    if not eyeq_p.is_file() and not deepdrid_p.is_file():
        raise FileNotFoundError(
            "No canonical dataset manifests found in data/manifests/.\n"
            "Please run scripts/prepare_eyeq.py or scripts/prepare_deepdrid.py after obtaining datasets."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if eyeq_p.is_file():
        print(f"Loading EyeQ manifest from {eyeq_p}...")
        eyeq_df = pd.read_csv(eyeq_p)
        eyeq_splits = split_eyeq_preserving_official_partitions(eyeq_df, seed=args.seed)
        saved_eyeq = save_split_manifests(eyeq_splits, output_dir=out_dir, prefix="eyeq")
        print("Saved EyeQ splits:")
        for k, p in saved_eyeq.items():
            print(f"  - {k}: {p} ({len(eyeq_splits[k])} samples)")

    if deepdrid_p.is_file():
        print(f"Loading DeepDRiD manifest from {deepdrid_p}...")
        deepdrid_df = pd.read_csv(deepdrid_p)
        deepdrid_splits = split_deepdrid_preserving_official_partitions(deepdrid_df, seed=args.seed)
        saved_deepdrid = save_split_manifests(
            deepdrid_splits, output_dir=out_dir, prefix="deepdrid"
        )
        print("Saved DeepDRiD splits:")
        for k, p in saved_deepdrid.items():
            print(f"  - {k}: {p} ({len(deepdrid_splits[k])} samples)")


if __name__ == "__main__":
    main()
