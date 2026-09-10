"""Data preparation script constructing canonical DeepDRiD manifest per blueprint Section 8."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.retinaguard.data.adapters import (
    build_canonical_manifest,
    load_deepdrid_label_mapping,
    parse_deepdrid_metadata,
)
from src.retinaguard.utils.hashing import compute_sha256


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepDRiD canonical manifest from official dataset."
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="data/raw/deepdrid/regular_fundus_quality.csv",
        help="Path to official DeepDRiD quality label CSV file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/deepdrid/images",
        help="Path to official DeepDRiD images directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/manifests/deepdrid_manifest.csv",
        help="Destination path for canonical manifest CSV",
    )
    parser.add_argument(
        "--exclusions-csv",
        type=str,
        default="data/manifests/deepdrid_exclusions.csv",
        help="Destination path for excluded records CSV",
    )
    args = parser.parse_args()

    dfs = []
    exclusions_list = []
    ex_p = Path(args.exclusions_csv)

    # Check for multi-fold official DeepDRiD structure
    ext_dir = Path("external/DeepDRiD/regular_fundus_images")
    if ext_dir.is_dir() and not Path(args.labels_csv).is_file():
        print(f"Found official multi-fold DeepDRiD repository at {ext_dir}...")
        train_csv = ext_dir / "regular-fundus-training" / "regular-fundus-training.csv"
        train_img = ext_dir / "regular-fundus-training" / "Images"
        val_csv = ext_dir / "regular-fundus-validation" / "regular-fundus-validation.csv"
        val_img = ext_dir / "regular-fundus-validation" / "Images"
        eval_csv = ext_dir / "Online-Challenge1&2-Evaluation" / "Challenge2_labels.xlsx"
        eval_img = ext_dir / "Online-Challenge1&2-Evaluation" / "Images"

        # Require all 3 official folds explicitly
        missing_components = []
        if not train_csv.is_file() or not train_img.is_dir():
            missing_components.append("training fold (regular-fundus-training)")
        if not val_csv.is_file() or not val_img.is_dir():
            missing_components.append("validation fold (regular-fundus-validation)")
        if not eval_csv.is_file() or not eval_img.is_dir():
            missing_components.append("external evaluation fold (Online-Challenge1&2-Evaluation)")

        if missing_components:
            raise FileNotFoundError(
                f"Incomplete DeepDRiD repository. Missing required folds: {missing_components}.\n"
                "Scientific evaluation requires all three official DeepDRiD folds."
            )

        def safe_save_exclusions():
            ex_p.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            from src.retinaguard.data.adapters import EXCLUSION_COLUMNS

            pd.DataFrame(exclusions_list, columns=EXCLUSION_COLUMNS).to_csv(ex_p, index=False)

        try:
            print(f"Parsing DeepDRiD training fold from {train_csv}...")
            df_train = parse_deepdrid_metadata(
                train_csv, train_img, source_split="train", exclusions_list=exclusions_list
            )
            assert len(df_train) == 1200, f"Expected 1200 training images, found {len(df_train)}"
            assert (
                df_train["patient_id"].nunique() == 300
            ), f"Expected 300 training patients, found {df_train['patient_id'].nunique()}"
            dfs.append(df_train)

            print(f"Parsing DeepDRiD validation fold from {val_csv}...")
            df_val = parse_deepdrid_metadata(
                val_csv, val_img, source_split="val", exclusions_list=exclusions_list
            )
            assert len(df_val) == 400, f"Expected 400 validation images, found {len(df_val)}"
            assert (
                df_val["patient_id"].nunique() == 100
            ), f"Expected 100 validation patients, found {df_val['patient_id'].nunique()}"
            dfs.append(df_val)

            print(f"Parsing DeepDRiD external evaluation fold from {eval_csv}...")
            df_eval = parse_deepdrid_metadata(
                eval_csv, eval_img, source_split="external_test", exclusions_list=exclusions_list
            )
            assert len(df_eval) == 400, f"Expected 400 evaluation images, found {len(df_eval)}"
            assert (
                df_eval["patient_id"].nunique() == 100
            ), f"Expected 100 evaluation patients, found {df_eval['patient_id'].nunique()}"
            dfs.append(df_eval)
        finally:
            safe_save_exclusions()

    else:
        labels_p = Path(args.labels_csv)
        if not labels_p.is_file():
            raise FileNotFoundError(
                f"Official DeepDRiD label file not found at: {args.labels_csv}.\n"
                "Please obtain the official DeepDRiD repository & annotations legally per docs/data-card.md."
            )

        images_p = Path(args.images_dir)
        if not images_p.is_dir():
            raise FileNotFoundError(
                f"Official DeepDRiD images directory not found at: {args.images_dir}.\n"
                "Please download authorized DeepDRiD images into data/raw/deepdrid/images/ per docs/data-card.md."
            )

        try:
            print(f"Parsing official DeepDRiD metadata from {labels_p} and images in {images_p}...")
            dfs.append(
                parse_deepdrid_metadata(labels_p, str(images_p), exclusions_list=exclusions_list)
            )
        finally:
            ex_p.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            from src.retinaguard.data.adapters import EXCLUSION_COLUMNS

            pd.DataFrame(exclusions_list, columns=EXCLUSION_COLUMNS).to_csv(ex_p, index=False)

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = build_canonical_manifest(dfs, out_p)

    mapping_info = load_deepdrid_label_mapping()
    manifest_sha = compute_sha256(out_p)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    source_folds_info = {}
    if ext_dir.is_dir() and not Path(args.labels_csv).is_file():
        source_folds_info = {
            "train": {
                "images": len(df_train),
                "patients": int(df_train["patient_id"].nunique()),
                "raw_labels_path": str(train_csv),
                "raw_labels_sha256": compute_sha256(train_csv),
            },
            "val": {
                "images": len(df_val),
                "patients": int(df_val["patient_id"].nunique()),
                "raw_labels_path": str(val_csv),
                "raw_labels_sha256": compute_sha256(val_csv),
            },
            "external_test": {
                "images": len(df_eval),
                "patients": int(df_eval["patient_id"].nunique()),
                "raw_labels_path": str(eval_csv),
                "raw_labels_sha256": compute_sha256(eval_csv),
            },
        }
    else:
        source_folds_info = {
            "custom": {
                "images": len(manifest_df),
                "patients": int(manifest_df["patient_id"].nunique()),
                "raw_labels_path": str(args.labels_csv),
                "raw_labels_sha256": (
                    compute_sha256(args.labels_csv) if Path(args.labels_csv).is_file() else "N/A"
                ),
            }
        }

    metadata = {
        "dataset": "DeepDRiD",
        "manifest_path": str(out_p),
        "manifest_sha256": manifest_sha,
        "mapping_path": mapping_info.get("mapping_path", "configs/deepdrid_label_mapping.yaml"),
        "mapping_sha256": mapping_info.get("mapping_sha256", "N/A"),
        "source_folds": source_folds_info,
        "total_records": len(manifest_df),
        "total_patients": int(manifest_df["patient_id"].nunique()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }

    meta_p = out_p.parent / f"{out_p.stem}.metadata.json"
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== DeepDRiD Canonical Label Mapping & Distribution Verification ===")
    print(f"Mapping Schema SHA-256: {mapping_info.get('mapping_sha256', 'N/A')}")
    print(f"Manifest SHA-256: {manifest_sha}")
    for col in ["overall_quality_canonical", "artifact", "clarity", "field_definition"]:
        counts = dict(manifest_df[col].value_counts(dropna=False).sort_index())
        print(f"  {col}: {counts}")

    print(
        f"\nSuccessfully constructed canonical DeepDRiD manifest ({len(manifest_df)} records, {manifest_df['patient_id'].nunique()} patients) at: {out_p}"
    )
    print(f"Persistent provenance metadata saved to: {meta_p}")
    print(f"Total exclusions across all folds: {len(exclusions_list)} (written to {ex_p})")


if __name__ == "__main__":
    main()
