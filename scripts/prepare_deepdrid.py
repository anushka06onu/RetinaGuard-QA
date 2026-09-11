"""Data preparation script constructing canonical DeepDRiD manifest per blueprint Section 8."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from retinaguard.data.adapters import (
    EXCLUSION_COLUMNS,
    build_canonical_manifest,
    load_deepdrid_label_mapping,
    parse_deepdrid_metadata,
)
from retinaguard.utils.hashing import compute_sha256


def prepare_deepdrid(
    external_root: Optional[Union[str, Path]] = None,
    output_csv: Union[str, Path] = "data/manifests/deepdrid_manifest.csv",
    exclusions_csv: Union[str, Path] = "data/manifests/deepdrid_exclusions.csv",
    labels_csv: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None,
    expected_fold_counts: Optional[Dict[str, Any]] = None,
    mapping_config_path: Union[str, Path] = "configs/deepdrid_label_mapping.yaml",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Construct canonical DeepDRiD manifest and persistent cryptographic metadata JSON."""
    dfs = []
    exclusions_list = []
    ex_p = Path(exclusions_csv)

    ext_dir = (
        Path(external_root)
        if external_root is not None
        else Path("external/DeepDRiD/regular_fundus_images")
    )

    if ext_dir.is_dir() and (labels_csv is None or not Path(labels_csv).is_file()):
        print(f"Found official multi-fold DeepDRiD repository at {ext_dir}...")
        train_csv = ext_dir / "regular-fundus-training" / "regular-fundus-training.csv"
        train_img = ext_dir / "regular-fundus-training" / "Images"
        if not train_img.is_dir():
            train_img = ext_dir / "regular-fundus-training"

        val_csv = ext_dir / "regular-fundus-validation" / "regular-fundus-validation.csv"
        val_img = ext_dir / "regular-fundus-validation" / "Images"
        if not val_img.is_dir():
            val_img = ext_dir / "regular-fundus-validation"

        eval_csv = ext_dir / "Online-Challenge1&2-Evaluation" / "Challenge2_labels.xlsx"
        if not eval_csv.is_file():
            eval_csv = ext_dir / "Online-Challenge1&2-Evaluation" / "Challenge2_labels.csv"
        eval_img = ext_dir / "Online-Challenge1&2-Evaluation" / "Images"
        if not eval_img.is_dir():
            eval_img = ext_dir / "Online-Challenge1&2-Evaluation"

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
            pd.DataFrame(exclusions_list, columns=EXCLUSION_COLUMNS).to_csv(ex_p, index=False)

        try:
            print(f"Parsing DeepDRiD training fold from {train_csv}...")
            df_train = parse_deepdrid_metadata(
                train_csv,
                train_img,
                source_split="train",
                exclusions_list=exclusions_list,
                mapping_config_path=mapping_config_path,
            )
            dfs.append(df_train)

            print(f"Parsing DeepDRiD validation fold from {val_csv}...")
            df_val = parse_deepdrid_metadata(
                val_csv,
                val_img,
                source_split="val",
                exclusions_list=exclusions_list,
                mapping_config_path=mapping_config_path,
            )
            dfs.append(df_val)

            print(f"Parsing DeepDRiD external evaluation fold from {eval_csv}...")
            df_eval = parse_deepdrid_metadata(
                eval_csv,
                eval_img,
                source_split="external_test",
                exclusions_list=exclusions_list,
                mapping_config_path=mapping_config_path,
            )
            dfs.append(df_eval)

            # Count validations
            if expected_fold_counts is None:
                train_exp = {"images": 1200, "patients": 300}
                val_exp = {"images": 400, "patients": 100}
                eval_exp = {"images": 400, "patients": 100}
            else:
                train_exp = expected_fold_counts.get("train", {})
                val_exp = expected_fold_counts.get("val", {})
                eval_exp = expected_fold_counts.get(
                    "external_test", expected_fold_counts.get("eval", {})
                )
                if isinstance(train_exp, int):
                    train_exp = {"images": train_exp}
                if isinstance(val_exp, int):
                    val_exp = {"images": val_exp}
                if isinstance(eval_exp, int):
                    eval_exp = {"images": eval_exp}

            if "images" in train_exp and train_exp["images"] is not None:
                assert len(df_train) == train_exp["images"], (
                    f"Expected {train_exp['images']} training images, found {len(df_train)}"
                )
            if "patients" in train_exp and train_exp["patients"] is not None:
                assert df_train["patient_id"].nunique() == train_exp["patients"], (
                    f"Expected {train_exp['patients']} training patients, found {df_train['patient_id'].nunique()}"
                )

            if "images" in val_exp and val_exp["images"] is not None:
                assert len(df_val) == val_exp["images"], (
                    f"Expected {val_exp['images']} validation images, found {len(df_val)}"
                )
            if "patients" in val_exp and val_exp["patients"] is not None:
                assert df_val["patient_id"].nunique() == val_exp["patients"], (
                    f"Expected {val_exp['patients']} validation patients, found {df_val['patient_id'].nunique()}"
                )

            if "images" in eval_exp and eval_exp["images"] is not None:
                assert len(df_eval) == eval_exp["images"], (
                    f"Expected {eval_exp['images']} evaluation images, found {len(df_eval)}"
                )
            if "patients" in eval_exp and eval_exp["patients"] is not None:
                assert df_eval["patient_id"].nunique() == eval_exp["patients"], (
                    f"Expected {eval_exp['patients']} evaluation patients, found {df_eval['patient_id'].nunique()}"
                )

        finally:
            safe_save_exclusions()

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
        labels_p = Path(
            labels_csv if labels_csv is not None else "data/raw/deepdrid/regular_fundus_quality.csv"
        )
        if not labels_p.is_file():
            raise FileNotFoundError(
                f"Official DeepDRiD label file not found at: {labels_p}.\n"
                "Please obtain the official DeepDRiD repository & annotations legally per docs/data-card.md."
            )

        images_p = Path(images_dir if images_dir is not None else "data/raw/deepdrid/images")
        if not images_p.is_dir():
            raise FileNotFoundError(
                f"Official DeepDRiD images directory not found at: {images_p}.\n"
                "Please download authorized DeepDRiD images into data/raw/deepdrid/images/ per docs/data-card.md."
            )

        try:
            print(f"Parsing official DeepDRiD metadata from {labels_p} and images in {images_p}...")
            manifest_df = parse_deepdrid_metadata(
                labels_p,
                str(images_p),
                exclusions_list=exclusions_list,
                mapping_config_path=mapping_config_path,
            )
            dfs.append(manifest_df)
        finally:
            ex_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(exclusions_list, columns=EXCLUSION_COLUMNS).to_csv(ex_p, index=False)

        source_folds_info = {
            "custom": {
                "images": len(dfs[0]),
                "patients": int(dfs[0]["patient_id"].nunique()),
                "raw_labels_path": str(labels_p),
                "raw_labels_sha256": compute_sha256(labels_p),
            }
        }

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = build_canonical_manifest(dfs, out_p)

    mapping_info = load_deepdrid_label_mapping(mapping_config_path)
    manifest_sha = compute_sha256(out_p)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "dataset": "DeepDRiD",
        "manifest_path": str(out_p),
        "manifest_sha256": manifest_sha,
        "mapping_path": mapping_info.get("mapping_path", str(mapping_config_path)),
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

    return manifest_df, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepDRiD canonical manifest from official dataset."
    )
    parser.add_argument(
        "--external-root",
        type=str,
        default="external/DeepDRiD/regular_fundus_images",
        help="Path to official multi-fold DeepDRiD dataset root",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Path to custom DeepDRiD quality label CSV/XLSX file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/deepdrid/images",
        help="Path to custom DeepDRiD images directory",
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
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="configs/deepdrid_label_mapping.yaml",
        help="Path to authoritative YAML mapping configuration",
    )
    args = parser.parse_args()

    manifest_df, metadata = prepare_deepdrid(
        external_root=args.external_root,
        output_csv=args.output_csv,
        exclusions_csv=args.exclusions_csv,
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        mapping_config_path=args.mapping_config,
    )

    print("\n=== DeepDRiD Canonical Label Mapping & Distribution Verification ===")
    print(f"Mapping Schema SHA-256: {metadata.get('mapping_sha256', 'N/A')}")
    print(f"Manifest SHA-256: {metadata.get('manifest_sha256')}")
    for col in ["overall_quality_canonical", "artifact", "clarity", "field_definition"]:
        counts = dict(manifest_df[col].value_counts(dropna=False).sort_index())
        print(f"  {col}: {counts}")

    print(
        f"\nSuccessfully constructed canonical DeepDRiD manifest ({len(manifest_df)} records, {manifest_df['patient_id'].nunique()} patients) at: {args.output_csv}"
    )
    meta_p = Path(args.output_csv).parent / f"{Path(args.output_csv).stem}.metadata.json"
    print(f"Persistent provenance metadata saved to: {meta_p}")


if __name__ == "__main__":
    main()
