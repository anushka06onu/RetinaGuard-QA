"""Data preparation script constructing canonical DeepDRiD manifest per blueprint Section 8."""

import argparse
from pathlib import Path

from src.retinaguard.data.adapters import (
    build_canonical_manifest,
    parse_deepdrid_metadata,
)


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

        if train_csv.is_file() and train_img.is_dir():
            print(f"Parsing DeepDRiD training fold from {train_csv}...")
            dfs.append(
                parse_deepdrid_metadata(
                    train_csv, train_img, source_split="train", exclusions_csv=ex_p
                )
            )
        if val_csv.is_file() and val_img.is_dir():
            print(f"Parsing DeepDRiD validation fold from {val_csv}...")
            dfs.append(
                parse_deepdrid_metadata(val_csv, val_img, source_split="val", exclusions_csv=ex_p)
            )
        if eval_csv.is_file() and eval_img.is_dir():
            print(f"Parsing DeepDRiD external evaluation fold from {eval_csv}...")
            dfs.append(
                parse_deepdrid_metadata(
                    eval_csv, eval_img, source_split="external_test", exclusions_csv=ex_p
                )
            )
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

        print(f"Parsing official DeepDRiD metadata from {labels_p} and images in {images_p}...")
        dfs.append(parse_deepdrid_metadata(labels_p, str(images_p), exclusions_csv=ex_p))

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = build_canonical_manifest(dfs, out_p)
    print(
        f"Successfully constructed canonical DeepDRiD manifest ({len(manifest_df)} records) at: {out_p}"
    )
    if ex_p.is_file():
        print(f"Exclusion audit report written to: {ex_p}")


if __name__ == "__main__":
    main()
