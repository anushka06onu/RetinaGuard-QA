"""Data preparation script constructing canonical DeepDRiD manifest per blueprint Section 8."""

import argparse
from pathlib import Path

from src.retinaguard.data.adapters import (
    build_canonical_manifest,
    parse_deepdrid_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepDRiD canonical manifest from official dataset.")
    parser.add_argument(
        "--labels-csv", type=str, default="data/raw/deepdrid/regular_fundus_quality.csv",
        help="Path to official DeepDRiD quality label CSV file"
    )
    parser.add_argument(
        "--images-dir", type=str, default="data/raw/deepdrid/images",
        help="Path to official DeepDRiD images directory"
    )
    parser.add_argument(
        "--output-csv", type=str, default="data/manifests/deepdrid_manifest.csv",
        help="Destination path for canonical manifest CSV"
    )
    args = parser.parse_args()

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

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing official DeepDRiD metadata from {labels_p} and images in {images_p}...")
    df = parse_deepdrid_metadata(labels_p, str(images_p))
    manifest_df = build_canonical_manifest([df], out_p)
    print(f"Successfully constructed canonical DeepDRiD manifest ({len(manifest_df)} records) at: {out_p}")


if __name__ == "__main__":
    main()
