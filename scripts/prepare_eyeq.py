"""Data preparation script constructing canonical EyeQ manifest per blueprint Section 8."""

import argparse
from pathlib import Path

from src.retinaguard.data.adapters import build_canonical_manifest, parse_eyeq_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EyeQ canonical manifest from official dataset."
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="data/raw/eyeq/labels/Label_EyeQ_Train.csv",
        help="Path to official EyeQ label CSV file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/eyeq/images",
        help="Path to official EyePACS/EyeQ images directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/manifests/eyeq_manifest.csv",
        help="Destination path for canonical manifest CSV",
    )
    parser.add_argument(
        "--exclusions-csv",
        type=str,
        default="data/manifests/eyeq_exclusions.csv",
        help="Destination path for excluded records CSV",
    )
    args = parser.parse_args()

    labels_p = Path(args.labels_csv)
    if not labels_p.is_file():
        raise FileNotFoundError(
            f"Official EyeQ label file not found at: {args.labels_csv}.\n"
            "Please obtain the official EyeQ labels and EyePACS images legally per docs/data-card.md."
        )

    images_p = Path(args.images_dir)
    if not images_p.is_dir():
        raise FileNotFoundError(
            f"Official EyeQ images directory not found at: {args.images_dir}.\n"
            "Please download authorized images into data/raw/eyeq/images/ per docs/data-card.md."
        )

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ex_p = Path(args.exclusions_csv)

    print(f"Parsing official EyeQ metadata from {labels_p} and images in {images_p}...")
    df = parse_eyeq_metadata(labels_p, str(images_p), exclusions_csv=ex_p)
    manifest_df = build_canonical_manifest([df], out_p)
    print(
        f"Successfully constructed canonical EyeQ manifest ({len(manifest_df)} records) at: {out_p}"
    )
    if ex_p.is_file():
        print(f"Exclusion audit report written to: {ex_p}")


if __name__ == "__main__":
    main()
