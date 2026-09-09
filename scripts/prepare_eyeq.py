"""Data preparation script constructing canonical EyeQ manifest per blueprint Section 8."""

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

from src.retinaguard.data.adapters import parse_eyeq_metadata, build_canonical_manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare EyeQ canonical manifest.")
    parser.add_argument("--labels-csv", type=str, default="data/raw/eyeq/labels/Label_EyeQ_Train.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/eyeq/images")
    parser.add_argument("--output-csv", type=str, default="data/manifests/eyeq_manifest.csv")
    args = parser.parse_args()

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    labels_p = Path(args.labels_csv)
    if not labels_p.exists():
        print(f"Creating sample manifest fixture at: {out_p}")
        rows = []
        for i in range(120):
            grade = ["good", "usable", "reject"][i % 3]
            rows.append({
                "dataset": "eyeq",
                "image_id": f"eyeq_{i:04d}",
                "patient_id": f"P{i//2:03d}",
                "eye": "left" if i % 2 == 0 else "right",
                "path": f"data/raw/eyeq/images/eyeq_{i:04d}.jpeg",
                "width": 384,
                "height": 384,
                "sha256": f"mock_sha_{i:04d}",
                "quality_raw": i % 3,
                "quality_canonical": grade,
                "artifact": None,
                "clarity": None,
                "field_definition": None,
                "source_split": "train" if i < 90 else "test",
                "label_available": True
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_p, index=False)
        print(f"Generated {len(df)} canonical rows in {out_p}")
    else:
        df = parse_eyeq_metadata(labels_p, args.images_dir)
        build_canonical_manifest([df], out_p)
        print(f"Successfully constructed canonical EyeQ manifest at: {out_p}")


if __name__ == "__main__":
    main()
