"""Helper utility for generating isolated test fixtures for unit and pipeline tests."""

from pathlib import Path

import pandas as pd


def generate_test_fixtures(output_dir: Path = Path("tests/fixtures")):
    manifests_dir = output_dir / "manifests"
    splits_dir = output_dir / "splits"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic EyeQ Fixture Manifest
    eyeq_rows = []
    for i in range(120):
        grade = ["good", "usable", "reject"][i % 3]
        eyeq_rows.append(
            {
                "dataset": "eyeq",
                "image_id": f"fixture_eyeq_{i:04d}",
                "patient_id": f"P{i // 2:03d}",
                "eye": "left" if i % 2 == 0 else "right",
                "path": f"tests/fixtures/images/eyeq_{i:04d}.jpg",
                "width": 384,
                "height": 384,
                "sha256": f"fixture_sha_{i:04d}",
                "quality_raw": i % 3,
                "quality_canonical": grade,
                "artifact": None,
                "clarity": None,
                "field_definition": None,
                "source_split": "train" if i < 90 else "test",
                "label_available": True,
                "is_fixture": True,
                "data_origin": "constructed_fixture",
                "eligible_for_scientific_analysis": False,
            }
        )
    df_eyeq = pd.DataFrame(eyeq_rows)
    df_eyeq.to_csv(manifests_dir / "synthetic_eyeq_fixture.csv", index=False)

    # 2. Synthetic DeepDRiD Fixture Manifest
    deepdrid_rows = []
    for i in range(80):
        grade = ["good", "usable", "reject"][i % 3]
        deepdrid_rows.append(
            {
                "dataset": "deepdrid",
                "image_id": f"fixture_deepdrid_{i:04d}",
                "patient_id": f"D{i // 2:03d}",
                "eye": "left" if i % 2 == 0 else "right",
                "path": f"tests/fixtures/images/deepdrid_{i:04d}.jpg",
                "width": 384,
                "height": 384,
                "sha256": f"fixture_deepdrid_sha_{i:04d}",
                "quality_raw": grade,
                "quality_canonical": grade,
                "artifact": i % 3,
                "clarity": i % 3,
                "field_definition": i % 3,
                "source_split": "train" if i < 50 else "external_test",
                "label_available": True,
                "is_fixture": True,
                "data_origin": "constructed_fixture",
                "eligible_for_scientific_analysis": False,
            }
        )
    df_deepdrid = pd.DataFrame(deepdrid_rows)
    df_deepdrid.to_csv(manifests_dir / "synthetic_deepdrid_fixture.csv", index=False)

    # 3. Splits
    df_eyeq.iloc[:70].to_csv(splits_dir / "eyeq_train.csv", index=False)
    df_eyeq.iloc[70:90].to_csv(splits_dir / "eyeq_val.csv", index=False)
    df_eyeq.iloc[90:].to_csv(splits_dir / "eyeq_test.csv", index=False)

    df_deepdrid.iloc[:40].to_csv(splits_dir / "deepdrid_train.csv", index=False)
    df_deepdrid.iloc[40:55].to_csv(splits_dir / "deepdrid_val.csv", index=False)
    df_deepdrid.iloc[55:].to_csv(splits_dir / "deepdrid_test.csv", index=False)


if __name__ == "__main__":
    generate_test_fixtures()
