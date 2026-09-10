"""Unit tests for Data Pipeline (manifests, leakage check, transforms, datasets)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retinaguard.data.adapters import (
    extract_patient_and_eye,
    parse_deepdrid_metadata,
    parse_eyeq_metadata,
)
from retinaguard.data.audit import run_leakage_and_duplicate_audit
from retinaguard.data.datasets import RetinalQualityDataset
from retinaguard.data.preprocessing import crop_retinal_fov
from retinaguard.data.splits import create_patient_grouped_splits


def test_patient_and_eye_extraction():
    p1, eye1 = extract_patient_and_eye("10_left.jpeg")
    assert p1 == "10" and eye1 == "left"

    p2, eye2 = extract_patient_and_eye("042_right.jpg")
    assert p2 == "042" and eye2 == "right"


def test_patient_grouped_splitting(tmp_path):
    rows = []
    for i in range(100):
        rows.append(
            {
                "image_id": f"img_{i}",
                "patient_id": f"patient_{i // 2}",
                "path": f"data/raw/eyeq/images/img_{i}.jpg",
                "quality_canonical": ["good", "usable", "reject"][i % 3],
                "sha256": f"sha_{i}",
            }
        )
    df = pd.DataFrame(rows)

    splits = create_patient_grouped_splits(df, val_ratio=0.15, test_ratio=0.15, seed=2026)
    audit = run_leakage_and_duplicate_audit(splits, reports_dir=tmp_path / "reports")

    assert audit["isolation_passed"] is True
    assert len(audit["patient_leakages"]) == 0
    assert len(audit["sha256_leakages"]) == 0


def test_canonical_fov_crop():
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[50:250, 50:250, 0] = 200
    img = Image.fromarray(arr)
    cropped = crop_retinal_fov(img)
    w, h = cropped.size
    assert w == h


def test_dataset_loader(tmp_path):
    # 1. Create real temporary test image
    img_file = tmp_path / "valid_fundus.png"
    Image.new("RGB", (100, 100), color=(180, 80, 30)).save(img_file)

    df = pd.DataFrame(
        [
            {
                "image_id": "test_1",
                "patient_id": "P001",
                "path": str(img_file),
                "quality_canonical": "good",
                "artifact": 0,
                "clarity": 1,
                "field_definition": 0,
            }
        ]
    )
    ds = RetinalQualityDataset(df, image_size=384)
    item = ds[0]
    assert item["image"].shape == (3, 384, 384)
    assert item["quality_target"].item() == 0
    assert item["quality_mask"].item() == 1.0

    # 2. Verify FileNotFoundError when missing image is encountered without fallback
    df_missing = pd.DataFrame(
        [
            {
                "image_id": "missing_1",
                "patient_id": "P002",
                "path": str(tmp_path / "nonexistent.jpg"),
                "quality_canonical": "usable",
            }
        ]
    )
    ds_strict = RetinalQualityDataset(df_missing, allow_synthetic_fallback=False)
    with pytest.raises(FileNotFoundError):
        _ = ds_strict[0]

    # 3. Verify fallback works when explicitly requested for synthetic test fixtures
    ds_fixture = RetinalQualityDataset(df_missing, allow_synthetic_fallback=True)
    item_fix = ds_fixture[0]
    assert item_fix["image"].shape == (3, 384, 384)


def test_eyeq_adapter_and_exclusions(tmp_path):

    img_dir = tmp_path / "eyeq_images"
    img_dir.mkdir()
    valid_img = img_dir / "101_left.jpeg"
    Image.new("RGB", (100, 100), color=(150, 70, 20)).save(valid_img)

    # Label CSV with: 1 valid, 1 missing image
    csv_file = tmp_path / "eyeq_labels.csv"
    pd.DataFrame(
        [
            {"image": "101_left.jpeg", "quality": 0},
            {"image": "102_right.jpeg", "quality": 1},  # missing
        ]
    ).to_csv(csv_file, index=False)

    ex_csv = tmp_path / "eyeq_ex.csv"
    manifest_df = parse_eyeq_metadata(csv_file, img_dir, exclusions_csv=ex_csv)

    assert len(manifest_df) == 1
    assert manifest_df.iloc[0]["quality_canonical"] == "good"
    assert manifest_df.iloc[0]["overall_quality_raw"] is None

    ex_df = pd.read_csv(ex_csv)
    assert len(ex_df) == 1
    assert ex_df.iloc[0]["reason"] == "missing_image"
    assert len(manifest_df) + len(ex_df) == 2


def test_deepdrid_adapter_label_validations_and_exclusions(tmp_path):
    from retinaguard.data.adapters import parse_deepdrid_metadata

    img_dir = tmp_path / "deepdrid_images"
    img_dir.mkdir()

    # Create 3 valid test images
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        Image.new("RGB", (100, 100), color=(160, 60, 20)).save(img_dir / name)

    # 1. Test valid accepted raw values (3-level & 5-level)
    csv_valid = tmp_path / "deepdrid_valid.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1.jpg",
                "overall_quality": 1,
                "artifact": 0,
                "clarity": 10,
                "field_definition": 10,
            },
            {
                "image_id": "img2.jpg",
                "overall_quality": "1",
                "artifact": 4,
                "clarity": 6,
                "field_definition": 6,
            },
            {
                "image_id": "img3.jpg",
                "overall_quality": 0,
                "artifact": 8,
                "clarity": 1,
                "field_definition": 4,
            },
            {
                "image_id": "missing_img.jpg",
                "overall_quality": 1,
                "artifact": 1,
                "clarity": 1,
                "field_definition": 1,
            },
        ]
    ).to_csv(csv_valid, index=False)

    ex_csv = tmp_path / "deepdrid_ex.csv"
    df_manifest = parse_deepdrid_metadata(csv_valid, img_dir, exclusions_csv=ex_csv)

    assert len(df_manifest) == 3
    assert df_manifest.iloc[0]["overall_quality_canonical"] == "good"
    assert df_manifest.iloc[1]["overall_quality_canonical"] == "good"
    assert df_manifest.iloc[2]["overall_quality_canonical"] == "reject"
    # Ensure quality_raw and quality_canonical are None for DeepDRiD
    assert df_manifest.iloc[0]["quality_canonical"] is None

    ex_df = pd.read_csv(ex_csv)
    assert len(ex_df) == 1
    assert len(df_manifest) + len(ex_df) == 4

    # 2. Test unknown/invalid raw label causes ValueError
    csv_invalid = tmp_path / "deepdrid_invalid.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1.jpg",
                "overall_quality": "UnknownGrade999",
                "artifact": 0,
                "clarity": 0,
                "field_definition": 0,
            }
        ]
    ).to_csv(csv_invalid, index=False)

    with pytest.raises(ValueError, match="Unrecognized DeepDRiD overall quality"):
        parse_deepdrid_metadata(csv_invalid, img_dir)


def test_deepdrid_adapter_excel_xlsx_ingestion(tmp_path):
    """Verify parsing Excel .xlsx annotation files using openpyxl."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "test_eval_1.jpg"
    Image.new("RGB", (100, 100), color=(180, 80, 30)).save(img_file)

    xlsx_file = tmp_path / "Challenge2_labels.xlsx"
    df = pd.DataFrame(
        [
            {
                "image_id": "test_eval_1",
                "Overall quality": 1,
                "Artifact": 0,
                "Clarity": 10,
                "Field definition": 10,
            }
        ]
    )
    df.to_excel(xlsx_file, index=False)

    parsed = parse_deepdrid_metadata(xlsx_file, img_dir, source_split="external_test")
    assert len(parsed) == 1
    assert parsed.iloc[0]["image_id"] == "test_eval_1"
    assert parsed.iloc[0]["overall_quality_canonical"] == "good"
    assert parsed.iloc[0]["artifact"] == 0
    assert parsed.iloc[0]["clarity"] == 0
    assert parsed.iloc[0]["field_definition"] == 0
    assert parsed.iloc[0]["source_split"] == "external_test"


def test_prepare_deepdrid_provenance_metadata_json(tmp_path):
    """Verify that deepdrid_manifest.metadata.json is produced with correct provenance fields and exact SHA-256 hashes."""
    import json

    from retinaguard.utils.hashing import compute_sha256
    from scripts.prepare_deepdrid import prepare_deepdrid

    # Create dummy multi-fold directory structure with Images subdirectories
    ext_dir = tmp_path / "external" / "DeepDRiD" / "regular_fundus_images"
    train_dir = ext_dir / "regular-fundus-training"
    val_dir = ext_dir / "regular-fundus-validation"
    eval_dir = ext_dir / "Online-Challenge1&2-Evaluation"
    for d in [train_dir, val_dir, eval_dir]:
        (d / "Images").mkdir(parents=True)

    # Images and labels
    train_csv = train_dir / "regular-fundus-training.csv"
    val_csv = val_dir / "regular-fundus-validation.csv"
    eval_xlsx = eval_dir / "Challenge2_labels.xlsx"

    for prefix, d, count, label_file, base_id in [
        ("train", train_dir, 3, train_csv, 100),
        ("val", val_dir, 2, val_csv, 200),
        ("eval", eval_dir, 2, eval_xlsx, 300),
    ]:
        rows = []
        for i in range(count):
            img_name = f"{base_id + i + 1}_1.jpg"
            Image.new("RGB", (50, 50), color=(100, 50, 20)).save(d / "Images" / img_name)
            rows.append(
                {
                    "image_id": img_name,
                    "overall_quality": 1,
                    "artifact": 0,
                    "clarity": 10,
                    "field_definition": 10,
                }
            )
        if prefix == "eval":
            pd.DataFrame(rows).to_excel(label_file, index=False)
        else:
            pd.DataFrame(rows).to_csv(label_file, index=False)

    out_csv = tmp_path / "manifests" / "deepdrid_manifest.csv"
    ex_csv = tmp_path / "manifests" / "deepdrid_exclusions.csv"
    meta_json = tmp_path / "manifests" / "deepdrid_manifest.metadata.json"

    expected_counts = {
        "train": {"images": 3, "patients": 3},
        "val": {"images": 2, "patients": 2},
        "external_test": {"images": 2, "patients": 2},
    }

    manifest_df, metadata = prepare_deepdrid(
        external_root=ext_dir,
        output_csv=out_csv,
        exclusions_csv=ex_csv,
        expected_fold_counts=expected_counts,
    )

    assert out_csv.is_file()
    assert meta_json.is_file()
    assert len(manifest_df) == 7

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Validate structure and exact hash parity
    assert meta["dataset"] == "DeepDRiD"
    assert meta["manifest_sha256"] == compute_sha256(out_csv)
    assert meta["mapping_path"] == "configs/deepdrid_label_mapping.yaml"
    assert meta["mapping_sha256"] == compute_sha256("configs/deepdrid_label_mapping.yaml")

    folds = meta["source_folds"]
    assert folds["train"]["images"] == 3
    assert folds["train"]["patients"] == 3
    assert folds["train"]["raw_labels_sha256"] == compute_sha256(train_csv)

    assert folds["val"]["images"] == 2
    assert folds["val"]["patients"] == 2
    assert folds["val"]["raw_labels_sha256"] == compute_sha256(val_csv)

    assert folds["external_test"]["images"] == 2
    assert folds["external_test"]["patients"] == 2
    assert folds["external_test"]["raw_labels_sha256"] == compute_sha256(eval_xlsx)

    assert meta["total_records"] == 7
    assert meta["total_patients"] == 7
    assert "generated_at_utc" in meta
    assert "git_commit" in meta
