"""Unit tests for Data Pipeline (manifests, leakage check, transforms, datasets)."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import pytest

from src.retinaguard.data.adapters import extract_patient_and_eye
from src.retinaguard.data.splits import create_patient_grouped_splits
from src.retinaguard.data.audit import run_leakage_and_duplicate_audit
from src.retinaguard.data.preprocessing import crop_retinal_fov, preprocess_image_canonical
from src.retinaguard.data.datasets import RetinalQualityDataset


def test_patient_and_eye_extraction():
    p1, eye1 = extract_patient_and_eye("10_left.jpeg")
    assert p1 == "10" and eye1 == "left"

    p2, eye2 = extract_patient_and_eye("042_right.jpg")
    assert p2 == "042" and eye2 == "right"


def test_patient_grouped_splitting():
    rows = []
    for i in range(100):
        rows.append({
            "image_id": f"img_{i}",
            "patient_id": f"patient_{i // 2}",
            "path": f"data/raw/eyeq/images/img_{i}.jpg",
            "quality_canonical": ["good", "usable", "reject"][i % 3],
            "sha256": f"sha_{i}"
        })
    df = pd.DataFrame(rows)

    splits = create_patient_grouped_splits(df, val_ratio=0.15, test_ratio=0.15, seed=2026)
    audit = run_leakage_and_duplicate_audit(splits)

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

    df = pd.DataFrame([{
        "image_id": "test_1",
        "patient_id": "P001",
        "path": str(img_file),
        "quality_canonical": "good",
        "artifact": 0,
        "clarity": 1,
        "field_definition": 0
    }])
    ds = RetinalQualityDataset(df, image_size=384)
    item = ds[0]
    assert item["image"].shape == (3, 384, 384)
    assert item["quality_target"].item() == 0
    assert item["quality_mask"].item() == 1.0

    # 2. Verify FileNotFoundError when missing image is encountered without fallback
    df_missing = pd.DataFrame([{
        "image_id": "missing_1",
        "patient_id": "P002",
        "path": str(tmp_path / "nonexistent.jpg"),
        "quality_canonical": "usable"
    }])
    ds_strict = RetinalQualityDataset(df_missing, allow_synthetic_fallback=False)
    with pytest.raises(FileNotFoundError):
        _ = ds_strict[0]

    # 3. Verify fallback works when explicitly requested for synthetic test fixtures
    ds_fixture = RetinalQualityDataset(df_missing, allow_synthetic_fallback=True)
    item_fix = ds_fixture[0]
    assert item_fix["image"].shape == (3, 384, 384)
