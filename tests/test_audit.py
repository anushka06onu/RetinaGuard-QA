"""Unit tests for cryptographic/perceptual hashing and patient leakage audit."""

import numpy as np
from PIL import Image

from retinaguard.data.adapters import extract_patient_and_eye
from retinaguard.data.audit import (
    find_duplicate_images,
    inspect_image_file,
    verify_patient_split_isolation,
)


def test_extract_patient_id():
    p1, _ = extract_patient_and_eye("10_left.jpeg")
    assert p1 == "10"
    p2, _ = extract_patient_and_eye("10_right.jpeg")
    assert p2 == "10"
    p3, _ = extract_patient_and_eye("P042_OD.png")
    assert p3 == "P042"
    p4, _ = extract_patient_and_eye("001_1_left.jpg")
    assert p4 == "001"


def test_verify_patient_split_isolation():
    # Clean partition without leak
    clean_splits = {
        "train": ["10_left.jpg", "10_right.jpg", "11_left.jpg"],
        "val": ["12_left.jpg", "12_right.jpg"],
        "test": ["13_left.jpg"],
    }
    report = verify_patient_split_isolation(clean_splits)
    assert report["isolation_passed"] is True
    assert len(report["leaked_patient_ids"]) == 0

    # Leaked partition
    leaked_splits = {
        "train": ["10_left.jpg", "11_left.jpg"],
        "val": ["10_right.jpg", "12_left.jpg"],
        "test": ["13_left.jpg"],
    }
    report_leaked = verify_patient_split_isolation(leaked_splits)
    assert report_leaked["isolation_passed"] is False
    assert "train_vs_val" in report_leaked["leaked_patient_ids"]
    assert "10" in report_leaked["leaked_patient_ids"]["train_vs_val"]


def test_duplicate_and_phash(tmp_path):
    img1 = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
    img2 = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
    img3 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    p3 = tmp_path / "img3.png"

    img1.save(p1)
    img2.save(p2)
    img3.save(p3)

    dup_report = find_duplicate_images([p1, p2, p3], check_perceptual=True)
    assert len(dup_report["exact_duplicates"]) == 1
    assert dup_report["total_scanned"] == 3

    info = inspect_image_file(p1)
    assert info["is_valid"] is True
    assert info["width"] == 100
    assert info["height"] == 100
