"""Unit tests for cryptographic/perceptual hashing and patient leakage audit."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.audit.hash_audit import compute_file_hash, find_duplicate_images, compute_phash_distance, compute_perceptual_hash
from src.audit.patient_leakage_check import extract_patient_id_from_filename, verify_patient_split_isolation
from src.audit.data_integrity import inspect_image_file


def test_extract_patient_id():
    assert extract_patient_id_from_filename("10_left.jpeg") == "10"
    assert extract_patient_id_from_filename("10_right.jpeg") == "10"
    assert extract_patient_id_from_filename("P042_OD.png") == "P042"
    assert extract_patient_id_from_filename("001_1_left.jpg") == "001"


def test_verify_patient_split_isolation():
    # Clean partition without leak
    clean_splits = {
        "train": ["10_left.jpg", "10_right.jpg", "11_left.jpg"],
        "val": ["12_left.jpg", "12_right.jpg"],
        "test": ["13_left.jpg"]
    }
    report = verify_patient_split_isolation(clean_splits)
    assert report["isolation_passed"] is True
    assert len(report["leaked_patient_ids"]) == 0

    # Leaked partition
    leaked_splits = {
        "train": ["10_left.jpg", "11_left.jpg"],
        "val": ["10_right.jpg", "12_left.jpg"],
        "test": ["13_left.jpg"]
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
