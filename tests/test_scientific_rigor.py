"""Scientific rigor, fail-fast boundary, and fixture isolation test suite."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retinaguard.data.datasets import RetinalQualityDataset
from retinaguard.evaluation.ood import compute_energy_score
from retinaguard.inference.predictor import RetinaGuardPredictor
from retinaguard.utils.hashing import compute_sha256


def test_committed_audit_report_provenance():
    """Verify that committed audit reports match currently committed scripts and configs."""
    report_p = Path("artifacts/reports/cross_split_isolation_audit.json")
    if report_p.is_file():
        with open(report_p, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["audit_script_sha256"] == compute_sha256("scripts/audit_dataset.py")
        assert report["mapping_sha256"] == compute_sha256("configs/deepdrid_label_mapping.yaml")
        assert report.get("cross_split_isolation_passed") is True
        assert report.get("intra_split_uniqueness_passed") is True
        assert report.get("image_integrity_passed") is True
        assert report.get("overall_audit_passed") is True


def test_production_data_directory_isolation():
    """Verify that production data/manifests/ and data/splits/ contain no mock or fixture entries."""
    for p in Path("data/manifests").glob("*.csv"):
        df = pd.read_csv(p)
        if "sha256" in df.columns:
            assert "mock_sha" not in str(
                df["sha256"].values
            ), f"Mock hashes found in production manifest: {p}"
        if "is_fixture" in df.columns:
            assert not df["is_fixture"].any(), f"Fixture records found in production manifest: {p}"

    for p in Path("data/splits").glob("*.csv"):
        df = pd.read_csv(p)
        if "sha256" in df.columns:
            assert "mock_sha" not in str(
                df["sha256"].values
            ), f"Mock hashes found in production split: {p}"
        if "is_fixture" in df.columns:
            assert not df["is_fixture"].any(), f"Fixture records found in production split: {p}"


def test_production_artifacts_contain_no_unsupported_metrics():
    """Verify that artifacts/metrics/ and artifacts/reports/ do not contain unsupported JSON metrics."""
    metrics_files = list(Path("artifacts/metrics").glob("*.json"))
    assert (
        len(metrics_files) == 0
    ), f"Unsupported metrics JSON files found in artifacts/metrics/: {metrics_files}"

    allowed_report_names = {"cross_split_isolation_audit.json", "data_audit.json"}
    report_files = [
        f for f in Path("artifacts/reports").glob("*.json") if f.name not in allowed_report_names
    ]
    assert (
        len(report_files) == 0
    ), f"Unsupported report JSON files found in artifacts/reports/: {report_files}"


def test_fixture_directory_metadata():
    """Verify that all fixture files in tests/fixtures/ are explicitly tagged as non-scientific."""
    fixture_files = list(Path("tests/fixtures").rglob("*.csv"))
    assert len(fixture_files) > 0, "Expected fixture files in tests/fixtures/"
    for p in fixture_files:
        df = pd.read_csv(p)
        assert "is_fixture" in df.columns, f"Missing is_fixture in {p}"
        assert df["is_fixture"].all(), f"is_fixture is not all True in {p}"
        assert (
            "eligible_for_scientific_analysis" in df.columns
        ), f"Missing eligible_for_scientific_analysis in {p}"
        assert not df[
            "eligible_for_scientific_analysis"
        ].any(), f"Fixture marked eligible for science in {p}"


def test_dataset_fail_fast_on_missing_images(tmp_path):
    """Verify RetinalQualityDataset immediately raises FileNotFoundError on missing files."""
    df = pd.DataFrame(
        [
            {
                "image_id": "nonexistent_001",
                "path": str(tmp_path / "nonexistent.jpg"),
                "quality_canonical": "good",
            }
        ]
    )
    ds = RetinalQualityDataset(df, allow_synthetic_fallback=False)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_predictor_fail_fast_without_model():
    """Verify RetinaGuardPredictor raises RuntimeError if no model is loaded."""
    predictor = RetinaGuardPredictor(model_path=None)
    with pytest.raises(RuntimeError):
        predictor.predict(np.zeros((100, 100, 3), dtype=np.uint8))


def test_energy_score_directionality():
    """Verify that confident in-distribution logits produce higher energy than flat OOD logits."""
    in_dist_logits = np.array([[6.0, 1.0, 0.0]])
    ood_flat_logits = np.array([[0.0, 0.0, 0.0]])

    energy_id = compute_energy_score(in_dist_logits)[0]
    energy_ood = compute_energy_score(ood_flat_logits)[0]

    # In-distribution confident predictions must have higher energy score
    assert energy_id > energy_ood, f"Expected energy_id ({energy_id}) > energy_ood ({energy_ood})"
