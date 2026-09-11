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
            assert "mock_sha" not in str(df["sha256"].values), (
                f"Mock hashes found in production manifest: {p}"
            )
        if "is_fixture" in df.columns:
            assert not df["is_fixture"].any(), f"Fixture records found in production manifest: {p}"

    for p in Path("data/splits").glob("*.csv"):
        df = pd.read_csv(p)
        if "sha256" in df.columns:
            assert "mock_sha" not in str(df["sha256"].values), (
                f"Mock hashes found in production split: {p}"
            )
        if "is_fixture" in df.columns:
            assert not df["is_fixture"].any(), f"Fixture records found in production split: {p}"


def test_production_artifacts_contain_no_unsupported_metrics():
    """Verify that artifacts/metrics/ and artifacts/reports/ only contain approved pipeline output files."""
    allowed_metric_names = {
        "train_history.json",
        "held_out_deepdrid.json",
        "internal_eyeq.json",
        "eyeq_test.json",
        "deepdrid_heldout.json",
        "zero_shot_transfer.json",
        "calibration.json",
        "selective_prediction.json",
        "corruptions.json",
        "ood.json",
        "latency.json",
        "onnx_parity.json",
    }
    metrics_files = [
        f for f in Path("artifacts/metrics").glob("*.json") if f.name not in allowed_metric_names
    ]
    assert len(metrics_files) == 0, (
        f"Unsupported metrics JSON files found in artifacts/metrics/: {metrics_files}"
    )

    allowed_report_names = {"cross_split_isolation_audit.json", "data_audit.json"}
    report_files = [
        f for f in Path("artifacts/reports").glob("*.json") if f.name not in allowed_report_names
    ]
    assert len(report_files) == 0, (
        f"Unsupported report JSON files found in artifacts/reports/: {report_files}"
    )


def test_fixture_directory_metadata():
    """Verify that all fixture files in tests/fixtures/ are explicitly tagged as non-scientific."""
    fixture_files = list(Path("tests/fixtures").rglob("*.csv"))
    assert len(fixture_files) > 0, "Expected fixture files in tests/fixtures/"
    for p in fixture_files:
        df = pd.read_csv(p)
        assert "is_fixture" in df.columns, f"Missing is_fixture in {p}"
        assert df["is_fixture"].all(), f"is_fixture is not all True in {p}"
        assert "eligible_for_scientific_analysis" in df.columns, (
            f"Missing eligible_for_scientific_analysis in {p}"
        )
        assert not df["eligible_for_scientific_analysis"].any(), (
            f"Fixture marked eligible for science in {p}"
        )


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


def test_empirical_result_artifact_schema_and_integrity(tmp_path):
    """Verify that all metric artifacts conform to rigorous evidence-status schema."""
    from retinaguard.evaluation.provenance import validate_empirical_result

    # 1. Verify all committed metrics files in artifacts/metrics/
    metrics_files = list(Path("artifacts/metrics").glob("*.json"))
    assert len(metrics_files) > 0, "Expected metrics files in artifacts/metrics/"
    for mf in metrics_files:
        res = validate_empirical_result(mf)
        assert res["valid"] is True

    # 2. Test rejection of unprovenanced mock result JSON
    mock_bad_json = tmp_path / "mock_bad.json"
    mock_bad_json.write_text(
        json.dumps({"macro_f1": 0.95, "dataset": "Fabricated", "status": "completed"})
    )
    with pytest.raises(ValueError, match="missing required provenance keys"):
        validate_empirical_result(mock_bad_json)

    # 3. Test rejection of completed result marked ineligible for science
    mock_ineligible = tmp_path / "ineligible.json"
    mock_ineligible.write_text(
        json.dumps(
            {
                "status": "completed",
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_sha256": "fake_ckpt_sha",
                "split_sha256": "fake_split_sha",
                "num_samples": 100,
                "created_at_utc": "2026-09-11T12:00:00Z",
                "eligible_as_final_result": False,
            }
        )
    )
    with pytest.raises(
        ValueError, match="Completed result cannot have eligible_as_final_result = False"
    ):
        validate_empirical_result(mock_ineligible)

    # 4. Test rejection of preliminary artifact missing head_type
    mock_bad_prelim = tmp_path / "bad_prelim.json"
    mock_bad_prelim.write_text(
        json.dumps(
            {"status": "preliminary_obsolete_architecture", "eligible_as_final_result": False}
        )
    )
    with pytest.raises(ValueError, match="Preliminary obsolete artifact must declare head_type"):
        validate_empirical_result(mock_bad_prelim)

    # 5. Test acceptance of fully provenanced completed result
    dummy_split = tmp_path / "split.csv"
    dummy_split.write_text("image_id,path\n1,img.png\n")
    split_sha = compute_sha256(dummy_split)

    valid_completed = tmp_path / "valid_completed.json"
    valid_completed.write_text(
        json.dumps(
            {
                "status": "completed",
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_sha256": "ckpt_hash_123",
                "split_path": str(dummy_split),
                "split_sha256": split_sha,
                "num_samples": 1,
                "macro_f1": 0.85,
                "created_at_utc": "2026-09-11T12:00:00Z",
            }
        )
    )
    valid_res = validate_empirical_result(valid_completed)
    assert valid_res["valid"] is True
    assert valid_res["type"] == "completed_final_result"
