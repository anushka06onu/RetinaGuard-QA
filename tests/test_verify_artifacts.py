"""Unit tests for artifact checksum and lineage verification."""

import json

from scripts.verify_artifacts import (
    compute_sha256,
    generate_checksum_manifest,
    verify_checksum_manifest,
)


def test_verify_artifacts_synthetic(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "models").mkdir()
    (artifacts_dir / "reports").mkdir()
    (artifacts_dir / "metrics").mkdir()
    (artifacts_dir / "provenance").mkdir()

    # Create dummy files
    f1 = artifacts_dir / "models/model.onnx"
    f1.write_text("dummy onnx content")
    f1_data = artifacts_dir / "models/model.onnx.data"
    f1_data.write_text("dummy onnx data content")
    f2 = artifacts_dir / "models/preprocessing.json"
    f2.write_text(json.dumps({"test": 123}))
    f3 = artifacts_dir / "models/calibration_metadata.json"
    f3.write_text(json.dumps({"onnx_model_sha256": compute_sha256(f1)}))
    f4 = artifacts_dir / "models/onnx_manifest.json"
    f4.write_text(json.dumps({"onnx_file_sha256": compute_sha256(f1)}))
    f5 = artifacts_dir / "metrics/onnx_parity.json"
    f5.write_text(json.dumps({"parity_passed": True}))
    f6 = artifacts_dir / "reports/data_audit.json"
    f6.write_text(json.dumps({"audit": True}))
    f7 = artifacts_dir / "reports/cross_split_isolation_audit.json"
    f7.write_text(json.dumps({"cross_split": True}))
    f8 = artifacts_dir / "reports/data_flow_report.json"
    f8.write_text(json.dumps({"flow": True}))
    f9 = artifacts_dir / "provenance/environment.txt"
    f9.write_text("torch==2.0.0\n")

    manifest_p = artifacts_dir / "provenance/SHA256SUMS"
    count = generate_checksum_manifest(artifacts_dir, manifest_p)
    assert count >= 9

    report = verify_checksum_manifest(manifest_p, artifacts_dir, tmp_path)
    assert report["passed"] is True
    assert report["mismatched_files"] == 0
    assert report["missing_files"] == 0
