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
                "checkpoint_path": "fake.ckpt",
                "checkpoint_sha256": "fake_ckpt_sha",
                "split_path": "fake.csv",
                "split_sha256": "fake_split_sha",
                "prediction_file": "fake_preds.csv",
                "prediction_file_sha256": "fake_pred_sha",
                "num_samples": 100,
                "accuracy": 0.9,
                "macro_f1": 0.9,
                "balanced_accuracy": 0.9,
                "confusion_matrix": [[50, 0], [0, 50]],
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

    dummy_pred = tmp_path / "predictions.csv"
    dummy_pred.write_text(
        "image_id,patient_id,dataset,split,target,prediction\n1,p1,test,split.csv,0,0\n"
    )
    pred_sha = compute_sha256(dummy_pred)

    dummy_ckpt = tmp_path / "model.ckpt"
    dummy_ckpt.write_text("dummy_ckpt_bytes")
    ckpt_sha = compute_sha256(dummy_ckpt)

    valid_completed = tmp_path / "valid_completed.json"
    valid_completed.write_text(
        json.dumps(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_path": str(dummy_ckpt),
                "checkpoint_sha256": ckpt_sha,
                "split_path": str(dummy_split),
                "split_sha256": split_sha,
                "prediction_file": str(dummy_pred),
                "prediction_file_sha256": pred_sha,
                "num_samples": 1,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "confusion_matrix": [[1]],
                "created_at_utc": "2026-09-11T12:00:00Z",
            }
        )
    )
    valid_res = validate_empirical_result(valid_completed)
    assert valid_res["valid"] is True
    assert valid_res["type"] == "completed_final_result"

    # 6. Test missing checkpoint file raises FileNotFoundError
    bad_ckpt_json = tmp_path / "bad_ckpt.json"
    bad_ckpt_json.write_text(
        json.dumps(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_path": str(tmp_path / "nonexistent.ckpt"),
                "checkpoint_sha256": "fake_sha",
                "split_path": str(dummy_split),
                "split_sha256": split_sha,
                "prediction_file": str(dummy_pred),
                "prediction_file_sha256": pred_sha,
                "num_samples": 1,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "confusion_matrix": [[1]],
                "created_at_utc": "2026-09-11T12:00:00Z",
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="Referenced checkpoint file not found"):
        validate_empirical_result(bad_ckpt_json)

    # 7. Test prediction IDs mismatch with split IDs raises ValueError
    mismatch_pred = tmp_path / "mismatch_preds.csv"
    mismatch_pred.write_text(
        "image_id,patient_id,dataset,split,target,prediction\n999,p1,test,split.csv,0,0\n"
    )
    mismatch_pred_sha = compute_sha256(mismatch_pred)
    mismatch_json = tmp_path / "mismatch.json"
    mismatch_json.write_text(
        json.dumps(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_path": str(dummy_ckpt),
                "checkpoint_sha256": ckpt_sha,
                "split_path": str(dummy_split),
                "split_sha256": split_sha,
                "prediction_file": str(mismatch_pred),
                "prediction_file_sha256": mismatch_pred_sha,
                "num_samples": 1,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "confusion_matrix": [[1]],
                "created_at_utc": "2026-09-11T12:00:00Z",
            }
        )
    )
    with pytest.raises(ValueError, match="Prediction image IDs do not exactly equal"):
        validate_empirical_result(mismatch_json)

    # 8. Test duplicate image_id in predictions raises ValueError
    dup_pred = tmp_path / "dup_preds.csv"
    dup_pred.write_text(
        "image_id,patient_id,dataset,split,target,prediction\n1,p1,test,split.csv,0,0\n1,p1,test,split.csv,0,0\n"
    )
    dup_pred_sha = compute_sha256(dup_pred)
    dup_json = tmp_path / "dup.json"
    dup_json.write_text(
        json.dumps(
            {
                "status": "completed",
                "eligible_as_final_result": True,
                "generated_by": "scripts/evaluate.py",
                "git_commit": "abc1234",
                "checkpoint_path": str(dummy_ckpt),
                "checkpoint_sha256": ckpt_sha,
                "split_path": str(dummy_split),
                "split_sha256": split_sha,
                "prediction_file": str(dup_pred),
                "prediction_file_sha256": dup_pred_sha,
                "num_samples": 2,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "confusion_matrix": [[2]],
                "created_at_utc": "2026-09-11T12:00:00Z",
            }
        )
    )
    with pytest.raises(ValueError, match="Duplicate image_id entries found in prediction file"):
        validate_empirical_result(dup_json)


def test_baseline_provenance_validation(tmp_path):
    """Test validation of baseline results schema without requiring neural checkpoint."""
    from retinaguard.evaluation.provenance import validate_baseline_result

    train_split = tmp_path / "train.csv"
    train_split.write_text("image_id,path\n1,img1.png\n2,img2.png\n")
    train_sha = compute_sha256(train_split)

    test_split = tmp_path / "test.csv"
    test_split.write_text("image_id,path\n3,img3.png\n4,img4.png\n")
    test_sha = compute_sha256(test_split)

    pred_file = tmp_path / "baseline_preds.csv"
    pred_file.write_text(
        "image_id,patient_id,dataset,split,target,prediction,confidence\n"
        "3,p3,EyeQ,test.csv,0,0,0.85\n"
        "4,p4,EyeQ,test.csv,1,1,0.85\n"
    )
    pred_sha = compute_sha256(pred_file)

    valid_baseline = {
        "status": "completed",
        "eligible_as_final_result": True,
        "generated_by": "scripts/train_baselines.py",
        "git_commit": "abc1234",
        "model": "Majority Class Baseline",
        "task": "eyeq_quality",
        "train_split_path": str(train_split),
        "train_split_sha256": train_sha,
        "test_split_path": str(test_split),
        "test_split_sha256": test_sha,
        "prediction_file": str(pred_file),
        "prediction_file_sha256": pred_sha,
        "num_train": 2,
        "num_test": 2,
        "created_at_utc": "2026-09-11T12:00:00Z",
        "subset_experiment": False,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "confusion_matrix": [[1, 0], [0, 1]],
    }

    res = validate_baseline_result(valid_baseline)
    assert res["valid"] is True
    assert res["type"] == "completed_baseline_result"


def test_verify_existing_seed_run_bidirectional(tmp_path):
    """Test strict bidirectional checksum and integrity verification for campaign seeds."""
    from scripts.run_campaign import generate_seed_checksums, verify_existing_seed_run

    seed_dir = tmp_path / "seed_2026"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missing manifest
    valid, _, reason = verify_existing_seed_run(seed_dir, "multitask", "cfg_hash")
    assert valid is False
    assert "Missing run_manifest.json" in reason

    # Setup valid structure
    metrics_dir = seed_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = seed_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    split_file = tmp_path / "test_split.csv"
    split_file.write_text("image_id,path\ns1,img1.png\n")
    split_sha = compute_sha256(split_file)

    pred_file = preds_dir / "eyeq_test_predictions.csv"
    pred_file.write_text(
        "image_id,patient_id,dataset,split,target,prediction\ns1,p1,eyeq,test,0,0\n"
    )
    pred_sha = compute_sha256(pred_file)

    ckpt_file = seed_dir / "best.ckpt"
    ckpt_file.write_text("ckpt_data")
    ckpt_sha = compute_sha256(ckpt_file)

    metric_file = metrics_dir / "eyeq_test.json"
    metric_payload = {
        "status": "completed",
        "eligible_as_final_result": True,
        "generated_by": "scripts/run_campaign.py",
        "git_commit": "abc1234",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha,
        "split_path": str(split_file),
        "split_sha256": split_sha,
        "prediction_file": str(pred_file),
        "prediction_file_sha256": pred_sha,
        "num_samples": 1,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "confusion_matrix": [[1]],
        "created_at_utc": "2026-09-11T12:00:00Z",
    }
    metric_file.write_text(json.dumps(metric_payload, indent=2))

    manifest_file = seed_dir / "run_manifest.json"
    manifest_payload = {
        "status": "completed",
        "eligible_for_aggregation": True,
        "campaign_mode": "multitask",
        "seed": 2026,
        "config_sha256": "cfg_hash_123",
        "git_commit": "abc1234",
        "summary_metrics": {"macro_f1": 1.0},
    }
    manifest_file.write_text(json.dumps(manifest_payload, indent=2))

    # Generate proper checksums
    generate_seed_checksums(seed_dir)

    # Valid run passes (with allow_cross_commit=True since git commit is mocked)
    valid, man, reason = verify_existing_seed_run(
        seed_dir, "multitask", "cfg_hash_123", allow_cross_commit=True
    )
    assert valid is True
    assert reason == "Verified"

    # Bidirectional test: add unlisted file on disk
    unlisted_file = seed_dir / "sneaky_file.txt"
    unlisted_file.write_text("untracked")
    valid, _, reason = verify_existing_seed_run(
        seed_dir, "multitask", "cfg_hash_123", allow_cross_commit=True
    )
    assert valid is False
    assert "Unlisted file on disk not recorded in SHA256SUMS" in reason
    unlisted_file.unlink()

    # Tampered checksum test - single token line
    sums_file = seed_dir / "SHA256SUMS"
    sums_file.write_text("singletokenline\n")
    valid, _, reason = verify_existing_seed_run(
        seed_dir, "multitask", "cfg_hash_123", allow_cross_commit=True
    )
    assert valid is False
    assert "Malformed line" in reason

    # Invalid SHA256 length / non-hex test
    sums_file.write_text("not_a_valid_64_char_hex_hash  best.ckpt\n")
    valid, _, reason = verify_existing_seed_run(
        seed_dir, "multitask", "cfg_hash_123", allow_cross_commit=True
    )
    assert valid is False
    assert "Invalid SHA-256 hash" in reason


def test_verify_campaign_archive_portable_independence(tmp_path):
    """Test that a campaign archive can be verified completely independently in a new location."""
    import shutil

    from scripts.run_campaign import generate_seed_checksums, verify_campaign_archive

    archive_dir = tmp_path / "multitask_campaign"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create splits dir
    splits_dir = archive_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_file = splits_dir / "eyeq_test.csv"
    split_file.write_text("image_id,path\nimg1,test.png\n")
    split_sha = compute_sha256(split_file)

    # 2. Create seed dir
    seed_dir = archive_dir / "seed_2026"
    seed_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = seed_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = seed_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = seed_dir / "best.ckpt"
    ckpt_file.write_text("ckpt_binary_data")
    ckpt_sha = compute_sha256(ckpt_file)

    pred_file = preds_dir / "eyeq_test_predictions.csv"
    pred_file.write_text(
        "image_id,patient_id,dataset,split,target,prediction\nimg1,p1,eyeq,test,0,0\n"
    )
    pred_sha = compute_sha256(pred_file)

    # Relative paths inside metric JSON
    metric_file = metrics_dir / "eyeq_test.json"
    metric_payload = {
        "status": "completed",
        "eligible_as_final_result": True,
        "generated_by": "scripts/run_campaign.py",
        "git_commit": "abc1234",
        "checkpoint_path": "../best.ckpt",
        "checkpoint_sha256": ckpt_sha,
        "split_path": "../../splits/eyeq_test.csv",
        "split_sha256": split_sha,
        "prediction_file": "../predictions/eyeq_test_predictions.csv",
        "prediction_file_sha256": pred_sha,
        "num_samples": 1,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "confusion_matrix": [[1]],
        "created_at_utc": "2026-09-11T12:00:00Z",
    }
    metric_file.write_text(json.dumps(metric_payload, indent=2))

    # Seed run manifest
    seed_manifest = seed_dir / "run_manifest.json"
    seed_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "campaign_mode": "multitask",
                "seed": 2026,
                "config_path": "../config.yaml",
                "git_commit": "abc1234",
            },
            indent=2,
        )
    )

    generate_seed_checksums(seed_dir)

    # Manifest and top-level sums
    manifest_file = archive_dir / "campaign_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "status": "completed",
                "campaign_mode": "multitask",
                "seeds": [2026],
                "archived_seed_directories": ["seed_2026"],
            },
            indent=2,
        )
    )
    generate_seed_checksums(archive_dir)

    # Verify original archive
    is_valid, msg = verify_campaign_archive(archive_dir)
    assert is_valid is True
    assert msg == "Archive verified successfully"

    # Move archive to completely different isolated folder and test again
    new_loc = tmp_path / "transferred_campaign"
    shutil.copytree(archive_dir, new_loc)
    is_valid_transferred, msg_transferred = verify_campaign_archive(new_loc)
    assert is_valid_transferred is True
    assert msg_transferred == "Archive verified successfully"


def test_ablation_script_configuration_and_execution_smoke(tmp_path):
    """Test ablation study variant configs and execution smoke run."""
    import yaml
    from PIL import Image

    from scripts.run_ablations import run_ablation_experiment

    # Create dummy image on disk
    dummy_img = tmp_path / "dummy.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(dummy_img)

    # Create dummy split files
    eyeq_split = tmp_path / "eyeq_test.csv"
    eyeq_split.write_text(f"image_id,patient_id,path,dataset,quality\ne1,p1,{dummy_img},EyeQ,0\n")
    dd_split = tmp_path / "dd_test.csv"
    dd_split.write_text(
        f"image_id,patient_id,path,dataset,overall_quality,artifact,clarity,field_definition\nd1,p1,{dummy_img},DeepDRiD,0,0,0,0\n"
    )

    # Create minimal base config
    base_cfg = {
        "experiment": {"name": "test_ablation", "seed": 42},
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": str(eyeq_split),
                    "val_split": str(eyeq_split),
                },
                "deepdrid": {
                    "enabled": True,
                    "train_split": str(dd_split),
                    "val_split": str(dd_split),
                },
            },
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": False,
            "dropout": 0.1,
            "latent_dim": 64,
            "heads": {
                "quality": {"num_classes": 3, "weight": 1.0},
                "overall_quality": {"num_classes": 2, "weight": 1.0},
                "artifact": {"num_classes": 3, "weight": 0.0},
                "clarity": {"num_classes": 3, "weight": 0.0},
                "field_definition": {"num_classes": 3, "weight": 0.0},
            },
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "epochs": 1,
            "batch_size": 2,
            "early_stopping_patience": 3,
            "selection_metric": "primary_macro_f1",
            "mixed_precision": False,
            "image_size": 64,
            "num_workers": 0,
        },
        "loss": {"label_smoothing": 0.0},
    }
    cfg_p = tmp_path / "base_config.yaml"
    with open(cfg_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f)

    # Run smoke ablation
    out_dir = tmp_path / "ablation_out"
    res = run_ablation_experiment(
        name="test_no_attr",
        base_config_path=str(cfg_p),
        overrides={},
        output_dir=out_dir,
        eyeq_split=str(eyeq_split),
        deepdrid_split=str(dd_split),
        seed=2026,
        smoke_test=True,
    )
    assert res["seed"] == 2026
    assert res["ablation_name"] == "test_no_attr"
    assert "best_val_macro_f1" in res
