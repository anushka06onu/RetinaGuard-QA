"""Unit tests for inference runtime, decision policy engine, and ONNX parity."""

import torch
from PIL import Image

from retinaguard.inference.decision_policy import DecisionPolicyEngine
from retinaguard.inference.predictor import RetinaGuardPredictor
from retinaguard.inference.schemas import DecisionAction
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


def test_decision_engine_rules():
    engine = DecisionPolicyEngine(uncertainty_threshold=0.85, ood_energy_threshold=1.0)

    # Case 1: High quality Good -> ACCEPT
    res_good = engine.evaluate(
        probs={"good": 0.90, "usable": 0.08, "reject": 0.02},
        uncertainty=0.35,
        ood_score=3.5,
        is_valid_modality=True,
        attributes_raw={"artifact": 0, "clarity": 0, "field_definition": 0},
    )
    assert res_good.decision == DecisionAction.ACCEPT
    assert res_good.quality == "good"
    assert "technically acceptable" in res_good.feedback[0]

    # Case 2: Reject quality with poor clarity -> RECAPTURE
    res_reject = engine.evaluate(
        probs={"good": 0.05, "usable": 0.15, "reject": 0.80},
        uncertainty=0.45,
        ood_score=2.8,
        is_valid_modality=True,
        attributes_raw={"artifact": 0, "clarity": 2, "field_definition": 0},
    )
    assert res_reject.decision == DecisionAction.RECAPTURE
    assert res_reject.quality == "reject"
    assert any("blur" in fb.lower() for fb in res_reject.feedback)

    # Case 3: High uncertainty -> MANUAL_REVIEW
    res_uncertain = engine.evaluate(
        probs={"good": 0.40, "usable": 0.35, "reject": 0.25},
        uncertainty=1.52,  # > 0.85
        ood_score=2.5,
        is_valid_modality=True,
    )
    assert res_uncertain.decision == DecisionAction.MANUAL_REVIEW

    # Case 4: Non-fundus modality -> UNSUPPORTED_INPUT
    res_ood = engine.evaluate(
        probs={"good": 0.33, "usable": 0.33, "reject": 0.34},
        uncertainty=1.58,
        ood_score=-5.0,
        is_valid_modality=False,
    )
    assert res_ood.decision == DecisionAction.UNSUPPORTED_INPUT


def test_predictor_onnx_inference(tmp_path):
    # Verify inference with a temporary test model in tmp_path
    onnx_path = tmp_path / "temp_test_model.onnx"
    model = RetinaGuardMultiTaskModel(backbone_name="mobilenetv3_large_100", pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input_image"],
        output_names=[
            "quality_logits",
            "artifact_logits",
            "clarity_logits",
            "field_logits",
            "features",
        ],
        opset_version=18,
    )

    predictor = RetinaGuardPredictor(model_path=str(onnx_path))
    test_img = Image.new("RGB", (200, 200), color=(180, 80, 30))
    res = predictor.predict(test_img)
    assert res.decision in [
        DecisionAction.ACCEPT,
        DecisionAction.MANUAL_REVIEW,
        DecisionAction.RECAPTURE,
        DecisionAction.UNSUPPORTED_INPUT,
    ]
    assert res.calibrated_confidence > 0
    assert res.probabilities.good >= 0
    assert res.latency_ms is not None and res.latency_ms > 0


def test_predictor_strict_config_validation(tmp_path):
    import json

    import pytest

    # 1. Invalid temperature (<= 0)
    cal_file = tmp_path / "bad_calibration.json"
    cal_file.write_text(json.dumps({"temperature": -0.5}))

    with pytest.raises(ValueError, match="Temperature must be a finite float > 0"):
        RetinaGuardPredictor(calibration_config_path=cal_file, model_path=None)

    # 2. Invalid preprocessing image_size
    prep_file = tmp_path / "bad_preprocessing.json"
    prep_file.write_text(json.dumps({"image_size": -100}))

    with pytest.raises(ValueError, match="Invalid image_size"):
        RetinaGuardPredictor(preprocessing_config_path=prep_file, model_path=None)

    # 3. Invalid uncertainty threshold (> log2(3) or <= 0)
    cal_bad_u = tmp_path / "bad_u_cal.json"
    cal_bad_u.write_text(json.dumps({"temperature": 1.0, "uncertainty_threshold": 2.5}))
    with pytest.raises(ValueError, match="Invalid uncertainty_threshold"):
        RetinaGuardPredictor(calibration_config_path=cal_bad_u, model_path=None)

    # 4. Valid configs pass
    cal_good = tmp_path / "good_calibration.json"
    cal_good.write_text(
        json.dumps(
            {
                "temperature": 1.25,
                "uncertainty_threshold": 0.85,
                "ood_energy_threshold": 1.0,
                "ood_direction": "lower_is_ood",
            }
        )
    )
    prep_good = tmp_path / "good_preprocessing.json"
    prep_good.write_text(
        json.dumps(
            {
                "image_size": 224,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "class_order": ["good", "usable", "reject"],
            }
        )
    )

    pred = RetinaGuardPredictor(
        model_path=None,
        preprocessing_config_path=prep_good,
        calibration_config_path=cal_good,
    )
    assert pred.temperature == 1.25
    assert pred.image_size == 224
    assert pred.policy_engine.uncertainty_threshold == 0.85
    assert pred.policy_engine.ood_energy_threshold == 1.0


def test_predictor_production_mode_requirements(tmp_path, monkeypatch):
    import json

    import pytest

    monkeypatch.setenv("APP_MODE", "production")
    monkeypatch.setenv("TEST_MODE", "0")

    # Incomplete calibration file missing required fields in production mode
    cal_incomplete = tmp_path / "incomplete_cal.json"
    cal_incomplete.write_text(json.dumps({"temperature": 1.2}))

    with pytest.raises(ValueError, match="Production calibration metadata missing required fields"):
        RetinaGuardPredictor(calibration_config_path=cal_incomplete, model_path=None)

    # Model hash mismatch in production mode
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_text("fake_model_bytes")

    cal_mismatch = tmp_path / "mismatch_cal.json"
    cal_mismatch.write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "uncertainty_threshold": 0.85,
                "ood_energy_threshold": 1.0,
                "ood_direction": "lower_is_ood",
                "ood_score_type": "energy",
                "validation_split_sha256": "abcdef",
                "model_checkpoint_sha256": "DIFFERENT_HASH_12345",
                "onnx_model_sha256": "DIFFERENT_ONNX_HASH_12345",
                "fitting_method": "temperature_scaling",
                "created_at_utc": "2026-09-10T12:00:00Z",
            }
        )
    )

    with pytest.raises(ValueError, match="Model hash mismatch in production mode"):
        RetinaGuardPredictor(calibration_config_path=cal_mismatch, model_path=dummy_model)
