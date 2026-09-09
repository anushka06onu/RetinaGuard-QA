"""Unit tests for inference runtime, decision engine, and ONNX parity."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

from src.models.multi_task_head import RetinaGuardNet
from src.inference.decision_engine import DecisionEngine, DecisionAction
from src.inference.engine import RetinaGuardInferenceEngine
from src.inference.onnx_exporter import export_to_onnx, verify_onnx_numerical_parity


def test_decision_engine_rules():
    engine = DecisionEngine()

    # Case 1: High quality -> ACCEPT
    res_good = engine.evaluate(
        grade_probs=np.array([0.90, 0.08, 0.02]),
        defect_probs=np.array([0.05, 0.02, 0.01, 0.04, 0.02, 0.01]),
        quality_score=94.0,
        energy_score=2.5,
        entropy=0.35,
        is_fundus_modality=True
    )
    assert res_good.decision == DecisionAction.ACCEPT
    assert res_good.quality_grade == "Good"

    # Case 2: Reject grade with Severe Blur -> RECAPTURE_WITH_GUIDANCE
    res_reject = engine.evaluate(
        grade_probs=np.array([0.05, 0.15, 0.80]),
        defect_probs=np.array([0.88, 0.10, 0.05, 0.10, 0.05, 0.02]),
        quality_score=25.0,
        energy_score=1.8,
        entropy=0.55,
        is_fundus_modality=True
    )
    assert res_reject.decision == DecisionAction.RECAPTURE_WITH_GUIDANCE
    assert len(res_reject.detected_defects) > 0
    assert "Blur" in res_reject.detected_defects[0]["name"]

    # Case 3: High Entropy -> MANUAL_REVIEW
    res_uncertain = engine.evaluate(
        grade_probs=np.array([0.40, 0.35, 0.25]),
        defect_probs=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        quality_score=60.0,
        energy_score=0.5,
        entropy=1.52, # > 0.85
        is_fundus_modality=True
    )
    assert res_uncertain.decision == DecisionAction.MANUAL_REVIEW

    # Case 4: Non-fundus -> UNSUPPORTED_OOD
    res_ood = engine.evaluate(
        grade_probs=np.array([0.33, 0.33, 0.33]),
        defect_probs=np.zeros(6),
        quality_score=0.0,
        energy_score=-6.0,
        entropy=1.58,
        is_fundus_modality=False
    )
    assert res_ood.decision == DecisionAction.UNSUPPORTED_OOD


def test_onnx_export_and_inference(tmp_path):
    model = RetinaGuardNet(backbone_name="mobilenetv3_large_100", pretrained=False)
    onnx_file = tmp_path / "test_model.onnx"

    export_to_onnx(model, onnx_file)
    assert onnx_file.exists()

    parity = verify_onnx_numerical_parity(model, onnx_file)
    assert parity["is_parity_verified"] is True

    # Test runtime engine with ONNX
    engine = RetinaGuardInferenceEngine(model_path=onnx_file, use_onnx=True)
    test_img = Image.new("RGB", (200, 200), color=(180, 80, 30))
    result = engine.predict(test_img)
    assert result.decision in DecisionAction
    assert result.latency_ms > 0
