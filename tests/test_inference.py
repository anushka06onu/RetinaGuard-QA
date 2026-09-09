"""Unit tests for inference runtime, decision policy engine, and ONNX parity."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel
from src.retinaguard.inference.schemas import DecisionAction
from src.retinaguard.inference.decision_policy import DecisionPolicyEngine
from src.retinaguard.inference.predictor import RetinaGuardPredictor


def test_decision_engine_rules():
    engine = DecisionPolicyEngine(uncertainty_threshold=0.85, ood_energy_threshold=1.0)

    # Case 1: High quality Good -> ACCEPT
    res_good = engine.evaluate(
        probs={"good": 0.90, "usable": 0.08, "reject": 0.02},
        uncertainty=0.35,
        ood_score=3.5,
        is_valid_modality=True,
        attributes_raw={"artifact": 0, "clarity": 0, "field_definition": 0}
    )
    assert res_good.decision == DecisionAction.ACCEPT
    assert res_good.quality == "good"
    assert "Ready for clinical review" in res_good.feedback[0]

    # Case 2: Reject quality with poor clarity -> RECAPTURE
    res_reject = engine.evaluate(
        probs={"good": 0.05, "usable": 0.15, "reject": 0.80},
        uncertainty=0.45,
        ood_score=2.8,
        is_valid_modality=True,
        attributes_raw={"artifact": 0, "clarity": 2, "field_definition": 0}
    )
    assert res_reject.decision == DecisionAction.RECAPTURE
    assert res_reject.quality == "reject"
    assert any("blur" in fb.lower() for fb in res_reject.feedback)

    # Case 3: High uncertainty -> MANUAL_REVIEW
    res_uncertain = engine.evaluate(
        probs={"good": 0.40, "usable": 0.35, "reject": 0.25},
        uncertainty=1.52,  # > 0.85
        ood_score=2.5,
        is_valid_modality=True
    )
    assert res_uncertain.decision == DecisionAction.MANUAL_REVIEW

    # Case 4: Non-fundus modality -> UNSUPPORTED_INPUT
    res_ood = engine.evaluate(
        probs={"good": 0.33, "usable": 0.33, "reject": 0.34},
        uncertainty=1.58,
        ood_score=-5.0,
        is_valid_modality=False
    )
    assert res_ood.decision == DecisionAction.UNSUPPORTED_INPUT


def test_predictor_onnx_inference(tmp_path):
    # Verify inference with existing ONNX model or synthetic test
    onnx_path = Path("artifacts/models/model.onnx")
    if not onnx_path.exists():
        model = RetinaGuardMultiTaskModel(backbone_name="mobilenetv3_large_100", pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 384, 384)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input_image"],
            output_names=["quality_logits", "artifact_logits", "clarity_logits", "field_logits", "features"],
            opset_version=18
        )

    predictor = RetinaGuardPredictor(model_path=str(onnx_path))
    test_img = Image.new("RGB", (200, 200), color=(180, 80, 30))
    res = predictor.predict(test_img)
    assert res.decision in [DecisionAction.ACCEPT, DecisionAction.MANUAL_REVIEW, DecisionAction.RECAPTURE, DecisionAction.UNSUPPORTED_INPUT]
    assert res.calibrated_confidence > 0
    assert res.probabilities.good >= 0
    assert res.latency_ms is not None and res.latency_ms > 0

