"""Unit tests for model architectures, multi-task losses, and classical baselines."""

import numpy as np
from PIL import Image
import torch
import pytest

from src.models.multi_task_head import RetinaGuardNet, MultiTaskLoss
from src.models.baselines import ClassicalImageQualityExtractor, ClassicalQualityClassifier, SingleTaskQualityClassifier


def test_retinaguard_net_forward():
    model = RetinaGuardNet(backbone_name="mobilenetv3_large_100", pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)

    assert "grade_logits" in out
    assert out["grade_logits"].shape == (2, 3)
    assert out["defect_logits"].shape == (2, 6)
    assert out["quality_score"].shape == (2, 1)
    assert out["latent_features"].shape == (2, 128)


def test_multitask_loss():
    criterion = MultiTaskLoss()
    preds = {
        "grade_logits": torch.randn(4, 3),
        "defect_logits": torch.randn(4, 6),
        "quality_score": torch.rand(4, 1) * 100.0
    }
    targets_grade = torch.tensor([0, 1, 2, 0])
    targets_defect = torch.randint(0, 2, (4, 6)).float()
    targets_score = torch.tensor([[85.0], [70.0], [20.0], [90.0]])

    loss_dict = criterion(preds, targets_grade, targets_defect, targets_score)
    assert "loss_total" in loss_dict
    assert loss_dict["loss_total"].item() > 0


def test_classical_iqa():
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    feat = ClassicalImageQualityExtractor.extract_features(img)
    assert feat.shape == (10,)
    assert not np.isnan(feat).any()

    clf = ClassicalQualityClassifier(classifier_type="random_forest")
    clf.fit([img, img], np.array([0, 1]))
    pred = clf.predict([img])
    assert len(pred) == 1
