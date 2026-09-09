"""Unit tests for models, masked multi-task losses, and baselines."""

import numpy as np
import torch
from PIL import Image

from src.retinaguard.models.baselines import (
    ClassicalFeatureExtractor,
    SingleTaskQualityModel,
)
from src.retinaguard.models.losses import MaskedMultiTaskLoss
from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel


def test_classical_feature_extractor():
    img = Image.new("RGB", (100, 100), color=(180, 80, 20))
    feat = ClassicalFeatureExtractor.extract_features(img)
    assert feat.shape == (10,)
    assert not np.isnan(feat).any()


def test_singletask_model_forward():
    model = SingleTaskQualityModel(backbone_name="mobilenetv3_small_100", pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    assert out.shape == (2, 3)


def test_multitask_model_forward_and_loss():
    model = RetinaGuardMultiTaskModel(backbone_name="mobilenetv3_large_100", pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    preds = model(x)

    assert "quality_logits" in preds
    assert preds["quality_logits"].shape == (2, 3)
    assert preds["artifact_logits"].shape == (2, 3)
    assert preds["latent_features"].shape == (2, 128)

    criterion = MaskedMultiTaskLoss()
    batch = {
        "quality_target": torch.tensor([0, 1]),
        "quality_mask": torch.tensor([1.0, 1.0]),
        "artifact_target": torch.tensor([0, 2]),
        "artifact_mask": torch.tensor([1.0, 0.0]),
        "clarity_target": torch.tensor([1, 1]),
        "clarity_mask": torch.tensor([0.0, 1.0]),
        "field_definition_target": torch.tensor([0, 0]),
        "field_definition_mask": torch.tensor([1.0, 1.0]),
    }

    loss_dict = criterion(preds, batch)
    assert "loss_total" in loss_dict
    assert loss_dict["loss_total"].item() > 0
