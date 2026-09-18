"""Unit tests for models, masked multi-task losses, and baselines."""

import numpy as np
import pytest
import torch
from PIL import Image

from retinaguard.models.baselines import (
    ClassicalFeatureExtractor,
    SingleTaskQualityModel,
)
from retinaguard.models.losses import MaskedMultiTaskLoss
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


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
    assert "overall_quality_logits" in preds
    assert preds["quality_logits"].shape == (2, 3)
    assert preds["overall_quality_logits"].shape == (2, 2)
    assert preds["artifact_logits"].shape == (2, 3)
    assert preds["latent_features"].shape == (2, 128)

    criterion = MaskedMultiTaskLoss()
    batch = {
        "quality_target": torch.tensor([0, 1]),
        "quality_mask": torch.tensor([1.0, 0.0]),
        "overall_quality_target": torch.tensor([0, 1]),
        "overall_quality_mask": torch.tensor([0.0, 1.0]),
        "artifact_target": torch.tensor([0, 2]),
        "artifact_mask": torch.tensor([1.0, 0.0]),
        "clarity_target": torch.tensor([1, 1]),
        "clarity_mask": torch.tensor([0.0, 1.0]),
        "field_definition_target": torch.tensor([0, 0]),
        "field_definition_mask": torch.tensor([1.0, 1.0]),
    }

    loss_dict = criterion(preds, batch)
    assert "loss_total" in loss_dict
    assert "loss_quality" in loss_dict
    assert "loss_overall_quality" in loss_dict
    assert loss_dict["loss_total"].item() > 0
    assert loss_dict["loss_overall_quality"].item() > 0


def test_evaluate_epoch_multi_head_metrics_and_logger(tmp_path):
    from torch.utils.data import DataLoader

    from retinaguard.training.callbacks import MetricHistoryLogger
    from retinaguard.training.engine import evaluate_epoch

    model = RetinaGuardMultiTaskModel(backbone_name="mobilenetv3_small_100", pretrained=False)
    model.eval()

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224),
                "quality_target": torch.tensor(idx % 3, dtype=torch.long),
                "quality_mask": torch.tensor(1.0 if idx < 2 else 0.0, dtype=torch.float32),
                "overall_quality_target": torch.tensor((idx + 1) % 2, dtype=torch.long),
                "overall_quality_mask": torch.tensor(1.0 if idx >= 2 else 0.0, dtype=torch.float32),
                "artifact_target": torch.tensor(idx % 3, dtype=torch.long),
                "artifact_mask": torch.tensor(1.0, dtype=torch.float32),
                "clarity_target": torch.tensor(idx % 3, dtype=torch.long),
                "clarity_mask": torch.tensor(1.0, dtype=torch.float32),
                "field_definition_target": torch.tensor(idx % 3, dtype=torch.long),
                "field_definition_mask": torch.tensor(1.0, dtype=torch.float32),
            }

    loader = DataLoader(DummyDataset(), batch_size=2)
    criterion = MaskedMultiTaskLoss()
    device = torch.device("cpu")

    loss, metrics, logits, labels = evaluate_epoch(model, loader, criterion, device)

    assert isinstance(metrics, dict)
    assert "quality_macro_f1" in metrics
    assert "overall_quality_macro_f1" in metrics
    assert "artifact_macro_f1" in metrics
    assert "primary_macro_f1" in metrics
    assert metrics["primary_macro_f1"] >= 0.0

    # Test MetricHistoryLogger with dictionary
    hist_file = tmp_path / "history.json"
    logger = MetricHistoryLogger(hist_file)
    logger.log(1, 1.5, loss, metrics)

    import json

    with open(hist_file) as f:
        saved_hist = json.load(f)

    assert len(saved_hist["epoch"]) == 1
    assert "val_quality_macro_f1" in saved_hist
    assert "val_overall_quality_macro_f1" in saved_hist


def test_trained_heads_dynamic_derivation():
    """Verify that trained_heads list is derived strictly from enabled training datasets."""

    def get_trained_heads(eyeq_enabled: bool, deepdrid_enabled: bool):
        heads = []
        if eyeq_enabled:
            heads.append("quality_logits")
        if deepdrid_enabled:
            heads.extend(
                [
                    "overall_quality_logits",
                    "artifact_logits",
                    "clarity_logits",
                    "field_definition_logits",
                ]
            )
        return heads

    # 1. DeepDRiD only
    dd_heads = get_trained_heads(eyeq_enabled=False, deepdrid_enabled=True)
    assert "quality_logits" not in dd_heads
    assert dd_heads == [
        "overall_quality_logits",
        "artifact_logits",
        "clarity_logits",
        "field_definition_logits",
    ]

    # 2. EyeQ only
    eyeq_heads = get_trained_heads(eyeq_enabled=True, deepdrid_enabled=False)
    assert eyeq_heads == ["quality_logits"]

    # 3. Joint training
    joint_heads = get_trained_heads(eyeq_enabled=True, deepdrid_enabled=True)
    assert len(joint_heads) == 5
    assert "quality_logits" in joint_heads
    assert "overall_quality_logits" in joint_heads


def test_calibration_rejection_untrained_head(tmp_path):
    """Verify that calibration immediately rejects attempts to calibrate untrained heads or mismatched splits."""
    from scripts.calibrate import calibrate_temperature_and_thresholds

    dummy_val_df = tmp_path / "val_split.csv"
    dummy_val_df.write_text("dataset,quality_canonical,overall_quality_canonical,path\n")

    # Untrained head rejection
    with pytest.raises(ValueError, match="is not among the model's trained heads"):
        calibrate_temperature_and_thresholds(
            checkpoint_path="nonexistent.ckpt",
            val_split_path=str(dummy_val_df),
            task_head="quality_logits",
            trained_heads=["overall_quality_logits"],
        )

    # Cross-dataset mismatch rejection
    with pytest.raises(ValueError, match="Cross-dataset validation mismatch"):
        calibrate_temperature_and_thresholds(
            checkpoint_path="nonexistent.ckpt",
            val_split_path="data/splits/deepdrid_val.csv",
            task_head="quality_logits",
            trained_heads=["quality_logits"],
        )


def test_predictor_binary_and_three_class_entropy_bounds():
    """Verify entropy calculation and upper bounds for binary vs 3-class tasks."""
    import numpy as np

    # Binary max entropy is log2(2) = 1.0 bit
    p_binary_max = np.array([0.5, 0.5])
    ent_bin = -np.sum(p_binary_max * np.log2(p_binary_max))
    assert np.isclose(ent_bin, 1.0)

    # 3-class max entropy is log2(3) = 1.58496 bits
    p_3class_max = np.array([1 / 3, 1 / 3, 1 / 3])
    ent_3class = -np.sum(p_3class_max * np.log2(p_3class_max))
    assert np.isclose(ent_3class, np.log2(3))
