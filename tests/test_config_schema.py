"""Unit tests for training configuration schema validation."""

import pytest
from pydantic import ValidationError

from retinaguard.training.config_schema import (
    validate_training_config,
)


def test_valid_training_config():
    valid_dict = {
        "experiment": {"name": "test_exp", "seed": 2026, "seeds": [2026, 2027, 2028]},
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                },
                "deepdrid": {
                    "enabled": True,
                    "train_split": "data/splits/deepdrid_train.csv",
                    "val_split": "data/splits/deepdrid_val.csv",
                },
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": True,
            "dropout": 0.2,
            "latent_dim": 128,
            "heads": {
                "quality": {"num_classes": 3, "weight": 1.0},
                "overall_quality": {"num_classes": 2, "weight": 0.8},
                "artifact": {"num_classes": 3, "weight": 0.5},
                "clarity": {"num_classes": 3, "weight": 0.5},
                "field_definition": {"num_classes": 3, "weight": 0.5},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 40,
            "batch_size": 32,
            "num_workers": 2,
            "optimizer": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "mixed_precision": True,
            "early_stopping_patience": 7,
            "selection_metric": "primary_macro_f1",
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
            "label_smoothing": 0.05,
        },
    }
    cfg = validate_training_config(valid_dict)
    assert cfg.experiment.seed == 2026
    assert cfg.model.backbone == "mobilenetv3_large_100"
    assert cfg.training.optimizer == "adamw"


def test_invalid_optimizer_rejected():
    invalid_dict = {
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                }
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": True,
            "dropout": 0.2,
            "latent_dim": 128,
            "heads": {
                "quality": {"num_classes": 3, "weight": 1.0},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 40,
            "batch_size": 32,
            "num_workers": 2,
            "optimizer": "unsupported_opt_name",  # Invalid optimizer
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "mixed_precision": True,
            "early_stopping_patience": 7,
            "selection_metric": "primary_macro_f1",
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
            "label_smoothing": 0.05,
        },
    }
    with pytest.raises(ValidationError):
        validate_training_config(invalid_dict)


def test_invalid_head_classes_rejected():
    invalid_dict = {
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                }
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": True,
            "dropout": 0.2,
            "latent_dim": 128,
            "heads": {
                "quality": {"num_classes": 5, "weight": 1.0},  # Invalid class count (must be 3)
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 40,
            "batch_size": 32,
            "num_workers": 2,
            "optimizer": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "mixed_precision": True,
            "early_stopping_patience": 7,
            "selection_metric": "primary_macro_f1",
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
            "label_smoothing": 0.05,
        },
    }
    with pytest.raises(ValidationError):
        validate_training_config(invalid_dict)


def test_extra_unknown_fields_rejected():
    invalid_dict = {
        "unknown_extra_top_level_key": 123,
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                }
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "heads": {
                "quality": {"num_classes": 3, "weight": 1.0},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 40,
            "batch_size": 32,
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
        },
    }
    with pytest.raises(ValidationError):
        validate_training_config(invalid_dict)
