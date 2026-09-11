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


def test_all_training_yamls_pass_validation():
    from pathlib import Path

    import yaml

    config_dir = Path("configs")
    train_yamls = list(config_dir.glob("train_*.yaml"))
    assert len(train_yamls) >= 2, f"Expected at least 2 training configs, found {train_yamls}"

    for yaml_path in train_yamls:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        validated = validate_training_config(cfg)
        assert validated is not None
        assert validated.training.batch_size > 0


def test_disabled_dataset_without_splits_is_valid():
    config_dict = {
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                },
                "deepdrid": {
                    "enabled": False,
                },
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
            "epochs": 10,
            "batch_size": 16,
            "scheduler": "reduce_on_plateau",
            "selection_metric": "quality_macro_f1",
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
        },
    }
    validated = validate_training_config(config_dict)
    assert validated.training.scheduler == "reduce_on_plateau"
    assert validated.training.selection_metric == "quality_macro_f1"


def test_missing_required_head_for_enabled_dataset():
    # DeepDRiD enabled but missing attribute heads
    config_dict = {
        "data": {
            "datasets": {
                "deepdrid": {
                    "enabled": True,
                    "train_split": "data/splits/deepdrid_train.csv",
                    "val_split": "data/splits/deepdrid_val.csv",
                },
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "heads": {
                "overall_quality": {"num_classes": 2, "weight": 1.0},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 10,
            "batch_size": 16,
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
        },
    }
    with pytest.raises(
        ValidationError, match="DeepDRiD dataset is enabled, but required heads are missing"
    ):
        validate_training_config(config_dict)


def test_unknown_head_name_rejected():
    config_dict = {
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                },
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "heads": {
                "quality": {"num_classes": 3, "weight": 1.0},
                "unknown_head": {"num_classes": 2, "weight": 0.5},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 10,
            "batch_size": 16,
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
        },
    }
    with pytest.raises(ValidationError, match="Unknown model heads configured"):
        validate_training_config(config_dict)


def test_all_zero_head_weights_rejected():
    config_dict = {
        "data": {
            "datasets": {
                "eyeq": {
                    "enabled": True,
                    "train_split": "data/splits/eyeq_train.csv",
                    "val_split": "data/splits/eyeq_val.csv",
                },
            }
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "heads": {
                "quality": {"num_classes": 3, "weight": 0.0},
            },
        },
        "training": {
            "image_size": 384,
            "epochs": 10,
            "batch_size": 16,
        },
        "loss": {
            "quality_loss": "cross_entropy",
            "attribute_loss": "ordinal_or_ce",
        },
    }
    with pytest.raises(ValidationError, match="Every enabled dataset head has weight 0.0"):
        validate_training_config(config_dict)
