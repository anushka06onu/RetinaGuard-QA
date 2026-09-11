"""Typed Pydantic configuration schemas for RetinaGuard training."""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "retinaguard_multitask"
    seed: int = 2026
    seeds: List[int] = Field(default_factory=lambda: [2026, 2027, 2028])


class DatasetSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    train_split: str
    val_split: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: Dict[str, DatasetSplitConfig]

    @model_validator(mode="after")
    def check_at_least_one_enabled(self) -> "DataConfig":
        enabled_count = sum(1 for ds in self.datasets.values() if ds.enabled)
        if enabled_count == 0:
            raise ValueError("At least one dataset must be enabled for training.")
        return self


class HeadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_classes: int = Field(gt=0)
    weight: float = Field(ge=0.0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: Literal["mobilenetv3_large_100", "efficientnet_b0", "convnext_tiny"] = (
        "mobilenetv3_large_100"
    )
    pretrained: bool = True
    dropout: float = Field(ge=0.0, le=0.9, default=0.2)
    latent_dim: int = Field(gt=0, default=128)
    heads: Dict[str, HeadConfig]

    @field_validator("heads")
    @classmethod
    def validate_head_counts(cls, v: Dict[str, HeadConfig]) -> Dict[str, HeadConfig]:
        if "quality" in v and v["quality"].num_classes != 3:
            raise ValueError(
                f"EyeQ quality head must have 3 classes, got {v['quality'].num_classes}"
            )
        if "overall_quality" in v and v["overall_quality"].num_classes != 2:
            raise ValueError(
                f"DeepDRiD overall_quality head must have 2 classes (binary), got {v['overall_quality'].num_classes}"
            )
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_size: int = Field(gt=0, default=384)
    epochs: int = Field(gt=0, default=40)
    batch_size: int = Field(gt=0, default=32)
    num_workers: int = Field(ge=0, default=2)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = Field(gt=0.0, default=0.0003)
    weight_decay: float = Field(ge=0.0, default=0.0001)
    scheduler: Literal["cosine", "step", "reduce_on_plateau", "none"] = "cosine"
    mixed_precision: bool = True
    early_stopping_patience: int = Field(gt=0, default=7)
    selection_metric: Literal[
        "primary_macro_f1", "val_loss", "val_macro_f1", "val_deepdrid_overall_macro_f1"
    ] = "primary_macro_f1"


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_loss: Literal["cross_entropy", "focal"] = "cross_entropy"
    attribute_loss: Literal["ordinal_or_ce", "cross_entropy", "mse"] = "ordinal_or_ce"
    label_smoothing: float = Field(ge=0.0, le=0.5, default=0.05)


class TrainMultiTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig


def validate_training_config(cfg_dict: Dict[str, Any]) -> TrainMultiTaskConfig:
    """Validate a raw configuration dictionary against strict Pydantic schemas."""
    return TrainMultiTaskConfig.model_validate(cfg_dict)
