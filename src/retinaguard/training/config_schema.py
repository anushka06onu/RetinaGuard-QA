"""Typed Pydantic configuration schemas for RetinaGuard training."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "retinaguard_multitask"
    seed: int = 2026
    seeds: List[int] = Field(default_factory=lambda: [2026, 2027, 2028])


class DatasetSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    train_split: Optional[str] = None
    val_split: Optional[str] = None

    @model_validator(mode="after")
    def validate_splits_when_enabled(self) -> "DatasetSplitConfig":
        if self.enabled:
            if not self.train_split or not self.val_split:
                raise ValueError(
                    "Enabled dataset must specify non-empty train_split and val_split paths."
                )
        return self


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
    heads: Dict[str, HeadConfig] = Field(default_factory=dict)

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
        "primary_macro_f1",
        "val_loss",
        "val_macro_f1",
        "quality_macro_f1",
        "val_deepdrid_overall_macro_f1",
    ] = "primary_macro_f1"


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_loss: Literal["cross_entropy"] = "cross_entropy"
    attribute_loss: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = Field(ge=0.0, le=0.5, default=0.05)


class TrainMultiTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig

    @model_validator(mode="after")
    def validate_heads_match_datasets(self) -> "TrainMultiTaskConfig":
        allowed_heads = {
            "quality",
            "overall_quality",
            "artifact",
            "clarity",
            "field_definition",
        }
        unknown_heads = set(self.model.heads.keys()) - allowed_heads
        if unknown_heads:
            raise ValueError(
                f"Unknown model heads configured: {unknown_heads}. Allowed: {allowed_heads}"
            )

        eyeq_cfg = self.data.datasets.get("eyeq")
        eyeq_enabled = eyeq_cfg.enabled if eyeq_cfg else False

        deepdrid_cfg = self.data.datasets.get("deepdrid")
        deepdrid_enabled = deepdrid_cfg.enabled if deepdrid_cfg else False

        if eyeq_enabled:
            if "quality" not in self.model.heads:
                raise ValueError(
                    "EyeQ dataset is enabled, but required 'quality' head is missing in model.heads."
                )

        if deepdrid_enabled:
            required_dd_heads = {
                "overall_quality",
                "artifact",
                "clarity",
                "field_definition",
            }
            missing_dd_heads = required_dd_heads - set(self.model.heads.keys())
            if missing_dd_heads:
                raise ValueError(
                    f"DeepDRiD dataset is enabled, but required heads are missing in model.heads: {missing_dd_heads}"
                )

        # Check that total weight across relevant heads is greater than zero
        active_heads = []
        if eyeq_enabled:
            active_heads.append("quality")
        if deepdrid_enabled:
            active_heads.extend(["overall_quality", "artifact", "clarity", "field_definition"])

        total_weight = sum(
            self.model.heads[h].weight for h in active_heads if h in self.model.heads
        )
        if total_weight <= 0.0:
            raise ValueError(
                "Every enabled dataset head has weight 0.0; at least one active head must have positive weight."
            )

        return self


def validate_training_config(cfg_dict: Dict[str, Any]) -> TrainMultiTaskConfig:
    """Validate a raw configuration dictionary against strict Pydantic schemas."""
    return TrainMultiTaskConfig.model_validate(cfg_dict)
