from pathlib import Path
from typing import Any, Dict

import timm
import torch
import torch.nn.functional as F
from torch import nn


class RetinaGuardMultiTaskModel(nn.Module):
    """Shared lightweight encoder with dataset-aware masked heads:
    - EyeQ Quality Head: Good / Usable / Reject (3 classes: 0=good, 1=usable, 2=reject)
    - DeepDRiD Overall Quality Head: Binary Diagnosable (2 classes: 0=good, 1=poor/reject)
    - Artifact Ordinal Head (3 levels: 0=none, 1=mild, 2=severe)
    - Clarity Ordinal Head (3 levels: 0=high, 1=moderate, 2=low)
    - Field Definition Ordinal Head (3 levels: 0=adequate, 1=acceptable, 2=poor)
    - Latent Projection Head (128-d) for OOD Mahalanobis distance
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        dropout: float = 0.2,
        latent_dim: int = 128,
        allow_random_initialization: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.initialization_source = "random"

        if pretrained:
            try:
                self.backbone = timm.create_model(
                    backbone_name, pretrained=True, num_classes=0, drop_rate=dropout
                )
                self.initialization_source = f"timm_pretrained_{backbone_name}"
            except Exception as e:
                if not allow_random_initialization:
                    raise RuntimeError(
                        f"Failed to load pretrained weights for '{backbone_name}': {e}. "
                        "To allow offline random initialization, pass allow_random_initialization=True."
                    ) from e
                self.backbone = timm.create_model(
                    backbone_name, pretrained=False, num_classes=0, drop_rate=dropout
                )
                self.initialization_source = "random_offline_fallback"
        else:
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, num_classes=0, drop_rate=dropout
            )
            self.initialization_source = "random_unpretrained"

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[-1]

        # Common neck
        self.neck = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(p=dropout)
        )

        # 1. EyeQ Quality Head (Good=0, Usable=1, Reject=2) - 3 classes
        self.quality_head = nn.Linear(512, 3)

        # 2. DeepDRiD Overall Quality Head (Good=0, Poor/Reject=1) - 2 classes
        self.overall_quality_head = nn.Linear(512, 2)

        # 3. Attribute Heads (Ordinal 3 levels: 0, 1, 2)
        self.artifact_head = nn.Linear(512, 3)
        self.clarity_head = nn.Linear(512, 3)
        self.field_def_head = nn.Linear(512, 3)

        # 4. Latent Projection for OOD
        self.projection_head = nn.Sequential(nn.Linear(512, latent_dim), nn.BatchNorm1d(latent_dim))

    @classmethod
    def from_checkpoint_metadata(
        cls, metadata_or_path: Dict[str, Any] | str | Path
    ) -> "RetinaGuardMultiTaskModel":
        """Reconstruct model architecture exactly matching recorded checkpoint metadata."""
        if isinstance(metadata_or_path, (str, Path)):
            try:
                state = torch.load(metadata_or_path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(metadata_or_path, map_location="cpu")
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        else:
            metadata = metadata_or_path
            state_dict = None

        backbone = metadata.get("backbone", "mobilenetv3_large_100")
        latent_dim = metadata.get("latent_dim")
        if latent_dim is None and "head_dimensions" in metadata:
            latent_dim = metadata["head_dimensions"].get("projection_head", 128)
        if latent_dim is None:
            latent_dim = 128
        dropout = metadata.get("dropout", 0.2)

        model = cls(
            backbone_name=backbone,
            pretrained=False,
            dropout=dropout,
            latent_dim=latent_dim,
            allow_random_initialization=True,
        )
        if state_dict is not None and isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        return model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        neck_out = self.neck(features)

        q_logits = self.quality_head(neck_out)
        oq_logits = self.overall_quality_head(neck_out)
        art_logits = self.artifact_head(neck_out)
        cla_logits = self.clarity_head(neck_out)
        fld_logits = self.field_def_head(neck_out)

        latent = F.normalize(self.projection_head(neck_out), p=2, dim=-1)

        return {
            "quality_logits": q_logits,
            "overall_quality_logits": oq_logits,
            "artifact_logits": art_logits,
            "clarity_logits": cla_logits,
            "field_definition_logits": fld_logits,
            "latent_features": latent,
            "features": neck_out,
        }
