"""Multi-task neural network architecture per Phase 8 of blueprint."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RetinaGuardMultiTaskModel(nn.Module):
    """Shared lightweight encoder with dataset-aware masked heads:
    - EyeQ Quality Head: Good / Usable / Reject (3 classes)
    - DeepDRiD Overall Quality Head (3 classes)
    - Artifact Ordinal Head (3 levels)
    - Clarity Ordinal Head (3 levels)
    - Field Definition Ordinal Head (3 levels)
    - Latent Projection Head (128-d) for OOD Mahalanobis distance
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        dropout: float = 0.2,
        latent_dim: int = 128
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=dropout)
        except Exception:
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=dropout)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[-1]

        # Common neck
        self.neck = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

        # 1. EyeQ Quality Head (Good, Usable, Reject)
        self.quality_head = nn.Linear(512, 3)

        # 2. DeepDRiD Overall Quality Head
        self.overall_quality_head = nn.Linear(512, 3)

        # 3. Attribute Heads
        self.artifact_head = nn.Linear(512, 3)
        self.clarity_head = nn.Linear(512, 3)
        self.field_def_head = nn.Linear(512, 3)

        # 4. Latent Projection for OOD
        self.projection_head = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        # Post-hoc temperature calibration parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        neck_out = self.neck(features)

        q_logits = self.quality_head(neck_out)
        cal_q_logits = q_logits / torch.clamp(self.temperature, min=0.01)

        oq_logits = self.overall_quality_head(neck_out)
        art_logits = self.artifact_head(neck_out)
        cla_logits = self.clarity_head(neck_out)
        fld_logits = self.field_def_head(neck_out)

        latent = F.normalize(self.projection_head(neck_out), p=2, dim=-1)

        return {
            "quality_logits": q_logits,
            "calibrated_quality_logits": cal_q_logits,
            "overall_quality_logits": oq_logits,
            "artifact_logits": art_logits,
            "clarity_logits": cla_logits,
            "field_definition_logits": fld_logits,
            "latent_features": latent,
            "features": neck_out
        }

    def set_temperature(self, temp: float):
        self.temperature.data.fill_(max(0.01, float(temp)))
