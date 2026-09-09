"""Multi-task neural network architecture for quality grading, defect diagnosis, and uncertainty estimation."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone


class RetinaGuardNet(nn.Module):
    """Unified Multi-Task Retinal Image Quality Assurance Architecture.

    Outputs:
    - grade_logits: [B, 3] for Good (0), Usable (1), Reject (2)
    - defect_logits: [B, 6] for multi-label acquisition defect identification
    - quality_score: [B, 1] continuous visual quality indicator in [0, 100]
    - latent_features: [B, latent_dim] L2-normalized deep embedding for OOD detection
    """

    DEFECT_NAMES = [
        "Severe Blur / Focus Loss",
        "Underexposure",
        "Overexposure",
        "Uneven Illumination / Shadowing",
        "Field Truncation / Off-Center",
        "Optical Artifacts / Lens Smear"
    ]

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        num_grades: int = 3,
        num_defects: int = 6,
        latent_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_grades = num_grades
        self.num_defects = num_defects
        self.latent_dim = latent_dim

        # Backbone feature extractor
        self.backbone, in_features = create_backbone(backbone_name, pretrained=pretrained, drop_rate=dropout_rate)

        # Common neck layer
        self.neck = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate)
        )

        # Head A: 3-class Quality Grade
        self.grade_head = nn.Linear(512, num_grades)

        # Head B: Multi-label Defect Classifier
        self.defect_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_defects)
        )

        # Head C: Continuous Quality Index [0.0, 100.0]
        self.score_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Head D: OOD Latent Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        # Calibrated temperature scaling parameter (learned or set post-hoc)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass extracting pooled representations and multi-task heads."""
        features = self.backbone(x)
        neck_out = self.neck(features)

        # Predictions
        grade_logits = self.grade_head(neck_out)
        defect_logits = self.defect_head(neck_out)
        raw_score = self.score_head(neck_out)
        quality_score = raw_score * 100.0  # Scale [0, 1] to [0, 100]

        # Normalized embedding for OOD Mahalanobis / clustering
        latent_features = F.normalize(self.projection_head(neck_out), p=2, dim=-1)

        return {
            "grade_logits": grade_logits,
            "calibrated_grade_logits": grade_logits / torch.clamp(self.temperature, min=0.1),
            "defect_logits": defect_logits,
            "quality_score": quality_score,
            "latent_features": latent_features,
            "shared_features": neck_out
        }

    def set_temperature(self, temp_val: float):
        """Set post-hoc probability calibration temperature."""
        self.temperature.data.fill_(max(0.01, float(temp_val)))


class MultiTaskLoss(nn.Module):
    """Composite loss function balancing classification, multi-label defects, and continuous score."""

    def __init__(
        self,
        weight_grade: float = 1.0,
        weight_defect: float = 0.5,
        weight_score: float = 0.3,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.weight_grade = weight_grade
        self.weight_defect = weight_defect
        self.weight_score = weight_score

        self.grade_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.defect_criterion = nn.BCEWithLogitsLoss()
        self.score_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets_grade: torch.Tensor,
        targets_defect: Optional[torch.Tensor] = None,
        targets_score: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss terms."""
        loss_grade = self.grade_criterion(predictions["grade_logits"], targets_grade)
        total_loss = self.weight_grade * loss_grade

        loss_defect = torch.tensor(0.0, device=targets_grade.device)
        if targets_defect is not None:
            loss_defect = self.defect_criterion(predictions["defect_logits"], targets_defect.float())
            total_loss = total_loss + self.weight_defect * loss_defect

        loss_score = torch.tensor(0.0, device=targets_grade.device)
        if targets_score is not None:
            loss_score = self.score_criterion(predictions["quality_score"], targets_score.float())
            total_loss = total_loss + self.weight_score * loss_score

        return {
            "loss_total": total_loss,
            "loss_grade": loss_grade,
            "loss_defect": loss_defect,
            "loss_score": loss_score
        }
