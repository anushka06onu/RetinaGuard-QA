"""Masked multi-task loss functions per Phase 8 of blueprint."""

from typing import Dict

import torch
from torch import nn


class MaskedMultiTaskLoss(nn.Module):
    """Masked Multi-Task Loss:
    L = lambda_q * m_q * L_q + lambda_a * m_a * L_a + lambda_c * m_c * L_c + lambda_f * m_f * L_f
    """

    def __init__(
        self,
        weight_quality: float = 1.0,
        weight_overall_quality: float = 0.8,
        weight_artifact: float = 0.5,
        weight_clarity: float = 0.5,
        weight_field_def: float = 0.5,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.w_q = weight_quality
        self.w_oq = weight_overall_quality
        self.w_art = weight_artifact
        self.w_cla = weight_clarity
        self.w_fld = weight_field_def

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(
        self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = preds["quality_logits"].device

        # Helper to calculate masked mean loss
        def calc_masked_loss(logits, targets, mask):
            if mask is None or mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            losses = self.ce(logits, targets)
            masked = (losses * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            return masked

        l_q = calc_masked_loss(
            preds["quality_logits"],
            batch.get("quality_target", torch.zeros(1, dtype=torch.long, device=device)),
            batch.get("quality_mask", torch.ones(1, device=device)),
        )

        l_art = calc_masked_loss(
            preds["artifact_logits"],
            batch.get("artifact_target", torch.zeros(1, dtype=torch.long, device=device)),
            batch.get("artifact_mask", torch.zeros(1, device=device)),
        )

        l_cla = calc_masked_loss(
            preds["clarity_logits"],
            batch.get("clarity_target", torch.zeros(1, dtype=torch.long, device=device)),
            batch.get("clarity_mask", torch.zeros(1, device=device)),
        )

        l_fld = calc_masked_loss(
            preds["field_definition_logits"],
            batch.get("field_definition_target", torch.zeros(1, dtype=torch.long, device=device)),
            batch.get("field_definition_mask", torch.zeros(1, device=device)),
        )

        total_loss = self.w_q * l_q + self.w_art * l_art + self.w_cla * l_cla + self.w_fld * l_fld

        return {
            "loss_total": total_loss,
            "loss_quality": l_q,
            "loss_artifact": l_art,
            "loss_clarity": l_cla,
            "loss_field_def": l_fld,
        }
