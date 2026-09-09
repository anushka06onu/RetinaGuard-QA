"""Backbone feature extractors with lightweight CPU and edge deployment support."""

from typing import Tuple
import torch
import torch.nn as nn
import timm


def create_backbone(
    backbone_name: str = "mobilenetv3_large_100", 
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> Tuple[nn.Module, int]:
    """Create a timm backbone with pooled feature extraction (num_classes=0).

    Args:
        backbone_name: Architecture identifier.
        pretrained: Whether to load ImageNet pre-trained weights.
        drop_rate: Dropout rate.

    Returns:
        Tuple of (nn.Module, feature_dimension).
    """
    try:
        model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
    except Exception:
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate
        )

    # Dynamic exact feature dimension calculation
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        feature_dim = out.shape[-1]

    return model, feature_dim


def get_feature_dim(backbone_name: str) -> int:
    """Return the output feature embedding dimension for the given backbone."""
    _, dim = create_backbone(backbone_name, pretrained=False)
    return dim
