"""Baseline models per Phase 7 of blueprint."""

from typing import List, Union, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import laplace, sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import timm


class ClassicalFeatureExtractor:
    """Extracts non-learning physical and statistical image quality attributes."""

    @staticmethod
    def extract_features(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extract a 10-dimensional physical quality descriptor vector.

        Features:
        1. Laplacian Variance (sharpness/focus)
        2. Tenengrad Gradient Energy
        3. Shannon Entropy
        4. RMS Contrast
        5. Mean Luminance
        6. Underexposed Pixel Proportion (< 15)
        7. Overexposed Pixel Proportion (> 240)
        8. Red/Green Chromatic Ratio
        9. Spatial Uniformity Index
        10. Circular Foreground Mask Area Ratio
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            arr = np.array(image, dtype=np.float32)

        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        lap = laplace(gray)
        lap_var = float(np.var(lap))

        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        tenengrad = float(np.mean(gx ** 2 + gy ** 2))

        hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        rms_contrast = float(np.std(gray))
        mean_lum = float(np.mean(gray))

        tot = max(1, gray.size)
        under = float(np.sum(gray < 15.0) / tot)
        over = float(np.sum(gray > 240.0) / tot)

        mean_r = float(np.mean(arr[:, :, 0]))
        mean_g = float(np.mean(arr[:, :, 1])) + 1e-5
        rg_ratio = mean_r / mean_g

        h, w = gray.shape
        quads = [
            gray[:h//2, :w//2].mean(),
            gray[:h//2, w//2:].mean(),
            gray[h//2:, :w//2].mean(),
            gray[h//2:, w//2:].mean()
        ]
        uniformity = float(np.std(quads))
        mask_ratio = float(np.sum(gray > 20.0) / tot)

        return np.array([
            lap_var, tenengrad, entropy, rms_contrast, mean_lum,
            under, over, rg_ratio, uniformity, mask_ratio
        ], dtype=np.float32)


class ClassicalQualityModel:
    """Classical baseline trained on physical feature vectors."""

    def __init__(self, model_type: str = "random_forest", seed: int = 2026):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.clf = LogisticRegression(max_iter=1000, random_state=seed)
        else:
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
        self.extractor = ClassicalFeatureExtractor()

    def fit(self, images: List[Union[Image.Image, np.ndarray]], labels: np.ndarray):
        X = np.stack([self.extractor.extract_features(img) for img in images])
        self.clf.fit(X, labels)

    def predict(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        X = np.stack([self.extractor.extract_features(img) for img in images])
        return self.clf.predict(X)

    def predict_proba(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        X = np.stack([self.extractor.extract_features(img) for img in images])
        return self.clf.predict_proba(X)


class SingleTaskQualityModel(nn.Module):
    """Deep CNN single-task baseline (MobileNetV3-Small or EfficientNet-B0)."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=dropout)
        except Exception:
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=dropout)

        # Dynamic feature dim check
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
