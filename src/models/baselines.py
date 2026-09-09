"""Classical feature-based Image Quality Assessment (IQA) and single-task deep baselines."""

from typing import Dict, List, Union, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import laplace, sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from .backbones import create_backbone


class ClassicalImageQualityExtractor:
    """Extract hand-engineered physical and statistical image quality attributes."""

    @staticmethod
    def extract_features(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extract a 10-dimensional image quality descriptor vector.

        Features:
        1. Laplacian Variance (Focus / Sharpness)
        2. Tenengrad Gradient Energy (Edge definition)
        3. Shannon Gray-level Entropy (Information content)
        4. RMS Contrast (Dynamic range)
        5. Mean Luminance (Overall illumination)
        6. Underexposure Ratio (Pixels < 15)
        7. Overexposure Ratio (Pixels > 240)
        8. Red/Green Chromatic Ratio
        9. Spatial Uniformity Index
        10. Circular Mask Area Ratio
        """
        if isinstance(image, Image.Image):
            img_arr = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            img_arr = np.array(image, dtype=np.float32)

        # Grayscale luminance
        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]

        # 1. Laplacian Variance
        lap = laplace(gray)
        lap_var = float(np.var(lap))

        # 2. Tenengrad Gradient
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        tenengrad = float(np.mean(gx ** 2 + gy ** 2))

        # 3. Shannon Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        # 4. RMS Contrast
        rms_contrast = float(np.std(gray))

        # 5. Mean Luminance
        mean_lum = float(np.mean(gray))

        # 6 & 7. Exposure Ratios
        total_pixels = max(1, gray.size)
        underexposed = float(np.sum(gray < 15.0) / total_pixels)
        overexposed = float(np.sum(gray > 240.0) / total_pixels)

        # 8. Red/Green Ratio
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1])) + 1e-5
        rg_ratio = mean_r / mean_g

        # 9. Spatial Uniformity (Quadrant standard deviation)
        h, w = gray.shape
        quads = [
            gray[:h//2, :w//2].mean(),
            gray[:h//2, w//2:].mean(),
            gray[h//2:, :w//2].mean(),
            gray[h//2:, w//2:].mean()
        ]
        uniformity = float(np.std(quads))

        # 10. Mask Area Ratio (Foreground ratio)
        mask_ratio = float(np.sum(gray > 20.0) / total_pixels)

        features = np.array([
            lap_var, tenengrad, entropy, rms_contrast, mean_lum,
            underexposed, overexposed, rg_ratio, uniformity, mask_ratio
        ], dtype=np.float32)

        return features


class ClassicalQualityClassifier:
    """Scikit-Learn baseline trained on hand-engineered physical features."""

    def __init__(self, classifier_type: str = "random_forest", seed: int = 42):
        self.classifier_type = classifier_type
        if classifier_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=seed)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
        self.extractor = ClassicalImageQualityExtractor()

    def fit(self, images: List[Union[Image.Image, np.ndarray]], labels: np.ndarray):
        X = np.stack([self.extractor.extract_features(img) for img in images])
        self.model.fit(X, labels)

    def predict(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        X = np.stack([self.extractor.extract_features(img) for img in images])
        return self.model.predict(X)

    def predict_proba(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        X = np.stack([self.extractor.extract_features(img) for img in images])
        return self.model.predict_proba(X)


class SingleTaskQualityClassifier(nn.Module):
    """Standard single-task deep CNN baseline (Quality Grade only)."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone, in_features = create_backbone(backbone_name, pretrained=pretrained, drop_rate=dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
