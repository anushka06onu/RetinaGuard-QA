"""API Service Layer managing model lifecycles and image analysis."""

from pathlib import Path
from typing import Optional

from PIL import Image

from src.retinaguard.inference.predictor import RetinaGuardPredictor
from src.retinaguard.inference.schemas import PredictionResponse


class QualityAssessmentService:
    """Service layer coordinating preprocessing, ONNX Runtime, and decision engine."""

    def __init__(self, model_path: str = "artifacts/models/model.onnx", image_size: int = 384):
        self.predictor = RetinaGuardPredictor(
            model_path=model_path if Path(model_path).exists() else None, image_size=image_size
        )

    def analyze_image(self, image: Image.Image) -> PredictionResponse:
        """Run transient quality assurance analysis."""
        return self.predictor.predict(image)


# Singleton instance
service_instance: Optional[QualityAssessmentService] = None


def get_service() -> QualityAssessmentService:
    global service_instance
    if service_instance is None:
        service_instance = QualityAssessmentService()
    return service_instance
