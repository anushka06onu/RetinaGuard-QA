"""API Service Layer managing model lifecycles and image analysis."""

import os
from pathlib import Path
from typing import Optional

from PIL import Image

from src.retinaguard.inference.predictor import RetinaGuardPredictor
from src.retinaguard.inference.schemas import PredictionResponse


class QualityAssessmentService:
    """Service layer coordinating preprocessing, ONNX Runtime, and decision engine."""

    def __init__(self, model_path: str = "artifacts/models/model.onnx", image_size: int = 384):
        p = Path(model_path)
        is_test_mode = os.environ.get("TEST_MODE", "0") == "1"

        if not p.is_file():
            if is_test_mode:
                # In test mode without exported ONNX, instantiate predictor with PyTorch model in eval mode
                from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel

                model = RetinaGuardMultiTaskModel(pretrained=False)
                model.eval()
                self.predictor = RetinaGuardPredictor(model_path=None, image_size=image_size)
                self.predictor.pt_model = model
            else:
                self.predictor = RetinaGuardPredictor(model_path=None, image_size=image_size)
        else:
            self.predictor = RetinaGuardPredictor(model_path=model_path, image_size=image_size)

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
