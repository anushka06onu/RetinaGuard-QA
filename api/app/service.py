"""API Service Layer managing model lifecycles and image analysis."""

from pathlib import Path
from typing import Optional

from PIL import Image

from retinaguard.inference.predictor import RetinaGuardPredictor
from retinaguard.inference.schemas import PredictionResponse

from .config import Settings, get_settings


class QualityAssessmentService:
    """Service layer coordinating preprocessing, ONNX Runtime, and decision engine."""

    def __init__(self, settings: Optional[Settings] = None, image_size: int = 384):
        self.settings = settings or get_settings()
        p = Path(self.settings.model_path)

        if not p.is_file():
            if self.settings.test_mode:
                from retinaguard.models.multitask import RetinaGuardMultiTaskModel

                model = RetinaGuardMultiTaskModel(pretrained=False)
                model.eval()
                self.predictor = RetinaGuardPredictor(
                    model_path=None,
                    preprocessing_config_path=self.settings.preprocessing_path,
                    calibration_config_path=self.settings.calibration_path,
                    image_size=image_size,
                    allow_test_fallback=self.settings.is_test,
                )
                self.predictor.pt_model = model
            else:
                self.predictor = RetinaGuardPredictor(
                    model_path=None,
                    preprocessing_config_path=self.settings.preprocessing_path,
                    calibration_config_path=self.settings.calibration_path,
                    image_size=image_size,
                    allow_test_fallback=self.settings.is_test,
                )
        else:
            self.predictor = RetinaGuardPredictor(
                model_path=self.settings.model_path,
                preprocessing_config_path=self.settings.preprocessing_path,
                calibration_config_path=self.settings.calibration_path,
                image_size=image_size,
                allow_test_fallback=self.settings.is_test,
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
