"""Typed application settings and environment variable management per Items 31 & 32."""

import os
from dataclasses import dataclass, field
from typing import List


def _get_validated_max_upload_size() -> int:
    val_str = os.environ.get("MAX_UPLOAD_SIZE_BYTES", "15728640")
    try:
        val = int(val_str)
    except (ValueError, TypeError):
        return 15728640
    # Enforce safe bounds: 1 MB to 100 MB
    if val < 1024 * 1024 or val > 100 * 1024 * 1024:
        return 15728640
    return val


@dataclass
class Settings:
    """Central authoritative application settings reading unified environment variables."""

    app_mode: str = field(default_factory=lambda: os.environ.get("APP_MODE", "development").lower())
    model_path: str = field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "artifacts/models/model.onnx")
    )
    preprocessing_path: str = field(
        default_factory=lambda: os.environ.get(
            "PREPROCESSING_PATH", "artifacts/models/preprocessing.json"
        )
    )
    calibration_path: str = field(
        default_factory=lambda: os.environ.get(
            "CALIBRATION_PATH", "artifacts/models/calibration_metadata.json"
        )
    )
    cors_origins_raw: str = field(
        default_factory=lambda: os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
        )
    )
    max_upload_size_bytes: int = field(default_factory=_get_validated_max_upload_size)
    test_mode: bool = field(default_factory=lambda: os.environ.get("TEST_MODE", "0") == "1")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_mode == "production" and not self.test_mode

    @property
    def is_test(self) -> bool:
        return self.test_mode or self.app_mode == "test"

    @property
    def is_development(self) -> bool:
        return self.app_mode == "development" and not self.test_mode


def get_settings() -> Settings:
    return Settings()
