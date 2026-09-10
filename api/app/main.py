"""FastAPI application matching Phase 18 & Phase 24 of blueprint."""

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from retinaguard.inference.schemas import PredictionResponse

from .service import QualityAssessmentService, get_service

logger = logging.getLogger("retinaguard.api")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_DIMENSION = 8192
MIN_DIMENSION = 32


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook strictly checking production artifact integrity."""
    app_mode = os.environ.get("APP_MODE", "development").lower()
    is_test_mode = os.environ.get("TEST_MODE", "0") == "1"

    if app_mode == "production" and not is_test_mode:
        onnx_path = Path(os.environ.get("MODEL_PATH", "artifacts/models/model.onnx"))
        preproc_path = Path(
            os.environ.get("PREPROCESSING_PATH", "artifacts/models/preprocessing.json")
        )
        calib_path = Path(
            os.environ.get("CALIBRATION_PATH", "artifacts/models/calibration_metadata.json")
        )

        missing = []
        if not onnx_path.is_file():
            missing.append(f"ONNX Model ({onnx_path})")
        if not preproc_path.is_file():
            missing.append(f"Preprocessing Metadata ({preproc_path})")
        if not calib_path.is_file():
            missing.append(f"Calibration Metadata ({calib_path})")

        if missing:
            err = (
                f"Strict production startup halted: Required production artifacts missing: "
                f"{', '.join(missing)}.\n"
                "To run in development mode, set APP_MODE=development."
            )
            logger.critical(err)
            raise RuntimeError(err)

        service = get_service()
        if service.predictor.ort_session is None and service.predictor.pt_model is None:
            raise RuntimeError(
                "Production startup failed: Inference model session could not be initialized."
            )
        logger.info("RetinaGuard-QA API production verification passed.")
    elif app_mode == "development":
        logger.info(
            "RetinaGuard-QA API running in development mode (un-trained fallback enabled if models missing)."
        )

    yield


app = FastAPI(
    title="RetinaGuard-QA API",
    description="Uncertainty-aware and device-robust quality control for retinal fundus images.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration from environment or defaults
cors_origins_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
)
allowed_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check(service: QualityAssessmentService = Depends(get_service)) -> Dict[str, Any]:
    """Liveness check and model loaded state."""
    model_loaded = bool(
        service.predictor.ort_session is not None or service.predictor.pt_model is not None
    )
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "runtime_engine": (
            "onnxruntime_cpu"
            if service.predictor.ort_session
            else ("pytorch_cpu" if service.predictor.pt_model else "none")
        ),
        "version": "1.0.0",
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """System metadata, intended input, quality classes, and clinical boundaries."""
    return {
        "system_name": "RetinaGuard-QA",
        "version": "1.0.0",
        "intended_input": "Color Retinal Fundus Photograph (Standard 45/50 degree FOV)",
        "supported_formats": ["JPEG", "PNG"],
        "quality_classes": ["good", "usable", "reject"],
        "quality_attributes": ["artifact", "clarity", "field_definition"],
        "triage_decisions": ["accept", "recapture", "manual_review", "unsupported_input"],
        "clinical_disclaimer": "Technical image-quality assessment only; not a diagnosis or clinical recommendation.",
        "prohibited_use": "Direct diagnostic disease grading or automated treatment dispatch without clinical oversight.",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_quality(
    file: UploadFile = File(...), service: QualityAssessmentService = Depends(get_service)
) -> PredictionResponse:
    """Analyze single uploaded fundus photograph."""
    req_id = str(uuid.uuid4())[:8]

    # 1. Read buffer with size safeguard
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum permitted size is 15MB."
        )

    # 2. Content decoding and decompression bomb prevention
    try:
        Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION
        raw_img = Image.open(io.BytesIO(contents))
        fmt = raw_img.format
        if fmt not in ["JPEG", "PNG", "MPO"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format: {fmt}. Please provide JPEG or PNG."
            )
        raw_img.verify()

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            raise HTTPException(
                status_code=400, detail=f"Image dimensions ({w}x{h}) are too small."
            )
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            raise HTTPException(
                status_code=400, detail=f"Image dimensions ({w}x{h}) exceed maximum permitted size."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[{req_id}] Image verification failure: {e}")
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

    # 3. Execute transient analysis without persisting uploaded bytes
    try:
        result = service.analyze_image(pil_img)
        return result
    except Exception as e:
        logger.error(f"[{req_id}] Quality prediction runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image quality assurance processing error [Request ID: {req_id}]",
        )
