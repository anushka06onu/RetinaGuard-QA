"""FastAPI application matching Phase 18 & Phase 24 of blueprint."""

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from retinaguard.inference.schemas import PredictionResponse

from .config import get_settings
from .service import QualityAssessmentService, get_service

logger = logging.getLogger("retinaguard.api")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_DIMENSION = 8192
MIN_DIMENSION = 32

# Initialize decompression bomb limit once at module level (Item 40)
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook strictly checking production artifact integrity."""
    settings = get_settings()

    if settings.is_production:
        onnx_path = Path(settings.model_path)
        preproc_path = Path(settings.preprocessing_path)
        calib_path = Path(settings.calibration_path)

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

        import json

        with open(calib_path, "r", encoding="utf-8") as f:
            calib_meta = json.load(f)

        if calib_meta.get("status") != "completed":
            err = (
                f"Production startup halted: Calibration metadata status is '{calib_meta.get('status')}'. "
                "Production requires completed final model metadata (status='completed')."
            )
            logger.critical(err)
            raise RuntimeError(err)

        if calib_meta.get("eligible_as_final_result") is not True:
            err = (
                "Production startup halted: Model metadata is marked not eligible for production inference "
                "(eligible_as_final_result != True)."
            )
            logger.critical(err)
            raise RuntimeError(err)

        service = get_service()
        if service.predictor.ort_session is None and service.predictor.pt_model is None:
            raise RuntimeError(
                "Production startup failed: Inference model session could not be initialized."
            )
        logger.info("RetinaGuard-QA API production verification passed.")
    elif settings.app_mode == "development":
        logger.info(
            "Development mode: model artifact unavailable; inference endpoints will remain unavailable."
        )

    yield


app = FastAPI(
    title="RetinaGuard-QA API",
    description="Retinal image-quality research prototype with planned cross-domain robustness evaluation.",
    version="0.2.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health/live")
def liveness_check() -> Dict[str, Any]:
    """Basic liveness probe confirming server process is running."""
    return {"status": "live", "service": "RetinaGuard-QA API", "version": "0.2.0"}


@app.get("/health/ready")
def readiness_check(
    response: Response, service: QualityAssessmentService = Depends(get_service)
) -> Dict[str, Any]:
    """Readiness probe verifying model session, preprocessing, calibration, and artifact status."""
    p = service.predictor
    model_loaded = bool(p.ort_session is not None or p.pt_model is not None)
    preproc_loaded = bool(p.preprocessing_metadata_loaded)
    calib_loaded = bool(p.calibration_metadata_loaded)
    status_valid = bool(p.artifact_status_valid)
    hash_verified = bool(p.model_hash_verified or not settings.is_production)
    arch_compatible = bool(p.model_architecture_compatible)

    all_ready = (
        model_loaded
        and preproc_loaded
        and calib_loaded
        and status_valid
        and hash_verified
        and arch_compatible
    )

    checks = {
        "model_loaded": model_loaded,
        "preprocessing_metadata_loaded": preproc_loaded,
        "calibration_metadata_loaded": calib_loaded,
        "model_hash_verified": hash_verified,
        "artifact_status_valid": status_valid,
        "model_architecture_compatible": arch_compatible,
    }

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "checks": checks,
            "version": "0.2.0",
        }

    return {
        "status": "ready",
        "checks": checks,
        "runtime_engine": (
            "onnxruntime_cpu" if p.ort_session else ("pytorch_cpu" if p.pt_model else "none")
        ),
        "version": "0.2.0",
    }


@app.get("/health")
def health_check_legacy(
    response: Response, service: QualityAssessmentService = Depends(get_service)
) -> Dict[str, Any]:
    """Legacy health endpoint combining liveness and loaded status."""
    return readiness_check(response=response, service=service)


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """System metadata, intended input, quality classes, and clinical boundaries."""
    return {
        "system_name": "RetinaGuard-QA",
        "version": "0.2.0",
        "status": "Research Prototype",
        "intended_input": "Color Retinal Fundus Photograph (Standard 45/50 degree FOV)",
        "supported_formats": ["JPEG", "PNG"],
        "quality_classes": ["good", "usable", "reject"],
        "quality_attributes": ["artifact", "clarity", "field_definition"],
        "triage_decisions": ["accept", "recapture", "manual_review", "unsupported_input"],
        "clinical_disclaimer": (
            "Model-assessed technical quality for research purposes only. "
            "Not clinically validated. Not intended for direct diagnostic disease grading."
        ),
    }


async def _process_image_upload(
    file: UploadFile, service: QualityAssessmentService
) -> PredictionResponse:
    """Internal handler for processing uploaded fundus image."""
    req_id = str(uuid.uuid4())[:8]

    # 1. Check content-type header if present
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Please provide an image (JPEG or PNG).",
        )

    # 2. Read buffer with size safeguard & reject empty uploads (Item 40)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty upload received. Please provide a valid JPEG or PNG image file.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum permitted size is 15MB."
        )

    # 2. Content decoding and image validation
    try:
        raw_img = Image.open(io.BytesIO(contents))
        fmt = raw_img.format
        if fmt not in ["JPEG", "PNG"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format: {fmt}. Please provide JPEG or PNG."
            )
        raw_img.verify()

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({w}x{h}) are too small (minimum {MIN_DIMENSION}x{MIN_DIMENSION}).",
            )
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({w}x{h}) exceed maximum permitted size ({MAX_DIMENSION}x{MAX_DIMENSION}).",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[{req_id}] Image verification failure: {e}")
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

    # 3. Execute transient analysis without persisting uploaded bytes
    try:
        return service.analyze_image(pil_img)
    except Exception as e:
        logger.error(f"[{req_id}] Quality prediction runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image quality assurance processing error [Request ID: {req_id}]",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_quality(
    file: UploadFile = File(...), service: QualityAssessmentService = Depends(get_service)
) -> PredictionResponse:
    """Analyze single uploaded fundus photograph."""
    return await _process_image_upload(file, service)


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict_quality(
    file: UploadFile = File(...), service: QualityAssessmentService = Depends(get_service)
) -> PredictionResponse:
    """Analyze single uploaded fundus photograph (canonical proxy route)."""
    return await _process_image_upload(file, service)
