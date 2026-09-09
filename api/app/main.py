"""FastAPI application matching Phase 18 & Phase 24 of blueprint."""

import io
from pathlib import Path
from typing import Dict, Any
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.retinaguard.inference.schemas import PredictionResponse
from .service import QualityAssessmentService, get_service

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_DIMENSION = 8192
MIN_DIMENSION = 32

app = FastAPI(
    title="RetinaGuard-QA API",
    description="Uncertainty-aware and device-robust quality control for retinal fundus images.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check(service: QualityAssessmentService = Depends(get_service)) -> Dict[str, Any]:
    """Liveness check and model loaded state."""
    model_loaded = bool(service.predictor.ort_session is not None or service.predictor.pt_model is not None)
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "runtime_engine": "onnxruntime_cpu" if service.predictor.ort_session else "pytorch_cpu",
        "version": "1.0.0"
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """System metadata, intended input, quality classes, and clinical boundaries."""
    return {
        "system_name": "RetinaGuard-QA",
        "version": "1.0.0",
        "intended_input": "Color Retinal Fundus Photograph (Standard 45/50 degree FOV)",
        "supported_formats": ["JPEG", "PNG", "TIFF"],
        "quality_classes": ["good", "usable", "reject"],
        "quality_attributes": ["artifact", "clarity", "field_definition"],
        "triage_decisions": ["accept", "recapture", "manual_review", "unsupported_input"],
        "clinical_disclaimer": "Technical image-quality assessment only; not a diagnosis or clinical recommendation.",
        "prohibited_use": "Direct diagnostic disease grading or automated treatment dispatch without clinical oversight."
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_quality(
    file: UploadFile = File(...),
    service: QualityAssessmentService = Depends(get_service)
) -> PredictionResponse:
    """Analyze single uploaded fundus photograph."""
    # 1. Read buffer with size safeguard
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum permitted size is 15MB.")

    # 2. Content decoding and decompression bomb prevention
    try:
        Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION
        raw_img = Image.open(io.BytesIO(contents))
        raw_img.verify() # Verify file header integrity

        # Re-open after verify to strip metadata and convert to RGB
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            raise HTTPException(status_code=400, detail=f"Image dimensions ({w}x{h}) are too small.")
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            raise HTTPException(status_code=400, detail=f"Image dimensions ({w}x{h}) exceed maximum permitted size.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image format: {str(e)}")

    # 3. Execute transient analysis without persisting uploaded bytes
    try:
        result = service.analyze_image(pil_img)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assurance analysis failed: {str(e)}")
