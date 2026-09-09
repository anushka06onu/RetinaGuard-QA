"""RetinaGuard-QA FastAPI Backend Application."""

import io
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.inference.engine import RetinaGuardInferenceEngine
from src.inference.decision_engine import DecisionEngine
from .schemas import QualityAssessmentResponse, HealthCheckResponse
from .report_generator import generate_json_report, generate_pdf_report


app = FastAPI(
    title="RetinaGuard-QA API",
    description="Uncertainty-aware, device-robust quality assurance and capture-feedback system for retinal fundus imaging.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
MODEL_ONNX_PATH = Path("results/models/retinaguard_net.onnx")
MODEL_PT_PATH = Path("results/models/retinaguard_net.pt")

engine = RetinaGuardInferenceEngine()
if MODEL_ONNX_PATH.exists():
    try:
        engine.load_model(MODEL_ONNX_PATH, use_onnx=True)
    except Exception:
        pass
elif MODEL_PT_PATH.exists():
    try:
        engine.load_model(MODEL_PT_PATH, use_onnx=False)
    except Exception:
        pass


@app.get("/api/v1/health", response_model=HealthCheckResponse)
def health_check():
    return {
        "status": "online",
        "system": "RetinaGuard-QA",
        "version": "1.0.0",
        "onnx_available": bool(engine and engine.ort_session is not None),
        "runtime_device": "cpu"
    }


@app.post("/api/v1/inspect", response_model=QualityAssessmentResponse)
async def inspect_fundus_image(file: UploadFile = File(...)):
    """Analyze an uploaded retinal fundus image and return structured quality assessment."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, TIFF, BMP).")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    result = engine.predict(pil_img)
    return result.to_dict()


@app.post("/api/v1/report/pdf")
async def export_pdf_report(file: UploadFile = File(...)):
    """Analyze image and stream back a clinical-grade PDF audit report."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    result = engine.predict(pil_img)
    pdf_bytes = generate_pdf_report(result.to_dict(), filename=file.filename or "retinal_image.jpg")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=RetinaGuard_Audit_{Path(file.filename or 'image').stem}.pdf"}
    )


@app.post("/api/v1/report/json")
async def export_json_report(file: UploadFile = File(...)):
    """Analyze image and return downloadable JSON audit log."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    result = engine.predict(pil_img)
    json_str = generate_json_report(result.to_dict(), filename=file.filename or "retinal_image.jpg")

    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=RetinaGuard_Audit_{Path(file.filename or 'image').stem}.json"}
    )


# Mount static frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>RetinaGuard-QA UI Loading...</h1>")


def start():
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
