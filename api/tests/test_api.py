"""API endpoint and safeguard unit tests per Phase 26 of blueprint."""

import io
import os

from PIL import Image

# Configure TEST_MODE before importing app
os.environ["TEST_MODE"] = "1"
from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "runtime_engine" in data


def test_api_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "intended_input" in data
    assert "quality_classes" in data
    assert "clinical_disclaimer" in data


def test_api_predict_valid_jpeg():
    img = Image.new("RGB", (200, 200), color=(180, 80, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post("/predict", files={"file": ("fundus_test.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert "quality" in data
    assert "probabilities" in data
    assert "calibrated_confidence" in data
    assert "uncertainty" in data
    assert "decision" in data
    assert "disclaimer" in data


def test_api_predict_corrupt_file():
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_DATA"
    res = client.post(
        "/predict", files={"file": ("bad_image.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}
    )
    assert res.status_code == 400
    assert "Invalid or unreadable" in res.json()["detail"]


def test_api_predict_grayscale_ood():
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post("/predict", files={"file": ("grayscale_xray.png", buf, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] in ["unsupported_input", "manual_review"]


def test_api_production_lifespan_failure_without_models(monkeypatch):
    import pytest

    monkeypatch.setenv("APP_MODE", "production")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("MODEL_PATH", "nonexistent/model.onnx")

    with pytest.raises(RuntimeError, match="Strict production startup halted"):
        with TestClient(app):
            pass
