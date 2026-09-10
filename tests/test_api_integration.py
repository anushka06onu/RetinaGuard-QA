"""API integration tests for FastAPI application."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import app


def create_test_image_bytes(format="JPEG", color=(180, 50, 20), size=(200, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_live():
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "live"


def test_health_ready():
    client = TestClient(app)
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_predict_empty_file():
    client = TestClient(app)
    response = client.post("/api/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert "Empty upload" in response.json()["detail"]


def test_predict_unsupported_media_type():
    client = TestClient(app)
    response = client.post("/api/predict", files={"file": ("doc.txt", b"plain text", "text/plain")})
    assert response.status_code == 415


def test_predict_both_routes_work():
    client = TestClient(app)
    img_bytes = create_test_image_bytes()

    # Test POST /api/predict
    resp1 = client.post("/api/predict", files={"file": ("fundus.jpg", img_bytes, "image/jpeg")})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "quality" in data1
    assert "decision" in data1
    assert "probabilities" in data1
    assert "quality_attributes" in data1
    assert "disclaimer" in data1

    # Test POST /predict (legacy route)
    resp2 = client.post("/predict", files={"file": ("fundus.jpg", img_bytes, "image/jpeg")})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["quality"] == data1["quality"]
