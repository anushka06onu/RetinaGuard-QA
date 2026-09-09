"""Integration tests for FastAPI endpoints."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["system"] == "RetinaGuard-QA"


def test_inspect_endpoint():
    img = Image.new("RGB", (100, 100), color=(180, 80, 20))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/inspect",
        files={"file": ("sample_fundus.jpg", buffer, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "quality_grade" in data
    assert "quality_score" in data
    assert "clinical_disclaimer" in data


def test_export_pdf_endpoint():
    img = Image.new("RGB", (100, 100), color=(180, 80, 20))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/report/pdf",
        files={"file": ("sample_fundus.jpg", buffer, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 100


def test_export_json_endpoint():
    img = Image.new("RGB", (100, 100), color=(180, 80, 20))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/report/json",
        files={"file": ("sample_fundus.jpg", buffer, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "report_metadata" in data
    assert "quality_assessment" in data
