import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from pathlib import Path

from src.service.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_no_file(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_invalid_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_with_image(client):
    img = Image.new("RGB", (200, 200), color="gray")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "probabilities" in data
