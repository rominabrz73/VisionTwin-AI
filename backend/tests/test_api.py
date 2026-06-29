from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_rejects_non_image() -> None:
    response = TestClient(app).post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
