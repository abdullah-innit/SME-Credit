"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /api/v1/health should return 200 with status ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "offline_mode" in data


def test_health_contains_version() -> None:
    """Health response must include the current app version."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert data["version"] == "0.1.0"
