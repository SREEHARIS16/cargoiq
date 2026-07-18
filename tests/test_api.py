"""
Basic tests for the CargoIQ API.

Run:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_payload():
    payload = {
        "route_id": "BLR-DXB",
        "cargo_type": "pharma",
        "date": "2026-07-20",
        "distance_km": 2800,
        "capacity_tons": 46.2,
        "fuel_index": 102.3,
        "demand_lag_1": 28.4,
        "demand_lag_7": 27.1,
        "demand_roll_mean_7": 27.9,
        "demand_roll_mean_14": 27.3,
        "rate_lag_1": 0.85,
        "rate_roll_mean_7": 0.83,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_demand_tons" in body
    assert "predicted_rate_per_kg" in body
    assert body["predicted_demand_tons"] > 0


def test_predict_missing_field():
    response = client.post("/predict", json={"route_id": "BLR-DXB"})
    assert response.status_code == 422  # FastAPI validation error
