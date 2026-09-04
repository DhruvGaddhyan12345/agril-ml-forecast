from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_contract():
    response = TestClient(app).post("/predict", json={
        "date": "2025-01-01", "farm_id": 1, "crop": "maize",
        "rainfall_mm": 24, "temperature_c": 22, "soil_moisture": 0.4,
        "ndvi": 0.6, "fertilizer_kg_ha": 100,
    })
    assert response.status_code == 200
    assert response.json()["predicted_yield"] > 0
    assert response.json()["model_version"] == "production"


def test_prediction_rejects_invalid_values():
    response = TestClient(app).post("/predict", json={
        "date": "2025-01-01", "farm_id": 0, "crop": "maize",
        "rainfall_mm": 24, "temperature_c": 22, "soil_moisture": 0.4,
        "ndvi": 0.6, "fertilizer_kg_ha": 100,
    })
    assert response.status_code == 422
