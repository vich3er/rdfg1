import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE = {
    "area": 7420,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 3,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["predicted_price"] > 0


def test_bigger_area_costs_more(client):
    small = client.post("/predict", json={**SAMPLE, "area": 3000}).json()
    big = client.post("/predict", json={**SAMPLE, "area": 12000}).json()
    assert big["predicted_price"] > small["predicted_price"]


def test_batch(client):
    r = client.post("/predict/batch", json={"items": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_validation_error(client):
    bad = {**SAMPLE, "mainroad": "maybe"}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "area"}
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "test_metrics" in r.json()
