import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodynamics.api.rest import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "lif" in data["models"]
    assert "hodgkin_huxley" in data["models"]


def test_simulate_lif():
    response = client.post("/simulate", json={
        "model": "lif",
        "duration": 300.0,
        "current": 3.0,
        "dt": 0.1,
    })
    assert response.status_code == 200
    data = response.json()
    assert "n_spikes" in data
    assert data["n_spikes"] >= 0
    assert "spike_times" in data


def test_list_experiments():
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert "n_back" in data["experiments"]


def test_analyze_entropy():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 100).tolist()
    response = client.post("/analyze", json={
        "data": data,
        "analysis": "entropy",
    })
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert isinstance(result["result"], (int, float))
    assert result["result"] > 0
