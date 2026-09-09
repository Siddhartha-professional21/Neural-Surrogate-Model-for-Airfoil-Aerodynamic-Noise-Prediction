
import json
from fastapi.testclient import TestClient

import app

client = TestClient(app.app)

sample_file = open("data/sample_input.json")
sample = json.load(sample_file)
sample_file.close()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert len(response.json()["inputs"]) == 5


def test_model_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json()["model"] == "deep_nn"


def test_prediction_matches_the_one_saved_during_training():
    response = client.post("/predict", json=sample["readings"])
    assert response.status_code == 200
    prediction = response.json()["predicted_sound_pressure_db"]
    assert abs(prediction - sample["expected_prediction"]) < 0.5


def test_prediction_is_close_to_the_true_value():
    response = client.post("/predict", json=sample["readings"])
    prediction = response.json()["predicted_sound_pressure_db"]
    assert abs(prediction - sample["true_sound_pressure"]) < 8


def test_prediction_works_with_the_default_values():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert 90 < response.json()["predicted_sound_pressure_db"] < 150


def test_zero_frequency_is_rejected():
    response = client.post("/predict", json={"frequency": 0})
    assert response.status_code == 400


def test_negative_thickness_is_rejected():
    response = client.post("/predict", json={"displacement_thickness": -0.01})
    assert response.status_code == 400


def test_value_outside_the_training_range_gives_a_warning():
    response = client.post("/predict", json={"frequency": 50000})
    assert response.status_code == 200
    assert len(response.json()["warnings"]) > 0


def test_text_instead_of_a_number_is_rejected():
    response = client.post("/predict", json={"frequency": "abc"})
    assert response.status_code == 422
