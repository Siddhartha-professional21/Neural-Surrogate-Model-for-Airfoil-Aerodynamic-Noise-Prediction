
import json
import time
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from tensorflow import keras

import features

app = FastAPI(title="Airfoil Self-Noise Prediction API")

# load the saved model, scaler and column order once, when the server starts
model = keras.models.load_model("models/airfoil_deep_nn.keras")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

info_file = open("models/model_info.json")
model_info = json.load(info_file)
info_file.close()

print("Loaded model, scaler and column order")

# the range each input had in the training data
training_ranges = {
    "frequency": [200.0, 20000.0],
    "angle_of_attack": [0.0, 22.2],
    "chord_length": [0.0254, 0.3048],
    "velocity": [31.7, 71.3],
    "displacement_thickness": [0.0004, 0.0584]
}


class AirfoilInput(BaseModel):
    frequency: float = 1000.0
    angle_of_attack: float = 5.0
    chord_length: float = 0.1524
    velocity: float = 55.5
    displacement_thickness: float = 0.005


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "inputs": feature_order}


@app.get("/model")
def model_details():
    return model_info


@app.post("/predict")
def predict(request: AirfoilInput):
    readings = {}
    readings["frequency"] = request.frequency
    readings["angle_of_attack"] = request.angle_of_attack
    readings["chord_length"] = request.chord_length
    readings["velocity"] = request.velocity
    readings["displacement_thickness"] = request.displacement_thickness

    # the log needs positive numbers
    if readings["frequency"] <= 0:
        raise HTTPException(status_code=400, detail="frequency must be greater than 0")

    if readings["displacement_thickness"] <= 0:
        raise HTTPException(status_code=400,
                            detail="displacement_thickness must be greater than 0")

    # warn if a value is outside the range the model was trained on
    warnings = []
    for name in feature_order:
        low = training_ranges[name][0]
        high = training_ranges[name][1]
        if readings[name] < low or readings[name] > high:
            warnings.append(name + " is outside the training range " +
                            str(low) + " to " + str(high))

    start = time.time()

    X_one = features.prepare_one_input(readings)
    X_scaled = scaler.transform(X_one)
    prediction = model.predict(X_scaled, verbose=0)[0][0]

    end = time.time()

    result = {}
    result["predicted_sound_pressure_db"] = round(float(prediction), 3)
    result["model"] = model_info["model"]
    result["test_rmse_db"] = round(model_info["rmse"], 3)
    result["warnings"] = warnings
    result["latency_ms"] = round((end - start) * 1000, 2)
    return result
