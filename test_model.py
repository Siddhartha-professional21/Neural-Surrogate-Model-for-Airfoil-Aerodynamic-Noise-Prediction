
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow import keras

import features

test_data = pd.read_csv("data/test_sample.csv")

model = keras.models.load_model("models/airfoil_deep_nn.keras")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

info_file = open("models/model_info.json")
model_info = json.load(info_file)
info_file.close()

# the limits the model has to stay inside
max_rmse = 2.6
min_r2 = 0.88


def get_predictions():
    prepared = features.prepare_data(test_data)
    X = prepared[feature_order].values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, verbose=0).flatten()


def test_saved_files_match_each_other():
    assert feature_order == features.input_cols
    assert len(feature_order) == 5
    assert model.input_shape == (None, 5)
    assert scaler.n_features_in_ == 5


def test_rmse_is_below_the_limit():
    y = test_data[features.target_col].values
    rmse = np.sqrt(mean_squared_error(y, get_predictions()))
    print("RMSE:", round(rmse, 3))
    assert rmse <= max_rmse


def test_r2_is_above_the_limit():
    y = test_data[features.target_col].values
    r2 = r2_score(y, get_predictions())
    print("R2:", round(r2, 3))
    assert r2 >= min_r2


def test_model_beats_the_linear_baseline():
    y = test_data[features.target_col].values
    rmse = np.sqrt(mean_squared_error(y, get_predictions()))
    assert rmse < model_info["baseline_rmse"]
