
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import features

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("models", exist_ok=True)

print("Loading data")
df = features.load_data("data")
data = features.prepare_data(df)

X = data[features.input_cols].values
y = data[features.target_col].values

# 70 / 15 / 15 split
row_numbers = np.arange(len(data))
temp_rows, test_rows = train_test_split(row_numbers, test_size=0.15, random_state=42)
train_rows, val_rows = train_test_split(temp_rows, test_size=0.1765, random_state=42)

X_train = X[train_rows]
X_val = X[val_rows]
X_test = X[test_rows]

y_train = y[train_rows]
y_val = y[val_rows]
y_test = y[test_rows]

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# scale using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("airfoil_noise_surrogate")

results = {}


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(name)
    print("  RMSE :", round(rmse, 3))
    print("  MAE  :", round(mae, 3))
    print("  R2   :", round(r2, 3))

    results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
    return rmse, mae, r2


def build_deep_model():
    model = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


print("")
print("######## Linear Regression ########")
with mlflow.start_run(run_name="linear_regression"):
    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)
    lin_pred = lin_model.predict(X_test_scaled)
    rmse, mae, r2 = evaluate("Linear Regression", y_test, lin_pred)

    mlflow.log_param("model", "linear_regression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

baseline_rmse = results["Linear Regression"]["rmse"]

print("")
print("######## Random Forest ########")
with mlflow.start_run(run_name="random_forest"):
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rmse, mae, r2 = evaluate("Random Forest", y_test, rf_pred)

    mlflow.log_param("model", "random_forest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("improvement_vs_baseline", (baseline_rmse - rmse) / baseline_rmse * 100)

print("")
print("######## Shallow Neural Network ########")
with mlflow.start_run(run_name="shallow_nn"):
    shallow_model = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    shallow_model.compile(optimizer="adam", loss="mse")

    stop = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    shallow_model.fit(X_train_scaled, y_train,
                      validation_data=(X_val_scaled, y_val),
                      epochs=300, batch_size=32,
                      callbacks=[stop], verbose=0)

    shallow_pred = shallow_model.predict(X_test_scaled, verbose=0).flatten()
    rmse, mae, r2 = evaluate("Shallow NN", y_test, shallow_pred)

    mlflow.log_param("model", "shallow_nn")
    mlflow.log_param("layers", "32")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("improvement_vs_baseline", (baseline_rmse - rmse) / baseline_rmse * 100)

print("")
print("######## Deep Neural Network ########")
with mlflow.start_run(run_name="deep_nn"):
    deep_model = build_deep_model()

    stop = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15)

    history = deep_model.fit(X_train_scaled, y_train,
                             validation_data=(X_val_scaled, y_val),
                             epochs=500, batch_size=32,
                             callbacks=[stop, reduce_lr], verbose=0)

    deep_pred = deep_model.predict(X_test_scaled, verbose=0).flatten()
    rmse, mae, r2 = evaluate("Deep NN", y_test, deep_pred)

    # 5 fold cross validation on the training and validation data together
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fold = 1

    for fold_train, fold_val in kfold.split(X_full):
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_full[fold_train])
        X_va = fold_scaler.transform(X_full[fold_val])

        fold_model = build_deep_model()
        fold_stop = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)
        fold_model.fit(X_tr, y_full[fold_train],
                       validation_data=(X_va, y_full[fold_val]),
                       epochs=500, batch_size=32,
                       callbacks=[fold_stop], verbose=0)

        fold_pred = fold_model.predict(X_va, verbose=0).flatten()
        fold_rmse = np.sqrt(mean_squared_error(y_full[fold_val], fold_pred))
        cv_scores.append(fold_rmse)
        print("  Fold", fold, "RMSE:", round(fold_rmse, 3))
        fold = fold + 1

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print("  5-Fold CV RMSE:", round(cv_mean, 3), "+/-", round(cv_std, 3))

    # measure how long one prediction takes
    single_sample = X_test_scaled[0:1]
    deep_model(single_sample, training=False)

    times = []
    for i in range(200):
        start = time.time()
        deep_model(single_sample, training=False)
        end = time.time()
        times.append((end - start) * 1000)

    median_latency = np.median(times)
    p95_latency = np.percentile(times, 95)
    print("  Median latency:", round(median_latency, 3), "ms")

    mlflow.log_param("model", "deep_nn")
    mlflow.log_param("layers", "128-64-32")
    mlflow.log_param("epochs_trained", len(history.history["loss"]))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("improvement_vs_baseline", (baseline_rmse - rmse) / baseline_rmse * 100)
    mlflow.log_metric("cv_rmse_mean", cv_mean)
    mlflow.log_metric("cv_rmse_std", cv_std)
    mlflow.log_metric("median_latency_ms", median_latency)
    mlflow.log_metric("p95_latency_ms", p95_latency)

print("")
print("######## Autoencoder Pipeline ########")
with mlflow.start_run(run_name="autoencoder_pipeline"):
    encoder_input = layers.Input(shape=(5,))
    encoded = layers.Dense(8, activation="relu")(encoder_input)
    latent = layers.Dense(3, activation="relu")(encoded)
    decoded = layers.Dense(8, activation="relu")(latent)
    decoder_output = layers.Dense(5)(decoded)

    autoencoder = keras.Model(encoder_input, decoder_output)
    encoder = keras.Model(encoder_input, latent)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X_train_scaled, X_train_scaled,
                    validation_data=(X_val_scaled, X_val_scaled),
                    epochs=200, batch_size=32, verbose=0)

    X_train_latent = encoder.predict(X_train_scaled, verbose=0)
    X_val_latent = encoder.predict(X_val_scaled, verbose=0)
    X_test_latent = encoder.predict(X_test_scaled, verbose=0)

    ae_model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    ae_model.compile(optimizer="adam", loss="mse")

    stop = keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)
    ae_model.fit(X_train_latent, y_train,
                 validation_data=(X_val_latent, y_val),
                 epochs=300, batch_size=32,
                 callbacks=[stop], verbose=0)

    ae_pred = ae_model.predict(X_test_latent, verbose=0).flatten()
    rmse, mae, r2 = evaluate("Autoencoder Pipeline", y_test, ae_pred)

    mlflow.log_param("model", "autoencoder_pipeline")
    mlflow.log_param("latent_size", 3)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("improvement_vs_baseline", (baseline_rmse - rmse) / baseline_rmse * 100)

# the deep network is the one we serve
print("")
print("Saving the deep network, the scaler and the column order")
deep_model.save("models/airfoil_deep_nn.keras")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features.input_cols, "models/feature_order.pkl")

model_info = {}
model_info["model"] = "deep_nn"
model_info["layers"] = "128-64-32"
model_info["rmse"] = results["Deep NN"]["rmse"]
model_info["mae"] = results["Deep NN"]["mae"]
model_info["r2"] = results["Deep NN"]["r2"]
model_info["baseline_rmse"] = baseline_rmse
model_info["cv_rmse_mean"] = float(cv_mean)
model_info["cv_rmse_std"] = float(cv_std)
model_info["median_latency_ms"] = float(median_latency)
model_info["n_train"] = int(len(X_train))
model_info["n_test"] = int(len(X_test))

info_file = open("models/model_info.json", "w")
json.dump(model_info, info_file, indent=2)
info_file.close()

# save the test rows so the tests can check the model later
test_data = df.iloc[test_rows].copy()
test_data.to_csv("data/test_sample.csv", index=False)
print("Saved data/test_sample.csv with shape:", test_data.shape)

# save one raw input so we can test the API later
first_test_row = df.iloc[test_rows[0]]

readings = {}
for name in features.input_cols:
    readings[name] = float(first_test_row[name])

sample = {}
sample["readings"] = readings
sample["true_sound_pressure"] = float(first_test_row[features.target_col])
sample["expected_prediction"] = float(deep_pred[0])

sample_file = open("data/sample_input.json", "w")
json.dump(sample, sample_file, indent=2)
sample_file.close()

print("Saved data/sample_input.json")
print("")
print("Summary")
for name in results:
    print(" ", name, "-> RMSE", round(results[name]["rmse"], 3),
          " R2", round(results[name]["r2"], 3))
