
# Neural Surrogate Model for Airfoil Aerodynamic Noise Prediction

![CI/CD](https://github.com/Siddhartha-professional21/Neural-Surrogate-Model-for-Airfoil-Aerodynamic-Noise-Prediction/actions/workflows/ci.yml/badge.svg)

**Live API:** https://airfoil-noise-api.onrender.com/docs
(free tier — the first request may take ~40 seconds to wake the service)

Predicting the sound pressure level of an airfoil from its shape and the air
flow around it. The NASA dataset holds 1,503 wind tunnel measurements with 5
inputs and one output in decibels. A neural network learns the relationship and
answers in milliseconds, which is the point of a surrogate model: it stands in
for a wind tunnel run or a CFD simulation.

## Results

70 / 15 / 15 split, so 1,051 rows for training, 226 for validation and 226 for
the held out test set. All numbers below are on the test set.

| Model | RMSE (dB) | MAE (dB) | R2 | Better than baseline |
|---|---|---|---|---|
| Linear Regression (baseline) | 4.980 | 3.873 | 0.524 | - |
| Random Forest | 1.843 | 1.353 | 0.935 | 63% |
| Shallow NN (32) | 3.512 | 2.725 | 0.763 | 29% |
| **Deep NN (128-64-32), served** | **1.992** | **1.568** | **0.924** | **60%** |
| Autoencoder pipeline | 5.639 | 4.455 | 0.389 | -13% |

Five fold cross validation of the deep network gives 2.158 +/- 0.103 RMSE across
folds of 2.211, 1.956, 2.180, 2.196 and 2.247, so the test score is not a lucky
split. A single prediction takes 12.5 ms on a warmed up CPU.

Two results worth pointing out honestly.

The Random Forest scores slightly better than the deep network, 1.843 against
1.992. The gap of 0.15 dB is inside the cross validation spread. The network is
the one that gets served because the goal of the project is a neural surrogate:
a smooth, constant time stand in for a wind tunnel run. The forest is there as
the comparison that shows the network is in the right range.

The autoencoder pipeline is worse than plain linear regression. Squeezing 5
inputs down to 3 dimensions throws away information, and with only 5 inputs
there was nothing to compress in the first place. It is kept in the comparison
because a negative result is still a result.

## Inputs

| Input | Unit | Range in the data |
|---|---|---|
| frequency | Hz | 200 to 20000 |
| angle_of_attack | degrees | 0 to 22.2 |
| chord_length | metres | 0.0254 to 0.3048 |
| velocity | m/s | 31.7 to 71.3 |
| displacement_thickness | metres | 0.0004 to 0.0584 |

Frequency and displacement thickness are put on a log10 scale before training,
because both span several orders of magnitude. The other three are left alone.
The scaler is fitted on the training rows only.

The API takes the raw physical values and applies the log itself, so a caller
never has to know about the transform.

## Files

| File | Purpose |
|---|---|
| features.py | loads the data, holds the column order and the log transform |
| train.py | trains the five models, logs every run to MLflow, saves the model |
| app.py | FastAPI service for the saved model and scaler |
| test_data.py | checks the dataset shape, ranges, missing values and the log |
| test_model.py | checks the saved model still scores above its limits |
| test_api.py | checks the API endpoints and the predictions |
| check_container.py | sends a real input to the running container |
| Dockerfile | builds the API image |
| .github/workflows/ci.yml | the CI/CD pipeline |

The dataset is 59 KB and the saved model is 196 KB, so both live in the
repository. Nothing needs downloading.

## Running it

requirements.txt holds only what the API image needs. Training and testing need
three more packages:

    pip install -r requirements.txt
    pip install mlflow pytest httpx

Then:

    python train.py
    python -m pytest -v
    uvicorn app:app --port 8000

With Docker:

    docker build -t airfoil-api .
    docker run -p 8000:8000 airfoil-api

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /health | check the service is up and the model is loaded |
| GET | /model | scores, layers and training details of the served model |
| POST | /predict | predict the sound pressure level for one set of inputs |
| GET | /docs | interactive documentation |

The request body is the 5 raw inputs. The response gives the predicted sound
pressure in decibels, the RMSE of the model so the number can be read with its
error, a warning for any input outside the range the model was trained on, and
the time the prediction took.

An input with a frequency or displacement thickness of zero or below is
rejected, because the log needs positive numbers. An input outside the training
range still gets a prediction, with a warning attached.

## Experiment tracking

Every model is one MLflow run, 5 in total: linear regression, random forest,
shallow network, deep network and the autoencoder pipeline. Each run stores its
name, RMSE, MAE and R2. The four models after the baseline also store how much
better they are than it. The deep network run additionally stores the number of
epochs it trained for, the cross validation mean and standard deviation, and the
median and 95th percentile prediction time.

The database is part of the repository, so the runs can be opened directly:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

## Pipeline

Every push to main runs four jobs in order:

1. data-validation checks the shape, the ranges, the missing values and that the
   log was applied to the right two columns
2. model-testing loads the saved model and fails the build if RMSE goes above
   2.6 dB, if R2 drops below 0.88, or if it stops beating the linear baseline
3. api-testing checks every endpoint, the rejected inputs and the warnings, and
   that the API returns the same prediction that was recorded during training
4. docker-and-deploy builds the image, starts the container, sends a real input
   to it, checks the answer agrees with the saved prediction, then pushes the
   image to the GitHub Container Registry

The image uses tensorflow-cpu rather than tensorflow, since a server has no GPU
and the CPU build is a fraction of the size. It runs as a non-root user and
reads the port from the PORT environment variable, so the same image runs
locally and on a host that assigns its own port.

## Dataset

Brooks, T., Pope, D., Marcolini, M. (1989). Airfoil self-noise and prediction.
NASA Reference Publication 1218. UCI Machine Learning Repository.
