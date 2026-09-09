
import json
import requests

sample_file = open("data/sample_input.json")
sample = json.load(sample_file)
sample_file.close()

response = requests.post("http://localhost:8000/predict", json=sample["readings"])
result = response.json()

predicted = result["predicted_sound_pressure_db"]
expected = sample["expected_prediction"]

print("Container prediction:", predicted, "dB")
print("Expected prediction :", round(expected, 3), "dB")
print("True value          :", round(sample["true_sound_pressure"], 3), "dB")

difference = abs(predicted - expected)
print("Difference          :", round(difference, 3), "dB")

if difference > 0.5:
    raise Exception("the container does not agree with the saved prediction")

print("")
print("Container matches the prediction made during training")
