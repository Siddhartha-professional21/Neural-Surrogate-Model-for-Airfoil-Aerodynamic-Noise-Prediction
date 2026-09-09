
import numpy as np
import pandas as pd

# the file has no header row, so we name the columns ourselves
column_names = ["frequency", "angle_of_attack", "chord_length",
                "velocity", "displacement_thickness", "sound_pressure"]

# the 5 inputs, always in this order
input_cols = ["frequency", "angle_of_attack", "chord_length",
              "velocity", "displacement_thickness"]

target_col = "sound_pressure"

# the two columns that get a log scale
log_cols = ["frequency", "displacement_thickness"]


def load_data(folder):
    file_path = folder + "/airfoil_self_noise.dat"
    df = pd.read_csv(file_path, sep="\t", names=column_names)
    return df


def prepare_data(df):
    # frequency spans 200 Hz to 20000 Hz, so log scale helps the model
    data = df.copy()
    for name in log_cols:
        data[name] = np.log10(data[name])
    return data


def prepare_one_input(readings):
    # readings is a dictionary with the 5 raw values, before any log
    row = []
    for name in input_cols:
        value = float(readings[name])
        if name in log_cols:
            value = np.log10(value)
        row.append(value)
    return np.array([row])
