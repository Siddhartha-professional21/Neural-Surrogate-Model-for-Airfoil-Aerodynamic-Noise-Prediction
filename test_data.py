
import features

df = features.load_data("data")
data = features.prepare_data(df)


def test_data_has_the_right_shape():
    assert df.shape == (1503, 6)


def test_no_missing_values():
    assert df.isnull().sum().sum() == 0


def test_no_duplicate_rows():
    assert df.duplicated().sum() == 0


def test_inputs_are_inside_their_known_ranges():
    assert df["frequency"].min() >= 200
    assert df["frequency"].max() <= 20000
    assert df["angle_of_attack"].min() >= 0
    assert df["angle_of_attack"].max() <= 22.2
    assert df["velocity"].min() >= 31.7
    assert df["velocity"].max() <= 71.3
    assert df["displacement_thickness"].min() > 0


def test_target_is_a_sensible_sound_level():
    assert df["sound_pressure"].min() > 100
    assert df["sound_pressure"].max() < 145


def test_log_changed_only_the_two_chosen_columns():
    assert data["frequency"].max() < 5
    assert data["displacement_thickness"].max() < 0
    assert data["velocity"].equals(df["velocity"])
    assert data["angle_of_attack"].equals(df["angle_of_attack"])
    assert data["chord_length"].equals(df["chord_length"])
