import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def read_testdata():

    path = os.path.join(data_path, "valid.parquet")

    data = pd.read_parquet(path)

    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'preprocessor', 'preprocessor.b'
    )

    with open(model_relative_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Ensure to separate features and target variable correctly
    X_val = data.drop(columns=["Age"])
    y_val = data["Age"]

    # Apply the LabelEncoder to the 'Sex' column
    X_val["Sex"] = preprocessor.transform(X_val["Sex"])

    return X_val, y_val


def test_model_loading():
    # Test if the model loads correctly

    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)

    assert rf_model is not None, "Model failed to load"


def test_predictAccuracy():
    # testinng accuracy if it will be accpted or not
    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)

    features_xval, features_yval = read_testdata()
    actual_prediction = rf_model.predict(features_xval)
    mse = mean_squared_error(features_yval, actual_prediction)
    rmse = np.sqrt(mse)

    # Assuming acceptable performance threshold is an RMSE below 10
    assert rmse <= 10


def test_predict_shape():
    # Test if the prediction shape matches the input shape

    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)

    features_xval, _ = read_testdata()
    actual_prediction = rf_model.predict(features_xval)
    assert (
        actual_prediction.shape[0] == features_xval.shape[0]
    ), "Mismatch in prediction shape"


def test_empty_input():
    # Test how the model handles an empty input
    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)
    try:
        rf_model.predict(np.array([]))
        assert False, "Model should raise an exception for empty input"
    except ValueError:
        pass  # Expected behavior

def test_predictability_with_outliers():
    # Test model's predictability with extreme values

    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)

    # Create a dataset with extreme values
    features_xval, _ = read_testdata()
    features_xval.loc[0, "Length"] = 1000  # Set an extreme value

    try:
        rf_model.predict(features_xval)
    except Exception as e:
        assert False, f"Model failed with extreme input: {e}"

def test_unexpected_labels():
    # Test how the preprocessor and model handle unexpected labels in 'Sex' column

    model_relative_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'model.pkl'
    )

    with open(model_relative_path, "rb") as f:
        rf_model = pickle.load(f)

    features_xval, _ = read_testdata()
    features_xval.loc[0, "Sex"] = 'M' # Set an unexpected label

    try:
        rf_model.predict(features_xval)
        assert False, "Model should raise an exception for input with unexpected label"
    except ValueError:
        pass  # Expected behavior
