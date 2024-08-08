import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

data_path = "/home/kaustubh/mlops_zoomcamp/final_project/project_1/data/processed"


def read_testdata():

    path = os.path.join(data_path, "valid.parquet")

    data = pd.read_parquet(path)

    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/preprocessor/preprocessor.b",
        "rb",
    ) as f:
        preprocessor = pickle.load(f)

    # Ensure to separate features and target variable correctly
    X_val = data.drop(columns=["Age"])
    y_val = data["Age"]

    # Apply the LabelEncoder to the 'Sex' column
    X_val["Sex"] = preprocessor.transform(X_val["Sex"])

    return X_val, y_val


def test_model_loading():
    # Test if the model loads correctly

    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)

    assert rf_model is not None, "Model failed to load"


def test_predictAccuracy():
    # testinng accuracy if it will be accpted or not
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)
    features_xval, features_yval = read_testdata()
    actual_prediction = rf_model.predict(features_xval)
    mse = mean_squared_error(features_yval, actual_prediction)
    rmse = np.sqrt(mse)

    # Assuming acceptable performance threshold is an RMSE below 10
    assert rmse <= 10


def test_predict_shape():
    # Test if the prediction shape matches the input shape
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)

    features_xval, _ = read_testdata()
    actual_prediction = rf_model.predict(features_xval)
    assert (
        actual_prediction.shape[0] == features_xval.shape[0]
    ), "Mismatch in prediction shape"


def test_empty_input():
    # Test how the model handles an empty input
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)
    try:
        rf_model.predict(np.array([]))
        assert False, "Model should raise an exception for empty input"
    except ValueError:
        pass  # Expected behavior


def test_consistent_predictions():
    # Test if the model gives consistent predictions for the same input
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)

    features_xval, _ = read_testdata()
    prediction_1 = rf_model.predict(features_xval)
    prediction_2 = rf_model.predict(features_xval)
    assert np.array_equal(
        prediction_1, prediction_2
    ), "Model predictions are inconsistent"


def test_predict():
    # testing the first row prediction is correct
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/models/model.pkl", "rb"
    ) as f:
        rf_model = pickle.load(f)

    features_xval, features_yval = read_testdata()
    actual_prediction = rf_model.predict(np.array([features_xval.iloc[0]]))
    assert actual_prediction == features_yval.iloc[0]
