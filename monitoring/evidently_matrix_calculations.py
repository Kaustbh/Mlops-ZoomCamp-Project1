import io
import os
import time
import pickle
import random
from datetime import datetime, timedelta

import mlflow
import pandas as pd
import psycopg
import mlflow.sklearn
from mlflow import MlflowClient
from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.ui.workspace import Workspace

HOME_DIRECTORY = os.environ.get("HOME")
DATAPATH = "/home/kaustubh/mlops_zoomcamp/final_project/project_1/data/processed"
TRAINPATH = os.path.join(DATAPATH, "train.parquet")
VALIDPATH = os.path.join(DATAPATH, "valid.parquet")
TRACKING_URI = (
    "sqlite:////home/kaustubh/mlops_zoomcamp/final_project/project_1/mlflow.db"
)
EXPERIMENT_NAME = "training-pipeline"
MODEL_NAME = "crab-age-predictor"
LOCAL_SERVE_FOLDER = os.getenv(
    "LOCAL_SERVER_FOLDER",
    f"{HOME_DIRECTORY}/mlops_zoomcamp/final_project/project_1/mnt/serve",
)


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
ws = Workspace("workspace")

project = ws.create_project("Crab Age Prediction Project")
project.description = "My project description"
project.save()

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists age_metrics;
create table age_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns varchar,
    share_missing_values float
)
"""

begin = None

categorical = ["Sex"]
numerical = [
    "Length",
    "Diameter",
    "Height",
    "Weight",
    "Shucked Weight",
    "Viscera Weight",
    "Shell Weight",
]
FEATURES = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Weight",
    "Shucked Weight",
    "Viscera Weight",
    "Shell Weight",
]

column_mapping = ColumnMapping(
    prediction="predicted_age",
    numerical_features=numerical,
    categorical_features=categorical,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="predicted_age"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task(name="Read Data", retries=3, retry_delay_seconds=2)
def read_dataframe():

    # Read data from data folder
    df = pd.read_parquet(VALIDPATH)

    print(df.head())
    return df


@task(name="Read Reference Data", retries=3, retry_delay_seconds=2)
def read_reference():

    df = pd.read_parquet(TRAINPATH)
    print(df.head())
    return df


@task(name="Transform Data", retries=3, retry_delay_seconds=2)
def transform_data(df):

    return df


def get_prod_run_id(tracking_uri, model_name, stage="Production"):
    """
    get the latest production run_id from model registry
    Args:
        tracking_uri (str): tracking uri of mlflow server
        model_name (str): experiment_name
        stage (str, optional): Staging or Production. Defaults to "Production".

    Returns:
        str: run_id of the latest production model
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    model_metadata = client.get_latest_versions(name=model_name, stages=[stage])[0]
    run_id = model_metadata.run_id

    return run_id


def get_latest_run_id():
    """
    Get the run_id of the production model inside the model registry.
    Returns:
        str: run_id of the latest production model
    """
    run_id = get_prod_run_id(
        tracking_uri=TRACKING_URI,
        model_name=MODEL_NAME,
    )

    return run_id


@task(name="Load Model", retries=3, retry_delay_seconds=2)
def load_model():

    run_id = get_latest_run_id()

    # Load the model as a PyFuncModel
    logged_model = f"runs:/{run_id}/model"

    load_model = mlflow.sklearn.load_model(logged_model)

    return load_model


def download_artifacts(run_id, artifact_path, dst_path):
    """
    Download the artifact(dict vectorizer) from the model registry
    Args:
        run_id (str): run_id of the model
        artifact_path (str): artifact path in the model registry
        dst_path (str): destination path in the local
    """
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )


def load_dv(run_id, artifact_path="preprocessor.b"):
    # pylint: disable=invalid-name
    """
    Download the Preprocessor and load it from the local path.
    Args:
        run_id (str): run_id of the model
        artifact_path (str, optional):  artifact path in the model registry.
                                        Defaults to "preprocessor.b".
    Returns:
        Preprocessor: A Preprocessor
    """

    download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=LOCAL_SERVE_FOLDER,
    )

    preprocessor_path = os.path.join(LOCAL_SERVE_FOLDER, artifact_path)

    with open(preprocessor_path, "rb") as f_in:
        le = pickle.load(f_in)

    return le


@task(name="Preprocess Data", retries=3, retry_delay_seconds=2)
def preprocess(serving_data):

    return serving_data


def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame, processor):

    X_train = df_train[FEATURES]
    X_train["Sex"] = processor.transform(X_train["Sex"])

    X_val = df_val[FEATURES]
    X_val["Sex"] = processor.transform(X_val["Sex"])

    y_train = df_train["Age"].values
    y_val = df_val["Age"].values

    return X_train, X_val, y_train, y_val


@task(name="Prepare Database", retries=3, retry_delay_seconds=2)
def prep_db():
    with psycopg.connect(
        "host=localhost port=5433 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5433 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


@task(name="Calculate Metrics", retries=3, retry_delay_seconds=2)
def calculate_metrics_postgresql(
    current_data, ref_df, regressor, preprocessor, curr, i
):

    X_train, X_test, y_train, y_test = add_features(current_data, ref_df, preprocessor)

    current_data["predicted_age"] = regressor.predict(X_train)
    ref_df["predicted_age"] = regressor.predict(X_test)

    report.run(
        reference_data=ref_df,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    ws.add_report(project.id, report)
    project.save()
    result = report.as_dict()

    # deriving some values (prediction drift)
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    # number of drifted columns
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    # share of missing values
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    curr.execute(
        "insert into age_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
        (
            datetime.now() + timedelta(hours=i),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
        ),
    )
    y_pred = current_data["predicted_age"].to_numpy()
    return (y_pred, current_data)


@task(name="Log Serve Run", retries=3, retry_delay_seconds=2)
def log_serve_run(df, y_pred):
    mlflow.end_run()
    with mlflow.start_run():
        # Add serve tags
        mlflow.set_tag("developer", "Kaustubh")
        mlflow.set_tag("model", "randomforest")
        # Log some variables to mlflow
        mlflow.log_metric("serving_data_row_count", len(df))
        mlflow.log_metric("predictions_row_count", len(y_pred))
        try:
            mlflow.log_metric("mean_predicted_age", y_pred.mean())
        except TypeError:
            print("Error: cannot convert the series to <class 'float'>")
        mlflow.set_tag("run_datetime", str(datetime.now()))
    mlflow.end_run()
    return None


@task(name="Write Predictions", retries=3, retry_delay_seconds=2)
def write_predictions(y_pred):
    # Persist the predictions
    output_filename = (
        f"predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_key = f"data/output/{output_filename}"
    csv_buffer = io.BytesIO()
    pd.DataFrame(y_pred, columns=["predicted_age"]).to_csv(csv_buffer, index=False)


@flow(name="ML Serve Flow")
def main():
    # Prepare database
    prep_db()
    last_send = datetime.now() - timedelta(seconds=10)
    # Read the data
    df = read_dataframe()

    # Read the reference data
    ref_df = read_reference()
    # Transform the data
    # df = transform_data(df)
    # Load the model
    regressor = load_model()
    # Load Preprocessor
    # preprocessor = load_dv(run_id)
    with open(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/mnt/serve/preprocessor.b",
        "rb",
    ) as f:
        preprocessor = pickle.load(f)
    # Preprocess the data
    # serving = preprocess(df)

    # Make predictions & calculate metrics
    with psycopg.connect(
        "host=localhost port=5433 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(0, 1):
            with conn.cursor() as curr:
                current_df, y_pred = calculate_metrics_postgresql(
                    df, ref_df, regressor, preprocessor, curr, i
                )
                log_serve_run(current_df, y_pred)
                # Persist the predictions
                write_predictions(y_pred)
            new_send = datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + timedelta(seconds=10)
                # Log the run details


if __name__ == "__main__":

    main()
