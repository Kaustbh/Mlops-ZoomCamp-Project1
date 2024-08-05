# pylint: disable=invalid-name, redefined-outer-name, ungrouped-imports
import os
import pickle
from datetime import datetime

import yaml
import numpy as np
import mlflow
import optuna
import pandas as pd
from prefect import flow, task, get_run_logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_EXPERIMENT_URI",
    "sqlite:////home/kaustubh/mlops_zoomcamp/final_project/project_1/mlflow.db",
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment_name = os.getenv("EXPERIMENT_NAME", "training-pipeline")
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()

RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
]


@task(retries=2, retry_delay_seconds=5)
def load_config(config_path):
    """
    Load the configuration path
    Args:
        config_path (str): A path where configuration file exists

    Returns:
        dict: the configuration
    """
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


@task(name="Load the Data", retries=2, retry_delay_seconds=5)
def load_data(file_path):
    """
    Load the data from csv file
    Args:
        file_path (str): csv file path

    Returns:
        pandas.DataFrame:
    """
    df = pd.read_csv(file_path)

    return df


@task(name="Splitting the data")
def data_split(
    df,
    train_path,
    valid_path,
    test_path,
):
    """
    Train test split and these data will be saved in data/processed
    Args:
        df (pandas.DataFrame): all data read from csv file
        train_path (str): training csv file
        valid_path (str): valid csv file
        test_path (str): test csv file
    """
    train, test_val = train_test_split(df, test_size=0.25, random_state=42)
    valid, test = train_test_split(test_val, test_size=0.5, random_state=42)

    os.makedirs(
        "/home/kaustubh/mlops_zoomcamp/final_project/project_1/data/processed",
        exist_ok=True,
    )
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)
    test.to_parquet(test_path, index=False)


def drop_features(train, valid, test, target_var):
    """
    Dropping features
    Args:
        train (pandas.DataFrame): training data
        valid (pandas.DataFrame): validation data
        target_var (str): target column

    Returns:
        tuple: (input_train, output_train, input_valid, output_valid)
    """
    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]
    x_valid = valid.drop(target_var, axis=1)
    y_valid = valid[target_var]
    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


@task(name="Processing the Features")
def process_features(train_path, valid_path, test_path, target_var, save_dv=True):
    # pylint: disable=too-many-locals
    """
    Dropped and processed features by using DictVectorizer
    Args:
        train_path (str): training data
        valid_path (str): validation data
        target_var (str): testing data
        save_dv (bool, optional): save DictVectorizer or not. Defaults to True.

    Returns:
        tuple: (input_train, output_train, input_valid, output_valid)
    """

    training = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)
    test = pd.read_parquet(test_path)

    x_train, y_train, x_valid, y_valid, x_test, y_test = drop_features(
        training, valid, test, target_var
    )

    le = LabelEncoder()
    # scaler = StandardScaler()

    # Fit scaler to training data and transform
    x_train["Sex"] = le.fit_transform(x_train["Sex"])
    x_valid["Sex"] = le.transform(x_valid["Sex"])
    x_test["Sex"] = le.transform(x_test["Sex"])

    # numerical_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()
    # x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])
    # x_valid[numerical_columns] = scaler.transform(x_valid[numerical_columns])
    # x_test[numerical_columns] = scaler.transform(x_test[numerical_columns])

    if save_dv:
        os.makedirs("model", exist_ok=True)
        with open("model/preprocessor.b", "wb") as f:
            pickle.dump(le, f)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


@task(name="Training and Optimization")
def hpo(X_train, y_train, X_valid, y_valid, n_trials=3):
    """
    Hyperparameter Optimization stage
    Args:
        X_train (pandas.DataFrame): input training data
        y_train (pandas.DataFrame): output training data
        X_valid (pandas.DataFrame): input validation data
        y_valid (pandas.DataFrame): output validation data
        n_trials (int, optional): number of trials. Defaults to 3.
    """
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
            "random_state": 42,
            "n_jobs": -1,
        }
        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_params(params=params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_artifact(local_path="model/preprocessor.b")
        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)


@task(name="Transition Model")
def transition_model_stage(
    reg_model_meta_data, model_name="crab-age-predictor", stage="Production"
):
    """
    Transitioned the best model from the training trials to the production stage
    Args:
        reg_model_meta_data (): metadata about registered model
        model_name (str, optional): model name used in mlflow.
                                    Defaults to "crag-age-predictor".
        stage (str, optional): stage inside mlflow model registry. Defaults to "production".
    """
    # Initialize the MlflowClient
    client = mlflow.tracking.MlflowClient()

    model_version = reg_model_meta_data.version

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
        archive_existing_versions=True,
    )

    date = datetime.today().date()
    client.update_model_version(
        name=model_name,
        version=reg_model_meta_data.version,
        description=f"The model version {reg_model_meta_data.version} "
        f"was transition to {stage} on {date}",
    )


def train_and_log_model(X_train, y_train, X_valid, y_valid, x_test, y_test, params):

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        filtered_params = {param: params[param] for param in RF_PARAMS}

        rf = RandomForestRegressor(**filtered_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(x_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


@task(name="Register best model")
def run_register_model(X_train, y_train, X_valid, y_valid, x_test, y_test, top_n: int):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(
            X_train, y_train, X_valid, y_valid, x_test, y_test, params=run.data.params
        )

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(experiment_name)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )[0]

    # Register the best model
    run_identifier = best_run.info.run_id
    model_uri = f"runs:/{run_identifier}/model"
    reg_model_meta_data = mlflow.register_model(
        model_uri=model_uri, name="crab-age-predictor"
    )

    return reg_model_meta_data


@flow(name="training_pipeline")
def train_pipeline(experiment_name, config_path):
    # pylint: disable=too-many-locals
    """
    Main training pipeline
    Args:
        experiement_name (str): An experiment name used in the mlflow
        config_path (str): A configuration path used in training
    """
    logger = get_run_logger()

    logger.info("Loading configuration")
    config = load_config(config_path)

    dataset_path = config["dataset_path"]
    train_path = config["train_path"]
    valid_path = config["valid_path"]
    test_path = config["test_path"]
    target_var = config["target_variable"]
    trials = config["trials"]

    logger.info("Loading the data")
    df = load_data(dataset_path)

    logger.info("Splitting the data into train, valid and test")
    data_split(df, train_path, valid_path, test_path)

    logger.info("Processing Features")
    X_train, y_train, X_valid, y_valid, x_test, y_test = process_features(
        train_path, valid_path, test_path, target_var
    )

    logger.info("Hyperparameter Tuning with XGBoost model")
    hpo(X_train, y_train, X_valid, y_valid, trials)

    logger.info("Searching the best model and registering it")
    best_model_meta_data = run_register_model(
        X_train, y_train, X_valid, y_valid, x_test, y_test, 10
    )

    logger.info("Transition the best model to the production stage")
    transition_model_stage(best_model_meta_data)


if __name__ == "__main__":

    train_pipeline(
        experiment_name,
        config_path="/home/kaustubh/mlops_zoomcamp/final_project/project_1/config.yaml",
    )
