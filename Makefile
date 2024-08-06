export PIPENV_VERBOSITY := -0

prerequisites:
	@echo "Building Python environment and unzipping dataset"
	# python3 -m pip install --upgrade pip
	# pip install --upgrade pipenv
	# pipenv install --python 3.12
	pipenv run python ./ingestion/UnZipRaw.py

mlflow:
	@echo "Preparing directories for MLflow..."
	mkdir -p ${HOME}/mlops_zoomcamp/final_project/project_1/mnt/mlruns/tracking
	mkdir -p ${HOME}/mlops_zoomcamp/final_project/project_1/mnt/mlruns/artifacts
	@echo "Starting MLflow server..."
	mlflow server \
	--backend-store-uri sqlite:///${HOME}/mlops_zoomcamp/final_project/project_1/mlflow.db \
	--default-artifact-root ${HOME}/mlops_zoomcamp/final_project/project_1/mnt/mlruns/artifacts \
	--host 0.0.0.0

prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start

run-training-pipeline:
	pipenv run python pipeline/training_pipeline.py

quality-check:
	pipenv run isort .
	pipenv run black .
	pipenv run pylint --recursive=y .