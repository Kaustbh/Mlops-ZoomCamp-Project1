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

web-service:
	@echo "Creating docker container for model deployment (as web service)"
	pipenv run docker build -f ./deployment/Dockerfile -t crab-age-prediction-service:v1
	@echo "Open a new terminal and run"
	@echo "cd web-service"
	@echo "docker run -it --rm -p 5010:5010 crab-age-prediction-service:v1"
	@echo "Open a new terminal and run"
	@echo "python test.py"
	@echo "To stop all running docker containers run"
	@echo "docker stop $(docker ps -a -q)"

quality-check:
	pipenv run isort .
	pipenv run black .
	pipenv run pylint --recursive=y .