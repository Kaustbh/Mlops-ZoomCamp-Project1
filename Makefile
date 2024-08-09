export PIPENV_VERBOSITY := -0

prerequisites:
	@echo "Building Python environment and unzipping dataset"
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.12
	@echo "Installing dependencies from Pipfile"
	pipenv install --dev
	pipenv run python ./ingestion/UnZipRaw.py

mlflow:
	@echo "Starting MLflow server..."
	mlflow server \
	--backend-store-uri sqlite:///${HOME}/mlops_zoomcamp/final_project/project_1/mlflow.db \

prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start

run-training-pipeline:
	pipenv run python pipeline/training_pipeline.py

monitoring:
	@echo "Starting monitoring with Evidently and Grafana dashboards"
	pipenv run docker-compose -f docker-compose.yaml up --build
	@echo "Open a new terminal and run"
	@echo "cd monitoring"
	@echo "python evidently_metrics_calculations.py"


deployment:
	@echo "Creating docker container for model deployment (as web service)"
	pipenv run docker build -f ./deployment/Dockerfile -t crab-age-prediction-service:v1
	@echo "Open a new terminal and run"
	@echo "cd deployment"
	@echo "docker run -it --rm -p 5010:5010 crab-age-prediction-service:v1"
	@echo "Open a new terminal and run"
	@echo "python test.py"
	@echo "To stop all running docker containers run"
	@echo "docker stop $(docker ps -a -q)"

quality-check:
	pipenv run isort .
	pipenv run black .
	pipenv run pylint --recursive=y .
