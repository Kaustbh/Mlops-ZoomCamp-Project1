# MLOps for Crab Age Prediction 
|![](/images/house.jpg)|
|:--:| 
|*[Image from https://pixabay.com](https://pixabay.com/de/photos/geld-heimat-m%C3%BCnze-anlage-gesch%C3%A4ft-2724235/)*|

## Problem description
The used dataset is from [Kaggle](https://www.kaggle.com/datasets/shalfey/extended-crab-age-prediction/data). The dataset is used to estimate the age of the crab based on the physical attributes
There are 10 columns:

<div align="center">

| field | description |
|------|------------| 
| id  | Used for Indexing |
|Sex| Gender of the Crab - Male, Female and Indeterminate |
|Length | Length of the Crab (in Feet; 1 foot = 30.48 cms) |
|Diameter   |   Diameter of the Crab (in Feet; 1 foot = 30.48 cms) |
|Height  |  Height of the Crab (in Feet; 1 foot = 30.48 cms) |
|Weight |  Weight of the Crab (in ounces; 1 Pound = 16 ounces) |
|Shucked Weight |  Weight without the shell (in ounces; 1 Pound = 16 ounces) |
|Viscera Weight   |   Weight that wraps around your abdominal organs deep inside body (in ounces; 1 Pound = 16 ounces) |
|Shell Weight |   Weight of the Shell (in ounces; 1 Pound = 16 ounces) |
|Age     |  Age of the Crab (in months) |

</div>
The main focus of the project is to apply the MLops principles like **experiment tracking**, **training pipeline**, **model monitoring** concepts to the machine learning projects rather than getting state-of-the-art accuracy.

This model tries to predict the Crab Age (column "Age") for this data set.

<p style="font-size:28px; color:red;"><strong>The Entire Project can be run using the Makefile</strong></p>


### Installing of make and git
```
sudo pacman -S make git
```


### Cloning the git repository.

Clone the repository.
```shell
git clone https://github.com/Kaustbh/Mlops-ZoomCamp-Project1.git
```
Go to the root directory by ```cd Mlops-ZoomCamp-Project1```.


### Installing of docker and docker compose

Since this project uses a lot of dockerized services, docker and docker compose are needed to be installed.

You need to follow these steps about **Install using the apt repository** from [here](https://docs.docker.com/engine/install/ubuntu/). You also need to install the post installation activity of docker from [here](https://docs.docker.com/engine/install/linux-postinstall/).



## 1. Prerequisites
To prepare the environment just run **make prerequisites**
```
prerequisites:
	@echo "Building Python environment and unzipping dataset"
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.11
	pipenv run python ./ingestion/unzipZipRaw.py
```
The step creates virtual environment using pipenv and install all the required packages present in the **Pipfile.lock** file.
It also unzips the data from zip file stored in ```data/``` folder.


## 2. Start MLFlow UI

To start MLFlow UI open new terminal and run **make mlflow**
```
mlflow:
	@echo "Starting MLflow server..."
	mlflow server \
	--backend-store-uri sqlite:///${HOME}/mlops_zoomcamp/final_project/project_1/mlflow.db 
```
You can access the initialized GUI at http://127.0.0.1:5000.


## 3. Start Prefect server

To start Prefect server open new terminal and run **make prefect**
```
prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start
```
You can access the initialized GUI at http://127.0.0.1:4200.


## 4. Workflow orchestration

Now everything is ready to start the orchestration workflow. Run **make run-training-pipeline** in a new terminal window.
```
run-training-pipeline:
	@echo "Start Training"
	pipenv run python pipeline/training_pipeline.py
```

Running the flows (and sub flows) and tasks can take some time.
This workflow includes a whole bunch of steps. 

First the datasets are provided. You can find all of them in the ```data/processed``` folder. After that I performed **Preprocessing** and **Normalization** on the dataset then applied **RandomForestRegressor** to the dataset.

You can find the run on MLFlow website. The model is now registered and I also promoted it to **Production** stage automatically.

All of this mentioned steps are shown in the Prefect GUI after the main flow has finished.

You want to know more about training pipeline, take a look at [README](./pipeline/README.md)


## 5. Monitoring with Evidently and Grafana

This step shows Evidently and Grafana in action. It is dockerized (have a look at docker-compose.yaml). To start this step open new terminal and run **make monitoring**. Run this make command in root directory.
```
monitoring:
	@echo "Starting monitoring with Evidently and Grafana dashboards"
	pipenv run docker-compose -f docker-compose.yaml up --build
	@echo "Open a new terminal and run"
	@echo "cd monitoring"
	@echo "python evidently_metrics_calculations.py"
```
This provides 3 running docker containers for you (database, [Grafana](http://localhost:3000/), and [Adminer](http://localhost:8081/)). The user credentials for Grafana are **admin:admin**.
Then you have to open new terminal and change directory to the monitoring folder and run **python evidently_metrics_calculation.py** manually.

The ML model monitoring proceeses and dashboard are fully explained and demonstrated in the [README](monitoring/README.md) file


## 6. Model deployment as simple web service

This step is about deploying the model as a web service. It is also dockerized (have a look at the Dockerfile in the deployment folder). The image building process can be triggered by running **make deployment**.
```
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
```
Then you have to change directory to the deployment folder. By running **docker run -it --rm -p 5010:5010 crag-age-prediction-service:v1** the docker container is started. The web service is listening at http://0.0.0.0:5010. 
Open a new terminal (in the web-service folder) and run **python test.py**. This triggers a request to get a prediction for one specific example. This triggers one request and outputs the result of the prediction to the terminal.


## 7. Testing

### 1. Check the unit test.

- You can run unit test by `make run-unit-test`.

### 2. Check the quality of the code by linting tools.

- You can run by `make quality-check`.

## 8. CICD (Github Actions)

I also implemented pre-commit hooks (see .pre-commit-config.yaml) and I added ci-tests (see .github/workflows/ci.yaml).

For running the hooks present in **pre-commit-config.yaml** you need to first install the pre-commit, for that just run ``` make install-hooks```

The CICD (GitHub Actions) workflow is fully explained and demonstrated in the [workflows README](.github/workflows/README.md) file.


## 9. Cleaning

To stop and clean everything open new terminal and run

You have already set up the mlops-pipeline. You can stop the services ```make stop-all-services```

```
stop-all-services:
	docker compose down 
```
 ```make clean```
```
clean:
	@echo "Cleaning"
	rm -rf __pycache__
	rm -rf data/processed
	rm -rf data/extended_crab_age_pred.csv
	rm -rf evidently
	rm -rf mlruns
	rm -rf mlflow.db
	pipenv --rm
```
This also removes the virtual environment in the project folder **.venv**

## Scoring

For the scoring purpose, if you want to find out which steps are implemented throughout this project, check [this](./docs/plan.md)
