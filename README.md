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
| id  | used for Indexing |
|Sex| Gender of the Crab - Male, Female and Indeterminate |
|Length | Length of the Crab (in Feet; 1 foot = 30.48 cms) |
|Diameter   |   Diameter of the Crab (in Feet; 1 foot = 30.48 cms) |
|Height  |  Height of the Crab (in Feet; 1 foot = 30.48 cms) |
|Weight |  Weight of the Crab (in ounces; 1 Pound = 16 ounces) |
|Shucked Weight |  Weight without the shell (in ounces; 1 Pound = 16 ounces) |
|Viscera Weight   |   is weight that wraps around your abdominal organs deep inside body (in ounces; 1 Pound = 16 ounces) |
|Shell Weight |   Weight of the Shell (in ounces; 1 Pound = 16 ounces) |
|Age     |  Age of the Crab (in months) |

</div>
The main focus of the project is to apply the MLops principles like experiment tracking, training pipeline, model monitoring concepts to the machine learning projects rather than getting state-of-the-art accuracy.

This model tries to predict the Crab Age (column "Age") for this data set.

The Entire Project can be runned using Makefile

### Installing of make and git
```
sudo pacman -S make git
```
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
	pipenv run python ./scripts/unzipZipRaw.py
```
The step "Installing dependencies from Pipfile.lock" needs some time.
That provide the virtual environment **.venv** in project folder. This commands also unzips the data from zip file stored in ```data/``` folder.

I recommend to run steps as it's described in Makefile.

## 2. Start MLFlow UI
To start MLFlow UI open new terminal and run **make mlflow**
```
mlflow:
	@echo "Starting MLflow server..."
	mlflow server \
	--backend-store-uri sqlite:///${HOME}/mlops_zoomcamp/final_project/project_1/mlflow.db 
```
You can access the initialized GUI at http://127.0.0.1:5000.

![](/images/mlflow-initialGUI.png)



## 3. Start Prefect server
To start Prefect server open new terminal and run **make prefect**
```
prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start
```
You can access the initialized GUI at http://127.0.0.1:4200.

![](/images/prefect-initialGUI.png)
<!---
## 4. Workflow orchestration
Now everything is ready to start the orchestration workflow. Run **make deploy** in a new terminal window.
```
deploy: ./data/raw/housing-prices-35.csv
	@echo "Starting workflow deployment with Prefect"
	# if error: Work pool named 'zoompool' already exists. 
	# Please try creating your work pool again with a different name.
	# uncomment next line
	#pipenv run prefect work-pool delete 'zoompool'
	pipenv run prefect work-pool create --type process zoompool
	pipenv run prefect --no-prompt deploy --all
	pipenv run prefect worker start -p zoompool
```
With that everything is prepared. You only need to go to [Prefect website](http://127.0.0.1:4200/deployments) and hit "Quick run"

![](/images/prefect-quickRun.png)

Running the flows (and sub flows) and tasks can take some time.
This workflow includes a whole bunch of steps. 
First the datasets are provided. You can find all of them in the data/processed folder. After that there is a preparation step. For the training I have chosen the columns "area_living", "area_land", "n_rooms", "price", and "year". I extracted the "year" information from the "date" column.
This specific features are used in the training step.
You can find the run on MLFlow website. The model is now registered and I also promoted it to Staging stage automatically.

![](/images/mlflow-Run.png)

After that a test is triggered. For this purpose I decided to transition that model to Production. This decision can make no sense in production environment but I wanted to see this transition in action.

![](/images/mlflow-prodModel.png)

As the last step there is a prediction. For that purpose I used a different dataset. The dataset has almost 4.000 rows and for the prediction a randomly choosen row is used. You can find the predicted price in the terminal.

All of this mentioned steps are shown in the Prefect GUI after the main flow has finished.

![](/images/prefect-FlowCompleted.png)

For the training and the test an automatically created HTML report is stored in the evidently folder.

![](/images/report.png)

## 5. Monitoring with Evidently and Grafana
This step shows Evidently and Grafana in action. It is dockerized (have a look at monitoring/docker-compose.yaml). To start this step open new terminal and run **make monitoring**. Run this make command in root directory.
```
monitoring:
	@echo "Starting monitoring with Evidently and Grafana dashboards"
	pipenv run docker-compose -f ./monitoring/docker-compose.yaml up --build
	@echo "Open a new terminal and run"
	@echo "cd monitoring"
	@echo "python evidently_metrics_calculation.py"
```
This provides 3 running docker containers for you (database, [Grafana](http://localhost:3000/), and [Adminer](http://localhost:8080/)). The user credentials for Grafana are admin:admin.
Then you have to open new terminal and change directory to the monitoring folder and run **python evidently_metrics_calculation.py** manually.
A process is triggered to simulate production usage of the model. For that purpose some metrics are calculated for 9 runs (3 for each of 3 different data sets). On Grafana website you can find a prepared dashboard "Housing Prices Prediction Dashboard". After finishing you can see the results.

![](/images/grafana-db.png)

I implemented also simple alerting to raise an error when one specific value is higher than expected.  

![](/images/grafana-alerting.png)


## 6. Model deployment as simple web service
This step is about deploying the model as a web service. It is also dockerized (have a look at the Dockerfile in the web-service folder). The image building process can be triggered by running **make web-service**.
```
web-service:
	@echo "Creating docker container for model deployment (as web service)"
	pipenv run docker build -f ./web-service/Dockerfile -t housing-price-prediction-service:v1
	@echo "Open a new terminal and run"
	@echo "cd web-service"
	@echo "docker run -it --rm -p 9696:9696 housing-price-prediction-service:v1"
	@echo "Open a new terminal and run"
	@echo "python test.py"
	@echo "To stop all running docker containers run"
	@echo "docker stop $(docker ps -a -q)"
```
Then you have to change directory to the web-service folder. By running **docker run -it --rm -p 9696:9696 housing-price-prediction-service:v1** the docker container is started. The web service is listening at http://0.0.0.0:9696. 
Open a new terminal (in the web-service folder) and run **python test.py**. This triggers a request to get a prediction for one specific example. This triggers one request and outputs the result of the prediction to the terminal.

## 7. Cleaning
To clean everything open new terminal and run **make clean**
```
clean:
	@echo "Cleaning"
	rm -rf __pycache__
	rm -rf data/processed
	rm -rf data/raw/housing-prices-35.csv
	rm -rf evidently
	rm -rf mlruns
	rm -rf mlflow.db
	pipenv --rm
```
This also removes the virtual environment in the project folder **.venv**


## Reproducibility
Following each step in the mentioned order should make it easy to reproduce the same results like me.

## Best Practices
There are unit tests implemented and I used black, isort and pylint as linter and code formatters (have a look at pyproject.toml). I used a Makefile for the most important steps. This order should guide you through the project. I also implemented pre-commit hooks (see .pre-commit-config.yaml) and I added ci-tests (see .github/workflows/ci-tests.yml). -->