FROM python:3.9-slim

WORKDIR /app

COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["model_loader.py", "model_service.py", "app.py", "./"]
COPY ["model", "/app/model"]
COPY ["artifact", "/app/artifact"]

EXPOSE 5010

CMD ["gunicorn", "--bind", "0.0.0.0:5010", "app:app"]
