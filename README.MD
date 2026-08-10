# Explainable AI API

ML REST API for XGBoost predictions and SHAP explanations built with FastAPI.

## Features
XGBoost classification
SHAP model explanations
FastAPI REST API
Pydantic validation
pytest tests
Docker & Docker Compose

## Project Structure
api/
├── app.py
├── explain.py
├── schemas.py
├── services.py
└── xgboost_pipeline.pkl

tests/
notebooks/

Dockerfile
docker-compose.yml
requirements.txt

## Run with Docker
docker compose up --build

API:

http://localhost:8000

Swagger:

http://localhost:8000/docs

## Run Tests
docker compose exec api python -m pytest -v

## Technologies

Python · FastAPI · XGBoost · SHAP · scikit-learn · pytest · Docker