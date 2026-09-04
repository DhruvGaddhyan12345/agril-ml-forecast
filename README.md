# AgriML — Crop Yield Prediction & MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)

AgriML is an end-to-end **regional crop-yield forecasting and MLOps platform** designed to support early agricultural planning.

The system combines leakage-safe feature engineering, temporal validation, multi-model benchmarking, experiment tracking, containerized model serving, drift monitoring, and controlled model promotion into a reproducible ML pipeline.

---

## Overview

Agricultural yield varies across regions and depends on interacting environmental and historical factors such as:

- Historical crop yield
- Rainfall
- Temperature
- Soil and environmental conditions
- Region
- Crop characteristics
- Historical agricultural patterns

AgriML transforms these inputs into a production-oriented forecasting workflow:

```text
Agricultural Data
       │
       ▼
Data Validation
       │
       ▼
Leakage-Safe Feature Engineering
       │
       ▼
Temporal / Cross-Validation
       │
       ▼
Multi-Model Benchmark
       │
       ├── Linear / Ridge
       ├── Random Forest
       ├── XGBoost
       ├── LightGBM
       └── PyTorch MLP
       │
       ▼
Model Selection
       │
       ▼
MLflow Experiment Tracking
       │
       ▼
FastAPI + Docker
       │
       ▼
Production Predictions
       │
       ▼
Inference Logging
       │
       ▼
Evidently Drift Monitoring
       │
       ▼
Candidate Retraining
       │
       ▼
Quality Gate
       │
       ▼
Controlled Model Promotion
