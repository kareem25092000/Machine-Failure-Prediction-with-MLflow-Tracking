# 🏭 Machine Failure Prediction with MLflow + FastAPI

This project builds an end-to-end ML system to predict machine failures using sensor data from Kaggle. It focuses on reproducible experiments, hyperparameter tuning, and deployable model design with real-time inference API.

---

## 🎯 Goals

- Build a full ML pipeline from scratch (preprocessing → training → evaluation)
- Track experiments using **MLflow**
- Optimize models using **Optuna**
- Deploy the best trained model using **FastAPI**
- Enable real-time machine failure prediction via REST API

---

## 📊 Dataset

- Kaggle: Machine Failure Prediction using Sensor Data  
  https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data  
- Binary classification problem based on sensor readings

---

## 🧠 Workflow

- Data preprocessing (cleaning, scaling, feature selection)
- Model training (baseline ML models)
- Hyperparameter tuning with Optuna
- Experiment tracking with MLflow
- Model selection based on best validation metrics
- Deployment of best model using FastAPI

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- PyTorch (if applicable)  
- MLflow  
- Optuna  
- FastAPI  
- Uvicorn  

---

## 📈 MLflow Tracking

- Logs parameters, metrics, and models  
- Enables comparison across experiments  
- MLflow UI used for visualization and model selection  
- Stores best model artifacts for deployment  

---

## 🚀 Model Deployment (FastAPI)

The best trained model is served using a REST API built with FastAPI.

### ▶️ Run API server

```bash
uvicorn app:app --reload
```