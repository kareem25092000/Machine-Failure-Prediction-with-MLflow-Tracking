from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import torch
import numpy as np

app = FastAPI()

mlflow.set_tracking_uri("file:./mlruns")

run_id = "9b8836539c6d4e14a88bc364388b9434"
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
model.eval()


# Input schema
class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"status": "MLflow model API running"}


@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features, dtype=np.float32)

    if x.shape[0] != 9:
        return {"error": "Model expects exactly 9 features"}

    x = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    return {"prediction": output.tolist()}