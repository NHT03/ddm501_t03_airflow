import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

app = FastAPI(
    title="WDBC Breast Cancer Prediction API",
    description="Serving predictions for WDBC Breast Cancer model logged in MLflow",
    version="1.0.0",
)

MODEL_NAME = "wdbc_breast_cancer_model"

class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictBatchRequest(BaseModel):
    samples: List[Dict[str, float]]

@app.get("/health")
def health():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    return {"status": "ok", "mlflow_tracking_uri": tracking_uri}

def load_latest_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model_uri = f"models:/{MODEL_NAME}/latest"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to latest run from experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("wdbc_breast_cancer")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                return mlflow.sklearn.load_model(model_uri)
        raise RuntimeError("No trained model found in MLflow.")

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model = load_latest_model()
        df = pd.DataFrame([request.features])
        pred_int = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][pred_int])
        label = "Malignant" if pred_int == 1 else "Benign"
        return {
            "prediction": label,
            "prediction_code": pred_int,
            "probability": round(prob, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
