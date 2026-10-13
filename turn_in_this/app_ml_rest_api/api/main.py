import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchRequest,
    BatchResponse,
    HouseFeatures,
    PredictionResponse,
)
from src.config import BINARY_FEATURES, FEATURES, METRICS_PATH, MODEL_PATH

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Модель не знайдено: {MODEL_PATH}. Запусти: python -m src.train"
        )
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Housing Price API",
    description="Прогноз ціни житла за характеристиками (датасет Housing.csv)",
    version="1.0.0",
    lifespan=lifespan,
)


def to_frame(items) -> pd.DataFrame:
    df = pd.DataFrame([i.model_dump() for i in items])
    for col in BINARY_FEATURES:
        df[col] = df[col].map({"yes": 1, "no": 0})
    return df[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model/info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json не знайдено")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    pred = state["model"].predict(to_frame([house]))[0]
    return PredictionResponse(predicted_price=round(float(pred), 2))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    preds = state["model"].predict(to_frame(req.items))
    return BatchResponse(predictions=[round(float(p), 2) for p in preds])
