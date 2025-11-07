# src/app/server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

MODEL_URI = os.getenv("MODEL_URI", "models:/titanic-rf/Production")  # or local path
PORT = int(os.getenv("PORT", 8000))

class InputData(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int = 0
    Embarked_S: int = 1

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.dict()])
    preds_proba = model.predict_proba(df) if hasattr(model, "predict_proba") else model.predict(df)
    return {"prediction": preds_proba.tolist()}

@app.get("/")
def root():
    return {"status": "ok"}

# run with: uvicorn server:app --host 0.0.0.0 --port 8000
