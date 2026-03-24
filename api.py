from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

# ✅ DEFINE APP FIRST
app = FastAPI()

# Load models
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature names (important if using DataFrame)
columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

@app.get("/")
def home():
    return {"message": "Fraud API Running"}

@app.post("/predict")
def predict(data: dict):
    features = pd.DataFrame([data["features"]], columns=columns)

    features = scaler.transform(features)

    rf_pred = rf_model.predict(features)[0]
    iso_pred = iso_model.predict(features)[0]

    fraud = 1 if (rf_pred == 1 or iso_pred == -1) else 0

    return {"fraud": int(fraud)}