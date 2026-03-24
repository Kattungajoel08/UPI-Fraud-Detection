from fastapi import FastAPI
import pickle
import pandas as pd

# ---------------- INIT APP ----------------
app = FastAPI()

# ---------------- LOAD MODELS ----------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- FEATURE COLUMNS ----------------
columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------- HOME ROUTE ----------------
@app.get("/")
def home():
    return {"message": "UPI Fraud Detection API Running 🚀"}

# ---------------- PREDICT ROUTE ----------------
@app.post("/predict")
def predict(data: dict):
    try:
        # ✅ VALIDATION
        if "features" not in data:
            return {"error": "Missing 'features' field"}

        if len(data["features"]) != 30:
            return {"error": "Feature length must be exactly 30"}

        amount = float(data["features"][-1])

        # ---------------- PREPROCESS ----------------
        df = pd.DataFrame([data["features"]], columns=columns)
        scaled = scaler.transform(df)

        # ---------------- ML MODELS ----------------
        rf_pred = rf_model.predict(scaled)[0]
        iso_pred = iso_model.predict(scaled)[0]

        # ML fraud decision
        ml_fraud = 1 if (rf_pred == 1 or iso_pred == -1) else 0

        # ---------------- RULE-BASED RISK ----------------
        if amount <= 2000:
            risk_level = "LOW"
        elif amount <= 5000:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # ---------------- FINAL RESPONSE ----------------
        return {
            "fraud": int(ml_fraud),
            "risk_level": risk_level,
            "amount": amount
        }

    except Exception as e:
        return {"error": str(e)}