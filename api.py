from fastapi import FastAPI
import sqlite3
from datetime import datetime
import numpy as np
import pickle

app = FastAPI()

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))   # SGD
rf_model = pickle.load(open("rf_model.pkl", "rb"))   # Random Forest
iso_model = pickle.load(open("iso_model.pkl", "rb")) # Isolation Forest
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        merchant TEXT,
        amount REAL,
        fraud INTEGER,
        risk TEXT,
        drift INTEGER,
        risk_score REAL,
        time TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- DRIFT DETECTION ----------------
recent_risks = []
WINDOW_SIZE = 5

def detect_drift(current_risk):
    global recent_risks
    drift_flag = 0

    if len(recent_risks) >= WINDOW_SIZE:
        avg = sum(recent_risks) / len(recent_risks)

        if abs(current_risk - avg) > 0.3:
            drift_flag = 1

    recent_risks.append(current_risk)

    if len(recent_risks) > WINDOW_SIZE:
        recent_risks.pop(0)

    return drift_flag

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "ML Fraud Detection API Running"}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):

    merchant = data.get("merchant", "Unknown")

    # ✅ SAFE INPUT HANDLING (fixes your error)
    if "features" in data:
        amount = data["features"][-1]
    else:
        amount = data.get("amount", 0)

    # ---------------- FEATURE VECTOR ----------------
    features = np.zeros((1, 30))
    features[0][-1] = amount
    scaled = scaler.transform(features)

    # ---------------- ADAPTIVE ML ----------------
    prob_sgd = model.predict_proba(scaled)[0][1]
    prob_rf = rf_model.predict_proba(scaled)[0][1]
    prob = (0.6 * prob_sgd) + (0.4 * prob_rf)

    # ---------------- ANOMALY DETECTION ----------------
    anomaly = abs(iso_model.decision_function(scaled)[0])

    # ---------------- BEHAVIORAL PROFILING ----------------
    if amount > 5000:
        behavior = 0.8
    elif amount > 2000:
        behavior = 0.5
    else:
        behavior = 0.2

    # ---------------- AMOUNT FACTOR ----------------
    amount_factor = min(amount / 10000, 1)

    # ---------------- FINAL RISK SCORE ----------------
    risk_score = (
        0.2 * prob +
        0.2 * anomaly +
        0.4 * behavior +
        0.4 * amount_factor
    )

    risk_score = min(risk_score, 1)

    # ---------------- CONCEPT DRIFT ----------------
    drift_flag = detect_drift(risk_score)

    # ---------------- DECISION ENGINE ----------------
    if risk_score < 0.4:
        risk = "LOW"
        fraud = 0
        status = "Approved"

    elif risk_score < 0.7:
        risk = "MEDIUM"
        fraud = 0
        status = "Approved (OTP Required)"

    else:
        risk = "HIGH"
        fraud = 1
        status = "Blocked"

    # ---------------- SAVE TO DATABASE ----------------
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        merchant,
        float(amount),
        fraud,
        risk,
        drift_flag,
        float(risk_score),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status
    ))

    conn.commit()
    conn.close()

    # ---------------- RESPONSE ----------------
    return {
        "fraud": fraud,
        "risk": risk,
        "risk_score": float(risk_score),
        "drift": drift_flag,
        "status": status
    }

# ---------------- LOG FRAUD (MANUAL) ----------------
@app.post("/log_fraud")
def log_fraud(data: dict):

    merchant = data.get("merchant", "Unknown")
    amount = data.get("amount", 0)

    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        merchant,
        float(amount),
        1,
        "HIGH",
        0,
        0.99,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Blocked"
    ))

    conn.commit()
    conn.close()

    return {"message": "Fraud logged successfully"}