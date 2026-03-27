from fastapi import FastAPI
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pickle
import requests
import random

app = FastAPI()

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- OTP CONFIG ----------------
MSG91_AUTH_KEY = "YOUR_AUTH_KEY"
TEMPLATE_ID = "YOUR_TEMPLATE_ID"

otp_store = {}

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

# ---------------- DRIFT ----------------
recent_risks = []
WINDOW_SIZE = 5

def detect_drift(current_risk):
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
    return {"message": "ML API Running"}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):

    merchant = data.get("merchant", "Unknown")
    amount = data["features"][-1] if "features" in data else data.get("amount", 0)

    features = np.zeros((1, 30))
    features[0][-1] = amount
    scaled = features

    prob = (0.6 * model.predict_proba(scaled)[0][1] +
            0.4 * rf_model.predict_proba(scaled)[0][1])

    anomaly = abs(iso_model.decision_function(scaled)[0])

    behavior = 0.8 if amount > 5000 else 0.5 if amount > 2000 else 0.2
    amount_factor = min(amount / 10000, 1)

    risk_score = min((0.2*prob + 0.2*anomaly + 0.4*behavior + 0.4*amount_factor),1)

    drift = detect_drift(risk_score)

    if risk_score < 0.4:
        risk, fraud, status = "LOW", 0, "Approved"
    elif risk_score < 0.7:
        risk, fraud, status = "MEDIUM", 0, "OTP Required"
    else:
        risk, fraud, status = "HIGH", 1, "Step-Up Required"

    return {
        "risk": risk,
        "risk_score": float(risk_score),
        "drift": drift,
        "status": status
    }

# ---------------- SAVE TRANSACTION ----------------
@app.post("/save_transaction")
def save_transaction(data: dict):
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("merchant"),
        float(data.get("amount")),
        int(data.get("fraud", 0)),
        data.get("risk"),
        int(data.get("drift", 0)),
        float(data.get("risk_score", 0)),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("status")
    ))

    conn.commit()
    conn.close()

    return {"message": "Saved"}

# ---------------- SEND OTP ----------------
@app.post("/send_otp")
def send_otp(data: dict):
    phone = data.get("phone")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(seconds=60)

    otp_store[phone] = (otp, expiry)

    print(f"OTP for {phone}: {otp}")

    try:
        requests.post(
            "https://control.msg91.com/api/v5/otp",
            data={
                "template_id": TEMPLATE_ID,
                "mobile": "91" + phone,
                "authkey": MSG91_AUTH_KEY,
                "otp": otp
            }
        )
    except:
        print("SMS not sent")

    return {"message": "OTP sent"}

# ---------------- VERIFY OTP ----------------
@app.post("/verify_otp")
def verify_otp(data: dict):
    phone = data.get("phone")
    user_otp = data.get("otp")

    if phone not in otp_store:
        return {"status": "expired"}

    real_otp, expiry = otp_store[phone]

    if datetime.now() > expiry:
        return {"status": "expired"}

    if user_otp == real_otp:
        return {"status": "verified"}

    return {"status": "invalid"}