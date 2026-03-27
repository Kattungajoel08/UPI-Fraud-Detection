from fastapi import FastAPI
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pickle
import random

app = FastAPI()

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))

# ---------------- FAST ML ----------------
def fast_predict(amount):
    features = np.zeros((1, 30))
    features[0][-1] = amount

    prob = (0.6 * model.predict_proba(features)[0][1] +
            0.4 * rf_model.predict_proba(features)[0][1])

    anomaly = abs(iso_model.decision_function(features)[0])

    behavior = 0.8 if amount > 5000 else 0.5 if amount > 2000 else 0.2
    amount_factor = min(amount / 10000, 1)

    return min((0.2*prob + 0.2*anomaly + 0.4*behavior + 0.4*amount_factor),1)

# ---------------- DB ----------------
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

# ---------------- OTP ----------------
otp_store = {}

@app.post("/send_otp")
def send_otp(data: dict):
    phone = data.get("phone")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=3)

    otp_store[phone] = {"otp": otp, "expiry": expiry, "attempts": 3}

    print(f"\nOTP for {phone}: {otp} (3 mins, 3 attempts)")

    return {"message": "OTP sent"}

@app.post("/verify_otp")
def verify_otp(data: dict):
    phone = data.get("phone")
    user_otp = data.get("otp")

    if phone not in otp_store:
        return {"status": "expired"}

    record = otp_store[phone]

    if datetime.now() > record["expiry"]:
        return {"status": "expired"}

    if record["attempts"] <= 0:
        return {"status": "blocked"}

    if user_otp == record["otp"]:
        return {"status": "verified"}

    record["attempts"] -= 1
    return {"status": "invalid", "attempts_left": record["attempts"]}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):
    amount = data.get("amount", 0)

    score = fast_predict(amount)

    if score < 0.4:
        risk = "LOW"
    elif score < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {"risk": risk, "risk_score": float(score)}

# ---------------- SAVE ----------------
@app.post("/save_transaction")
def save_transaction(data: dict):
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["merchant"],
        data["amount"],
        1 if data["status"] == "Fraud" else 0,
        data["risk"],
        0,
        data["risk_score"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["status"]
    ))

    conn.commit()
    conn.close()

    return {"message": "Saved"}