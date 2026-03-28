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
    
    prob = min(max(prob, 0.1), 0.9)

    anomaly = abs(iso_model.decision_function(features)[0])
    anomaly = min(anomaly, 1)


# Behavior based on amount (soft influence, not dominating)
    if amount > 5000:
        behavior = 0.7
    elif amount > 2000:
        behavior = 0.5
    else:
        behavior = 0.2

# Amount contribution (scaled)
    amount_factor = min(amount / 10000, 1)

# ✅ Balanced scoring (sum = 1)
    score = (
        0.2 * prob +          # ML importance
        0.2 * anomaly +       # anomaly importance
        0.3 * behavior +       # behavior
        0.3 * amount_factor    # amount influence
    )

    return min(score, 1)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT,
        receiver TEXT,
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

# ---------------- OTP SYSTEM ----------------
otp_store = {}

@app.post("/send_otp")
def send_otp(data: dict):
    phone = data.get("phone")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=3)

    otp_store[phone] = {
        "otp": otp,
        "expiry": expiry,
        "attempts": 3
    }

    print(f"OTP for {phone}: {otp}")

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

# ---------------- FRAUD PREDICTION ----------------
@app.post("/predict")
def predict(data: dict):
    amount = data.get("amount", 0)

    score = fast_predict(amount)

    if score < 0.3:
        risk = "LOW"
    elif score < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    print(f"Amount: {amount}, Score: {score}, Risk: {risk}")

    return {
        "risk": risk,
        "risk_score": float(score)
    }

# ---------------- SAVE TRANSACTION ----------------
@app.post("/save_transaction")
def save_transaction(data: dict):
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions 
    (sender, receiver, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["sender"],
        data["receiver"],
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

    return {"message": "Transaction Saved"}

# ---------------- SEND MONEY ----------------
@app.post("/send")
def send_money(data: dict):
    return save_transaction(data)

# ---------------- GET TRANSACTIONS ----------------
@app.get("/transactions/{user}")
def get_transactions(user: str):
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT sender, receiver, amount, status, time
    FROM transactions
    WHERE sender=? OR receiver=?
    ORDER BY id DESC
    """, (user, user))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "sender": r[0],
            "receiver": r[1],
            "amount": r[2],
            "status": r[3],
            "time": r[4]
        }
        for r in rows
    ]