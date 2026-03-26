from fastapi import FastAPI
import sqlite3
from datetime import datetime
import random

app = FastAPI()

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

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "API Running"}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):
    print("Incoming Data:", data)

    merchant = data.get("merchant", "Unknown")
    amount = data["features"][-1]
    

    # logic
    if amount <= 2000:
        risk = "LOW"
        fraud = 0
    elif amount <= 5000:
        risk = "MEDIUM"
        fraud = 0
    else:
        risk = "HIGH"
        fraud = 0  # still success (handled by app)

    risk_score = round(random.uniform(0.1, 0.9), 3)

    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        merchant,
        amount,
        fraud,
        risk,
        0,
        risk_score,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Approved"
    ))

    conn.commit()
    conn.close()

    return {
        "fraud": fraud,
        "risk": risk,
        "risk_score": risk_score
    }

# ---------------- FRAUD LOG ----------------
@app.post("/log_fraud")
def log_fraud(data: dict):
    merchant = data.get("merchant")
    amount = data.get("amount")

    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transactions (merchant, amount, fraud, risk, drift, risk_score, time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        merchant,
        amount,
        1,
        "HIGH",
        0,
        0.99,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Blocked"
    ))

    conn.commit()
    conn.close()

    return {"message": "Fraud logged"}