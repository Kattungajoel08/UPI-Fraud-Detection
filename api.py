from fastapi import FastAPI
import pickle
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------- INIT ----------------
app = FastAPI()

# ---------------- LOAD MODELS ----------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- DB SETUP ----------------
conn = sqlite3.connect("fraud.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant TEXT,
    amount REAL,
    risk TEXT,
    fraud INTEGER,
    time TEXT
)
""")
conn.commit()

# ---------------- FEATURES ----------------
columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "API Running"}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):
    amount = float(data["features"][-1])
    merchant = data.get("merchant", "Unknown")

    df = pd.DataFrame([data["features"]], columns=columns)
    scaled = scaler.transform(df)

    rf_pred = rf_model.predict(scaled)[0]
    iso_pred = iso_model.predict(scaled)[0]

    ml_fraud = 1 if (rf_pred == 1 or iso_pred == -1) else 0

    # Risk logic
    if amount <= 2000:
        risk = "LOW"
    elif amount <= 5000:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # ✅ STORE IN DATABASE
    cursor.execute("""
    INSERT INTO transactions (merchant, amount, risk, fraud, time)
    VALUES (?, ?, ?, ?, ?)
    """, (
        merchant,
        amount,
        risk,
        ml_fraud,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

    return {
        "fraud": ml_fraud,
        "risk_level": risk,
        "amount": amount
    }

# ---------------- GET ALL TRANSACTIONS ----------------
@app.get("/transactions")
def get_transactions():
    cursor.execute("SELECT * FROM transactions ORDER BY id DESC")
    rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "merchant": r[1],
            "amount": r[2],
            "risk": r[3],
            "fraud": r[4],
            "time": r[5]
        }
        for r in rows
    ]