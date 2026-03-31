from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime, timedelta
import random

# ✅ NEW IMPORT (ML ENGINE)
from services.risk_engine import compute_risk, update_model

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect("fraud.db", check_same_thread=False)

def init_db():
    conn = get_connection()
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

    print(f"[OTP] {phone}: {otp}")

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
    try:
        amount = data.get("amount", 0)
        user = data.get("sender", "unknown")
        result = compute_risk(amount, user)
        return result 
    except Exception as e:
        print("ERROR:", e)
        return {"risk": "LOW", "risk_score": 0.1}

# ---------------- SAVE TRANSACTION ----------------
@app.post("/send")
def save_transaction(data: dict):
    try:

        print("RECEIVED DATA:", data)
        conn = get_connection()
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

    # 🔥 ✅ ADAPTIVE LEARNING (MOST IMPORTANT)
        label = 1 if data["status"] == "Fraud" else 0

        update_model(data["amount"], data["sender"], label)

        return {"message": "Transaction Saved"}
    except Exception as e:
        print("SAVE ERROR: ", e)
        return {"error": str(e)}


# ---------------- GET TRANSACTIONS ----------------
@app.get("/transactions/{user}")
def get_transactions(user: str):
    conn = get_connection()
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